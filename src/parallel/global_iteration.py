"""
Distributed global K-Means iteration utilities.

This module implements the Spark-side operations required for one
global K-Means iteration. Cluster centers are broadcast to Spark
workers, each data point is assigned to its nearest center, and
cluster statistics are aggregated across partitions.

The functions in this module intentionally perform one iteration at
a time. The convergence loop is implemented separately so that the
distributed mathematics can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pyspark import RDD

from src.parallel.local_kmeans import (
    Point,
    nearest_center,
    squared_euclidean_distance,
)


@dataclass(frozen=True)
class ClusterStatistics:
    """
    Aggregated statistics for one cluster.

    Attributes
    ----------
    coordinate_sums:
        Sum of every feature coordinate assigned to the cluster.

    count:
        Number of points assigned to the cluster.

    sse:
        Sum of squared distances between assigned points and the
        cluster center used during the iteration.
    """

    coordinate_sums: Point
    count: int
    sse: float


@dataclass(frozen=True)
class GlobalIterationResult:
    """
    Result of one distributed global K-Means iteration.
    """

    centers: tuple[Point, ...]
    cluster_counts: tuple[int, ...]
    sse: float
    maximum_center_shift: float


def _point_to_cluster_statistics(
    point: Point,
    centers: Sequence[Point],
) -> tuple[int, ClusterStatistics]:
    """
    Assign one point to its nearest center and create statistics
    suitable for distributed aggregation.
    """

    cluster_index = nearest_center(
        point,
        centers,
    )

    center = centers[cluster_index]

    statistics = ClusterStatistics(
        coordinate_sums=point,
        count=1,
        sse=squared_euclidean_distance(
            point,
            center,
        ),
    )

    return cluster_index, statistics


def _merge_cluster_statistics(
    left: ClusterStatistics,
    right: ClusterStatistics,
) -> ClusterStatistics:
    """
    Combine statistics produced by separate Spark records or
    partitions.
    """

    if len(left.coordinate_sums) != len(right.coordinate_sums):
        raise ValueError(
            "Cluster statistics must have the same dimensionality."
        )

    coordinate_sums = tuple(
        left_value + right_value
        for left_value, right_value in zip(
            left.coordinate_sums,
            right.coordinate_sums,
        )
    )

    return ClusterStatistics(
        coordinate_sums=coordinate_sums,
        count=left.count + right.count,
        sse=left.sse + right.sse,
    )


def _calculate_updated_center(
    statistics: ClusterStatistics,
) -> Point:
    """
    Calculate a cluster center from aggregated coordinate sums.
    """

    if statistics.count <= 0:
        raise ValueError(
            "Cluster statistics must contain at least one record."
        )

    return tuple(
        coordinate_sum / statistics.count
        for coordinate_sum in statistics.coordinate_sums
    )


def _maximum_center_shift(
    old_centers: Sequence[Point],
    new_centers: Sequence[Point],
) -> float:
    """
    Return the largest squared movement among all cluster centers.
    """

    if len(old_centers) != len(new_centers):
        raise ValueError(
            "Old and new center collections must have the same size."
        )

    if not old_centers:
        raise ValueError(
            "At least one center is required."
        )

    return max(
        squared_euclidean_distance(
            old_center,
            new_center,
        )
        for old_center, new_center in zip(
            old_centers,
            new_centers,
        )
    )


def run_global_iteration(
    rdd: RDD,
    centers: Sequence[Point],
) -> GlobalIterationResult:
    """
    Execute one distributed K-Means iteration.

    The current centers are broadcast to Spark workers. Workers assign
    points to their nearest center and create partial cluster
    statistics. Spark then combines those statistics by cluster.

    Empty clusters retain their previous center.
    """

    if not centers:
        raise ValueError(
            "At least one center is required."
        )

    center_tuple = tuple(centers)

    dimensions = len(center_tuple[0])

    if dimensions == 0:
        raise ValueError(
            "Centers must contain at least one feature."
        )

    if any(
        len(center) != dimensions
        for center in center_tuple
    ):
        raise ValueError(
            "All centers must have the same dimensionality."
        )

    spark_context = rdd.context

    broadcast_centers = spark_context.broadcast(
        center_tuple
    )

    try:
        cluster_statistics = (
            rdd
            .map(
                lambda point: _point_to_cluster_statistics(
                    point,
                    broadcast_centers.value,
                )
            )
            .reduceByKey(
                _merge_cluster_statistics
            )
            .collectAsMap()
        )

        new_centers: list[Point] = []
        cluster_counts: list[int] = []
        total_sse = 0.0

        for cluster_index, previous_center in enumerate(
            center_tuple
        ):
            statistics = cluster_statistics.get(
                cluster_index
            )

            if statistics is None:
                new_centers.append(
                    previous_center
                )
                cluster_counts.append(0)
                continue

            new_centers.append(
                _calculate_updated_center(
                    statistics
                )
            )

            cluster_counts.append(
                statistics.count
            )

            total_sse += statistics.sse

        updated_centers = tuple(new_centers)

        maximum_center_shift = _maximum_center_shift(
            center_tuple,
            updated_centers,
        )

        return GlobalIterationResult(
            centers=updated_centers,
            cluster_counts=tuple(cluster_counts),
            sse=total_sse,
            maximum_center_shift=maximum_center_shift,
        )

    finally:
        broadcast_centers.destroy()
