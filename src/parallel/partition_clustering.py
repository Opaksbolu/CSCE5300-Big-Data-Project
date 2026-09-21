"""
Partition-level K-Means clustering for Spark RDDs.

This module connects the tested local K-Means implementation to
Spark's partition execution model. Each non-empty Spark partition is
clustered independently and returns a small set of local centers.

These local centers will later be aggregated and re-clustered to
produce global initialization centers for the parallel K-Means
algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from pyspark import RDD

from src.parallel.local_kmeans import (
    Point,
    fit_local_kmeans,
)


@dataclass(frozen=True)
class PartitionClusteringResult:
    """
    Result produced by K-Means on one Spark partition.
    """

    partition_index: int
    record_count: int
    centers: tuple[Point, ...]
    iterations: int
    converged: bool
    sse: float


def cluster_partition(
    partition_index: int,
    records: Iterable[Point],
    *,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42,
) -> Iterator[PartitionClusteringResult]:
    """
    Run local K-Means on one Spark partition.

    Parameters
    ----------
    partition_index:
        Index assigned to the partition by Spark.

    records:
        Iterator containing the feature vectors stored in the
        partition.

    k:
        Number of local clusters.

    max_iterations:
        Maximum local K-Means iterations.

    tolerance:
        Convergence threshold used by local K-Means.

    random_seed:
        Base seed. The partition index is added to this value so
        partitions receive deterministic but distinct seeds.

    Yields
    ------
    PartitionClusteringResult
        One result for each non-empty partition.

    Notes
    -----
    A partition must contain at least k records. This requirement
    prevents an invalid local K-Means configuration.
    """

    points = tuple(records)

    if not points:
        return

    if len(points) < k:
        raise ValueError(
            f"Partition {partition_index} contains "
            f"{len(points)} records, but k={k}. "
            "Each non-empty partition must contain at least k records."
        )

    local_result = fit_local_kmeans(
        points,
        k=k,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_seed=random_seed + partition_index,
    )

    yield PartitionClusteringResult(
        partition_index=partition_index,
        record_count=len(points),
        centers=local_result.centers,
        iterations=local_result.iterations,
        converged=local_result.converged,
        sse=local_result.sse,
    )


def cluster_rdd_partitions(
    rdd: RDD,
    *,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42,
) -> RDD:
    """
    Execute local K-Means independently across Spark partitions.

    The returned RDD contains one PartitionClusteringResult for each
    non-empty input partition.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return rdd.mapPartitionsWithIndex(
        lambda partition_index, records: cluster_partition(
            partition_index,
            records,
            k=k,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_seed=random_seed,
        )
    )


def collect_candidate_centers(
    partition_results: Iterable[PartitionClusteringResult],
) -> tuple[Point, ...]:
    """
    Flatten partition-level centers into one candidate-center set.

    These centers are intended for the next stage of the algorithm,
    where they will be re-clustered to obtain global initial centers.
    """

    centers: list[Point] = []

    for result in partition_results:
        centers.extend(result.centers)

    return tuple(centers)
