"""
End-to-end improved Parallel K-Means pipeline.

This module connects the three major stages of the implementation:

    1. partition-level local K-Means,
    2. candidate-center aggregation,
    3. distributed global K-Means convergence.

The orchestration layer intentionally delegates clustering work to
the independently tested components rather than duplicating their
algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark import RDD

from src.parallel.center_aggregation import (
    CenterAggregationResult,
    aggregate_partition_results,
)
from src.parallel.global_kmeans import (
    GlobalKMeansResult,
    fit_global_kmeans,
)
from src.parallel.local_kmeans import Point
from src.parallel.partition_clustering import (
    PartitionClusteringResult,
    cluster_rdd_partitions,
)


@dataclass(frozen=True)
class ParallelKMeansResult:
    """
    Result of the complete improved Parallel K-Means pipeline.
    """

    centers: tuple[Point, ...]
    iterations: int
    converged: bool
    sse: float
    cluster_counts: tuple[int, ...]
    partition_results: tuple[PartitionClusteringResult, ...]
    initialization: CenterAggregationResult
    global_result: GlobalKMeansResult


def fit_parallel_kmeans(
    rdd: RDD,
    *,
    k: int,
    local_max_iterations: int = 100,
    global_max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42,
) -> ParallelKMeansResult:
    """
    Execute the complete improved Parallel K-Means workflow.

    Parameters
    ----------
    rdd:
        Spark RDD containing feature vectors.

    k:
        Number of clusters.

    local_max_iterations:
        Maximum iterations used by partition-level K-Means and
        candidate-center aggregation.

    global_max_iterations:
        Maximum iterations used by distributed global K-Means.

    tolerance:
        Maximum squared center movement allowed before convergence.

    random_seed:
        Seed used to make local initialization reproducible.

    Returns
    -------
    ParallelKMeansResult
        Complete initialization and global clustering results.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if local_max_iterations <= 0:
        raise ValueError(
            "local_max_iterations must be greater than zero."
        )

    if global_max_iterations <= 0:
        raise ValueError(
            "global_max_iterations must be greater than zero."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    if rdd.isEmpty():
        raise ValueError(
            "RDD must contain at least one point."
        )

    partition_results = tuple(
        cluster_rdd_partitions(
            rdd,
            k=k,
            max_iterations=local_max_iterations,
            tolerance=tolerance,
            random_seed=random_seed,
        ).collect()
    )

    if not partition_results:
        raise ValueError(
            "Partition clustering produced no results."
        )

    initialization = aggregate_partition_results(
        partition_results,
        k=k,
        max_iterations=local_max_iterations,
        tolerance=tolerance,
        random_seed=random_seed,
    )

    global_result = fit_global_kmeans(
        rdd,
        initialization.global_centers,
        max_iterations=global_max_iterations,
        tolerance=tolerance,
    )

    return ParallelKMeansResult(
        centers=global_result.centers,
        iterations=global_result.iterations,
        converged=global_result.converged,
        sse=global_result.sse,
        cluster_counts=global_result.cluster_counts,
        partition_results=partition_results,
        initialization=initialization,
        global_result=global_result,
    )
