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
from time import perf_counter

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
class ParallelKMeansTiming:
    """
    Runtime measurements for the major Parallel K-Means stages.

    All values are measured in seconds using a monotonic
    high-resolution performance counter.

    global_clustering_seconds measures only the iterative distributed
    K-Means loop. Final-center SSE and cluster-count evaluation are
    reported separately in final_evaluation_seconds.
    """

    partition_clustering_seconds: float
    center_aggregation_seconds: float
    initialization_seconds: float
    global_clustering_seconds: float
    final_evaluation_seconds: float
    total_runtime_seconds: float


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
    timing: ParallelKMeansTiming


def fit_parallel_kmeans(
    rdd: RDD,
    *,
    k: int,
    local_max_iterations: int = 100,
    global_max_iterations: int = 100,
    local_tolerance: float = 1e-6,
    global_cost_tolerance: float = 1e-6,
    random_seed: int = 42,
    aggregation_restarts: int = 5,
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

    local_tolerance:
        Maximum squared center movement allowed before convergence
        during partition-level K-Means and candidate-center
        aggregation.

    global_cost_tolerance:
        Maximum absolute change in consecutive global iteration SSE
        values allowed before convergence.

    random_seed:
        Seed used to make local initialization reproducible.

    aggregation_restarts:
        Number of deterministic candidate-center aggregation
        initializations to evaluate. The lowest-SSE result is retained.

    Returns
    -------
    ParallelKMeansResult
        Complete initialization, timing, and global clustering results.
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

    if local_tolerance < 0:
        raise ValueError(
            "local_tolerance cannot be negative."
        )

    if global_cost_tolerance < 0:
        raise ValueError(
            "global_cost_tolerance cannot be negative."
        )

    if aggregation_restarts <= 0:
        raise ValueError(
            "aggregation_restarts must be greater than zero."
        )

    if rdd.isEmpty():
        raise ValueError(
            "RDD must contain at least one point."
        )

    total_start = perf_counter()

    partition_start = perf_counter()

    partition_results = tuple(
        cluster_rdd_partitions(
            rdd,
            k=k,
            max_iterations=local_max_iterations,
            tolerance=local_tolerance,
            random_seed=random_seed,
        ).collect()
    )

    partition_clustering_seconds = (
        perf_counter() - partition_start
    )

    if not partition_results:
        raise ValueError(
            "Partition clustering produced no results."
        )

    aggregation_start = perf_counter()

    initialization = aggregate_partition_results(
        partition_results,
        k=k,
        max_iterations=local_max_iterations,
        tolerance=local_tolerance,
        random_seed=random_seed,
        num_restarts=aggregation_restarts,
    )

    center_aggregation_seconds = (
        perf_counter() - aggregation_start
    )

    initialization_seconds = (
        partition_clustering_seconds
        + center_aggregation_seconds
    )

    global_result = fit_global_kmeans(
        rdd,
        initialization.global_centers,
        max_iterations=global_max_iterations,
        tolerance=global_cost_tolerance,
    )

    global_clustering_seconds = (
        global_result.timing.iteration_seconds
    )

    final_evaluation_seconds = (
        global_result.timing.final_evaluation_seconds
    )

    total_runtime_seconds = (
        perf_counter() - total_start
    )

    timing = ParallelKMeansTiming(
        partition_clustering_seconds=(
            partition_clustering_seconds
        ),
        center_aggregation_seconds=(
            center_aggregation_seconds
        ),
        initialization_seconds=(
            initialization_seconds
        ),
        global_clustering_seconds=(
            global_clustering_seconds
        ),
        final_evaluation_seconds=(
            final_evaluation_seconds
        ),
        total_runtime_seconds=(
            total_runtime_seconds
        ),
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
        timing=timing,
    )
