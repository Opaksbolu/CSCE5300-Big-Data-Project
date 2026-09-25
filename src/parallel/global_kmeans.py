"""
Distributed global K-Means convergence loop.

This module repeatedly executes the tested Spark global-iteration
operation until the cluster centers converge or the configured
maximum number of iterations is reached.

The implementation records per-iteration metrics and separates
iterative clustering time from final evaluation time so that
experimental runtime measurements have clear boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from pyspark import RDD

from src.parallel.global_iteration import (
    run_global_iteration,
)
from src.parallel.local_kmeans import (
    Point,
    nearest_center,
    squared_euclidean_distance,
)


@dataclass(frozen=True)
class IterationMetrics:
    """
    Metrics recorded after one distributed K-Means iteration.
    """

    iteration: int
    sse: float
    maximum_center_shift: float
    cluster_counts: tuple[int, ...]


@dataclass(frozen=True)
class GlobalKMeansTiming:
    """
    Runtime measurements for distributed global K-Means.

    iteration_seconds measures only the repeated center-assignment
    and center-update loop.

    final_evaluation_seconds measures the separate Spark operation
    used to evaluate SSE and cluster counts against the final centers.

    total_seconds covers both measured phases plus negligible
    orchestration overhead between them.
    """

    iteration_seconds: float
    final_evaluation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class GlobalKMeansResult:
    """
    Final result of the distributed global K-Means convergence loop.
    """

    centers: tuple[Point, ...]
    iterations: int
    converged: bool
    sse: float
    cluster_counts: tuple[int, ...]
    history: tuple[IterationMetrics, ...]
    timing: GlobalKMeansTiming


def evaluate_centers(
    rdd: RDD,
    centers: Sequence[Point],
) -> tuple[float, tuple[int, ...]]:
    """
    Evaluate SSE and cluster counts for a fixed set of centers.

    Unlike a K-Means iteration, this operation does not update the
    centers. It is used after the convergence loop so that the final
    reported SSE and cluster counts correspond exactly to the centers
    returned to the caller.
    """

    if not centers:
        raise ValueError(
            "At least one center is required."
        )

    center_tuple = tuple(centers)

    spark_context = rdd.context
    broadcast_centers = spark_context.broadcast(
        center_tuple
    )

    try:
        cluster_statistics = (
            rdd
            .map(
                lambda point: (
                    nearest_center(
                        point,
                        broadcast_centers.value,
                    ),
                    point,
                )
            )
            .map(
                lambda assignment: (
                    assignment[0],
                    (
                        1,
                        squared_euclidean_distance(
                            assignment[1],
                            broadcast_centers.value[
                                assignment[0]
                            ],
                        ),
                    ),
                )
            )
            .reduceByKey(
                lambda left, right: (
                    left[0] + right[0],
                    left[1] + right[1],
                )
            )
            .collectAsMap()
        )

        cluster_counts = tuple(
            cluster_statistics.get(
                cluster_index,
                (0, 0.0),
            )[0]
            for cluster_index in range(
                len(center_tuple)
            )
        )

        total_sse = sum(
            statistics[1]
            for statistics in cluster_statistics.values()
        )

        return total_sse, cluster_counts

    finally:
        broadcast_centers.destroy()


def fit_global_kmeans(
    rdd: RDD,
    initial_centers: Sequence[Point],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> GlobalKMeansResult:
    """
    Run distributed K-Means until convergence.

    Parameters
    ----------
    rdd:
        Spark RDD containing feature vectors.

    initial_centers:
        Global initialization centers produced by the center
        aggregation stage.

    max_iterations:
        Maximum number of distributed K-Means iterations.

    tolerance:
        Maximum squared center movement allowed before convergence.

    Returns
    -------
    GlobalKMeansResult
        Final centers, convergence information, cluster counts,
        SSE, per-iteration metrics, and separated runtime
        measurements.
    """

    if not initial_centers:
        raise ValueError(
            "At least one initial center is required."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    if rdd.isEmpty():
        raise ValueError(
            "The input RDD cannot be empty."
        )

    centers = tuple(initial_centers)

    history: list[IterationMetrics] = []

    converged = False

    total_start = perf_counter()
    iteration_start = perf_counter()

    for iteration_number in range(
        1,
        max_iterations + 1,
    ):
        iteration_result = run_global_iteration(
            rdd,
            centers,
        )

        centers = iteration_result.centers

        history.append(
            IterationMetrics(
                iteration=iteration_number,
                sse=iteration_result.sse,
                maximum_center_shift=(
                    iteration_result.maximum_center_shift
                ),
                cluster_counts=(
                    iteration_result.cluster_counts
                ),
            )
        )

        if (
            iteration_result.maximum_center_shift
            <= tolerance
        ):
            converged = True
            break

    iteration_seconds = (
        perf_counter() - iteration_start
    )

    evaluation_start = perf_counter()

    final_sse, final_cluster_counts = evaluate_centers(
        rdd,
        centers,
    )

    final_evaluation_seconds = (
        perf_counter() - evaluation_start
    )

    total_seconds = (
        perf_counter() - total_start
    )

    timing = GlobalKMeansTiming(
        iteration_seconds=iteration_seconds,
        final_evaluation_seconds=(
            final_evaluation_seconds
        ),
        total_seconds=total_seconds,
    )

    return GlobalKMeansResult(
        centers=centers,
        iterations=len(history),
        converged=converged,
        sse=final_sse,
        cluster_counts=final_cluster_counts,
        history=tuple(history),
        timing=timing,
    )
