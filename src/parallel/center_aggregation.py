"""
Global center initialization from partition-level candidate centers.

The improved Parallel K-Means workflow first performs local K-Means
independently inside Spark partitions. Each partition returns a small
set of local cluster centers.

Those local centers form a candidate-center set. This module
re-clusters the candidate centers to obtain exactly k global
initialization centers.

Keeping this step separate from Spark execution makes the aggregation
logic deterministic, testable, and reusable by the later distributed
K-Means iteration stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.parallel.local_kmeans import (
    LocalKMeansResult,
    Point,
    fit_local_kmeans,
)
from src.parallel.partition_clustering import (
    PartitionClusteringResult,
    collect_candidate_centers,
)


@dataclass(frozen=True)
class CenterAggregationResult:
    """
    Result of re-clustering partition-level candidate centers.
    """

    candidate_centers: tuple[Point, ...]
    global_centers: tuple[Point, ...]
    candidate_count: int
    iterations: int
    converged: bool
    sse: float


def aggregate_candidate_centers(
    candidate_centers: Iterable[Point],
    *,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42,
) -> CenterAggregationResult:
    """
    Re-cluster candidate centers into k global initialization centers.

    Parameters
    ----------
    candidate_centers:
        Local cluster centers produced by Spark partitions.

    k:
        Number of global centers required.

    max_iterations:
        Maximum number of K-Means iterations used during aggregation.

    tolerance:
        Convergence threshold for center movement.

    random_seed:
        Seed used to make center initialization reproducible.

    Returns
    -------
    CenterAggregationResult
        Candidate centers and the resulting global initialization
        centers.

    Raises
    ------
    ValueError
        If k is invalid or fewer than k candidate centers are supplied.
    """

    centers = tuple(candidate_centers)

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if len(centers) < k:
        raise ValueError(
            "The number of candidate centers must be at least k."
        )

    clustering_result: LocalKMeansResult = fit_local_kmeans(
        centers,
        k=k,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_seed=random_seed,
    )

    return CenterAggregationResult(
        candidate_centers=centers,
        global_centers=clustering_result.centers,
        candidate_count=len(centers),
        iterations=clustering_result.iterations,
        converged=clustering_result.converged,
        sse=clustering_result.sse,
    )


def aggregate_partition_results(
    partition_results: Iterable[PartitionClusteringResult],
    *,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42,
) -> CenterAggregationResult:
    """
    Convert partition results directly into global initial centers.

    This function represents the transition between Spark's local
    partition-clustering stage and the later global parallel K-Means
    stage.
    """

    candidate_centers = collect_candidate_centers(partition_results)

    return aggregate_candidate_centers(
        candidate_centers,
        k=k,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_seed=random_seed,
    )
