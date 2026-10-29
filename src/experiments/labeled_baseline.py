
"""
Traditional K-Means baseline for small labeled datasets.

Uses the existing local K-Means implementation without Spark.
Ground-truth labels are excluded from clustering.

Record IDs, labels, and predicted assignments are retained
for later evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from src.experiments.dataset_bridge import LabeledDataset
from src.parallel.local_kmeans import fit_local_kmeans


@dataclass(frozen=True)
class LabeledBaselineResult:
    """Traditional K-Means metrics and aligned evaluation data."""

    dataset_name: str
    num_records: int
    num_features: int
    num_clusters: int
    random_seed: int
    num_initializations: int
    best_initialization_seed: int
    max_iterations: int
    tolerance: float
    converged: bool
    iterations: int
    sse: float
    cluster_counts: tuple[int, ...]
    total_runtime_seconds: float
    record_ids: tuple[int, ...]
    true_labels: tuple[int, ...]
    predicted_clusters: tuple[int, ...]


def run_labeled_baseline(
    dataset: LabeledDataset,
    *,
    num_clusters: int,
    random_seed: int = 42,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    num_initializations: int = 1,
) -> LabeledBaselineResult:
    """
    Run traditional K-Means on a validated labeled dataset.

    Dataset loading is excluded from clustering runtime.
    The best initialization is selected by minimum SSE.
    """

    if num_clusters <= 0:
        raise ValueError(
            "num_clusters must be greater than zero."
        )

    if num_clusters > dataset.num_records:
        raise ValueError(
            "num_clusters cannot exceed dataset size."
        )

    if num_initializations <= 0:
        raise ValueError(
            "num_initializations must be greater than zero."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    if not (
        len(dataset.points)
        == len(dataset.labels)
        == len(dataset.record_ids)
    ):
        raise ValueError(
            "Dataset features, labels, and IDs must align."
        )

    if len(set(dataset.record_ids)) != dataset.num_records:
        raise ValueError(
            "Dataset record IDs must be unique."
        )

    best_result = None
    best_seed = None

    start = perf_counter()

    for initialization_index in range(num_initializations):
        initialization_seed = (
            random_seed + initialization_index
        )

        result = fit_local_kmeans(
            dataset.points,
            k=num_clusters,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_seed=initialization_seed,
        )

        if (
            best_result is None
            or result.sse < best_result.sse
        ):
            best_result = result
            best_seed = initialization_seed

    runtime = perf_counter() - start

    assignments = tuple(best_result.assignments)

    if len(assignments) != dataset.num_records:
        raise RuntimeError(
            "Cluster assignment count does not match "
            "dataset record count."
        )

    cluster_counts = tuple(
        assignments.count(cluster_index)
        for cluster_index in range(num_clusters)
    )

    if sum(cluster_counts) != dataset.num_records:
        raise RuntimeError(
            "Cluster counts do not sum to dataset size."
        )

    return LabeledBaselineResult(
        dataset_name=dataset.name,
        num_records=dataset.num_records,
        num_features=dataset.num_features,
        num_clusters=num_clusters,
        random_seed=random_seed,
        num_initializations=num_initializations,
        best_initialization_seed=best_seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
        converged=best_result.converged,
        iterations=best_result.iterations,
        sse=best_result.sse,
        cluster_counts=cluster_counts,
        total_runtime_seconds=runtime,
        record_ids=dataset.record_ids,
        true_labels=dataset.labels,
        predicted_clusters=assignments,
    )
