"""
Standardized experiment-result records for Parallel K-Means benchmarks.

This module converts a controlled benchmark execution into a stable,
flat record that can later be written to CSV or analyzed across
repeated experiments.

Keeping this transformation in the experiment layer prevents reporting
and storage concerns from becoming part of the clustering algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.experiments.benchmark_runner import BenchmarkResult


@dataclass(frozen=True)
class ExperimentResult:
    """Flat record describing one completed benchmark execution."""

    experiment_id: str
    dataset_name: str

    num_records: int
    num_features: int
    num_clusters: int
    num_partitions: int

    spark_master: str
    random_seed: int

    local_max_iterations: int
    global_max_iterations: int
    tolerance: float

    materialized_record_count: int

    converged: bool
    iterations: int
    sse: float
    cluster_counts: tuple[int, ...]

    partition_clustering_seconds: float
    center_aggregation_seconds: float
    initialization_seconds: float
    global_clustering_seconds: float
    final_evaluation_seconds: float
    total_runtime_seconds: float


def create_experiment_result(
    benchmark: BenchmarkResult,
    *,
    experiment_id: str,
    dataset_name: str,
    local_max_iterations: int,
    global_max_iterations: int,
    tolerance: float,
) -> ExperimentResult:
    """
    Convert one controlled benchmark result into a flat experiment record.

    Parameters supplied to the benchmark but not currently stored by
    BenchmarkResult are provided explicitly so that every experimental
    record preserves the complete configuration used for that run.
    """

    if not experiment_id.strip():
        raise ValueError("experiment_id cannot be empty.")

    if not dataset_name.strip():
        raise ValueError("dataset_name cannot be empty.")

    if local_max_iterations <= 0:
        raise ValueError(
            "local_max_iterations must be greater than zero."
        )

    if global_max_iterations <= 0:
        raise ValueError(
            "global_max_iterations must be greater than zero."
        )

    if not isfinite(tolerance):
        raise ValueError("tolerance must be finite.")

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    clustering = benchmark.clustering_result
    timing = clustering.timing

    return ExperimentResult(
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        num_records=benchmark.num_records,
        num_features=benchmark.num_features,
        num_clusters=benchmark.num_clusters,
        num_partitions=benchmark.num_partitions,
        spark_master=benchmark.spark_master,
        random_seed=benchmark.random_seed,
        local_max_iterations=local_max_iterations,
        global_max_iterations=global_max_iterations,
        tolerance=tolerance,
        materialized_record_count=(
            benchmark.materialized_record_count
        ),
        converged=clustering.converged,
        iterations=clustering.iterations,
        sse=clustering.sse,
        cluster_counts=clustering.cluster_counts,
        partition_clustering_seconds=(
            timing.partition_clustering_seconds
        ),
        center_aggregation_seconds=(
            timing.center_aggregation_seconds
        ),
        initialization_seconds=timing.initialization_seconds,
        global_clustering_seconds=(
            timing.global_clustering_seconds
        ),
        final_evaluation_seconds=(
            timing.final_evaluation_seconds
        ),
        total_runtime_seconds=timing.total_runtime_seconds,
    )
