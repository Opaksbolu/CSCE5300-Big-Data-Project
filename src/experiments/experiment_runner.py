"""
Controlled execution of one complete Parallel K-Means experiment.

This module coordinates dataset generation, benchmark execution,
standardized result creation, and optional CSV persistence.

The orchestration remains separate from the clustering implementation
so that experimental concerns do not alter algorithm behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiments.benchmark_runner import (
    run_parallel_kmeans_benchmark,
)
from src.experiments.experiment_result import (
    ExperimentResult,
    create_experiment_result,
)
from src.experiments.result_storage import append_experiment_result
from src.experiments.synthetic_data import generate_synthetic_dataset


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Configuration for one synthetic Parallel K-Means experiment."""

    experiment_id: str
    dataset_name: str = "synthetic"

    num_records: int = 1000
    num_features: int = 20
    num_clusters: int = 5
    num_partitions: int = 4

    cluster_spread: float = 1.0
    center_separation: float = 10.0

    random_seed: int = 42

    local_max_iterations: int = 100
    global_max_iterations: int = 100
    tolerance: float = 1e-6


def run_synthetic_experiment(
    spark,
    config: SyntheticExperimentConfig,
    *,
    output_path: str | Path | None = None,
) -> ExperimentResult:
    """
    Execute one complete synthetic Parallel K-Means experiment.

    The dataset is generated deterministically from the configuration,
    materialized by the benchmark layer, processed by the complete
    Parallel K-Means pipeline, converted into a standardized result,
    and optionally appended to a CSV file.
    """

    dataset = generate_synthetic_dataset(
        num_records=config.num_records,
        num_features=config.num_features,
        num_clusters=config.num_clusters,
        cluster_spread=config.cluster_spread,
        center_separation=config.center_separation,
        random_seed=config.random_seed,
    )

    benchmark = run_parallel_kmeans_benchmark(
        spark,
        dataset,
        num_partitions=config.num_partitions,
        random_seed=config.random_seed,
        local_max_iterations=config.local_max_iterations,
        global_max_iterations=config.global_max_iterations,
        tolerance=config.tolerance,
    )

    result = create_experiment_result(
        benchmark,
        experiment_id=config.experiment_id,
        dataset_name=config.dataset_name,
        local_max_iterations=config.local_max_iterations,
        global_max_iterations=config.global_max_iterations,
        tolerance=config.tolerance,
    )

    if output_path is not None:
        append_experiment_result(
            result,
            output_path,
        )

    return result
