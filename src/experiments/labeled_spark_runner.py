"""
Integration of labeled datasets with Spark Parallel K-Means.

This module reuses the existing file-backed Spark benchmark and
evaluates its final centers against original labeled records.

The independent evaluation occurs after the measured Spark run.
No existing clustering algorithms are modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from src.experiments.benchmark_runner import (
    BenchmarkResult,
    run_file_parallel_kmeans_benchmark,
)
from src.experiments.dataset_bridge import (
    LabeledDataset,
    PreparedSparkDataset,
    prepare_spark_csv,
)
from src.experiments.labeled_spark_evaluation import (
    LabeledSparkEvaluation,
    evaluate_labeled_spark_centers,
)


@dataclass(frozen=True)
class LabeledSparkRunResult:
    """Combined Spark benchmark and independent evaluation."""

    dataset_name: str
    prepared_dataset: PreparedSparkDataset
    benchmark: BenchmarkResult
    evaluation: LabeledSparkEvaluation


def run_labeled_spark_benchmark(
    spark: SparkSession,
    dataset: LabeledDataset,
    *,
    output_directory: str | Path,
    num_clusters: int,
    num_partitions: int,
    random_seed: int = 42,
    local_max_iterations: int = 100,
    global_max_iterations: int = 100,
    local_tolerance: float = 1e-6,
    global_cost_tolerance: float = 1e-6,
    aggregation_restarts: int = 5,
) -> LabeledSparkRunResult:
    """
    Execute Parallel K-Means on a small labeled dataset.

    Steps:
    1. Validate configuration.
    2. Prepare a features-only Spark CSV.
    3. Execute the existing Spark benchmark.
    4. Evaluate final centers using original record order.
    5. Verify independent SSE and cluster counts.

    Ground-truth labels are never used for clustering.
    """

    if num_clusters <= 0:
        raise ValueError(
            "num_clusters must be positive."
        )

    if num_clusters > dataset.num_records:
        raise ValueError(
            "num_clusters cannot exceed record count."
        )

    if num_partitions <= 0:
        raise ValueError(
            "num_partitions must be positive."
        )

    if num_partitions > dataset.num_records:
        raise ValueError(
            "num_partitions cannot exceed record count."
        )

    if aggregation_restarts <= 0:
        raise ValueError(
            "aggregation_restarts must be positive."
        )

    prepared = prepare_spark_csv(
        dataset,
        output_directory=output_directory,
        num_clusters=num_clusters,
    )

    benchmark = run_file_parallel_kmeans_benchmark(
        spark,
        prepared.dataset,
        num_partitions=num_partitions,
        random_seed=random_seed,
        local_max_iterations=local_max_iterations,
        global_max_iterations=global_max_iterations,
        local_tolerance=local_tolerance,
        global_cost_tolerance=global_cost_tolerance,
        aggregation_restarts=aggregation_restarts,
    )

    evaluation = evaluate_labeled_spark_centers(
        dataset,
        benchmark.clustering_result.centers,
    )

    if evaluation.num_records != benchmark.num_records:
        raise RuntimeError(
            "Spark and labeled record counts disagree."
        )

    if evaluation.num_features != benchmark.num_features:
        raise RuntimeError(
            "Spark and labeled feature counts disagree."
        )

    if evaluation.num_clusters != benchmark.num_clusters:
        raise RuntimeError(
            "Spark and labeled cluster counts disagree."
        )

    if (
        evaluation.cluster_counts
        != benchmark.clustering_result.cluster_counts
    ):
        raise RuntimeError(
            "Spark and independent cluster assignments disagree."
        )

    if not math.isclose(
        evaluation.sse,
        benchmark.clustering_result.sse,
        rel_tol=1e-7,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "Spark and independent SSE calculations disagree."
        )

    return LabeledSparkRunResult(
        dataset_name=dataset.name,
        prepared_dataset=prepared,
        benchmark=benchmark,
        evaluation=evaluation,
    )
