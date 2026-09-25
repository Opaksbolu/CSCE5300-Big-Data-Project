"""
Tests for controlled Parallel K-Means benchmark execution.
"""

from __future__ import annotations

import pytest

from src.experiments.benchmark_runner import (
    _create_persisted_rdd,
    run_parallel_kmeans_benchmark,
)
from src.experiments.synthetic_data import (
    generate_synthetic_dataset,
)


def test_create_persisted_rdd_materializes_dataset(spark) -> None:
    """The benchmark RDD should be persisted and fully materialized."""

    dataset = generate_synthetic_dataset(
        num_records=20,
        num_features=2,
        num_clusters=2,
        random_seed=42,
    )

    rdd, record_count = _create_persisted_rdd(
        spark,
        dataset,
        num_partitions=2,
    )

    try:
        assert record_count == 20
        assert rdd.getNumPartitions() == 2
        assert rdd.is_cached
    finally:
        rdd.unpersist()


def test_create_persisted_rdd_rejects_invalid_partition_count(
    spark,
) -> None:
    """A benchmark requires at least one Spark partition."""

    dataset = generate_synthetic_dataset(
        num_records=20,
        num_features=2,
        num_clusters=2,
    )

    with pytest.raises(
        ValueError,
        match="num_partitions must be greater than zero",
    ):
        _create_persisted_rdd(
            spark,
            dataset,
            num_partitions=0,
        )


def test_benchmark_executes_complete_pipeline(spark) -> None:
    """A controlled benchmark should execute the full clustering flow."""

    dataset = generate_synthetic_dataset(
        num_records=40,
        num_features=2,
        num_clusters=2,
        cluster_spread=0.25,
        center_separation=10.0,
        random_seed=42,
    )

    result = run_parallel_kmeans_benchmark(
        spark,
        dataset,
        num_partitions=2,
        random_seed=42,
        tolerance=1e-12,
    )

    assert result.num_records == 40
    assert result.num_features == 2
    assert result.num_clusters == 2
    assert result.num_partitions == 2
    assert result.materialized_record_count == 40
    assert result.random_seed == 42

    assert result.clustering_result.converged
    assert result.clustering_result.iterations >= 1
    assert len(result.clustering_result.centers) == 2
    assert sum(result.clustering_result.cluster_counts) == 40


def test_benchmark_records_spark_master(spark) -> None:
    """Benchmark metadata should record the active Spark master."""

    dataset = generate_synthetic_dataset(
        num_records=40,
        num_features=2,
        num_clusters=2,
        random_seed=42,
    )

    result = run_parallel_kmeans_benchmark(
        spark,
        dataset,
        num_partitions=2,
    )

    assert result.spark_master == spark.sparkContext.master


def test_benchmark_reports_nonnegative_timings(spark) -> None:
    """The clustering result should expose valid timing measurements."""

    dataset = generate_synthetic_dataset(
        num_records=40,
        num_features=2,
        num_clusters=2,
        random_seed=42,
    )

    result = run_parallel_kmeans_benchmark(
        spark,
        dataset,
        num_partitions=2,
    )

    timing = result.clustering_result.timing

    assert timing.partition_clustering_seconds >= 0.0
    assert timing.center_aggregation_seconds >= 0.0
    assert timing.initialization_seconds >= 0.0
    assert timing.global_clustering_seconds >= 0.0
    assert timing.final_evaluation_seconds >= 0.0
    assert timing.total_runtime_seconds >= 0.0


def test_create_persisted_rdd_unpersists_when_materialization_fails(
    spark,
    monkeypatch,
) -> None:
    """A failed materialization should unpersist the benchmark RDD."""

    dataset = generate_synthetic_dataset(
        num_records=20,
        num_features=2,
        num_clusters=2,
    )

    probe_rdd = spark.sparkContext.parallelize([1], 1)
    rdd_type = type(probe_rdd)

    unpersist_called = False
    original_unpersist = rdd_type.unpersist

    def failing_count(self):
        raise RuntimeError("forced materialization failure")

    def tracking_unpersist(self, blocking=False):
        nonlocal unpersist_called
        unpersist_called = True
        return original_unpersist(self, blocking=blocking)

    monkeypatch.setattr(
        rdd_type,
        "count",
        failing_count,
    )
    monkeypatch.setattr(
        rdd_type,
        "unpersist",
        tracking_unpersist,
    )

    with pytest.raises(
        RuntimeError,
        match="forced materialization failure",
    ):
        _create_persisted_rdd(
            spark,
            dataset,
            num_partitions=2,
        )

    assert unpersist_called is True
