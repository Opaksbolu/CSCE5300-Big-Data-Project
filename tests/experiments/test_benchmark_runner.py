"""
Tests for controlled Parallel K-Means benchmark execution.
"""

from __future__ import annotations

import pytest

from src.experiments.benchmark_runner import (
    _create_persisted_rdd,
    run_file_parallel_kmeans_benchmark,
    run_parallel_kmeans_benchmark,
    run_persisted_rdd_benchmark,
)
from src.experiments.file_dataset import FileDataset
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



def test_benchmark_propagates_aggregation_restarts(spark) -> None:
    """Benchmark should accept the aggregation restart configuration."""

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
        aggregation_restarts=5,
    )

    assert result.aggregation_restarts == 5


def test_persisted_rdd_benchmark_executes_complete_pipeline(
    spark,
) -> None:
    """A materialized RDD should use the shared benchmark path."""

    points = [
        (0.0, 0.0),
        (0.1, 0.1),
        (10.0, 10.0),
        (10.1, 10.1),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )
    rdd.persist()

    try:
        materialized_record_count = rdd.count()

        result = run_persisted_rdd_benchmark(
            spark,
            rdd,
            num_records=4,
            num_features=2,
            num_clusters=2,
            materialized_record_count=materialized_record_count,
            random_seed=42,
            tolerance=1e-12,
        )

        assert result.num_records == 4
        assert result.num_features == 2
        assert result.num_clusters == 2
        assert result.num_partitions == 2
        assert result.materialized_record_count == 4
        assert result.spark_master == spark.sparkContext.master
        assert result.random_seed == 42
        assert result.aggregation_restarts == 5

        assert result.clustering_result.converged
        assert sum(
            result.clustering_result.cluster_counts
        ) == 4

    finally:
        rdd.unpersist()


def test_persisted_rdd_benchmark_does_not_own_cleanup(
    spark,
) -> None:
    """The shared benchmark must not unpersist a caller-owned RDD."""

    points = [
        (0.0, 0.0),
        (0.1, 0.1),
        (10.0, 10.0),
        (10.1, 10.1),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )
    rdd.persist()

    try:
        materialized_record_count = rdd.count()

        run_persisted_rdd_benchmark(
            spark,
            rdd,
            num_records=4,
            num_features=2,
            num_clusters=2,
            materialized_record_count=materialized_record_count,
            random_seed=42,
        )

        assert rdd.is_cached

    finally:
        rdd.unpersist()


def test_file_benchmark_executes_complete_pipeline(
    spark,
    tmp_path,
) -> None:
    """A file-backed dataset should use the complete benchmark path."""

    dataset_path = tmp_path / "points.csv"

    points = [
        (float(index), float(index))
        for index in range(20)
    ] + [
        (
            100.0 + float(index),
            100.0 + float(index),
        )
        for index in range(20)
    ]

    with dataset_path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(
                ",".join(repr(value) for value in point)
            )
            handle.write("\n")

    dataset = FileDataset(
        path=dataset_path,
        num_records=40,
        num_features=2,
        num_clusters=2,
    )

    result = run_file_parallel_kmeans_benchmark(
        spark,
        dataset,
        num_partitions=2,
        random_seed=42,
        tolerance=1e-12,
        aggregation_restarts=5,
    )

    assert result.num_records == 40
    assert result.num_features == 2
    assert result.num_clusters == 2
    assert result.num_partitions >= 2
    assert result.materialized_record_count == 40
    assert result.spark_master == spark.sparkContext.master
    assert result.random_seed == 42
    assert result.aggregation_restarts == 5

    assert result.clustering_result.converged
    assert len(result.clustering_result.centers) == 2
    assert sum(
        result.clustering_result.cluster_counts
    ) == 40


def test_file_benchmark_unpersists_after_clustering_failure(
    spark,
    tmp_path,
    monkeypatch,
) -> None:
    """The file-backed wrapper should clean up after failure."""

    dataset_path = tmp_path / "points.csv"
    dataset_path.write_text(
        "0.0,0.0\n"
        "0.1,0.1\n"
        "10.0,10.0\n"
        "10.1,10.1\n",
        encoding="utf-8",
    )

    dataset = FileDataset(
        path=dataset_path,
        num_records=4,
        num_features=2,
        num_clusters=2,
    )

    probe_rdd = spark.sparkContext.parallelize([1], 1)
    rdd_type = type(probe_rdd)

    unpersist_called = False
    original_unpersist = rdd_type.unpersist

    def tracking_unpersist(self, blocking=False):
        nonlocal unpersist_called
        unpersist_called = True
        return original_unpersist(
            self,
            blocking=blocking,
        )

    def failing_benchmark(*args, **kwargs):
        raise RuntimeError("forced clustering failure")

    monkeypatch.setattr(
        rdd_type,
        "unpersist",
        tracking_unpersist,
    )

    monkeypatch.setattr(
        "src.experiments.benchmark_runner."
        "run_persisted_rdd_benchmark",
        failing_benchmark,
    )

    with pytest.raises(
        RuntimeError,
        match="forced clustering failure",
    ):
        run_file_parallel_kmeans_benchmark(
            spark,
            dataset,
            num_partitions=2,
        )

    assert unpersist_called is True
