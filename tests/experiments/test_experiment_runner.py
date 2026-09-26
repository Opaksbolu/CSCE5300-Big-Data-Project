import csv

from src.experiments.experiment_runner import (
    SyntheticExperimentConfig,
    run_synthetic_experiment,
)
from src.experiments.experiment_result import ExperimentResult


def _make_config(
    *,
    experiment_id="synthetic-test-run",
):
    return SyntheticExperimentConfig(
        experiment_id=experiment_id,
        num_records=40,
        num_features=2,
        num_clusters=2,
        num_partitions=2,
        cluster_spread=0.25,
        center_separation=10.0,
        random_seed=42,
        local_max_iterations=20,
        global_max_iterations=20,
        tolerance=1e-6,
    )


def test_run_synthetic_experiment_returns_standardized_result(
    spark,
):
    config = _make_config()

    result = run_synthetic_experiment(
        spark,
        config,
    )

    assert isinstance(result, ExperimentResult)

    assert result.experiment_id == config.experiment_id
    assert result.dataset_name == config.dataset_name

    assert result.num_records == config.num_records
    assert result.materialized_record_count == config.num_records
    assert result.num_features == config.num_features
    assert result.num_clusters == config.num_clusters
    assert result.num_partitions == config.num_partitions

    assert result.random_seed == config.random_seed

    assert result.local_max_iterations == (
        config.local_max_iterations
    )
    assert result.global_max_iterations == (
        config.global_max_iterations
    )
    assert result.tolerance == config.tolerance

    assert sum(result.cluster_counts) == config.num_records
    assert len(result.cluster_counts) == config.num_clusters

    assert result.converged is True
    assert result.iterations > 0
    assert result.sse >= 0.0


def test_run_synthetic_experiment_records_spark_master(
    spark,
):
    result = run_synthetic_experiment(
        spark,
        _make_config(),
    )

    assert result.spark_master == spark.sparkContext.master


def test_run_synthetic_experiment_reports_nonnegative_timings(
    spark,
):
    result = run_synthetic_experiment(
        spark,
        _make_config(),
    )

    assert result.partition_clustering_seconds >= 0.0
    assert result.center_aggregation_seconds >= 0.0
    assert result.initialization_seconds >= 0.0
    assert result.global_clustering_seconds >= 0.0
    assert result.final_evaluation_seconds >= 0.0
    assert result.total_runtime_seconds >= 0.0


def test_run_synthetic_experiment_can_persist_result(
    spark,
    tmp_path,
):
    output_path = tmp_path / "results.csv"
    config = _make_config(
        experiment_id="persisted-run",
    )

    result = run_synthetic_experiment(
        spark,
        config,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1

    assert rows[0]["experiment_id"] == result.experiment_id
    assert rows[0]["dataset_name"] == result.dataset_name
    assert rows[0]["num_records"] == str(result.num_records)
    assert rows[0]["sse"] == str(result.sse)


def test_run_synthetic_experiment_is_reproducible_in_quality(
    spark,
):
    config = _make_config()

    first = run_synthetic_experiment(
        spark,
        config,
    )

    second = run_synthetic_experiment(
        spark,
        config,
    )

    assert first.num_records == second.num_records
    assert first.cluster_counts == second.cluster_counts
    assert first.iterations == second.iterations
    assert first.sse == second.sse
