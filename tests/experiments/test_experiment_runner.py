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
        local_tolerance=1e-4,
        global_cost_tolerance=0.25,
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
    assert result.local_tolerance == config.local_tolerance
    assert result.global_cost_tolerance == (
        config.global_cost_tolerance
    )

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



def test_run_synthetic_experiment_records_aggregation_restarts(
    spark,
):
    """Experiment results should preserve aggregation restart metadata."""

    config = SyntheticExperimentConfig(
        experiment_id="aggregation-restart-test",
        num_records=40,
        num_features=2,
        num_clusters=2,
        num_partitions=2,
        cluster_spread=0.25,
        center_separation=10.0,
        random_seed=42,
        local_max_iterations=20,
        global_max_iterations=20,
        local_tolerance=1e-4,
        global_cost_tolerance=0.25,
        aggregation_restarts=5,
    )

    result = run_synthetic_experiment(
        spark,
        config,
    )

    assert result.aggregation_restarts == 5


def test_file_backed_synthetic_experiment_returns_standardized_result(
    spark,
    tmp_path,
):
    from src.experiments.experiment_runner import (
        run_file_backed_synthetic_experiment,
    )

    config = _make_config(
        experiment_id="file-backed-run",
    )
    dataset_path = tmp_path / "synthetic-points.csv"

    result = run_file_backed_synthetic_experiment(
        spark,
        config,
        dataset_path=dataset_path,
    )

    assert isinstance(result, ExperimentResult)
    assert dataset_path.exists()

    assert result.experiment_id == config.experiment_id
    assert result.dataset_name == config.dataset_name
    assert result.num_records == config.num_records
    assert result.materialized_record_count == config.num_records
    assert result.num_features == config.num_features
    assert result.num_clusters == config.num_clusters
    assert result.num_partitions >= config.num_partitions
    assert result.random_seed == config.random_seed
    assert result.aggregation_restarts == (
        config.aggregation_restarts
    )

    assert sum(result.cluster_counts) == config.num_records
    assert len(result.cluster_counts) == config.num_clusters
    assert result.converged is True
    assert result.iterations > 0
    assert result.sse >= 0.0


def test_file_backed_matches_in_memory_quality(
    spark,
    tmp_path,
):
    from src.experiments.experiment_runner import (
        run_file_backed_synthetic_experiment,
    )

    config = _make_config(
        experiment_id="quality-equivalence",
    )

    in_memory = run_synthetic_experiment(
        spark,
        config,
    )

    file_backed = run_file_backed_synthetic_experiment(
        spark,
        config,
        dataset_path=tmp_path / "equivalent-points.csv",
    )

    assert file_backed.num_records == in_memory.num_records
    assert file_backed.random_seed == in_memory.random_seed
    assert file_backed.cluster_counts == in_memory.cluster_counts
    assert file_backed.iterations == in_memory.iterations
    assert file_backed.sse == in_memory.sse


def test_file_backed_synthetic_experiment_can_persist_result(
    spark,
    tmp_path,
):
    from src.experiments.experiment_runner import (
        run_file_backed_synthetic_experiment,
    )

    config = _make_config(
        experiment_id="file-backed-persisted",
    )
    dataset_path = tmp_path / "points.csv"
    output_path = tmp_path / "results.csv"

    result = run_file_backed_synthetic_experiment(
        spark,
        config,
        dataset_path=dataset_path,
        output_path=output_path,
    )

    assert dataset_path.exists()
    assert output_path.exists()

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["experiment_id"] == result.experiment_id
    assert rows[0]["num_records"] == str(result.num_records)
    assert rows[0]["sse"] == str(result.sse)


def test_file_backed_synthetic_experiment_is_reproducible(
    spark,
    tmp_path,
):
    from src.experiments.experiment_runner import (
        run_file_backed_synthetic_experiment,
    )

    config = _make_config(
        experiment_id="file-backed-reproducible",
    )

    first = run_file_backed_synthetic_experiment(
        spark,
        config,
        dataset_path=tmp_path / "first.csv",
    )

    second = run_file_backed_synthetic_experiment(
        spark,
        config,
        dataset_path=tmp_path / "second.csv",
    )

    assert first.cluster_counts == second.cluster_counts
    assert first.iterations == second.iterations
    assert first.sse == second.sse

    assert (
        (tmp_path / "first.csv").read_text()
        == (tmp_path / "second.csv").read_text()
    )
