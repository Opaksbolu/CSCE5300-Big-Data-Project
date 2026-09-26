import csv

import pytest

from src.experiments.experiment_result import ExperimentResult
from src.experiments.experiment_runner import SyntheticExperimentConfig
from src.experiments.repeated_runner import (
    run_repeated_synthetic_experiments,
)


def _make_config():
    return SyntheticExperimentConfig(
        experiment_id="repeated-test",
        dataset_name="synthetic-repeated-test",
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


def test_repeated_runner_returns_requested_number_of_results(
    spark,
):
    results = run_repeated_synthetic_experiments(
        spark,
        _make_config(),
        repetitions=3,
    )

    assert isinstance(results, tuple)
    assert len(results) == 3

    assert all(
        isinstance(result, ExperimentResult)
        for result in results
    )


def test_repeated_runner_assigns_unique_experiment_ids(
    spark,
):
    results = run_repeated_synthetic_experiments(
        spark,
        _make_config(),
        repetitions=3,
    )

    assert tuple(
        result.experiment_id
        for result in results
    ) == (
        "repeated-test-run-01",
        "repeated-test-run-02",
        "repeated-test-run-03",
    )


def test_repeated_runner_uses_deterministic_incrementing_seeds(
    spark,
):
    results = run_repeated_synthetic_experiments(
        spark,
        _make_config(),
        repetitions=3,
    )

    assert tuple(
        result.random_seed
        for result in results
    ) == (
        42,
        43,
        44,
    )


def test_repeated_runner_preserves_base_configuration(
    spark,
):
    config = _make_config()

    results = run_repeated_synthetic_experiments(
        spark,
        config,
        repetitions=2,
    )

    for result in results:
        assert result.dataset_name == config.dataset_name
        assert result.num_records == config.num_records
        assert result.num_features == config.num_features
        assert result.num_clusters == config.num_clusters
        assert result.num_partitions == config.num_partitions
        assert result.local_max_iterations == (
            config.local_max_iterations
        )
        assert result.global_max_iterations == (
            config.global_max_iterations
        )
        assert result.tolerance == config.tolerance


def test_repeated_runner_persists_every_result(
    spark,
    tmp_path,
):
    output_path = tmp_path / "repeated-results.csv"

    results = run_repeated_synthetic_experiments(
        spark,
        _make_config(),
        repetitions=3,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3

    assert tuple(
        row["experiment_id"]
        for row in rows
    ) == tuple(
        result.experiment_id
        for result in results
    )

    assert tuple(
        int(row["random_seed"])
        for row in rows
    ) == (
        42,
        43,
        44,
    )


@pytest.mark.parametrize(
    "repetitions",
    [
        0,
        -1,
    ],
)
def test_repeated_runner_rejects_nonpositive_repetitions(
    spark,
    repetitions,
):
    with pytest.raises(
        ValueError,
        match="repetitions must be greater than zero",
    ):
        run_repeated_synthetic_experiments(
            spark,
            _make_config(),
            repetitions=repetitions,
        )
