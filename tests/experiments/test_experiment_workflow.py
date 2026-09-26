import csv
from dataclasses import FrozenInstanceError

import pytest

from src.experiments.experiment_runner import SyntheticExperimentConfig
from src.experiments.experiment_workflow import (
    RepeatedExperimentWorkflowResult,
    run_repeated_synthetic_experiment_workflow,
)
from src.experiments.result_summary import ExperimentSummary


def _make_config():
    return SyntheticExperimentConfig(
        experiment_id="workflow-test",
        dataset_name="synthetic-workflow-test",
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


def test_workflow_returns_raw_results_and_summary(spark):
    workflow = run_repeated_synthetic_experiment_workflow(
        spark,
        _make_config(),
        repetitions=2,
    )

    assert isinstance(
        workflow,
        RepeatedExperimentWorkflowResult,
    )
    assert len(workflow.results) == 2
    assert isinstance(
        workflow.summary,
        ExperimentSummary,
    )
    assert workflow.summary.repetitions == 2


def test_workflow_preserves_controlled_run_metadata(spark):
    workflow = run_repeated_synthetic_experiment_workflow(
        spark,
        _make_config(),
        repetitions=3,
    )

    assert tuple(
        result.experiment_id
        for result in workflow.results
    ) == (
        "workflow-test-run-01",
        "workflow-test-run-02",
        "workflow-test-run-03",
    )

    assert tuple(
        result.random_seed
        for result in workflow.results
    ) == (
        42,
        43,
        44,
    )


def test_workflow_summary_matches_raw_results(spark):
    workflow = run_repeated_synthetic_experiment_workflow(
        spark,
        _make_config(),
        repetitions=3,
    )

    expected_converged_runs = sum(
        1
        for result in workflow.results
        if result.converged
    )

    expected_mean_sse = sum(
        result.sse
        for result in workflow.results
    ) / len(workflow.results)

    expected_mean_iterations = sum(
        result.iterations
        for result in workflow.results
    ) / len(workflow.results)

    assert workflow.summary.repetitions == 3
    assert (
        workflow.summary.converged_runs
        == expected_converged_runs
    )
    assert workflow.summary.convergence_rate == pytest.approx(
        expected_converged_runs / 3
    )
    assert workflow.summary.mean_sse == pytest.approx(
        expected_mean_sse
    )
    assert workflow.summary.mean_iterations == pytest.approx(
        expected_mean_iterations
    )


def test_workflow_can_persist_every_raw_result(
    spark,
    tmp_path,
):
    output_path = tmp_path / "workflow-results.csv"

    workflow = run_repeated_synthetic_experiment_workflow(
        spark,
        _make_config(),
        repetitions=3,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    assert len(rows) == 3

    assert tuple(
        row["experiment_id"]
        for row in rows
    ) == tuple(
        result.experiment_id
        for result in workflow.results
    )

    assert tuple(
        int(row["random_seed"])
        for row in rows
    ) == tuple(
        result.random_seed
        for result in workflow.results
    )


@pytest.mark.parametrize(
    "repetitions",
    [
        0,
        -1,
    ],
)
def test_workflow_rejects_nonpositive_repetitions(
    spark,
    repetitions,
):
    with pytest.raises(
        ValueError,
        match="repetitions must be greater than zero",
    ):
        run_repeated_synthetic_experiment_workflow(
            spark,
            _make_config(),
            repetitions=repetitions,
        )


def test_workflow_result_is_immutable(spark):
    workflow = run_repeated_synthetic_experiment_workflow(
        spark,
        _make_config(),
        repetitions=1,
    )

    with pytest.raises(FrozenInstanceError):
        workflow.summary = None
