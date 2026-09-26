from dataclasses import replace
from math import sqrt

import pytest

from src.experiments.experiment_result import ExperimentResult
from src.experiments.result_summary import (
    ExperimentSummary,
    summarize_experiment_results,
)


def _make_result(
    *,
    experiment_id="summary-test",
    converged=True,
    iterations=2,
    sse=100.0,
    partition_clustering_seconds=1.0,
    center_aggregation_seconds=0.1,
    initialization_seconds=1.1,
    global_clustering_seconds=2.0,
    final_evaluation_seconds=0.2,
    total_runtime_seconds=3.3,
):
    return ExperimentResult(
        experiment_id=experiment_id,
        dataset_name="synthetic-summary-test",
        num_records=1000,
        num_features=20,
        num_clusters=5,
        num_partitions=4,
        spark_master="local[4]",
        random_seed=42,
        local_max_iterations=100,
        global_max_iterations=100,
        tolerance=1e-6,
        materialized_record_count=1000,
        converged=converged,
        iterations=iterations,
        sse=sse,
        cluster_counts=(200, 200, 200, 200, 200),
        partition_clustering_seconds=(
            partition_clustering_seconds
        ),
        center_aggregation_seconds=(
            center_aggregation_seconds
        ),
        initialization_seconds=initialization_seconds,
        global_clustering_seconds=(
            global_clustering_seconds
        ),
        final_evaluation_seconds=(
            final_evaluation_seconds
        ),
        total_runtime_seconds=total_runtime_seconds,
    )


def test_summary_returns_experiment_summary():
    summary = summarize_experiment_results(
        (_make_result(),)
    )

    assert isinstance(summary, ExperimentSummary)


def test_summary_records_repetition_and_convergence_counts():
    results = (
        _make_result(
            experiment_id="run-01",
            converged=True,
        ),
        _make_result(
            experiment_id="run-02",
            converged=False,
        ),
        _make_result(
            experiment_id="run-03",
            converged=True,
        ),
    )

    summary = summarize_experiment_results(results)

    assert summary.repetitions == 3
    assert summary.converged_runs == 2
    assert summary.convergence_rate == pytest.approx(
        2 / 3
    )


def test_summary_calculates_iteration_statistics():
    results = (
        _make_result(
            experiment_id="run-01",
            iterations=2,
        ),
        _make_result(
            experiment_id="run-02",
            iterations=4,
        ),
        _make_result(
            experiment_id="run-03",
            iterations=6,
        ),
    )

    summary = summarize_experiment_results(results)

    assert summary.mean_iterations == pytest.approx(4.0)
    assert summary.std_iterations == pytest.approx(2.0)


def test_summary_calculates_sse_statistics():
    results = (
        _make_result(
            experiment_id="run-01",
            sse=100.0,
        ),
        _make_result(
            experiment_id="run-02",
            sse=110.0,
        ),
        _make_result(
            experiment_id="run-03",
            sse=120.0,
        ),
    )

    summary = summarize_experiment_results(results)

    assert summary.mean_sse == pytest.approx(110.0)
    assert summary.std_sse == pytest.approx(10.0)


def test_summary_calculates_timing_statistics():
    first = _make_result(
        experiment_id="run-01",
        partition_clustering_seconds=1.0,
        center_aggregation_seconds=0.1,
        initialization_seconds=1.1,
        global_clustering_seconds=2.0,
        final_evaluation_seconds=0.2,
        total_runtime_seconds=3.3,
    )

    second = replace(
        first,
        experiment_id="run-02",
        partition_clustering_seconds=3.0,
        center_aggregation_seconds=0.3,
        initialization_seconds=3.3,
        global_clustering_seconds=4.0,
        final_evaluation_seconds=0.4,
        total_runtime_seconds=7.7,
    )

    summary = summarize_experiment_results(
        (first, second)
    )

    assert summary.mean_partition_clustering_seconds == (
        pytest.approx(2.0)
    )
    assert summary.std_partition_clustering_seconds == (
        pytest.approx(sqrt(2.0))
    )

    assert summary.mean_center_aggregation_seconds == (
        pytest.approx(0.2)
    )
    assert summary.std_center_aggregation_seconds == (
        pytest.approx(sqrt(0.02))
    )

    assert summary.mean_initialization_seconds == (
        pytest.approx(2.2)
    )
    assert summary.std_initialization_seconds == (
        pytest.approx(sqrt(2.42))
    )

    assert summary.mean_global_clustering_seconds == (
        pytest.approx(3.0)
    )
    assert summary.std_global_clustering_seconds == (
        pytest.approx(sqrt(2.0))
    )

    assert summary.mean_final_evaluation_seconds == (
        pytest.approx(0.3)
    )
    assert summary.std_final_evaluation_seconds == (
        pytest.approx(sqrt(0.02))
    )

    assert summary.mean_total_runtime_seconds == (
        pytest.approx(5.5)
    )
    assert summary.std_total_runtime_seconds == (
        pytest.approx(sqrt(9.68))
    )


def test_single_result_reports_zero_standard_deviation():
    result = _make_result()

    summary = summarize_experiment_results(
        (result,)
    )

    assert summary.std_iterations == 0.0
    assert summary.std_sse == 0.0

    assert summary.std_partition_clustering_seconds == 0.0
    assert summary.std_center_aggregation_seconds == 0.0
    assert summary.std_initialization_seconds == 0.0
    assert summary.std_global_clustering_seconds == 0.0
    assert summary.std_final_evaluation_seconds == 0.0
    assert summary.std_total_runtime_seconds == 0.0


def test_summary_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match="results must contain at least one experiment result",
    ):
        summarize_experiment_results(())


def test_experiment_summary_is_immutable():
    summary = summarize_experiment_results(
        (_make_result(),)
    )

    with pytest.raises(Exception):
        summary.repetitions = 10
