from types import SimpleNamespace

import pytest

from src.experiments.experiment_result import (
    ExperimentResult,
    create_experiment_result,
)


def _make_benchmark():
    timing = SimpleNamespace(
        partition_clustering_seconds=0.18,
        center_aggregation_seconds=0.01,
        initialization_seconds=0.19,
        global_clustering_seconds=0.27,
        final_evaluation_seconds=0.08,
        total_runtime_seconds=0.54,
    )

    clustering_result = SimpleNamespace(
        converged=True,
        iterations=2,
        sse=19853.25,
        cluster_counts=(200, 200, 200, 200, 200),
        timing=timing,
    )

    return SimpleNamespace(
        num_records=1000,
        num_features=20,
        num_clusters=5,
        num_partitions=4,
        spark_master="local[4]",
        random_seed=42,
        materialized_record_count=1000,
        clustering_result=clustering_result,
    )


def test_create_experiment_result_maps_benchmark_fields():
    result = create_experiment_result(
        _make_benchmark(),
        experiment_id="synthetic-1000-local4-run1",
        dataset_name="synthetic",
        local_max_iterations=100,
        global_max_iterations=100,
        tolerance=1e-6,
    )

    assert isinstance(result, ExperimentResult)

    assert result.experiment_id == "synthetic-1000-local4-run1"
    assert result.dataset_name == "synthetic"

    assert result.num_records == 1000
    assert result.num_features == 20
    assert result.num_clusters == 5
    assert result.num_partitions == 4

    assert result.spark_master == "local[4]"
    assert result.random_seed == 42

    assert result.local_max_iterations == 100
    assert result.global_max_iterations == 100
    assert result.tolerance == 1e-6

    assert result.materialized_record_count == 1000

    assert result.converged is True
    assert result.iterations == 2
    assert result.sse == 19853.25
    assert result.cluster_counts == (
        200,
        200,
        200,
        200,
        200,
    )

    assert result.partition_clustering_seconds == 0.18
    assert result.center_aggregation_seconds == 0.01
    assert result.initialization_seconds == 0.19
    assert result.global_clustering_seconds == 0.27
    assert result.final_evaluation_seconds == 0.08
    assert result.total_runtime_seconds == 0.54


@pytest.mark.parametrize(
    "experiment_id",
    ["", " ", "\t", "\n"],
)
def test_create_experiment_result_rejects_empty_experiment_id(
    experiment_id,
):
    with pytest.raises(
        ValueError,
        match="experiment_id cannot be empty",
    ):
        create_experiment_result(
            _make_benchmark(),
            experiment_id=experiment_id,
            dataset_name="synthetic",
            local_max_iterations=100,
            global_max_iterations=100,
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    "dataset_name",
    ["", " ", "\t", "\n"],
)
def test_create_experiment_result_rejects_empty_dataset_name(
    dataset_name,
):
    with pytest.raises(
        ValueError,
        match="dataset_name cannot be empty",
    ):
        create_experiment_result(
            _make_benchmark(),
            experiment_id="run-1",
            dataset_name=dataset_name,
            local_max_iterations=100,
            global_max_iterations=100,
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    "local_max_iterations",
    [0, -1],
)
def test_create_experiment_result_rejects_invalid_local_iterations(
    local_max_iterations,
):
    with pytest.raises(
        ValueError,
        match="local_max_iterations must be greater than zero",
    ):
        create_experiment_result(
            _make_benchmark(),
            experiment_id="run-1",
            dataset_name="synthetic",
            local_max_iterations=local_max_iterations,
            global_max_iterations=100,
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    "global_max_iterations",
    [0, -1],
)
def test_create_experiment_result_rejects_invalid_global_iterations(
    global_max_iterations,
):
    with pytest.raises(
        ValueError,
        match="global_max_iterations must be greater than zero",
    ):
        create_experiment_result(
            _make_benchmark(),
            experiment_id="run-1",
            dataset_name="synthetic",
            local_max_iterations=100,
            global_max_iterations=global_max_iterations,
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_create_experiment_result_rejects_invalid_tolerance(
    tolerance,
):
    with pytest.raises(ValueError):
        create_experiment_result(
            _make_benchmark(),
            experiment_id="run-1",
            dataset_name="synthetic",
            local_max_iterations=100,
            global_max_iterations=100,
            tolerance=tolerance,
        )


def test_experiment_result_is_immutable():
    result = create_experiment_result(
        _make_benchmark(),
        experiment_id="run-1",
        dataset_name="synthetic",
        local_max_iterations=100,
        global_max_iterations=100,
        tolerance=1e-6,
    )

    with pytest.raises(Exception):
        result.num_records = 2000
