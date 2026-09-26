import csv
from dataclasses import FrozenInstanceError

import pytest

from src.experiments.experiment_matrix import (
    ExperimentMatrixCaseResult,
    ExperimentMatrixResult,
    run_synthetic_experiment_matrix,
)
from src.experiments.experiment_runner import SyntheticExperimentConfig
from src.experiments.experiment_workflow import (
    RepeatedExperimentWorkflowResult,
)


def _make_config():
    return SyntheticExperimentConfig(
        experiment_id="matrix-test",
        dataset_name="synthetic-matrix-test",
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


def test_matrix_returns_expected_result_types(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40,),
        partition_counts=(2,),
        repetitions=1,
    )

    assert isinstance(
        matrix,
        ExperimentMatrixResult,
    )
    assert len(matrix.cases) == 1
    assert isinstance(
        matrix.cases[0],
        ExperimentMatrixCaseResult,
    )
    assert isinstance(
        matrix.cases[0].workflow,
        RepeatedExperimentWorkflowResult,
    )


def test_matrix_builds_complete_cartesian_product(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40, 60),
        partition_counts=(2, 3),
        repetitions=1,
    )

    observed_cases = tuple(
        (
            case.config.num_records,
            case.config.num_partitions,
        )
        for case in matrix.cases
    )

    assert observed_cases == (
        (40, 2),
        (40, 3),
        (60, 2),
        (60, 3),
    )


def test_matrix_assigns_deterministic_case_ids(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40, 60),
        partition_counts=(2, 3),
        repetitions=1,
    )

    observed_ids = tuple(
        case.config.experiment_id
        for case in matrix.cases
    )

    assert observed_ids == (
        "matrix-test-n40-p2",
        "matrix-test-n40-p3",
        "matrix-test-n60-p2",
        "matrix-test-n60-p3",
    )


def test_matrix_preserves_base_configuration(spark):
    base_config = _make_config()

    matrix = run_synthetic_experiment_matrix(
        spark,
        base_config,
        record_counts=(60,),
        partition_counts=(3,),
        repetitions=1,
    )

    case_config = matrix.cases[0].config

    assert base_config.experiment_id == "matrix-test"
    assert base_config.num_records == 40
    assert base_config.num_partitions == 2

    assert case_config.experiment_id == "matrix-test-n60-p3"
    assert case_config.num_records == 60
    assert case_config.num_partitions == 3

    assert case_config.dataset_name == base_config.dataset_name
    assert case_config.num_features == base_config.num_features
    assert case_config.num_clusters == base_config.num_clusters
    assert case_config.cluster_spread == base_config.cluster_spread
    assert (
        case_config.center_separation
        == base_config.center_separation
    )
    assert case_config.random_seed == base_config.random_seed
    assert (
        case_config.local_max_iterations
        == base_config.local_max_iterations
    )
    assert (
        case_config.global_max_iterations
        == base_config.global_max_iterations
    )
    assert case_config.tolerance == base_config.tolerance


def test_matrix_executes_requested_repetitions_for_every_case(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40, 60),
        partition_counts=(2,),
        repetitions=2,
    )

    assert len(matrix.cases) == 2

    for case in matrix.cases:
        assert len(case.workflow.results) == 2
        assert case.workflow.summary.repetitions == 2

        assert tuple(
            result.random_seed
            for result in case.workflow.results
        ) == (
            42,
            43,
        )


def test_matrix_summaries_match_each_case(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40, 60),
        partition_counts=(2,),
        repetitions=2,
    )

    for case in matrix.cases:
        assert case.workflow.summary.repetitions == 2

        expected_converged_runs = sum(
            1
            for result in case.workflow.results
            if result.converged
        )

        assert (
            case.workflow.summary.converged_runs
            == expected_converged_runs
        )


def test_matrix_persists_separate_csv_for_each_case(
    spark,
    tmp_path,
):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40, 60),
        partition_counts=(2,),
        repetitions=2,
        output_directory=tmp_path,
    )

    expected_paths = (
        tmp_path / "matrix-test-n40-p2.csv",
        tmp_path / "matrix-test-n60-p2.csv",
    )

    assert len(matrix.cases) == 2

    for path in expected_paths:
        assert path.exists()

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            rows = list(
                csv.DictReader(csv_file)
            )

        assert len(rows) == 2


@pytest.mark.parametrize(
    "record_counts",
    [
        (),
        (0,),
        (-1,),
        (40, 0),
    ],
)
def test_matrix_rejects_invalid_record_counts(
    spark,
    record_counts,
):
    with pytest.raises(ValueError):
        run_synthetic_experiment_matrix(
            spark,
            _make_config(),
            record_counts=record_counts,
            partition_counts=(2,),
            repetitions=1,
        )


@pytest.mark.parametrize(
    "partition_counts",
    [
        (),
        (0,),
        (-1,),
        (2, 0),
    ],
)
def test_matrix_rejects_invalid_partition_counts(
    spark,
    partition_counts,
):
    with pytest.raises(ValueError):
        run_synthetic_experiment_matrix(
            spark,
            _make_config(),
            record_counts=(40,),
            partition_counts=partition_counts,
            repetitions=1,
        )


@pytest.mark.parametrize(
    "repetitions",
    [
        0,
        -1,
    ],
)
def test_matrix_rejects_nonpositive_repetitions(
    spark,
    repetitions,
):
    with pytest.raises(
        ValueError,
        match="repetitions must be greater than zero",
    ):
        run_synthetic_experiment_matrix(
            spark,
            _make_config(),
            record_counts=(40,),
            partition_counts=(2,),
            repetitions=repetitions,
        )


def test_matrix_result_is_immutable(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40,),
        partition_counts=(2,),
        repetitions=1,
    )

    with pytest.raises(FrozenInstanceError):
        matrix.cases = ()


def test_matrix_case_result_is_immutable(spark):
    matrix = run_synthetic_experiment_matrix(
        spark,
        _make_config(),
        record_counts=(40,),
        partition_counts=(2,),
        repetitions=1,
    )

    with pytest.raises(FrozenInstanceError):
        matrix.cases[0].workflow = None
