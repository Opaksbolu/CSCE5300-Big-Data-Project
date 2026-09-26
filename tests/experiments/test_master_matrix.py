"""Tests for Spark master experiment metadata helpers."""

from dataclasses import FrozenInstanceError

import pytest

from src.experiments.experiment_matrix import ExperimentMatrixResult
from src.experiments.master_matrix import (
    MasterMatrixCaseResult,
    MasterMatrixResult,
    master_output_name,
    validate_spark_masters,
)


@pytest.mark.parametrize(
    ("master", "expected"),
    [
        ("local[1]", "local-1"),
        ("local[2]", "local-2"),
        ("local[4]", "local-4"),
        ("spark://host:7077", "spark---host-7077"),
    ],
)
def test_master_output_name_is_filesystem_safe(
    master,
    expected,
):
    assert master_output_name(master) == expected


@pytest.mark.parametrize(
    "master",
    [
        "",
        " ",
        "\t",
        "\n",
        None,
    ],
)
def test_master_output_name_rejects_invalid_master(master):
    with pytest.raises(ValueError):
        master_output_name(master)


@pytest.mark.parametrize(
    "spark_masters",
    [
        ("local[1]",),
        ("local[1]", "local[2]"),
        ("local[1]", "local[2]", "local[4]"),
    ],
)
def test_validate_spark_masters_accepts_valid_masters(
    spark_masters,
):
    validate_spark_masters(spark_masters)


@pytest.mark.parametrize(
    "spark_masters",
    [
        (),
        ("",),
        (" ",),
        ("\t",),
        ("\n",),
        ("local[1]", ""),
        ("local[1]", None),
    ],
)
def test_validate_spark_masters_rejects_invalid_masters(
    spark_masters,
):
    with pytest.raises(ValueError):
        validate_spark_masters(spark_masters)


def test_master_matrix_result_preserves_cases():
    matrix = ExperimentMatrixResult(cases=())

    case = MasterMatrixCaseResult(
        spark_master="local[1]",
        matrix=matrix,
    )

    result = MasterMatrixResult(
        cases=(case,),
    )

    assert result.cases == (case,)
    assert result.cases[0].spark_master == "local[1]"
    assert result.cases[0].matrix is matrix


def test_master_matrix_result_is_immutable():
    result = MasterMatrixResult(cases=())

    with pytest.raises(FrozenInstanceError):
        result.cases = ()


def test_master_matrix_case_result_is_immutable():
    case = MasterMatrixCaseResult(
        spark_master="local[1]",
        matrix=ExperimentMatrixResult(cases=()),
    )

    with pytest.raises(FrozenInstanceError):
        case.spark_master = "local[4]"
