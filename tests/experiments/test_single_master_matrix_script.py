import argparse
from pathlib import Path

import pytest

from scripts.run_single_master_matrix import (
    build_parser,
    parse_positive_int_csv,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", (1,)),
        ("1000", (1000,)),
        ("1000,10000", (1000, 10000)),
        ("2,4,8", (2, 4, 8)),
        (" 1000 , 10000 ", (1000, 10000)),
    ],
)
def test_parse_positive_int_csv_accepts_valid_values(
    value,
    expected,
):
    assert parse_positive_int_csv(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "0",
        "-1",
        "1000,0",
        "0,1000",
        "1000,-1",
        "abc",
        "1000,abc",
        "1000,,2000",
    ],
)
def test_parse_positive_int_csv_rejects_invalid_values(
    value,
):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_int_csv(value)


def test_parser_requires_master():
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args([])

    assert error.value.code == 2


def test_parser_accepts_single_master_configuration():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--master",
            "local[1]",
            "--record-counts",
            "1000,10000",
            "--partition-counts",
            "2,4",
            "--repetitions",
            "3",
            "--output-directory",
            "results/raw/test",
            "--log-level",
            "ERROR",
        ]
    )

    assert args.master == "local[1]"
    assert args.record_counts == (1000, 10000)
    assert args.partition_counts == (2, 4)
    assert args.repetitions == 3
    assert args.output_directory == Path("results/raw/test")
    assert args.log_level == "ERROR"


def test_parser_uses_expected_defaults():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--master",
            "local[2]",
        ]
    )

    assert args.master == "local[2]"
    assert args.record_counts == (1000,)
    assert args.partition_counts == (4,)
    assert args.repetitions == 1
    assert args.output_directory == Path(
        "results/raw/cross-master"
    )
    assert args.log_level == "WARN"
