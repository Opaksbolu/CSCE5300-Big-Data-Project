import argparse
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_cross_master_matrix import (
    DEFAULT_MASTERS,
    build_child_command,
    build_parser,
    format_positive_int_csv,
    parse_master_csv,
    run_cross_master_processes,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("local[1]", ("local[1]",)),
        (
            "local[1],local[2]",
            ("local[1]", "local[2]"),
        ),
        (
            "local[1],local[2],local[4]",
            ("local[1]", "local[2]", "local[4]"),
        ),
        (
            " local[1] , local[4] ",
            ("local[1]", "local[4]"),
        ),
    ],
)
def test_parse_master_csv_accepts_valid_values(
    value,
    expected,
):
    assert parse_master_csv(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        ",",
        "local[1],",
        ",local[1]",
        "local[1], ,local[2]",
    ],
)
def test_parse_master_csv_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_master_csv(value)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((1,), "1"),
        ((1000,), "1000"),
        ((1000, 10000), "1000,10000"),
        ((2, 4, 8), "2,4,8"),
    ],
)
def test_format_positive_int_csv_accepts_valid_values(
    values,
    expected,
):
    assert format_positive_int_csv(values) == expected


@pytest.mark.parametrize(
    "values",
    [
        (),
        (0,),
        (-1,),
        (1000, 0),
        (True,),
        (1.5,),
        ("1000",),
    ],
)
def test_format_positive_int_csv_rejects_invalid_values(
    values,
):
    with pytest.raises(ValueError):
        format_positive_int_csv(values)


def test_build_child_command_uses_module_execution():
    command = build_child_command(
        python_executable="/example/python",
        master="local[2]",
        record_counts=(1000, 10000),
        partition_counts=(2, 4),
        repetitions=3,
        warmup_runs=1,
        aggregation_restarts=5,
        output_directory=Path("results/raw/example"),
        log_level="ERROR",
    )

    assert command == [
        "/example/python",
        "-m",
        "scripts.run_single_master_matrix",
        "--master",
        "local[2]",
        "--record-counts",
        "1000,10000",
        "--partition-counts",
        "2,4",
                "--repetitions",
        "3",
        "--warmup-runs",
        "1",
        "--aggregation-restarts",
        "5",
        "--output-directory",
        "results/raw/example",
        "--log-level",
        "ERROR",
    ]


@pytest.mark.parametrize(
    "python_executable",
    [
        "",
        " ",
    ],
)
def test_build_child_command_rejects_invalid_python_executable(
    python_executable,
):
    with pytest.raises(ValueError):
        build_child_command(
            python_executable=python_executable,
            master="local[1]",
            record_counts=(1000,),
            partition_counts=(4,),
            repetitions=1,
            warmup_runs=1,
            aggregation_restarts=5,
            output_directory=Path("results/raw/example"),
            log_level="WARN",
        )


@pytest.mark.parametrize(
    "repetitions",
    [
        0,
        -1,
    ],
)
def test_build_child_command_rejects_invalid_repetitions(
    repetitions,
):
    with pytest.raises(ValueError):
        build_child_command(
            python_executable="/example/python",
            master="local[1]",
            record_counts=(1000,),
            partition_counts=(4,),
            repetitions=repetitions,
            warmup_runs=1,
            aggregation_restarts=5,
            output_directory=Path("results/raw/example"),
            log_level="WARN",
        )


@pytest.mark.parametrize(
    "log_level",
    [
        "",
        " ",
    ],
)
def test_build_child_command_rejects_invalid_log_level(
    log_level,
):
    with pytest.raises(ValueError):
        build_child_command(
            python_executable="/example/python",
            master="local[1]",
            record_counts=(1000,),
            partition_counts=(4,),
            repetitions=1,
            warmup_runs=1,
            aggregation_restarts=5,
            output_directory=Path("results/raw/example"),
            log_level=log_level,
        )


def test_cross_master_runner_launches_one_process_per_master(
    monkeypatch,
):
    calls = []

    def fake_run(command, *, check):
        calls.append(
            (command, check)
        )
        return subprocess.CompletedProcess(
            command,
            0,
        )

    monkeypatch.setattr(
        "scripts.run_cross_master_matrix.subprocess.run",
        fake_run,
    )

    run_cross_master_processes(
        spark_masters=(
            "local[1]",
            "local[2]",
            "local[4]",
        ),
        record_counts=(1000,),
        partition_counts=(4,),
        repetitions=2,
        warmup_runs=1,
        aggregation_restarts=5,
        output_directory=Path(
            "results/raw/cross-master-test"
        ),
        log_level="WARN",
        python_executable="/example/python",
    )

    assert len(calls) == 3

    assert [
        command[
            command.index("--master") + 1
        ]
        for command, _ in calls
    ] == [
        "local[1]",
        "local[2]",
        "local[4]",
    ]

    assert all(
        check is True
        for _, check in calls
    )


def test_cross_master_runner_uses_common_output_root(
    monkeypatch,
):
    commands = []

    def fake_run(command, *, check):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
        )

    monkeypatch.setattr(
        "scripts.run_cross_master_matrix.subprocess.run",
        fake_run,
    )

    output_root = Path(
        "results/raw/common-root"
    )

    run_cross_master_processes(
        spark_masters=DEFAULT_MASTERS,
        record_counts=(1000,),
        partition_counts=(4,),
        repetitions=1,
        warmup_runs=1,
        aggregation_restarts=5,
        output_directory=output_root,
        log_level="WARN",
        python_executable="/example/python",
    )

    for command in commands:
        output_index = (
            command.index("--output-directory") + 1
        )

        assert command[output_index] == str(
            output_root
        )


def test_cross_master_runner_uses_check_true(
    monkeypatch,
):
    observed_checks = []

    def fake_run(command, *, check):
        observed_checks.append(check)
        return subprocess.CompletedProcess(
            command,
            0,
        )

    monkeypatch.setattr(
        "scripts.run_cross_master_matrix.subprocess.run",
        fake_run,
    )

    run_cross_master_processes(
        spark_masters=("local[1]",),
        record_counts=(1000,),
        partition_counts=(4,),
        repetitions=1,
        warmup_runs=1,
        aggregation_restarts=5,
        output_directory=Path(
            "results/raw/test"
        ),
        log_level="WARN",
        python_executable="/example/python",
    )

    assert observed_checks == [True]


def test_cross_master_runner_stops_after_child_failure(
    monkeypatch,
):
    calls = []

    def fake_run(command, *, check):
        calls.append(command)

        if len(calls) == 2:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=command,
            )

        return subprocess.CompletedProcess(
            command,
            0,
        )

    monkeypatch.setattr(
        "scripts.run_cross_master_matrix.subprocess.run",
        fake_run,
    )

    with pytest.raises(
        subprocess.CalledProcessError
    ):
        run_cross_master_processes(
            spark_masters=DEFAULT_MASTERS,
            record_counts=(1000,),
            partition_counts=(4,),
            repetitions=1,
            warmup_runs=1,
            aggregation_restarts=5,
            output_directory=Path(
                "results/raw/test"
            ),
            log_level="WARN",
            python_executable="/example/python",
        )

    assert len(calls) == 2


def test_parser_uses_expected_defaults():
    parser = build_parser()

    args = parser.parse_args([])

    assert args.masters == DEFAULT_MASTERS
    assert args.record_counts == (1000,)
    assert args.partition_counts == (4,)
    assert args.repetitions == 1
    assert args.output_directory == Path(
        "results/raw/cross-master"
    )
    assert args.log_level == "WARN"


def test_parser_accepts_custom_configuration():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--masters",
            "local[1],local[4]",
            "--record-counts",
            "1000,10000",
            "--partition-counts",
            "2,4",
            "--repetitions",
            "3",
            "--output-directory",
            "results/raw/custom",
            "--log-level",
            "ERROR",
        ]
    )

    assert args.masters == (
        "local[1]",
        "local[4]",
    )
    assert args.record_counts == (
        1000,
        10000,
    )
    assert args.partition_counts == (2, 4)
    assert args.repetitions == 3
    assert args.output_directory == Path(
        "results/raw/custom"
    )
    assert args.log_level == "ERROR"


def test_parser_accepts_warmup_and_aggregation_restarts():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--warmup-runs",
            "2",
            "--aggregation-restarts",
            "7",
        ]
    )

    assert args.warmup_runs == 2
    assert args.aggregation_restarts == 7


def test_parser_uses_expected_benchmark_defaults():
    parser = build_parser()

    args = parser.parse_args([])

    assert args.warmup_runs == 1
    assert args.aggregation_restarts == 5


@pytest.mark.parametrize("warmup_runs", [-1, -2])
def test_build_child_command_rejects_negative_warmup_runs(
    warmup_runs,
):
    with pytest.raises(
        ValueError,
        match="warmup_runs must be zero or greater",
    ):
        build_child_command(
            python_executable=sys.executable,
            master="local[2]",
            record_counts=(1000,),
            partition_counts=(4,),
            repetitions=1,
            warmup_runs=warmup_runs,
            aggregation_restarts=5,
            output_directory=Path("results/raw/test"),
            log_level="WARN",
        )


@pytest.mark.parametrize("aggregation_restarts", [0, -1])
def test_build_child_command_rejects_nonpositive_aggregation_restarts(
    aggregation_restarts,
):
    with pytest.raises(
        ValueError,
        match="aggregation_restarts must be greater than zero",
    ):
        build_child_command(
            python_executable=sys.executable,
            master="local[2]",
            record_counts=(1000,),
            partition_counts=(4,),
            repetitions=1,
            warmup_runs=1,
            aggregation_restarts=aggregation_restarts,
            output_directory=Path("results/raw/test"),
            log_level="WARN",
        )
