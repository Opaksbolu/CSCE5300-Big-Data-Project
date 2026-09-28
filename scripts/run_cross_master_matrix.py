"""
Launch controlled synthetic experiment matrices across Spark masters.

Each Spark master is executed in a separate child Python process.
The parent process never creates a Spark session. This preserves
isolation between Spark application contexts while allowing one
command to coordinate local[1], local[2], local[4], or other masters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.experiments.master_matrix import (
    validate_spark_masters,
)
from scripts.run_single_master_matrix import (
    parse_positive_int_csv,
)


DEFAULT_MASTERS = (
    "local[1]",
    "local[2]",
    "local[4]",
)


def parse_master_csv(value: str) -> tuple[str, ...]:
    """Parse and validate comma-separated Spark masters."""

    masters = tuple(
        item.strip()
        for item in value.split(",")
    )

    try:
        validate_spark_masters(masters)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            str(exc)
        ) from exc

    return masters


def format_positive_int_csv(
    values: tuple[int, ...],
) -> str:
    """Convert positive integers into CLI CSV form."""

    if not values or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in values
    ):
        raise ValueError(
            "values must contain only positive integers."
        )

    return ",".join(
        str(value)
        for value in values
    )


def build_child_command(
    *,
    python_executable: str,
    master: str,
    record_counts: tuple[int, ...],
    partition_counts: tuple[int, ...],
    repetitions: int,
    warmup_runs: int,
    aggregation_restarts: int,
    input_mode: str = "memory",
    dataset_directory: Path = Path("data/generated/cross-master"),
    output_directory: Path,
    log_level: str,
) -> list[str]:
    """Build one isolated single-master child command."""

    if not isinstance(python_executable, str) or not (
        python_executable.strip()
    ):
        raise ValueError(
            "python_executable must be a nonempty string."
        )

    validate_spark_masters((master,))

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be greater than zero."
        )
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs must be zero or greater."
        )

    if aggregation_restarts <= 0:
        raise ValueError(
            "aggregation_restarts must be greater than zero."
        )
    if input_mode not in {"memory", "file"}:
        raise ValueError(
            "input_mode must be 'memory' or 'file'."
        )
    if not isinstance(log_level, str) or not log_level.strip():
        raise ValueError(
            "log_level must be a nonempty string."
        )

    return [
        python_executable,
        "-m",
        "scripts.run_single_master_matrix",
        "--master",
        master,
        "--record-counts",
        format_positive_int_csv(record_counts),
        "--partition-counts",
        format_positive_int_csv(partition_counts),
        "--repetitions",
        str(repetitions),
        "--warmup-runs",
        str(warmup_runs),
        "--aggregation-restarts",
        str(aggregation_restarts),
        "--input-mode",
        input_mode,
        "--dataset-directory",
        str(dataset_directory),
        "--output-directory",
        str(output_directory),
        "--log-level",
        log_level,
    ]


def run_cross_master_processes(
    *,
    spark_masters: tuple[str, ...],
    record_counts: tuple[int, ...],
    partition_counts: tuple[int, ...],
    repetitions: int,
    warmup_runs: int,
    aggregation_restarts: int,
    input_mode: str = "memory",
    dataset_directory: Path = Path("data/generated/cross-master"),
    output_directory: Path,
    log_level: str,
    python_executable: str = sys.executable,
) -> None:
    """Run each Spark master sequentially in its own process."""

    validate_spark_masters(spark_masters)

    for index, master in enumerate(
        spark_masters,
        start=1,
    ):
        command = build_child_command(
            python_executable=python_executable,
            master=master,
            record_counts=record_counts,
            partition_counts=partition_counts,
            repetitions=repetitions,
            warmup_runs=warmup_runs,
            aggregation_restarts=aggregation_restarts,
            input_mode=input_mode,
            dataset_directory=dataset_directory,
            output_directory=output_directory,
            log_level=log_level,
        )

        print(
            f"[{index}/{len(spark_masters)}] "
            f"Starting isolated Spark master {master}."
        )

        subprocess.run(
            command,
            check=True,
        )

        print(
            f"[{index}/{len(spark_masters)}] "
            f"Completed isolated Spark master {master}."
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the cross-master command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run controlled Parallel K-Means experiments "
            "across isolated Spark-master processes."
        )
    )

    parser.add_argument(
        "--masters",
        type=parse_master_csv,
        default=DEFAULT_MASTERS,
        help=(
            "Comma-separated Spark masters. "
            "Default: local[1],local[2],local[4]"
        ),
    )

    parser.add_argument(
        "--record-counts",
        type=parse_positive_int_csv,
        default=(1000,),
        help=(
            "Comma-separated dataset sizes. "
            "Default: 1000"
        ),
    )

    parser.add_argument(
        "--partition-counts",
        type=parse_positive_int_csv,
        default=(4,),
        help=(
            "Comma-separated Spark partition counts. "
            "Default: 4"
        ),
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repeated runs per matrix case.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help=(
            "Number of warm-up runs per isolated Spark master. "
            "Default: 1"
        ),
    )

    parser.add_argument(
        "--aggregation-restarts",
        type=int,
        default=5,
        help=(
            "Number of deterministic candidate-center aggregation "
            "initializations. Default: 5"
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=("memory", "file"),
        default="memory",
        help=(
            "Synthetic input mode: memory or file. "
            "Default: memory"
        ),
    )

    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=Path("data/generated/cross-master"),
        help=(
            "Root directory for generated file-backed datasets. "
            "Default: data/generated/cross-master"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/raw/cross-master"),
        help="Root directory for raw experiment results.",
    )

    parser.add_argument(
        "--log-level",
        default="WARN",
        help="Spark log level. Default: WARN",
    )

    return parser


def main() -> int:
    """Execute the process-isolated cross-master experiment."""

    args = build_parser().parse_args()

    if args.repetitions <= 0:
        raise SystemExit(
            "--repetitions must be greater than zero."
        )
    if args.warmup_runs < 0:
        raise SystemExit(
            "--warmup-runs must be zero or greater."
        )

    if args.aggregation_restarts <= 0:
        raise SystemExit(
            "--aggregation-restarts must be greater than zero."
        )
    print(
        "Cross-master experiment controller started."
    )
    print(
        f"Spark masters: {args.masters}"
    )
    print(
        f"Record counts: {args.record_counts}"
    )
    print(
        f"Partition counts: {args.partition_counts}"
    )
    print(
        f"Repetitions: {args.repetitions}"
    )
    print(
        f"Warm-up runs: {args.warmup_runs}"
    )
    print(
        "Aggregation restarts: "
        f"{args.aggregation_restarts}"
    )
    print(
        f"Input mode: {args.input_mode}"
    )
    print(
        f"Dataset root: {args.dataset_directory}"
    )
    print(
        f"Output root: {args.output_directory}"
    )

    run_cross_master_processes(
        spark_masters=args.masters,
        record_counts=args.record_counts,
        partition_counts=args.partition_counts,
        repetitions=args.repetitions,
        warmup_runs=args.warmup_runs,
        aggregation_restarts=args.aggregation_restarts,
        input_mode=args.input_mode,
        dataset_directory=args.dataset_directory,
        output_directory=args.output_directory,
        log_level=args.log_level,
    )

    print(
        "Cross-master experiment controller completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
