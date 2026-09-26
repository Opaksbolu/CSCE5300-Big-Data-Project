"""
Run one controlled synthetic experiment matrix under one Spark master.

This script is intentionally responsible for exactly one Spark master
per process. Cross-master experiments launch this script in separate
processes so that Spark application contexts remain isolated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.experiment_matrix import (
    run_synthetic_experiment_matrix,
)
from src.experiments.experiment_runner import (
    SyntheticExperimentConfig,
)
from src.experiments.master_matrix import master_output_name
from src.parallel.spark_session import (
    create_spark_session,
    stop_spark_session,
)


def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence of positive integers."""

    try:
        values = tuple(
            int(item.strip())
            for item in value.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from exc

    if not values or any(number <= 0 for number in values):
        raise argparse.ArgumentTypeError(
            "values must contain only positive integers"
        )

    return values


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run one synthetic Parallel K-Means experiment matrix "
            "under one Spark master."
        )
    )

    parser.add_argument(
        "--master",
        required=True,
        help="Spark master, for example local[1].",
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
    """Execute one isolated Spark-master experiment matrix."""

    args = build_parser().parse_args()

    if args.repetitions <= 0:
        raise SystemExit(
            "--repetitions must be greater than zero."
        )

    output_directory = (
        args.output_directory
        / master_output_name(args.master)
    )

    base_config = SyntheticExperimentConfig(
        experiment_id="cross-master",
        dataset_name="synthetic-cross-master",
        num_records=args.record_counts[0],
        num_features=20,
        num_clusters=5,
        num_partitions=args.partition_counts[0],
        cluster_spread=1.0,
        center_separation=10.0,
        random_seed=42,
        local_max_iterations=100,
        global_max_iterations=100,
        tolerance=1e-6,
    )

    spark = create_spark_session(
        app_name=(
            "CSCE5300-CrossMaster-"
            f"{master_output_name(args.master)}"
        ),
        master=args.master,
        log_level=args.log_level,
    )

    try:
        actual_master = spark.sparkContext.master

        if actual_master != args.master:
            raise RuntimeError(
                "Spark master mismatch: "
                f"requested {args.master!r}, "
                f"received {actual_master!r}."
            )

        print(
            "Cross-master experiment process started."
        )
        print(
            f"Requested Spark master: {args.master}"
        )
        print(
            f"Active Spark master: {actual_master}"
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
            f"Output directory: {output_directory}"
        )

        matrix = run_synthetic_experiment_matrix(
            spark,
            base_config,
            record_counts=args.record_counts,
            partition_counts=args.partition_counts,
            repetitions=args.repetitions,
            output_directory=output_directory,
        )

        print(
            f"Completed matrix cases: {len(matrix.cases)}"
        )

        for case in matrix.cases:
            summary = case.workflow.summary

            print(
                "CASE "
                f"{case.config.experiment_id}: "
                f"records={case.config.num_records}, "
                f"partitions={case.config.num_partitions}, "
                f"repetitions={summary.repetitions}, "
                f"converged={summary.converged_runs}"
            )

        print(
            "Cross-master experiment process completed."
        )

    finally:
        stop_spark_session(spark)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
