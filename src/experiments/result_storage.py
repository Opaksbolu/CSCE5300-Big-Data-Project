"""
Persistent storage for standardized experiment results.

This module writes ExperimentResult records to CSV so that benchmark
executions can be preserved for later statistical analysis,
visualization, and comparison.

CSV persistence remains separate from the clustering and benchmark
execution layers.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from src.experiments.experiment_result import ExperimentResult


CSV_FIELDNAMES = (
    "experiment_id",
    "dataset_name",
    "num_records",
    "num_features",
    "num_clusters",
    "num_partitions",
    "spark_master",
    "random_seed",
    "local_max_iterations",
    "global_max_iterations",
    "tolerance",
    "materialized_record_count",
    "converged",
    "iterations",
    "sse",
    "cluster_counts",
    "partition_clustering_seconds",
    "center_aggregation_seconds",
    "initialization_seconds",
    "global_clustering_seconds",
    "final_evaluation_seconds",
    "total_runtime_seconds",
)


def _serialize_experiment_result(
    result: ExperimentResult,
) -> dict[str, object]:
    """Convert an ExperimentResult into a CSV-compatible row."""

    row = asdict(result)

    row["cluster_counts"] = ";".join(
        str(count)
        for count in result.cluster_counts
    )

    return row


def append_experiment_result(
    result: ExperimentResult,
    output_path: str | Path,
) -> Path:
    """
    Append one experiment result to a CSV file.

    The parent directory is created when necessary. A header is written
    only when the destination file does not already contain data.

    Parameters
    ----------
    result:
        Standardized experiment result to persist.

    output_path:
        Destination CSV path.

    Returns
    -------
    Path
        Resolved destination path used for the write.
    """

    path = Path(output_path).expanduser()

    if path.exists() and path.is_dir():
        raise ValueError(
            "output_path must reference a file, not a directory."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDNAMES,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            _serialize_experiment_result(result)
        )

    return path
