"""
Controlled experiment-matrix execution for synthetic Parallel K-Means.

This module expands a base synthetic experiment configuration across
selected dataset sizes and partition counts. Each matrix case is
executed through the existing repeated-experiment workflow so that raw
results and statistical summaries remain consistent with single-case
benchmark execution.

Spark master selection is intentionally outside this module. A matrix
executes against the active SparkSession supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from src.experiments.experiment_runner import SyntheticExperimentConfig
from src.experiments.experiment_workflow import (
    RepeatedExperimentWorkflowResult,
    run_repeated_synthetic_experiment_workflow,
)


@dataclass(frozen=True)
class ExperimentMatrixCaseResult:
    """Configuration and completed workflow for one matrix case."""

    config: SyntheticExperimentConfig
    workflow: RepeatedExperimentWorkflowResult


@dataclass(frozen=True)
class ExperimentMatrixResult:
    """Completed cases from one controlled experiment matrix."""

    cases: tuple[ExperimentMatrixCaseResult, ...]


def _validate_positive_values(
    values: tuple[int, ...],
    *,
    name: str,
) -> None:
    """Validate a nonempty tuple containing only positive integers."""

    if not values:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if any(value <= 0 for value in values):
        raise ValueError(
            f"{name} must contain only positive values."
        )


def run_synthetic_experiment_matrix(
    spark,
    base_config: SyntheticExperimentConfig,
    *,
    record_counts: tuple[int, ...],
    partition_counts: tuple[int, ...],
    repetitions: int,
    output_directory: str | Path | None = None,
) -> ExperimentMatrixResult:
    """
    Execute repeated experiments across dataset-size and partition cases.

    Matrix cases are generated in deterministic record-count-major
    order. Each case preserves the base configuration except for the
    experiment identifier, record count, and partition count.

    Parameters
    ----------
    spark:
        Active SparkSession used for every matrix case.

    base_config:
        Base synthetic experiment configuration.

    record_counts:
        Dataset sizes to evaluate.

    partition_counts:
        Spark partition counts to evaluate.

    repetitions:
        Number of repeated runs for every matrix case.

    output_directory:
        Optional directory for one raw-result CSV per matrix case.

    Returns
    -------
    ExperimentMatrixResult
        Immutable collection of completed matrix cases.
    """

    _validate_positive_values(
        record_counts,
        name="record_counts",
    )
    _validate_positive_values(
        partition_counts,
        name="partition_counts",
    )

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be greater than zero."
        )

    output_root = (
        Path(output_directory).expanduser()
        if output_directory is not None
        else None
    )

    cases = []

    for num_records in record_counts:
        for num_partitions in partition_counts:
            case_id = (
                f"{base_config.experiment_id}"
                f"-n{num_records}"
                f"-p{num_partitions}"
            )

            case_config = replace(
                base_config,
                experiment_id=case_id,
                num_records=num_records,
                num_partitions=num_partitions,
            )

            output_path = (
                output_root / f"{case_id}.csv"
                if output_root is not None
                else None
            )

            workflow = run_repeated_synthetic_experiment_workflow(
                spark,
                case_config,
                repetitions=repetitions,
                output_path=output_path,
            )

            cases.append(
                ExperimentMatrixCaseResult(
                    config=case_config,
                    workflow=workflow,
                )
            )

    return ExperimentMatrixResult(
        cases=tuple(cases),
    )
