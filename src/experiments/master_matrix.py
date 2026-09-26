"""
Configuration helpers for controlled experiments across Spark masters.

Cross-master Spark execution is intentionally isolated at the process
level. A Spark master belongs to the Spark application context, so this
module does not create or stop Spark sessions itself.

Actual local[1], local[2], and local[4] runs are launched in separate
processes by the cross-master benchmark execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.experiments.experiment_matrix import ExperimentMatrixResult


@dataclass(frozen=True)
class MasterMatrixCaseResult:
    """Completed experiment matrix for one Spark master."""

    spark_master: str
    matrix: ExperimentMatrixResult


@dataclass(frozen=True)
class MasterMatrixResult:
    """Completed experiment matrices across Spark masters."""

    cases: tuple[MasterMatrixCaseResult, ...]


def validate_spark_masters(
    spark_masters: tuple[str, ...],
) -> None:
    """Validate a nonempty tuple of Spark master strings."""

    if not spark_masters:
        raise ValueError(
            "spark_masters cannot be empty."
        )

    if any(
        not isinstance(master, str) or not master.strip()
        for master in spark_masters
    ):
        raise ValueError(
            "spark_masters must contain only nonempty strings."
        )


def master_output_name(master: str) -> str:
    """
    Convert a Spark master into a filesystem-safe directory name.

    Examples
    --------
    local[1] -> local-1
    local[2] -> local-2
    local[4] -> local-4
    """

    if not isinstance(master, str) or not master.strip():
        raise ValueError(
            "master must be a nonempty string."
        )

    return (
        master.strip()
        .replace("[", "-")
        .replace("]", "")
        .replace(":", "-")
        .replace("/", "-")
    )
