"""
High-level workflow for repeated synthetic Parallel K-Means experiments.

This module connects controlled repeated execution with statistical
summarization. It provides one orchestration entry point while keeping
dataset generation, clustering, persistence, and statistical analysis
in their existing specialized modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiments.experiment_result import ExperimentResult
from src.experiments.experiment_runner import SyntheticExperimentConfig
from src.experiments.repeated_runner import (
    run_repeated_synthetic_experiments,
)
from src.experiments.result_summary import (
    ExperimentSummary,
    summarize_experiment_results,
)


@dataclass(frozen=True)
class RepeatedExperimentWorkflowResult:
    """Raw and summarized results from one repeated experiment workflow."""

    results: tuple[ExperimentResult, ...]
    summary: ExperimentSummary


def run_repeated_synthetic_experiment_workflow(
    spark,
    config: SyntheticExperimentConfig,
    *,
    repetitions: int,
    output_path: str | Path | None = None,
) -> RepeatedExperimentWorkflowResult:
    """
    Execute and summarize repeated synthetic Parallel K-Means runs.

    The repeated-execution layer controls experiment identifiers,
    deterministic seed progression, and optional CSV persistence.
    Completed results are then summarized using the statistical
    analysis layer.

    Parameters
    ----------
    spark:
        Active SparkSession shared across all experiment repetitions.

    config:
        Base synthetic experiment configuration.

    repetitions:
        Number of controlled repetitions to execute.

    output_path:
        Optional CSV destination for the individual experiment results.

    Returns
    -------
    RepeatedExperimentWorkflowResult
        Individual experiment records together with their statistical
        summary.
    """

    results = run_repeated_synthetic_experiments(
        spark,
        config,
        repetitions=repetitions,
        output_path=output_path,
    )

    summary = summarize_experiment_results(results)

    return RepeatedExperimentWorkflowResult(
        results=results,
        summary=summary,
    )
