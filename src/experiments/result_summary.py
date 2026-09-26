"""
Statistical summaries for repeated Parallel K-Means experiments.

This module converts multiple standardized ExperimentResult records
into a compact statistical summary suitable for later reporting,
comparison, and visualization.

It does not execute experiments or persist results. Its responsibility
is limited to summarizing completed experiment runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from src.experiments.experiment_result import ExperimentResult


@dataclass(frozen=True)
class ExperimentSummary:
    """Statistical summary of repeated experiment executions."""

    repetitions: int
    converged_runs: int
    convergence_rate: float

    mean_iterations: float
    std_iterations: float

    mean_sse: float
    std_sse: float

    mean_partition_clustering_seconds: float
    std_partition_clustering_seconds: float

    mean_center_aggregation_seconds: float
    std_center_aggregation_seconds: float

    mean_initialization_seconds: float
    std_initialization_seconds: float

    mean_global_clustering_seconds: float
    std_global_clustering_seconds: float

    mean_final_evaluation_seconds: float
    std_final_evaluation_seconds: float

    mean_total_runtime_seconds: float
    std_total_runtime_seconds: float


def _sample_standard_deviation(
    values: tuple[float, ...],
) -> float:
    """
    Return sample standard deviation for repeated measurements.

    A single observation has no observed run-to-run variation, so this
    experiment summary reports a standard deviation of zero when only
    one result is available.
    """

    if len(values) == 1:
        return 0.0

    return stdev(values)


def summarize_experiment_results(
    results: tuple[ExperimentResult, ...],
) -> ExperimentSummary:
    """
    Summarize repeated experiment results using mean and sample spread.

    Parameters
    ----------
    results:
        One or more completed ExperimentResult records.

    Returns
    -------
    ExperimentSummary
        Aggregate statistics across the supplied experiment runs.
    """

    if not results:
        raise ValueError(
            "results must contain at least one experiment result."
        )

    repetitions = len(results)

    converged_runs = sum(
        1
        for result in results
        if result.converged
    )

    iterations = tuple(
        float(result.iterations)
        for result in results
    )

    sse_values = tuple(
        result.sse
        for result in results
    )

    partition_clustering_seconds = tuple(
        result.partition_clustering_seconds
        for result in results
    )

    center_aggregation_seconds = tuple(
        result.center_aggregation_seconds
        for result in results
    )

    initialization_seconds = tuple(
        result.initialization_seconds
        for result in results
    )

    global_clustering_seconds = tuple(
        result.global_clustering_seconds
        for result in results
    )

    final_evaluation_seconds = tuple(
        result.final_evaluation_seconds
        for result in results
    )

    total_runtime_seconds = tuple(
        result.total_runtime_seconds
        for result in results
    )

    return ExperimentSummary(
        repetitions=repetitions,
        converged_runs=converged_runs,
        convergence_rate=(
            converged_runs / repetitions
        ),
        mean_iterations=mean(iterations),
        std_iterations=_sample_standard_deviation(
            iterations
        ),
        mean_sse=mean(sse_values),
        std_sse=_sample_standard_deviation(
            sse_values
        ),
        mean_partition_clustering_seconds=mean(
            partition_clustering_seconds
        ),
        std_partition_clustering_seconds=(
            _sample_standard_deviation(
                partition_clustering_seconds
            )
        ),
        mean_center_aggregation_seconds=mean(
            center_aggregation_seconds
        ),
        std_center_aggregation_seconds=(
            _sample_standard_deviation(
                center_aggregation_seconds
            )
        ),
        mean_initialization_seconds=mean(
            initialization_seconds
        ),
        std_initialization_seconds=(
            _sample_standard_deviation(
                initialization_seconds
            )
        ),
        mean_global_clustering_seconds=mean(
            global_clustering_seconds
        ),
        std_global_clustering_seconds=(
            _sample_standard_deviation(
                global_clustering_seconds
            )
        ),
        mean_final_evaluation_seconds=mean(
            final_evaluation_seconds
        ),
        std_final_evaluation_seconds=(
            _sample_standard_deviation(
                final_evaluation_seconds
            )
        ),
        mean_total_runtime_seconds=mean(
            total_runtime_seconds
        ),
        std_total_runtime_seconds=(
            _sample_standard_deviation(
                total_runtime_seconds
            )
        ),
    )
