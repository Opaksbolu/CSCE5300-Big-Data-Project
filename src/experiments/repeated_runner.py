"""
Repeated controlled execution of synthetic Parallel K-Means experiments.

This module builds on the single-experiment runner by executing the
same experimental configuration multiple times with deterministic,
distinct random seeds and experiment identifiers.

Statistical summarization is intentionally kept outside this module.
Its responsibility is only controlled repeated execution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.experiments.experiment_result import ExperimentResult
from src.experiments.experiment_runner import (
    SyntheticExperimentConfig,
    run_file_backed_synthetic_experiment,
    run_synthetic_experiment,
)


def run_repeated_synthetic_experiments(
    spark,
    config: SyntheticExperimentConfig,
    *,
    repetitions: int,
    output_path: str | Path | None = None,
) -> tuple[ExperimentResult, ...]:
    """
    Execute a synthetic experiment repeatedly under controlled seeds.

    Each repetition preserves the base experimental configuration while
    assigning a deterministic seed and a unique experiment identifier.

    The first repetition uses the base random seed. Each later
    repetition increments that seed by one.

    Parameters
    ----------
    spark:
        Active SparkSession used for every experiment.

    config:
        Base synthetic experiment configuration.

    repetitions:
        Number of controlled experiment runs to execute.

    output_path:
        Optional CSV destination. When provided, every completed result
        is appended to the same file.

    Returns
    -------
    tuple[ExperimentResult, ...]
        Results in execution order.
    """

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be greater than zero."
        )

    results: list[ExperimentResult] = []

    for repetition_index in range(repetitions):
        run_number = repetition_index + 1

        run_config = replace(
            config,
            experiment_id=(
                f"{config.experiment_id}-run-{run_number:02d}"
            ),
            random_seed=(
                config.random_seed + repetition_index
            ),
        )

        result = run_synthetic_experiment(
            spark,
            run_config,
            output_path=output_path,
        )

        results.append(result)

    return tuple(results)


def run_repeated_file_backed_synthetic_experiments(
    spark,
    config: SyntheticExperimentConfig,
    *,
    repetitions: int,
    dataset_directory: str | Path,
    output_path: str | Path | None = None,
) -> tuple[ExperimentResult, ...]:
    """
    Execute repeated synthetic experiments through file-backed input.

    Each repetition preserves the same deterministic identifier and
    seed progression as the in-memory repeated runner. A separate
    synthetic dataset file is generated for every repetition.

    Dataset files remain owned by the caller and are not deleted after
    execution so that they can be inspected or reused by later
    experiment orchestration.
    """

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be greater than zero."
        )

    dataset_root = Path(dataset_directory).expanduser()
    dataset_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[ExperimentResult] = []

    for repetition_index in range(repetitions):
        run_number = repetition_index + 1

        run_config = replace(
            config,
            experiment_id=(
                f"{config.experiment_id}-run-{run_number:02d}"
            ),
            random_seed=(
                config.random_seed + repetition_index
            ),
        )

        dataset_path = (
            dataset_root
            / f"{run_config.experiment_id}-seed-{run_config.random_seed}.csv"
        )

        result = run_file_backed_synthetic_experiment(
            spark,
            run_config,
            dataset_path=dataset_path,
            output_path=output_path,
        )

        results.append(result)

    return tuple(results)
