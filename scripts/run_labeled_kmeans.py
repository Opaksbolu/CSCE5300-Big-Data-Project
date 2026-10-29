"""
Command-line traditional K-Means benchmarks for labeled CSV datasets.

Uses the project's existing labeled baseline implementation.
Dataset loading and CSV output are excluded from clustering runtime.

Produces:
1. Benchmark metrics CSV.
2. Record-level predictions CSV for external evaluation.

Never overwrites existing output files.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from src.experiments.dataset_bridge import load_labeled_csv
from src.experiments.labeled_baseline import run_labeled_baseline


METRIC_FIELDS = [
    "algorithm",
    "dataset_name",
    "num_records",
    "num_features",
    "num_clusters",
    "random_seed",
    "max_iterations",
    "tolerance",
    "converged",
    "iterations",
    "sse",
    "cluster_counts",
    "total_runtime_seconds",
    "num_initializations",
    "best_initialization_seed",
    "repetition",
]

PREDICTION_FIELDS = [
    "dataset_name",
    "repetition",
    "random_seed",
    "record_id",
    "true_label",
    "predicted_cluster",
]


def build_parser() -> argparse.ArgumentParser:
    """Create the labeled baseline command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run traditional K-Means on a labeled numerical CSV."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a labeled CSV dataset.",
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        required=True,
        help="Number of K-Means clusters.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of measured repetitions.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of unmeasured warm-up runs.",
    )

    parser.add_argument(
        "--num-initializations",
        type=int,
        default=1,
        help="Number of K-Means initializations per run.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Starting random seed.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for benchmark metrics CSV.",
    )

    parser.add_argument(
        "--predictions-output",
        required=True,
        help="Path for record-level predictions CSV.",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Validate configuration and protect existing outputs."""

    if args.num_clusters <= 0:
        parser.error("--num-clusters must be positive")

    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")

    if args.num_initializations <= 0:
        parser.error("--num-initializations must be positive")

    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")

    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error(
            "--tolerance must be finite and nonnegative"
        )

    output = Path(args.output).expanduser()
    predictions_output = Path(
        args.predictions_output
    ).expanduser()

    if output.resolve() == predictions_output.resolve():
        parser.error(
            "Metrics and predictions must use different files."
        )

    for target in (output, predictions_output):
        if target.exists():
            parser.error(
                f"STOP: Output already exists: {target}"
            )

    return output, predictions_output


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output, predictions_output = validate_arguments(
        parser,
        args,
    )

    dataset = load_labeled_csv(args.dataset)

    if args.num_clusters > dataset.num_records:
        parser.error(
            "--num-clusters cannot exceed dataset size"
        )

    print("=" * 65)
    print("LABELED TRADITIONAL K-MEANS BENCHMARK")
    print("=" * 65)

    print(f"Dataset: {dataset.name}")
    print(f"Records: {dataset.num_records}")
    print(f"Features: {dataset.num_features}")
    print(f"Ground-truth classes: {dataset.num_classes}")
    print(f"Configured clusters: {args.num_clusters}")

    metric_rows = []
    prediction_rows = []

    # Warm-up runs do not contribute rows to either output.
    for warmup_index in range(args.warmup_runs):
        run_labeled_baseline(
            dataset,
            num_clusters=args.num_clusters,
            random_seed=args.random_seed,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            num_initializations=args.num_initializations,
        )

        print(
            f"Warm-up {warmup_index + 1}/"
            f"{args.warmup_runs} complete"
        )

    # Every repetition clusters the same loaded dataset.
    # Only the initialization seed changes.
    for repetition_index in range(args.repetitions):
        seed = args.random_seed + repetition_index

        result = run_labeled_baseline(
            dataset,
            num_clusters=args.num_clusters,
            random_seed=seed,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            num_initializations=args.num_initializations,
        )

        repetition = repetition_index + 1

        metric_rows.append(
            {
                "algorithm": "traditional-kmeans",
                "dataset_name": result.dataset_name,
                "num_records": result.num_records,
                "num_features": result.num_features,
                "num_clusters": result.num_clusters,
                "random_seed": result.random_seed,
                "max_iterations": result.max_iterations,
                "tolerance": result.tolerance,
                "converged": result.converged,
                "iterations": result.iterations,
                "sse": result.sse,
                "cluster_counts": ";".join(
                    map(str, result.cluster_counts)
                ),
                "total_runtime_seconds": (
                    result.total_runtime_seconds
                ),
                "num_initializations": (
                    result.num_initializations
                ),
                "best_initialization_seed": (
                    result.best_initialization_seed
                ),
                "repetition": repetition,
            }
        )

        if not (
            len(result.record_ids)
            == len(result.true_labels)
            == len(result.predicted_clusters)
            == result.num_records
        ):
            raise RuntimeError(
                "Record-level evaluation arrays are misaligned."
            )

        for record_id, true_label, predicted_cluster in zip(
            result.record_ids,
            result.true_labels,
            result.predicted_clusters,
        ):
            prediction_rows.append(
                {
                    "dataset_name": result.dataset_name,
                    "repetition": repetition,
                    "random_seed": result.random_seed,
                    "record_id": record_id,
                    "true_label": true_label,
                    "predicted_cluster": predicted_cluster,
                }
            )

        print(
            f"Run {repetition}/{args.repetitions}: "
            f"time={result.total_runtime_seconds:.4f}s, "
            f"SSE={result.sse:.6f}, "
            f"converged={result.converged}, "
            f"best_seed={result.best_initialization_seed}"
        )

    # Validate both outputs before writing anything.
    if len(metric_rows) != args.repetitions:
        raise RuntimeError("Incorrect benchmark row count.")

    expected_predictions = (
        args.repetitions * dataset.num_records
    )

    if len(prediction_rows) != expected_predictions:
        raise RuntimeError("Incorrect prediction row count.")

    # Recheck output protection immediately before writing.
    for target in (output, predictions_output):
        if target.exists():
            raise FileExistsError(
                f"STOP: Output already exists: {target}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exclusive creation prevents overwriting an existing file.
    with output.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRIC_FIELDS,
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    with predictions_output.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PREDICTION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)
    print(f"Measured repetitions: {len(metric_rows)}")
    print(f"Prediction rows: {len(prediction_rows)}")
    print(f"Metrics saved: {output}")
    print(f"Predictions saved: {predictions_output}")


if __name__ == "__main__":
    main()
