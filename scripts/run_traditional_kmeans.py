"""
Conventional single-process K-Means benchmark.

Uses the project's existing deterministic local K-Means implementation
on the complete synthetic dataset, without Spark partitioning.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from src.experiments.synthetic_data import generate_synthetic_dataset
from src.parallel.local_kmeans import fit_local_kmeans


def run_baseline(
    *,
    num_records: int,
    num_features: int,
    num_clusters: int,
    random_seed: int = 42,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    num_initializations: int = 1,
) -> dict:
    """
    Execute conventional K-Means with configurable initializations.

    The dataset is generated once. Each initialization uses the same
    data but a different deterministic random seed.

    The clustering result with the lowest SSE is retained.
    """

    if num_initializations <= 0:
        raise ValueError(
            "num_initializations must be greater than zero."
        )

    dataset = generate_synthetic_dataset(
        num_records=num_records,
        num_features=num_features,
        num_clusters=num_clusters,
        random_seed=random_seed,
    )

    start = perf_counter()

    best_result = None
    best_seed = None

    for initialization_index in range(num_initializations):
        initialization_seed = random_seed + initialization_index

        result = fit_local_kmeans(
            dataset.points,
            k=num_clusters,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_seed=initialization_seed,
        )

        if best_result is None or result.sse < best_result.sse:
            best_result = result
            best_seed = initialization_seed

    runtime = perf_counter() - start

    cluster_counts = tuple(
        best_result.assignments.count(index)
        for index in range(num_clusters)
    )

    return {
        "algorithm": "traditional-kmeans",
        "num_records": num_records,
        "num_features": num_features,
        "num_clusters": num_clusters,
        "random_seed": random_seed,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "converged": best_result.converged,
        "iterations": best_result.iterations,
        "sse": best_result.sse,
        "cluster_counts": ";".join(map(str, cluster_counts)),
        "total_runtime_seconds": runtime,
        "num_initializations": num_initializations,
        "best_initialization_seed": best_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run conventional K-Means benchmarks."
    )

    parser.add_argument(
        "--record-counts",
        default="1000",
    )

    parser.add_argument(
        "--num-features",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of unmeasured warm-up runs per dataset size.",
    )

    parser.add_argument(
        "--num-initializations",
        type=int,
        default=1,
        help="Number of deterministic K-Means initializations.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default="results/raw/traditional-kmeans-baseline.csv",
    )

    args = parser.parse_args()

    if args.repetitions <= 0:
        parser.error("--repetitions must be greater than zero")

    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")

    if args.num_initializations <= 0:
        parser.error("--num-initializations must be greater than zero")

    if args.num_features <= 0:
        parser.error("--num-features must be positive")

    if args.num_clusters <= 0:
        parser.error("--num-clusters must be positive")

    try:
        record_counts = [
            int(value.strip())
            for value in args.record_counts.split(",")
        ]
    except ValueError:
        parser.error("--record-counts must contain integers")

    if not record_counts or any(n <= 0 for n in record_counts):
        parser.error("--record-counts must contain positive integers")

    if any(n < args.num_clusters for n in record_counts):
        parser.error(
            "--num-clusters cannot exceed any dataset size"
        )

    output = Path(args.output)

    if output.exists():
        raise SystemExit(
            f"STOP: Output file already exists: {output}"
        )

    rows = []

    for num_records in record_counts:
        for warmup_index in range(args.warmup_runs):
            run_baseline(
                num_records=num_records,
                num_features=args.num_features,
                num_clusters=args.num_clusters,
                random_seed=args.random_seed,
                num_initializations=args.num_initializations,
            )

            print(
                f"Warm-up {warmup_index + 1}/"
                f"{args.warmup_runs} completed for n={num_records}"
            )

        for repetition in range(args.repetitions):
            seed = args.random_seed + repetition

            row = run_baseline(
                num_records=num_records,
                num_features=args.num_features,
                num_clusters=args.num_clusters,
                random_seed=seed,
                num_initializations=args.num_initializations,
            )

            row["repetition"] = repetition + 1
            rows.append(row)

            print(
                f"n={num_records}, "
                f"run={repetition + 1}, "
                f"time={row['total_runtime_seconds']:.4f}s, "
                f"SSE={row['sse']:.4f}, "
                f"converged={row['converged']}"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        raise SystemExit(
            f"STOP: Output file already exists: {output}"
        )

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
