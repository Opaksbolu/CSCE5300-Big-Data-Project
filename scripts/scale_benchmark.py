"""
benchmark_runner.py proved the harness works end-to-end on a tiny
12-point demo dataset. This script is a separate, second step: it
generates its own synthetic datasets at increasing sizes and runs
Spark-Kmeans across a grid of (dataset size x Spark master) so we can
see real scalability behavior *before* Person 2's real datasets land.

This is throwaway/exploratory data, not the final experiment. out, but the grid-running logic stays the
same.

Usage (same env setup as benchmark_runner.py):

    python -m scripts.scale_benchmark
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from sklearn.datasets import make_blobs

from src.parallel.parallel_kmeans import fit_parallel_kmeans
from src.parallel.spark_session import create_spark_session

RESULTS_PATH = Path("results/raw/spark_kmeans_scale_trials.csv")

# --- experiment grid: edit these to control what gets tested -----------

DATASET_SIZES = [1_000, 10_000, 50_000]
SPARK_MASTERS = ["local[1]", "local[2]", "local[4]"]
K = 5                     # matches the "5 categories" of the paper's synthetic dataset
N_FEATURES = 20           # matches the paper's synthetic dataset feature count
TRIALS_PER_CONFIG = 3
RANDOM_SEED = 42
TOLERANCE = 1e-6

FIELDNAMES = [
    "run_id",
    "algorithm",
    "n_records",
    "n_features",
    "k",
    "spark_master",
    "num_partitions",
    "random_seed",
    "trial",
    "tolerance",
    "wall_clock_seconds",
    "partition_clustering_seconds",
    "center_aggregation_seconds",
    "initialization_seconds",
    "global_clustering_seconds",
    "total_runtime_seconds",
    "iterations",
    "converged",
    "sse",
    "cluster_counts",
]


def generate_synthetic_points(
    n_samples: int,
    *,
    n_features: int,
    centers: int,
    random_seed: int,
) -> list[tuple[float, ...]]:
    """
    Generate a synthetic dataset with sklearn's make_blobs.

    This mirrors the paper's own approach ("sklearn randomly generated
    dataset") for Dataset II. Note: make_blobs also returns ground-truth
    labels, which we are NOT using here — this script is timing/SSE
    only. Purity evaluation is a separate step once we either keep
    these labels or switch to Person 2's labeled data.
    """
    X, _labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_seed,
    )
    return [tuple(row) for row in X.tolist()]


def run_trial(
    points: list[tuple[float, ...]],
    *,
    k: int,
    spark_master: str,
    num_partitions: int,
    random_seed: int,
    trial: int,
    tolerance: float = TOLERANCE,
) -> dict:
    """Run one Spark-Kmeans trial and return a flat results row."""

    spark = create_spark_session(
        app_name="CSCE5300-Scale-Benchmark",
        master=spark_master,
        log_level="ERROR",
    )

    try:
        rdd = spark.sparkContext.parallelize(points, num_partitions)

        wall_start = time.perf_counter()
        result = fit_parallel_kmeans(
            rdd,
            k=k,
            tolerance=tolerance,
            random_seed=random_seed,
        )
        wall_clock_seconds = time.perf_counter() - wall_start

        return {
            "run_id": f"spark_{len(points)}_{spark_master}_{trial}",
            "algorithm": "spark-kmeans",
            "n_records": len(points),
            "n_features": len(points[0]) if points else 0,
            "k": k,
            "spark_master": spark_master,
            "num_partitions": num_partitions,
            "random_seed": random_seed,
            "trial": trial,
            "tolerance": tolerance,
            "wall_clock_seconds": wall_clock_seconds,
            "partition_clustering_seconds": result.timing.partition_clustering_seconds,
            "center_aggregation_seconds": result.timing.center_aggregation_seconds,
            "initialization_seconds": result.timing.initialization_seconds,
            "global_clustering_seconds": result.timing.global_clustering_seconds,
            "total_runtime_seconds": result.timing.total_runtime_seconds,
            "iterations": result.iterations,
            "converged": result.converged,
            "sse": result.sse,
            "cluster_counts": list(result.cluster_counts),
        }

    finally:
        spark.stop()


def append_row(row: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    total_runs = len(DATASET_SIZES) * len(SPARK_MASTERS) * TRIALS_PER_CONFIG
    run_number = 0

    for size in DATASET_SIZES:
        print(f"\nGenerating {size} synthetic points ({N_FEATURES} features, k={K})...")
        points = generate_synthetic_points(
            size,
            n_features=N_FEATURES,
            centers=K,
            random_seed=RANDOM_SEED,
        )

        for master in SPARK_MASTERS:
            # keep partition count reasonable relative to master's core count
            num_partitions = int(master.split("[")[1].rstrip("]")) if "[" in master else 4

            for trial in range(1, TRIALS_PER_CONFIG + 1):
                run_number += 1
                print(
                    f"[{run_number}/{total_runs}] "
                    f"size={size} master={master} trial={trial} ..."
                )

                row = run_trial(
                    points,
                    k=K,
                    spark_master=master,
                    num_partitions=num_partitions,
                    random_seed=RANDOM_SEED,
                    trial=trial,
                )
                append_row(row)

                print(
                    f"    sse={row['sse']:.4f} "
                    f"wall_clock={row['wall_clock_seconds']:.2f}s "
                    f"iterations={row['iterations']} "
                    f"converged={row['converged']}"
                )

    print(f"\nAll runs complete. Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
