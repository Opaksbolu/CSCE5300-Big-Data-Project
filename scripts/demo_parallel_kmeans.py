"""
End-to-end demonstration of the improved Parallel K-Means pipeline.

This script creates a reproducible synthetic dataset and executes the
complete Spark-based clustering workflow:

    Spark RDD
        -> partition-level local K-Means
        -> candidate-center aggregation
        -> global initialization
        -> distributed global K-Means
        -> final clustering result

The demonstration is intentionally small enough to run comfortably on
a development machine while still exercising the complete parallel
architecture.
"""

from __future__ import annotations

from src.parallel.parallel_kmeans import fit_parallel_kmeans
from src.parallel.spark_session import create_spark_session


def format_point(point: tuple[float, ...]) -> str:
    """Return a feature vector in a readable fixed-precision format."""

    values = ", ".join(
        f"{value:.4f}"
        for value in point
    )

    return f"({values})"


def main() -> None:
    """Run the complete Parallel K-Means demonstration."""

    spark = create_spark_session(
        app_name="CSCE5300-Parallel-KMeans-Demo",
        master="local[4]",
        log_level="ERROR",
    )

    try:
        points = [
            (0.8, 1.0),
            (1.0, 0.8),
            (1.0, 1.0),
            (1.0, 1.2),
            (1.2, 1.0),
            (1.2, 1.2),
            (4.8, 5.0),
            (5.0, 4.8),
            (5.0, 5.0),
            (5.0, 5.2),
            (5.2, 5.0),
            (5.2, 5.2),
            (8.8, 9.0),
            (9.0, 8.8),
            (9.0, 9.0),
            (9.0, 9.2),
            (9.2, 9.0),
            (9.2, 9.2),
            (12.8, 13.0),
            (13.0, 12.8),
            (13.0, 13.0),
            (13.0, 13.2),
            (13.2, 13.0),
            (13.2, 13.2),
        ]

        k = 4
        partitions = 4
        random_seed = 42
        tolerance = 1e-12

        rdd = spark.sparkContext.parallelize(
            points,
            partitions,
        )

        partition_sizes = rdd.mapPartitions(
            lambda iterator: [sum(1 for _ in iterator)]
        ).collect()

        print()
        print("=" * 72)
        print("CSCE 5300 - IMPROVED PARALLEL K-MEANS DEMONSTRATION")
        print("=" * 72)

        print()
        print("SPARK CONFIGURATION")
        print("-" * 72)
        print(f"Spark version:       {spark.version}")
        print(f"Spark master:        {spark.sparkContext.master}")
        print(f"Application name:    {spark.sparkContext.appName}")

        print()
        print("DATASET CONFIGURATION")
        print("-" * 72)
        print(f"Records:             {len(points)}")
        print(f"Features per record: {len(points[0])}")
        print(f"Requested clusters:  {k}")
        print(f"RDD partitions:      {rdd.getNumPartitions()}")
        print(f"Partition sizes:     {partition_sizes}")
        print(f"Random seed:         {random_seed}")
        print(f"Tolerance:           {tolerance}")

        result = fit_parallel_kmeans(
            rdd,
            k=k,
            local_max_iterations=100,
            global_max_iterations=100,
            tolerance=tolerance,
            random_seed=random_seed,
        )

        print()
        print("PARTITION-LEVEL LOCAL K-MEANS")
        print("-" * 72)

        for partition_result in sorted(
            result.partition_results,
            key=lambda item: item.partition_index,
        ):
            print(
                f"Partition {partition_result.partition_index}: "
                f"{partition_result.record_count} records, "
                f"{partition_result.iterations} local iterations, "
                f"converged={partition_result.converged}"
            )

            for center_index, center in enumerate(
                partition_result.centers,
                start=1,
            ):
                print(
                    f"    Local center {center_index}: "
                    f"{format_point(center)}"
                )

        print()
        print("GLOBAL INITIALIZATION")
        print("-" * 72)
        print(
            "Candidate centers:   "
            f"{result.initialization.candidate_count}"
        )

        for center_index, center in enumerate(
            result.initialization.global_centers,
            start=1,
        ):
            print(
                f"Initial center {center_index}: "
                f"{format_point(center)}"
            )

        print()
        print("GLOBAL DISTRIBUTED K-MEANS")
        print("-" * 72)

        for metric in result.global_result.history:
            print(
                f"Iteration {metric.iteration:>2}: "
                f"SSE={metric.sse:.6f}, "
                f"max squared shift="
                f"{metric.maximum_center_shift:.8f}, "
                f"cluster counts={metric.cluster_counts}"
            )

        print()
        print("FINAL RESULT")
        print("-" * 72)
        print(f"Converged:           {result.converged}")
        print(f"Global iterations:   {result.iterations}")
        print(f"Final SSE:           {result.sse:.6f}")
        print(f"Cluster counts:      {result.cluster_counts}")

        for center_index, center in enumerate(
            result.centers,
            start=1,
        ):
            print(
                f"Final center {center_index}: "
                f"{format_point(center)}"
            )

        print()
        print("VALIDATION")
        print("-" * 72)

        total_clustered_records = sum(
            result.cluster_counts
        )

        print(
            "All records assigned: "
            f"{total_clustered_records == len(points)}"
        )

        print(
            "Expected records:      "
            f"{len(points)}"
        )

        print(
            "Assigned records:      "
            f"{total_clustered_records}"
        )

        print()
        print("=" * 72)
        print("DEMONSTRATION COMPLETE")
        print("=" * 72)
        print()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
