"""
Demonstration of Spark RDD partitioning for the CSCE 5300 project.

This script validates the distributed execution foundation that will
later be used by the Parallel K-Means implementation.
"""

from __future__ import annotations

from src.parallel.partitioning import (
    count_partition_records,
    validate_partition_count,
)
from src.parallel.spark_session import (
    create_spark_session,
    stop_spark_session,
)


def main() -> None:
    """Run the Spark partitioning demonstration."""

    spark = create_spark_session(
        app_name="CSCE5300-Partition-Demo",
        master="local[4]",
        log_level="ERROR",
    )

    sc = spark.sparkContext

    try:
        print()
        print("=" * 64)
        print("CSCE 5300 — SPARK PARTITIONING DEMONSTRATION")
        print("=" * 64)

        # A controlled dataset makes it easy to verify correctness.
        records = list(range(1, 1001))

        number_of_partitions = 4

        rdd = sc.parallelize(
            records,
            numSlices=number_of_partitions,
        )

        partition_counts = count_partition_records(rdd)

        total_records = rdd.count()
        total_sum = rdd.sum()

        print()
        print("Spark Configuration")
        print("-" * 64)
        print(f"Spark version:        {spark.version}")
        print(f"Spark master:         {sc.master}")
        print(f"Application name:     {sc.appName}")

        print()
        print("Dataset")
        print("-" * 64)
        print(f"Records:              {total_records:,}")
        print(f"Expected sum:         {sum(records):,}")
        print(f"Spark-computed sum:   {int(total_sum):,}")

        print()
        print("Partition Distribution")
        print("-" * 64)

        for partition_index, record_count in partition_counts:
            print(
                f"Partition {partition_index:<3} "
                f"{record_count:>6,} records"
            )

        print()
        print("Validation")
        print("-" * 64)

        partition_test = validate_partition_count(
            rdd,
            expected_partitions=number_of_partitions,
        )

        record_test = total_records == len(records)
        sum_test = int(total_sum) == sum(records)

        print(
            "Partition count:      "
            f"{'PASS' if partition_test else 'FAIL'}"
        )

        print(
            "Record count:         "
            f"{'PASS' if record_test else 'FAIL'}"
        )

        print(
            "Distributed sum:      "
            f"{'PASS' if sum_test else 'FAIL'}"
        )

        all_tests_pass = (
            partition_test
            and record_test
            and sum_test
        )

        print()
        print("=" * 64)

        if all_tests_pass:
            print("RESULT: SPARK DISTRIBUTED FOUNDATION PASSED")
        else:
            print("RESULT: VALIDATION FAILURE")

        print("=" * 64)
        print()

        if not all_tests_pass:
            raise RuntimeError(
                "Spark partition demonstration failed validation."
            )

    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
