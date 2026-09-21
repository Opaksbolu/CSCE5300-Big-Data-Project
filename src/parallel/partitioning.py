"""
RDD partition utilities for the Parallel K-Means implementation.

The functions in this module make Spark's data distribution visible
and measurable. This is useful for validating the parallel execution
strategy and documenting experimental configurations.
"""

from __future__ import annotations

from typing import Any

from pyspark import RDD


def count_partition_records(rdd: RDD) -> list[tuple[int, int]]:
    """
    Count the number of records stored in each RDD partition.

    Parameters
    ----------
    rdd:
        Input Spark RDD.

    Returns
    -------
    list[tuple[int, int]]
        Pairs containing (partition_index, record_count).
    """

    def count_partition(
        partition_index: int,
        iterator: Any,
    ):
        count = sum(1 for _ in iterator)
        yield partition_index, count

    return sorted(
        rdd.mapPartitionsWithIndex(count_partition).collect(),
        key=lambda item: item[0],
    )


def validate_partition_count(
    rdd: RDD,
    expected_partitions: int,
) -> bool:
    """
    Check whether an RDD has the expected number of partitions.
    """

    return rdd.getNumPartitions() == expected_partitions
