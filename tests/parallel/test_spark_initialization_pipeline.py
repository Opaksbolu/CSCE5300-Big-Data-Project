"""
End-to-end integration test for the Spark K-Means initialization pipeline.

This test verifies the complete initialization path:

    Spark RDD
        -> partition-level local K-Means
        -> candidate centers
        -> center aggregation
        -> global initialization centers

The global centers produced here will later be supplied to the
iterative parallel K-Means stage.
"""

from __future__ import annotations

import pytest

from src.parallel.center_aggregation import aggregate_partition_results
from src.parallel.partition_clustering import cluster_rdd_partitions


def test_spark_initialization_pipeline_produces_global_centers(
    spark,
) -> None:
    """
    Verify the complete distributed initialization pipeline.

    The input contains two clearly separated natural clusters.
    Spark divides the records into partitions, each partition performs
    local K-Means, and the resulting candidate centers are aggregated
    into two global initialization centers.
    """

    points = [
        (0.8, 1.0),
        (1.0, 0.8),
        (1.0, 1.0),
        (1.0, 1.2),
        (1.2, 1.0),
        (1.2, 1.2),
        (8.8, 9.0),
        (9.0, 8.8),
        (9.0, 9.0),
        (9.0, 9.2),
        (9.2, 9.0),
        (9.2, 9.2),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        numSlices=2,
    )

    partition_results = (
        cluster_rdd_partitions(
            rdd,
            k=2,
            random_seed=7,
        )
        .collect()
    )

    aggregation_result = aggregate_partition_results(
        partition_results,
        k=2,
        random_seed=7,
    )

    assert rdd.getNumPartitions() == 2
    assert len(partition_results) == 2

    assert (
        sum(
            result.record_count
            for result in partition_results
        )
        == len(points)
    )

    assert all(
        result.converged
        for result in partition_results
    )

    assert aggregation_result.candidate_count == 4
    assert len(aggregation_result.candidate_centers) == 4
    assert len(aggregation_result.global_centers) == 2

    assert aggregation_result.converged is True
    assert aggregation_result.iterations > 0
    assert aggregation_result.sse >= 0.0

    sorted_centers = sorted(
        aggregation_result.global_centers
    )

    assert sorted_centers[0] == pytest.approx(
        (1.0333333333333334, 1.0333333333333334),
        abs=0.25,
    )

    assert sorted_centers[1] == pytest.approx(
        (9.033333333333333, 9.033333333333333),
        abs=0.25,
    )


def test_spark_initialization_pipeline_is_reproducible(
    spark,
) -> None:
    """Verify deterministic results when the same seed is reused."""

    points = [
        (0.8, 1.0),
        (1.0, 0.8),
        (1.0, 1.0),
        (1.0, 1.2),
        (1.2, 1.0),
        (1.2, 1.2),
        (8.8, 9.0),
        (9.0, 8.8),
        (9.0, 9.0),
        (9.0, 9.2),
        (9.2, 9.0),
        (9.2, 9.2),
    ]

    def run_pipeline():
        rdd = spark.sparkContext.parallelize(
            points,
            numSlices=2,
        )

        partition_results = (
            cluster_rdd_partitions(
                rdd,
                k=2,
                random_seed=11,
            )
            .collect()
        )

        return aggregate_partition_results(
            partition_results,
            k=2,
            random_seed=11,
        )

    first_result = run_pipeline()
    second_result = run_pipeline()

    assert first_result == second_result
