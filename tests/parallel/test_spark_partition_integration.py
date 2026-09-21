"""
Spark integration tests for partition-level K-Means clustering.

These tests verify that the local K-Means implementation can execute
inside actual Spark RDD partitions and return partition-level
clustering results to the driver.
"""

from __future__ import annotations

import pytest

from src.parallel.partition_clustering import (
    cluster_rdd_partitions,
    collect_candidate_centers,
)
from src.parallel.spark_session import (
    create_spark_session,
    stop_spark_session,
)


@pytest.fixture(scope="module")
def spark():
    """
    Create one Spark session for this integration-test module.
    """
    session = create_spark_session(
        app_name="CSCE5300-Partition-Integration-Tests",
        master="local[2]",
        log_level="ERROR",
    )

    yield session

    stop_spark_session(session)


def test_spark_executes_local_kmeans_across_partitions(spark) -> None:
    """
    Verify that Spark executes local K-Means independently in
    multiple RDD partitions.
    """
    sc = spark.sparkContext

    points = [
        (1.0, 1.0),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.2, 0.8),
        (8.8, 9.2),
        (9.0, 9.0),
        (9.1, 8.9),
        (8.9, 9.1),
    ]

    rdd = sc.parallelize(
        points,
        numSlices=2,
    )

    results = cluster_rdd_partitions(
        rdd,
        k=2,
        random_seed=42,
    ).collect()

    assert len(results) == 2

    assert {
        result.partition_index
        for result in results
    } == {0, 1}

    assert sum(
        result.record_count
        for result in results
    ) == len(points)

    assert all(
        result.converged
        for result in results
    )

    assert all(
        len(result.centers) == 2
        for result in results
    )


def test_spark_partition_results_produce_candidate_centers(spark) -> None:
    """
    Verify that centers returned by Spark partitions can be flattened
    into the candidate-center collection required by the next stage.
    """
    sc = spark.sparkContext

    points = [
        (1.0, 1.0),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.2, 0.8),
        (8.8, 9.2),
        (9.0, 9.0),
        (9.1, 8.9),
        (8.9, 9.1),
    ]

    rdd = sc.parallelize(
        points,
        numSlices=2,
    )

    results = cluster_rdd_partitions(
        rdd,
        k=2,
        random_seed=42,
    ).collect()

    candidate_centers = collect_candidate_centers(results)

    assert len(candidate_centers) == 4

    assert all(
        len(center) == 2
        for center in candidate_centers
    )
