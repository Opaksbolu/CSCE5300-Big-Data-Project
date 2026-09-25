"""
End-to-end tests for the improved Parallel K-Means pipeline.

These tests verify that the complete workflow connects correctly:

    Spark RDD
        -> partition-level local K-Means
        -> candidate-center aggregation
        -> distributed global K-Means
        -> final clustering result

The individual mathematical stages are tested separately elsewhere.
This module verifies their integration as one complete algorithm.
"""

from __future__ import annotations

import pytest

from pyspark.errors import PythonException

from src.parallel.parallel_kmeans import fit_parallel_kmeans


def test_parallel_kmeans_executes_complete_pipeline(spark) -> None:
    """
    The complete pipeline should identify two clearly separated
    natural clusters.
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
        2,
    )

    result = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=7,
        tolerance=1e-12,
    )

    assert result.converged is True

    assert len(result.centers) == 2

    sorted_centers = sorted(
        result.centers,
        key=lambda center: center[0],
    )

    assert sorted_centers[0] == pytest.approx(
        (1.0333333333333334, 1.0333333333333334)
    )

    assert sorted_centers[1] == pytest.approx(
        (9.033333333333333, 9.033333333333333)
    )

    assert sorted(result.cluster_counts) == [6, 6]

    assert sum(result.cluster_counts) == len(points)

    assert result.sse >= 0.0


def test_parallel_kmeans_preserves_initialization_metadata(
    spark,
) -> None:
    """
    The orchestration result should expose the partition and
    candidate-center information needed for later analysis.
    """

    points = [
        (0.8, 1.0),
        (1.0, 0.8),
        (1.0, 1.2),
        (1.2, 1.0),
        (8.8, 9.0),
        (9.0, 8.8),
        (9.0, 9.2),
        (9.2, 9.0),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    result = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=42,
    )

    assert len(result.partition_results) == 2

    assert all(
        partition_result.record_count == 4
        for partition_result in result.partition_results
    )

    assert all(
        len(partition_result.centers) == 2
        for partition_result in result.partition_results
    )

    assert result.initialization.candidate_count == 4

    assert len(result.initialization.global_centers) == 2


def test_parallel_kmeans_exposes_global_result(spark) -> None:
    """
    Top-level final values should match the underlying global
    K-Means result.
    """

    points = [
        (0.0, 0.0),
        (0.0, 2.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (8.0, 8.0),
        (8.0, 10.0),
        (10.0, 8.0),
        (10.0, 10.0),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    result = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=42,
    )

    assert result.centers == result.global_result.centers

    assert result.iterations == result.global_result.iterations

    assert result.converged == result.global_result.converged

    assert result.sse == pytest.approx(
        result.global_result.sse
    )

    assert (
        result.cluster_counts
        == result.global_result.cluster_counts
    )


def test_parallel_kmeans_is_reproducible(spark) -> None:
    """
    Identical data and random seeds should produce identical
    clustering results.
    """

    points = [
        (0.8, 1.0),
        (1.0, 0.8),
        (1.0, 1.2),
        (1.2, 1.0),
        (8.8, 9.0),
        (9.0, 8.8),
        (9.0, 9.2),
        (9.2, 9.0),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    first = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=17,
    )

    second = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=17,
    )

    assert first.centers == second.centers

    assert first.cluster_counts == second.cluster_counts

    assert first.sse == pytest.approx(second.sse)

    assert (
        first.initialization.global_centers
        == second.initialization.global_centers
    )


def test_parallel_kmeans_rejects_nonpositive_k(spark) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        fit_parallel_kmeans(
            rdd,
            k=0,
        )


def test_parallel_kmeans_rejects_nonpositive_local_iterations(
    spark,
) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="local_max_iterations must be greater than zero",
    ):
        fit_parallel_kmeans(
            rdd,
            k=1,
            local_max_iterations=0,
        )


def test_parallel_kmeans_rejects_nonpositive_global_iterations(
    spark,
) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="global_max_iterations must be greater than zero",
    ):
        fit_parallel_kmeans(
            rdd,
            k=1,
            global_max_iterations=0,
        )


def test_parallel_kmeans_rejects_negative_tolerance(spark) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="tolerance cannot be negative",
    ):
        fit_parallel_kmeans(
            rdd,
            k=1,
            tolerance=-1.0,
        )


def test_parallel_kmeans_rejects_empty_rdd(spark) -> None:
    rdd = spark.sparkContext.parallelize(
        [],
        2,
    )

    with pytest.raises(
        ValueError,
        match="RDD must contain at least one point",
    ):
        fit_parallel_kmeans(
            rdd,
            k=2,
        )


def test_parallel_kmeans_rejects_partition_smaller_than_k(
    spark,
) -> None:
    """
    The complete pipeline should fail clearly when a non-empty
    Spark partition contains fewer records than the requested
    number of clusters.

    Spark wraps exceptions raised inside Python workers in a
    PythonException when the distributed job is evaluated.
    """

    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (8.0, 8.0),
        (9.0, 9.0),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    with pytest.raises(
        PythonException,
        match="Each non-empty partition must contain at least k records",
    ):
        fit_parallel_kmeans(
            rdd,
            k=3,
            random_seed=42,
        )


def test_parallel_kmeans_reports_nonnegative_timing(
    spark,
) -> None:
    """
    The complete pipeline should expose nonnegative runtime
    measurements for every instrumented stage.
    """

    points = [
        (1.0, 1.0),
        (1.1, 1.0),
        (1.0, 1.1),
        (9.0, 9.0),
        (9.1, 9.0),
        (9.0, 9.1),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    result = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=42,
    )

    timing = result.timing

    assert timing.partition_clustering_seconds >= 0.0
    assert timing.center_aggregation_seconds >= 0.0
    assert timing.initialization_seconds >= 0.0
    assert timing.global_clustering_seconds >= 0.0
    assert timing.final_evaluation_seconds >= 0.0
    assert timing.total_runtime_seconds >= 0.0


def test_parallel_kmeans_timing_is_internally_consistent(
    spark,
) -> None:
    """
    Timing metadata should preserve the relationships between
    initialization, global clustering, and total runtime.
    """

    points = [
        (1.0, 1.0),
        (1.1, 1.0),
        (1.0, 1.1),
        (9.0, 9.0),
        (9.1, 9.0),
        (9.0, 9.1),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    result = fit_parallel_kmeans(
        rdd,
        k=2,
        random_seed=42,
    )

    timing = result.timing

    expected_initialization = (
        timing.partition_clustering_seconds
        + timing.center_aggregation_seconds
    )

    assert (
        timing.initialization_seconds
        == expected_initialization
    )

    assert (
        timing.total_runtime_seconds
        >= timing.initialization_seconds
    )

    assert (
        timing.total_runtime_seconds
        >= timing.global_clustering_seconds
    )

    assert (
        timing.total_runtime_seconds
        >= timing.final_evaluation_seconds
    )

    assert (
        timing.total_runtime_seconds
        >= (
            timing.initialization_seconds
            + timing.global_clustering_seconds
            + timing.final_evaluation_seconds
        )
    )
