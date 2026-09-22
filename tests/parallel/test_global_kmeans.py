"""
Tests for the distributed global K-Means convergence loop.
"""

from __future__ import annotations

import pytest

from src.parallel.global_kmeans import (
    evaluate_centers,
    fit_global_kmeans,
)


def test_global_kmeans_converges_on_obvious_clusters(spark) -> None:
    """
    Distributed K-Means should converge to the means of two
    clearly separated clusters.
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

    result = fit_global_kmeans(
        rdd,
        initial_centers=(
            (0.0, 0.0),
            (10.0, 10.0),
        ),
        max_iterations=20,
        tolerance=1e-12,
    )

    assert result.converged is True

    assert result.iterations >= 1
    assert result.iterations <= 20

    sorted_centers = sorted(
        result.centers,
        key=lambda center: center[0],
    )

    assert sorted_centers[0] == pytest.approx(
        (1.0, 1.0)
    )

    assert sorted_centers[1] == pytest.approx(
        (9.0, 9.0)
    )

    assert sorted(result.cluster_counts) == [4, 4]

    assert result.sse >= 0.0


def test_global_kmeans_records_iteration_history(spark) -> None:
    """
    Every completed iteration should have a corresponding
    history record.
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

    result = fit_global_kmeans(
        rdd,
        initial_centers=(
            (0.0, 0.0),
            (10.0, 10.0),
        ),
        max_iterations=20,
        tolerance=1e-12,
    )

    assert len(result.history) == result.iterations

    assert [
        metric.iteration
        for metric in result.history
    ] == list(
        range(
            1,
            result.iterations + 1,
        )
    )

    for metric in result.history:
        assert metric.sse >= 0.0
        assert metric.maximum_center_shift >= 0.0
        assert sum(metric.cluster_counts) == len(points)


def test_global_kmeans_stops_at_maximum_iterations(spark) -> None:
    """
    A deliberately strict tolerance and one-iteration limit should
    stop execution before convergence when centers still move.
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

    result = fit_global_kmeans(
        rdd,
        initial_centers=(
            (0.0, 0.0),
            (10.0, 10.0),
        ),
        max_iterations=1,
        tolerance=0.0,
    )

    assert result.iterations == 1
    assert result.converged is False
    assert len(result.history) == 1


def test_global_kmeans_reports_final_cluster_counts(spark) -> None:
    """
    Final cluster counts should account for every input record.
    """

    points = [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (9.0, 9.0),
        (9.0, 10.0),
        (10.0, 9.0),
    ]

    rdd = spark.sparkContext.parallelize(
        points,
        2,
    )

    result = fit_global_kmeans(
        rdd,
        initial_centers=(
            (1.0, 1.0),
            (10.0, 10.0),
        ),
    )

    assert sum(result.cluster_counts) == len(points)

    assert sorted(result.cluster_counts) == [3, 3]


def test_global_kmeans_rejects_empty_initial_centers(spark) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="At least one initial center",
    ):
        fit_global_kmeans(
            rdd,
            initial_centers=(),
        )


def test_global_kmeans_rejects_nonpositive_max_iterations(
    spark,
) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="max_iterations must be greater than zero",
    ):
        fit_global_kmeans(
            rdd,
            initial_centers=((1.0, 1.0),),
            max_iterations=0,
        )


def test_global_kmeans_rejects_negative_tolerance(spark) -> None:
    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        1,
    )

    with pytest.raises(
        ValueError,
        match="tolerance cannot be negative",
    ):
        fit_global_kmeans(
            rdd,
            initial_centers=((1.0, 1.0),),
            tolerance=-1.0,
        )
def test_final_sse_matches_returned_centers_after_iteration_limit(
    spark,
) -> None:
    """
    Final SSE must be evaluated against the centers returned by the
    algorithm, even when execution stops because of max_iterations.
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

    result = fit_global_kmeans(
        rdd,
        initial_centers=(
            (0.0, 0.0),
            (10.0, 10.0),
        ),
        max_iterations=1,
        tolerance=0.0,
    )

    expected_sse, expected_counts = evaluate_centers(
        rdd,
        result.centers,
    )

    assert result.converged is False

    assert result.sse == pytest.approx(
        expected_sse
    )

    assert result.cluster_counts == expected_counts

    assert result.sse == pytest.approx(16.0)
    assert sorted(result.cluster_counts) == [4, 4]
def test_global_kmeans_rejects_empty_rdd(spark) -> None:
    """
    Distributed K-Means requires at least one input record.
    """

    rdd = spark.sparkContext.emptyRDD()

    with pytest.raises(
        ValueError,
        match="input RDD cannot be empty",
    ):
        fit_global_kmeans(
            rdd,
            initial_centers=((1.0, 1.0),),
        )