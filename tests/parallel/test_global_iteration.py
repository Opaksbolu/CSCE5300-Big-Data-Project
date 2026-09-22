"""
Tests for one distributed global K-Means iteration.

These tests verify both the mathematical helper functions and the
actual Spark execution used to update global cluster centers.
"""

from __future__ import annotations

import pytest

from src.parallel.global_iteration import (
    ClusterStatistics,
    _calculate_updated_center,
    _maximum_center_shift,
    _merge_cluster_statistics,
    _point_to_cluster_statistics,
    run_global_iteration,
)
from src.parallel.spark_session import (
    create_spark_session,
    stop_spark_session,
)


@pytest.fixture(scope="module")
def spark():
    """
    Create one Spark session for the global-iteration tests.
    """

    session = create_spark_session(
        app_name="CSCE5300-Global-Iteration-Tests",
        master="local[2]",
        log_level="ERROR",
    )

    yield session

    stop_spark_session(session)


def test_point_is_assigned_to_nearest_center() -> None:
    """
    Verify that one point produces statistics for its nearest center.
    """

    centers = (
        (1.0, 1.0),
        (9.0, 9.0),
    )

    cluster_index, statistics = _point_to_cluster_statistics(
        (1.2, 0.8),
        centers,
    )

    assert cluster_index == 0
    assert statistics.coordinate_sums == (1.2, 0.8)
    assert statistics.count == 1
    assert statistics.sse == pytest.approx(0.08)


def test_cluster_statistics_are_merged_correctly() -> None:
    """
    Verify coordinate sums, counts, and SSE values are accumulated.
    """

    left = ClusterStatistics(
        coordinate_sums=(2.0, 4.0),
        count=2,
        sse=0.25,
    )

    right = ClusterStatistics(
        coordinate_sums=(3.0, 5.0),
        count=3,
        sse=0.75,
    )

    merged = _merge_cluster_statistics(
        left,
        right,
    )

    assert merged.coordinate_sums == (5.0, 9.0)
    assert merged.count == 5
    assert merged.sse == pytest.approx(1.0)


def test_updated_center_is_calculated_from_cluster_statistics() -> None:
    """
    Verify that aggregated coordinate sums are converted to a mean.
    """

    statistics = ClusterStatistics(
        coordinate_sums=(8.0, 12.0),
        count=4,
        sse=1.0,
    )

    center = _calculate_updated_center(
        statistics
    )

    assert center == pytest.approx(
        (2.0, 3.0)
    )


def test_maximum_center_shift_is_calculated_correctly() -> None:
    """
    Verify that the largest squared center movement is returned.
    """

    old_centers = (
        (1.0, 1.0),
        (9.0, 9.0),
    )

    new_centers = (
        (1.0, 2.0),
        (11.0, 9.0),
    )

    shift = _maximum_center_shift(
        old_centers,
        new_centers,
    )

    assert shift == pytest.approx(4.0)


def test_global_iteration_updates_centers_with_spark(spark) -> None:
    """
    Verify one complete global K-Means iteration using a real RDD.
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

    initial_centers = (
        (0.0, 0.0),
        (10.0, 10.0),
    )

    rdd = spark.sparkContext.parallelize(
        points,
        numSlices=2,
    )

    result = run_global_iteration(
        rdd,
        initial_centers,
    )

    assert result.centers[0] == pytest.approx(
        (1.0, 1.0)
    )

    assert result.centers[1] == pytest.approx(
        (9.0, 9.0)
    )

    assert result.cluster_counts == (
        4,
        4,
    )

    assert result.sse == pytest.approx(
        16.32
    )

    assert result.maximum_center_shift == pytest.approx(
        2.0
    )


def test_global_iteration_preserves_empty_cluster_center(spark) -> None:
    """
    Verify that a center is retained when no points are assigned to it.
    """

    points = [
        (0.0, 0.0),
        (0.2, 0.0),
        (0.0, 0.2),
        (0.2, 0.2),
    ]

    initial_centers = (
        (0.0, 0.0),
        (100.0, 100.0),
    )

    rdd = spark.sparkContext.parallelize(
        points,
        numSlices=2,
    )

    result = run_global_iteration(
        rdd,
        initial_centers,
    )

    assert result.centers[0] == pytest.approx(
        (0.1, 0.1)
    )

    assert result.centers[1] == (
        100.0,
        100.0,
    )

    assert result.cluster_counts == (
        4,
        0,
    )


def test_global_iteration_rejects_empty_centers(spark) -> None:
    """
    Verify that at least one global center is required.
    """

    rdd = spark.sparkContext.parallelize(
        [(1.0, 1.0)],
        numSlices=1,
    )

    with pytest.raises(
        ValueError,
        match="At least one center",
    ):
        run_global_iteration(
            rdd,
            (),
        )


def test_merge_rejects_mismatched_dimensions() -> None:
    """
    Verify incompatible cluster statistics cannot be combined.
    """

    left = ClusterStatistics(
        coordinate_sums=(1.0, 2.0),
        count=1,
        sse=0.0,
    )

    right = ClusterStatistics(
        coordinate_sums=(1.0, 2.0, 3.0),
        count=1,
        sse=0.0,
    )

    with pytest.raises(
        ValueError,
        match="same dimensionality",
    ):
        _merge_cluster_statistics(
            left,
            right,
        )
