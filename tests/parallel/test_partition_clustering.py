"""
Tests for Spark partition-level K-Means support.

These tests validate the partition clustering logic independently
before the complete parallel K-Means workflow is constructed.
"""

import pytest

from src.parallel.partition_clustering import (
    cluster_partition,
    collect_candidate_centers,
)


def test_cluster_partition_returns_expected_metadata() -> None:
    points = (
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (9.0, 9.0),
        (8.9, 9.1),
        (9.1, 8.9),
    )

    results = tuple(
        cluster_partition(
            0,
            points,
            k=2,
            random_seed=7,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.partition_index == 0
    assert result.record_count == 6
    assert len(result.centers) == 2
    assert result.converged is True
    assert result.sse == pytest.approx(0.08)


def test_cluster_partition_is_reproducible() -> None:
    points = tuple(
        (float(value), float(value))
        for value in range(20)
    )

    first_result = tuple(
        cluster_partition(
            2,
            points,
            k=3,
            random_seed=42,
        )
    )

    second_result = tuple(
        cluster_partition(
            2,
            points,
            k=3,
            random_seed=42,
        )
    )

    assert first_result == second_result


def test_empty_partition_returns_no_result() -> None:
    results = tuple(
        cluster_partition(
            0,
            (),
            k=2,
        )
    )

    assert results == ()


def test_partition_rejects_fewer_records_than_k() -> None:
    points = (
        (1.0, 1.0),
    )

    with pytest.raises(
        ValueError,
        match="must contain at least k records",
    ):
        tuple(
            cluster_partition(
                3,
                points,
                k=2,
            )
        )


def test_collect_candidate_centers() -> None:
    first_points = (
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (9.0, 9.0),
        (8.9, 9.1),
        (9.1, 8.9),
    )

    second_points = (
        (20.0, 20.0),
        (19.9, 20.1),
        (20.1, 19.9),
        (30.0, 30.0),
        (29.9, 30.1),
        (30.1, 29.9),
    )

    first_result = tuple(
        cluster_partition(
            0,
            first_points,
            k=2,
            random_seed=7,
        )
    )[0]

    second_result = tuple(
        cluster_partition(
            1,
            second_points,
            k=2,
            random_seed=7,
        )
    )[0]

    candidate_centers = collect_candidate_centers(
        (
            first_result,
            second_result,
        )
    )

    assert len(candidate_centers) == 4
