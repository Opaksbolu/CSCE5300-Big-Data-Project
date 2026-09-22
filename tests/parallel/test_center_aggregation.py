"""
Tests for candidate-center aggregation.

These tests verify the stage that converts local centers produced by
Spark partitions into global initialization centers for the later
distributed K-Means algorithm.
"""

from __future__ import annotations

import pytest

from src.parallel.center_aggregation import (
    aggregate_candidate_centers,
    aggregate_partition_results,
)
from src.parallel.partition_clustering import (
    PartitionClusteringResult,
)


def test_candidate_centers_are_reduced_to_global_centers() -> None:
    candidate_centers = (
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (9.0, 9.0),
        (9.2, 8.8),
        (8.8, 9.2),
    )

    result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=7,
    )

    assert result.candidate_count == 6
    assert len(result.candidate_centers) == 6
    assert len(result.global_centers) == 2
    assert result.converged is True
    assert result.iterations > 0
    assert result.sse >= 0.0

    sorted_centers = sorted(result.global_centers)

    assert sorted_centers[0] == pytest.approx((1.0, 1.0))
    assert sorted_centers[1] == pytest.approx((9.0, 9.0))


def test_candidate_center_aggregation_is_reproducible() -> None:
    candidate_centers = (
        (1.0, 1.0),
        (1.1, 0.9),
        (0.9, 1.1),
        (9.0, 9.0),
        (9.1, 8.9),
        (8.9, 9.1),
    )

    first_result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=17,
    )

    second_result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=17,
    )

    assert first_result == second_result


def test_partition_results_are_aggregated_into_global_centers() -> None:
    partition_results = (
        PartitionClusteringResult(
            partition_index=0,
            record_count=100,
            centers=((1.0, 1.0), (9.0, 9.0)),
            iterations=2,
            converged=True,
            sse=1.0,
        ),
        PartitionClusteringResult(
            partition_index=1,
            record_count=100,
            centers=((1.2, 0.8), (8.8, 9.2)),
            iterations=3,
            converged=True,
            sse=1.2,
        ),
        PartitionClusteringResult(
            partition_index=2,
            record_count=100,
            centers=((0.8, 1.2), (9.2, 8.8)),
            iterations=2,
            converged=True,
            sse=0.9,
        ),
    )

    result = aggregate_partition_results(
        partition_results,
        k=2,
        random_seed=7,
    )

    assert result.candidate_count == 6
    assert len(result.global_centers) == 2

    sorted_centers = sorted(result.global_centers)

    assert sorted_centers[0] == pytest.approx((1.0, 1.0))
    assert sorted_centers[1] == pytest.approx((9.0, 9.0))


def test_rejects_fewer_candidate_centers_than_k() -> None:
    candidate_centers = (
        (1.0, 1.0),
        (9.0, 9.0),
    )

    with pytest.raises(
        ValueError,
        match="candidate centers must be at least k",
    ):
        aggregate_candidate_centers(
            candidate_centers,
            k=3,
        )


def test_rejects_nonpositive_k() -> None:
    candidate_centers = (
        (1.0, 1.0),
        (9.0, 9.0),
    )

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        aggregate_candidate_centers(
            candidate_centers,
            k=0,
        )


def test_empty_partition_results_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="candidate centers must be at least k",
    ):
        aggregate_partition_results(
            (),
            k=2,
        )
