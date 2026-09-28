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

def test_multiple_restarts_select_lowest_sse_aggregation() -> None:
    """
    Multiple aggregation restarts should retain the lowest-SSE result.

    This fixture reproduces an initialization failure discovered during
    parallel K-Means validation. A single aggregation run with seed 42
    converges to a poor local optimum, while later deterministic seeds
    find a substantially better clustering.
    """

    candidate_centers = (
        (-0.7317086365368399, -1.7628963352893607),
        (1.9693375952864272, -1.603458695675816),
        (-0.8609456115113736, 0.09501216591092154),
        (1.024854072653738, 1.5085007195864208),
        (9.222684727683513, 0.522972230539271),
        (10.32012589776767, 1.2173563001258825),
        (8.34188888207252, 1.5511435300021321),
        (9.82159809833, -0.47592526035837635),
        (1.1533840288683153, 10.42693919665697),
        (-1.5325661032310243, 11.543021561294974),
        (0.22235367257533367, 9.953083760216886),
        (-0.6890060890089766, 10.929828071216907),
        (8.57902899185439, 9.265485605187301),
        (8.275916078487786, 10.25695462713237),
        (8.558225772709864, 9.360282475292909),
        (9.355999793734616, 8.777710827093234),
    )

    result = aggregate_candidate_centers(
        candidate_centers,
        k=4,
        random_seed=42,
        num_restarts=5,
    )

    assert result.converged is True
    assert result.sse == pytest.approx(
        24.684190995687825
    )

def test_rejects_nonpositive_num_restarts() -> None:
    """Aggregation requires at least one initialization restart."""

    candidate_centers = (
        (0.0, 0.0),
        (1.0, 1.0),
    )

    for num_restarts in (0, -1):
        with pytest.raises(
            ValueError,
            match="num_restarts must be greater than zero",
        ):
            aggregate_candidate_centers(
                candidate_centers,
                k=1,
                num_restarts=num_restarts,
            )


def test_multiple_restarts_are_reproducible() -> None:
    """Identical restart settings should produce identical results."""

    candidate_centers = (
        (0.0, 0.0),
        (0.1, 0.1),
        (9.9, 9.9),
        (10.0, 10.0),
    )

    first_result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=42,
        num_restarts=5,
    )

    second_result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=42,
        num_restarts=5,
    )

    assert first_result == second_result


def test_one_restart_preserves_existing_behavior() -> None:
    """One restart should match the original single-run behavior."""

    candidate_centers = (
        (0.0, 0.0),
        (0.1, 0.1),
        (9.9, 9.9),
        (10.0, 10.0),
    )

    default_result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=42,
    )

    explicit_result = aggregate_candidate_centers(
        candidate_centers,
        k=2,
        random_seed=42,
        num_restarts=1,
    )

    assert explicit_result == default_result
