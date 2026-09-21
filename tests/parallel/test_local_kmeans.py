"""
Automated tests for the deterministic local K-Means implementation.

These tests validate the mathematical foundation before the
algorithm is executed inside Spark partitions.
"""

import pytest

from src.parallel.local_kmeans import (
    calculate_sse,
    fit_local_kmeans,
    nearest_center,
    squared_euclidean_distance,
)


def test_squared_euclidean_distance() -> None:
    """Squared distance should follow the Euclidean distance formula."""

    point = (1.0, 2.0)
    center = (4.0, 6.0)

    result = squared_euclidean_distance(
        point,
        center,
    )

    assert result == pytest.approx(25.0)


def test_nearest_center() -> None:
    """Points should be assigned to the nearest center."""

    centers = (
        (0.0, 0.0),
        (10.0, 10.0),
    )

    assert nearest_center(
        (1.0, 1.0),
        centers,
    ) == 0

    assert nearest_center(
        (9.0, 9.0),
        centers,
    ) == 1


def test_local_kmeans_finds_two_obvious_clusters() -> None:
    """
    K-Means should recover two clearly separated synthetic clusters.
    """

    points = (
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (9.0, 9.0),
        (8.9, 9.1),
        (9.1, 8.9),
    )

    result = fit_local_kmeans(
        points,
        k=2,
        random_seed=7,
    )

    ordered_centers = sorted(
        result.centers,
        key=lambda center: center[0],
    )

    assert ordered_centers[0] == pytest.approx(
        (1.0, 1.0)
    )

    assert ordered_centers[1] == pytest.approx(
        (9.0, 9.0)
    )

    assert result.converged is True
    assert result.sse == pytest.approx(0.08)


def test_local_kmeans_is_reproducible() -> None:
    """The same random seed should produce the same result."""

    points = tuple(
        (float(value), float(value))
        for value in range(20)
    )

    first_result = fit_local_kmeans(
        points,
        k=3,
        random_seed=42,
    )

    second_result = fit_local_kmeans(
        points,
        k=3,
        random_seed=42,
    )

    assert first_result == second_result


def test_calculate_sse() -> None:
    """SSE should equal the total squared within-cluster distance."""

    points = (
        (0.0, 0.0),
        (2.0, 0.0),
    )

    centers = (
        (1.0, 0.0),
    )

    assignments = (0, 0)

    result = calculate_sse(
        points,
        assignments,
        centers,
    )

    assert result == pytest.approx(2.0)


def test_rejects_empty_dataset() -> None:
    """K-Means should reject an empty dataset."""

    with pytest.raises(
        ValueError,
        match="at least one data point",
    ):
        fit_local_kmeans(
            (),
            k=1,
        )


def test_rejects_invalid_k() -> None:
    """The number of clusters must be positive."""

    points = (
        (1.0, 1.0),
        (2.0, 2.0),
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        fit_local_kmeans(
            points,
            k=0,
        )


def test_rejects_inconsistent_dimensions() -> None:
    """Every point must have the same number of features."""

    points = (
        (1.0, 2.0),
        (3.0,),
    )

    with pytest.raises(
        ValueError,
        match="same dimensionality",
    ):
        fit_local_kmeans(
            points,
            k=1,
        )


def test_initialization_separates_well_spaced_clusters() -> None:
    """
    Regression test for the initialization failure discovered during
    development.

    With the fixed seed below, uniform random initialization
    previously selected poor starting centers and converged to an
    incorrect solution. K-Means++ should reliably separate these
    clearly distinct groups.
    """

    points = (
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (9.0, 9.0),
        (8.9, 9.1),
        (9.1, 8.9),
    )

    result = fit_local_kmeans(
        points,
        k=2,
        random_seed=7,
    )

    ordered_centers = sorted(
        result.centers,
        key=lambda center: center[0],
    )

    assert ordered_centers[0] == pytest.approx(
        (1.0, 1.0)
    )

    assert ordered_centers[1] == pytest.approx(
        (9.0, 9.0)
    )

    assert result.converged is True


def test_rejects_k_larger_than_dataset() -> None:
    """K cannot exceed the number of available data points."""

    points = (
        (1.0, 1.0),
        (2.0, 2.0),
    )

    with pytest.raises(
        ValueError,
        match="cannot be greater",
    ):
        fit_local_kmeans(
            points,
            k=3,
        )


def test_rejects_negative_tolerance() -> None:
    """Convergence tolerance cannot be negative."""

    points = (
        (1.0, 1.0),
        (2.0, 2.0),
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        fit_local_kmeans(
            points,
            k=1,
            tolerance=-0.1,
        )


def test_rejects_nonpositive_max_iterations() -> None:
    """The algorithm must allow at least one iteration."""

    points = (
        (1.0, 1.0),
        (2.0, 2.0),
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        fit_local_kmeans(
            points,
            k=1,
            max_iterations=0,
        )


def test_rejects_nonfinite_values() -> None:
    """NaN and infinity must not enter the clustering calculation."""

    points = (
        (1.0, 1.0),
        (float("nan"), 2.0),
    )

    with pytest.raises(
        ValueError,
        match="finite numeric values",
    ):
        fit_local_kmeans(
            points,
            k=1,
        )


def test_sse_rejects_assignment_length_mismatch() -> None:
    """SSE requires exactly one assignment for every data point."""

    points = (
        (1.0, 1.0),
        (2.0, 2.0),
    )

    centers = (
        (1.5, 1.5),
    )

    with pytest.raises(
        ValueError,
        match="exactly one assignment",
    ):
        calculate_sse(
            points,
            (0,),
            centers,
        )