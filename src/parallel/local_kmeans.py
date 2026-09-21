"""
Deterministic local K-Means implementation.

This module provides the mathematical clustering core used during
partition-level pre-clustering in the Parallel K-Means project.

The implementation intentionally does not depend on Spark,
scikit-learn, or NumPy. This allows the clustering mathematics to be
tested independently before it is executed inside Spark partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Sequence


Point = tuple[float, ...]


@dataclass(frozen=True)
class LocalKMeansResult:
    """Result returned by the local K-Means algorithm."""

    centers: tuple[Point, ...]
    assignments: tuple[int, ...]
    iterations: int
    converged: bool
    sse: float


def _validate_points(points: Sequence[Point]) -> int:
    """
    Validate input points and return their dimensionality.

    All points must contain at least one feature, have identical
    dimensionality, and contain only finite numeric values.
    """

    if not points:
        raise ValueError("K-Means requires at least one data point.")

    dimensions = len(points[0])

    if dimensions == 0:
        raise ValueError(
            "Data points must contain at least one feature."
        )

    for point in points:
        if len(point) != dimensions:
            raise ValueError(
                "All data points must have the same dimensionality."
            )

        if not all(isfinite(value) for value in point):
            raise ValueError(
                "Data points must contain only finite numeric values."
            )

    return dimensions


def squared_euclidean_distance(
    point: Point,
    center: Point,
) -> float:
    """
    Calculate squared Euclidean distance between two vectors.

    The square root is unnecessary for nearest-center comparisons
    because it does not change the ordering of distances.
    """

    if len(point) != len(center):
        raise ValueError(
            "Point and center must have the same dimensionality."
        )

    return sum(
        (point_value - center_value) ** 2
        for point_value, center_value in zip(point, center)
    )


def nearest_center(
    point: Point,
    centers: Sequence[Point],
) -> int:
    """
    Return the index of the center nearest to a data point.

    If two centers are equally distant, the center with the smaller
    index is selected deterministically.
    """

    if not centers:
        raise ValueError("At least one center is required.")

    return min(
        range(len(centers)),
        key=lambda index: squared_euclidean_distance(
            point,
            centers[index],
        ),
    )


def _mean_point(
    points: Sequence[Point],
    dimensions: int,
) -> Point:
    """Calculate the coordinate-wise mean of a collection of points."""

    return tuple(
        sum(point[dimension] for point in points) / len(points)
        for dimension in range(dimensions)
    )


def _initialize_centers(
    points: Sequence[Point],
    k: int,
    random_seed: int,
) -> tuple[Point, ...]:
    """
    Initialize cluster centers reproducibly using K-Means++.

    The first center is selected randomly using the supplied seed.
    Each subsequent center is sampled with probability proportional
    to its squared distance from the nearest center already selected.

    This generally produces better-spread initial centers than
    uniform random initialization.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if k > len(points):
        raise ValueError(
            "k cannot be greater than the number of data points."
        )

    random_generator = Random(random_seed)

    first_index = random_generator.randrange(len(points))

    centers: list[Point] = [
        points[first_index]
    ]

    selected_indices = {first_index}

    while len(centers) < k:
        distances: list[float] = []

        for index, point in enumerate(points):
            if index in selected_indices:
                distances.append(0.0)
                continue

            nearest_distance = min(
                squared_euclidean_distance(
                    point,
                    center,
                )
                for center in centers
            )

            distances.append(nearest_distance)

        total_distance = sum(distances)

        if total_distance == 0.0:
            remaining_indices = [
                index
                for index in range(len(points))
                if index not in selected_indices
            ]

            selected_index = random_generator.choice(
                remaining_indices
            )

        else:
            threshold = (
                random_generator.random()
                * total_distance
            )

            cumulative_distance = 0.0
            selected_index: int | None = None

            for index, distance in enumerate(distances):
                cumulative_distance += distance

                if cumulative_distance >= threshold:
                    selected_index = index
                    break

            if selected_index is None:
                selected_index = max(
                    (
                        index
                        for index in range(len(points))
                        if index not in selected_indices
                    ),
                    key=lambda index: distances[index],
                )

        selected_indices.add(selected_index)
        centers.append(points[selected_index])

    return tuple(centers)


def _assign_points(
    points: Sequence[Point],
    centers: Sequence[Point],
) -> tuple[int, ...]:
    """Assign every point to its nearest cluster center."""

    return tuple(
        nearest_center(point, centers)
        for point in points
    )


def _update_centers(
    points: Sequence[Point],
    assignments: Sequence[int],
    previous_centers: Sequence[Point],
    dimensions: int,
) -> tuple[Point, ...]:
    """
    Recalculate cluster centers from current assignments.

    If a cluster receives no points during an iteration, its previous
    center is retained rather than producing an undefined mean.
    """

    new_centers: list[Point] = []

    for cluster_index in range(len(previous_centers)):
        cluster_points = [
            point
            for point, assignment in zip(points, assignments)
            if assignment == cluster_index
        ]

        if cluster_points:
            new_centers.append(
                _mean_point(
                    cluster_points,
                    dimensions,
                )
            )
        else:
            new_centers.append(
                previous_centers[cluster_index]
            )

    return tuple(new_centers)


def _maximum_center_shift(
    old_centers: Sequence[Point],
    new_centers: Sequence[Point],
) -> float:
    """
    Return the largest squared movement of any cluster center.
    """

    return max(
        squared_euclidean_distance(
            old_center,
            new_center,
        )
        for old_center, new_center in zip(
            old_centers,
            new_centers,
        )
    )


def calculate_sse(
    points: Sequence[Point],
    assignments: Sequence[int],
    centers: Sequence[Point],
) -> float:
    """
    Calculate the Sum of Squared Errors (SSE).

    SSE measures the total squared distance between each data point
    and its assigned cluster center. Lower values indicate more
    compact clusters for the same dataset and value of k.
    """

    if len(points) != len(assignments):
        raise ValueError(
            "Each data point must have exactly one assignment."
        )

    if not centers:
        raise ValueError(
            "At least one center is required to calculate SSE."
        )

    for assignment in assignments:
        if assignment < 0 or assignment >= len(centers):
            raise ValueError(
                "Cluster assignment index is out of range."
            )

    return sum(
        squared_euclidean_distance(
            point,
            centers[assignment],
        )
        for point, assignment in zip(
            points,
            assignments,
        )
    )


def fit_local_kmeans(
    points: Sequence[Point],
    k: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    random_seed: int = 42,
) -> LocalKMeansResult:
    """
    Cluster a local collection of feature vectors using K-Means.

    Parameters
    ----------
    points:
        Input feature vectors.

    k:
        Number of clusters.

    max_iterations:
        Maximum number of center-update iterations.

    tolerance:
        Convergence threshold applied to squared center movement.

    random_seed:
        Seed used for reproducible K-Means++ initialization.

    Returns
    -------
    LocalKMeansResult
        Final centers, assignments, iteration count, convergence
        status, and SSE.
    """

    dimensions = _validate_points(points)

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if k > len(points):
        raise ValueError(
            "k cannot be greater than the number of data points."
        )

    if max_iterations <= 0:
        raise ValueError(
            "max_iterations must be greater than zero."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    centers = _initialize_centers(
        points,
        k,
        random_seed,
    )

    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        assignments = _assign_points(
            points,
            centers,
        )

        new_centers = _update_centers(
            points,
            assignments,
            centers,
            dimensions,
        )

        shift = _maximum_center_shift(
            centers,
            new_centers,
        )

        centers = new_centers
        iterations = iteration

        if shift <= tolerance:
            converged = True
            break

    # Perform one final assignment using the final centers so that
    # assignments and SSE correspond exactly to the returned model.
    assignments = _assign_points(
        points,
        centers,
    )

    sse = calculate_sse(
        points,
        assignments,
        centers,
    )

    return LocalKMeansResult(
        centers=centers,
        assignments=assignments,
        iterations=iterations,
        converged=converged,
        sse=sse,
    )