"""
Deterministic synthetic dataset generation for clustering experiments.

The experimental framework requires datasets whose size, feature
dimensionality, cluster structure, and random seed can be controlled
precisely. This module provides a lightweight generator without
depending on NumPy or scikit-learn.

Synthetic datasets are useful for scalability experiments because the
number of records can be increased while keeping the data-generation
procedure reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random

from src.parallel.local_kmeans import Point


@dataclass(frozen=True)
class SyntheticDataset:
    """
    Generated synthetic clustering dataset.

    Attributes
    ----------
    points:
        Generated feature vectors.

    labels:
        Ground-truth cluster index for each generated point.

    cluster_centers:
        Centers used to generate the synthetic clusters.

    num_records:
        Total number of generated records.

    num_features:
        Number of features in each record.

    num_clusters:
        Number of synthetic clusters.
    """

    points: tuple[Point, ...]
    labels: tuple[int, ...]
    cluster_centers: tuple[Point, ...]
    num_records: int
    num_features: int
    num_clusters: int


def _validate_generation_parameters(
    *,
    num_records: int,
    num_features: int,
    num_clusters: int,
    cluster_spread: float,
) -> None:
    """Validate synthetic dataset generation parameters."""

    if num_records <= 0:
        raise ValueError(
            "num_records must be greater than zero."
        )

    if num_features <= 0:
        raise ValueError(
            "num_features must be greater than zero."
        )

    if num_clusters <= 0:
        raise ValueError(
            "num_clusters must be greater than zero."
        )

    if num_clusters > num_records:
        raise ValueError(
            "num_clusters cannot exceed num_records."
        )

    if not isfinite(cluster_spread):
        raise ValueError(
            "cluster_spread must be finite."
        )

    if cluster_spread < 0:
        raise ValueError(
            "cluster_spread cannot be negative."
        )


def _generate_cluster_centers(
    *,
    num_clusters: int,
    num_features: int,
    center_separation: float,
) -> tuple[Point, ...]:
    """
    Create deterministic, well-separated synthetic cluster centers.

    Each cluster is shifted along every feature dimension. A small
    feature-specific offset prevents all dimensions from being
    identical while preserving clear cluster separation.
    """

    return tuple(
        tuple(
            (
                cluster_index * center_separation
                + feature_index
            )
            for feature_index in range(num_features)
        )
        for cluster_index in range(num_clusters)
    )


def generate_synthetic_dataset(
    *,
    num_records: int,
    num_features: int,
    num_clusters: int,
    cluster_spread: float = 1.0,
    center_separation: float = 10.0,
    random_seed: int = 42,
) -> SyntheticDataset:
    """
    Generate a reproducible clustered synthetic dataset.

    Records are distributed as evenly as possible across the requested
    clusters. Each feature is sampled from a Gaussian distribution
    centered on its cluster center.

    Parameters
    ----------
    num_records:
        Number of records to generate.

    num_features:
        Number of numeric features per record.

    num_clusters:
        Number of ground-truth clusters.

    cluster_spread:
        Standard deviation of the Gaussian noise around each center.

    center_separation:
        Distance used to separate neighboring synthetic centers.

    random_seed:
        Seed controlling reproducible random-number generation.

    Returns
    -------
    SyntheticDataset
        Generated points, labels, centers, and dataset metadata.
    """

    _validate_generation_parameters(
        num_records=num_records,
        num_features=num_features,
        num_clusters=num_clusters,
        cluster_spread=cluster_spread,
    )

    if not isfinite(center_separation):
        raise ValueError(
            "center_separation must be finite."
        )

    if center_separation <= 0:
        raise ValueError(
            "center_separation must be greater than zero."
        )

    random_generator = Random(random_seed)

    cluster_centers = _generate_cluster_centers(
        num_clusters=num_clusters,
        num_features=num_features,
        center_separation=center_separation,
    )

    base_cluster_size = num_records // num_clusters
    remainder = num_records % num_clusters

    points: list[Point] = []
    labels: list[int] = []

    for cluster_index, center in enumerate(cluster_centers):
        cluster_size = (
            base_cluster_size
            + (1 if cluster_index < remainder else 0)
        )

        for _ in range(cluster_size):
            point = tuple(
                random_generator.gauss(
                    coordinate,
                    cluster_spread,
                )
                for coordinate in center
            )

            points.append(point)
            labels.append(cluster_index)

    return SyntheticDataset(
        points=tuple(points),
        labels=tuple(labels),
        cluster_centers=cluster_centers,
        num_records=num_records,
        num_features=num_features,
        num_clusters=num_clusters,
    )
