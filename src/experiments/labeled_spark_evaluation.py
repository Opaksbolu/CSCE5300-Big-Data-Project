"""
Evaluate final Spark K-Means centers on small labeled datasets.

Record-level assignments are generated using the original dataset
order, never the potentially shuffled Spark RDD order.

This module does not run Spark or modify cluster centers.
Evaluation is performed outside the measured clustering runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.experiments.dataset_bridge import LabeledDataset
from src.parallel.local_kmeans import (
    Point,
    calculate_sse,
    nearest_center,
)


@dataclass(frozen=True)
class LabeledSparkEvaluation:
    """Aligned predictions and independently calculated metrics."""

    dataset_name: str
    num_records: int
    num_features: int
    num_clusters: int
    record_ids: tuple[int, ...]
    true_labels: tuple[int, ...]
    predicted_clusters: tuple[int, ...]
    cluster_counts: tuple[int, ...]
    sse: float


def evaluate_labeled_spark_centers(
    dataset: LabeledDataset,
    centers: Sequence[Point],
) -> LabeledSparkEvaluation:
    """
    Evaluate final Spark cluster centers against original records.

    This function is intended for small in-memory datasets.
    Ground-truth labels are never used to choose assignments.
    """

    if not centers:
        raise ValueError(
            "At least one cluster center is required."
        )

    if dataset.num_records == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not (
        len(dataset.points)
        == len(dataset.labels)
        == len(dataset.record_ids)
    ):
        raise ValueError(
            "Dataset features, labels, and IDs must align."
        )

    if len(set(dataset.record_ids)) != dataset.num_records:
        raise ValueError(
            "Record IDs must be unique."
        )

    num_clusters = len(centers)

    if num_clusters > dataset.num_records:
        raise ValueError(
            "Cluster count cannot exceed record count."
        )

    for center in centers:
        if len(center) != dataset.num_features:
            raise ValueError(
                "Cluster center dimensionality does not match dataset."
            )

        if not all(math.isfinite(value) for value in center):
            raise ValueError(
                "Cluster centers must contain finite values."
            )

    predicted_clusters = tuple(
        nearest_center(point, centers)
        for point in dataset.points
    )

    cluster_counts = tuple(
        predicted_clusters.count(cluster_index)
        for cluster_index in range(num_clusters)
    )

    if sum(cluster_counts) != dataset.num_records:
        raise RuntimeError(
            "Cluster counts do not match record count."
        )

    sse = calculate_sse(
        dataset.points,
        predicted_clusters,
        centers,
    )

    if not math.isfinite(sse) or sse < 0:
        raise RuntimeError(
            "Calculated SSE is invalid."
        )

    return LabeledSparkEvaluation(
        dataset_name=dataset.name,
        num_records=dataset.num_records,
        num_features=dataset.num_features,
        num_clusters=num_clusters,
        record_ids=dataset.record_ids,
        true_labels=dataset.labels,
        predicted_clusters=predicted_clusters,
        cluster_counts=cluster_counts,
        sse=sse,
    )
