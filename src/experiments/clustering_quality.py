"""Independent clustering-quality evaluation.

Ground-truth labels are used only after clustering.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


@dataclass(frozen=True)
class ClusteringQuality:
    adjusted_rand_index: float
    normalized_mutual_information: float
    purity: float
    num_records: int
    num_true_classes: int
    num_predicted_clusters: int


def evaluate_clustering_quality(true_labels, predicted_clusters):
    """Evaluate clustering independently of cluster-ID numbering."""

    true_labels = tuple(true_labels)
    predicted_clusters = tuple(predicted_clusters)

    if not true_labels:
        raise ValueError("Labels cannot be empty.")

    if len(true_labels) != len(predicted_clusters):
        raise ValueError("Label lengths must match.")

    for label in (*true_labels, *predicted_clusters):
        if not isinstance(label, (int, str)):
            raise ValueError("Labels must be integers or strings.")

    groups = defaultdict(list)

    for actual, predicted in zip(
        true_labels,
        predicted_clusters,
    ):
        groups[predicted].append(actual)

    dominant_total = sum(
        max(Counter(values).values())
        for values in groups.values()
    )

    purity = dominant_total / len(true_labels)

    ari = float(
        adjusted_rand_score(
            true_labels,
            predicted_clusters,
        )
    )

    nmi = float(
        normalized_mutual_info_score(
            true_labels,
            predicted_clusters,
        )
    )

    if not all(
        math.isfinite(value)
        for value in (ari, nmi, purity)
    ):
        raise RuntimeError("Non-finite quality metric.")

    return ClusteringQuality(
        adjusted_rand_index=ari,
        normalized_mutual_information=nmi,
        purity=purity,
        num_records=len(true_labels),
        num_true_classes=len(set(true_labels)),
        num_predicted_clusters=len(set(predicted_clusters)),
    )
