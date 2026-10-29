"""
Regression tests for labeled Spark K-Means evaluation.

These tests verify assignment correctness, label alignment,
cluster counts, SSE, determinism, and input validation.

No Spark session is required.
"""

from math import isfinite

import pytest

from src.experiments.dataset_bridge import (
    LabeledDataset,
    load_labeled_csv,
)
from src.experiments.labeled_spark_evaluation import (
    evaluate_labeled_spark_centers,
)


@pytest.fixture
def known_dataset():
    return LabeledDataset(
        name="known_example",
        points=(
            (0.0, 0.0),
            (1.0, 0.0),
            (10.0, 10.0),
            (11.0, 10.0),
        ),
        labels=(5, 5, 9, 9),
        record_ids=(100, 101, 102, 103),
    )


@pytest.fixture
def known_centers():
    return (
        (0.0, 0.0),
        (10.0, 10.0),
    )


def test_correct_assignments(known_dataset, known_centers):
    result = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    assert result.predicted_clusters == (0, 0, 1, 1)


def test_correct_cluster_counts(known_dataset, known_centers):
    result = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    assert result.cluster_counts == (2, 2)
    assert sum(result.cluster_counts) == 4


def test_correct_sse(known_dataset, known_centers):
    result = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    assert result.sse == pytest.approx(2.0)


def test_record_ids_preserved(known_dataset, known_centers):
    result = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    assert result.record_ids == (100, 101, 102, 103)


def test_ground_truth_labels_preserved(
    known_dataset,
    known_centers,
):
    result = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    assert result.true_labels == (5, 5, 9, 9)


def test_deterministic_results(known_dataset, known_centers):
    first = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    second = evaluate_labeled_spark_centers(
        known_dataset,
        known_centers,
    )

    assert first == second


def test_rejects_invalid_dimensions(known_dataset):
    with pytest.raises(ValueError):
        evaluate_labeled_spark_centers(
            known_dataset,
            ((0.0,), (10.0,)),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_nonfinite_centers(
    known_dataset,
    invalid_value,
):
    with pytest.raises(ValueError):
        evaluate_labeled_spark_centers(
            known_dataset,
            (
                (invalid_value, 0.0),
                (10.0, 10.0),
            ),
        )


def test_rejects_empty_centers(known_dataset):
    with pytest.raises(ValueError):
        evaluate_labeled_spark_centers(
            known_dataset,
            (),
        )


def test_nearest_center_tie_breaking():
    dataset = LabeledDataset(
        name="tie_example",
        points=(
            (1.0, 0.0),
            (2.0, 0.0),
        ),
        labels=(7, 9),
        record_ids=(55, 56),
    )

    result = evaluate_labeled_spark_centers(
        dataset,
        (
            (0.0, 0.0),
            (2.0, 0.0),
        ),
    )

    assert result.predicted_clusters == (0, 1)
    assert result.cluster_counts == (1, 1)
    assert result.record_ids == (55, 56)
    assert result.true_labels == (7, 9)


@pytest.mark.parametrize(
    "filename,k,features",
    [
        ("data/samples/synthetic_1K_sample.csv", 5, 20),
        ("data/samples/kdd99_sample.csv", 4, 41),
    ],
)
def test_real_sample_alignment(filename, k, features):
    dataset = load_labeled_csv(filename)

    result = evaluate_labeled_spark_centers(
        dataset,
        dataset.points[:k],
    )

    assert result.num_records == 1000
    assert result.num_features == features
    assert result.num_clusters == k

    assert result.record_ids == dataset.record_ids
    assert result.true_labels == dataset.labels

    assert len(result.predicted_clusters) == 1000
    assert sum(result.cluster_counts) == 1000

    assert all(
        0 <= cluster < k
        for cluster in result.predicted_clusters
    )

    assert isfinite(result.sse)
    assert result.sse >= 0
