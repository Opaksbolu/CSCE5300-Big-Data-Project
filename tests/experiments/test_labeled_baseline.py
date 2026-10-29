"""
Tests for the labeled traditional K-Means baseline.

These tests verify deterministic clustering, complete assignments,
label preservation, initialization selection, and dataset support.
"""

from pathlib import Path

import math
import pytest

from src.experiments.dataset_bridge import (
    LabeledDataset,
    load_labeled_csv,
)
from src.experiments.labeled_baseline import (
    run_labeled_baseline,
)
from src.parallel.local_kmeans import fit_local_kmeans


def make_dataset():
    """Construct a small deterministic labeled dataset."""

    return LabeledDataset(
        name="unit_test",
        points=(
            (0.0, 0.0),
            (0.1, 0.2),
            (0.2, 0.1),
            (10.0, 10.0),
            (10.1, 10.2),
            (10.2, 10.1),
        ),
        labels=(0, 0, 0, 1, 1, 1),
        record_ids=(0, 1, 2, 3, 4, 5),
    )


def run_test_baseline(
    dataset=None,
    *,
    num_clusters=2,
    random_seed=42,
    num_initializations=3,
):
    """Run a small baseline with consistent test settings."""

    if dataset is None:
        dataset = make_dataset()

    return run_labeled_baseline(
        dataset,
        num_clusters=num_clusters,
        random_seed=random_seed,
        max_iterations=100,
        tolerance=1e-6,
        num_initializations=num_initializations,
    )


def test_baseline_returns_correct_metadata():

    dataset = make_dataset()

    result = run_test_baseline(dataset)

    assert result.dataset_name == dataset.name
    assert result.num_records == dataset.num_records
    assert result.num_features == dataset.num_features
    assert result.num_clusters == 2
    assert result.num_initializations == 3


def test_every_record_receives_valid_assignment():

    dataset = make_dataset()

    result = run_test_baseline(dataset)

    assert len(result.predicted_clusters) == dataset.num_records

    assert all(
        0 <= cluster < result.num_clusters
        for cluster in result.predicted_clusters
    )

    assert sum(result.cluster_counts) == dataset.num_records


def test_ground_truth_labels_are_preserved():

    dataset = make_dataset()

    result = run_test_baseline(dataset)

    assert result.true_labels == dataset.labels
    assert result.record_ids == dataset.record_ids

    assert len(result.true_labels) == dataset.num_records
    assert len(result.record_ids) == dataset.num_records


def test_sse_and_runtime_are_valid():

    result = run_test_baseline()

    assert math.isfinite(result.sse)
    assert result.sse >= 0

    assert math.isfinite(result.total_runtime_seconds)
    assert result.total_runtime_seconds >= 0


def test_repeated_runs_are_reproducible():

    first = run_test_baseline()
    second = run_test_baseline()

    assert first.sse == second.sse

    assert first.predicted_clusters == second.predicted_clusters

    assert first.cluster_counts == second.cluster_counts

    assert (
        first.best_initialization_seed
        == second.best_initialization_seed
    )


def test_best_initialization_has_lowest_sse():

    dataset = make_dataset()

    result = run_test_baseline(
        dataset,
        random_seed=42,
        num_initializations=3,
    )

    independent_results = []

    for seed in (42, 43, 44):

        independent = fit_local_kmeans(
            dataset.points,
            k=2,
            max_iterations=100,
            tolerance=1e-6,
            random_seed=seed,
        )

        independent_results.append(
            (seed, independent.sse)
        )

    expected_seed, expected_sse = min(
        independent_results,
        key=lambda item: item[1],
    )

    assert math.isclose(
        result.sse,
        expected_sse,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert result.best_initialization_seed == expected_seed


def test_single_initialization_matches_local_kmeans():

    dataset = make_dataset()

    result = run_test_baseline(
        dataset,
        random_seed=42,
        num_initializations=1,
    )

    expected = fit_local_kmeans(
        dataset.points,
        k=2,
        max_iterations=100,
        tolerance=1e-6,
        random_seed=42,
    )

    assert result.sse == expected.sse

    assert result.predicted_clusters == expected.assignments

    assert result.best_initialization_seed == 42


@pytest.mark.parametrize(
    "initializations",
    [0, -1, -5],
)
def test_invalid_initialization_counts_are_rejected(
    initializations,
):

    with pytest.raises(ValueError):

        run_test_baseline(
            num_initializations=initializations,
        )


@pytest.mark.parametrize(
    "sample_path,clusters,features",
    [
        (
            "data/samples/kdd99_sample.csv",
            4,
            41,
        ),
        (
            "data/samples/synthetic_1K_sample.csv",
            5,
            20,
        ),
    ],
)
def test_real_sample_datasets(
    sample_path,
    clusters,
    features,
):

    path = Path(sample_path)

    assert path.is_file()

    dataset = load_labeled_csv(path)

    result = run_test_baseline(
        dataset,
        num_clusters=clusters,
        num_initializations=2,
    )

    assert result.num_records == 1000
    assert result.num_features == features
    assert result.num_clusters == clusters

    assert len(result.predicted_clusters) == 1000

    assert sum(result.cluster_counts) == 1000

    assert result.true_labels == dataset.labels
    assert result.record_ids == dataset.record_ids

    assert math.isfinite(result.sse)
    assert result.sse >= 0
