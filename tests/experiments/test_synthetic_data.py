"""
Tests for deterministic synthetic clustering dataset generation.
"""

from collections import Counter
from math import inf, nan

import pytest

from src.experiments.synthetic_data import (
    generate_synthetic_dataset,
)


def test_generates_requested_dataset_shape():
    dataset = generate_synthetic_dataset(
        num_records=1000,
        num_features=20,
        num_clusters=5,
        random_seed=42,
    )

    assert len(dataset.points) == 1000
    assert len(dataset.labels) == 1000
    assert dataset.num_records == 1000
    assert dataset.num_features == 20
    assert dataset.num_clusters == 5

    assert all(
        len(point) == 20
        for point in dataset.points
    )

    assert len(dataset.cluster_centers) == 5

    assert all(
        len(center) == 20
        for center in dataset.cluster_centers
    )


def test_clusters_are_balanced_when_records_divide_evenly():
    dataset = generate_synthetic_dataset(
        num_records=1000,
        num_features=4,
        num_clusters=5,
        random_seed=42,
    )

    label_counts = Counter(dataset.labels)

    assert label_counts == {
        0: 200,
        1: 200,
        2: 200,
        3: 200,
        4: 200,
    }


def test_remainder_records_are_distributed_deterministically():
    dataset = generate_synthetic_dataset(
        num_records=11,
        num_features=2,
        num_clusters=3,
        random_seed=42,
    )

    label_counts = Counter(dataset.labels)

    assert label_counts == {
        0: 4,
        1: 4,
        2: 3,
    }


def test_same_seed_produces_identical_dataset():
    first = generate_synthetic_dataset(
        num_records=100,
        num_features=3,
        num_clusters=4,
        random_seed=123,
    )

    second = generate_synthetic_dataset(
        num_records=100,
        num_features=3,
        num_clusters=4,
        random_seed=123,
    )

    assert first == second


def test_different_seeds_produce_different_points():
    first = generate_synthetic_dataset(
        num_records=100,
        num_features=3,
        num_clusters=4,
        random_seed=123,
    )

    second = generate_synthetic_dataset(
        num_records=100,
        num_features=3,
        num_clusters=4,
        random_seed=456,
    )

    assert first.points != second.points

    assert first.cluster_centers == second.cluster_centers


def test_zero_spread_places_points_exactly_on_centers():
    dataset = generate_synthetic_dataset(
        num_records=12,
        num_features=3,
        num_clusters=3,
        cluster_spread=0.0,
        random_seed=42,
    )

    for point, label in zip(
        dataset.points,
        dataset.labels,
    ):
        assert point == dataset.cluster_centers[label]


@pytest.mark.parametrize(
    "num_records",
    [0, -1],
)
def test_rejects_nonpositive_record_count(num_records):
    with pytest.raises(
        ValueError,
        match="num_records must be greater than zero",
    ):
        generate_synthetic_dataset(
            num_records=num_records,
            num_features=2,
            num_clusters=2,
        )


@pytest.mark.parametrize(
    "num_features",
    [0, -1],
)
def test_rejects_nonpositive_feature_count(num_features):
    with pytest.raises(
        ValueError,
        match="num_features must be greater than zero",
    ):
        generate_synthetic_dataset(
            num_records=10,
            num_features=num_features,
            num_clusters=2,
        )


@pytest.mark.parametrize(
    "num_clusters",
    [0, -1],
)
def test_rejects_nonpositive_cluster_count(num_clusters):
    with pytest.raises(
        ValueError,
        match="num_clusters must be greater than zero",
    ):
        generate_synthetic_dataset(
            num_records=10,
            num_features=2,
            num_clusters=num_clusters,
        )


def test_rejects_more_clusters_than_records():
    with pytest.raises(
        ValueError,
        match="num_clusters cannot exceed num_records",
    ):
        generate_synthetic_dataset(
            num_records=3,
            num_features=2,
            num_clusters=4,
        )


def test_rejects_negative_cluster_spread():
    with pytest.raises(
        ValueError,
        match="cluster_spread cannot be negative",
    ):
        generate_synthetic_dataset(
            num_records=10,
            num_features=2,
            num_clusters=2,
            cluster_spread=-0.1,
        )


@pytest.mark.parametrize(
    "cluster_spread",
    [nan, inf, -inf],
)
def test_rejects_nonfinite_cluster_spread(cluster_spread):
    with pytest.raises(
        ValueError,
        match="cluster_spread must be finite",
    ):
        generate_synthetic_dataset(
            num_records=10,
            num_features=2,
            num_clusters=2,
            cluster_spread=cluster_spread,
        )


def test_rejects_nonpositive_center_separation():
    with pytest.raises(
        ValueError,
        match="center_separation must be greater than zero",
    ):
        generate_synthetic_dataset(
            num_records=10,
            num_features=2,
            num_clusters=2,
            center_separation=0.0,
        )


@pytest.mark.parametrize(
    "center_separation",
    [nan, inf, -inf],
)
def test_rejects_nonfinite_center_separation(
    center_separation,
):
    with pytest.raises(
        ValueError,
        match="center_separation must be finite",
    ):
        generate_synthetic_dataset(
            num_records=10,
            num_features=2,
            num_clusters=2,
            center_separation=center_separation,
        )
