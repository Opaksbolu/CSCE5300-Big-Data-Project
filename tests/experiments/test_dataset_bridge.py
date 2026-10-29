"""
Tests for the labeled dataset bridge.

These tests verify dataset validation, feature-label separation,
record alignment, and safe Spark-compatible CSV preparation.
"""

import csv
import math

import pytest

from src.experiments.dataset_bridge import (
    load_labeled_csv,
    prepare_spark_csv,
)


def write_dataset(path, rows):
    """Create a temporary labeled CSV for testing."""

    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return path


def test_load_valid_dataset(tmp_path):
    path = write_dataset(
        tmp_path / "sample.csv",
        [
            ["f0", "f1", "label"],
            [1.0, 2.0, 0],
            [3.0, 4.0, 1],
            [5.0, 6.0, 0],
        ],
    )

    dataset = load_labeled_csv(path)

    assert dataset.name == "sample"
    assert dataset.num_records == 3
    assert dataset.num_features == 2
    assert dataset.num_classes == 2
    assert dataset.points == (
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
    )
    assert dataset.labels == (0, 1, 0)
    assert dataset.record_ids == (0, 1, 2)


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labeled_csv(tmp_path / "missing.csv")


def test_missing_label_column(tmp_path):
    path = write_dataset(
        tmp_path / "missing_label.csv",
        [
            ["f0", "f1"],
            [1.0, 2.0],
        ],
    )

    with pytest.raises(ValueError, match="label column"):
        load_labeled_csv(path)


def test_duplicate_columns(tmp_path):
    path = write_dataset(
        tmp_path / "duplicate.csv",
        [
            ["f0", "f0", "label"],
            [1.0, 2.0, 0],
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_labeled_csv(path)


def test_record_id_column_rejected(tmp_path):
    path = write_dataset(
        tmp_path / "record_id.csv",
        [
            ["record_id", "f0", "label"],
            [10, 1.0, 0],
        ],
    )

    with pytest.raises(ValueError, match="record_id"):
        load_labeled_csv(path)


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-inf"],
)
def test_nonfinite_features_rejected(tmp_path, value):
    path = write_dataset(
        tmp_path / "nonfinite.csv",
        [
            ["f0", "label"],
            [value, 0],
        ],
    )

    with pytest.raises(ValueError, match="Non-finite"):
        load_labeled_csv(path)


def test_nonnumeric_feature_rejected(tmp_path):
    path = write_dataset(
        tmp_path / "invalid.csv",
        [
            ["f0", "label"],
            ["invalid", 0],
        ],
    )

    with pytest.raises(ValueError, match="Invalid feature"):
        load_labeled_csv(path)


def test_invalid_label_rejected(tmp_path):
    path = write_dataset(
        tmp_path / "invalid_label.csv",
        [
            ["f0", "label"],
            [1.0, "unknown"],
        ],
    )

    with pytest.raises(ValueError, match="Invalid label"):
        load_labeled_csv(path)


def test_empty_dataset_rejected(tmp_path):
    path = write_dataset(
        tmp_path / "empty.csv",
        [
            ["f0", "label"],
        ],
    )

    with pytest.raises(ValueError, match="empty"):
        load_labeled_csv(path)


def test_malformed_row_rejected(tmp_path):
    path = write_dataset(
        tmp_path / "malformed.csv",
        [
            ["f0", "f1", "label"],
            [1.0, 2.0],
        ],
    )

    with pytest.raises(ValueError, match="Malformed"):
        load_labeled_csv(path)


def test_export_preserves_features_and_labels(tmp_path):
    path = write_dataset(
        tmp_path / "sample.csv",
        [
            ["f0", "f1", "label"],
            [1.5, 2.5, 0],
            [3.5, 4.5, 1],
            [5.5, 6.5, 0],
        ],
    )

    dataset = load_labeled_csv(path)

    prepared = prepare_spark_csv(
        dataset,
        output_directory=tmp_path / "prepared",
        num_clusters=2,
    )

    assert prepared.dataset.num_records == 3
    assert prepared.dataset.num_features == 2
    assert prepared.dataset.num_clusters == 2

    with prepared.dataset.path.open(newline="") as handle:
        feature_rows = list(csv.reader(handle))

    with prepared.labels_path.open(newline="") as handle:
        label_rows = list(csv.DictReader(handle))

    assert feature_rows == [
        ["1.5", "2.5"],
        ["3.5", "4.5"],
        ["5.5", "6.5"],
    ]

    assert label_rows == [
        {"record_id": "0", "label": "0"},
        {"record_id": "1", "label": "1"},
        {"record_id": "2", "label": "0"},
    ]

    assert all(
        math.isfinite(float(value))
        for row in feature_rows
        for value in row
    )


def test_export_rejects_invalid_cluster_count(tmp_path):
    path = write_dataset(
        tmp_path / "sample.csv",
        [
            ["f0", "label"],
            [1.0, 0],
            [2.0, 1],
        ],
    )

    dataset = load_labeled_csv(path)

    for clusters in (0, -1, 3):
        with pytest.raises(ValueError):
            prepare_spark_csv(
                dataset,
                output_directory=tmp_path / "prepared",
                num_clusters=clusters,
            )


def test_export_never_overwrites_existing_files(tmp_path):
    path = write_dataset(
        tmp_path / "sample.csv",
        [
            ["f0", "label"],
            [1.0, 0],
            [2.0, 1],
        ],
    )

    dataset = load_labeled_csv(path)

    prepared = prepare_spark_csv(
        dataset,
        output_directory=tmp_path / "prepared",
        num_clusters=2,
    )

    original_features = prepared.dataset.path.read_bytes()
    original_labels = prepared.labels_path.read_bytes()

    with pytest.raises(FileExistsError):
        prepare_spark_csv(
            dataset,
            output_directory=tmp_path / "prepared",
            num_clusters=2,
        )

    assert prepared.dataset.path.read_bytes() == original_features
    assert prepared.labels_path.read_bytes() == original_labels
