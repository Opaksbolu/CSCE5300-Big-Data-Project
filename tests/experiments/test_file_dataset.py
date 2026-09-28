"""
Tests for file-backed Spark dataset ingestion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.file_dataset import (
    FileDataset,
    _parse_point,
    create_persisted_file_rdd,
)


def _write_dataset(
    path: Path,
    points: list[tuple[float, ...]],
) -> None:
    """Write points using the experimental CSV representation."""

    with path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(
                ",".join(
                    repr(value)
                    for value in point
                )
            )
            handle.write("\n")


def test_parse_point() -> None:
    """A CSV feature row should become a numeric point tuple."""

    assert _parse_point("1.0,2.5,-3.0") == (
        1.0,
        2.5,
        -3.0,
    )


def test_create_persisted_file_rdd_materializes_dataset(
    spark,
    tmp_path,
) -> None:
    """Spark should load and materialize the complete file dataset."""

    points = [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
    ]

    dataset_path = tmp_path / "points.csv"
    _write_dataset(dataset_path, points)

    dataset = FileDataset(
        path=dataset_path,
        num_records=4,
        num_features=2,
        num_clusters=2,
    )

    rdd, record_count = create_persisted_file_rdd(
        spark,
        dataset,
        num_partitions=2,
    )

    try:
        assert record_count == 4
        assert rdd.getNumPartitions() >= 2
        assert rdd.is_cached
        assert rdd.collect() == points
    finally:
        rdd.unpersist()


def test_create_persisted_file_rdd_rejects_invalid_partition_count(
    spark,
    tmp_path,
) -> None:
    """File-backed benchmarks require at least one partition."""

    dataset_path = tmp_path / "points.csv"
    _write_dataset(
        dataset_path,
        [(1.0, 2.0)],
    )

    dataset = FileDataset(
        path=dataset_path,
        num_records=1,
        num_features=2,
        num_clusters=1,
    )

    with pytest.raises(
        ValueError,
        match="num_partitions must be greater than zero",
    ):
        create_persisted_file_rdd(
            spark,
            dataset,
            num_partitions=0,
        )


def test_create_persisted_file_rdd_validates_record_count(
    spark,
    tmp_path,
) -> None:
    """Materialized input must agree with dataset metadata."""

    dataset_path = tmp_path / "points.csv"
    _write_dataset(
        dataset_path,
        [
            (1.0, 2.0),
            (3.0, 4.0),
        ],
    )

    dataset = FileDataset(
        path=dataset_path,
        num_records=3,
        num_features=2,
        num_clusters=2,
    )

    with pytest.raises(
        ValueError,
        match="Materialized record count does not match",
    ):
        create_persisted_file_rdd(
            spark,
            dataset,
            num_partitions=2,
        )


def test_file_dataset_preserves_numeric_content(
    spark,
    tmp_path,
) -> None:
    """File ingestion should preserve feature values exactly."""

    points = [
        (-0.14409032957792836, 0.8270963996684807),
        (41.002879956553016, 42.119793201987214),
    ]

    dataset_path = tmp_path / "points.csv"
    _write_dataset(dataset_path, points)

    dataset = FileDataset(
        path=dataset_path,
        num_records=2,
        num_features=2,
        num_clusters=2,
    )

    rdd, _ = create_persisted_file_rdd(
        spark,
        dataset,
        num_partitions=2,
    )

    try:
        assert rdd.collect() == points
    finally:
        rdd.unpersist()
