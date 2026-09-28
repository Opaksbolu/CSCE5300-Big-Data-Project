"""
File-backed dataset support for Spark clustering experiments.

Large experimental datasets should be loaded by Spark from storage
rather than first materialized as a complete Python collection on the
driver. This module provides the file-backed input boundary while
keeping dataset preparation separate from clustering measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspark import RDD
from pyspark.sql import SparkSession

from src.parallel.local_kmeans import Point


@dataclass(frozen=True)
class FileDataset:
    """Metadata describing a file-backed clustering dataset."""

    path: Path
    num_records: int
    num_features: int
    num_clusters: int


def _parse_point(line: str) -> Point:
    """Parse one comma-separated feature vector."""

    return tuple(
        float(value)
        for value in line.split(",")
    )


def _ensure_exact_partition_count(
    rdd: RDD,
    *,
    num_partitions: int,
) -> RDD:
    """Return an RDD with exactly the requested partition count."""

    current_partitions = rdd.getNumPartitions()

    if current_partitions > num_partitions:
        return rdd.coalesce(num_partitions)

    if current_partitions < num_partitions:
        return rdd.repartition(num_partitions)

    return rdd


def create_persisted_file_rdd(
    spark: SparkSession,
    dataset: FileDataset,
    *,
    num_partitions: int,
) -> tuple[RDD, int]:
    """
    Load, persist, and materialize a file-backed point RDD.

    File ingestion and materialization occur before the clustering
    pipeline begins so that input preparation is not included in
    clustering measurements.
    """

    if num_partitions <= 0:
        raise ValueError(
            "num_partitions must be greater than zero."
        )

    rdd = (
        spark.sparkContext.textFile(
            str(dataset.path),
            minPartitions=num_partitions,
        )
        .map(_parse_point)
    )

    rdd = _ensure_exact_partition_count(
        rdd,
        num_partitions=num_partitions,
    )
    rdd.persist()

    try:
        materialized_record_count = rdd.count()
    except Exception:
        rdd.unpersist()
        raise

    if materialized_record_count != dataset.num_records:
        rdd.unpersist()

        raise ValueError(
            "Materialized record count does not match "
            "dataset metadata: "
            f"expected {dataset.num_records}, "
            f"received {materialized_record_count}."
        )

    return rdd, materialized_record_count
