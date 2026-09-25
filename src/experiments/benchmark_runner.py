"""
Controlled Spark benchmark execution for Parallel K-Means experiments.

This module owns the experimental lifecycle of a Spark RDD:

1. create the RDD,
2. persist it,
3. materialize it before clustering,
4. execute Parallel K-Means,
5. collect experiment metadata,
6. unpersist the RDD.

Keeping persistence in the experiment layer prevents the clustering
algorithm from taking ownership of an RDD supplied by its caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark import RDD
from pyspark.sql import SparkSession

from src.experiments.synthetic_data import SyntheticDataset
from src.parallel.parallel_kmeans import (
    ParallelKMeansResult,
    fit_parallel_kmeans,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Result and metadata from one controlled benchmark execution."""

    num_records: int
    num_features: int
    num_clusters: int
    num_partitions: int
    spark_master: str
    random_seed: int
    materialized_record_count: int
    clustering_result: ParallelKMeansResult


def _create_persisted_rdd(
    spark: SparkSession,
    dataset: SyntheticDataset,
    *,
    num_partitions: int,
) -> tuple[RDD, int]:
    """
    Create, persist, and materialize an RDD for benchmarking.

    Materialization occurs before Parallel K-Means begins so that
    initial RDD computation is not mixed into clustering measurements.
    """

    if num_partitions <= 0:
        raise ValueError(
            "num_partitions must be greater than zero."
        )

    rdd = spark.sparkContext.parallelize(
        dataset.points,
        num_partitions,
    )

    rdd.persist()

    try:
        materialized_record_count = rdd.count()
    except Exception:
        rdd.unpersist()
        raise

    return rdd, materialized_record_count


def run_parallel_kmeans_benchmark(
    spark: SparkSession,
    dataset: SyntheticDataset,
    *,
    num_partitions: int,
    random_seed: int = 42,
    local_max_iterations: int = 100,
    global_max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> BenchmarkResult:
    """
    Execute one controlled Parallel K-Means benchmark.

    The input RDD is persisted and fully materialized before the
    clustering pipeline runs. The RDD is always unpersisted afterward,
    including when clustering raises an exception.
    """

    rdd, materialized_record_count = (
        _create_persisted_rdd(
            spark,
            dataset,
            num_partitions=num_partitions,
        )
    )

    try:
        clustering_result = fit_parallel_kmeans(
            rdd,
            k=dataset.num_clusters,
            local_max_iterations=local_max_iterations,
            global_max_iterations=global_max_iterations,
            tolerance=tolerance,
            random_seed=random_seed,
        )

        return BenchmarkResult(
            num_records=dataset.num_records,
            num_features=dataset.num_features,
            num_clusters=dataset.num_clusters,
            num_partitions=rdd.getNumPartitions(),
            spark_master=spark.sparkContext.master,
            random_seed=random_seed,
            materialized_record_count=materialized_record_count,
            clustering_result=clustering_result,
        )

    finally:
        rdd.unpersist()
