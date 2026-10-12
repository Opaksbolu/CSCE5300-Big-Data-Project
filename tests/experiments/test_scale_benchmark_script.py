"""Regression tests for the scale benchmark launcher."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts import scale_benchmark


def test_run_trial_forwards_both_tolerances(monkeypatch):
    """Verify that the benchmark uses the current algorithm API."""

    spark = MagicMock()
    rdd = MagicMock()

    spark.sparkContext.parallelize.return_value = rdd

    result = SimpleNamespace(
        timing=SimpleNamespace(
            partition_clustering_seconds=0.1,
            center_aggregation_seconds=0.2,
            initialization_seconds=0.3,
            global_clustering_seconds=0.4,
            total_runtime_seconds=0.7,
        ),
        iterations=3,
        converged=True,
        sse=12.5,
        cluster_counts=(3, 3),
    )

    fit_mock = MagicMock(return_value=result)

    monkeypatch.setattr(
        scale_benchmark,
        "create_spark_session",
        lambda **kwargs: spark,
    )

    monkeypatch.setattr(
        scale_benchmark,
        "fit_parallel_kmeans",
        fit_mock,
    )

    row = scale_benchmark.run_trial(
        [(0.0, 0.0), (1.0, 1.0)],
        k=2,
        spark_master="local[1]",
        num_partitions=1,
        random_seed=42,
        trial=1,
        tolerance=0.0001,
    )

    fit_mock.assert_called_once_with(
        rdd,
        k=2,
        local_tolerance=0.0001,
        global_cost_tolerance=0.0001,
        random_seed=42,
    )

    assert row["tolerance"] == 0.0001
    assert row["sse"] == 12.5
    assert row["converged"] is True

    spark.stop.assert_called_once()
