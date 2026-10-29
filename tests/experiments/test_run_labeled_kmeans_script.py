"""
Regression tests for the labeled traditional K-Means CLI.

All generated CSV outputs are isolated in pytest temporary directories.
Existing project benchmark files are never modified.
"""

import csv
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts.run_labeled_kmeans import (
    METRIC_FIELDS,
    PREDICTION_FIELDS,
)
from src.experiments.dataset_bridge import load_labeled_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_labeled_kmeans.py"

SYNTHETIC = PROJECT_ROOT / "data/samples/synthetic_1K_sample.csv"
KDD99 = PROJECT_ROOT / "data/samples/kdd99_sample.csv"


def run_cli(
    tmp_path,
    *,
    dataset=SYNTHETIC,
    clusters=5,
    repetitions=2,
    warmups=1,
    initializations=2,
    seed=42,
    extra_args=(),
    metrics=None,
    predictions=None,
):
    """Run the CLI with outputs confined to a temporary directory."""

    if metrics is None:
        metrics = tmp_path / "metrics.csv"

    if predictions is None:
        predictions = tmp_path / "predictions.csv"

    command = [
        sys.executable,
        "-m",
        "scripts.run_labeled_kmeans",
        "--dataset",
        str(dataset),
        "--num-clusters",
        str(clusters),
        "--repetitions",
        str(repetitions),
        "--warmup-runs",
        str(warmups),
        "--num-initializations",
        str(initializations),
        "--random-seed",
        str(seed),
        "--output",
        str(metrics),
        "--predictions-output",
        str(predictions),
        *extra_args,
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    return result, metrics, predictions


def read_csv(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize(
    "dataset_path,clusters,features",
    [
        (SYNTHETIC, 5, 20),
        (KDD99, 4, 41),
    ],
)
def test_cli_writes_valid_csv_files(
    tmp_path,
    dataset_path,
    clusters,
    features,
):
    result, metrics_path, predictions_path = run_cli(
        tmp_path,
        dataset=dataset_path,
        clusters=clusters,
    )

    assert result.returncode == 0, result.stderr
    assert metrics_path.is_file()
    assert predictions_path.is_file()

    metric_fields, metrics = read_csv(metrics_path)
    prediction_fields, predictions = read_csv(predictions_path)

    assert metric_fields == METRIC_FIELDS
    assert prediction_fields == PREDICTION_FIELDS

    assert len(metrics) == 2
    assert len(predictions) == 2000

    source = load_labeled_csv(dataset_path)

    for index, metric in enumerate(metrics, start=1):
        assert metric["algorithm"] == "traditional-kmeans"
        assert metric["dataset_name"] == source.name
        assert int(metric["num_records"]) == 1000
        assert int(metric["num_features"]) == features
        assert int(metric["num_clusters"]) == clusters
        assert int(metric["repetition"]) == index
        assert int(metric["random_seed"]) == 41 + index
        assert int(metric["num_initializations"]) == 2

        assert math.isfinite(float(metric["sse"]))
        assert float(metric["sse"]) >= 0

        assert math.isfinite(
            float(metric["total_runtime_seconds"])
        )
        assert float(metric["total_runtime_seconds"]) >= 0

        counts = tuple(
            int(value)
            for value in metric["cluster_counts"].split(";")
        )

        assert len(counts) == clusters
        assert sum(counts) == 1000

        repetition_predictions = [
            row
            for row in predictions
            if int(row["repetition"]) == index
        ]

        assert len(repetition_predictions) == 1000

        actual_counts = Counter(
            int(row["predicted_cluster"])
            for row in repetition_predictions
        )

        assert counts == tuple(
            actual_counts[cluster]
            for cluster in range(clusters)
        )

        for expected_id, row in enumerate(
            repetition_predictions
        ):
            assert row["dataset_name"] == source.name
            assert int(row["record_id"]) == expected_id

            assert (
                int(row["true_label"])
                == source.labels[expected_id]
            )

            assert int(row["random_seed"]) == 41 + index

            assert (
                0 <= int(row["predicted_cluster"]) < clusters
            )


def test_warmups_are_not_saved(tmp_path):
    result, metrics_path, predictions_path = run_cli(
        tmp_path,
        repetitions=1,
        warmups=2,
        initializations=1,
    )

    assert result.returncode == 0, result.stderr

    _, metrics = read_csv(metrics_path)
    _, predictions = read_csv(predictions_path)

    assert len(metrics) == 1
    assert len(predictions) == 1000

    assert {
        int(row["repetition"])
        for row in predictions
    } == {1}


def test_repetition_seeds_are_sequential(tmp_path):
    result, metrics_path, predictions_path = run_cli(
        tmp_path,
        repetitions=3,
        warmups=0,
        initializations=1,
        seed=100,
    )

    assert result.returncode == 0, result.stderr

    _, metrics = read_csv(metrics_path)
    _, predictions = read_csv(predictions_path)

    assert [
        int(row["random_seed"])
        for row in metrics
    ] == [100, 101, 102]

    assert len(predictions) == 3000

    assert [
        int(row["best_initialization_seed"])
        for row in metrics
    ] == [100, 101, 102]


@pytest.mark.parametrize(
    "argument,value",
    [
        ("--num-clusters", "0"),
        ("--repetitions", "0"),
        ("--warmup-runs", "-1"),
        ("--num-initializations", "0"),
        ("--max-iterations", "0"),
        ("--tolerance", "-1"),
        ("--tolerance", "nan"),
        ("--tolerance", "inf"),
    ],
)
def test_invalid_arguments_are_rejected(
    tmp_path,
    argument,
    value,
):
    result, metrics_path, predictions_path = run_cli(
        tmp_path,
        repetitions=1,
        warmups=0,
        initializations=1,
        extra_args=(argument, value),
    )

    assert result.returncode != 0
    assert not metrics_path.exists()
    assert not predictions_path.exists()


def test_cluster_count_cannot_exceed_dataset(tmp_path):
    result, metrics_path, predictions_path = run_cli(
        tmp_path,
        clusters=1001,
        warmups=0,
    )

    assert result.returncode != 0
    assert not metrics_path.exists()
    assert not predictions_path.exists()


@pytest.mark.parametrize(
    "existing_target",
    ["metrics", "predictions"],
)
def test_existing_outputs_are_never_overwritten(
    tmp_path,
    existing_target,
):
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"

    protected = (
        metrics_path
        if existing_target == "metrics"
        else predictions_path
    )

    original_content = "DO NOT OVERWRITE"
    protected.write_text(original_content)

    result, _, _ = run_cli(
        tmp_path,
        repetitions=1,
        warmups=0,
        metrics=metrics_path,
        predictions=predictions_path,
    )

    assert result.returncode != 0
    assert protected.read_text() == original_content

    other = (
        predictions_path
        if existing_target == "metrics"
        else metrics_path
    )

    assert not other.exists()


def test_metrics_and_predictions_cannot_share_path(tmp_path):
    shared_path = tmp_path / "shared.csv"

    result, _, _ = run_cli(
        tmp_path,
        repetitions=1,
        warmups=0,
        metrics=shared_path,
        predictions=shared_path,
    )

    assert result.returncode != 0
    assert not shared_path.exists()


def test_missing_dataset_creates_no_outputs(tmp_path):
    missing = tmp_path / "missing.csv"

    result, metrics_path, predictions_path = run_cli(
        tmp_path,
        dataset=missing,
        repetitions=1,
        warmups=0,
    )

    assert result.returncode != 0
    assert not metrics_path.exists()
    assert not predictions_path.exists()
