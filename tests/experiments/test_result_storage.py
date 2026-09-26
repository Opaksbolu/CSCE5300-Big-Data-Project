import csv

import pytest

from src.experiments.experiment_result import ExperimentResult
from src.experiments.result_storage import (
    CSV_FIELDNAMES,
    _serialize_experiment_result,
    append_experiment_result,
)


def _make_result(
    *,
    experiment_id="run-1",
    num_records=1000,
):
    return ExperimentResult(
        experiment_id=experiment_id,
        dataset_name="synthetic",
        num_records=num_records,
        num_features=20,
        num_clusters=5,
        num_partitions=4,
        spark_master="local[4]",
        random_seed=42,
        local_max_iterations=100,
        global_max_iterations=100,
        tolerance=1e-6,
        materialized_record_count=num_records,
        converged=True,
        iterations=2,
        sse=19853.25,
        cluster_counts=(200, 200, 200, 200, 200),
        partition_clustering_seconds=0.18,
        center_aggregation_seconds=0.01,
        initialization_seconds=0.19,
        global_clustering_seconds=0.27,
        final_evaluation_seconds=0.08,
        total_runtime_seconds=0.54,
    )


def test_serialize_experiment_result_preserves_fields():
    result = _make_result()

    row = _serialize_experiment_result(result)

    assert tuple(row.keys()) == CSV_FIELDNAMES
    assert row["experiment_id"] == "run-1"
    assert row["dataset_name"] == "synthetic"
    assert row["num_records"] == 1000
    assert row["cluster_counts"] == "200;200;200;200;200"
    assert row["sse"] == 19853.25


def test_append_experiment_result_creates_csv(tmp_path):
    output_path = tmp_path / "results.csv"

    returned_path = append_experiment_result(
        _make_result(),
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "run-1"
    assert rows[0]["num_records"] == "1000"
    assert rows[0]["cluster_counts"] == "200;200;200;200;200"


def test_append_experiment_result_writes_header_once(tmp_path):
    output_path = tmp_path / "results.csv"

    append_experiment_result(
        _make_result(experiment_id="run-1"),
        output_path,
    )

    append_experiment_result(
        _make_result(
            experiment_id="run-2",
            num_records=2000,
        ),
        output_path,
    )

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "run-1"
    assert rows[1]["experiment_id"] == "run-2"
    assert rows[1]["num_records"] == "2000"

    lines = output_path.read_text(
        encoding="utf-8",
    ).splitlines()

    header = ",".join(CSV_FIELDNAMES)

    assert lines.count(header) == 1


def test_append_experiment_result_creates_parent_directories(
    tmp_path,
):
    output_path = (
        tmp_path
        / "nested"
        / "benchmark"
        / "results.csv"
    )

    append_experiment_result(
        _make_result(),
        output_path,
    )

    assert output_path.exists()


def test_append_experiment_result_writes_header_to_empty_file(
    tmp_path,
):
    output_path = tmp_path / "results.csv"
    output_path.touch()

    append_experiment_result(
        _make_result(),
        output_path,
    )

    with output_path.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "run-1"


def test_append_experiment_result_rejects_directory_path(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="output_path must reference a file",
    ):
        append_experiment_result(
            _make_result(),
            tmp_path,
        )
