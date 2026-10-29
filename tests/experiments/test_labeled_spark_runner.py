
"""Regression tests for labeled Spark benchmark integration."""

from types import SimpleNamespace

import pytest

from src.experiments.dataset_bridge import LabeledDataset
from src.experiments import labeled_spark_runner as runner


@pytest.fixture
def dataset():
    return LabeledDataset(
        name="runner_test",
        points=(
            (0.0, 0.0),
            (0.0, 1.0),
            (10.0, 10.0),
            (10.0, 11.0),
        ),
        labels=(0, 0, 1, 1),
        record_ids=(0, 1, 2, 3),
    )


@pytest.fixture
def fake_benchmark(monkeypatch):
    """Replace Spark execution with a deterministic benchmark."""

    calls = []

    def execute(spark, prepared_dataset, **kwargs):
        calls.append(
            (spark, prepared_dataset, kwargs)
        )

        return SimpleNamespace(
            num_records=4,
            num_features=2,
            num_clusters=2,
            clustering_result=SimpleNamespace(
                centers=(
                    (0.0, 0.5),
                    (10.0, 10.5),
                ),
                cluster_counts=(2, 2),
                sse=1.0,
            ),
        )

    monkeypatch.setattr(
        runner,
        "run_file_parallel_kmeans_benchmark",
        execute,
    )

    return calls


def run_example(dataset, tmp_path, **overrides):
    options = {
        "output_directory": tmp_path / "prepared",
        "num_clusters": 2,
        "num_partitions": 2,
        "random_seed": 42,
    }

    options.update(overrides)

    return runner.run_labeled_spark_benchmark(
        None,
        dataset,
        **options,
    )


def test_successful_integration(
    dataset,
    tmp_path,
    fake_benchmark,
):
    result = run_example(dataset, tmp_path)

    assert result.dataset_name == "runner_test"
    assert result.evaluation.num_records == 4
    assert result.evaluation.num_features == 2
    assert result.evaluation.num_clusters == 2

    assert result.evaluation.cluster_counts == (2, 2)
    assert result.evaluation.sse == pytest.approx(1.0)

    assert result.evaluation.record_ids == (0, 1, 2, 3)
    assert result.evaluation.true_labels == (0, 0, 1, 1)

    assert len(fake_benchmark) == 1


def test_features_and_labels_saved_separately(
    dataset,
    tmp_path,
    fake_benchmark,
):
    result = run_example(dataset, tmp_path)

    features = result.prepared_dataset.dataset.path
    labels = result.prepared_dataset.labels_path

    assert features.exists()
    assert labels.exists()

    feature_rows = features.read_text().splitlines()
    label_rows = labels.read_text().splitlines()

    assert len(feature_rows) == 4
    assert len(label_rows) == 5

    assert label_rows[0] == "record_id,label"


def test_configuration_forwarded(
    dataset,
    tmp_path,
    fake_benchmark,
):
    run_example(
        dataset,
        tmp_path,
        random_seed=123,
        local_max_iterations=25,
        global_max_iterations=30,
        aggregation_restarts=3,
    )

    _, prepared, kwargs = fake_benchmark[0]

    assert prepared.num_records == 4
    assert prepared.num_features == 2
    assert prepared.num_clusters == 2

    assert kwargs["num_partitions"] == 2
    assert kwargs["random_seed"] == 123
    assert kwargs["local_max_iterations"] == 25
    assert kwargs["global_max_iterations"] == 30
    assert kwargs["aggregation_restarts"] == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_clusters": 0},
        {"num_clusters": 5},
        {"num_partitions": 0},
        {"num_partitions": 5},
        {"aggregation_restarts": 0},
    ],
)
def test_invalid_configuration_rejected(
    dataset,
    tmp_path,
    overrides,
):
    with pytest.raises(ValueError):
        run_example(
            dataset,
            tmp_path,
            **overrides,
        )

    assert not (tmp_path / "prepared").exists()


def test_existing_files_not_overwritten(
    dataset,
    tmp_path,
    fake_benchmark,
):
    run_example(dataset, tmp_path)

    features = (
        tmp_path
        / "prepared"
        / "runner_test_features.csv"
    )

    original = features.read_bytes()

    with pytest.raises(FileExistsError):
        run_example(dataset, tmp_path)

    assert features.read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_records", 5),
        ("num_features", 3),
        ("num_clusters", 3),
    ],
)
def test_benchmark_metadata_mismatch_rejected(
    dataset,
    tmp_path,
    monkeypatch,
    field,
    value,
):
    def execute(*args, **kwargs):
        result = SimpleNamespace(
            num_records=4,
            num_features=2,
            num_clusters=2,
            clustering_result=SimpleNamespace(
                centers=((0.0, 0.5), (10.0, 10.5)),
                cluster_counts=(2, 2),
                sse=1.0,
            ),
        )

        setattr(result, field, value)
        return result

    monkeypatch.setattr(
        runner,
        "run_file_parallel_kmeans_benchmark",
        execute,
    )

    with pytest.raises(RuntimeError):
        run_example(dataset, tmp_path)


def test_sse_mismatch_rejected(
    dataset,
    tmp_path,
    monkeypatch,
):
    def execute(*args, **kwargs):
        return SimpleNamespace(
            num_records=4,
            num_features=2,
            num_clusters=2,
            clustering_result=SimpleNamespace(
                centers=((0.0, 0.5), (10.0, 10.5)),
                cluster_counts=(2, 2),
                sse=999.0,
            ),
        )

    monkeypatch.setattr(
        runner,
        "run_file_parallel_kmeans_benchmark",
        execute,
    )

    with pytest.raises(
        RuntimeError,
        match="SSE",
    ):
        run_example(dataset, tmp_path)


def test_cluster_count_mismatch_rejected(
    dataset,
    tmp_path,
    monkeypatch,
):
    def execute(*args, **kwargs):
        return SimpleNamespace(
            num_records=4,
            num_features=2,
            num_clusters=2,
            clustering_result=SimpleNamespace(
                centers=((0.0, 0.5), (10.0, 10.5)),
                cluster_counts=(3, 1),
                sse=1.0,
            ),
        )

    monkeypatch.setattr(
        runner,
        "run_file_parallel_kmeans_benchmark",
        execute,
    )

    with pytest.raises(
        RuntimeError,
        match="assignments",
    ):
        run_example(dataset, tmp_path)
