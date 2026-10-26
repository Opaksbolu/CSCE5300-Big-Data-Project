"""
Tests for the conventional single-process K-Means benchmark.

These tests verify correctness, reproducibility, input validation,
and safe CSV output without modifying existing experiment results.
"""

import csv
import sys

import pytest

from scripts import run_traditional_kmeans


def test_baseline_returns_expected_fields():
    """A baseline experiment should report its key metrics."""

    result = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=2,
        num_clusters=2,
        random_seed=42,
    )

    assert result["algorithm"] == "traditional-kmeans"
    assert result["num_records"] == 100
    assert result["num_features"] == 2
    assert result["num_clusters"] == 2

    assert result["iterations"] >= 1
    assert result["sse"] >= 0
    assert result["total_runtime_seconds"] >= 0

    assert isinstance(result["converged"], bool)


def test_cluster_counts_match_dataset_size():
    """Every generated point must belong to one cluster."""

    result = run_traditional_kmeans.run_baseline(
        num_records=103,
        num_features=3,
        num_clusters=4,
        random_seed=42,
    )

    counts = [
        int(value)
        for value in result["cluster_counts"].split(";")
    ]

    assert len(counts) == 4
    assert sum(counts) == 103
    assert all(count >= 0 for count in counts)


def test_baseline_is_reproducible():
    """Identical seeds should produce identical clustering results."""

    first = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
    )

    second = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
    )

    assert first["sse"] == pytest.approx(second["sse"])
    assert first["cluster_counts"] == second["cluster_counts"]
    assert first["iterations"] == second["iterations"]
    assert first["converged"] == second["converged"]


@pytest.mark.parametrize(
    "num_records,num_features,num_clusters",
    [
        (0, 2, 2),
        (100, 0, 2),
        (100, 2, 0),
        (2, 2, 3),
    ],
)
def test_baseline_rejects_invalid_dataset_parameters(
    num_records,
    num_features,
    num_clusters,
):
    """Invalid dataset configurations should raise ValueError."""

    with pytest.raises(ValueError):
        run_traditional_kmeans.run_baseline(
            num_records=num_records,
            num_features=num_features,
            num_clusters=num_clusters,
        )


def test_main_writes_valid_csv(tmp_path, monkeypatch):
    """The command-line benchmark should produce valid CSV rows."""

    output = tmp_path / "baseline.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--record-counts",
            "20,30",
            "--num-features",
            "2",
            "--num-clusters",
            "2",
            "--repetitions",
            "2",
            "--random-seed",
            "42",
            "--output",
            str(output),
        ],
    )

    run_traditional_kmeans.main()

    assert output.exists()

    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 4

    assert [int(row["num_records"]) for row in rows] == [
        20, 20, 30, 30
    ]

    assert [int(row["random_seed"]) for row in rows] == [
        42, 43, 42, 43
    ]

    assert [int(row["repetition"]) for row in rows] == [
        1, 2, 1, 2
    ]

    assert all(float(row["sse"]) >= 0 for row in rows)


def test_main_does_not_overwrite_existing_csv(
    tmp_path,
    monkeypatch,
):
    """An existing result file must remain unchanged."""

    output = tmp_path / "existing.csv"
    original_content = "important existing results\\n"

    output.write_text(original_content)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--record-counts",
            "20",
            "--num-features",
            "2",
            "--num-clusters",
            "2",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="already exists"):
        run_traditional_kmeans.main()

    assert output.read_text() == original_content


@pytest.mark.parametrize(
    "record_counts",
    [
        "0",
        "-10",
        "abc",
        "10,abc",
    ],
)
def test_main_rejects_invalid_record_counts(
    record_counts,
    tmp_path,
    monkeypatch,
):
    """The CLI must reject invalid record-count arguments."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--record-counts",
            record_counts,
            "--output",
            str(tmp_path / "invalid.csv"),
        ],
    )

    with pytest.raises(SystemExit):
        run_traditional_kmeans.main()



def test_warmup_runs_are_excluded_from_csv(
    tmp_path,
    monkeypatch,
):
    """Warm-up experiments must not appear in measured CSV output."""

    output = tmp_path / "warmup_results.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--record-counts",
            "20",
            "--num-features",
            "2",
            "--num-clusters",
            "2",
            "--repetitions",
            "3",
            "--warmup-runs",
            "2",
            "--random-seed",
            "42",
            "--output",
            str(output),
        ],
    )

    run_traditional_kmeans.main()

    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 3

    assert [int(row["repetition"]) for row in rows] == [
        1, 2, 3
    ]

    assert [int(row["random_seed"]) for row in rows] == [
        42, 43, 44
    ]


def test_warmup_runs_do_not_change_measured_results(
    tmp_path,
    monkeypatch,
):
    """Warm-ups must not change deterministic clustering results."""

    results = []

    for warmup_count in (0, 2):
        output = tmp_path / f"warmup_{warmup_count}.csv"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_traditional_kmeans.py",
                "--record-counts",
                "40",
                "--num-features",
                "3",
                "--num-clusters",
                "2",
                "--repetitions",
                "2",
                "--warmup-runs",
                str(warmup_count),
                "--random-seed",
                "42",
                "--output",
                str(output),
            ],
        )

        run_traditional_kmeans.main()

        with output.open(newline="") as file:
            rows = list(csv.DictReader(file))

        results.append(rows)

    without_warmup, with_warmup = results

    assert len(without_warmup) == len(with_warmup) == 2

    for first, second in zip(
        without_warmup,
        with_warmup,
    ):
        assert first["random_seed"] == second["random_seed"]
        assert first["iterations"] == second["iterations"]
        assert first["converged"] == second["converged"]
        assert first["cluster_counts"] == second["cluster_counts"]

        assert float(first["sse"]) == pytest.approx(
            float(second["sse"])
        )


def test_negative_warmup_count_is_rejected(
    tmp_path,
    monkeypatch,
):
    """Negative warm-up counts must be rejected."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--warmup-runs",
            "-1",
            "--output",
            str(tmp_path / "invalid.csv"),
        ],
    )

    with pytest.raises(SystemExit):
        run_traditional_kmeans.main()

def test_multiple_initializations_select_lowest_sse():
    """More initializations must retain the lowest observed SSE."""

    results = {}

    for count in (1, 5, 10):
        results[count] = run_traditional_kmeans.run_baseline(
            num_records=100,
            num_features=3,
            num_clusters=3,
            random_seed=42,
            num_initializations=count,
        )

    assert results[5]["sse"] <= results[1]["sse"] + 1e-9
    assert results[10]["sse"] <= results[5]["sse"] + 1e-9


def test_multiple_initializations_are_reproducible():
    """Repeated experiments must select identical clustering results."""

    first = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
        num_initializations=5,
    )

    second = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
        num_initializations=5,
    )

    assert first["sse"] == pytest.approx(second["sse"])

    assert (
        first["best_initialization_seed"]
        == second["best_initialization_seed"]
    )

    assert first["cluster_counts"] == second["cluster_counts"]
    assert first["iterations"] == second["iterations"]
    assert first["converged"] == second["converged"]


@pytest.mark.parametrize("count", [0, -1, -5])
def test_rejects_nonpositive_initialization_counts(count):
    """Initialization counts must be positive integers."""

    with pytest.raises(
        ValueError,
        match="num_initializations must be greater than zero",
    ):
        run_traditional_kmeans.run_baseline(
            num_records=100,
            num_features=3,
            num_clusters=3,
            random_seed=42,
            num_initializations=count,
        )


def test_single_initialization_preserves_original_behavior():
    """Explicitly requesting one initialization matches the default."""

    original = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
    )

    explicit = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
        num_initializations=1,
    )

    assert original["sse"] == pytest.approx(explicit["sse"])
    assert original["cluster_counts"] == explicit["cluster_counts"]
    assert original["iterations"] == explicit["iterations"]

    assert explicit["num_initializations"] == 1
    assert explicit["best_initialization_seed"] == 42


def test_multiple_initializations_assign_all_records():
    """The selected clustering result must assign every data point."""

    result = run_traditional_kmeans.run_baseline(
        num_records=103,
        num_features=3,
        num_clusters=4,
        random_seed=42,
        num_initializations=5,
    )

    counts = [
        int(value)
        for value in result["cluster_counts"].split(";")
    ]

    assert len(counts) == 4
    assert sum(counts) == 103

    assert result["num_initializations"] == 5

    assert (
        42 <= result["best_initialization_seed"] <= 46
    )


def test_multiple_initializations_match_independent_sse_minimum():
    """The selected SSE must equal the best independent fit."""

    from src.experiments.synthetic_data import (
        generate_synthetic_dataset,
    )
    from src.parallel.local_kmeans import fit_local_kmeans

    dataset = generate_synthetic_dataset(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
    )

    individual_results = [
        fit_local_kmeans(
            dataset.points,
            k=3,
            random_seed=seed,
        )
        for seed in range(42, 47)
    ]

    expected_sse = min(
        result.sse
        for result in individual_results
    )

    actual = run_traditional_kmeans.run_baseline(
        num_records=100,
        num_features=3,
        num_clusters=3,
        random_seed=42,
        num_initializations=5,
    )

    assert actual["sse"] == pytest.approx(expected_sse)

def test_cli_multiple_initializations_written_to_csv(
    tmp_path,
    monkeypatch,
):
    """CLI must preserve the requested initialization count in CSV."""

    output = tmp_path / "multiple_initializations.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--record-counts", "100",
            "--num-features", "3",
            "--num-clusters", "3",
            "--repetitions", "2",
            "--warmup-runs", "0",
            "--num-initializations", "5",
            "--random-seed", "42",
            "--output", str(output),
        ],
    )

    run_traditional_kmeans.main()

    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2

    assert [int(row["num_initializations"]) for row in rows] == [
        5, 5
    ]

    assert [int(row["random_seed"]) for row in rows] == [
        42, 43
    ]

    for row in rows:
        dataset_seed = int(row["random_seed"])
        best_seed = int(row["best_initialization_seed"])

        assert dataset_seed <= best_seed < dataset_seed + 5
        assert float(row["sse"]) >= 0
        assert sum(
            int(count)
            for count in row["cluster_counts"].split(";")
        ) == int(row["num_records"])


@pytest.mark.parametrize(
    "initialization_count",
    ["0", "-1", "-5"],
)
def test_cli_rejects_invalid_initialization_counts(
    initialization_count,
    tmp_path,
    monkeypatch,
):
    """CLI must reject nonpositive initialization counts."""

    output = tmp_path / "invalid_initializations.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--num-initializations",
            initialization_count,
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit):
        run_traditional_kmeans.main()

    assert not output.exists()


def test_cli_warmups_use_requested_initializations(
    tmp_path,
    monkeypatch,
):
    """Warm-up and measured runs must use the same restart count."""

    output = tmp_path / "warmup_initializations.csv"

    observed_initializations = []

    original_run_baseline = run_traditional_kmeans.run_baseline

    def recording_run_baseline(**kwargs):
        observed_initializations.append(
            kwargs["num_initializations"]
        )
        return original_run_baseline(**kwargs)

    monkeypatch.setattr(
        run_traditional_kmeans,
        "run_baseline",
        recording_run_baseline,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_traditional_kmeans.py",
            "--record-counts", "30",
            "--num-features", "2",
            "--num-clusters", "2",
            "--repetitions", "2",
            "--warmup-runs", "2",
            "--num-initializations", "5",
            "--output", str(output),
        ],
    )

    run_traditional_kmeans.main()

    assert observed_initializations == [5, 5, 5, 5]

    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert all(
        int(row["num_initializations"]) == 5
        for row in rows
    )
