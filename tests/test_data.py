"""Tests for the data pipeline. Run from the repo root:  pytest tests/"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.data.datasets import (
    ATTACK_CATEGORY,
    KDD_COLUMNS,
    KDD_LABELS,
    generate_synthetic_raw,
    list_datasets,
    load_dataset,
    preprocess_kdd,
    save_dataset,
    size_label,
    standardize_features,
    synthetic_centers,
    validate_dataset,
    write_sample,
)


# ---------------------------------------------------------------- synthetic
def test_centers_match_make_blobs_seed_42():
    """Our centers equal what Person 4's make_blobs(random_state=42) draws."""
    _, _, centers = make_blobs(
        n_samples=10, n_features=20, centers=5, random_state=42, return_centers=True
    )
    assert np.allclose(centers, synthetic_centers(42))


def test_synthetic_shape_and_balance():
    X, y = generate_synthetic_raw(5_000, chunk_size=2_000)
    assert X.shape == (5_000, 20)
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}
    counts = np.bincount(y)
    assert counts.min() > 900  # roughly balanced (1000 each)


def test_synthetic_is_deterministic():
    X1, y1 = generate_synthetic_raw(3_000, chunk_size=1_000)
    X2, y2 = generate_synthetic_raw(3_000, chunk_size=1_000)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_size_label():
    assert size_label(1_000) == "1K"
    assert size_label(500_000) == "500K"
    assert size_label(1_000_000) == "1M"
    assert size_label(2_500_000) == "2.5M"
    assert size_label(5_000_000) == "5M"


# ------------------------------------------------------------- standardize
def test_standardize_zero_mean_unit_std():
    X = np.random.default_rng(0).normal(5, 3, size=(1_000, 4)).astype(np.float32)
    Z = standardize_features(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-3)
    assert Z.dtype == np.float32


def test_standardize_constant_column_has_no_nan():
    X = np.column_stack([np.zeros(50), np.arange(50.0)])
    Z = standardize_features(X)
    assert np.isfinite(Z).all()
    assert np.all(Z[:, 0] == 0)


# --------------------------------------------------------------------- KDD
def _fake_kdd(n_per_label: dict) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for label, n in n_per_label.items():
        for _ in range(n):
            row = dict(zip(KDD_COLUMNS, rng.random(len(KDD_COLUMNS))))
            row["protocol_type"] = rng.choice(["tcp", "udp", "icmp"])
            row["service"] = rng.choice(["http", "smtp", "ftp"])
            row["flag"] = rng.choice(["SF", "S0"])
            row["label"] = label + "."  # real files have a trailing dot
            rows.append(row)
    return pd.DataFrame(rows)


def test_preprocess_kdd_drops_u2r_and_encodes():
    df = _fake_kdd({"normal": 10, "smurf": 10, "satan": 5, "guess_passwd": 5,
                    "buffer_overflow": 3, "rootkit": 2})
    X, y, info = preprocess_kdd(df)
    assert info["rows_raw"] == 35
    assert info["rows_dropped_u2r"] == 5
    assert X.shape == (30, 41)
    assert X.dtype == np.float64 and np.isfinite(X).all()
    assert set(info["label_names"]) == set(KDD_LABELS)
    assert sum(info["label_counts"].values()) == 30
    assert len(y) == 30


def test_preprocess_kdd_rejects_unknown_label():
    df = _fake_kdd({"normal": 3, "totally_new_attack": 2})
    with pytest.raises(ValueError):
        preprocess_kdd(df)


def test_attack_map_u2r_count_matches_kdd_10_percent():
    u2r = [k for k, v in ATTACK_CATEGORY.items() if v == "u2r"]
    assert sorted(u2r) == ["buffer_overflow", "loadmodule", "perl", "rootkit"]


# -------------------------------------------------- validate / save / load
def test_validate_rejects_nan_and_mismatch():
    X = np.zeros((5, 2))
    y = np.zeros(5, dtype=int)
    assert validate_dataset(X, y)["rows"] == 5
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_dataset(X_bad, y)
    with pytest.raises(ValueError):
        validate_dataset(X, y[:4])
    with pytest.raises(ValueError):
        validate_dataset(X, y, expected_features=3)


def test_save_load_roundtrip_and_sample(tmp_path):
    X, y = generate_synthetic_raw(1_200, chunk_size=1_200)
    X = standardize_features(X)
    save_dataset("synthetic_test", X, y, tmp_path)
    assert "synthetic_test" in list_datasets(tmp_path)
    X2, y2 = load_dataset("synthetic_test", tmp_path)
    assert np.allclose(X, X2, atol=1e-6)
    assert np.array_equal(y, y2)

    sample = write_sample("synthetic_test", n=100,
                          out_dir=tmp_path / "samples", data_dir=tmp_path)
    assert len(pd.read_csv(sample)) == 100


def test_load_missing_dataset_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_datasets"):
        load_dataset("nope", tmp_path)
