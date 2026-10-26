"""
Dataset pipeline for the CSCE 5300 Parallel K-Means / Spark-Kmeans project.

Owner: Person 2 (Data & Preprocessing Lead).

Two datasets mirror the approved paper's experiments:

  Dataset I  - KDD Cup 99 (10% file, ~494K records, 41 features, 4 labels
               after dropping the tiny u2r class: dos, normal, probe, r2l).
  Dataset II - synthetic sklearn make_blobs data (20 features, 5 classes),
               produced at increasing sizes (1K ... 5M) for scalability runs.

Every dataset is standardized (zero mean, unit variance per feature) and
saved with its ground-truth labels, which are used ONLY for purity.

Public API (what Persons 3 and 4 should call):

    X, y = load_dataset("synthetic_1M")   # or "kdd99", "synthetic_10K", ...
    list_datasets()                       # names that exist on disk

Everything is deterministic given the seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# --------------------------------------------------------------------------
# Paths (resolved from this file, so they work from any working directory)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLES_DIR = ROOT / "data" / "samples"

# --------------------------------------------------------------------------
# Synthetic settings (match the paper's Dataset II and Person 4's benchmark)
# --------------------------------------------------------------------------
SEED = 42
N_FEATURES = 20
N_CENTERS = 5
CLUSTER_STD = 1.0
CENTER_BOX = (-10.0, 10.0)  # make_blobs default
SYNTHETIC_SIZES = [
    1_000, 10_000, 50_000, 100_000, 500_000,
    1_000_000, 2_500_000, 5_000_000,
]
SYNTHETIC_DTYPE = np.float32  # keeps the 5M file ~400 MB in memory

# --------------------------------------------------------------------------
# KDD Cup 99 definitions
# --------------------------------------------------------------------------
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]  # 41 features
KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
KDD_DEFAULT_FILE = RAW_DIR / "kddcup.data_10_percent.gz"

ATTACK_CATEGORY = {"normal": "normal"}
for _name in ["back", "land", "neptune", "pod", "smurf", "teardrop"]:
    ATTACK_CATEGORY[_name] = "dos"
for _name in ["ipsweep", "nmap", "portsweep", "satan"]:
    ATTACK_CATEGORY[_name] = "probe"
for _name in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
              "warezclient", "warezmaster"]:
    ATTACK_CATEGORY[_name] = "r2l"
for _name in ["buffer_overflow", "loadmodule", "perl", "rootkit"]:
    ATTACK_CATEGORY[_name] = "u2r"

# Labels kept after dropping u2r (the paper keeps 4 classes). Fixed order so
# integer ids are stable: 0=dos, 1=normal, 2=probe, 3=r2l.
KDD_LABELS = ["dos", "normal", "probe", "r2l"]
KDD_LABEL_ID = {name: i for i, name in enumerate(KDD_LABELS)}


# ==========================================================================
# Small helpers
# ==========================================================================
def size_label(n: int) -> str:
    """1000 -> '1K', 2_500_000 -> '2.5M', 5_000_000 -> '5M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _use_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Z-score each column. Constant columns (std == 0, e.g. KDD's all-zero
    num_outbound_cmds) become all zeros instead of dividing by zero.
    Statistics are computed in float64; the output keeps X's dtype.
    """
    X = np.asarray(X)
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    return (X - mean.astype(X.dtype)) / std.astype(X.dtype)


def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str = "dataset",
    expected_rows: int | None = None,
    expected_features: int | None = None,
    expected_classes: int | None = None,
) -> dict:
    """Raise ValueError if the dataset is malformed; otherwise return a summary."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"{name}: X must be 2-D, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"{name}: X has {len(X)} rows but y has {len(y)}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: X contains NaN or infinite values")
    if expected_rows is not None and len(X) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, got {len(X)}")
    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            f"{name}: expected {expected_features} features, got {X.shape[1]}"
        )
    labels, counts = np.unique(y, return_counts=True)
    if expected_classes is not None and len(labels) != expected_classes:
        raise ValueError(
            f"{name}: expected {expected_classes} classes, got {len(labels)}"
        )
    return {
        "name": name,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "label_counts": {int(k): int(v) for k, v in zip(labels, counts)},
    }


# ==========================================================================
# Saving / loading
# ==========================================================================
def save_dataset(name: str, X: np.ndarray, y: np.ndarray,
                 out_dir: Path = PROCESSED_DIR) -> Path:
    """Write features (f0..fN) plus an int `label` column."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = np.asarray(y, dtype=np.int64)
    if _use_parquet():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:  # fallback only; Spark reads parquet much faster
        print("WARNING: pyarrow not installed, falling back to .csv.gz "
              "(run: pip install pyarrow)")
        path = out_dir / f"{name}.csv.gz"
        df.to_csv(path, index=False)
    return path


def _find_dataset_file(name: str, data_dir: Path) -> Path:
    for ext in (".parquet", ".csv.gz"):
        path = Path(data_dir) / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Dataset '{name}' not found in {data_dir}. "
        "Generate it with:  python -m scripts.make_datasets"
    )


def load_dataset(name: str, data_dir: Path = PROCESSED_DIR):
    """Return (X, y): X is a 2-D float array, y an int label array."""
    path = _find_dataset_file(name, data_dir)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    y = df["label"].to_numpy()
    X = df.drop(columns="label").to_numpy()
    return X, y


def list_datasets(data_dir: Path = PROCESSED_DIR) -> list[str]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    names = {
        p.name.replace(".parquet", "").replace(".csv.gz", "")
        for p in data_dir.iterdir()
        if p.name.endswith((".parquet", ".csv.gz"))
    }
    return sorted(names)


def write_sample(name: str, n: int = 1000, seed: int = SEED,
                 out_dir: Path = SAMPLES_DIR,
                 data_dir: Path = PROCESSED_DIR) -> Path:
    """Write a small CSV (safe to commit to GitHub) for tests and live demos."""
    X, y = load_dataset(name, data_dir)
    if len(X) > n:
        idx = np.sort(np.random.default_rng(seed).choice(len(X), n, replace=False))
        X, y = X[idx], y[idx]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    path = out_dir / f"{name}_sample.csv"
    df.to_csv(path, index=False, float_format="%.5f")
    return path


# ==========================================================================
# Synthetic data (paper Dataset II)
# ==========================================================================
def synthetic_centers(seed: int = SEED, n_features: int = N_FEATURES,
                      centers: int = N_CENTERS) -> np.ndarray:
    """
    The same cluster centers make_blobs(random_state=seed) would draw.
    Fixing them lets us generate huge datasets in chunks and still get ONE
    consistent set of clusters (and the same centers as Person 4's benchmark).
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(CENTER_BOX[0], CENTER_BOX[1], size=(centers, n_features))


def generate_synthetic_raw(
    n_samples: int,
    *,
    n_features: int = N_FEATURES,
    centers: int = N_CENTERS,
    seed: int = SEED,
    cluster_std: float = CLUSTER_STD,
    chunk_size: int = 500_000,
    dtype=SYNTHETIC_DTYPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate UNSTANDARDIZED blobs data with labels, chunk by chunk so peak
    memory stays low. Rows are shuffled within each chunk, so any prefix of
    the result is a random, class-balanced sample.
    """
    centers_arr = synthetic_centers(seed, n_features, centers)
    X = np.empty((n_samples, n_features), dtype=dtype)
    y = np.empty(n_samples, dtype=np.int64)
    for i, start in enumerate(range(0, n_samples, chunk_size)):
        m = min(chunk_size, n_samples - start)
        Xc, yc = make_blobs(
            n_samples=m,
            n_features=n_features,
            centers=centers_arr,
            cluster_std=cluster_std,
            random_state=seed + 1 + i,
        )
        X[start:start + m] = Xc
        y[start:start + m] = yc
    return X, y


def make_synthetic_datasets(
    sizes: list[int] | None = None,
    *,
    out_dir: Path = PROCESSED_DIR,
    seed: int = SEED,
) -> list[Path]:
    """
    Generate the largest size once, then slice nested prefixes for the smaller
    sizes (so size is the only thing that changes between experiments).
    Each size is standardized on its own statistics and saved.
    """
    sizes = sorted(sizes or SYNTHETIC_SIZES)
    print(f"Generating {sizes[-1]:,} raw synthetic rows "
          f"({N_FEATURES} features, {N_CENTERS} classes)...")
    X_raw, y_raw = generate_synthetic_raw(sizes[-1], seed=seed)

    paths = []
    for n in sizes:
        name = f"synthetic_{size_label(n)}"
        X = standardize_features(X_raw[:n])
        y = y_raw[:n]
        info = validate_dataset(
            X, y, name=name, expected_rows=n,
            expected_features=N_FEATURES, expected_classes=N_CENTERS,
        )
        path = save_dataset(name, X, y, out_dir)
        print(f"  saved {path.name}: {info['rows']:,} rows, "
              f"labels {info['label_counts']}")
        paths.append(path)
    return paths


# ==========================================================================
# KDD Cup 99 (paper Dataset I)
# ==========================================================================
def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def load_kdd_raw(path: Path | str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame with the 41 KDD_COLUMNS plus a `label` column.

    Uses a local copy of kddcup.data_10_percent(.gz) if present (header-less
    CSV, 42 columns). Otherwise falls back to sklearn's fetch_kddcup99, which
    downloads into data/raw/ (needs internet access).
    """
    path = Path(path) if path else KDD_DEFAULT_FILE
    if path.exists():
        return pd.read_csv(path, header=None, names=KDD_COLUMNS + ["label"])

    from sklearn.datasets import fetch_kddcup99
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_kddcup99(percent10=True, as_frame=True, data_home=str(RAW_DIR))
    df = bunch.data.copy()
    df.columns = KDD_COLUMNS  # sklearn's names differ slightly; same order
    for col in KDD_CATEGORICAL:
        df[col] = df[col].map(_decode)
    df["label"] = pd.Series(bunch.target).map(_decode).to_numpy()
    return df


def preprocess_kdd(df: pd.DataFrame):
    """
    Paper-style preprocessing, BEFORE standardization.

      1. Strip the trailing '.' from attack names and map them to
         dos / probe / r2l / u2r / normal.
      2. Drop u2r (too few samples; the paper eliminates it).
      3. Ordinal-encode protocol_type, service, flag (sorted categories ->
         deterministic codes). This keeps the paper's 41 dimensions.

    Returns (X_unscaled float64, y int64, info dict).
    """
    labels = df["label"].astype(str).str.strip().str.rstrip(".")
    category = labels.map(ATTACK_CATEGORY)
    if category.isna().any():
        unknown = sorted(labels[category.isna()].unique())
        raise ValueError(f"Unknown KDD attack labels: {unknown}")

    n_raw = len(df)
    keep = (category != "u2r").to_numpy()
    n_u2r = int((~keep).sum())

    feats = df.loc[keep, KDD_COLUMNS].copy()
    for col in KDD_CATEGORICAL:
        feats[col] = pd.Categorical(feats[col].astype(str)).codes
    X = feats.to_numpy(dtype=np.float64)
    y = category[keep].map(KDD_LABEL_ID).to_numpy(dtype=np.int64)

    info = {
        "rows_raw": n_raw,
        "rows_dropped_u2r": n_u2r,
        "rows_final": int(len(X)),
        "features": int(X.shape[1]),
        "label_names": KDD_LABELS,
        "label_counts": {
            KDD_LABELS[k]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        "paper_rows": 493_970,
        "difference_vs_paper": int(len(X)) - 493_970,
    }
    return X, y, info


def make_kdd_dataset(
    raw_path: Path | str | None = None,
    out_dir: Path = PROCESSED_DIR,
) -> Path:
    """Load raw KDD, preprocess, standardize, validate, save as 'kdd99'."""
    df = load_kdd_raw(raw_path)
    X, y, info = preprocess_kdd(df)
    X = standardize_features(X)
    validate_dataset(
        X, y, name="kdd99", expected_features=41,
        expected_classes=len(KDD_LABELS),
    )
    path = save_dataset("kdd99", X, y, out_dir)
    summary_path = Path(out_dir) / "kdd99_summary.json"
    summary_path.write_text(json.dumps(info, indent=2))
    print(f"  saved {path.name}: {info['rows_final']:,} rows "
          f"(dropped {info['rows_dropped_u2r']} u2r; "
          f"paper reports {info['paper_rows']:,})")
    print(f"  label counts: {info['label_counts']}")
    return path
