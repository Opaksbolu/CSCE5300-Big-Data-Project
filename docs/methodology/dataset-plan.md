# Dataset Plan (Person 2: Data & Preprocessing Lead)

Goal: provide the same two kinds of data as the approved paper, "Big data clustering
method based on parallel K-means" (IEEE ICPECA 2024), plus a size ladder for
scalability experiments.

## 1. Datasets

| | Dataset I: KDD Cup 99 | Dataset II: synthetic |
| --- | --- | --- |
| Paper | 493,970 records, 41 features, 4 labels (DOS, "r2u", probing, normal) | 5,000,000 records, 20 features, 5 classes, sklearn-generated |
| Ours | KDD 10% file, u2r dropped, 41 features, 4 labels (dos, normal, probe, r2l) | `make_blobs`, 20 features, 5 centers, seed 42 |
| Sizes | one dataset (~494K) | 1K, 10K, 50K, 100K, 500K, 1M, 2.5M, 5M (as hardware allows) |
| Name | `kdd99` | `synthetic_<size>` |

## 2. KDD Cup 99 preprocessing

1. Load the 10% file (494,021 rows) and strip the trailing "." from attack names.
2. Map attacks to dos / probe / r2l / u2r / normal.
3. Drop **u2r** (52 rows in the 10% file), as the paper eliminates this class.
   Result: about 493,969 rows, within one row of the paper's 493,970 (to be confirmed by the run).
4. Ordinal-encode the three text columns (`protocol_type`, `service`, `flag`) to keep the paper's
   41 dimensions. Limitation: ordinal codes impose an arbitrary order under Euclidean distance.
5. Standardize every feature (z-score). Constant columns become 0 (no divide-by-zero).
6. Save features and labels together; labels are used for purity only.

Notes on the paper: it writes "r2u", which we interpret as r2l, and it says "about 1000" u2r
rows where the 10% file has 52. We follow the standard KDD categories and document this.

## 3. Synthetic pipeline

- `make_blobs(n_features=20, centers=5, random_state=42)`: same settings as Person 4's benchmark.
  Centers are fixed so data can be generated in chunks with one consistent set of clusters.
- The largest size is generated once; smaller sizes are nested prefixes, so only size changes.
- Each size is standardized on its own statistics.
- Stored as float32 Parquet (about 400 MB in memory at 5M rows). Labels saved alongside.
- The paper does not say which sklearn function it used or its `cluster_std`; we assume
  `make_blobs` with defaults (`cluster_std=1.0`) and report this as an assumption.

## 4. Interface for the team

```python
from src.data.datasets import load_dataset
X, y = load_dataset("synthetic_1M")   # standardized X, int labels y
```

Standardization is done once in the data pipeline, so both algorithms receive identical inputs.
Spark code should not standardize again.

## 5. Validation

Row/feature/class counts, no NaN or infinite values, fixed seeds (reproducible), tests in
`tests/test_data.py`. Run `python -m scripts.make_datasets` to rebuild everything.

## 6. Repository rules

Large files stay out of git (`data/raw`, `data/processed`). Only `data/samples/*.csv`
(1,000 rows each) are committed, for tests and the live demo.
