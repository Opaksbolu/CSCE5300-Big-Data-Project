# Data

Large datasets are **not** stored in GitHub. Regenerate them with the scripts below.

## Quick start (from the repository root)

```bash
pip install -r requirements-data.txt
python -m scripts.make_datasets                      # everything
python -m scripts.make_datasets --max-size 1000000   # skip 2.5M and 5M if memory is tight
```

## Using the data in code

```python
from src.data.datasets import load_dataset, list_datasets

X, y = load_dataset("synthetic_1M")   # standardized features, int labels
X, y = load_dataset("kdd99")
print(list_datasets())
```

`y` (ground truth) is for **purity only**. Never pass it to K-Means.

## Folders

| Folder | Contents | In git? |
| --- | --- | --- |
| `raw/` | Downloaded KDD Cup 99 file | No |
| `processed/` | Standardized datasets (`*.parquet`) | No |
| `samples/` | 1,000-row CSV samples for tests and the live demo | Yes |

## KDD Cup 99 source

Place `kddcup.data_10_percent.gz` (the 10% file) in `data/raw/`, from the UCI KDD
archive (kdd.ics.uci.edu/databases/kddcup99). If the file is absent, the script
falls back to `sklearn.datasets.fetch_kddcup99`, which needs internet access.

## Dataset names

`synthetic_1K`, `synthetic_10K`, `synthetic_50K`, `synthetic_100K`, `synthetic_500K`,
`synthetic_1M`, `synthetic_2.5M`, `synthetic_5M`, `kdd99`

All are standardized (mean 0, std 1 per feature). Seed = 42.
