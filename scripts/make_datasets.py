"""
Build every dataset used by the project (Person 2).

Run from the repository root:

    python -m scripts.make_datasets                        # everything
    python -m scripts.make_datasets --only synthetic       # synthetic only
    python -m scripts.make_datasets --only kdd             # KDD Cup 99 only
    python -m scripts.make_datasets --max-size 1000000     # skip 2.5M and 5M
    python -m scripts.make_datasets --only kdd --kdd-path data/raw/kddcup.data_10_percent.gz

Large outputs go to data/processed/ (git-ignored). Small CSV samples go to
data/samples/ (safe to commit).
"""

from __future__ import annotations

import argparse

from src.data.datasets import (
    SYNTHETIC_SIZES,
    list_datasets,
    make_kdd_dataset,
    make_synthetic_datasets,
    size_label,
    write_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--only", choices=["all", "synthetic", "kdd"], default="all")
    parser.add_argument("--max-size", type=int, default=max(SYNTHETIC_SIZES),
                        help="largest synthetic size to generate (default: 5,000,000)")
    parser.add_argument("--kdd-path", default=None,
                        help="local kddcup.data_10_percent(.gz); otherwise sklearn downloads it")
    parser.add_argument("--no-samples", action="store_true",
                        help="do not write the small CSV samples")
    args = parser.parse_args()

    if args.only in ("all", "synthetic"):
        sizes = [n for n in SYNTHETIC_SIZES if n <= args.max_size]
        print(f"\n== Synthetic datasets: {[size_label(n) for n in sizes]} ==")
        make_synthetic_datasets(sizes)

    if args.only in ("all", "kdd"):
        print("\n== KDD Cup 99 ==")
        make_kdd_dataset(args.kdd_path)

    if not args.no_samples:
        print("\n== Samples (commit these, not the big files) ==")
        available = list_datasets()
        for name in ("synthetic_1K", "kdd99"):
            if name in available:
                print(f"  wrote {write_sample(name).name}")

    print(f"\nDatasets on disk: {list_datasets()}")


if __name__ == "__main__":
    main()
