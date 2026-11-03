"""
Report-ready summaries for independently verified clustering quality.

This module reads the verified clustering-quality comparison CSV
produced from `src/experiments/clustering_quality.py` evaluations and
turns it into a compact, per-dataset comparison suitable for the
methodology document, final report, and presentation.

It does not compute purity, ARI, or NMI itself, and it does not run
any clustering. Its responsibility is limited to loading already
verified rows and summarizing them for reporting.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QualityComparisonRow:
    """One verified clustering-quality measurement for one run."""

    dataset_name: str
    algorithm: str
    master: str
    num_records: int
    num_features: int
    num_clusters: int
    sse: float
    purity: float
    adjusted_rand_index: float
    normalized_mutual_information: float
    cluster_counts: tuple[int, ...]
    elapsed_seconds: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class QualityComparisonSummary:
    """
    Per-dataset comparison of Spark and traditional clustering quality.

    `spark_master` identifies which non-`local[1]` Spark master the
    comparison master's runtime was drawn from; when only `local[1]`
    is present for a dataset, `spark_comparison_master_seconds` and
    `speedup` are left as `None` rather than invented.
    """

    dataset_name: str
    spark_base_master: str
    spark_base_seconds: float
    spark_comparison_master: str | None
    spark_comparison_master_seconds: float | None
    speedup: float | None
    spark_purity: float
    spark_adjusted_rand_index: float
    spark_normalized_mutual_information: float
    traditional_seconds: float
    traditional_purity: float
    traditional_adjusted_rand_index: float
    traditional_normalized_mutual_information: float
    purity_gap: float
    adjusted_rand_index_gap: float
    normalized_mutual_information_gap: float


REQUIRED_COLUMNS = (
    "dataset",
    "algorithm",
    "master",
    "records",
    "features",
    "clusters",
    "sse",
    "purity",
    "ari",
    "nmi",
    "cluster_counts",
    "elapsed_seconds",
    "iterations",
    "converged",
)


def _parse_bool(value: str, *, row_number: int) -> bool:
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid boolean value in row {row_number}: {value!r}"
    )


def _parse_cluster_counts(
    value: str,
    *,
    row_number: int,
) -> tuple[int, ...]:
    if not value.strip():
        raise ValueError(
            f"Empty cluster_counts in row {row_number}."
        )

    try:
        counts = tuple(
            int(count)
            for count in value.split(";")
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid cluster_counts in row {row_number}: {value!r}"
        ) from exc

    if any(count < 0 for count in counts):
        raise ValueError(
            f"Negative cluster count in row {row_number}."
        )

    return counts


def load_quality_comparison_csv(
    path: str | Path,
) -> tuple[QualityComparisonRow, ...]:
    """
    Load verified clustering-quality rows from a comparison CSV.

    The CSV is treated as already-verified data; this function
    validates structure and numeric well-formedness only. It never
    recomputes purity, ARI, or NMI.
    """

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Quality comparison file does not exist: {path}"
        )

    rows = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        columns = reader.fieldnames

        if not columns or set(REQUIRED_COLUMNS) - set(columns):
            missing = (
                set(REQUIRED_COLUMNS) - set(columns or ())
            )
            raise ValueError(
                f"Quality comparison file is missing columns: {missing}"
            )

        for row_index, raw_row in enumerate(reader):
            row_number = row_index + 2

            if None in raw_row or any(
                value is None for value in raw_row.values()
            ):
                raise ValueError(
                    f"Malformed CSV row: {row_number}"
                )

            try:
                sse = float(raw_row["sse"])
                purity = float(raw_row["purity"])
                ari = float(raw_row["ari"])
                nmi = float(raw_row["nmi"])
                elapsed_seconds = float(
                    raw_row["elapsed_seconds"]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value in row {row_number}."
                ) from exc

            for name, value in (
                ("sse", sse),
                ("purity", purity),
                ("ari", ari),
                ("nmi", nmi),
                ("elapsed_seconds", elapsed_seconds),
            ):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite {name} in row {row_number}."
                    )

            if not (0.0 <= purity <= 1.0 + 1e-9):
                raise ValueError(
                    f"Purity out of range in row {row_number}: {purity}"
                )

            if elapsed_seconds < 0:
                raise ValueError(
                    f"Negative elapsed_seconds in row {row_number}."
                )

            rows.append(
                QualityComparisonRow(
                    dataset_name=raw_row["dataset"],
                    algorithm=raw_row["algorithm"],
                    master=raw_row["master"],
                    num_records=int(raw_row["records"]),
                    num_features=int(raw_row["features"]),
                    num_clusters=int(raw_row["clusters"]),
                    sse=sse,
                    purity=purity,
                    adjusted_rand_index=ari,
                    normalized_mutual_information=nmi,
                    cluster_counts=_parse_cluster_counts(
                        raw_row["cluster_counts"],
                        row_number=row_number,
                    ),
                    elapsed_seconds=elapsed_seconds,
                    iterations=int(raw_row["iterations"]),
                    converged=_parse_bool(
                        raw_row["converged"],
                        row_number=row_number,
                    ),
                )
            )

    if not rows:
        raise ValueError(
            "Quality comparison file contains no data rows."
        )

    return tuple(rows)


def summarize_quality_comparisons(
    rows: tuple[QualityComparisonRow, ...],
) -> tuple[QualityComparisonSummary, ...]:
    """
    Build one Spark-vs-traditional quality comparison per dataset.

    For each dataset, the `local[1]` Spark run is treated as the base
    measurement. If a second Spark master is present for the same
    dataset, its runtime is used to report a speedup; otherwise the
    comparison master and speedup are left as `None` rather than
    fabricated. Exactly one `parallel_kmeans` row at `local[1]` and
    exactly one `traditional_kmeans` row are required per dataset.
    """

    if not rows:
        raise ValueError(
            "rows must contain at least one measurement."
        )

    dataset_names = sorted(
        {row.dataset_name for row in rows}
    )

    summaries = []

    for dataset_name in dataset_names:
        dataset_rows = tuple(
            row
            for row in rows
            if row.dataset_name == dataset_name
        )

        spark_rows = tuple(
            row
            for row in dataset_rows
            if row.algorithm == "parallel_kmeans"
        )

        traditional_rows = tuple(
            row
            for row in dataset_rows
            if row.algorithm == "traditional_kmeans"
        )

        if len(traditional_rows) != 1:
            raise ValueError(
                "Expected exactly one traditional_kmeans row for "
                f"dataset {dataset_name!r}, found "
                f"{len(traditional_rows)}."
            )

        base_rows = tuple(
            row
            for row in spark_rows
            if row.master == "local[1]"
        )

        if len(base_rows) != 1:
            raise ValueError(
                "Expected exactly one parallel_kmeans local[1] row "
                f"for dataset {dataset_name!r}, found "
                f"{len(base_rows)}."
            )

        base = base_rows[0]
        traditional = traditional_rows[0]

        comparison_rows = tuple(
            row
            for row in spark_rows
            if row.master != "local[1]"
        )

        comparison_master = None
        comparison_seconds = None
        speedup = None

        if comparison_rows:
            if len(comparison_rows) != 1:
                raise ValueError(
                    "Expected at most one non-local[1] Spark master "
                    f"for dataset {dataset_name!r}, found "
                    f"{len(comparison_rows)}."
                )

            comparison = comparison_rows[0]
            comparison_master = comparison.master
            comparison_seconds = comparison.elapsed_seconds

            if comparison_seconds > 0:
                speedup = base.elapsed_seconds / comparison_seconds

        summaries.append(
            QualityComparisonSummary(
                dataset_name=dataset_name,
                spark_base_master=base.master,
                spark_base_seconds=base.elapsed_seconds,
                spark_comparison_master=comparison_master,
                spark_comparison_master_seconds=comparison_seconds,
                speedup=speedup,
                spark_purity=base.purity,
                spark_adjusted_rand_index=base.adjusted_rand_index,
                spark_normalized_mutual_information=(
                    base.normalized_mutual_information
                ),
                traditional_seconds=traditional.elapsed_seconds,
                traditional_purity=traditional.purity,
                traditional_adjusted_rand_index=(
                    traditional.adjusted_rand_index
                ),
                traditional_normalized_mutual_information=(
                    traditional.normalized_mutual_information
                ),
                purity_gap=(
                    base.purity - traditional.purity
                ),
                adjusted_rand_index_gap=(
                    base.adjusted_rand_index
                    - traditional.adjusted_rand_index
                ),
                normalized_mutual_information_gap=(
                    base.normalized_mutual_information
                    - traditional.normalized_mutual_information
                ),
            )
        )

    return tuple(summaries)
