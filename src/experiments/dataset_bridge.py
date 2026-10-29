"""
Dataset format bridge for CSCE 5300 Parallel K-Means.

Converts small labeled CSV datasets into numerical clustering
features while retaining record identifiers and ground-truth
labels for later evaluation.

The bridge does not perform clustering or calculate purity.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from src.experiments.file_dataset import FileDataset


@dataclass(frozen=True)
class LabeledDataset:
    """Validated small dataset with aligned features and labels."""

    name: str
    points: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    record_ids: tuple[int, ...]

    @property
    def num_records(self) -> int:
        return len(self.points)

    @property
    def num_features(self) -> int:
        return len(self.points[0])

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))


@dataclass(frozen=True)
class PreparedSparkDataset:
    """Paths and metadata for Spark input and evaluation."""

    dataset: FileDataset
    labels_path: Path
    dataset_name: str


def load_labeled_csv(path: str | Path) -> LabeledDataset:
    """
    Load a small headered CSV with numerical features and labels.

    The label column is never included in the clustering features.
    Record identifiers follow the original CSV row positions.
    """

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {path}"
        )

    points = []
    labels = []
    record_ids = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        columns = reader.fieldnames

        if not columns or "label" not in columns:
            raise ValueError(
                "Dataset must contain a header and label column."
            )

        if len(columns) != len(set(columns)):
            raise ValueError(
                "Dataset contains duplicate column names."
            )

        if "record_id" in columns:
            raise ValueError(
                "Input record_id columns are not supported "
                "by this small-sample adapter."
            )

        feature_columns = [
            column
            for column in columns
            if column != "label"
        ]

        if not feature_columns:
            raise ValueError(
                "Dataset must contain numerical features."
            )

        for row_index, row in enumerate(reader):
            if None in row or any(
                value is None for value in row.values()
            ):
                raise ValueError(
                    f"Malformed CSV row: {row_index + 2}"
                )

            point = []

            for column in feature_columns:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid feature in row "
                        f"{row_index + 2}, column {column}."
                    ) from exc

                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite feature in row "
                        f"{row_index + 2}, column {column}."
                    )

                point.append(value)

            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid label in row {row_index + 2}."
                ) from exc

            points.append(tuple(point))
            labels.append(label)
            record_ids.append(row_index)

    if not points:
        raise ValueError("Dataset cannot be empty.")

    return LabeledDataset(
        name=path.stem,
        points=tuple(points),
        labels=tuple(labels),
        record_ids=tuple(record_ids),
    )


def prepare_spark_csv(
    dataset: LabeledDataset,
    *,
    output_directory: str | Path,
    num_clusters: int,
) -> PreparedSparkDataset:
    """
    Write a features-only Spark CSV and aligned label CSV.

    This initial implementation targets small validated datasets.
    It never overwrites existing output files.
    """

    if num_clusters <= 0:
        raise ValueError(
            "num_clusters must be positive."
        )

    if num_clusters > dataset.num_records:
        raise ValueError(
            "num_clusters cannot exceed record count."
        )

    output_directory = Path(
        output_directory
    ).expanduser()

    features_path = (
        output_directory / f"{dataset.name}_features.csv"
    )

    labels_path = (
        output_directory / f"{dataset.name}_labels.csv"
    )

    for path in (features_path, labels_path):
        if path.exists():
            raise FileExistsError(
                f"STOP: Output already exists: {path}"
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with features_path.open(
            "x",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerows(dataset.points)

        with labels_path.open(
            "x",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["record_id", "label"])

            writer.writerows(
                zip(
                    dataset.record_ids,
                    dataset.labels,
                )
            )

    except Exception:
        # Preserve generated files for inspection if an error
        # occurs; never silently delete user data.
        raise

    return PreparedSparkDataset(
        dataset=FileDataset(
            path=features_path,
            num_records=dataset.num_records,
            num_features=dataset.num_features,
            num_clusters=num_clusters,
        ),
        labels_path=labels_path,
        dataset_name=dataset.name,
    )
