import pytest

from src.experiments.clustering_quality import (
    evaluate_clustering_quality,
)


def test_perfect_clustering():
    result = evaluate_clustering_quality(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    )

    assert result.adjusted_rand_index == pytest.approx(1)
    assert result.normalized_mutual_information == pytest.approx(1)
    assert result.purity == pytest.approx(1)


def test_permuted_cluster_identifiers():
    result = evaluate_clustering_quality(
        [0, 0, 1, 1],
        [9, 9, 4, 4],
    )

    assert result.adjusted_rand_index == pytest.approx(1)
    assert result.normalized_mutual_information == pytest.approx(1)
    assert result.purity == pytest.approx(1)


def test_imperfect_clustering():
    result = evaluate_clustering_quality(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    )

    assert result.purity == pytest.approx(0.5)
    assert result.adjusted_rand_index < 1
    assert result.normalized_mutual_information < 1


def test_empty_labels_rejected():
    with pytest.raises(ValueError):
        evaluate_clustering_quality([], [])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        evaluate_clustering_quality(
            [0, 1],
            [0],
        )


def test_invalid_label_type_rejected():
    with pytest.raises(ValueError):
        evaluate_clustering_quality(
            [0, 1],
            [0, None],
        )


def test_string_labels_supported():
    result = evaluate_clustering_quality(
        ["normal", "normal", "attack", "attack"],
        [1, 1, 0, 0],
    )

    assert result.purity == pytest.approx(1)
