import pytest

from src.experiments.clustering_quality_report import (
    load_quality_comparison_csv,
    summarize_quality_comparisons,
)


HEADER = (
    "dataset,algorithm,master,records,features,clusters,sse,purity,"
    "ari,nmi,cluster_counts,elapsed_seconds,iterations,converged\n"
)


def _write_csv(tmp_path, body: str):
    path = tmp_path / "clustering_quality_comparison.csv"
    path.write_text(HEADER + body)
    return path


SAMPLE_BODY = (
    "synthetic,parallel_kmeans,local[1],1000,20,5,1032.37,1.0,1.0,1.0,"
    "218;191;206;192;193,1.6055,3,True\n"
    "synthetic,parallel_kmeans,local[2],1000,20,5,1032.37,1.0,1.0,1.0,"
    "218;191;206;192;193,0.9978,3,True\n"
    "kdd99,parallel_kmeans,local[1],1000,41,5,21072.55,0.797,-0.0054,"
    "0.0111,1;936;1;61;1,1.2167,4,True\n"
    "kdd99,parallel_kmeans,local[2],1000,41,5,21072.55,0.797,-0.0054,"
    "0.0111,1;936;1;61;1,1.0420,4,True\n"
    "synthetic,traditional_kmeans,single_process,1000,20,5,1032.37,"
    "1.0,1.0,0.9999999999999999,192;193;191;218;206,0.4779,2,True\n"
    "kdd99,traditional_kmeans,single_process,1000,41,5,9342.83,0.975,"
    "0.4318,0.5352,541;1;199;61;198,0.6013,4,True\n"
)


def test_load_quality_comparison_csv(tmp_path):
    path = _write_csv(tmp_path, SAMPLE_BODY)

    rows = load_quality_comparison_csv(path)

    assert len(rows) == 6
    assert rows[0].dataset_name == "synthetic"
    assert rows[0].algorithm == "parallel_kmeans"
    assert rows[0].master == "local[1]"
    assert rows[0].cluster_counts == (218, 191, 206, 192, 193)
    assert rows[0].converged is True


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_quality_comparison_csv(tmp_path / "missing.csv")


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("dataset,algorithm\nsynthetic,parallel_kmeans\n")

    with pytest.raises(ValueError):
        load_quality_comparison_csv(path)


def test_invalid_numeric_value_rejected(tmp_path):
    body = (
        "synthetic,parallel_kmeans,local[1],1000,20,5,notanumber,1.0,"
        "1.0,1.0,218;191;206;192;193,1.6055,3,True\n"
    )
    path = _write_csv(tmp_path, body)

    with pytest.raises(ValueError):
        load_quality_comparison_csv(path)


def test_purity_out_of_range_rejected(tmp_path):
    body = (
        "synthetic,parallel_kmeans,local[1],1000,20,5,1032.37,1.5,"
        "1.0,1.0,218;191;206;192;193,1.6055,3,True\n"
    )
    path = _write_csv(tmp_path, body)

    with pytest.raises(ValueError):
        load_quality_comparison_csv(path)


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(HEADER)

    with pytest.raises(ValueError):
        load_quality_comparison_csv(path)


def test_summarize_quality_comparisons(tmp_path):
    path = _write_csv(tmp_path, SAMPLE_BODY)
    rows = load_quality_comparison_csv(path)

    summaries = summarize_quality_comparisons(rows)
    by_dataset = {
        summary.dataset_name: summary
        for summary in summaries
    }

    assert set(by_dataset) == {"synthetic", "kdd99"}

    synthetic = by_dataset["synthetic"]
    assert synthetic.spark_base_master == "local[1]"
    assert synthetic.spark_comparison_master == "local[2]"
    assert synthetic.speedup == pytest.approx(
        1.6055 / 0.9978, rel=1e-3
    )
    assert synthetic.spark_purity == pytest.approx(1.0)
    assert synthetic.purity_gap == pytest.approx(0.0)

    kdd99 = by_dataset["kdd99"]
    assert kdd99.spark_adjusted_rand_index < 0
    assert kdd99.traditional_adjusted_rand_index > (
        kdd99.spark_adjusted_rand_index
    )
    assert kdd99.purity_gap < 0
    assert kdd99.adjusted_rand_index_gap < 0
    assert kdd99.normalized_mutual_information_gap < 0


def test_summarize_requires_traditional_row(tmp_path):
    body = (
        "synthetic,parallel_kmeans,local[1],1000,20,5,1032.37,1.0,1.0,"
        "1.0,218;191;206;192;193,1.6055,3,True\n"
    )
    path = _write_csv(tmp_path, body)
    rows = load_quality_comparison_csv(path)

    with pytest.raises(ValueError):
        summarize_quality_comparisons(rows)


def test_summarize_requires_local1_row(tmp_path):
    body = (
        "synthetic,parallel_kmeans,local[2],1000,20,5,1032.37,1.0,1.0,"
        "1.0,218;191;206;192;193,0.9978,3,True\n"
        "synthetic,traditional_kmeans,single_process,1000,20,5,"
        "1032.37,1.0,1.0,1.0,218;191;206;192;193,0.4779,2,True\n"
    )
    path = _write_csv(tmp_path, body)
    rows = load_quality_comparison_csv(path)

    with pytest.raises(ValueError):
        summarize_quality_comparisons(rows)


def test_summarize_without_comparison_master(tmp_path):
    body = (
        "synthetic,parallel_kmeans,local[1],1000,20,5,1032.37,1.0,1.0,"
        "1.0,218;191;206;192;193,1.6055,3,True\n"
        "synthetic,traditional_kmeans,single_process,1000,20,5,"
        "1032.37,1.0,1.0,1.0,218;191;206;192;193,0.4779,2,True\n"
    )
    path = _write_csv(tmp_path, body)
    rows = load_quality_comparison_csv(path)

    summaries = summarize_quality_comparisons(rows)

    assert len(summaries) == 1
    assert summaries[0].spark_comparison_master is None
    assert summaries[0].spark_comparison_master_seconds is None
    assert summaries[0].speedup is None


def test_summarize_rejects_empty_rows():
    with pytest.raises(ValueError):
        summarize_quality_comparisons(())
