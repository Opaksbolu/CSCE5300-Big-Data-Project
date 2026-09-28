import argparse
from pathlib import Path

import pytest

from scripts.run_single_master_matrix import (
    build_parser,
    parse_positive_int_csv,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", (1,)),
        ("1000", (1000,)),
        ("1000,10000", (1000, 10000)),
        ("2,4,8", (2, 4, 8)),
        (" 1000 , 10000 ", (1000, 10000)),
    ],
)
def test_parse_positive_int_csv_accepts_valid_values(
    value,
    expected,
):
    assert parse_positive_int_csv(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "0",
        "-1",
        "1000,0",
        "0,1000",
        "1000,-1",
        "abc",
        "1000,abc",
        "1000,,2000",
    ],
)
def test_parse_positive_int_csv_rejects_invalid_values(
    value,
):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_int_csv(value)


def test_parser_requires_master():
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args([])

    assert error.value.code == 2


def test_parser_accepts_single_master_configuration():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--master",
            "local[1]",
            "--record-counts",
            "1000,10000",
            "--partition-counts",
            "2,4",
            "--repetitions",
            "3",
            "--output-directory",
            "results/raw/test",
            "--log-level",
            "ERROR",
        ]
    )

    assert args.master == "local[1]"
    assert args.record_counts == (1000, 10000)
    assert args.partition_counts == (2, 4)
    assert args.repetitions == 3
    assert args.output_directory == Path("results/raw/test")
    assert args.log_level == "ERROR"


def test_parser_uses_expected_defaults():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--master",
            "local[2]",
        ]
    )

    assert args.master == "local[2]"
    assert args.record_counts == (1000,)
    assert args.partition_counts == (4,)
    assert args.repetitions == 1
    assert args.output_directory == Path(
        "results/raw/cross-master"
    )
    assert args.log_level == "WARN"


def test_parser_accepts_warmup_and_aggregation_restarts():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--master",
            "local[4]",
            "--warmup-runs",
            "2",
            "--aggregation-restarts",
            "7",
        ]
    )

    assert args.warmup_runs == 2
    assert args.aggregation_restarts == 7


def test_parser_uses_expected_benchmark_defaults():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--master",
            "local[4]",
        ]
    )

    assert args.warmup_runs == 1
    assert args.aggregation_restarts == 5

def test_run_warmups_executes_requested_number_without_output(
    monkeypatch,
):
    import scripts.run_single_master_matrix as module

    calls = []

    def fake_run_synthetic_experiment(
        spark,
        config,
        *,
        output_path=None,
    ):
        calls.append(
            (
                spark,
                config,
                output_path,
            )
        )
        return object()

    monkeypatch.setattr(
        module,
        "run_synthetic_experiment",
        fake_run_synthetic_experiment,
    )

    spark = object()

    module.run_warmup_experiments(
        spark,
        module.SyntheticExperimentConfig(
            experiment_id="warmup-test",
        ),
        warmup_runs=2,
    )

    assert len(calls) == 2
    assert calls[0][2] is None
    assert calls[1][2] is None


def test_run_warmups_does_not_shift_base_seed(
    monkeypatch,
):
    import scripts.run_single_master_matrix as module

    seeds = []
    experiment_ids = []

    def fake_run_synthetic_experiment(
        spark,
        config,
        *,
        output_path=None,
    ):
        seeds.append(config.random_seed)
        experiment_ids.append(config.experiment_id)
        return object()

    monkeypatch.setattr(
        module,
        "run_synthetic_experiment",
        fake_run_synthetic_experiment,
    )

    base_config = module.SyntheticExperimentConfig(
        experiment_id="cross-master",
        random_seed=42,
    )

    module.run_warmup_experiments(
        object(),
        base_config,
        warmup_runs=2,
    )

    assert seeds == [42, 42]
    assert experiment_ids == [
        "cross-master-warmup-01",
        "cross-master-warmup-02",
    ]

    assert base_config.random_seed == 42
    assert base_config.experiment_id == "cross-master"


def test_run_warmups_allows_zero_runs(
    monkeypatch,
):
    import scripts.run_single_master_matrix as module

    calls = []

    def fake_run_synthetic_experiment(
        spark,
        config,
        *,
        output_path=None,
    ):
        calls.append(config)
        return object()

    monkeypatch.setattr(
        module,
        "run_synthetic_experiment",
        fake_run_synthetic_experiment,
    )

    module.run_warmup_experiments(
        object(),
        module.SyntheticExperimentConfig(
            experiment_id="warmup-test",
        ),
        warmup_runs=0,
    )

    assert calls == []


def test_run_warmups_rejects_negative_runs():
    import scripts.run_single_master_matrix as module

    with pytest.raises(
        ValueError,
        match="warmup_runs must be zero or greater",
    ):
        module.run_warmup_experiments(
            object(),
            module.SyntheticExperimentConfig(
                experiment_id="warmup-test",
            ),
            warmup_runs=-1,
        )


def test_main_runs_warmup_before_measured_matrix(
    monkeypatch,
    tmp_path,
):
    import scripts.run_single_master_matrix as module

    events = []

    class FakeSparkContext:
        master = "local[2]"

    class FakeSpark:
        sparkContext = FakeSparkContext()

    class FakeMatrix:
        cases = ()

    fake_spark = FakeSpark()

    monkeypatch.setattr(
        module,
        "create_spark_session",
        lambda **kwargs: fake_spark,
    )

    monkeypatch.setattr(
        module,
        "stop_spark_session",
        lambda spark: events.append("stop"),
    )

    def fake_warmup(
        spark,
        config,
        *,
        warmup_runs,
    ):
        assert spark is fake_spark
        assert warmup_runs == 1
        events.append("warmup")

    monkeypatch.setattr(
        module,
        "run_warmup_experiments",
        fake_warmup,
    )

    def fake_matrix(
        spark,
        base_config,
        *,
        record_counts,
        partition_counts,
        repetitions,
        output_directory,
    ):
        assert spark is fake_spark
        assert repetitions == 1
        events.append("matrix")
        return FakeMatrix()

    monkeypatch.setattr(
        module,
        "run_synthetic_experiment_matrix",
        fake_matrix,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_single_master_matrix",
            "--master",
            "local[2]",
            "--warmup-runs",
            "1",
            "--output-directory",
            str(tmp_path),
        ],
    )

    assert module.main() == 0

    assert events == [
        "warmup",
        "matrix",
        "stop",
    ]
