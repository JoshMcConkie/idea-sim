from pathlib import Path

import pandas as pd
import pytest

from coverage_planner.experiments import plotting


def _row(
    *,
    agents=1,
    steps=2,
    chunksize=1,
    start_row=0,
    start_col=0,
    method="full_horizon_greedy_solve",
    score=10.0,
    runtime=2.0,
):
    return {
        "agents": agents,
        "steps": steps,
        "chunksize": chunksize,
        "start_row": start_row,
        "start_col": start_col,
        "method": method,
        "score": score,
        "runtime": runtime,
    }


def test_efficiency_summary_calculates_reference_normalized_ratios():
    raw_df = pd.DataFrame([
        _row(start_col=0, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(start_col=0, method="rolling_horizon_greedy_solve", score=12, runtime=2),
        _row(start_col=1, method="full_horizon_greedy_solve", score=8, runtime=4),
        _row(start_col=1, method="rolling_horizon_greedy_solve", score=12, runtime=3),
    ])

    summary = plotting._efficiency_summary(
        raw_df,
        x_axis="agents",
        series_by="method",
    )

    ref = summary[summary["method"] == "full_horizon_greedy_solve"].iloc[0]
    split = summary[summary["method"] == "rolling_horizon_greedy_solve"].iloc[0]

    assert ref["mean"] == pytest.approx(1.0)
    assert ref["min"] == pytest.approx(1.0)
    assert ref["max"] == pytest.approx(1.0)

    assert split["mean"] == pytest.approx(1.6)
    assert split["min"] == pytest.approx(1.2)
    assert split["max"] == pytest.approx(2.0)


def test_efficiency_ratios_drop_nonpositive_reference_runtime():
    raw_df = pd.DataFrame([
        _row(start_col=0, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(start_col=0, method="rolling_horizon_greedy_solve", score=12, runtime=2),
        _row(start_col=1, method="full_horizon_greedy_solve", score=8, runtime=0),
        _row(start_col=1, method="rolling_horizon_greedy_solve", score=12, runtime=3),
    ])

    ratios = plotting._efficiency_ratios(
        raw_df,
        x_axis="agents",
        series_by="method",
    )

    assert set(ratios["start_col"]) == {0}
    split = ratios[ratios["method"] == "rolling_horizon_greedy_solve"].iloc[0]
    assert split["efficiency_ratio"] == pytest.approx(1.2)


def test_efficiency_method_filter_keeps_reference_for_denominator():
    raw_df = pd.DataFrame([
        _row(start_col=0, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(start_col=0, method="rolling_horizon_greedy_solve", score=12, runtime=2),
    ])

    ratios = plotting._efficiency_ratios(
        raw_df,
        x_axis="agents",
        series_by="method",
        filters={"method": "rolling_horizon_greedy_solve"},
    )

    assert list(ratios["method"]) == ["rolling_horizon_greedy_solve"]
    assert ratios.iloc[0]["efficiency_ratio"] == pytest.approx(1.2)


def test_efficiency_rejects_same_axis_and_series():
    raw_df = pd.DataFrame([
        _row(method="full_horizon_greedy_solve"),
        _row(method="rolling_horizon_greedy_solve"),
    ])

    with pytest.raises(ValueError, match="x_axis and series_by"):
        plotting._efficiency_summary(
            raw_df,
            x_axis="agents",
            series_by="agents",
        )


def _pareto_raw_df():
    return pd.DataFrame([
        # chunksize=1: rolling is twice as fast but loses score.
        _row(chunksize=1, start_col=0, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(chunksize=1, start_col=0, method="rolling_horizon_greedy_solve", score=8, runtime=1),
        _row(chunksize=1, start_col=1, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(chunksize=1, start_col=1, method="rolling_horizon_greedy_solve", score=9, runtime=1),
        # chunksize=2 (== steps): rolling matches the reference exactly.
        _row(chunksize=2, start_col=0, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(chunksize=2, start_col=0, method="rolling_horizon_greedy_solve", score=10, runtime=2),
        _row(chunksize=2, start_col=1, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(chunksize=2, start_col=1, method="rolling_horizon_greedy_solve", score=10, runtime=2),
    ])


def test_pareto_summary_aggregates_one_point_per_chunksize():
    summary = plotting._pareto_summary(
        _pareto_raw_df(),
        series_by="agents",
        filters={"method": "rolling_horizon_greedy_solve"},
    )

    assert len(summary) == 2

    chunk1 = summary[summary["chunksize"] == 1].iloc[0]
    assert chunk1["runtime_ratio_mean"] == pytest.approx(0.5)
    assert chunk1["score_ratio_mean"] == pytest.approx(0.85)
    assert chunk1["score_ratio_min"] == pytest.approx(0.8)

    chunk2 = summary[summary["chunksize"] == 2].iloc[0]
    assert chunk2["runtime_ratio_mean"] == pytest.approx(1.0)
    assert chunk2["score_ratio_mean"] == pytest.approx(1.0)
    assert chunk2["score_ratio_min"] == pytest.approx(1.0)


def test_pareto_summary_rejects_chunksize_series():
    with pytest.raises(ValueError, match="chunksize"):
        plotting._pareto_summary(
            _pareto_raw_df(),
            series_by="chunksize",
        )


def test_render_pareto_defaults_method_filter_and_writes_file(tmp_path: Path):
    meta = {"name": "unit", "grid_size": 4}

    out_dir = plotting.render_pareto(
        _pareto_raw_df(),
        meta,
        output_root=tmp_path,
    )

    expected = (
        out_dir
        / (
            "unit__pareto__seriesby_agents"
            "__method_rolling_horizon_greedy_solve__grid_4x4.png"
        )
    )
    assert out_dir == tmp_path / "unit" / "grid_4x4"
    assert expected.is_file()


def test_render_efficiency_lines_writes_expected_file(tmp_path: Path):
    raw_df = pd.DataFrame([
        _row(start_col=0, method="full_horizon_greedy_solve", score=10, runtime=2),
        _row(start_col=0, method="rolling_horizon_greedy_solve", score=12, runtime=2),
        _row(agents=2, start_col=0, method="full_horizon_greedy_solve", score=12, runtime=3),
        _row(agents=2, start_col=0, method="rolling_horizon_greedy_solve", score=18, runtime=3),
    ])
    meta = {"name": "unit", "grid_size": 4}

    out_dir = plotting.render_efficiency_lines(
        raw_df,
        meta,
        output_root=tmp_path,
    )

    expected = (
        out_dir
        / "unit__efficiency__x_agents__seriesby_method__grid_4x4.png"
    )
    assert out_dir == tmp_path / "unit" / "grid_4x4"
    assert expected.is_file()
