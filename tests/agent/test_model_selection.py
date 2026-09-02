"""Unit tests for agent.model_selection (Stage 1 warm-up + incumbent selection)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.model_selection import (
    CandidateResult,
    SelectionGoal,
    evaluate_candidates,
    goal_value,
    render_markdown,
    select_incumbent,
)
from src.config import EVAL_END_DATE, EVAL_START_DATE, generate_eval_cutoffs

SERIES = ["06", "US"]
GOOD = "tests.agent.test_model_selection:GoodModel"
BAD = "tests.agent.test_model_selection:BrokenModel"


def _cdc_frame(n_weeks: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2022-02-05", periods=n_weeks, freq="W-SAT")
    rows = []
    for i, loc in enumerate(SERIES):
        values = 100.0 * (i + 1) + 30.0 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)
        for d, v in zip(dates, values, strict=True):
            rows.append({"date": d, "location": loc, "location_name": loc,
                         "value": float(v), "weekly_rate": 0.0})
    return pd.DataFrame(rows)


from src.model_bank import ForecastModel, quantile_column  # noqa: E402


class GoodModel(ForecastModel):
    family = "test_good"
    description = "persistence with fixed bands"

    def fit(self, history):
        self.is_fitted = True

    def predict(self, history):
        rows = []
        for uid, s in history.groupby("unique_id"):
            last_date, last = s["ds"].max(), float(s["y"].iloc[-1])
            for h in range(1, self.horizon + 1):
                row = {
                    "unique_id": uid,
                    "ds": last_date + pd.Timedelta(7 * h, unit="D"),
                    "horizon": h,
                }
                for lvl in self.quantile_levels:
                    row[quantile_column(lvl)] = max(last + (lvl - 0.5) * 80.0, 0.0)
                rows.append(row)
        return pd.DataFrame(rows)


class BrokenModel(ForecastModel):
    family = "test_broken"
    description = "always fails"

    def fit(self, history):
        raise RuntimeError("boom")

    def predict(self, history):
        raise RuntimeError("unreachable")


def _metrics(wis: float, peak_wis: float = None, coverage: float = 0.9):
    m = {"overall": {"wis": wis, "coverage_95": coverage, "bias": 1.0}}
    if peak_wis is not None:
        m["by_phase"] = {"peak": {"wis": peak_wis}}
    return m


def _cand(family: str, metrics=None, error=None) -> CandidateResult:
    return CandidateResult(family=family, params={}, n_cutoffs=1, n_forecast_rows=1,
                           fit_seconds_total=0.0, metrics=metrics or {}, error=error)


# ---- config split -----------------------------------------------------------

def test_generate_eval_cutoffs_are_saturdays_inside_pinned_window():
    cutoffs = generate_eval_cutoffs()
    dates = pd.to_datetime(cutoffs)
    assert (dates.dayofweek == 5).all()
    assert dates.min() >= pd.Timestamp(EVAL_START_DATE)
    assert dates.max() <= pd.Timestamp(EVAL_END_DATE)
    assert len(generate_eval_cutoffs(stride_weeks=4)) < len(cutoffs)
    with pytest.raises(ValueError):
        generate_eval_cutoffs(stride_weeks=0)


# ---- goal -------------------------------------------------------------------

def test_selection_goal_validates_metric_and_phase():
    assert SelectionGoal().describe() == "wis (overall)"
    assert SelectionGoal("mape", "peak").describe() == "mape (peak phase)"
    with pytest.raises(ValueError):
        SelectionGoal(metric="accuracy")
    with pytest.raises(ValueError):
        SelectionGoal(phase="summer")


def test_goal_value_reads_overall_or_phase():
    m = _metrics(10.0, peak_wis=25.0)
    assert goal_value(m, SelectionGoal()) == 10.0
    assert goal_value(m, SelectionGoal(phase="peak")) == 25.0
    assert goal_value(m, SelectionGoal(phase="onset")) is None


# ---- select_incumbent -------------------------------------------------------

def test_select_incumbent_lower_wis_wins_and_failed_are_skipped():
    cands = [_cand("a", _metrics(12.0)), _cand("b", _metrics(8.0)), _cand("c", error="boom")]
    result = select_incumbent(cands, SelectionGoal(), cutoffs=["2025-10-04"])
    assert result.incumbent == "b"
    assert [r["family"] for r in result.ranking] == ["b", "a"]
    assert result.to_json()["incumbent"] == "b"


def test_select_incumbent_coverage_is_distance_to_95():
    cands = [_cand("a", _metrics(1.0, coverage=0.80)), _cand("b", _metrics(1.0, coverage=0.99))]
    result = select_incumbent(cands, SelectionGoal(metric="coverage_95"))
    assert result.incumbent == "b"


def test_select_incumbent_by_phase_ignores_candidates_without_that_phase():
    cands = [_cand("a", _metrics(5.0)), _cand("b", _metrics(9.0, peak_wis=3.0))]
    result = select_incumbent(cands, SelectionGoal(phase="peak"))
    assert result.incumbent == "b"
    assert len(result.ranking) == 1


def test_select_incumbent_none_when_nothing_scored():
    result = select_incumbent([_cand("a", error="x")])
    assert result.incumbent is None and result.ranking == []


# ---- evaluate_candidates ----------------------------------------------------

def test_evaluate_candidates_rolls_over_cutoffs_and_scores():
    cdc = _cdc_frame()
    cutoffs = ["2025-01-04", "2025-02-01"]
    results = evaluate_candidates([GOOD], cutoffs, actuals=cdc, cdc=cdc)
    assert len(results) == 1
    r = results[0]
    assert r.ok
    assert r.n_cutoffs == 2
    assert r.n_forecast_rows == len(SERIES) * 4 * 2
    assert r.metrics["overall"]["n_forecasts"] == r.n_forecast_rows
    assert r.metrics["overall"]["wis"] >= 0


def test_evaluate_candidates_isolates_a_broken_family():
    cdc = _cdc_frame()
    results = evaluate_candidates([BAD, GOOD], ["2025-01-04"], actuals=cdc, cdc=cdc)
    # Results are keyed by the *requested* family string (a dotted path for
    # in-house models) because that is what --model-family takes next.
    by_family = {r.family: r for r in results}
    assert not by_family[BAD].ok
    assert "boom" in by_family[BAD].error
    assert by_family[GOOD].ok


def test_evaluate_candidates_excludes_locations_from_scoring():
    cdc = _cdc_frame()
    results = evaluate_candidates([GOOD], ["2025-01-04"], actuals=cdc, cdc=cdc,
                                  exclude_locations=["US"])
    assert results[0].n_forecast_rows == 4


def test_evaluate_candidates_rejects_empty_inputs():
    cdc = _cdc_frame()
    with pytest.raises(ValueError):
        evaluate_candidates([], ["2025-01-04"], actuals=cdc, cdc=cdc)
    with pytest.raises(ValueError):
        evaluate_candidates([GOOD], [], actuals=cdc, cdc=cdc)


def test_render_markdown_lists_best_first_and_failures():
    cands = [_cand("a", _metrics(12.0, peak_wis=30.0)), _cand("b", _metrics(8.0)),
             _cand("c", error="boom")]
    result = select_incumbent(cands, SelectionGoal(), cutoffs=["2025-10-04", "2025-11-01"])
    md = render_markdown(result, title="T")
    assert md.startswith("# T")
    assert "incumbent: **b**" in md
    assert md.index("| b (incumbent)") < md.index("| a |")
    assert "- c: boom" in md
    assert "2025-10-04 .. 2025-11-01" in md


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
