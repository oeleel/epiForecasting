"""Unit tests for the named calendar phase schemes (agent.phase_evaluator).

Covers A. Adiga's 2026-10-15 scheme (adiga_2026_10, the default) at every
boundary, the legacy calendar_v1 month map staying exactly as it was, unknown
scheme names failing loudly, training-side by_phase weights labelling each
row's target date under the same scheme as evaluation, every metrics dict being labelled with its scheme,
and every comparison of two metrics dicts refusing to mix schemes.

Run with:
    PYTHONPATH=. python tests/agent/test_phase_scheme.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter
from agent import cli
from agent.knowledge.experiment_import import rows_to_entries
from agent.knowledge.schema import KnowledgeEntry
from agent.knowledge.store import KnowledgeBank
from agent.knowledge.schema import KNOWN_PHASES
from agent.model_selection import CandidateResult, select_incumbent
from agent.orchestrator import IterationRecord, RunResult
from agent.phase_evaluator import (
    ADIGA_OCTOBER_PHASE,
    DEFAULT_PHASE_SCHEME,
    PHASE_SCHEMES,
    PhaseEvaluator,
    _validate_scheme,
    assign_phases,
    phase_order,
)
from agent.phase_segmentation import VALID_CURVE_PHASES
from agent.run_report import build_report, render_markdown
from src.direct_forecast import _compute_sample_weights

ADIGA = "adiga_2026_10"
CALENDAR_V1 = "calendar_v1"

# The month map PhaseEvaluator.PHASE_MAP held before schemes existed, verbatim.
OLD_PHASE_MAP = {
    10: "onset", 11: "onset",
    12: "peak", 1: "peak",
    2: "decline", 3: "decline", 4: "decline",
    5: "off_season", 6: "off_season", 7: "off_season",
    8: "off_season", 9: "off_season",
}
MID_MONTH_DATES = [f"2025-{month:02d}-15" for month in range(1, 13)]
MONTH_START_DATES = [f"2025-{month:02d}-01" for month in range(1, 13)]
MONTH_END_DATES = [
    "2025-01-31", "2024-02-29", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30",
    "2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30", "2025-12-31",
]


# ---- factories --------------------------------------------------------------

def _forecasts() -> pd.DataFrame:
    """Four quantile forecasts, one per adiga_2026_10 scored phase."""
    dates = ["2025-11-15", "2026-01-17", "2026-03-14", "2026-05-02"]
    return pd.DataFrame({
        "location": ["06"] * 4,
        "forecast_date": dates,
        "horizon": [1] * 4,
        "forecast": [100.0] * 4,
        "predicted_q05": [80.0] * 4,
        "predicted_q25": [90.0] * 4,
        "predicted_q50": [100.0] * 4,
        "predicted_q75": [110.0] * 4,
        "predicted_q95": [120.0] * 4,
    })


def _actuals() -> pd.DataFrame:
    dates = ["2025-11-15", "2026-01-17", "2026-03-14", "2026-05-02"]
    return pd.DataFrame({
        "date": dates,
        "location": ["06"] * 4,
        "value": [95.0, 130.0, 105.0, 90.0],
        "location_name": ["California"] * 4,
    })


def _adapter() -> FluForecastAdapter:
    adapter = FluForecastAdapter()
    adapter._location_names = {}  # skip the CDC cache lookup; names come from actuals
    return adapter


def _report_metrics(wis: float, phase_scheme: str | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "overall": {"mape": 30.0, "bias": -5.0, "n_forecasts": 212, "wis": wis,
                    "coverage_95": 0.9},
        "by_phase": {"peak": {"mape": 30.0, "bias": -5.0, "n": 53, "wis": wis}},
        "by_horizon": {},
        "date_range": {"min": "2026-01-31", "max": "2026-02-21"},
        "n_locations": 53,
    }
    if phase_scheme is not None:
        metrics["phase_scheme"] = phase_scheme
    return metrics


def _run_result(baseline_metrics: dict[str, Any], best_metrics: dict[str, Any]) -> RunResult:
    config = {"model": {"family": "xgboost_direct"}, "data": {"cutoff_date": "2026-01-24"}}
    records = [
        IterationRecord(iteration=0, config=config, forecast_path="iter_0.csv",
                        metrics=baseline_metrics, action_status="baseline",
                        target_value=baseline_metrics["overall"]["wis"]),
        IterationRecord(iteration=1, config=config, forecast_path="iter_1.csv",
                        metrics=best_metrics,
                        action={"name": "adjust_hyperparameter",
                                "params": {"name": "max_depth", "value": 4}},
                        action_status="applied",
                        target_value=best_metrics["overall"]["wis"]),
    ]
    return RunResult(run_id="r", iterations=records, best_iteration=1, best_forecast_path="",
                     stop_reason="max_iterations", target_metric="wis")


def _candidate(family: str, phase_scheme: str | None) -> CandidateResult:
    metrics: dict[str, Any] = {"overall": {"wis": 10.0}, "by_phase": {"peak": {"wis": 20.0}}}
    if phase_scheme is not None:
        metrics["phase_scheme"] = phase_scheme
    return CandidateResult(family=family, params={}, n_cutoffs=1, n_forecast_rows=1,
                           fit_seconds_total=0.0, metrics=metrics)


_SPLIT = {"train_start_date": "2022-02-05", "eval_start_date": "2025-10-01",
          "eval_end_date": "2026-05-31", "stride_weeks": 4}


def _harness_row(config_id: str, delta: dict[str, Any], wis: float,
                 phase_scheme: str | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "by_phase": {"peak": {"wis": wis, "bias": -50.0, "coverage_95": 0.85}},
    }
    if phase_scheme is not None:
        metrics["phase_scheme"] = phase_scheme
    return {
        "run_at": "2026-10-15T00:00:00+00:00",
        "config_id": config_id,
        "config_delta": delta,
        "state": {"model": "xgboost_direct", "target_phase": "peak", "metric": "wis",
                  "split": _SPLIT},
        "metrics": metrics,
        "reward": {"metric": "wis", "phase": "peak", "baseline_value": 121.0, "value": wis,
                   "delta": round(wis - 121.0, 2), "better": wis < 121.0},
    }


def _lambda_delta() -> dict[str, Any]:
    return {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}}


# ---- adiga_2026_10 boundaries -----------------------------------------------

def test_phase_scheme_default_is_adiga():
    assert DEFAULT_PHASE_SCHEME == ADIGA
    assert PhaseEvaluator.assign_phase("2026-01-10") == "peak"


def test_phase_scheme_adiga_growth_starts_nov_1():
    assert PhaseEvaluator.assign_phase("2025-10-31", ADIGA) == ADIGA_OCTOBER_PHASE
    assert PhaseEvaluator.assign_phase("2025-11-01", ADIGA) == "growth"


def test_phase_scheme_adiga_october_is_off_season_assumption():
    assert ADIGA_OCTOBER_PHASE == "off_season"
    assert PhaseEvaluator.assign_phase("2025-10-01", ADIGA) == "off_season"


def test_phase_scheme_adiga_peak_starts_jan_1():
    assert PhaseEvaluator.assign_phase("2025-12-31", ADIGA) == "growth"
    assert PhaseEvaluator.assign_phase("2026-01-01", ADIGA) == "peak"


def test_phase_scheme_adiga_peak_runs_through_end_of_february():
    assert PhaseEvaluator.assign_phase("2025-02-28", ADIGA) == "peak"
    assert PhaseEvaluator.assign_phase("2024-02-29", ADIGA) == "peak"
    assert PhaseEvaluator.assign_phase("2025-03-01", ADIGA) == "decline"
    assert PhaseEvaluator.assign_phase("2024-03-01", ADIGA) == "decline"


def test_phase_scheme_adiga_plateau_starts_apr_16():
    assert PhaseEvaluator.assign_phase("2026-04-15", ADIGA) == "decline"
    assert PhaseEvaluator.assign_phase("2026-04-16", ADIGA) == "plateau"


def test_phase_scheme_adiga_off_season_starts_jun_1():
    assert PhaseEvaluator.assign_phase("2026-05-31", ADIGA) == "plateau"
    assert PhaseEvaluator.assign_phase("2026-06-01", ADIGA) == "off_season"


def test_phase_scheme_adiga_phase_order_is_season_order():
    assert phase_order(ADIGA) == ("growth", "peak", "decline", "plateau")


# ---- calendar_v1 unchanged --------------------------------------------------

def test_phase_scheme_calendar_v1_matches_old_month_map_mid_month():
    expected = [OLD_PHASE_MAP[month] for month in range(1, 13)]

    assert assign_phases(MID_MONTH_DATES, CALENDAR_V1).tolist() == expected


def test_phase_scheme_calendar_v1_matches_old_month_map_at_month_edges():
    expected = [OLD_PHASE_MAP[month] for month in range(1, 13)]

    assert assign_phases(MONTH_START_DATES, CALENDAR_V1).tolist() == expected
    assert assign_phases(MONTH_END_DATES, CALENDAR_V1).tolist() == expected


def test_phase_scheme_calendar_v1_phase_order_unchanged():
    assert phase_order(CALENDAR_V1) == ("onset", "peak", "decline")


# ---- unknown / malformed schemes --------------------------------------------

def test_phase_scheme_unknown_name_raises_naming_it():
    with pytest.raises(ValueError, match="calendar_v9"):
        PhaseEvaluator.assign_phase("2026-01-10", "calendar_v9")


def test_phase_scheme_unknown_name_raises_in_merge():
    with pytest.raises(ValueError, match="nope"):
        PhaseEvaluator.merge_forecasts_actuals(_forecasts(), _actuals(), phase_scheme="nope")


def test_phase_scheme_gap_in_windows_is_rejected():
    gapped = PHASE_SCHEMES[CALENDAR_V1][:-1]  # drop December

    with pytest.raises(ValueError, match="12-01"):
        _validate_scheme(CALENDAR_V1, gapped)


def test_phase_scheme_every_scheme_phase_is_in_knowledge_vocab():
    scheme_phases = {phase for windows in PHASE_SCHEMES.values() for _, _, phase in windows}

    assert scheme_phases <= set(KNOWN_PHASES)
    assert set(VALID_CURVE_PHASES) <= set(KNOWN_PHASES)


# ---- training labels match evaluation labels --------------------------------

def test_phase_scheme_by_phase_weights_follow_default_scheme():
    meta = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-15", "2025-02-10", "2025-04-20"]),
        "location": ["06", "06", "06"],
        "horizon": [1, 1, 1],
    })

    weights = _compute_sample_weights(meta, {"by_phase": {"peak": 2.0, "plateau": 3.0}})

    # Target dates (origin + 1 week): Dec 22 = growth (calendar_v1 would say
    # peak), Feb 17 = peak (v1: decline), Apr 27 = plateau (v1: decline).
    assert weights.tolist() == [1.0, 2.0, 3.0]


def test_phase_scheme_by_phase_weights_label_the_target_date_like_the_evaluator():
    # Same origin, two horizons: the evaluator labels each forecast by its
    # target date, Dec 27 (growth) at h1 and Jan 3 (peak) at h2.
    meta = pd.DataFrame({
        "date": pd.to_datetime(["2025-12-20", "2025-12-20"]),
        "location": ["06", "06"],
        "horizon": [1, 2],
    })

    weights = _compute_sample_weights(meta, {"by_phase": {"peak": 2.0}})

    assert PhaseEvaluator.assign_phase("2025-12-27") == "growth"
    assert PhaseEvaluator.assign_phase("2026-01-03") == "peak"
    assert weights.tolist() == [1.0, 2.0]


def test_phase_scheme_by_phase_weights_reject_legacy_onset():
    meta = pd.DataFrame({
        "date": pd.to_datetime(["2024-10-15"]),
        "location": ["06"],
        "horizon": [1],
    })

    with pytest.raises(ValueError, match="'onset'.*calendar_v1"):
        _compute_sample_weights(meta, {"by_phase": {"onset": 3.0}})


def test_phase_scheme_by_phase_weights_reject_unknown_phase():
    meta = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-10"]),
        "location": ["06"],
        "horizon": [1],
    })

    with pytest.raises(ValueError, match="peka"):
        _compute_sample_weights(meta, {"by_phase": {"peka": 3.0}})


def test_phase_scheme_by_phase_weights_need_horizon_column():
    meta = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-10"]),
        "location": ["06"],
    })

    with pytest.raises(ValueError, match="'horizon'"):
        _compute_sample_weights(meta, {"by_phase": {"peak": 3.0}})


def test_phase_scheme_reweight_guardrail_uses_default_phases():
    catalog = {a["name"]: a for a in _adapter().get_available_actions()}

    phase_values = catalog["reweight_training_samples"]["guardrails"]["phase_values"]

    assert phase_values == ["growth", "peak", "decline", "plateau"]


# ---- metrics are labelled ---------------------------------------------------

def test_phase_scheme_adapter_metrics_record_scheme():
    metrics = _adapter().compute_metrics(_forecasts(), _actuals())

    assert metrics["phase_scheme"] == ADIGA
    assert sorted(metrics["by_phase"]) == ["decline", "growth", "peak", "plateau"]


# ---- cross-scheme comparisons are refused -----------------------------------

def test_phase_scheme_run_report_withholds_improvement_across_schemes():
    result = _run_result(_report_metrics(100.0, None), _report_metrics(80.0, ADIGA))

    report = build_report(result)
    md = render_markdown(report)

    assert report.phase_schemes_match is False
    assert report.baseline_phase_scheme == CALENDAR_V1
    assert report.phase_scheme == ADIGA
    assert report.improvement_frac is None
    assert "(not comparable: phase schemes differ)" in md
    assert "phase table withheld" in md
    assert "Recommendation: inconclusive" in md
    assert "adopt iteration" not in md


def test_phase_scheme_run_report_missing_key_reads_as_calendar_v1():
    result = _run_result(_report_metrics(100.0, None), _report_metrics(80.0, CALENDAR_V1))

    report = build_report(result)

    assert report.phase_schemes_match is True
    assert report.improvement_frac is not None
    assert "- phase scheme: calendar_v1" in render_markdown(report)


def test_phase_scheme_model_selection_refuses_mixed_schemes():
    candidates = [_candidate("a", ADIGA), _candidate("b", None)]

    with pytest.raises(ValueError, match="different phase schemes"):
        select_incumbent(candidates)


def test_phase_scheme_model_selection_ranks_single_scheme():
    candidates = [_candidate("a", ADIGA), _candidate("b", ADIGA)]

    result = select_incumbent(candidates)

    assert result.incumbent == "a"


# ---- experiment import ------------------------------------------------------

def test_phase_scheme_import_marks_legacy_rows_calendar_v1():
    rows = [_harness_row("baseline", {}, 121.0, None),
            _harness_row("lambda_3", _lambda_delta(), 118.0, None)]

    entries = rows_to_entries(rows, created_at="2026-10-15")

    assert entries[0].id == "exp-lambda-3-2025"
    assert entries[0].payload["phase_scheme"] == CALENDAR_V1
    assert entries[0].statement.startswith("[phase scheme calendar_v1] ")


def _bank_with_one_unlabelled_experiential_entry(directory: Path) -> KnowledgeBank:
    rows = [_harness_row("baseline", {}, 121.0, None),
            _harness_row("lambda_3", _lambda_delta(), 118.0, None)]
    labelled = rows_to_entries(rows, created_at="2026-10-15")[0]
    unlabelled = KnowledgeEntry.from_dict({
        **labelled.to_dict(),
        "id": "exp-pre-scheme-2025",
        "payload": {k: v for k, v in labelled.payload.items() if k != "phase_scheme"},
    })
    bank = KnowledgeBank(directory / "k.db")
    bank.upsert(labelled)
    bank.upsert(unlabelled)
    return bank


def test_phase_scheme_rebuild_warns_on_experiential_entries_without_scheme():
    stderr = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        bank = _bank_with_one_unlabelled_experiential_entry(Path(tmp))
        with mock.patch.object(cli, "_open_bank", return_value=bank), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            cli._knowledge_rebuild()

    assert "1 experiential entries carry no payload.phase_scheme (exp-pre-scheme-2025)" in stderr.getvalue()
    assert "import-experiment" in stderr.getvalue()


def test_phase_scheme_import_new_scheme_gets_its_own_id():
    rows = [_harness_row("baseline", {}, 121.0, ADIGA),
            _harness_row("lambda_3", _lambda_delta(), 118.0, ADIGA)]

    entries = rows_to_entries(rows, created_at="2026-10-15")

    assert entries[0].id == "exp-lambda-3-2025-adiga-2026-10"
    assert entries[0].payload["phase_scheme"] == ADIGA


def test_phase_scheme_import_row_and_baseline_schemes_must_match():
    rows = [_harness_row("baseline", {}, 121.0, None),
            _harness_row("lambda_3", _lambda_delta(), 118.0, ADIGA)]

    with pytest.raises(ValueError, match="phase scheme"):
        rows_to_entries(rows, created_at="2026-10-15")


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main() -> None:
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001 - the fallback runner reports every failure
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
