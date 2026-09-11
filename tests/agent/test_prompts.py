"""Unit tests for prompt formatters and JSON validators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.model_selection import VALID_METRICS, VALID_PHASES
from agent.prompt_templates import (
    extract_json_from_response,
    format_action_proposal_prompt,
    format_goal_parse_prompt,
    format_structured_diagnosis_prompt,
    validate_action_proposal,
    validate_diagnosis,
    validate_goal,
)


# ---- fixtures --------------------------------------------------------------

def _fake_metrics() -> dict:
    return {
        "overall": {"mape": 38.4, "mae": 145.2, "rmse": 200.1, "bias": 12.3, "n_forecasts": 1500},
        "by_horizon": {
            1: {"mape": 22.1, "mae": 95, "bias": 5.0, "n": 375},
            4: {"mape": 58.2, "mae": 210, "bias": 22.0, "n": 375},
        },
        "by_phase": {
            "peak": {"mape": 55.0, "mae": 220, "bias": 25.0, "n": 500},
        },
        "worst_locations": [{"location": "California", "fips": "06",
                             "mape": 65.0, "mae": 300, "bias": 30.0}],
        "best_locations": [{"location": "Vermont", "fips": "50",
                            "mape": 18.0, "mae": 12, "bias": 1.0}],
        "worst_segments": [{"location": "CA", "date": "2025-01-04", "horizon": 4,
                            "predicted": 800, "actual": 1500, "error": -700.0,
                            "pct_error": -46.7}],
        "date_range": {"min": "2024-11-02", "max": "2025-04-26"},
        "n_locations": 52,
    }


def _valid_diagnosis() -> dict:
    return {
        "summary": "Model under-predicts during peak.",
        "headline_metrics": {"mape": 38.4, "mae": 145.2, "bias": 12.3, "n_forecasts": 1500},
        "weak_segments": [
            {"dimension": "phase", "value": "peak", "metric": "mape",
             "delta_vs_overall": 16.6, "severity": "high"},
        ],
        "hypotheses": [
            {"id": "h1", "text": "Log compression at peak.", "confidence": 0.7},
        ],
        "suggested_focus": "peak_underprediction",
    }


def _valid_action() -> dict:
    return {
        "name": "adjust_hyperparameter",
        "params": {"name": "max_depth", "value": 5},
        "rationale": "Allow deeper trees.",
        "expected_effect": "Reduce peak MAPE.",
    }


# ---- diagnosis prompt + validator -----------------------------------------

def test_diagnosis_prompt_includes_required_phrases():
    prompt = format_structured_diagnosis_prompt(_fake_metrics(), "context")
    assert "JSON" in prompt
    assert "weak_segments" in prompt
    assert "suggested_focus" in prompt


def test_extract_json_bare():
    obj = extract_json_from_response(json.dumps(_valid_diagnosis()))
    assert obj["suggested_focus"] == "peak_underprediction"


def test_extract_json_fenced():
    fenced = "```json\n" + json.dumps(_valid_diagnosis()) + "\n```"
    obj = extract_json_from_response(fenced)
    assert obj["suggested_focus"] == "peak_underprediction"


def test_extract_json_noisy_prose():
    noisy = "Sure, here is the diagnosis:\n\n" + json.dumps(_valid_diagnosis()) + "\n\nThanks!"
    obj = extract_json_from_response(noisy)
    assert obj["suggested_focus"] == "peak_underprediction"


def test_extract_json_qwen3_think_tags():
    """Qwen3 wraps output in <think>...</think> tags — extractor strips them."""
    think_wrapped = (
        "<think>\nLet me analyze the metrics...\n</think>\n\n"
        + json.dumps(_valid_diagnosis())
    )
    obj = extract_json_from_response(think_wrapped)
    assert obj["suggested_focus"] == "peak_underprediction"


def test_extract_json_no_json_raises():
    try:
        extract_json_from_response("I cannot help.")
        assert False
    except ValueError:
        pass


def test_validate_diagnosis_valid_passes():
    validate_diagnosis(_valid_diagnosis())  # should not raise


def test_validate_diagnosis_missing_keys_raises():
    try:
        validate_diagnosis({"summary": "x"})
        assert False
    except ValueError:
        pass


def test_validate_diagnosis_bad_severity_raises():
    bad = _valid_diagnosis()
    bad["weak_segments"][0]["severity"] = "critical"
    try:
        validate_diagnosis(bad)
        assert False
    except ValueError:
        pass


def test_validate_diagnosis_confidence_out_of_range_raises():
    bad = _valid_diagnosis()
    bad["hypotheses"][0]["confidence"] = 1.5
    try:
        validate_diagnosis(bad)
        assert False
    except ValueError:
        pass


def test_validate_diagnosis_empty_weak_segments_allowed():
    healthy = {
        "summary": "All good.",
        "headline_metrics": {"mape": 12.0},
        "weak_segments": [],
        "hypotheses": [{"id": "h1", "text": "no issues", "confidence": 0.9}],
        "suggested_focus": "none",
    }
    validate_diagnosis(healthy)  # should not raise


# ---- action prompt + validator --------------------------------------------

def test_action_prompt_with_empty_history():
    catalog = FluForecastAdapter().get_available_actions()
    prompt = format_action_proposal_prompt(
        diagnosis=_valid_diagnosis(),
        history=[],
        action_catalog=catalog,
        domain_context="ctx",
        target_metric="wis",
    )
    assert "first action" in prompt
    assert "adjust_hyperparameter" in prompt


def test_action_prompt_with_history_renders_actions():
    catalog = FluForecastAdapter().get_available_actions()
    history = [
        {"iteration": 0, "action": None, "metrics": {"wis": 145.0}, "delta_vs_baseline": None},
        {"iteration": 1,
         "action": {"name": "adjust_hyperparameter", "params": {"name": "max_depth", "value": 5}},
         "metrics": {"wis": 138.0}, "delta_vs_baseline": -7.0},
    ]
    prompt = format_action_proposal_prompt(
        diagnosis=_valid_diagnosis(),
        history=history,
        action_catalog=catalog,
        domain_context="ctx",
        target_metric="wis",
    )
    assert "iter 1" in prompt
    assert "adjust_hyperparameter(name=max_depth" in prompt


def test_validate_action_valid_passes():
    catalog = FluForecastAdapter().get_available_actions()
    validate_action_proposal(_valid_action(), catalog)


def test_validate_action_unknown_name_raises():
    catalog = FluForecastAdapter().get_available_actions()
    bad = _valid_action()
    bad["name"] = "fly"
    try:
        validate_action_proposal(bad, catalog)
        assert False
    except ValueError:
        pass


def test_validate_action_missing_keys_raises():
    catalog = FluForecastAdapter().get_available_actions()
    try:
        validate_action_proposal({"name": "stop"}, catalog)
        assert False
    except ValueError:
        pass


def test_validate_action_empty_rationale_raises():
    catalog = FluForecastAdapter().get_available_actions()
    bad = _valid_action()
    bad["rationale"] = ""
    try:
        validate_action_proposal(bad, catalog)
        assert False
    except ValueError:
        pass


def test_validate_action_params_must_be_dict():
    catalog = FluForecastAdapter().get_available_actions()
    bad = _valid_action()
    bad["params"] = "not a dict"
    try:
        validate_action_proposal(bad, catalog)
        assert False
    except ValueError:
        pass


def test_validate_goal_rejects_missing_keys():
    try:
        validate_goal({"metric": "wis"}, VALID_METRICS, VALID_PHASES)
        assert False, "should have raised"
    except ValueError as e:
        assert "phase" in str(e)
    validate_goal(
        {"metric": "wis", "phase": "peak", "rationale": "the sentence said peak"},
        VALID_METRICS,
        VALID_PHASES,
    )


def test_format_goal_parse_prompt_lists_legal_values():
    prompt = format_goal_parse_prompt("best at the peak", VALID_METRICS, VALID_PHASES)
    for metric in VALID_METRICS:
        assert metric in prompt
    for phase in VALID_PHASES:
        assert phase in prompt
    assert "Output ONLY the JSON object" in prompt
    assert "off_season" not in prompt


ALL = [
    test_diagnosis_prompt_includes_required_phrases,
    test_extract_json_bare,
    test_extract_json_fenced,
    test_extract_json_noisy_prose,
    test_extract_json_no_json_raises,
    test_validate_diagnosis_valid_passes,
    test_validate_diagnosis_missing_keys_raises,
    test_validate_diagnosis_bad_severity_raises,
    test_validate_diagnosis_confidence_out_of_range_raises,
    test_validate_diagnosis_empty_weak_segments_allowed,
    test_action_prompt_with_empty_history,
    test_action_prompt_with_history_renders_actions,
    test_validate_action_valid_passes,
    test_validate_action_unknown_name_raises,
    test_validate_action_missing_keys_raises,
    test_validate_action_empty_rationale_raises,
    test_validate_action_params_must_be_dict,
    test_extract_json_qwen3_think_tags,  # was defined but never registered (drift fix)
    test_validate_goal_rejects_missing_keys,
    test_format_goal_parse_prompt_lists_legal_values,
]


def main():
    failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print()
    print("OK" if failed == 0 else f"{failed} FAILED")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
