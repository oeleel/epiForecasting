"""Unit tests for prompt formatters and JSON validators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.adapters.flu_forecast import FluForecastAdapter
from agent.prompt_templates import (
    NO_KNOWN_FACTS_TEXT,
    extract_json_from_response,
    format_action_proposal_prompt,
    format_known_facts_block,
    format_structured_diagnosis_prompt,
    validate_action_proposal,
    validate_diagnosis,
)
from tests.agent.fakes import fake_entry


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


# ---- known facts block + cited_entries (design unit 4) ---------------------

def test_action_prompt_without_bank_omits_known_facts_section_and_cited_entries():
    # known_facts=None means "no bank consulted": the prompt must be the
    # pre-bank prompt, byte for byte, so a --no-knowledge run is a true control.
    catalog = FluForecastAdapter().get_available_actions()
    prompt = format_action_proposal_prompt(
        diagnosis=_valid_diagnosis(),
        history=[],
        action_catalog=catalog,
        domain_context="ctx",
        target_metric="wis",
    )
    assert "## Known facts for this diagnosis" not in prompt
    assert NO_KNOWN_FACTS_TEXT not in prompt
    assert "cited_entries" not in prompt
    assert "Known facts are advisory" not in prompt
    # The spliced fragments collapse without leaving extra blank lines behind.
    assert "Log compression at peak.\n\n## Iteration History (most recent last)\n" in prompt
    assert "\n\n\n" not in prompt
    assert (
        '"expected_effect": "<one sentence: what you expect to happen to wis>"\n}\n\n'
        "Output ONLY the JSON object. No code fences, no commentary.\n"
    ) in prompt


def test_action_prompt_with_bank_but_no_matches_keeps_section_and_schema_line():
    catalog = FluForecastAdapter().get_available_actions()
    prompt = format_action_proposal_prompt(
        diagnosis=_valid_diagnosis(),
        history=[],
        action_catalog=catalog,
        domain_context="ctx",
        target_metric="wis",
        known_facts=format_known_facts_block([]),
    )
    assert "## Known facts for this diagnosis" in prompt
    assert NO_KNOWN_FACTS_TEXT in prompt
    assert '"cited_entries": ["<entry id>", ...]' in prompt
    assert "`cited_entries` is optional" in prompt
    assert "Known facts are advisory" in prompt
    assert "\n\n\n" not in prompt


def test_action_prompt_known_facts_block_sits_between_diagnosis_and_history():
    catalog = FluForecastAdapter().get_available_actions()
    entry = fake_entry("peak-depth-v1", statement="Deeper trees help at peak.")
    block = format_known_facts_block([entry])
    prompt = format_action_proposal_prompt(
        diagnosis=_valid_diagnosis(),
        history=[],
        action_catalog=catalog,
        domain_context="ctx",
        target_metric="wis",
        known_facts=block,
    )
    i_diag = prompt.index("## Diagnosis from Agent 1")
    i_facts = prompt.index("## Known facts for this diagnosis")
    i_hist = prompt.index("## Iteration History")
    assert i_diag < i_facts < i_hist
    assert "- [peak-depth-v1] [C, high] Deeper trees help at peak." in prompt
    assert NO_KNOWN_FACTS_TEXT not in prompt
    assert '"cited_entries": ["<entry id>", ...]' in prompt
    assert "`cited_entries` is optional" in prompt


def test_format_known_facts_block_shows_id_tag_recommendation_and_omitted():
    entries = [
        fake_entry(
            "peak-reweight-v1",
            statement="Up-weight peak rows.",
            action="reweight_training_samples",
            params={"dimension": "phase", "value": "peak", "weight": 2.0},
        ),
        fake_entry("peak-plain-v1", statement="Peak is under-predicted.", confidence="medium"),
    ]
    block = format_known_facts_block(entries, omitted=3)
    lines = block.split("\n")
    assert lines[0] == (
        "- [peak-reweight-v1] [C, high] Up-weight peak rows. "
        "-> try reweight_training_samples(dimension=phase, value=peak, weight=2.0)"
    )
    assert lines[1] == "- [peak-plain-v1] [C, medium] Peak is under-predicted."
    assert lines[2] == "- (3 more matching entries omitted; narrow the retrieval context)"
    assert len(lines) == 3


def test_format_known_facts_block_empty_is_none_retrieved():
    assert format_known_facts_block([]) == NO_KNOWN_FACTS_TEXT


def test_format_known_facts_block_negative_omitted_raises():
    try:
        format_known_facts_block([], omitted=-1)
        assert False
    except ValueError as e:
        assert "omitted" in str(e)


def test_validate_action_cited_entries_absent_passes():
    catalog = FluForecastAdapter().get_available_actions()
    action = _valid_action()
    assert "cited_entries" not in action
    validate_action_proposal(action, catalog)


def test_validate_action_cited_entries_list_of_strings_passes():
    catalog = FluForecastAdapter().get_available_actions()
    action = _valid_action()
    action["cited_entries"] = ["peak-depth-v1", "peak-reweight-v1"]
    validate_action_proposal(action, catalog)
    action["cited_entries"] = []
    validate_action_proposal(action, catalog)


def test_validate_action_cited_entries_not_a_list_raises():
    catalog = FluForecastAdapter().get_available_actions()
    action = _valid_action()
    action["cited_entries"] = "peak-depth-v1"
    try:
        validate_action_proposal(action, catalog)
        assert False
    except ValueError as e:
        assert "cited_entries" in str(e)


def test_validate_action_cited_entries_non_string_item_raises():
    catalog = FluForecastAdapter().get_available_actions()
    action = _valid_action()
    action["cited_entries"] = ["peak-depth-v1", 7]
    try:
        validate_action_proposal(action, catalog)
        assert False
    except ValueError as e:
        assert "cited_entries" in str(e)


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
    test_action_prompt_without_bank_omits_known_facts_section_and_cited_entries,
    test_action_prompt_with_bank_but_no_matches_keeps_section_and_schema_line,
    test_action_prompt_known_facts_block_sits_between_diagnosis_and_history,
    test_format_known_facts_block_shows_id_tag_recommendation_and_omitted,
    test_format_known_facts_block_empty_is_none_retrieved,
    test_format_known_facts_block_negative_omitted_raises,
    test_validate_action_cited_entries_absent_passes,
    test_validate_action_cited_entries_list_of_strings_passes,
    test_validate_action_cited_entries_not_a_list_raises,
    test_validate_action_cited_entries_non_string_item_raises,
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
