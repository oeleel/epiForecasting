"""Unit tests for agent.goal_parser (English sentence -> SelectionGoal)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.goal_parser import parse_goal
from agent.model_selection import DEFAULT_GOAL, SelectionGoal
from tests.agent.fakes import FakeLLM

# A sentence containing no METRIC_KEYWORDS and no PHASE_KEYWORDS substring, so
# the LLM rungs of the ladder are actually reached.
NO_KEYWORDS = "make the numbers less wrong"


# ---- factories -------------------------------------------------------------

def _llm(*responses: str) -> FakeLLM:
    return FakeLLM(list(responses))


def _goal_json(metric: str = "wis", phase: str = "peak", rationale: str = "because") -> str:
    return json.dumps({"metric": metric, "phase": phase, "rationale": rationale})


class _BoomLLM:
    """An LLM handle whose transport is down."""

    def invoke(self, prompt: str) -> str:
        raise ConnectionError("Could not connect to LLM server at http://localhost:11434/v1")

    def is_available(self) -> bool:
        return False


# ---- tests -----------------------------------------------------------------

def test_keyword_resolves_peak_without_llm():
    parsed = parse_goal("which model is best at the peak", llm=None)
    assert parsed.goal == SelectionGoal(metric="wis", phase="peak")
    assert parsed.source == "keyword"


def test_keyword_maps_calibration_to_coverage_95():
    parsed = parse_goal("I care about calibration overall", llm=None)
    assert parsed.goal == SelectionGoal(metric="coverage_95", phase="all")
    assert parsed.source == "keyword"


def test_llm_not_called_when_keywords_resolve_both():
    llm = _llm("junk")
    parsed = parse_goal("minimize wis at the peak", llm=llm)
    assert llm.calls == []
    assert parsed.source == "keyword"
    assert parsed.goal == SelectionGoal(metric="wis", phase="peak")


def test_llm_path_parses_valid_json():
    llm = _llm(_goal_json())
    parsed = parse_goal(NO_KEYWORDS, llm=llm)
    assert parsed.source == "llm"
    assert parsed.rationale == "because"
    assert len(llm.calls) == 1
    assert parsed.goal == SelectionGoal(metric="wis", phase="peak")


def test_llm_repair_retry_on_first_junk():
    llm = _llm("I think you want WIS.", _goal_json())
    parsed = parse_goal(NO_KEYWORDS, llm=llm)
    assert parsed.source == "llm_repair"
    assert len(llm.calls) == 2
    assert "Previous attempt failed validation" in llm.calls[1]


def test_llm_second_failure_falls_back_without_raising():
    llm = _llm("junk", "still junk")
    parsed = parse_goal(NO_KEYWORDS, llm=llm)
    assert parsed.goal == DEFAULT_GOAL
    assert parsed.source == "fallback"
    assert parsed.warnings


def test_connection_error_falls_back():
    parsed = parse_goal(NO_KEYWORDS, llm=_BoomLLM())
    assert parsed.source == "fallback"
    assert parsed.goal == DEFAULT_GOAL
    assert any("LLM" in w for w in parsed.warnings)


def test_out_of_enum_metric_triggers_repair_then_falls_back():
    llm = _llm(_goal_json(metric="crps"), _goal_json(metric="crps"))
    parsed = parse_goal(NO_KEYWORDS, llm=llm)
    assert parsed.source == "fallback"
    assert len(llm.calls) == 2
    assert "crps" in llm.calls[1]


def test_keyword_overrides_llm_disagreement():
    llm = _llm(_goal_json(metric="mape", phase="all"))
    parsed = parse_goal("how well does it do at the peak", llm=llm)
    assert parsed.goal.phase == "peak"
    assert parsed.goal.metric == "mape"
    assert parsed.warnings


def test_off_season_request_raises():
    try:
        parse_goal("best in the summer off-season", llm=None)
        assert False, "should have raised"
    except ValueError as e:
        assert "onset" in str(e)
        assert "decline" in str(e)


def test_rmse_with_phase_is_rejected():
    try:
        parse_goal("rmse at the peak", llm=None)
        assert False, "should have raised"
    except ValueError as e:
        assert "rmse" in str(e)


def test_empty_goal_text_raises():
    try:
        parse_goal("   ")
        assert False, "should have raised"
    except ValueError:
        pass


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
