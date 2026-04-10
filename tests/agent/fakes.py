"""Fake LLM client + scripted pipeline runner for orchestrator tests.

These fakes implement the same `invoke()` / callable signatures as the
real LLMClient and pipeline runner, but produce deterministic, scriptable
output. They let us exercise the full orchestrator loop in milliseconds
without spinning up Ollama or actually retraining XGBoost.

Usage in tests:

    fake_llm = FakeLLM([
        json.dumps(diag1), json.dumps(action1),
        json.dumps(diag2), json.dumps(action2),
    ])
    fake_pipeline = ScriptedPipeline(target_sequence=[145.0, 130.0, 138.0])
    fake_adapter = FakeAdapter(target_sequence=[145.0, 130.0, 138.0])

    orch = Orchestrator(
        adapter=fake_adapter,
        llm=fake_llm,
        pipeline=fake_pipeline,
        tracker=tracker,
        confirmer=auto_apply_confirmer,
        max_iterations=5,
    )
    result = orch.run(initial_forecast="dummy.csv", cutoff_date="2024-11-02")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from agent.domain_adapter import DomainAdapter
from agent.adapters.flu_forecast import FluForecastAdapter


# ----------------------------------------------------------------------------
# FakeLLM
# ----------------------------------------------------------------------------

class FakeLLM:
    """LLM client that returns canned responses in order.

    Pass a list of strings — each `invoke()` call pops the next one.
    Use `valid_diagnosis()` and `valid_action()` helpers below to build
    well-formed JSON responses without hand-typing schemas in tests.
    """

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls: List[str] = []  # full prompts seen, for assertions

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("FakeLLM ran out of canned responses")
        return self._responses.pop(0)

    def is_available(self) -> bool:
        return True


def valid_diagnosis(
    summary: str = "Model shows peak under-prediction.",
    headline: Optional[Dict[str, float]] = None,
    weak_segments: Optional[List[Dict[str, Any]]] = None,
    hypotheses: Optional[List[Dict[str, Any]]] = None,
    suggested_focus: str = "peak_underprediction",
) -> Dict[str, Any]:
    """Build a schema-valid diagnosis dict for tests."""
    return {
        "summary": summary,
        "headline_metrics": headline or {"mape": 38.0, "mae": 100.0, "bias": 5.0, "n_forecasts": 100},
        "weak_segments": weak_segments or [
            {
                "dimension": "phase",
                "value": "peak",
                "metric": "mape",
                "delta_vs_overall": 0.4,
                "severity": "high",
            }
        ],
        "hypotheses": hypotheses or [
            {"id": "h1", "text": "Log compression at peak.", "confidence": 0.7}
        ],
        "suggested_focus": suggested_focus,
    }


def valid_action(
    name: str = "adjust_hyperparameter",
    params: Optional[Dict[str, Any]] = None,
    rationale: str = "Test rationale",
    expected_effect: str = "Test effect",
) -> Dict[str, Any]:
    """Build a schema-valid action proposal dict for tests."""
    if params is None:
        if name == "adjust_hyperparameter":
            params = {"name": "max_depth", "value": 5}
        elif name == "reweight_training_samples":
            params = {"dimension": "phase", "value": "peak", "weight": 2.0}
        elif name == "toggle_feature":
            params = {"feature_group": "yoy", "enabled": False}
        elif name == "adjust_floor_constraint":
            params = {"floor_pct": 0.4}
        elif name == "change_target_transform":
            params = {"transform": "raw"}
        else:
            params = {}
    return {
        "name": name,
        "params": params,
        "rationale": rationale,
        "expected_effect": expected_effect,
    }


# ----------------------------------------------------------------------------
# ScriptedPipeline
# ----------------------------------------------------------------------------

class ScriptedPipeline:
    """Pipeline runner that writes a tiny CSV per iteration.

    Each call writes a forecast CSV containing exactly enough columns for
    the FakeAdapter to compute a metric value. Doesn't run XGBoost.

    The CSV's content is dependent on which call this is — a global counter
    is bumped each call so that successive iterations produce different
    forecasts (which the FakeAdapter then maps to the next target value).
    """

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        config: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> str:
        self.calls.append({"config": config, "output_path": output_path})
        out = Path(output_path) if output_path else Path(f"/tmp/scripted_{len(self.calls)}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Bake the call index into the file so the FakeAdapter can read it
        # back. Baseline CSV uses _call_index=0; pipeline calls produce 1,2,...
        df = pd.DataFrame([
            {
                "location": "06",
                "forecast_date": "2024-11-09",
                "horizon": 1,
                "forecast": 100.0,
                "_call_index": len(self.calls),  # 1-based, matches seq[idx]
            }
        ])
        df.to_csv(out, index=False)
        return str(out)


# ----------------------------------------------------------------------------
# FakeAdapter
# ----------------------------------------------------------------------------

class FakeAdapter(DomainAdapter):
    """Adapter that ignores the forecast file and returns scripted metrics.

    Each call to compute_metrics() returns the next entry in target_sequence
    (or repeats the last one if the sequence is exhausted). This is how a
    test scripts a "first iteration improves, second regresses, third
    regresses again" pattern.

    Note: this fake also implements get_available_actions() and apply_action()
    by delegating to the real FluForecastAdapter, since they're already pure
    functions over a config dict — no reason to mock them.
    """

    def __init__(self, target_sequence: List[float], target_metric: str = "wis"):
        self._sequence = list(target_sequence)
        self._call_idx = 0
        self.target_metric = target_metric
        self._real = FluForecastAdapter()
        self.compute_calls: List[str] = []

    # ---- Required DomainAdapter methods --------------------------------

    def load_data(self, config: Dict[str, Any]):
        # Read the marker out of the CSV so we can index into target_sequence
        path = config.get("forecast_csv")
        df = pd.read_csv(path)
        return df, df  # actuals unused by FakeAdapter

    def compute_metrics(self, forecasts, actuals) -> Dict[str, Any]:
        # Use the call index from the CSV marker if present, else our counter.
        # baseline.csv has _call_index=0 -> seq[0]
        # ScriptedPipeline calls produce _call_index=1,2,... -> seq[1], seq[2], ...
        if "_call_index" in forecasts.columns:
            idx = int(forecasts["_call_index"].iloc[0])
        else:
            idx = self._call_idx
        idx = min(idx, len(self._sequence) - 1)
        target_val = float(self._sequence[idx])
        self._call_idx += 1
        self.compute_calls.append(str(forecasts.iloc[0].to_dict()))
        return {
            "overall": {
                self.target_metric: target_val,
                "mape": target_val,  # for prompt formatter
                "mae": target_val,
                "rmse": target_val * 1.2,
                "bias": 0.0,
                "n_forecasts": 100,
            },
            "by_horizon": {1: {"mape": target_val, "mae": target_val, "bias": 0.0, "n": 25}},
            "by_phase": {
                "peak": {"mape": target_val * 1.5, "mae": target_val, "bias": 5.0, "n": 30}
            },
            "worst_locations": [],
            "best_locations": [],
            "worst_segments": [],
            "date_range": {"min": "2024-11-02", "max": "2025-04-26"},
            "n_locations": 1,
        }

    def get_domain_context(self) -> str:
        return "Fake domain for tests."

    def get_available_actions(self):
        return self._real.get_available_actions()

    def apply_action(self, action, config=None):
        return self._real.apply_action(action, config)
