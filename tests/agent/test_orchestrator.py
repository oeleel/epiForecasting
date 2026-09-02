"""Integration tests for agent.orchestrator using the FakeLLM/ScriptedPipeline harness.

These tests cover the orchestrator's stop conditions, best-iteration retention,
shadow-mode flow, and SQLite tracker round-trip — without spinning up Ollama
or actually retraining XGBoost. Each test runs in well under a second.

Run with:
    PYTHONPATH=. python tests/agent/test_orchestrator.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow running directly without pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.orchestrator import (
    Orchestrator,
    auto_apply_confirmer,
)
from agent.run_tracker import RunTracker
from tests.agent.fakes import (
    FakeAdapter,
    FakeLLM,
    ScriptedPipeline,
    valid_action,
    valid_diagnosis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_baseline_csv(tmp_dir: Path) -> Path:
    """Write a tiny CSV the FakeAdapter reads as iteration 0 (seq[0])."""
    import pandas as pd
    p = tmp_dir / "baseline.csv"
    pd.DataFrame([{
        "location": "06",
        "forecast_date": "2024-11-09",
        "horizon": 1,
        "forecast": 100.0,
        "_call_index": 0,  # baseline maps to target_sequence[0]
    }]).to_csv(p, index=False)
    return p


_DISTINCT_ACTIONS = [
    valid_action(name="adjust_hyperparameter", params={"name": "max_depth", "value": 5}),
    valid_action(name="adjust_hyperparameter", params={"name": "max_depth", "value": 6}),
    valid_action(name="adjust_hyperparameter", params={"name": "max_depth", "value": 7}),
    valid_action(name="adjust_hyperparameter", params={"name": "max_depth", "value": 8}),
    valid_action(name="adjust_hyperparameter", params={"name": "max_depth", "value": 9}),
]


def _build_llm(num_iterations: int, action_name: str = "adjust_hyperparameter") -> FakeLLM:
    """Build a FakeLLM that responds to N (diagnose, propose) pairs."""
    responses = []
    for i in range(num_iterations):
        responses.append(json.dumps(valid_diagnosis()))
        responses.append(json.dumps(_DISTINCT_ACTIONS[i % len(_DISTINCT_ACTIONS)]))
    return FakeLLM(responses)


def _make_orch(tmp_dir: Path, llm, target_sequence, **kwargs) -> Orchestrator:
    return Orchestrator(
        adapter=FakeAdapter(target_sequence=target_sequence, target_metric="wis"),
        llm=llm,
        pipeline=ScriptedPipeline(),
        tracker=RunTracker(db_path=tmp_dir / "test.db"),
        confirmer=auto_apply_confirmer,
        target_metric="wis",
        run_dir=tmp_dir / "run",
        verbose=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_max_iterations_stop():
    """Loop halts when iteration count reaches max_iterations."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        # Always-improving sequence so no other stop kicks in
        seq = [200.0, 180.0, 160.0, 140.0]
        orch = _make_orch(tmp_dir, _build_llm(3), seq, max_iterations=3)
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        assert result.stop_reason == "max_iterations", result.stop_reason
        assert len(result.iterations) == 4, f"baseline + 3 iters: {len(result.iterations)}"
        assert result.best_iteration == 3, result.best_iteration


def test_agent_stop_action():
    """Loop halts when Agent 2 chooses the stop action."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        # First iter improves, second emits stop
        responses = [
            json.dumps(valid_diagnosis()),
            json.dumps(valid_action()),  # iter 1
            json.dumps(valid_diagnosis()),
            json.dumps(valid_action(name="stop", params={})),  # iter 2 -> stop
        ]
        llm = FakeLLM(responses)
        orch = _make_orch(tmp_dir, llm, [200.0, 150.0], max_iterations=10)
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        assert result.stop_reason == "agent_stop", result.stop_reason
        # baseline + iter1 (applied) + iter2 (stop) = 3 records
        assert len(result.iterations) == 3
        assert result.best_iteration == 1, "iter 1 was the improvement"


def test_two_regressions_stop():
    """Loop halts after two consecutive regressions."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        # Sequence: baseline=100, iter1=90 (better), iter2=110 (regress 1),
        # iter3=120 (regress 2 -> stop)
        seq = [100.0, 90.0, 110.0, 120.0]
        orch = _make_orch(tmp_dir, _build_llm(5), seq, max_iterations=10)
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        assert result.stop_reason == "regressed_x2", result.stop_reason
        # Best is iter 1 (the only improvement), NOT the latest
        assert result.best_iteration == 1
        assert result.iterations[1].target_value == 90.0


def test_best_forecast_retained_on_regression():
    """Even when the loop ends in regression, the best forecast wins."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        seq = [100.0, 80.0, 95.0, 99.0]  # iter1 is best, iter2/3 regress
        orch = _make_orch(tmp_dir, _build_llm(5), seq, max_iterations=10)
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        assert result.best_iteration == 1
        assert result.iterations[1].target_value == 80.0
        # best_forecast.csv should exist and match iter_1.csv
        assert Path(result.best_forecast_path).exists()


def test_no_improvement_x2_stop():
    """Loop halts after two consecutive iterations with <1% improvement."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        # baseline=100, iter1=99.5 (0.5% better), iter2=99.0 (~0.5% better),
        # both below 1% threshold -> stop after iter 2
        seq = [100.0, 99.5, 99.0, 98.5, 98.0]
        orch = _make_orch(
            tmp_dir, _build_llm(5), seq,
            max_iterations=10, no_improvement_threshold=0.01,
        )
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        assert result.stop_reason == "no_improvement_x2", result.stop_reason


def test_shadow_mode_skip():
    """skip decision records the proposal but does not retrain."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        seq = [100.0, 80.0]  # only baseline + 1 retrain would happen if applied
        skips_left = [True, False]  # skip first, accept second

        def confirmer(iteration, action):
            if skips_left and skips_left.pop(0):
                return "skip"
            return "apply"

        orch = Orchestrator(
            adapter=FakeAdapter(target_sequence=seq, target_metric="wis"),
            llm=_build_llm(3),
            pipeline=ScriptedPipeline(),
            tracker=RunTracker(db_path=tmp_dir / "test.db"),
            confirmer=confirmer,
            target_metric="wis",
            max_iterations=2,
            run_dir=tmp_dir / "run",
            verbose=False,
        )
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        # baseline + iter1 (skipped) + iter2 (applied)
        assert len(result.iterations) == 3
        assert result.iterations[1].action_status == "skipped"
        assert result.iterations[2].action_status == "applied"


def test_tracker_persists_full_state():
    """Every iteration is recoverable from SQLite alone."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        db_path = tmp_dir / "test.db"

        seq = [100.0, 90.0, 85.0]
        orch = Orchestrator(
            adapter=FakeAdapter(target_sequence=seq, target_metric="wis"),
            llm=_build_llm(2),
            pipeline=ScriptedPipeline(),
            tracker=RunTracker(db_path=db_path),
            confirmer=auto_apply_confirmer,
            target_metric="wis",
            max_iterations=2,
            run_dir=tmp_dir / "run",
            verbose=False,
        )
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        # Re-open the tracker fresh and verify the full state is recoverable
        tracker2 = RunTracker(db_path=db_path)
        run = tracker2.get_run(result.run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert len(run["iterations"]) == 3
        # Each non-baseline iteration has a recorded action and config
        assert run["iterations"][0]["action"] is None
        assert run["iterations"][1]["action"]["name"] == "adjust_hyperparameter"
        assert run["iterations"][1]["config"]["xgboost"]["max_depth"] == 5
        assert run["iterations"][2]["config"]["xgboost"]["max_depth"] == 6

        # Best iteration query
        best = tracker2.get_best_iteration(result.run_id, metric="wis")
        assert best["iteration"] == 2  # 85 is best
        assert best["wis"] == 85.0


def test_invalid_llm_response_repair():
    """Orchestrator retries once when validation fails, succeeds on second try."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        # First diagnosis is malformed (missing keys), second attempt is valid
        bad_diag = json.dumps({"summary": "broken"})  # missing required keys
        good_diag = json.dumps(valid_diagnosis())
        good_action = json.dumps(valid_action())
        llm = FakeLLM([bad_diag, good_diag, good_action])

        orch = _make_orch(tmp_dir, llm, [100.0, 90.0], max_iterations=1)
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        assert result.stop_reason == "max_iterations"
        assert len(result.iterations) == 2
        assert result.iterations[1].action_status == "applied"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_loop_refines_a_bank_family_through_model_params():
    """End to end with a non-legacy family: catalog, prompt, apply, and retrain all key off model.family."""
    from src.config import get_default_config

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        seed = _seed_baseline_csv(tmp_dir)
        cfg = get_default_config()
        cfg["model"]["family"] = "persistence"
        fake_llm = FakeLLM([
            json.dumps(valid_diagnosis()),
            json.dumps(valid_action(
                name="adjust_hyperparameter",
                params={"name": "residual_window_weeks", "value": 52},
            )),
            json.dumps(valid_diagnosis()),
            json.dumps(valid_action(name="stop", params={})),
        ])
        pipeline = ScriptedPipeline()
        orch = Orchestrator(
            adapter=FakeAdapter(target_sequence=[100.0, 90.0]),
            llm=fake_llm,
            pipeline=pipeline,
            tracker=RunTracker(db_path=tmp_dir / "test.db"),
            confirmer=auto_apply_confirmer,
            max_iterations=3,
            run_dir=tmp_dir / "run",
            verbose=False,
        )
        result = orch.run(
            initial_forecast=str(seed), cutoff_date="2025-12-06",
            initial_config=cfg, regenerate_baseline=False,
        )

    # The retrain received the bank-family config with the param written under model.params
    assert len(pipeline.calls) == 1
    retrain_cfg = pipeline.calls[0]["config"]
    assert retrain_cfg["model"]["family"] == "persistence"
    assert retrain_cfg["model"]["params"] == {"residual_window_weeks": 52}
    assert retrain_cfg["xgboost"] == cfg["xgboost"]  # legacy section untouched

    # Agent 2 saw the family-specific catalog and context, not the XGBoost one
    proposal_prompt = fake_llm.calls[1]
    assert "Model family: persistence" in proposal_prompt
    assert "residual_window_weeks" in proposal_prompt
    assert "toggle_feature" not in proposal_prompt
    assert "XGBoost-based" not in proposal_prompt

    assert result.iterations[1].action_status == "applied"
    assert result.stop_reason == "agent_stop"

ALL_TESTS = [
    test_loop_refines_a_bank_family_through_model_params,
    test_max_iterations_stop,
    test_agent_stop_action,
    test_two_regressions_stop,
    test_best_forecast_retained_on_regression,
    test_no_improvement_x2_stop,
    test_shadow_mode_skip,
    test_tracker_persists_full_state,
    test_invalid_llm_response_repair,
]


def main():
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"{'OK' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
