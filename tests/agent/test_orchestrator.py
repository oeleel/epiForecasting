"""Integration tests for agent.orchestrator using the FakeLLM/ScriptedPipeline harness.

These tests cover the orchestrator's stop conditions, best-iteration retention,
shadow-mode flow, and SQLite tracker round-trip — without spinning up Ollama
or actually retraining XGBoost. Each test runs in well under a second.

Run with:
    PYTHONPATH=. python tests/agent/test_orchestrator.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

# Allow running directly without pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.orchestrator import (
    DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS,
    DEFAULT_PROPOSAL_FACTS_LIMIT,
    Orchestrator,
    auto_apply_confirmer,
)
from agent.run_tracker import RunTracker
from tests.agent.fakes import (
    FakeAdapter,
    FakeKnowledgeBank,
    FakeLLM,
    ScriptedPipeline,
    fake_entry,
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

def test_run_writes_report_into_run_dir():
    """A completed run leaves report.md + report.json next to its forecasts (roadmap 6.1)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)

        seq = [200.0, 180.0, 160.0]
        orch = _make_orch(tmp_dir, _build_llm(2), seq, max_iterations=2)
        result = orch.run(
            initial_forecast=str(baseline),
            cutoff_date="2024-11-02",
            regenerate_baseline=False,
        )

        md_path = tmp_dir / "run" / "report.md"
        json_path = tmp_dir / "run" / "report.json"
        assert md_path.exists(), "orchestrator must write report.md into the run dir"
        assert json_path.exists(), "orchestrator must write report.json into the run dir"

        payload = json.loads(json_path.read_text())
        assert payload["run_id"] == result.run_id
        assert payload["stop_reason"] == result.stop_reason
        assert payload["source"] == "run_result"
        assert md_path.read_text().startswith("# ")


# ---------------------------------------------------------------------------
# Knowledge bank at the proposal step (design unit 4)
# ---------------------------------------------------------------------------

def _peak_bank() -> FakeKnowledgeBank:
    """Two peak facts (one recommending adjust_hyperparameter) and one decline fact."""
    return FakeKnowledgeBank([
        fake_entry(
            "peak-depth-v1", statement="Deeper trees help at peak.", phase="peak",
            action="adjust_hyperparameter", params={"name": "max_depth", "value": 6},
        ),
        fake_entry(
            "peak-reweight-v1", statement="Up-weight peak rows.", phase="peak",
            action="reweight_training_samples",
            params={"dimension": "phase", "value": "peak", "weight": 2.0},
        ),
        fake_entry("decline-floor-v1", statement="Relax the floor in decline.", phase="decline"),
    ])


def _llm_citing(cited, action_name="adjust_hyperparameter", params=None) -> FakeLLM:
    """One (diagnose, propose) pair whose proposal cites `cited`."""
    params = params or {"name": "max_depth", "value": 5}
    return FakeLLM([
        json.dumps(valid_diagnosis()),  # weak segment: phase=peak
        json.dumps(valid_action(name=action_name, params=params, cited_entries=cited)),
    ])


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_knowledge_bank_retrieval_uses_diagnosed_phase_and_shows_ids():
    """The retrieval context is phase=peak/model/metric; the prompt shows only the peak ids."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        bank = _peak_bank()
        llm = _llm_citing(["peak-depth-v1"])
        orch = _make_orch(tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=bank)

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        assert len(bank.queries) == 1
        ctx = bank.queries[0]
        assert ctx.phase == "peak"
        assert ctx.model == "xgboost_direct"
        assert ctx.metric == "wis"

        proposal_prompt = llm.calls[1]
        assert "## Known facts for this diagnosis" in proposal_prompt
        assert "- [peak-depth-v1] [C, high] Deeper trees help at peak." in proposal_prompt
        assert "[peak-reweight-v1]" in proposal_prompt
        assert "decline-floor-v1" not in proposal_prompt

        action = result.iterations[1].action
        assert action["retrieved_entry_ids"] == ["peak-depth-v1", "peak-reweight-v1"]
        assert action["cited_entries"] == ["peak-depth-v1"]
        assert result.iterations[1].action_status == "applied"


def test_knowledge_bank_supported_by_matches_recommended_action():
    """supported_by lists the retrieved entries whose recommendation names the proposed action."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        orch = _make_orch(
            tmp_dir, _llm_citing([]), [200.0, 180.0], max_iterations=1, knowledge_bank=_peak_bank(),
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        action = result.iterations[1].action
        assert action["supported_by"] == ["peak-depth-v1"]
        assert action["cited_entries"] == []
        # The entry recommends max_depth=6, the proposal says 5: same action,
        # different value, so the mismatch list names it too.
        assert action["supported_by_params_mismatch"] == ["peak-depth-v1"]


def test_knowledge_bank_params_mismatch_empty_when_proposal_matches_recommended_value():
    """supported_by_params_mismatch is [] when the proposal uses the recommended params."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        llm = _llm_citing(["peak-depth-v1"], params={"name": "max_depth", "value": 6})
        orch = _make_orch(
            tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=_peak_bank(),
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        action = result.iterations[1].action
        assert action["params"] == {"name": "max_depth", "value": 6}
        assert action["supported_by"] == ["peak-depth-v1"]
        assert action["supported_by_params_mismatch"] == []


def test_knowledge_bank_params_mismatch_lists_supporting_entry_with_other_value():
    """An entry recommending the same action with a different value is recorded as a mismatch."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        bank = FakeKnowledgeBank([
            fake_entry(
                "peak-depth-six", statement="Depth 6 at peak.", phase="peak",
                action="adjust_hyperparameter", params={"name": "max_depth", "value": 6},
            ),
            fake_entry(
                "peak-depth-any", statement="Deeper trees at peak.", phase="peak",
                action="adjust_hyperparameter", params={"name": "max_depth"},
            ),
        ])
        llm = _llm_citing([], params={"name": "max_depth", "value": 7})
        orch = _make_orch(tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=bank)

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        action = result.iterations[1].action
        assert action["supported_by"] == ["peak-depth-any", "peak-depth-six"]  # store order: id asc
        # peak-depth-any leaves `value` open, so it does not disagree.
        assert action["supported_by_params_mismatch"] == ["peak-depth-six"]
        assert result.iterations[1].action_status == "applied"


def test_knowledge_bank_supported_by_empty_when_no_recommendation_matches():
    """An action no retrieved entry recommends is still applied; supported_by is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        llm = _llm_citing([], action_name="adjust_floor_constraint", params={"floor_pct": 0.4})
        orch = _make_orch(
            tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=_peak_bank(),
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        action = result.iterations[1].action
        assert action["name"] == "adjust_floor_constraint"
        assert action["supported_by"] == []
        assert result.iterations[1].action_status == "applied"


def test_knowledge_bank_hallucinated_citation_dropped_with_warning():
    """A cited id that was never shown is dropped and warned about; the action still applies."""
    capture = _WarningCapture()
    logger = logging.getLogger("agent.orchestrator")
    logger.addHandler(capture)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            baseline = _seed_baseline_csv(tmp_dir)
            llm = _llm_citing(["made-up-id", "peak-depth-v1", "peak-depth-v1"])
            orch = _make_orch(
                tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=_peak_bank(),
            )

            result = orch.run(
                initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
            )
    finally:
        logger.removeHandler(capture)

    action = result.iterations[1].action
    assert action["cited_entries"] == ["peak-depth-v1"], action["cited_entries"]
    assert result.iterations[1].action_status == "applied"
    assert len(capture.messages) == 1, capture.messages
    assert "made-up-id" in capture.messages[0]
    assert "hallucinated" in capture.messages[0]


def test_knowledge_bank_bracketed_citation_is_normalised_not_dropped():
    """qwen3:8b echoes ids as "[id]" (seen live 2026-10-01); brackets are stripped, not penalised."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        llm = _llm_citing(["[peak-depth-v1]", " [peak-reweight-v1] "])
        orch = _make_orch(
            tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=_peak_bank(),
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        action = result.iterations[1].action
        assert action["cited_entries"] == ["peak-depth-v1", "peak-reweight-v1"]


def test_knowledge_bank_citations_persist_to_tracker_and_report():
    """cited_entries survive the SQLite round trip and show up in report.md."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        orch = _make_orch(
            tmp_dir, _llm_citing(["peak-depth-v1"]), [200.0, 180.0],
            max_iterations=1, knowledge_bank=_peak_bank(),
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        run = orch.tracker.get_run(result.run_id)
        stored = run["iterations"][1]["action"]
        assert stored["cited_entries"] == ["peak-depth-v1"]
        assert stored["supported_by"] == ["peak-depth-v1"]

        md = (tmp_dir / "run" / "report.md").read_text()
        assert "cites: peak-depth-v1" in md
        payload = json.loads((tmp_dir / "run" / "report.json").read_text())
        assert payload["lines"][1]["cited_entries"] == ["peak-depth-v1"]


def test_knowledge_bank_unknown_phase_retrieves_without_phase_constraint():
    """A phase spelling the bank does not know degrades to phase=None, not a crash."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        bank = _peak_bank()
        diag = valid_diagnosis(weak_segments=[{
            "dimension": "phase", "value": "Winter Surge", "metric": "mape",
            "delta_vs_overall": 0.4, "severity": "high",
        }])
        llm = FakeLLM([json.dumps(diag), json.dumps(valid_action(cited_entries=[]))])
        orch = _make_orch(tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=bank)

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        assert bank.queries[0].phase is None
        assert bank.queries[0].model == "xgboost_direct"
        # No phase constraint: every entry matches, including the decline one,
        # in store order (same trust/confidence tier, so id asc).
        assert result.iterations[1].action["retrieved_entry_ids"] == [
            "decline-floor-v1", "peak-depth-v1", "peak-reweight-v1",
        ]


def test_knowledge_bank_experiential_entries_survive_the_proposal_limit():
    """Many curated matches never crowd out the experiential ones (finding: 12-limit cut all [E])."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        n_curated = DEFAULT_PROPOSAL_FACTS_LIMIT + 10
        curated = [
            fake_entry(f"peak-curated-{i:02d}", statement=f"Curated fact {i}.", phase="peak")
            for i in range(n_curated)
        ]
        experiential = [
            fake_entry(
                f"exp-run-{i}", statement=f"Run {i} at peak.", phase="peak",
                provenance="experiential", confidence="low", n_observations=1,
            )
            for i in range(3)
        ]
        bank = FakeKnowledgeBank(curated + experiential)
        llm = _llm_citing(["exp-run-0"])
        orch = _make_orch(tmp_dir, llm, [200.0, 180.0], max_iterations=1, knowledge_bank=bank)

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        facts = llm.calls[1]
        block = facts[facts.index("## Known facts for this diagnosis"):facts.index("## Iteration History")]
        assert "[exp-run-0]" in block and "[exp-run-1]" in block and "[exp-run-2]" in block
        shown = result.iterations[1].action["retrieved_entry_ids"]
        assert len(shown) == DEFAULT_PROPOSAL_FACTS_LIMIT
        assert shown[-3:] == ["exp-run-0", "exp-run-1", "exp-run-2"]
        # The lowest-ranked curated entries made room (store order within a
        # trust/confidence tier is id asc, so the highest ids are the ones cut).
        n_curated_shown = DEFAULT_PROPOSAL_FACTS_LIMIT - 3
        assert shown[:n_curated_shown] == [f"peak-curated-{i:02d}" for i in range(n_curated_shown)]
        assert f"peak-curated-{n_curated_shown:02d}" not in block
        omitted = n_curated + 3 - DEFAULT_PROPOSAL_FACTS_LIMIT
        assert f"- ({omitted} more matching entries omitted" in block
        assert result.iterations[1].action["cited_entries"] == ["exp-run-0"]
        assert 3 <= DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS


def test_knowledge_bank_experiential_slots_are_capped():
    """More experiential matches than slots: exactly the slot count is shown, in store order."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        curated = [
            fake_entry(f"peak-curated-{i:02d}", phase="peak")
            for i in range(DEFAULT_PROPOSAL_FACTS_LIMIT)
        ]
        n_exp = DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS + 4
        experiential = [
            fake_entry(
                f"exp-run-{i:02d}", phase="peak", provenance="experiential",
                confidence="low", n_observations=1,
            )
            for i in range(n_exp)
        ]
        bank = FakeKnowledgeBank(curated + experiential)
        orch = _make_orch(
            tmp_dir, _llm_citing([]), [200.0, 180.0], max_iterations=1, knowledge_bank=bank,
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        shown = result.iterations[1].action["retrieved_entry_ids"]
        assert len(shown) == DEFAULT_PROPOSAL_FACTS_LIMIT
        shown_exp = [i for i in shown if i.startswith("exp-")]
        assert shown_exp == [f"exp-run-{i:02d}" for i in range(DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS)]
        n_curated_shown = DEFAULT_PROPOSAL_FACTS_LIMIT - DEFAULT_PROPOSAL_EXPERIENTIAL_SLOTS
        assert shown[:n_curated_shown] == [f"peak-curated-{i:02d}" for i in range(n_curated_shown)]


def test_knowledge_bank_under_limit_shows_store_order_unchanged():
    """When nothing is cut, the shown list is exactly the store's ranking (one query, no reshuffle)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        bank = FakeKnowledgeBank([
            fake_entry("exp-run-0", phase="peak", provenance="experiential", confidence="low"),
            fake_entry("peak-b", phase="peak", confidence="medium"),
            fake_entry("peak-a", phase="peak"),
        ])
        orch = _make_orch(
            tmp_dir, _llm_citing([]), [200.0, 180.0], max_iterations=1, knowledge_bank=bank,
        )

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        assert len(bank.queries) == 1
        assert result.iterations[1].action["retrieved_entry_ids"] == ["peak-a", "peak-b", "exp-run-0"]


def test_no_knowledge_bank_leaves_proposal_untouched():
    """Without a bank the prompt has no known-facts section and the action carries no bank keys."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        llm = _build_llm(1)
        orch = _make_orch(tmp_dir, llm, [200.0, 180.0], max_iterations=1)

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        assert orch.knowledge_bank is None
        proposal_prompt = llm.calls[1]
        assert "Known facts" not in proposal_prompt
        assert "(none retrieved)" not in proposal_prompt
        assert "cited_entries" not in proposal_prompt
        action = result.iterations[1].action
        assert "cited_entries" not in action
        assert "retrieved_entry_ids" not in action
        assert "supported_by" not in action
        assert set(action) == {"name", "params", "rationale", "expected_effect"}


def test_no_knowledge_bank_drops_stray_cited_entries_from_stored_action():
    """An LLM that emits cited_entries anyway, with no bank wired, stores the pre-bank 4-key action."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        baseline = _seed_baseline_csv(tmp_dir)
        llm = _llm_citing(["peak-depth-v1"])
        orch = _make_orch(tmp_dir, llm, [200.0, 180.0], max_iterations=1)

        result = orch.run(
            initial_forecast=str(baseline), cutoff_date="2024-11-02", regenerate_baseline=False,
        )

        assert orch.knowledge_bank is None
        action = result.iterations[1].action
        assert set(action) == {"name", "params", "rationale", "expected_effect"}
        stored = orch.tracker.get_run(result.run_id)["iterations"][1]["action"]
        assert set(stored) == {"name", "params", "rationale", "expected_effect"}
        assert result.iterations[1].action_status == "applied"


ALL_TESTS = [
    test_loop_refines_a_bank_family_through_model_params,
    test_run_writes_report_into_run_dir,
    test_max_iterations_stop,
    test_agent_stop_action,
    test_two_regressions_stop,
    test_best_forecast_retained_on_regression,
    test_no_improvement_x2_stop,
    test_shadow_mode_skip,
    test_tracker_persists_full_state,
    test_invalid_llm_response_repair,
    test_knowledge_bank_retrieval_uses_diagnosed_phase_and_shows_ids,
    test_knowledge_bank_supported_by_matches_recommended_action,
    test_knowledge_bank_params_mismatch_empty_when_proposal_matches_recommended_value,
    test_knowledge_bank_params_mismatch_lists_supporting_entry_with_other_value,
    test_knowledge_bank_supported_by_empty_when_no_recommendation_matches,
    test_knowledge_bank_hallucinated_citation_dropped_with_warning,
    test_knowledge_bank_bracketed_citation_is_normalised_not_dropped,
    test_knowledge_bank_citations_persist_to_tracker_and_report,
    test_knowledge_bank_unknown_phase_retrieves_without_phase_constraint,
    test_knowledge_bank_experiential_entries_survive_the_proposal_limit,
    test_knowledge_bank_experiential_slots_are_capped,
    test_knowledge_bank_under_limit_shows_store_order_unchanged,
    test_no_knowledge_bank_leaves_proposal_untouched,
    test_no_knowledge_bank_drops_stray_cited_entries_from_stored_action,
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
