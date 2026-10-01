"""Unit tests for agent.knowledge.experiment_import (harness log -> experiential entries).

Every test is linear setup -> execute -> verify and uses
`tempfile.TemporaryDirectory()` rather than pytest fixtures, so the same
functions run under pytest and under `tests/agent/run_all.py`.

Run with:
    PYTHONPATH=. python tests/agent/test_experiment_import.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from agent.knowledge import (
    ImportResult,
    KnowledgeBank,
    KnowledgeEntry,
    RetrievalContext,
    import_experiment_log,
    render_known_facts,
    rows_to_entries,
)
from agent.knowledge.experiment_import import (
    ACTION_REWEIGHT,
    ACTION_SET_CONFIG,
    ACTION_TRAINING_WINDOW,
    SINGLE_OBSERVATION_CONFIDENCE,
    read_experiment_log,
)

CREATED_AT = "2026-10-01"

# ---- factories ------------------------------------------------------------

_STATE: dict[str, Any] = {
    "model": "xgboost_direct",
    "target_phase": "peak",
    "metric": "wis",
    "split": {
        "train_start_date": "2022-02-05",
        "eval_start_date": "2025-10-01",
        "eval_end_date": "2026-05-31",
        "stride_weeks": 4,
    },
    "n_cutoffs": 9,
}


def _row(
    *,
    run_at: str,
    config_id: str,
    config_delta: dict[str, Any],
    peak_wis: float,
    peak_bias: float,
    peak_cov: float,
    baseline_wis: float = 121.01,
    guard_wis: float = 50.05,
) -> dict[str, Any]:
    """One harness row; `metrics.by_phase` carries only what the importer reads plus noise."""
    delta = round(peak_wis - baseline_wis, 2)
    return {
        "run_at": run_at,
        "arm": config_id.split("_")[0],
        "config_id": config_id,
        "config_delta": config_delta,
        "describe": f"synthetic {config_id}",
        "state": json.loads(json.dumps(_STATE)),
        "metrics": {
            "overall": {"wis": guard_wis, "bias": -20.0, "coverage_95": 0.83},
            "by_phase": {
                "onset": {"wis": 11.0, "bias": -9.0, "coverage_95": 0.9},
                "peak": {"wis": peak_wis, "bias": peak_bias, "coverage_95": peak_cov},
                "decline": {"wis": 48.0, "bias": -26.0, "coverage_95": 0.73},
            },
            "by_horizon": {"1": {"wis": 29.0}},
        },
        "reward": {
            "metric": "wis",
            "phase": "peak",
            "baseline_value": baseline_wis,
            "value": peak_wis,
            "delta": delta,
            "improvement_frac": round(-delta / baseline_wis, 4),
            "better": delta < 0,
            "guard": {"metric": "overall_wis", "baseline_value": 50.05, "value": guard_wis},
        },
        "seconds": 1.0,
    }


def _baseline_row() -> dict[str, Any]:
    return _row(
        run_at="2026-09-30T20:10:48+00:00",
        config_id="baseline",
        config_delta={},
        peak_wis=121.01,
        peak_bias=-58.8,
        peak_cov=0.855,
    )


def _lambda_row() -> dict[str, Any]:
    return _row(
        run_at="2026-09-30T20:17:23+00:00",
        config_id="lambda_3",
        config_delta={"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}},
        peak_wis=123.71,
        peak_bias=-6.4,
        peak_cov=0.808,
        guard_wis=50.4,
    )


def _window_row() -> dict[str, Any]:
    return _row(
        run_at="2026-09-30T20:25:00+00:00",
        config_id="window_12",
        config_delta={"data.train_window_weeks": 12},
        peak_wis=238.26,
        peak_bias=-120.3,
        peak_cov=0.602,
        guard_wis=96.02,
    )


def _three_rows() -> list[dict[str, Any]]:
    return [_baseline_row(), _lambda_row(), _window_row()]


def _write_log(directory: Path, rows: list[dict[str, Any]], name: str = "log.jsonl") -> Path:
    path = directory / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_curated(directory: Path) -> Path:
    """A one-file curated dir with a peak entry (high) and an onset-only entry."""
    curated = directory / "curated"
    curated.mkdir()
    entries = [
        {
            "id": "curated-peak-fact",
            "provenance": "curated",
            "category": "model_characteristics",
            "statement": "Curated peak fact.",
            "context": {"phase": ["peak", "approaching_peak"]},
            "evidence": {"source": "A. Adiga"},
            "confidence": "high",
            "created_at": "2026-09-17",
        },
        {
            "id": "curated-onset-fact",
            "provenance": "curated",
            "category": "domain_dynamics",
            "statement": "Curated onset fact.",
            "context": {"phase": ["onset"]},
            "evidence": {"source": "A. Adiga"},
            "confidence": "low",
            "created_at": "2026-09-17",
        },
    ]
    (curated / "seed.yaml").write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    return curated


def _expect_value_error(fn: Callable[[], Any], *needles: str) -> str:
    try:
        fn()
    except ValueError as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, f"expected {needle!r} in error: {message}"
        return message
    raise AssertionError(f"expected ValueError mentioning {needles}, nothing raised")


# ---- rows_to_entries: ids, statements, fields -------------------------------


def test_rows_to_entries_skips_baseline_and_yields_one_entry_per_config():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    assert [e.id for e in entries] == ["exp-lambda-3-2025", "exp-window-12-2025"]


def test_entry_id_normalises_dots_and_underscores_to_dashes():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["config_id"] = "lambda_1.5"
    rows[1]["config_delta"]["sample_weights.approaching_peak"]["weight"] = 1.5

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert entries[0].id == "exp-lambda-1-5-2025"


def test_entry_id_uses_eval_start_year_not_run_year():
    rows = [_baseline_row(), _lambda_row()]
    rows[0]["state"]["split"]["eval_start_date"] = "2024-10-01"
    rows[1]["state"]["split"]["eval_start_date"] = "2024-10-01"

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert entries[0].id == "exp-lambda-3-2024"


def test_approaching_peak_statement_is_exact():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    assert entries[0].statement == (
        "approaching_peak weight 3.0 (weeks_before 6): peak WIS 123.71 vs 121.01 baseline "
        "(+2.70, worse); peak bias -58.8 -> -6.4; peak coverage_95 0.855 -> 0.808"
    )


def test_window_statement_is_exact():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    assert entries[1].statement == (
        "train_window_weeks 12: peak WIS 238.26 vs 121.01 baseline (+117.25, worse); "
        "peak bias -58.8 -> -120.3; peak coverage_95 0.855 -> 0.602"
    )


def test_calendar_phase_weight_statement_and_action():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["config_id"] = "lambda_calendar_2"
    rows[1]["config_delta"] = {"sample_weights.by_phase.peak": 2.0}

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert entries[0].id == "exp-lambda-calendar-2-2025"
    assert entries[0].statement.startswith("phase peak weight 2.0: peak WIS")
    assert entries[0].payload["action"] == {
        "name": ACTION_REWEIGHT,
        "params": {"dimension": "phase", "value": "peak", "weight": 2.0},
        "config_delta": {"sample_weights.by_phase.peak": 2.0},
    }


def test_better_row_says_better_and_negative_delta():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["reward"]["value"] = 118.51
    rows[1]["reward"]["delta"] = -2.5
    rows[1]["reward"]["better"] = True

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert "peak WIS 118.51 vs 121.01 baseline (-2.50, better)" in entries[0].statement
    assert entries[0].payload["reward"]["better"] is True
    assert entries[0].evidence["reward_delta"] == -2.5


def test_zero_delta_row_says_no_change():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["reward"]["value"] = 121.01
    rows[1]["reward"]["delta"] = 0.0
    rows[1]["reward"]["better"] = False

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert "(+0.00, no change)" in entries[0].statement


def test_entities_context_provenance_category_confidence():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    for entry in entries:
        assert entry.provenance == "experiential"
        assert entry.category == "model_characteristics"
        assert entry.confidence == SINGLE_OBSERVATION_CONFIDENCE == "low"
        assert entry.entities == {"model": "xgboost_direct", "phase": "peak", "metric": "wis"}
        assert entry.context == {
            "model": ["xgboost_direct"],
            "phase": ["peak"],
            "metric": ["wis"],
        }
        assert entry.llm_gloss is None
        assert entry.created_at == CREATED_AT


def test_evidence_carries_source_single_observation_and_delta():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT, source_label="run.jsonl")

    assert entries[0].evidence == {
        "source": "run.jsonl#lambda_3 run_at 2026-09-30T20:17:23+00:00",
        "n_observations": 1,
        "reward_delta": 2.7,
    }
    assert entries[1].evidence["source"] == "run.jsonl#window_12 run_at 2026-09-30T20:25:00+00:00"
    assert entries[1].evidence["reward_delta"] == 117.25


def test_payload_reward_action_state_and_before_after_values():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    lam = entries[0].payload
    assert lam["reward"] == {
        "metric": "wis",
        "phase": "peak",
        "baseline": 121.01,
        "value": 123.71,
        "delta": 2.7,
        "better": False,
    }
    assert lam["action"] == {
        "name": ACTION_REWEIGHT,
        "params": {"dimension": "approaching_peak", "value": 6, "weight": 3.0},
        "config_delta": {"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 3.0}},
    }
    assert lam["state"] == _STATE
    assert lam["guard"] == {"metric": "overall_wis", "baseline_value": 50.05, "value": 50.4}
    assert lam["bias_before"] == -58.8 and lam["bias_after"] == -6.4
    assert lam["coverage_95_before"] == 0.855 and lam["coverage_95_after"] == 0.808

    win = entries[1].payload
    assert win["action"] == {
        "name": ACTION_TRAINING_WINDOW,
        "params": {"weeks": 12},
        "config_delta": {"data.train_window_weeks": 12},
    }


def test_payload_omits_guard_when_row_has_none():
    rows = [_baseline_row(), _lambda_row()]
    del rows[1]["reward"]["guard"]

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert "guard" not in entries[0].payload


def test_unknown_config_path_falls_back_to_raw_path_and_set_config():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["config_id"] = "mystery_7"
    rows[1]["config_delta"] = {"model.params.max_depth": 7}

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert entries[0].id == "exp-mystery-7-2025"
    assert entries[0].statement.startswith("model.params.max_depth=7: peak WIS")
    assert entries[0].payload["action"] == {
        "name": ACTION_SET_CONFIG,
        "params": {"path": "model.params.max_depth", "value": 7},
        "config_delta": {"model.params.max_depth": 7},
    }


def test_multi_path_delta_is_recorded_as_composite():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["config_id"] = "combo"
    rows[1]["config_delta"] = {
        "sample_weights.by_phase.peak": 2.0,
        "data.train_window_weeks": 52,
    }

    entries = rows_to_entries(rows, created_at=CREATED_AT)

    assert entries[0].statement.startswith(
        "phase peak weight 2.0, train_window_weeks 52: peak WIS"
    )
    action = entries[0].payload["action"]
    assert action["name"] == "composite"
    assert [a["name"] for a in action["actions"]] == [ACTION_REWEIGHT, ACTION_TRAINING_WINDOW]


def test_entries_validate_and_round_trip_via_to_dict_from_dict():
    entries = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    for entry in entries:
        dumped = entry.to_dict()
        json.dumps(dumped)  # plain JSON, no fallback
        assert KnowledgeEntry.from_dict(dumped) == entry


def test_rows_to_entries_is_deterministic():
    first = rows_to_entries(_three_rows(), created_at=CREATED_AT)
    second = rows_to_entries(_three_rows(), created_at=CREATED_AT)

    assert first == second
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


# ---- rows_to_entries: baseline pairing and loud failures --------------------


def test_missing_baseline_raises():
    rows = [_lambda_row(), _window_row()]

    _expect_value_error(
        lambda: rows_to_entries(rows, created_at=CREATED_AT), "no baseline row", "config_delta"
    )


def test_row_logged_before_its_baseline_still_pairs():
    # Arms run concurrently: in the real log window_12 landed 45s before the
    # baseline row. run_at must not be treated as an ordering key.
    baseline = _baseline_row()
    early = _lambda_row()
    early["run_at"] = "2026-09-30T19:00:00+00:00"

    entries = rows_to_entries([early, baseline], created_at=CREATED_AT)

    assert [e.id for e in entries] == ["exp-lambda-3-2025"]
    assert entries[0].payload["bias_before"] == -58.8


def test_row_whose_baseline_value_matches_no_baseline_raises():
    baseline = _baseline_row()
    orphan = _lambda_row()
    orphan["reward"]["baseline_value"] = 99.99

    _expect_value_error(
        lambda: rows_to_entries([baseline, orphan], created_at=CREATED_AT),
        "lambda_3",
        "no baseline row with the same state",
        "99.99",
        "121.01",
    )


def test_row_pairs_with_baseline_of_matching_value_across_appended_runs():
    first_baseline = _baseline_row()
    first_lambda = _lambda_row()
    second_baseline = _row(
        run_at="2026-10-02T09:00:00+00:00",
        config_id="baseline",
        config_delta={},
        peak_wis=119.5,
        peak_bias=-40.0,
        peak_cov=0.87,
        baseline_wis=119.5,
    )
    second_lambda = _row(
        run_at="2026-10-02T09:30:00+00:00",
        config_id="lambda_5",
        config_delta={"sample_weights.approaching_peak": {"weeks_before": 6, "weight": 5.0}},
        peak_wis=122.0,
        peak_bias=-10.0,
        peak_cov=0.8,
        baseline_wis=119.5,
    )

    entries = rows_to_entries(
        [first_baseline, first_lambda, second_baseline, second_lambda], created_at=CREATED_AT
    )

    assert [e.id for e in entries] == ["exp-lambda-3-2025", "exp-lambda-5-2025"]
    assert entries[0].payload["bias_before"] == -58.8
    assert entries[1].payload["bias_before"] == -40.0
    assert entries[1].payload["reward"]["baseline"] == 119.5


def test_identical_baselines_resolve_to_the_nearest_run_at():
    first_baseline = _baseline_row()
    second_baseline = _baseline_row()
    second_baseline["run_at"] = "2026-10-02T09:00:00+00:00"
    second_baseline["metrics"]["by_phase"]["peak"]["bias"] = -40.0  # same WIS, drifted bias
    late_lambda = _lambda_row()
    late_lambda["run_at"] = "2026-10-02T09:30:00+00:00"

    entries = rows_to_entries(
        [first_baseline, second_baseline, late_lambda], created_at=CREATED_AT
    )

    assert entries[0].payload["bias_before"] == -40.0


def test_missing_phase_block_in_metrics_raises_with_row_name():
    rows = [_baseline_row(), _lambda_row()]
    del rows[1]["metrics"]["by_phase"]["peak"]

    _expect_value_error(
        lambda: rows_to_entries(rows, created_at=CREATED_AT), "lambda_3", "by_phase", "'peak'"
    )


def test_missing_required_row_key_raises_with_row_name():
    rows = [_baseline_row(), _lambda_row()]
    del rows[1]["reward"]["delta"]

    _expect_value_error(
        lambda: rows_to_entries(rows, created_at=CREATED_AT), "row[1] lambda_3.reward", "delta"
    )


def test_malformed_approaching_peak_delta_raises():
    rows = [_baseline_row(), _lambda_row()]
    rows[1]["config_delta"] = {"sample_weights.approaching_peak": {"weight": 3.0}}

    _expect_value_error(
        lambda: rows_to_entries(rows, created_at=CREATED_AT),
        "sample_weights.approaching_peak",
        "weeks_before",
    )


def test_bad_created_at_raises_from_schema():
    _expect_value_error(
        lambda: rows_to_entries(_three_rows(), created_at="yesterday"),
        "KnowledgeEntry.created_at",
    )


# ---- read_experiment_log ----------------------------------------------------


def test_read_experiment_log_skips_blank_lines_and_reports_bad_json():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        good = _write_log(directory, _three_rows())
        good.write_text(good.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        bad = directory / "bad.jsonl"
        bad.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
        empty = directory / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")

        rows = read_experiment_log(good)

        assert len(rows) == 3
        _expect_value_error(lambda: read_experiment_log(bad), "bad.jsonl:2", "invalid JSON")
        _expect_value_error(lambda: read_experiment_log(empty), "empty")
        _expect_value_error(lambda: read_experiment_log(directory / "nope.jsonl"), "does not exist")


# ---- import_experiment_log: bank integration --------------------------------


def test_import_writes_entries_and_returns_counts():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        log_path = _write_log(directory, _three_rows())
        bank = KnowledgeBank(directory / "k.db")

        result = import_experiment_log(log_path, bank, created_at=CREATED_AT)

        assert result == ImportResult(
            n_rows=3, n_entries=2, ids=("exp-lambda-3-2025", "exp-window-12-2025")
        )
        assert bank.count("experiential") == 2
        stored = bank.get("exp-lambda-3-2025")
        assert stored is not None
        assert stored.evidence["source"] == (
            f"{log_path}#lambda_3 run_at 2026-09-30T20:17:23+00:00"
        )
        assert stored == rows_to_entries(
            _three_rows(), created_at=CREATED_AT, source_label=str(log_path)
        )[0]


def test_reimport_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        log_path = _write_log(directory, _three_rows())
        bank = KnowledgeBank(directory / "k.db")
        first = import_experiment_log(log_path, bank, created_at=CREATED_AT)
        before = [bank.get(i) for i in first.ids]

        second = import_experiment_log(log_path, bank, created_at=CREATED_AT)
        after = [bank.get(i) for i in second.ids]

        assert first == second
        assert bank.count("experiential") == 2
        assert bank.count() == 2
        assert before == after
        assert [e.to_dict() for e in before] == [e.to_dict() for e in after]


def test_import_missing_baseline_leaves_bank_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        log_path = _write_log(directory, [_lambda_row(), _window_row()])
        bank = KnowledgeBank(directory / "k.db")

        _expect_value_error(
            lambda: import_experiment_log(log_path, bank, created_at=CREATED_AT), "no baseline row"
        )

        assert bank.count() == 0


def test_import_rejects_id_already_held_by_another_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        log_path = _write_log(directory, _three_rows())
        bank = KnowledgeBank(directory / "k.db")
        squatter = KnowledgeEntry.from_dict(
            {
                "id": "exp-lambda-3-2025",
                "provenance": "curated",
                "category": "forecasts",
                "statement": "Squatting on an experiential id.",
                "evidence": {"source": "test"},
                "confidence": "low",
                "created_at": "2026-09-17",
            }
        )
        bank.upsert(squatter)

        _expect_value_error(
            lambda: import_experiment_log(log_path, bank, created_at=CREATED_AT),
            "exp-lambda-3-2025",
            "'curated'",
        )

        assert bank.get("exp-lambda-3-2025") == squatter


def test_query_for_peak_returns_curated_before_experiential_after_import():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        curated_dir = _write_curated(directory)
        log_path = _write_log(directory, _three_rows())
        bank = KnowledgeBank.open(curated_dir=curated_dir, db_path=directory / "k.db")

        import_experiment_log(log_path, bank, created_at=CREATED_AT)
        peak = bank.query(RetrievalContext(phase="peak", model="xgboost_direct"))
        onset = bank.query(RetrievalContext(phase="onset"))
        rebuilt = bank.rebuild_curated(curated_dir)

        assert [e.id for e in peak] == [
            "curated-peak-fact",
            "exp-lambda-3-2025",
            "exp-window-12-2025",
        ]
        assert [e.provenance for e in peak] == ["curated", "experiential", "experiential"]
        assert [e.id for e in onset] == ["curated-onset-fact"]
        # A curated rebuild keeps the imported rows (store invariant, re-checked here).
        assert rebuilt == 2
        assert bank.count("experiential") == 2
        assert [e.id for e in bank.list(provenance="experiential")] == [
            "exp-lambda-3-2025",
            "exp-window-12-2025",
        ]


def test_imported_entries_render_as_single_run_experiential_lines():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        log_path = _write_log(directory, _three_rows())
        bank = KnowledgeBank(directory / "k.db")
        import_experiment_log(log_path, bank, created_at=CREATED_AT)

        text = render_known_facts(bank.query(RetrievalContext(phase="peak")))

        assert text.splitlines()[1] == (
            "- [E, low, 1 run] approaching_peak weight 3.0 (weeks_before 6): peak WIS 123.71 vs "
            "121.01 baseline (+2.70, worse); peak bias -58.8 -> -6.4; "
            "peak coverage_95 0.855 -> 0.808"
        )


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main() -> None:
    passed = failed = 0
    for fn in ALL:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001 - test runner surface
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
