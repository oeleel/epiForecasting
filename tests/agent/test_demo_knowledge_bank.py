"""Tests for scripts/demo_knowledge_bank.py (the 2026-10-01 meeting demo).

Drives the non-live steps against a bank built from the real curated YAML
into a temp DB, so a schema or curated-file change that would break the demo
in the meeting breaks here first. The live step (step 7) needs Ollama and is
deliberately not exercised.

Every test is linear setup -> execute -> verify and uses
`tempfile.TemporaryDirectory()` rather than pytest fixtures, so the same
functions run under pytest and under `tests/agent/run_all.py`.

Run with:
    PYTHONPATH=. python tests/agent/test_demo_knowledge_bank.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.knowledge import DEFAULT_CURATED_DIR, DEFAULT_HEADER, KnowledgeBank  # noqa: E402
from scripts import demo_knowledge_bank as demo  # noqa: E402

CREATED_AT = "2026-10-01"
# Three rows copied from the harness log (baseline, lambda_3, window_12),
# trimmed to the keys experiment_import reads. Committed, so the import path is
# always exercised: the real log is gitignored and absent on a fresh clone.
FIXTURE_LOG = PROJECT_ROOT / "tests" / "agent" / "fixtures" / "peak_rectification_log_sample.jsonl"
FIXTURE_ENTRY_IDS = ("exp-lambda-3-2025", "exp-window-12-2025")


def _temp_bank(tmp: str) -> KnowledgeBank:
    return KnowledgeBank.open(curated_dir=DEFAULT_CURATED_DIR, db_path=Path(tmp) / "knowledge.db")


# ---- individual steps -------------------------------------------------------


def test_step_validate_yaml_counts_every_curated_file():
    out = io.StringIO()

    record = demo.step_validate_yaml(DEFAULT_CURATED_DIR, out)

    text = out.getvalue()
    assert "1. Validate the curated YAML" in text
    assert "python -m agent knowledge validate" in text
    assert record["n_entries"] == sum(record["per_file"].values())
    assert record["n_files"] == len(record["per_file"]) >= 1
    assert record["by_provenance"] == {"curated": record["n_entries"]}
    assert "_TEMPLATE.yaml" in record["skipped_files"]
    for name in record["per_file"]:
        assert name in text


def test_step_open_bank_reports_curated_count_and_zero_others_on_fresh_db():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()

        record = demo.step_open_bank(bank, out)

        text = out.getvalue()
        assert record["counts"]["curated"] == bank.count(provenance="curated") > 0
        assert record["counts"]["derived"] == 0
        assert record["counts"]["experiential"] == 0
        assert record["total"] == record["counts"]["curated"]
        assert str(Path(tmp) / "knowledge.db") in text


def test_step_retrieval_prints_known_facts_for_peak_and_plateau():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()

        record = demo.step_retrieval(bank, out)

        text = out.getvalue()
        assert text.count(DEFAULT_HEADER) == 2
        assert record["peak"]["block"].startswith(DEFAULT_HEADER)
        assert record["plateau"]["block"].startswith(DEFAULT_HEADER)
        assert demo.LAMBDA_ENTRY_ID in record["peak"]["ids"]
        assert demo.LAMBDA_ENTRY_ID not in record["plateau"]["ids"]
        assert demo.LAMBDA_ENTRY_ID in record["only_peak"]
        assert "train-long-window-in-lull" in record["only_plateau"]
        assert "only in the peak block" in text


def test_step_show_entry_dumps_lambda_entry_schema():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()

        record = demo.step_show_entry(bank, out)

        text = out.getvalue()
        assert record["entry_id"] == demo.LAMBDA_ENTRY_ID
        assert record["entry"]["id"] == demo.LAMBDA_ENTRY_ID
        assert record["entry"]["payload"]["recommendation"]["action"] == "reweight_training_samples"
        assert f"id: {demo.LAMBDA_ENTRY_ID}" in text
        assert "action: reweight_training_samples" in text
        assert "provenance: curated" in text


def test_step_show_entry_unknown_id_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()

        try:
            demo.step_show_entry(bank, out, entry_id="no-such-entry")
        except ValueError as exc:
            assert "no-such-entry" in str(exc)
        else:
            raise AssertionError("expected ValueError for an unknown entry id")


def test_step_experiment_memory_skips_with_harness_command_when_log_missing():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()
        missing = Path(tmp) / "no-such-log.jsonl"

        record = demo.step_experiment_memory(bank, out, missing, CREATED_AT)

        text = out.getvalue()
        assert record["skipped"] is True
        assert "SKIP" in text
        assert str(missing) in text
        assert demo.HARNESS_COMMAND in text
        assert bank.count(provenance="experiential") == 0


def test_step_experiment_memory_imports_fixture_log_as_single_run_entries():
    assert FIXTURE_LOG.is_file(), f"fixture missing: {FIXTURE_LOG}"
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()

        record = demo.step_experiment_memory(bank, out, FIXTURE_LOG, CREATED_AT)

        text = out.getvalue()
        assert record["skipped"] is False
        assert record["import_result"]["n_rows"] == 3
        assert record["import_result"]["n_entries"] == 2
        assert tuple(record["import_result"]["ids"]) == FIXTURE_ENTRY_IDS
        assert bank.count(provenance="experiential") == 2
        assert len(record["experiential_lines"]) == 2
        for line in record["experiential_lines"]:
            assert line.startswith("- [E, low, 1 run]")
            assert line in text
        assert "approaching_peak weight 3.0 (weeks_before 6)" in text
        assert "train_window_weeks 12" in text
        assert "ImportResult(n_rows=3, n_entries=2)" in text
        assert "experiential rows now in bank: 2" in text


def test_step_experiment_memory_import_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)

        first = demo.step_experiment_memory(bank, io.StringIO(), FIXTURE_LOG, CREATED_AT)
        second = demo.step_experiment_memory(bank, io.StringIO(), FIXTURE_LOG, CREATED_AT)

        assert first["import_result"] == second["import_result"]
        assert bank.count(provenance="experiential") == 2


def test_step_loop_reach_prints_domain_context_tail_and_action_keys():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()

        record = demo.step_loop_reach(bank, out)

        text = out.getvalue()
        assert record["domain_context_tail"].startswith(DEFAULT_HEADER)
        assert demo.LAMBDA_ENTRY_ID not in record["domain_context_tail"]  # statements, not ids
        tail = record["domain_context_tail"]
        assert "reweight_training_samples(dimension=approaching_peak" in tail
        assert DEFAULT_HEADER in text
        assert "python -m agent improve" in text
        for key in ("retrieved_entry_ids", "cited_entries", "supported_by"):
            assert key in text
            assert key in record["action_keys"]


# ---- the whole non-live demo ------------------------------------------------


def test_run_demo_without_live_covers_steps_one_to_six_and_is_json_serialisable():
    with tempfile.TemporaryDirectory() as tmp:
        bank = _temp_bank(tmp)
        out = io.StringIO()
        missing = Path(tmp) / "no-such-log.jsonl"

        record = demo.run_demo(bank, out, log_path=missing, created_at=CREATED_AT)

        text = out.getvalue()
        assert list(record["steps"]) == [
            "1_validate_yaml", "2_open_bank", "3_retrieval", "4_entry",
            "5_experiment_memory", "6_loop_reach",
        ]
        assert "7_live" not in record["steps"]
        assert "KNOWLEDGE BANK DEMO" in text
        assert DEFAULT_HEADER in text
        assert demo.LAMBDA_ENTRY_ID in text
        assert "SKIP" in text
        assert "NEXT" in text
        json.dumps(record)  # --json must be able to write it


def test_main_writes_json_record_for_fixture_log_into_temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "nested" / "demo.json"
        db_path = Path(tmp) / "knowledge.db"
        argv = [
            "demo_knowledge_bank.py",
            "--json", str(json_path),
            "--log", str(FIXTURE_LOG),
            "--db", str(db_path),
        ]
        out = io.StringIO()
        saved_argv = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(out):
                demo.main()
        finally:
            sys.argv = saved_argv

        text = out.getvalue()
        assert json_path.is_file()
        assert db_path.is_file()
        record = json.loads(json_path.read_text())
        assert list(record["steps"]) == [
            "1_validate_yaml", "2_open_bank", "3_retrieval", "4_entry",
            "5_experiment_memory", "6_loop_reach",
        ]
        step5 = record["steps"]["5_experiment_memory"]
        assert step5["skipped"] is False
        assert step5["log_path"] == str(FIXTURE_LOG)
        assert tuple(step5["import_result"]["ids"]) == FIXTURE_ENTRY_IDS
        assert record["steps"]["2_open_bank"]["db_path"] == str(db_path)
        assert f"wrote {json_path}" in text
        assert KnowledgeBank(db_path).count(provenance="experiential") == 2


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
