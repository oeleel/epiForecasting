"""Unit tests for agent.knowledge (schema, YAML intake, SQLite store, render).

Every test is linear setup -> execute -> verify and uses
`tempfile.TemporaryDirectory()` rather than pytest fixtures, so the same
functions run under pytest and under `tests/agent/run_all.py`.

Run with:
    PYTHONPATH=. python tests/agent/test_knowledge_bank.py
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from agent.knowledge import (
    DEFAULT_CURATED_DIR,
    DEFAULT_HEADER,
    EMPTY_BLOCK,
    KNOWN_PHASES,
    RECOMMENDATION_NOT_YET_AVAILABLE,
    KnowledgeBank,
    KnowledgeEntry,
    RetrievalContext,
    load_curated_dir,
    render_known_facts,
)

# ---- factories ------------------------------------------------------------


def _entry_dict(**overrides: Any) -> dict[str, Any]:
    """A valid curated entry in the YAML shape; override any key."""
    base: dict[str, Any] = {
        "id": "onset-definition-v1",
        "provenance": "curated",
        "category": "domain_dynamics",
        "statement": "Season onset = 3 consecutive weeks of increase above threshold T.",
        "entities": {"phase": "onset"},
        "context": {"season_week": [38, 46]},
        "payload": {"rule": {"consecutive_weeks": 3}},
        "evidence": {
            "source": "advisor, meeting 2026-09-10",
            "n_observations": None,
            "reward_delta": None,
        },
        "confidence": "high",
        "llm_gloss": None,
        "created_at": "2026-09-17",
    }
    base.update(overrides)
    return base


def _entry(**overrides: Any) -> KnowledgeEntry:
    return KnowledgeEntry.from_dict(_entry_dict(**overrides))


def _expect_value_error(fn: Callable[[], Any], *needles: str) -> str:
    """Call `fn`, assert it raises ValueError mentioning every needle; return the message."""
    try:
        fn()
    except ValueError as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, f"expected {needle!r} in error: {message}"
        return message
    raise AssertionError(f"expected ValueError mentioning {needles}, nothing raised")


def _write_yaml(directory: Path, name: str, content: Any) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


# ---- schema: happy path ----------------------------------------------------


def test_from_dict_to_dict_round_trips():
    data = _entry_dict()
    entry = KnowledgeEntry.from_dict(data)
    assert entry.to_dict() == data
    assert KnowledgeEntry.from_dict(entry.to_dict()) == entry


def test_to_dict_emits_canonical_evidence_shape():
    data = _entry_dict(evidence={"source": "advisor"})
    entry = KnowledgeEntry.from_dict(data)
    assert entry.to_dict()["evidence"] == {
        "source": "advisor",
        "n_observations": None,
        "reward_delta": None,
    }
    assert KnowledgeEntry.from_dict(entry.to_dict()) == entry


def test_from_dict_defaults_optional_sections():
    data = _entry_dict()
    del data["entities"], data["context"], data["payload"], data["llm_gloss"]
    entry = KnowledgeEntry.from_dict(data)
    assert entry.entities == {}
    assert entry.context == {}
    assert entry.payload == {}
    assert entry.llm_gloss is None
    assert entry.season_week is None
    assert entry.recommendation is None


def test_statement_strips_folded_yaml_trailing_newline():
    entry = _entry(statement="  Folded scalar keeps a newline.\n")
    assert entry.statement == "Folded scalar keeps a newline."


def test_created_at_accepts_yaml_date_object():
    entry = _entry(created_at=date(2026, 9, 17))
    assert entry.created_at == "2026-09-17"


def test_created_at_rejects_yaml_timestamp():
    # An unquoted `created_at: 2026-09-30 10:00:00` loads as a datetime, which
    # is a date subclass; it must not be stored as '2026-09-30T10:00:00'.
    _expect_value_error(
        lambda: _entry(created_at=datetime(2026, 9, 30, 10, 0, 0)),
        "KnowledgeEntry.created_at",
        "timestamp",
    )
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "a.yaml").write_text(
            "- id: ts-entry\n  provenance: curated\n  category: forecasts\n"
            "  statement: A fact.\n  evidence: {source: advisor}\n  confidence: low\n"
            "  created_at: 2026-09-30 10:00:00\n",
            encoding="utf-8",
        )
        _expect_value_error(lambda: load_curated_dir(directory), "a.yaml[0] ts-entry", "timestamp")


def test_from_dict_rejects_falsy_non_mapping_sections():
    for key in ("entities", "context", "payload"):
        for bad in ("", [], 0, False):
            _expect_value_error(
                lambda: KnowledgeEntry.from_dict(_entry_dict(**{key: bad})),
                f"KnowledgeEntry.{key}",
                "expected a mapping",
            )
    # null still means "no section"
    entry = KnowledgeEntry.from_dict(_entry_dict(entities=None, context=None, payload=None))
    assert entry.entities == {} and entry.context == {} and entry.payload == {}


def test_payload_rejects_values_that_are_not_plain_json():
    _expect_value_error(
        lambda: _entry(payload={"when": date(2026, 1, 1)}),
        "KnowledgeEntry.payload",
        "plain JSON",
    )
    _expect_value_error(lambda: _entry(payload={"x": {1, 2}}), "KnowledgeEntry.payload")


def test_entries_are_hashable_and_usable_in_sets():
    a = _entry()
    same = KnowledgeEntry.from_dict(a.to_dict())
    other = _entry(id="other-entry")
    assert hash(a) == hash(same)
    assert a == same
    assert len({a, same, other}) == 2


def test_evidence_normalises_reward_delta_to_float_and_fills_n_observations():
    entry = _entry(evidence={"source": "run x", "reward_delta": 3})
    assert entry.evidence["reward_delta"] == 3.0
    assert isinstance(entry.evidence["reward_delta"], float)
    assert entry.evidence["n_observations"] is None


def test_to_dict_is_a_deep_copy():
    entry = _entry()
    dumped = entry.to_dict()
    dumped["payload"]["rule"]["consecutive_weeks"] = 99
    assert entry.payload["rule"]["consecutive_weeks"] == 3


def test_known_phases_cover_both_vocabularies():
    for phase in ("onset", "peak", "decline", "off_season", "surge", "plateau", "approaching_peak"):
        assert phase in KNOWN_PHASES, phase


# ---- schema: each validation branch names its field ------------------------


def test_id_rejects_non_slug():
    _expect_value_error(lambda: _entry(id="Bad_ID"), "KnowledgeEntry.id", "Bad_ID")
    _expect_value_error(lambda: _entry(id="-leading-dash"), "KnowledgeEntry.id")


def test_provenance_rejects_unknown_value():
    _expect_value_error(lambda: _entry(provenance="guess"), "KnowledgeEntry.provenance", "guess")


def test_category_rejects_unknown_value():
    _expect_value_error(lambda: _entry(category="misc"), "KnowledgeEntry.category", "misc")


def test_statement_rejects_empty_and_multiline():
    _expect_value_error(lambda: _entry(statement="   "), "KnowledgeEntry.statement", "non-empty")
    _expect_value_error(
        lambda: _entry(statement="line one\nline two"), "KnowledgeEntry.statement", "single line"
    )


def test_entities_rejects_unknown_key_and_unknown_phase():
    _expect_value_error(
        lambda: _entry(entities={"location": "US"}), "KnowledgeEntry.entities", "location"
    )
    _expect_value_error(
        lambda: _entry(entities={"phase": "winter"}), "KnowledgeEntry.entities.phase", "winter"
    )
    _expect_value_error(lambda: _entry(entities={"model": ""}), "KnowledgeEntry.entities.model")


def test_context_rejects_unknown_key():
    _expect_value_error(
        lambda: _entry(context={"location": ["US"]}), "KnowledgeEntry.context", "location"
    )


def test_context_list_keys_require_non_empty_lists():
    _expect_value_error(
        lambda: _entry(context={"phase": "peak"}), "KnowledgeEntry.context.phase", "list"
    )
    _expect_value_error(
        lambda: _entry(context={"model": []}), "KnowledgeEntry.context.model", "empty list"
    )
    _expect_value_error(
        lambda: _entry(context={"phase": ["peak", "winter"]}),
        "KnowledgeEntry.context.phase",
        "winter",
    )
    _expect_value_error(
        lambda: _entry(context={"metric": ["wis", "wis"]}),
        "KnowledgeEntry.context.metric",
        "duplicate",
    )


def test_context_model_and_metric_are_free_strings():
    entry = _entry(context={"model": ["not_wrapped_yet"], "metric": ["wis"]})
    assert entry.context["model"] == ["not_wrapped_yet"]


def test_context_season_week_bounds():
    _expect_value_error(
        lambda: _entry(context={"season_week": [0, 10]}), "KnowledgeEntry.context.season_week[0]"
    )
    _expect_value_error(
        lambda: _entry(context={"season_week": [10, 54]}), "KnowledgeEntry.context.season_week[1]"
    )
    _expect_value_error(
        lambda: _entry(context={"season_week": [20, 10]}),
        "KnowledgeEntry.context.season_week",
        "lo must be <= hi",
    )
    _expect_value_error(
        lambda: _entry(context={"season_week": [10]}),
        "KnowledgeEntry.context.season_week",
        "[lo, hi]",
    )
    _expect_value_error(
        lambda: _entry(context={"season_week": [True, 10]}), "KnowledgeEntry.context.season_week[0]"
    )
    assert _entry(context={"season_week": [1, 53]}).season_week == (1, 53)


def test_evidence_requires_source_and_validates_numbers():
    _expect_value_error(lambda: _entry(evidence={}), "KnowledgeEntry.evidence", "source")
    _expect_value_error(lambda: _entry(evidence={"source": ""}), "KnowledgeEntry.evidence.source")
    _expect_value_error(
        lambda: _entry(evidence={"source": "x", "n_observations": -1}),
        "KnowledgeEntry.evidence.n_observations",
    )
    _expect_value_error(
        lambda: _entry(evidence={"source": "x", "n_observations": True}),
        "KnowledgeEntry.evidence.n_observations",
    )
    _expect_value_error(
        lambda: _entry(evidence={"source": "x", "reward_delta": "big"}),
        "KnowledgeEntry.evidence.reward_delta",
    )
    _expect_value_error(
        lambda: _entry(evidence={"source": "x", "n_observation": 1}),
        "KnowledgeEntry.evidence",
        "n_observation",
    )


def test_confidence_rejects_unknown_value():
    _expect_value_error(
        lambda: _entry(confidence="certain"), "KnowledgeEntry.confidence", "certain"
    )


def test_llm_gloss_must_be_string_or_none():
    _expect_value_error(lambda: _entry(llm_gloss=42), "KnowledgeEntry.llm_gloss")
    assert _entry(llm_gloss="a reading").llm_gloss == "a reading"


def test_created_at_rejects_non_iso_and_impossible_dates():
    _expect_value_error(
        lambda: _entry(created_at="17/09/2026"), "KnowledgeEntry.created_at", "YYYY-MM-DD"
    )
    _expect_value_error(
        lambda: _entry(created_at="2026-13-01"), "KnowledgeEntry.created_at", "2026-13-01"
    )
    _expect_value_error(lambda: _entry(created_at=None), "KnowledgeEntry.created_at")


def test_from_dict_rejects_unknown_key_by_name():
    data = _entry_dict()
    data["confidnce"] = "high"
    _expect_value_error(lambda: KnowledgeEntry.from_dict(data), "unknown key", "confidnce")


def test_from_dict_rejects_missing_required_key_by_name():
    data = _entry_dict()
    del data["created_at"]
    _expect_value_error(
        lambda: KnowledgeEntry.from_dict(data), "missing required key", "created_at"
    )


def test_from_dict_rejects_non_mapping():
    _expect_value_error(
        lambda: KnowledgeEntry.from_dict(["not", "a", "dict"]), "KnowledgeEntry", "mapping"
    )


# ---- schema: payload.recommendation convention ----------------------------


def test_recommendation_missing_params_raises():
    payload = {"recommendation": {"action": "reweight_training_samples"}}
    _expect_value_error(
        lambda: _entry(payload=payload), "KnowledgeEntry.payload.recommendation", "params"
    )


def test_recommendation_missing_action_raises():
    payload = {"recommendation": {"params": {}}}
    _expect_value_error(
        lambda: _entry(payload=payload), "KnowledgeEntry.payload.recommendation", "action"
    )


def test_recommendation_rejects_unknown_key_and_bad_note():
    payload = {"recommendation": {"action": "stop", "params": {}, "why": "x"}}
    _expect_value_error(
        lambda: _entry(payload=payload), "KnowledgeEntry.payload.recommendation", "why"
    )
    payload = {"recommendation": {"action": "stop", "params": {}, "note": 3}}
    _expect_value_error(
        lambda: _entry(payload=payload), "KnowledgeEntry.payload.recommendation.note"
    )
    payload = {"recommendation": {"action": "stop", "params": "dimension=peak"}}
    _expect_value_error(
        lambda: _entry(payload=payload), "KnowledgeEntry.payload.recommendation.params", "mapping"
    )


def test_recommendation_not_yet_available_accepted():
    payload = {
        "recommendation": {
            "action": RECOMMENDATION_NOT_YET_AVAILABLE,
            "params": {"factor": ">1"},
            "note": "planned",
        }
    }
    entry = _entry(payload=payload)
    assert entry.recommendation is not None
    assert entry.recommendation["action"] == "not_yet_available"


def test_payload_without_recommendation_is_free_json():
    entry = _entry(payload={"loss": "L = 1/N sum w_i l(yhat_i, y_i)", "nested": {"a": [1, 2]}})
    assert entry.payload["nested"]["a"] == [1, 2]


# ---- schema: RetrievalContext ---------------------------------------------


def test_retrieval_context_validates_single_values():
    _expect_value_error(
        lambda: RetrievalContext(phase="winter"), "RetrievalContext.phase", "winter"
    )
    _expect_value_error(lambda: RetrievalContext(season_week=0), "RetrievalContext.season_week")
    _expect_value_error(lambda: RetrievalContext(season_week=54), "RetrievalContext.season_week")
    _expect_value_error(lambda: RetrievalContext(model=""), "RetrievalContext.model")
    _expect_value_error(lambda: RetrievalContext(metric=""), "RetrievalContext.metric")
    ctx = RetrievalContext(phase="peak", model="xgboost_direct", metric="wis", season_week=12)
    assert not ctx.is_empty()
    assert RetrievalContext().is_empty()


# ---- curated YAML intake ---------------------------------------------------


def test_load_curated_dir_loads_lists_and_skips_underscore_files():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_yaml(directory, "a.yaml", [_entry_dict(id="a-one"), _entry_dict(id="a-two")])
        _write_yaml(directory, "b.yaml", [_entry_dict(id="b-one")])
        _write_yaml(
            directory, "_TEMPLATE.yaml", [_entry_dict(id="a-one")]
        )  # duplicate id, but skipped
        _write_yaml(directory, "_stub.yaml", {"not": "a list"})  # also skipped
        (directory / "notes.txt").write_text("ignored", encoding="utf-8")

        entries = load_curated_dir(directory)

    assert [e.id for e in entries] == ["a-one", "a-two", "b-one"]


def test_load_curated_dir_duplicate_id_names_both_locations():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_yaml(directory, "a.yaml", [_entry_dict(id="same-id")])
        _write_yaml(directory, "b.yaml", [_entry_dict(id="other"), _entry_dict(id="same-id")])

        message = _expect_value_error(
            lambda: load_curated_dir(directory), "b.yaml[1] same-id", "duplicate id", "a.yaml[0]"
        )
    assert "1 curated knowledge error" in message


def test_load_curated_dir_non_list_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_yaml(directory, "a.yaml", _entry_dict(id="mapping-not-list"))
        _expect_value_error(lambda: load_curated_dir(directory), "a.yaml", "must be a list", "dict")


def test_load_curated_dir_empty_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "a.yaml").write_text("", encoding="utf-8")
        _expect_value_error(lambda: load_curated_dir(directory), "a.yaml", "empty file")


def test_load_curated_dir_empty_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _expect_value_error(lambda: load_curated_dir(tmp), "no curated knowledge files")
    with tempfile.TemporaryDirectory() as tmp:
        _write_yaml(Path(tmp), "_TEMPLATE.yaml", [_entry_dict()])
        _expect_value_error(lambda: load_curated_dir(tmp), "no curated knowledge files")
    with tempfile.TemporaryDirectory() as tmp:
        _write_yaml(Path(tmp), "a.yaml", [])
        _expect_value_error(lambda: load_curated_dir(tmp), "contain no entries")


def test_load_curated_dir_missing_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _expect_value_error(lambda: load_curated_dir(Path(tmp) / "nope"), "does not exist")


def test_load_curated_dir_reports_file_index_id_and_reason():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bad = _entry_dict(id="bad-conf", confidence="certain")
        _write_yaml(directory, "a.yaml", [_entry_dict(id="good"), bad])
        _expect_value_error(
            lambda: load_curated_dir(directory),
            "a.yaml[1] bad-conf: KnowledgeEntry.confidence",
            "certain",
        )


def test_load_curated_dir_collects_errors_across_files():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_yaml(directory, "a.yaml", [_entry_dict(id="Bad Id")])
        _write_yaml(directory, "b.yaml", ["not a mapping"])
        message = _expect_value_error(
            lambda: load_curated_dir(directory),
            "2 curated knowledge error(s)",
            "a.yaml[0] Bad Id:",
            "b.yaml[0] ?:",
        )
    assert message.count("\n") == 2, message


def test_list_curated_files_rejects_misnamed_yaml_suffix():
    from agent.knowledge import list_curated_files

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _write_yaml(directory, "good.yaml", [_entry_dict(id="a")])
        _write_yaml(directory, "typo.yml", [_entry_dict(id="b")])
        _write_yaml(directory, "note.YAML", [_entry_dict(id="c")])
        _write_yaml(directory, "_stub.yml", [_entry_dict(id="d")])  # skipped prefix: fine

        message = _expect_value_error(
            lambda: list_curated_files(directory), "not loaded", "typo.yml", "note.YAML"
        )
        assert "_stub.yml" not in message
        _expect_value_error(lambda: load_curated_dir(directory), "typo.yml")

        (directory / "typo.yml").unlink()
        (directory / "note.YAML").unlink()
        assert [p.name for p in list_curated_files(directory)] == ["good.yaml"]


def test_load_curated_dir_yaml_parse_error_is_reported_with_file():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "a.yaml").write_text("- id: x\n  statement: [unclosed\n", encoding="utf-8")
        _expect_value_error(lambda: load_curated_dir(directory), "a.yaml", "YAML parse error")


# ---- store: writes ---------------------------------------------------------


def test_bank_creates_db_and_upsert_get_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "nested" / "knowledge.db"
        bank = KnowledgeBank(db_path)
        entry = _entry(
            llm_gloss="a gloss", evidence={"source": "s", "n_observations": 2, "reward_delta": -1.5}
        )

        bank.upsert(entry)

        assert db_path.exists()
        assert bank.get(entry.id) == entry
        assert bank.get("missing") is None
        assert bank.count() == 1


def test_bank_upsert_replaces_row_and_context_rows():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="e", context={"phase": ["peak"]}, statement="v1"))

        bank.upsert(_entry(id="e", context={"phase": ["decline"]}, statement="v2"))

        assert bank.count() == 1
        assert bank.get("e").statement == "v2"
        assert [e.id for e in bank.query(RetrievalContext(phase="peak"))] == []
        assert [e.id for e in bank.query(RetrievalContext(phase="decline"))] == ["e"]


def test_bank_upsert_rejects_non_entry():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        _expect_value_error(lambda: bank.upsert(_entry_dict()), "upsert", "KnowledgeEntry")


def test_delete_provenance_cascades_and_validates():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="c1", provenance="curated", context={"phase": ["peak"]}))
        bank.upsert(_entry(id="x1", provenance="experiential", context={"phase": ["peak"]}))

        removed = bank.delete_provenance("curated")

        assert removed == 1
        assert bank.count("curated") == 0
        assert bank.count("experiential") == 1
        assert [e.id for e in bank.query(RetrievalContext(phase="peak"))] == ["x1"]
        _expect_value_error(lambda: bank.delete_provenance("guess"), "provenance", "guess")


def test_rebuild_curated_is_idempotent_and_keeps_derived_rows():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "curated"
        directory.mkdir()
        _write_yaml(directory, "a.yaml", [_entry_dict(id="c-one"), _entry_dict(id="c-two")])
        bank = KnowledgeBank(Path(tmp) / "k.db")
        derived = _entry(id="d-one", provenance="derived", evidence={"source": "job"})
        bank.upsert(derived)
        bank.upsert(_entry(id="stale-curated", provenance="curated"))

        first = bank.rebuild_curated(directory)
        second = bank.rebuild_curated(directory)

        assert first == second == 2
        assert bank.count("curated") == 2
        assert bank.get("stale-curated") is None
        assert bank.count("derived") == 1
        assert bank.get("d-one") == derived
        assert bank.count() == 3


def test_rebuild_curated_bad_files_leave_db_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "curated"
        directory.mkdir()
        _write_yaml(directory, "a.yaml", [_entry_dict(id="c-one")])
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.rebuild_curated(directory)
        _write_yaml(directory, "b.yaml", [_entry_dict(id="broken", confidence="certain")])

        _expect_value_error(lambda: bank.rebuild_curated(directory), "b.yaml[0] broken")

        assert bank.count("curated") == 1
        assert bank.get("c-one") is not None


def test_upsert_rejects_provenance_change_for_existing_id():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="shared-id", provenance="curated"))

        _expect_value_error(
            lambda: bank.upsert(_entry(id="shared-id", provenance="derived")),
            "shared-id",
            "'curated'",
            "'derived'",
        )

        assert bank.get("shared-id").provenance == "curated"
        assert bank.count("curated") == 1 and bank.count("derived") == 0
        # Same provenance still replaces in place.
        bank.upsert(_entry(id="shared-id", provenance="curated", statement="Updated."))
        assert bank.get("shared-id").statement == "Updated."


def test_rebuild_curated_rolls_back_when_a_curated_id_collides_with_a_derived_row():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "curated"
        directory.mkdir()
        _write_yaml(directory, "a.yaml", [_entry_dict(id="cur-a"), _entry_dict(id="cur-b")])
        bank = KnowledgeBank.open(curated_dir=directory, db_path=Path(tmp) / "k.db")
        assert bank.count("curated") == 2
        bank.upsert(_entry(id="derived-x", provenance="derived"))
        _write_yaml(
            directory, "b.yaml", [_entry_dict(id="derived-x", statement="Stolen id.")]
        )

        _expect_value_error(lambda: bank.rebuild_curated(directory), "derived-x", "'derived'")

        # The failed rebuild left every row exactly as it was.
        assert bank.count("curated") == 2
        assert bank.get("derived-x").provenance == "derived"
        assert bank.get("derived-x").statement != "Stolen id."


def test_count_matching_reports_how_many_query_would_return_unlimited():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        for i in range(5):
            bank.upsert(_entry(id=f"peak-{i}", context={"phase": ["peak"]}))
        bank.upsert(_entry(id="onset-only", context={"phase": ["onset"]}))

        ctx = RetrievalContext(phase="peak")
        shown = bank.query(ctx, limit=2)

        assert len(shown) == 2
        assert bank.count_matching(ctx) == 5
        assert bank.count_matching(RetrievalContext()) == 6
        assert bank.count_matching(RetrievalContext(phase="decline")) == 0


def test_open_classmethod_constructs_and_rebuilds():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "curated"
        directory.mkdir()
        _write_yaml(directory, "a.yaml", [_entry_dict(id="c-one")])
        db_path = Path(tmp) / "k.db"

        bank = KnowledgeBank.open(curated_dir=directory, db_path=db_path)

        assert bank.db_path == db_path
        assert bank.count("curated") == 1


# ---- store: query semantics -----------------------------------------------


def test_query_unconstrained_entry_matches_any_context():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="always", context={}))

        everything = bank.query(RetrievalContext())
        specific = bank.query(
            RetrievalContext(phase="peak", model="nhits", metric="wis", season_week=7)
        )

        assert [e.id for e in everything] == ["always"]
        assert [e.id for e in specific] == ["always"]


def test_query_phase_filter_requires_membership():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="peak-only", context={"phase": ["peak", "approaching_peak"]}))
        bank.upsert(_entry(id="decline-only", context={"phase": ["decline"]}))
        bank.upsert(_entry(id="any-phase"))

        peak = bank.query(RetrievalContext(phase="peak"))
        decline = bank.query(RetrievalContext(phase="decline"))
        onset = bank.query(RetrievalContext(phase="onset"))

        assert [e.id for e in peak] == ["any-phase", "peak-only"]
        assert [e.id for e in decline] == ["any-phase", "decline-only"]
        assert [e.id for e in onset] == ["any-phase"]


def test_query_model_and_metric_filters_combine_with_and():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="xgb-wis", context={"model": ["xgboost_direct"], "metric": ["wis"]}))
        bank.upsert(_entry(id="xgb-any-metric", context={"model": ["xgboost_direct"]}))
        bank.upsert(_entry(id="nhits-wis", context={"model": ["nhits"], "metric": ["wis"]}))

        hits = bank.query(RetrievalContext(model="xgboost_direct", metric="mape"))

        assert [e.id for e in hits] == ["xgb-any-metric"]


def test_query_season_week_edges_are_inclusive():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="window", context={"season_week": [38, 46]}))

        assert [e.id for e in bank.query(RetrievalContext(season_week=37))] == []
        assert [e.id for e in bank.query(RetrievalContext(season_week=38))] == ["window"]
        assert [e.id for e in bank.query(RetrievalContext(season_week=46))] == ["window"]
        assert [e.id for e in bank.query(RetrievalContext(season_week=47))] == []


def test_query_limit_truncates_after_ordering():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="low", confidence="low"))
        bank.upsert(_entry(id="high", confidence="high"))
        bank.upsert(_entry(id="medium", confidence="medium"))

        top_two = bank.query(RetrievalContext(), limit=2)

        assert [e.id for e in top_two] == ["high", "medium"]
        _expect_value_error(lambda: bank.query(RetrievalContext(), limit=0), "limit")
        _expect_value_error(lambda: bank.query({"phase": "peak"}), "RetrievalContext")


def test_query_orders_curated_before_experiential_regardless_of_insertion():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(
            _entry(
                id="exp",
                provenance="experiential",
                confidence="high",
                evidence={"source": "run", "n_observations": 9},
            )
        )
        bank.upsert(
            _entry(
                id="der",
                provenance="derived",
                confidence="high",
                evidence={"source": "job", "n_observations": 9},
            )
        )
        bank.upsert(_entry(id="cur", provenance="curated", confidence="low"))

        ordered = bank.query(RetrievalContext())

        assert [e.id for e in ordered] == ["cur", "der", "exp"]


def test_query_orders_by_confidence_then_n_observations_then_id():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="z-low", confidence="low"))
        bank.upsert(_entry(id="b-high-none", confidence="high"))
        bank.upsert(_entry(id="a-high-none", confidence="high"))
        bank.upsert(
            _entry(id="c-high-3", confidence="high", evidence={"source": "s", "n_observations": 3})
        )
        bank.upsert(_entry(id="m-medium", confidence="medium"))

        ordered = bank.query(RetrievalContext())

        assert [e.id for e in ordered] == [
            "c-high-3",
            "a-high-none",
            "b-high-none",
            "m-medium",
            "z-low",
        ]


def test_list_filters_by_provenance_and_category_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        bank = KnowledgeBank(Path(tmp) / "k.db")
        bank.upsert(_entry(id="c-dyn", provenance="curated", category="domain_dynamics"))
        bank.upsert(_entry(id="c-model", provenance="curated", category="model_characteristics"))
        bank.upsert(
            _entry(id="x-model", provenance="experiential", category="model_characteristics")
        )

        assert [e.id for e in bank.list()] == ["c-dyn", "c-model", "x-model"]
        assert [e.id for e in bank.list(provenance="curated")] == ["c-dyn", "c-model"]
        assert [e.id for e in bank.list(category="model_characteristics")] == ["c-model", "x-model"]
        assert [
            e.id for e in bank.list(provenance="experiential", category="domain_dynamics")
        ] == []
        assert bank.count() == 3
        assert bank.count("experiential") == 1
        _expect_value_error(lambda: bank.list(category="misc"), "category", "misc")
        _expect_value_error(lambda: bank.count("guess"), "provenance", "guess")


# ---- render ----------------------------------------------------------------


def test_render_header_matches_design_doc():
    assert (
        DEFAULT_HEADER == "KNOWN FACTS (knowledge bank; [C]=curated [D]=derived [E]=experiential):"
    )


def test_render_curated_entry_with_recommendation_exact():
    entry = _entry(
        id="lambda-approaching-peak",
        statement="Loss weight lambda>1 on approaching-peak rows when the model fails at the peak.",
        payload={
            "recommendation": {
                "action": "reweight_training_samples",
                "params": {"dimension": "approaching_peak", "weight": ">1"},
            }
        },
    )

    text = render_known_facts([entry])

    assert text == (
        "KNOWN FACTS (knowledge bank; [C]=curated [D]=derived [E]=experiential):\n"
        "- [C, high] Loss weight lambda>1 on approaching-peak rows when the model fails at the"
        " peak."
        " -> try reweight_training_samples(dimension=approaching_peak, weight='>1')"
    )


def test_render_derived_entry_hides_observation_count():
    entry = _entry(
        id="onset-dates",
        provenance="derived",
        statement="Observed onsets: 2022 wk41, 2023 wk42, 2024 wk41, 2025 wk43; std ~0.9 wk.",
        evidence={"source": "job derive_onsets", "n_observations": 4},
    )

    text = render_known_facts([entry])

    assert (
        text.splitlines()[1]
        == "- [D, high] Observed onsets: 2022 wk41, 2023 wk42, 2024 wk41, 2025 wk43; std ~0.9 wk."
    )


def test_render_experiential_entry_with_run_count_exact():
    one_run = _entry(
        id="exp-one",
        provenance="experiential",
        confidence="low",
        statement="input_size=52 on NHITS regressed WIS ~4x vs 26 (decline).",
        evidence={"source": "run 20260903-xxxxxx", "n_observations": 1, "reward_delta": 301.4},
    )
    three_runs = _entry(
        id="exp-three",
        provenance="experiential",
        confidence="medium",
        statement="Three.",
        evidence={"source": "runs", "n_observations": 3},
    )
    unknown = _entry(
        id="exp-unknown",
        provenance="experiential",
        confidence="low",
        statement="Unknown count.",
        evidence={"source": "run"},
    )

    text = render_known_facts([one_run, three_runs, unknown])

    assert text.splitlines()[1:] == [
        "- [E, low, 1 run] input_size=52 on NHITS regressed WIS ~4x vs 26 (decline).",
        "- [E, medium, 3 runs] Three.",
        "- [E, low] Unknown count.",
    ]


def test_render_not_yet_available_and_gloss_markers():
    entry = _entry(
        id="smote",
        confidence="low",
        statement="SMOTE on failing-phase rows is expected to fail.",
        payload={
            "recommendation": {
                "action": RECOMMENDATION_NOT_YET_AVAILABLE,
                "params": {},
                "note": "planned arm",
            }
        },
        llm_gloss="oversampling a time series breaks autocorrelation",
    )

    line = render_known_facts([entry]).splitlines()[1]

    assert line == (
        "- [C, low] SMOTE on failing-phase rows is expected to fail. -> no adapter action yet"
        " [interpretation: oversampling a time series breaks autocorrelation]"
    )


def test_render_custom_header_and_order_preserved():
    first = _entry(id="first", statement="First.")
    second = _entry(id="second", statement="Second.")

    text = render_known_facts([second, first], header="FACTS:")

    assert text == "FACTS:\n- [C, high] Second.\n- [C, high] First."


def test_render_quotes_param_values_that_are_not_plain_tokens():
    entry = _entry(
        id="lambda-entry",
        statement="Weight the rows.",
        payload={
            "recommendation": {
                "action": "reweight_training_samples",
                "params": {
                    "dimension": "approaching_peak",
                    "weight": ">1, to be learned",
                    "weeks": [6, 12],
                    "factor": 2.5,
                },
            }
        },
    )

    line = render_known_facts([entry]).splitlines()[1]

    assert line == (
        "- [C, high] Weight the rows. -> try reweight_training_samples("
        "dimension=approaching_peak, weight='>1, to be learned', weeks=[6, 12], factor=2.5)"
    )


def test_render_omitted_count_appends_explicit_line():
    entry = _entry(statement="A fact.")

    text = render_known_facts([entry], omitted=3)
    one = render_known_facts([entry], omitted=1)

    assert text.splitlines()[-1] == (
        "- (3 more matching entries omitted; narrow the retrieval context)"
    )
    assert one.splitlines()[-1] == "- (1 more matching entry omitted; narrow the retrieval context)"
    assert render_known_facts([entry], omitted=0).count("omitted") == 0
    _expect_value_error(lambda: render_known_facts([entry], omitted=-1), "omitted")
    # No entries at all: the placeholder, never a dangling omitted line.
    assert render_known_facts([], omitted=4) == EMPTY_BLOCK


def test_render_empty_list_is_explicit_placeholder():
    assert render_known_facts([]) == EMPTY_BLOCK
    assert render_known_facts([]) == "(knowledge bank: no matching entries)"


# ---- seed files (knowledge/curated) ----------------------------------------

# The advisor's five peak-rectification actions (Teams 2026-09-27), in his order.
SEED_RECTIFICATION_IDS = (
    "rectify-peak-add-examples-other-seasons",
    "rectify-peak-oversample-windows",
    "rectify-peak-smote",
    "rectify-peak-shorter-window-on-divergence",
    "rectify-peak-loss-weight-approaching-peak",
)


def test_seed_files_parse_and_recommend_real_actions():
    from agent.adapters.flu_forecast import FluForecastAdapter

    entries = load_curated_dir(DEFAULT_CURATED_DIR)
    action_names = {a["name"] for a in FluForecastAdapter().get_available_actions()}

    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate seed ids"
    assert entries, "seed directory loaded no entries"
    for entry in entries:
        assert entry.provenance == "curated", f"{entry.id}: seed files are curated only"
        rec = entry.recommendation
        if rec is None:
            continue
        assert rec["action"] in action_names or rec["action"] == RECOMMENDATION_NOT_YET_AVAILABLE, (
            f"{entry.id}: recommendation.action {rec['action']!r} is not an adapter action"
        )


def test_seed_files_carry_the_five_rectification_actions():
    entries = {e.id: e for e in load_curated_dir(DEFAULT_CURATED_DIR)}

    for entry_id in SEED_RECTIFICATION_IDS:
        assert entry_id in entries, f"missing rectification seed {entry_id}"
        entry = entries[entry_id]
        assert entry.recommendation is not None, f"{entry_id}: no recommendation"
        assert set(entry.context["phase"]) == {"peak", "approaching_peak"}
        assert entry.entities["phase"] == "peak"
    assert entries["rectify-peak-smote"].confidence == "low"
    lambda_rec = entries["rectify-peak-loss-weight-approaching-peak"].recommendation
    assert lambda_rec["action"] == "reweight_training_samples"
    assert lambda_rec["params"]["dimension"] == "approaching_peak"


def test_seed_files_hold_the_migrated_adapter_domain_context():
    entries = {e.id: e for e in load_curated_dir(DEFAULT_CURATED_DIR)}

    assert entries["calendar-phase-onset"].statement == (
        "Onset phase (Oct-Nov): flu activity begins rising."
    )
    assert entries["calendar-phase-onset"].context["season_week"] == [40, 48]
    assert entries["calendar-phase-decline"].context["season_week"] == [5, 17]
    assert entries["bias-sign-convention"].category == "forecasts"
    for entry_id in ("calendar-phase-onset", "mape-quality-bands", "bias-sign-convention"):
        assert entries[entry_id].evidence["source"].startswith("migrated from agent/adapters/")


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
