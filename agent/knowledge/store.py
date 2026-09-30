"""SQLite knowledge bank (design doc `knowledge-bank-design.md` sections 4 and 6).

One SQLite file, `knowledge/knowledge.db` (gitignored via `*.db`), holds every
entry regardless of provenance. Curated rows are rebuilt from
`knowledge/curated/*.yaml` on `KnowledgeBank.open()`; derived and
experiential rows are written directly by jobs and the end-of-run
bank-writer (not in v1) and survive every rebuild.

Schema
------
    entries
        id               TEXT PRIMARY KEY
        provenance       TEXT      curated | derived | experiential
        category         TEXT
        statement        TEXT
        confidence       TEXT      low | medium | high
        created_at       TEXT      ISO date, authored
        updated_at       TEXT      ISO8601 UTC, set on every upsert
        n_observations   INTEGER   denormalised from evidence (nullable)
        reward_delta     REAL      denormalised from evidence (nullable)
        source           TEXT      denormalised from evidence
        season_week_lo   INTEGER   denormalised from context (nullable)
        season_week_hi   INTEGER   denormalised from context (nullable)
        entities_json    TEXT      JSON
        context_json     TEXT      JSON (the authoritative context)
        payload_json     TEXT      JSON
        evidence_json    TEXT      JSON
        llm_gloss        TEXT      nullable

    entry_context (entry_id, key, value)
        one row per value of the list-valued context keys (phase, model,
        metric); FK to entries with ON DELETE CASCADE. Retrieval is a SQL
        filter over this table, not a Python scan: "a database that can be
        queried" is the advisor's explicit ask.

Invariants
----------
- Rows go in only through `upsert`, which takes a validated `KnowledgeEntry`,
  so every row reads back as a valid entry.
- An id belongs to one provenance for life: upserting an entry whose id is
  already stored under a different provenance raises. Without this a derived
  row could silently take over a curated id (or vice versa) and the next
  rebuild would flip it back, discarding content with no trace.
- `rebuild_curated` is one transaction: delete curated rows, insert all
  loaded entries. A parse error leaves the DB untouched (files are loaded
  before the transaction opens). It is idempotent and never touches derived
  or experiential rows.
- `query` matches an entry when, for each key set in the `RetrievalContext`,
  the entry has no constraint on that key or contains the value
  (`season_week`: `lo <= week <= hi`). Ordering is total and deterministic:
  `TRUST_RANK` desc, `CONFIDENCE_RANK` desc, `n_observations` desc (NULL
  last), `updated_at` desc, `id` asc. `count_matching` runs the same WHERE
  without the limit, so a caller can tell when `query` truncated.
- Connections open with `timeout=DEFAULT_BUSY_TIMEOUT_S` and foreign keys on,
  so parallel pytest workers or a concurrent CLI never see "database is
  locked" and cascade deletes actually cascade.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.knowledge.curated import DEFAULT_CURATED_DIR, REPO_ROOT, load_curated_dir
from agent.knowledge.schema import (
    CATEGORIES,
    CONFIDENCES,
    LIST_CONTEXT_KEYS,
    PROVENANCES,
    KnowledgeEntry,
    RetrievalContext,
)

__all__ = [
    "CONFIDENCE_RANK",
    "DEFAULT_BUSY_TIMEOUT_S",
    "DEFAULT_DB_PATH",
    "DEFAULT_QUERY_LIMIT",
    "TRUST_RANK",
    "KnowledgeBank",
]

_log = logging.getLogger(__name__)

DEFAULT_DB_PATH = REPO_ROOT / "knowledge" / "knowledge.db"

# How long a connection waits on a lock before raising. Generous on purpose:
# a rebuild is milliseconds, so waiting always beats "database is locked".
DEFAULT_BUSY_TIMEOUT_S = 30.0

# Bounds the KNOWN FACTS block as the bank grows.
DEFAULT_QUERY_LIMIT = 30

# Higher ranks first. Human guardrails outrank machine observations.
TRUST_RANK: Mapping[str, int] = {"curated": 3, "derived": 2, "experiential": 1}
CONFIDENCE_RANK: Mapping[str, int] = {"high": 3, "medium": 2, "low": 1}
assert set(TRUST_RANK) == set(PROVENANCES), "TRUST_RANK must cover every provenance"
assert set(CONFIDENCE_RANK) == set(CONFIDENCES), "CONFIDENCE_RANK must cover every confidence"

_DDL = """
CREATE TABLE IF NOT EXISTS entries (
    id               TEXT PRIMARY KEY,
    provenance       TEXT NOT NULL,
    category         TEXT NOT NULL,
    statement        TEXT NOT NULL,
    confidence       TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    n_observations   INTEGER,
    reward_delta     REAL,
    source           TEXT NOT NULL,
    season_week_lo   INTEGER,
    season_week_hi   INTEGER,
    entities_json    TEXT NOT NULL,
    context_json     TEXT NOT NULL,
    payload_json     TEXT NOT NULL,
    evidence_json    TEXT NOT NULL,
    llm_gloss        TEXT
);
CREATE TABLE IF NOT EXISTS entry_context (
    entry_id  TEXT NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (entry_id, key, value)
);
CREATE INDEX IF NOT EXISTS idx_entries_provenance ON entries(provenance);
CREATE INDEX IF NOT EXISTS idx_entries_category ON entries(category);
CREATE INDEX IF NOT EXISTS idx_entry_context_key_value ON entry_context(key, value);
"""

_UPSERT_SQL = """
INSERT INTO entries (
    id, provenance, category, statement, confidence, created_at, updated_at,
    n_observations, reward_delta, source, season_week_lo, season_week_hi,
    entities_json, context_json, payload_json, evidence_json, llm_gloss
) VALUES (
    :id, :provenance, :category, :statement, :confidence, :created_at, :updated_at,
    :n_observations, :reward_delta, :source, :season_week_lo, :season_week_hi,
    :entities_json, :context_json, :payload_json, :evidence_json, :llm_gloss
)
ON CONFLICT(id) DO UPDATE SET
    provenance     = excluded.provenance,
    category       = excluded.category,
    statement      = excluded.statement,
    confidence     = excluded.confidence,
    created_at     = excluded.created_at,
    updated_at     = excluded.updated_at,
    n_observations = excluded.n_observations,
    reward_delta   = excluded.reward_delta,
    source         = excluded.source,
    season_week_lo = excluded.season_week_lo,
    season_week_hi = excluded.season_week_hi,
    entities_json  = excluded.entities_json,
    context_json   = excluded.context_json,
    payload_json   = excluded.payload_json,
    evidence_json  = excluded.evidence_json,
    llm_gloss      = excluded.llm_gloss
"""

# One clause per list-valued context key: unconstrained OR contains the value.
_LIST_KEY_MATCH_SQL = """
(
    NOT EXISTS (SELECT 1 FROM entry_context c WHERE c.entry_id = e.id AND c.key = :{key}_key)
    OR EXISTS (SELECT 1 FROM entry_context c
               WHERE c.entry_id = e.id AND c.key = :{key}_key AND c.value = :{key}_value)
)
"""

_SEASON_WEEK_MATCH_SQL = """
(
    e.season_week_lo IS NULL
    OR (e.season_week_lo <= :season_week AND :season_week <= e.season_week_hi)
)
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rank_case(column: str, ranks: Mapping[str, int]) -> str:
    """`CASE column WHEN 'x' THEN n ... ELSE 0 END` from a constant rank table."""
    whens = " ".join(f"WHEN '{value}' THEN {rank}" for value, rank in ranks.items())
    return f"CASE {column} {whens} ELSE 0 END"


_ORDER_BY_SQL = (
    f"ORDER BY {_rank_case('e.provenance', TRUST_RANK)} DESC, "
    f"{_rank_case('e.confidence', CONFIDENCE_RANK)} DESC, "
    "e.n_observations DESC, e.updated_at DESC, e.id ASC"
)


def _entry_to_row(entry: KnowledgeEntry, updated_at: str) -> dict[str, Any]:
    window = entry.season_week
    return {
        "id": entry.id,
        "provenance": entry.provenance,
        "category": entry.category,
        "statement": entry.statement,
        "confidence": entry.confidence,
        "created_at": entry.created_at,
        "updated_at": updated_at,
        "n_observations": entry.evidence.get("n_observations"),
        "reward_delta": entry.evidence.get("reward_delta"),
        "source": entry.evidence["source"],
        "season_week_lo": window[0] if window else None,
        "season_week_hi": window[1] if window else None,
        "entities_json": json.dumps(dict(entry.entities), default=str),
        "context_json": json.dumps(dict(entry.context), default=str),
        "payload_json": json.dumps(dict(entry.payload), default=str),
        "evidence_json": json.dumps(dict(entry.evidence), default=str),
        "llm_gloss": entry.llm_gloss,
    }


def _row_to_entry(row: sqlite3.Row) -> KnowledgeEntry:
    # Re-validating on read is cheap and guarantees a row can never leak a
    # shape the schema would reject.
    return KnowledgeEntry.from_dict(
        {
            "id": row["id"],
            "provenance": row["provenance"],
            "category": row["category"],
            "statement": row["statement"],
            "entities": json.loads(row["entities_json"]),
            "context": json.loads(row["context_json"]),
            "payload": json.loads(row["payload_json"]),
            "evidence": json.loads(row["evidence_json"]),
            "confidence": row["confidence"],
            "llm_gloss": row["llm_gloss"],
            "created_at": row["created_at"],
        }
    )


def _check_provenance(provenance: str | None) -> None:
    if provenance is not None and provenance not in PROVENANCES:
        raise ValueError(f"provenance: expected one of {list(PROVENANCES)}, got {provenance!r}")


def _check_category(category: str | None) -> None:
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"category: expected one of {list(CATEGORIES)}, got {category!r}")


class KnowledgeBank:
    """The SQLite-backed bank. Construct directly for an empty/existing DB, or
    use `KnowledgeBank.open()` to also (re)load the curated YAML files.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def open(
        cls,
        curated_dir: str | Path = DEFAULT_CURATED_DIR,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> KnowledgeBank:
        """Construct and rebuild curated rows from `curated_dir`: the one-call entry point."""
        bank = cls(db_path)
        bank.rebuild_curated(curated_dir)
        return bank

    # ------------------------------------------------------------------ connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=DEFAULT_BUSY_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------ writes

    @staticmethod
    def _upsert_in(conn: sqlite3.Connection, entry: KnowledgeEntry, updated_at: str) -> None:
        existing = conn.execute(
            "SELECT provenance FROM entries WHERE id = ?", (entry.id,)
        ).fetchone()
        if existing is not None and existing["provenance"] != entry.provenance:
            raise ValueError(
                f"upsert: id {entry.id!r} is already stored with provenance "
                f"{existing['provenance']!r}; refusing to overwrite it as "
                f"{entry.provenance!r} (ids belong to one provenance for life)"
            )
        conn.execute(_UPSERT_SQL, _entry_to_row(entry, updated_at))
        conn.execute("DELETE FROM entry_context WHERE entry_id = ?", (entry.id,))
        conn.executemany(
            "INSERT INTO entry_context (entry_id, key, value) VALUES (?, ?, ?)",
            [
                (entry.id, key, value)
                for key in LIST_CONTEXT_KEYS
                for value in entry.context.get(key, [])
            ],
        )

    def upsert(self, entry: KnowledgeEntry) -> None:
        """Insert or replace one entry (and its context rows); bumps `updated_at`.

        Raises ValueError when the id exists under another provenance.
        """
        if not isinstance(entry, KnowledgeEntry):
            raise ValueError(f"upsert: expected a KnowledgeEntry, got {type(entry).__name__}")
        with self._connect() as conn:
            self._upsert_in(conn, entry, _now_iso())

    def delete_provenance(self, provenance: str) -> int:
        """Delete every row of one provenance; returns the number removed."""
        _check_provenance(provenance)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM entries WHERE provenance = ?", (provenance,))
            return int(cur.rowcount)

    def rebuild_curated(self, curated_dir: str | Path = DEFAULT_CURATED_DIR) -> int:
        """Replace all curated rows with the YAML files' entries, in one transaction.

        Files are parsed before the transaction opens, so a bad file leaves
        the DB exactly as it was; so does a curated id that collides with a
        derived/experiential row (the transaction rolls back). Returns the
        number of curated entries loaded.
        """
        entries = load_curated_dir(curated_dir)
        updated_at = _now_iso()
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE provenance = 'curated'")
            for entry in entries:
                self._upsert_in(conn, entry, updated_at)
        _log.info("Rebuilt %d curated entries into %s", len(entries), self.db_path)
        return len(entries)

    # ------------------------------------------------------------------ reads

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return _row_to_entry(row) if row is not None else None

    def list(
        self, provenance: str | None = None, category: str | None = None
    ) -> list[KnowledgeEntry]:
        """All entries, optionally filtered, in the same order `query` uses."""
        _check_provenance(provenance)
        _check_category(category)
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if provenance is not None:
            clauses.append("e.provenance = :provenance")
            params["provenance"] = provenance
        if category is not None:
            clauses.append("e.category = :category")
            params["category"] = category
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT e.* FROM entries e {where} {_ORDER_BY_SQL}", params
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def count(self, provenance: str | None = None) -> int:
        _check_provenance(provenance)
        with self._connect() as conn:
            if provenance is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM entries WHERE provenance = ?", (provenance,)
                ).fetchone()
        return int(row["n"])

    @staticmethod
    def _match_where(ctx: RetrievalContext) -> tuple[str, dict[str, Any]]:
        """The WHERE clause + bound params implementing the match rule for `ctx`."""
        if not isinstance(ctx, RetrievalContext):
            raise ValueError(f"query: expected a RetrievalContext, got {type(ctx).__name__}")
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for key in LIST_CONTEXT_KEYS:
            value = getattr(ctx, key)
            if value is None:
                continue
            clauses.append(_LIST_KEY_MATCH_SQL.format(key=key))
            params[f"{key}_key"] = key
            params[f"{key}_value"] = value
        if ctx.season_week is not None:
            clauses.append(_SEASON_WEEK_MATCH_SQL)
            params["season_week"] = ctx.season_week
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def count_matching(self, ctx: RetrievalContext) -> int:
        """How many entries `query(ctx)` would return with no limit."""
        where, params = self._match_where(ctx)
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM entries e {where}", params).fetchone()
        return int(row["n"])

    def query(
        self, ctx: RetrievalContext, *, limit: int = DEFAULT_QUERY_LIMIT
    ) -> list[KnowledgeEntry]:
        """Entries relevant to `ctx`, ranked; see the module docstring for semantics."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"query: limit must be an integer >= 1, got {limit!r}")
        where, params = self._match_where(ctx)
        params["limit"] = limit
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT e.* FROM entries e {where} {_ORDER_BY_SQL} LIMIT :limit", params
            ).fetchall()
        return [_row_to_entry(r) for r in rows]
