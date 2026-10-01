"""Knowledge-bank entry schema (design doc `knowledge-bank-design.md` section 3).

One schema for all three provenances. Curated entries are authored in YAML in
exactly this shape (`knowledge/curated/*.yaml`); derived and experiential
entries are the same records written by code. Everything the bank stores,
queries, or renders passes through `KnowledgeEntry`, so this module is the
single place where "what is a valid entry" is decided.

Invariants
----------
- A `KnowledgeEntry` that exists is valid. `__post_init__` validates every
  field and raises a `ValueError` that names the field and the offending
  value; nothing downstream re-checks shape.
- `from_dict` is strict: an unknown key (at the entry level, inside
  `evidence`, or inside `payload.recommendation`) raises with the key name.
  A silently ignored typo in a YAML file is worse than a loud error.
- `to_dict()` round-trips: `KnowledgeEntry.from_dict(e.to_dict()) == e`, and
  the dict is plain JSON-serialisable data (deep-copied, never aliased).
- `llm_gloss` is interpretation, never a retrieval key. Retrieval keys are
  exactly the `context` keys (`CONTEXT_KEYS`).
- Two YAML conveniences are normalised rather than rejected because the
  design doc's own example relies on them: a folded `statement: >` scalar
  carries a trailing newline (edge whitespace is stripped; interior newlines
  are still rejected), and an unquoted `created_at: 2026-09-17` parses as a
  `datetime.date` (converted to its ISO string). A `datetime.datetime` (an
  unquoted YAML timestamp such as `2026-09-30 10:00:00`) is rejected: the
  field is a date, and its ISO form would fail the round-trip on read-back.
- Optional sections (`entities`, `context`, `payload`, `llm_gloss`) may be
  absent or `null`; any other value must have the right type. A falsy typo
  such as `payload: []` or `entities: ""` is an error, not an empty section.
- `payload` must be plain JSON data (`json.dumps` without a fallback), so a
  stored entry always reads back equal to the authored one.
- Entries hash by `id` alone (the mapping fields are excluded from the
  generated `__hash__`), so they can live in sets and as dict keys; equality
  still compares every field.
- `KNOWN_PHASES` accepts both phase vocabularies in the repo (calendar phases
  from `PhaseEvaluator.PHASE_MAP`, curve phases from `phase_segmentation`)
  plus `approaching_peak`. They are deliberately not unified here; that is an
  open question for the lab.
- `payload` is free JSON, with one validated convention:
  `payload.recommendation = {action, params, note?}` for "what to do"
  entries. `action` is an adapter action name or the literal
  `RECOMMENDATION_NOT_YET_AVAILABLE`; the schema is domain-agnostic, so it
  checks the shape, not the adapter catalogue (the seed test does that).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from datetime import date, datetime
from typing import Any

__all__ = [
    "CATEGORIES",
    "CONFIDENCES",
    "CONTEXT_KEYS",
    "ENTITY_KEYS",
    "ENTRY_ID_PATTERN",
    "EVIDENCE_KEYS",
    "KNOWN_PHASES",
    "LIST_CONTEXT_KEYS",
    "PROVENANCES",
    "RECOMMENDATION_KEYS",
    "RECOMMENDATION_NOT_YET_AVAILABLE",
    "SEASON_WEEK_MAX",
    "SEASON_WEEK_MIN",
    "KnowledgeEntry",
    "RetrievalContext",
]

# ---- vocabularies ----------------------------------------------------------

PROVENANCES = ("curated", "derived", "experiential")
CATEGORIES = ("model_characteristics", "input_data", "forecasts", "domain_dynamics")
CONFIDENCES = ("low", "medium", "high")

# Calendar phases (PhaseEvaluator.PHASE_MAP) + curve phases (phase_segmentation)
# + the approaching-peak label used by the peak-rectification experiment.
KNOWN_PHASES = ("onset", "peak", "decline", "surge", "plateau", "approaching_peak", "off_season")

# Typed references: the graph-ready part of an entry (design section 7).
ENTITY_KEYS = ("model", "phase", "data_source", "season", "metric")

# Retrieval keys: WHEN an entry is relevant. Absence of a key = applies always.
CONTEXT_KEYS = ("phase", "model", "metric", "season_week")
LIST_CONTEXT_KEYS = ("phase", "model", "metric")

EVIDENCE_KEYS = ("source", "n_observations", "reward_delta")
RECOMMENDATION_KEYS = ("action", "params", "note")
RECOMMENDATION_NOT_YET_AVAILABLE = "not_yet_available"

SEASON_WEEK_MIN = 1
SEASON_WEEK_MAX = 53

ENTRY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ENTRY_KEYS = (
    "id",
    "provenance",
    "category",
    "statement",
    "entities",
    "context",
    "payload",
    "evidence",
    "confidence",
    "llm_gloss",
    "created_at",
)
_REQUIRED_ENTRY_KEYS = (
    "id",
    "provenance",
    "category",
    "statement",
    "evidence",
    "confidence",
    "created_at",
)


# ---- small validators ------------------------------------------------------
# Each returns the normalised value or raises ValueError naming `where`.


def _require_str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _require_choice(where: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{where}: expected one of {list(choices)}, got {value!r}")
    return value


def _require_mapping(where: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__} {value!r}")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{where}: keys must be strings, got {key!r}")
    return dict(value)


def _none_to_empty(value: Any) -> Any:
    # Only an absent/null section means "empty"; a falsy typo ('' , [], 0)
    # must reach _require_mapping and fail loudly.
    return {} if value is None else value


def _reject_unknown_keys(where: str, value: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    unknown = [k for k in value if k not in allowed]
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed keys are {list(allowed)}")


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; a YAML `true` must not pass as a count.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_id(value: Any) -> str:
    if not isinstance(value, str) or not ENTRY_ID_PATTERN.match(value):
        raise ValueError(
            f"KnowledgeEntry.id: expected a slug matching {ENTRY_ID_PATTERN.pattern!r}, "
            f"got {value!r}"
        )
    return value


def _validate_statement(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"KnowledgeEntry.statement: expected a string, got {value!r}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"KnowledgeEntry.statement: must be non-empty, got {value!r}")
    if "\n" in stripped or "\r" in stripped:
        raise ValueError(f"KnowledgeEntry.statement: must be a single line, got {value!r}")
    return stripped


def _validate_phase(where: str, value: Any) -> str:
    if value not in KNOWN_PHASES:
        raise ValueError(f"{where}: unknown phase {value!r}; known phases are {list(KNOWN_PHASES)}")
    return value


def _validate_season_week(where: str, value: Any) -> int:
    if not _is_int(value) or not (SEASON_WEEK_MIN <= value <= SEASON_WEEK_MAX):
        raise ValueError(
            f"{where}: expected an integer in [{SEASON_WEEK_MIN}, {SEASON_WEEK_MAX}], got {value!r}"
        )
    return value


def _validate_entities(value: Any) -> dict[str, str]:
    where = "KnowledgeEntry.entities"
    entities = _require_mapping(where, value)
    _reject_unknown_keys(where, entities, ENTITY_KEYS)
    for key, val in entities.items():
        _require_str(f"{where}.{key}", val)
    if "phase" in entities:
        _validate_phase(f"{where}.phase", entities["phase"])
    return entities


def _validate_context(value: Any) -> dict[str, Any]:
    where = "KnowledgeEntry.context"
    context = _require_mapping(where, value)
    _reject_unknown_keys(where, context, CONTEXT_KEYS)
    normalised: dict[str, Any] = {}
    for key in LIST_CONTEXT_KEYS:
        if key not in context:
            continue
        values = context[key]
        if not isinstance(values, (list, tuple)) or isinstance(values, str):
            raise ValueError(f"{where}.{key}: expected a list of strings, got {values!r}")
        if not values:
            raise ValueError(f"{where}.{key}: empty list; omit the key to mean 'applies always'")
        for item in values:
            _require_str(f"{where}.{key}", item)
            if key == "phase":
                _validate_phase(f"{where}.phase", item)
        if len(set(values)) != len(values):
            raise ValueError(f"{where}.{key}: duplicate values in {list(values)!r}")
        normalised[key] = list(values)
    if "season_week" in context:
        window = context["season_week"]
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(f"{where}.season_week: expected [lo, hi], got {window!r}")
        lo = _validate_season_week(f"{where}.season_week[0]", window[0])
        hi = _validate_season_week(f"{where}.season_week[1]", window[1])
        if lo > hi:
            raise ValueError(f"{where}.season_week: lo must be <= hi, got [{lo}, {hi}]")
        normalised["season_week"] = [lo, hi]
    return normalised


def _validate_recommendation(value: Any) -> dict[str, Any]:
    where = "KnowledgeEntry.payload.recommendation"
    rec = _require_mapping(where, value)
    _reject_unknown_keys(where, rec, RECOMMENDATION_KEYS)
    if "action" not in rec:
        raise ValueError(f"{where}: missing required key 'action'")
    _require_str(f"{where}.action", rec["action"])
    if "params" not in rec:
        raise ValueError(
            f"{where}: missing required key 'params' (use {{}} when the action takes none)"
        )
    _require_mapping(f"{where}.params", rec["params"])
    if "note" in rec:
        _require_str(f"{where}.note", rec["note"])
    return rec


def _validate_payload(value: Any) -> dict[str, Any]:
    where = "KnowledgeEntry.payload"
    payload = _require_mapping(where, value)
    if "recommendation" in payload:
        _validate_recommendation(payload["recommendation"])
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{where}: must be plain JSON data (quote dates and other scalars in YAML); "
            f"{exc}"
        ) from exc
    return payload


def _validate_evidence(value: Any) -> dict[str, Any]:
    where = "KnowledgeEntry.evidence"
    evidence = _require_mapping(where, value)
    _reject_unknown_keys(where, evidence, EVIDENCE_KEYS)
    if "source" not in evidence:
        raise ValueError(f"{where}: missing required key 'source'")
    _require_str(f"{where}.source", evidence["source"])
    n_obs = evidence.get("n_observations")
    if n_obs is not None and (not _is_int(n_obs) or n_obs < 0):
        raise ValueError(f"{where}.n_observations: expected an integer >= 0 or null, got {n_obs!r}")
    delta = evidence.get("reward_delta")
    if delta is not None and not _is_number(delta):
        raise ValueError(f"{where}.reward_delta: expected a number or null, got {delta!r}")
    normalised = dict(evidence)
    normalised.setdefault("n_observations", None)
    normalised["reward_delta"] = float(delta) if delta is not None else None
    return normalised


def _validate_created_at(value: Any) -> str:
    where = "KnowledgeEntry.created_at"
    if isinstance(value, datetime):
        # datetime is a subclass of date; an unquoted YAML timestamp would
        # otherwise be stored as '2026-09-30T10:00:00' and fail on read-back.
        raise ValueError(
            f"{where}: expected an ISO date 'YYYY-MM-DD', got a timestamp {value!r} "
            "(drop the time of day)"
        )
    if isinstance(value, date):
        # yaml.safe_load turns an unquoted 2026-09-17 into a date object.
        return value.isoformat()
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.match(value):
        raise ValueError(f"{where}: expected an ISO date 'YYYY-MM-DD', got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{where}: not a real calendar date: {value!r} ({exc})") from exc
    return value


# ---- records ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """One fact in the bank. See the module docstring for field semantics."""

    id: str
    provenance: str
    category: str
    statement: str
    evidence: Mapping[str, Any] = field(hash=False)
    confidence: str
    created_at: str
    # Mapping fields are excluded from the generated __hash__ (dicts are not
    # hashable); `id` is the primary key, so hashing by the scalar fields is safe.
    entities: Mapping[str, str] = field(default_factory=dict, hash=False)
    context: Mapping[str, Any] = field(default_factory=dict, hash=False)
    payload: Mapping[str, Any] = field(default_factory=dict, hash=False)
    llm_gloss: str | None = None

    def __post_init__(self) -> None:
        # Frozen + slots: normalised values go in through object.__setattr__.
        _set = object.__setattr__
        _set(self, "id", _validate_id(self.id))
        _set(
            self,
            "provenance",
            _require_choice("KnowledgeEntry.provenance", self.provenance, PROVENANCES),
        )
        _set(
            self, "category", _require_choice("KnowledgeEntry.category", self.category, CATEGORIES)
        )
        _set(self, "statement", _validate_statement(self.statement))
        _set(self, "entities", _validate_entities(self.entities))
        _set(self, "context", _validate_context(self.context))
        _set(self, "payload", copy.deepcopy(_validate_payload(self.payload)))
        _set(self, "evidence", _validate_evidence(self.evidence))
        _set(
            self,
            "confidence",
            _require_choice("KnowledgeEntry.confidence", self.confidence, CONFIDENCES),
        )
        if self.llm_gloss is not None:
            _require_str("KnowledgeEntry.llm_gloss", self.llm_gloss)
        _set(self, "created_at", _validate_created_at(self.created_at))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KnowledgeEntry:
        """Build from the YAML/JSON shape. Unknown or missing keys raise by name."""
        where = "KnowledgeEntry"
        entry = _require_mapping(where, data)
        _reject_unknown_keys(where, entry, _ENTRY_KEYS)
        missing = [k for k in _REQUIRED_ENTRY_KEYS if k not in entry]
        if missing:
            raise ValueError(f"{where}: missing required key(s) {missing}")
        return cls(
            id=entry["id"],
            provenance=entry["provenance"],
            category=entry["category"],
            statement=entry["statement"],
            entities=_none_to_empty(entry.get("entities")),
            context=_none_to_empty(entry.get("context")),
            payload=_none_to_empty(entry.get("payload")),
            evidence=entry["evidence"],
            confidence=entry["confidence"],
            llm_gloss=entry.get("llm_gloss"),
            created_at=entry["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain, deep-copied, JSON-serialisable dict in schema key order."""
        return {
            "id": self.id,
            "provenance": self.provenance,
            "category": self.category,
            "statement": self.statement,
            "entities": dict(self.entities),
            "context": copy.deepcopy(dict(self.context)),
            "payload": copy.deepcopy(dict(self.payload)),
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
            "llm_gloss": self.llm_gloss,
            "created_at": self.created_at,
        }

    @property
    def season_week(self) -> tuple[int, int] | None:
        """Inclusive [lo, hi] week-of-season window, or None when unconstrained."""
        window = self.context.get("season_week")
        return (window[0], window[1]) if window is not None else None

    @property
    def recommendation(self) -> Mapping[str, Any] | None:
        """`payload.recommendation` when the entry carries one (validated shape)."""
        return self.payload.get("recommendation")


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    """The situation the loop is in: single values, one per `CONTEXT_KEYS` key.

    `None` means "no constraint on this key". An entry matches a context when,
    for every key set here, the entry either has no constraint on that key or
    contains the value (`season_week`: `lo <= week <= hi`).
    """

    phase: str | None = None
    model: str | None = None
    metric: str | None = None
    season_week: int | None = None

    def __post_init__(self) -> None:
        if self.phase is not None:
            _validate_phase("RetrievalContext.phase", self.phase)
        if self.model is not None:
            _require_str("RetrievalContext.model", self.model)
        if self.metric is not None:
            _require_str("RetrievalContext.metric", self.metric)
        if self.season_week is not None:
            _validate_season_week("RetrievalContext.season_week", self.season_week)

    def is_empty(self) -> bool:
        """True when nothing is constrained (every entry matches)."""
        return all(v is None for v in (self.phase, self.model, self.metric, self.season_week))
