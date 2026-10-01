"""Experiential entries from the peak-rectification experiment log (design doc section 5).

The design doc's write path says experiential knowledge is written
*deterministically* from logged (state, action, reward) rows: no LLM touches
the statement, the numbers, or the retrieval keys, so a retrieved [E] fact
can never be a hallucination. The orchestrator's end-of-run bank-writer is
not built yet; this module is the first concrete instance of that write path,
fed by `scripts/experiments/peak_rectification.py`, whose
`outputs/experiments/peak_rectification/log.jsonl` already has exactly the
(state, action, reward) shape the design calls for.

Row shape consumed (one JSON object per line, written by the harness)
----------------------------------------------------------------------
    run_at        ISO8601 timestamp
    config_id     e.g. "baseline", "lambda_3", "window_12"
    config_delta  {dotted config path: value}; {} marks the baseline row
    state         {model, target_phase, metric, split{eval_start_date, ...}, n_cutoffs}
    metrics       {overall, by_phase{phase: {bias, coverage_95, ...}}, by_horizon}
    reward        {metric, phase, baseline_value, value, delta, better, guard?}

Invariants
----------
- Deterministic: the same rows and `created_at` always produce byte-identical
  entries. No clock, no randomness, no LLM. `created_at` is a parameter the
  caller (the CLI) fills with today's date; a library function must not read
  the clock, or re-imports would never compare equal.
- One entry per non-baseline row. The baseline row carries no action and
  becomes no entry; it supplies the "before" numbers (bias, coverage) that
  the row's own `reward` does not carry. A row pairs with the baseline row
  that has the same `state` and whose `reward.value` equals the row's
  `reward.baseline_value`: that number is copied from the baseline by the
  harness, so it is the link the data actually carries. `run_at` is NOT an
  ordering key (arms run concurrently, so a config row can land before its
  own baseline row); it only breaks ties when several baselines match,
  which happens when the harness is re-run with an identical baseline into
  the same appended log. A row with no matching baseline is an error, not a
  silent gap.
- Entry id = `exp-<config_id>-<eval-start year>` with the config id
  normalised to the schema's slug alphabet (`lambda_1.5` -> `lambda-1-5`).
  Re-importing the same log therefore upserts the same ids with the same
  content: the import is idempotent. Re-running the harness and importing
  again replaces each entry with the newer observation; merging repeated
  observations on (entities, action) into one entry with `n_observations > 1`
  is the end-of-run bank-writer's job and is not done here.
- Confidence is always `SINGLE_OBSERVATION_CONFIDENCE`: one experiment run is
  one observation, whatever the effect size (design section 5 step 3).
- The action description is derived from `config_delta` by a small, explicit
  table (`_describe_delta`): the two reweighting paths and the training
  window map onto adapter action names; anything else falls back to the raw
  `path=value` and the `set_config` action so an unforeseen path is still
  recorded faithfully rather than rejected.
- Every missing or malformed field raises `ValueError` naming the row
  (`config_id`) and the field. A bank with a half-imported log is worse than
  an import that stopped loudly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.knowledge.schema import KnowledgeEntry
from agent.knowledge.store import KnowledgeBank

__all__ = [
    "DEFAULT_EXPERIMENT_LOG",
    "ENTRY_ID_PREFIX",
    "EXPERIMENT_CATEGORY",
    "SINGLE_OBSERVATION_CONFIDENCE",
    "ImportResult",
    "import_experiment_log",
    "read_experiment_log",
    "rows_to_entries",
]

DEFAULT_EXPERIMENT_LOG = Path("outputs/experiments/peak_rectification/log.jsonl")

# Design section 5 step 3: a single observation is low confidence, full stop.
SINGLE_OBSERVATION_CONFIDENCE = "low"
SINGLE_OBSERVATION_COUNT = 1

ENTRY_ID_PREFIX = "exp"
EXPERIMENT_CATEGORY = "model_characteristics"
EXPERIMENT_PROVENANCE = "experiential"

# Dotted config paths the harness writes, mapped onto adapter action names.
CONFIG_PATH_APPROACHING_PEAK = "sample_weights.approaching_peak"
CONFIG_PATH_BY_PHASE_PREFIX = "sample_weights.by_phase."
CONFIG_PATH_TRAIN_WINDOW = "data.train_window_weeks"
ACTION_REWEIGHT = "reweight_training_samples"
ACTION_TRAINING_WINDOW = "set_training_window"
ACTION_SET_CONFIG = "set_config"
ACTION_COMPOSITE = "composite"

# Statement number formats. The harness already rounds; these fix the width
# so two imports of the same row can never differ in the printed digits.
_METRIC_FORMAT = "{:.2f}"
_DELTA_FORMAT = "{:+.2f}"
_BIAS_FORMAT = "{:.1f}"
_COVERAGE_FORMAT = "{:.3f}"

_REQUIRED_ROW_KEYS = ("run_at", "config_id", "config_delta", "state", "metrics", "reward")
_REQUIRED_STATE_KEYS = ("model", "target_phase", "metric", "split")
_REQUIRED_REWARD_KEYS = ("metric", "phase", "baseline_value", "value", "delta", "better")

_SLUG_SEPARATOR = "-"
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ImportResult:
    """What one `import_experiment_log` call did; `ids` are in log order."""

    n_rows: int
    n_entries: int
    ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ActionDescription:
    """One config path turned into words and into an adapter-shaped action."""

    text: str
    name: str
    params: dict[str, Any]


# ---- row access -------------------------------------------------------------


def _require_keys(where: str, mapping: Any, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(mapping).__name__} {mapping!r}")
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")
    return mapping


def _row_label(row: Mapping[str, Any], index: int) -> str:
    config_id = row.get("config_id") if isinstance(row, Mapping) else None
    return f"row[{index}] {config_id if config_id is not None else '?'}"


def _validate_row(row: Any, index: int) -> Mapping[str, Any]:
    where = _row_label(row, index)
    _require_keys(where, row, _REQUIRED_ROW_KEYS)
    if not isinstance(row["config_delta"], Mapping):
        raise ValueError(f"{where}.config_delta: expected a mapping, got {row['config_delta']!r}")
    _require_keys(f"{where}.state", row["state"], _REQUIRED_STATE_KEYS)
    _require_keys(f"{where}.state.split", row["state"]["split"], ("eval_start_date",))
    _require_keys(f"{where}.reward", row["reward"], _REQUIRED_REWARD_KEYS)
    _require_keys(f"{where}.metrics", row["metrics"], ("by_phase",))
    if not isinstance(row["run_at"], str) or not row["run_at"]:
        raise ValueError(
            f"{where}.run_at: expected a non-empty ISO timestamp, got {row['run_at']!r}"
        )
    return row


def _is_baseline(row: Mapping[str, Any]) -> bool:
    return len(row["config_delta"]) == 0


def _parse_run_at(row: Mapping[str, Any], where: str) -> datetime:
    try:
        return datetime.fromisoformat(row["run_at"])
    except ValueError as exc:
        raise ValueError(f"{where}.run_at: not an ISO8601 timestamp: {row['run_at']!r}") from exc


def _matching_baseline(
    row: Mapping[str, Any], baselines: Sequence[Mapping[str, Any]], where: str
) -> Mapping[str, Any]:
    """The baseline row this row was scored against; see the module docstring."""
    candidates = [
        b
        for b in baselines
        if b["state"] == row["state"] and b["reward"]["value"] == row["reward"]["baseline_value"]
    ]
    if not candidates:
        raise ValueError(
            f"{where}: no baseline row with the same state and reward.value == "
            f"{row['reward']['baseline_value']!r} (baseline reward.value seen: "
            f"{[b['reward']['value'] for b in baselines]})"
        )
    if len(candidates) == 1:
        return candidates[0]
    row_at = _parse_run_at(row, where)
    # Nearest in time wins; `min` keeps the first on an exact tie, so the
    # choice is stable across imports of the same log.
    return min(candidates, key=lambda b: abs(_parse_run_at(b, "baseline") - row_at))


def _phase_metrics(row: Mapping[str, Any], phase: str, where: str) -> Mapping[str, Any]:
    by_phase = row["metrics"]["by_phase"]
    if phase not in by_phase:
        raise ValueError(
            f"{where}.metrics.by_phase: no block for reward phase {phase!r}; "
            f"have {sorted(by_phase)}"
        )
    block = by_phase[phase]
    missing = [k for k in ("bias", "coverage_95") if k not in block]
    if missing:
        raise ValueError(f"{where}.metrics.by_phase.{phase}: missing {missing}")
    return block


def _eval_start_year(row: Mapping[str, Any], where: str) -> str:
    eval_start = row["state"]["split"]["eval_start_date"]
    if not isinstance(eval_start, str) or not re.match(r"^\d{4}-\d{2}-\d{2}", eval_start):
        raise ValueError(
            f"{where}.state.split.eval_start_date: expected 'YYYY-MM-DD', got {eval_start!r}"
        )
    return eval_start[:4]


# ---- naming -----------------------------------------------------------------


def _slugify(text: str) -> str:
    slug = _NON_SLUG_CHARS.sub(_SLUG_SEPARATOR, text.lower()).strip(_SLUG_SEPARATOR)
    if not slug:
        raise ValueError(f"config_id {text!r} has no slug characters (a-z, 0-9)")
    return slug


def _entry_id(config_id: str, eval_start_year: str) -> str:
    return _SLUG_SEPARATOR.join((ENTRY_ID_PREFIX, _slugify(config_id), eval_start_year))


# ---- action description -----------------------------------------------------


def _format_weight(value: Any) -> str:
    """Weights print as floats (`3.0`, `1.5`, `2.25`): lossless, and `3.0` reads as a multiplier."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sample weight must be a number, got {value!r}")
    return str(float(value))


def _describe_path(path: str, value: Any) -> _ActionDescription:
    if path == CONFIG_PATH_APPROACHING_PEAK:
        spec = _require_keys(f"config_delta.{path}", value, ("weeks_before", "weight"))
        weight = _format_weight(spec["weight"])
        return _ActionDescription(
            text=f"approaching_peak weight {weight} (weeks_before {spec['weeks_before']})",
            name=ACTION_REWEIGHT,
            params={
                "dimension": "approaching_peak",
                "value": spec["weeks_before"],
                "weight": spec["weight"],
            },
        )
    if path.startswith(CONFIG_PATH_BY_PHASE_PREFIX):
        phase = path[len(CONFIG_PATH_BY_PHASE_PREFIX):]
        return _ActionDescription(
            text=f"phase {phase} weight {_format_weight(value)}",
            name=ACTION_REWEIGHT,
            params={"dimension": "phase", "value": phase, "weight": value},
        )
    if path == CONFIG_PATH_TRAIN_WINDOW:
        return _ActionDescription(
            text=f"train_window_weeks {value}",
            name=ACTION_TRAINING_WINDOW,
            params={"weeks": value},
        )
    return _ActionDescription(
        text=f"{path}={value}",
        name=ACTION_SET_CONFIG,
        params={"path": path, "value": value},
    )


def _describe_delta(config_delta: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Words for the statement + the `payload.action` block, from a non-empty delta.

    A delta with one path is that path's action. The harness never writes
    more than one, but a multi-path delta is recorded as `composite` with the
    per-path actions listed, not collapsed into one misleading name.
    """
    described = [_describe_path(path, value) for path, value in config_delta.items()]
    text = ", ".join(d.text for d in described)
    delta_copy = json.loads(json.dumps(dict(config_delta)))
    if len(described) == 1:
        only = described[0]
        return text, {"name": only.name, "params": only.params, "config_delta": delta_copy}
    return text, {
        "name": ACTION_COMPOSITE,
        "params": {},
        "actions": [{"name": d.name, "params": d.params} for d in described],
        "config_delta": delta_copy,
    }


def _verdict(reward: Mapping[str, Any]) -> str:
    if reward["better"]:
        return "better"
    if reward["delta"] == 0:
        return "no change"
    return "worse"


# ---- entry construction -----------------------------------------------------


def _row_to_entry(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    where: str,
    source_label: str,
    created_at: str,
) -> KnowledgeEntry:
    state = row["state"]
    reward = row["reward"]
    phase = reward["phase"]
    metric = reward["metric"]
    before = _phase_metrics(baseline, phase, where=f"baseline for {where}")
    after = _phase_metrics(row, phase, where=where)
    action_text, action = _describe_delta(row["config_delta"])

    statement = (
        f"{action_text}: {phase} {metric.upper()} "
        f"{_METRIC_FORMAT.format(reward['value'])} vs "
        f"{_METRIC_FORMAT.format(reward['baseline_value'])} baseline "
        f"({_DELTA_FORMAT.format(reward['delta'])}, {_verdict(reward)}); "
        f"{phase} bias {_BIAS_FORMAT.format(before['bias'])} -> "
        f"{_BIAS_FORMAT.format(after['bias'])}; "
        f"{phase} coverage_95 {_COVERAGE_FORMAT.format(before['coverage_95'])} -> "
        f"{_COVERAGE_FORMAT.format(after['coverage_95'])}"
    )

    payload: dict[str, Any] = {
        "state": json.loads(json.dumps(dict(state))),
        "action": action,
        "reward": {
            "metric": metric,
            "phase": phase,
            "baseline": reward["baseline_value"],
            "value": reward["value"],
            "delta": reward["delta"],
            "better": bool(reward["better"]),
        },
        "bias_before": before["bias"],
        "bias_after": after["bias"],
        "coverage_95_before": before["coverage_95"],
        "coverage_95_after": after["coverage_95"],
    }
    if reward.get("guard") is not None:
        payload["guard"] = json.loads(json.dumps(reward["guard"]))

    return KnowledgeEntry(
        id=_entry_id(row["config_id"], _eval_start_year(row, where)),
        provenance=EXPERIMENT_PROVENANCE,
        category=EXPERIMENT_CATEGORY,
        statement=statement,
        entities={"model": state["model"], "phase": state["target_phase"], "metric": metric},
        context={"model": [state["model"]], "phase": [state["target_phase"]], "metric": [metric]},
        payload=payload,
        evidence={
            "source": f"{source_label}#{row['config_id']} run_at {row['run_at']}",
            "n_observations": SINGLE_OBSERVATION_COUNT,
            "reward_delta": reward["delta"],
        },
        confidence=SINGLE_OBSERVATION_CONFIDENCE,
        llm_gloss=None,
        created_at=created_at,
    )


def rows_to_entries(
    rows: Iterable[Mapping[str, Any]],
    *,
    created_at: str,
    source_label: str = "experiment log",
) -> list[KnowledgeEntry]:
    """Turn harness rows into experiential entries, one per non-baseline row.

    `source_label` prefixes `evidence.source` (the log path when imported from
    a file). Raises ValueError when a non-baseline row has no matching
    baseline row, or when any row is malformed.
    """
    validated = [_validate_row(row, i) for i, row in enumerate(rows)]
    baselines = [r for r in validated if _is_baseline(r)]
    if not baselines:
        raise ValueError(
            "no baseline row (empty config_delta) in the experiment log; "
            "the baseline supplies the before-values every entry compares against"
        )

    entries: list[KnowledgeEntry] = []
    for index, row in enumerate(validated):
        if _is_baseline(row):
            continue
        where = _row_label(row, index)
        baseline = _matching_baseline(row, baselines, where)
        entries.append(
            _row_to_entry(
                row, baseline, where=where, source_label=source_label, created_at=created_at
            )
        )
    return entries


# ---- file I/O ---------------------------------------------------------------


def read_experiment_log(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL log; blank lines are skipped, a bad line raises with its number."""
    log_path = Path(path)
    if not log_path.is_file():
        raise ValueError(f"experiment log does not exist: {log_path}")
    rows: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{log_path}:{line_no}: expected a JSON object, got {row!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"experiment log is empty: {log_path}")
    return rows


def import_experiment_log(
    path: str | Path, bank: KnowledgeBank, *, created_at: str
) -> ImportResult:
    """Read `path`, build entries, upsert each into `bank`. Idempotent for the same log."""
    rows = read_experiment_log(path)
    entries = rows_to_entries(rows, created_at=created_at, source_label=str(path))
    for entry in entries:
        bank.upsert(entry)
    return ImportResult(
        n_rows=len(rows), n_entries=len(entries), ids=tuple(e.id for e in entries)
    )
