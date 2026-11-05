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
- Phase schemes are never mixed silently. A row's phase numbers mean the
  dates of the scheme it was scored under: `metrics.phase_scheme`, or
  `LEGACY_PHASE_SCHEME` (calendar_v1) when the key is absent, which is every
  round-1 row (scored before schemes existed, with the Dec-Jan "peak"). Each
  entry records it in `payload.phase_scheme` and as a `[phase scheme X]`
  prefix of its statement,
  so a retrieved fact says which "peak" it means. A row and its baseline must
  share a scheme, or the import raises. Entries scored under the legacy scheme
  keep their original ids; any other scheme appends its slug to the id, so a
  later sweep under the new windows adds entries instead of overwriting the
  round-1 ones.
- Multi-seed logs (round 2 on) are merged before import. Rows that carry a
  `seed` key are grouped by `base_config_id`; the seed path is removed from
  `config_delta`, and the reward and the reward phase's bias and coverage are
  the means over the seeds. Every group must hold exactly the baseline's seeds,
  or the import raises. The entry records `n_observations` = number of seeds,
  the seeds, and the per-seed deltas, so a reader sees the spread, not only the
  mean. A merged row is "better" only when every seed improves; seeds of both
  signs give the verdict "mixed across seeds". Confidence stays `low`: seeds on one split are not a held-out test, and
  no significance test is run here. A log that mixes seeded and unseeded rows
  raises.
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
    "LEGACY_PHASE_SCHEME",
    "SEED_CONFIG_PATH",
    "SINGLE_OBSERVATION_CONFIDENCE",
    "ImportResult",
    "import_experiment_log",
    "merge_seed_rows",
    "read_experiment_log",
    "rows_to_entries",
]

DEFAULT_EXPERIMENT_LOG = Path("outputs/experiments/peak_rectification/log.jsonl")

# Rows without metrics.phase_scheme predate the key and were scored under the
# month map now named calendar_v1 (agent.phase_evaluator.LEGACY_PHASE_SCHEME).
# Literal, not imported: the knowledge package stays free of pandas.
LEGACY_PHASE_SCHEME = "calendar_v1"

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

# The harness writes each seed as this config path (peak_rectification.SEED_CONFIG_PATH).
# Literal, not imported: the knowledge package does not import the scripts.
SEED_CONFIG_PATH = "xgboost.random_state"
# Seed means print with the harness's rounding.
_SEED_MEAN_DIGITS = 2
_SEED_MEAN_BIAS_DIGITS = 1
_SEED_MEAN_COVERAGE_DIGITS = 3
# A merged row is "better" only if every seed's delta is negative, valid only where lower is
# better. A mean gain with seeds of both signs is inside seed noise, so it is not "better".
_LOWER_IS_BETTER_METRICS = ("wis", "mae", "rmse", "mape")
VERDICT_MIXED = "mixed across seeds"

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


def _phase_scheme(row: Mapping[str, Any], where: str) -> str:
    scheme = row["metrics"].get("phase_scheme", LEGACY_PHASE_SCHEME)
    if not isinstance(scheme, str) or not scheme:
        raise ValueError(f"{where}.metrics.phase_scheme: expected a scheme name, got {scheme!r}")
    return scheme


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


def _entry_id(config_id: str, eval_start_year: str, phase_scheme: str) -> str:
    parts = [ENTRY_ID_PREFIX, _slugify(config_id), eval_start_year]
    if phase_scheme != LEGACY_PHASE_SCHEME:
        parts.append(_slugify(phase_scheme))
    return _SLUG_SEPARATOR.join(parts)


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


def _verdict(reward: Mapping[str, Any], per_seed_delta: Sequence[float] | None = None) -> str:
    if per_seed_delta and not (
        all(d < 0 for d in per_seed_delta) or all(d > 0 for d in per_seed_delta)
    ):
        return VERDICT_MIXED
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
    scheme = _phase_scheme(row, where)
    baseline_scheme = _phase_scheme(baseline, f"baseline for {where}")
    if scheme != baseline_scheme:
        raise ValueError(
            f"{where}: scored under phase scheme {scheme!r} but its baseline under "
            f"{baseline_scheme!r}; phase numbers across schemes cover different dates"
        )
    before = _phase_metrics(baseline, phase, where=f"baseline for {where}")
    after = _phase_metrics(row, phase, where=where)
    action_text, action = _describe_delta(row["config_delta"])
    seeds = row.get("seeds")
    seed_note = (
        f"; mean of seeds {', '.join(str(x) for x in seeds)}, per-seed delta "
        f"{'/'.join(_DELTA_FORMAT.format(d) for d in row['per_seed_delta'])}"
        if seeds
        else ""
    )

    # Scheme first, not as a trailing note: the reader must know which dates
    # "peak" means before reading any number, and a truncated line keeps it.
    statement = (
        f"[phase scheme {scheme}] "
        f"{action_text}: {phase} {metric.upper()} "
        f"{_METRIC_FORMAT.format(reward['value'])} vs "
        f"{_METRIC_FORMAT.format(reward['baseline_value'])} baseline "
        f"({_DELTA_FORMAT.format(reward['delta'])}, {_verdict(reward, row.get('per_seed_delta'))}); "
        f"{phase} bias {_BIAS_FORMAT.format(before['bias'])} -> "
        f"{_BIAS_FORMAT.format(after['bias'])}; "
        f"{phase} coverage_95 {_COVERAGE_FORMAT.format(before['coverage_95'])} -> "
        f"{_COVERAGE_FORMAT.format(after['coverage_95'])}"
        f"{seed_note}"
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
        "phase_scheme": scheme,
    }
    if reward.get("guard") is not None:
        payload["guard"] = json.loads(json.dumps(reward["guard"]))
    if seeds:
        payload["seeds"] = list(seeds)
        payload["per_seed_delta"] = list(row["per_seed_delta"])

    return KnowledgeEntry(
        id=_entry_id(row["config_id"], _eval_start_year(row, where), scheme),
        provenance=EXPERIMENT_PROVENANCE,
        category=EXPERIMENT_CATEGORY,
        statement=statement,
        entities={"model": state["model"], "phase": state["target_phase"], "metric": metric},
        context={"model": [state["model"]], "phase": [state["target_phase"]], "metric": [metric]},
        payload=payload,
        evidence={
            "source": f"{source_label}#{row['config_id']} run_at {row['run_at']}",
            "n_observations": len(seeds) if seeds else SINGLE_OBSERVATION_COUNT,
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
    rows = list(rows)
    if any("seed" in row for row in rows):
        rows = merge_seed_rows(rows)
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


# ---- seed merge -------------------------------------------------------------


def merge_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per `base_config_id`, averaged over seeds; see the module docstring."""
    unseeded = [_row_label(r, i) for i, r in enumerate(rows) if "seed" not in r]
    if unseeded:
        raise ValueError(f"log mixes seeded and unseeded rows; unseeded: {unseeded}")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        _validate_row(row, index)
        if "base_config_id" not in row:
            raise ValueError(f"{_row_label(row, index)}: seeded row has no base_config_id")
        groups.setdefault(row["base_config_id"], []).append(row)

    baseline_groups = [g for g in groups.values() if all(_is_baseline_seeded(r) for r in g)]
    if len(baseline_groups) != 1:
        raise ValueError(
            f"expected one baseline group (config_delta only {SEED_CONFIG_PATH}), "
            f"found {len(baseline_groups)}"
        )
    baseline_seeds = sorted(r["seed"] for r in baseline_groups[0])
    return [_merge_group(base_id, group, baseline_seeds) for base_id, group in groups.items()]


def _is_baseline_seeded(row: Mapping[str, Any]) -> bool:
    return set(row["config_delta"]) == {SEED_CONFIG_PATH}


def _mean(values: Sequence[float], digits: int) -> float:
    return round(sum(values) / len(values), digits)


def _merge_group(
    base_id: str, group: Sequence[Mapping[str, Any]], baseline_seeds: Sequence[int]
) -> dict[str, Any]:
    group = sorted(group, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in group]
    if seeds != list(baseline_seeds):
        raise ValueError(f"{base_id}: seeds {seeds} differ from the baseline's {list(baseline_seeds)}")
    first = group[0]
    deltas = [{k: v for k, v in r["config_delta"].items() if k != SEED_CONFIG_PATH} for r in group]
    for name, values in (
        ("config_delta (seed removed)", deltas),
        ("state", [r["state"] for r in group]),
        ("phase_scheme", [_phase_scheme(r, base_id) for r in group]),
        ("reward metric/phase", [(r["reward"]["metric"], r["reward"]["phase"]) for r in group]),
    ):
        if any(v != values[0] for v in values):
            raise ValueError(f"{base_id}: seeds disagree on {name}: {values}")
    metric, phase = first["reward"]["metric"], first["reward"]["phase"]
    if metric not in _LOWER_IS_BETTER_METRICS:
        raise ValueError(
            f"{base_id}: cannot merge seeds for metric {metric!r}; "
            f"supported: {_LOWER_IS_BETTER_METRICS}"
        )

    rewards = [r["reward"] for r in group]
    per_seed_delta = [rw["delta"] for rw in rewards]
    mean_delta = _mean(per_seed_delta, _SEED_MEAN_DIGITS)
    reward: dict[str, Any] = {
        "metric": metric,
        "phase": phase,
        "baseline_value": _mean([rw["baseline_value"] for rw in rewards], _SEED_MEAN_DIGITS),
        "value": _mean([rw["value"] for rw in rewards], _SEED_MEAN_DIGITS),
        "delta": mean_delta,
        "better": all(d < 0 for d in per_seed_delta),
    }
    guards = [rw.get("guard") for rw in rewards]
    if all(g is not None for g in guards):
        reward["guard"] = {
            "metric": guards[0]["metric"],
            "baseline_value": _mean([g["baseline_value"] for g in guards], _SEED_MEAN_DIGITS),
            "value": _mean([g["value"] for g in guards], _SEED_MEAN_DIGITS),
        }
    blocks = [_phase_metrics(r, phase, base_id) for r in group]
    return {
        "run_at": max(r["run_at"] for r in group),
        "config_id": base_id,
        "config_delta": deltas[0],
        "state": json.loads(json.dumps(first["state"])),
        "metrics": {
            "phase_scheme": _phase_scheme(first, base_id),
            "by_phase": {
                phase: {
                    "bias": _mean([b["bias"] for b in blocks], _SEED_MEAN_BIAS_DIGITS),
                    "coverage_95": _mean([b["coverage_95"] for b in blocks], _SEED_MEAN_COVERAGE_DIGITS),
                }
            },
        },
        "reward": reward,
        "seeds": seeds,
        "per_seed_delta": per_seed_delta,
    }


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
