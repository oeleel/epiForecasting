"""End-of-run report for one `agent improve` run (roadmap 6.1).

The improvement loop already prints a running commentary and persists every
iteration to SQLite, but neither is the artifact a reader actually needs. The
question a run has to answer is narrow and practical:

    "Did the search find a better config than the baseline, which edit did it,
     and how much do I believe the number?"

This module answers exactly that, and nothing else. It turns a `RunResult`
(live, in-process) or a `RunTracker.get_run()` row (historical, from
`outputs/agent_runs/runs.db`) into a `RunReport`, renders it as markdown, and
writes `report.md` + `report.json` next to the run's forecasts.

Design notes
------------
- **Direction is not the renderer's opinion.** `_is_better` / `_improvement_pct`
  are imported from `agent.orchestrator` rather than reimplemented. For `wis` a
  *negative* absolute delta is an improvement; for `coverage_95` "better" means
  closer to 0.95; for `bias` it means closer to zero. Every delta in the report
  is a direction-aware **fraction** where positive always means better, so a
  reader never has to know which way a metric points. Duplicating that logic
  here is the real risk — reaching through an underscore is not.
- **The report describes the search, never the epidemic.** Every sentence is
  about models, actions, metric deltas, and the recommendation. Summarizing what
  the forecast says about flu is a separate agent and deliberately out of scope
  (advisor meeting 2026-09-02).
- **Carry-forward rows are flagged, not credited.** `skipped`, `no_op`, `stop`,
  and `user_stop` iterations reuse the previous iteration's forecast and metrics
  verbatim. Crediting them with a delta would invent an improvement that no
  retrain produced, so they get `delta_vs_prev=None`, `became_best=False`, and a
  marker in the table.
- **Best-iteration is recomputed here, from the lines.** `RunTracker.
  get_best_iteration` only supports four metrics while `improve --target-metric`
  allows six, and the tracker persists no `action_status`. Computing it locally
  with `_is_better` gives the live path and the historical path one code path
  and one answer. It agrees with `Orchestrator._find_best_iteration` by
  construction: both scan in order with the same strict comparison, and
  carry-forward rows duplicate a value so they can never strictly win.
- **Honest degradation beats a schema migration.** `action_status`,
  `change_desc`, `elapsed_s`, and `stop_reason` are not columns in `runs.db`. A
  report rebuilt from the DB names them in `degraded_fields` and says so in the
  markdown instead of silently guessing.
- **The evaluation window is never optional.** The window, the location count,
  and the forecast count are printed on every report so a 1-cutoff smoke test
  can never be mistaken for a citable result. Where two iterations with
  different configs produce byte-identical metrics (exactly what
  `--fake-pipeline` does), the report says so.
- `render_markdown` is pure, matching `agent.model_selection.render_markdown`.
  `write_report` is the one deliberate I/O helper: the orchestrator and the CLI
  both need the same two files in the same place, so the idiom exists once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Private, on purpose: these two functions are the single source of truth for
# "which direction is better" in this repo. Re-deriving them here is the bug.
from agent.orchestrator import RunResult, _extract_target, _improvement_pct, _is_better

__all__ = [
    "CARRY_FORWARD_STATUSES",
    "MATERIAL_IMPROVEMENT_FRAC",
    "REPORT_JSON_NAME",
    "REPORT_MD_NAME",
    "IterationLine",
    "RunReport",
    "build_report",
    "build_report_from_tracker",
    "render_markdown",
    "write_report",
]

_log = logging.getLogger(__name__)

# Statuses whose metrics are copied from the previous iteration rather than
# produced by a retrain. They are reported, never credited.
CARRY_FORWARD_STATUSES = frozenset({"stop", "user_stop", "skipped", "no_op"})

# Below this fractional gain the recommendation says "marginal" rather than
# "adopt". Mirrors the loop's own no_improvement_threshold default.
MATERIAL_IMPROVEMENT_FRAC = 0.01

REPORT_MD_NAME = "report.md"
REPORT_JSON_NAME = "report.json"

# Fields the SQLite schema does not persist; a DB-rebuilt report degrades these.
TRACKER_DEGRADED_FIELDS = ("action_status", "change_desc", "elapsed_s", "stop_reason")

_UNKNOWN_STATUS = "unknown"
_METRIC_DIGITS = {"wis": 2, "coverage_95": 3}
_DEFAULT_DIGITS = 1
_MAX_LOCATIONS_REPORTED = 5


@dataclass(frozen=True, slots=True)
class IterationLine:
    """One row of the "what was tried" table, with direction-aware deltas."""

    iteration: int
    action_name: str | None
    action_params: dict[str, Any]
    action_status: str
    change_desc: str | None
    rationale: str | None
    expected_effect: str | None
    diagnosis_summary: str | None
    suggested_focus: str | None
    target_value: float | None
    delta_vs_prev: float | None  # direction-aware FRACTION, + = better
    delta_vs_best_before: float | None  # vs the best target seen strictly before
    became_best: bool
    carry_forward: bool
    error: str | None  # set when the iteration recorded {"error": ...}
    metrics: dict[str, Any] = field(default_factory=dict, repr=False)

    def action_label(self) -> str:
        """Human-readable `name(k=v, ...)`, or `(baseline)` when no action ran."""
        if not self.action_name:
            return "(baseline)"
        if not self.action_params:
            return self.action_name
        params = ", ".join(f"{k}={v}" for k, v in self.action_params.items())
        return f"{self.action_name}({params})"


@dataclass(frozen=True, slots=True)
class RunReport:
    """Everything the end-of-run markdown needs, already direction-corrected."""

    run_id: str
    target_metric: str
    stop_reason: str | None
    cutoff_date: str | None
    model_family: str | None
    n_iterations: int
    best_iteration: int | None
    baseline_value: float | None
    best_value: float | None
    improvement_frac: float | None  # direction-aware, + = better
    lines: list[IterationLine]
    baseline_metrics: dict[str, Any]
    best_metrics: dict[str, Any]
    date_range: dict[str, str]
    n_locations: int | None
    n_forecasts: int | None
    source: str  # "run_result" | "tracker"
    degraded_fields: list[str]
    generated_at: str  # ISO8601 UTC

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable form, written verbatim to report.json."""
        return asdict(self)


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------

def build_report(result: RunResult) -> RunReport:
    """Build a report from a live `RunResult` (the orchestrator's own output)."""
    if result is None:
        raise ValueError("result must be a RunResult, got None")

    target_metric = result.target_metric or "wis"
    records = list(result.iterations)
    lines = _build_lines(
        [_normalize_record(rec, target_metric) for rec in records], target_metric
    )

    baseline_metrics = dict(records[0].metrics) if records else {}
    best_index = _best_index(lines, target_metric)
    best_metrics = dict(records[best_index].metrics) if best_index is not None else {}
    configs = [rec.config for rec in records]

    return _assemble_report(
        run_id=result.run_id,
        target_metric=target_metric,
        stop_reason=result.stop_reason,
        lines=lines,
        best_index=best_index,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        configs=configs,
        cutoff_date=_cutoff_date(configs),
        source="run_result",
        degraded_fields=[],
    )


def build_report_from_tracker(run_row: dict[str, Any]) -> RunReport:
    """Build a report from a `RunTracker.get_run()` row (a historical run).

    The SQLite schema stores no `action_status`, `change_desc`, `elapsed_s`, or
    `stop_reason`, so those come back as `degraded_fields` and the markdown says
    so rather than guessing. Takes the dict, not the tracker, so this module
    never imports `RunTracker` and stays trivially testable.
    """
    if not isinstance(run_row, dict):
        raise ValueError(f"run_row must be a dict from RunTracker.get_run, got {type(run_row)}")
    if "run_id" not in run_row:
        raise ValueError(f"run_row is missing 'run_id'; keys were {sorted(run_row)}")

    target_metric = run_row.get("target_metric") or "wis"
    rows = list(run_row.get("iterations") or [])
    lines = _build_lines([_normalize_row(row, target_metric) for row in rows], target_metric)

    baseline_metrics = dict(rows[0].get("metrics") or {}) if rows else {}
    best_index = _best_index(lines, target_metric)
    best_metrics = dict(rows[best_index].get("metrics") or {}) if best_index is not None else {}
    configs = [row.get("config") for row in rows]

    return _assemble_report(
        run_id=str(run_row["run_id"]),
        target_metric=target_metric,
        stop_reason=None,
        lines=lines,
        best_index=best_index,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        configs=configs,
        cutoff_date=run_row.get("cutoff_date") or _cutoff_date(configs),
        source="tracker",
        degraded_fields=list(TRACKER_DEGRADED_FIELDS),
    )


# ----------------------------------------------------------------------------
# Rendering + writing
# ----------------------------------------------------------------------------

def render_markdown(report: RunReport, title: str | None = None) -> str:
    """Render the professor-facing view of a RunReport. Pure: string in, out."""
    lines: list[str] = [f"# {title or f'Run report — {report.run_id}'}", ""]
    lines += _header_block(report)
    lines += ["", "What was tried:", ""]
    lines += _tried_table(report)
    lines += ["", "Where the benefit came from (baseline -> best):", ""]
    lines += _phase_table(report)
    lines += [""]
    lines += _horizon_table(report)
    lines += _location_block(report)

    diagnoses = [ln for ln in report.lines if ln.diagnosis_summary]
    if diagnoses:
        lines += ["", "Diagnoses:", ""]
        for ln in diagnoses:
            focus = f" (focus: {ln.suggested_focus})" if ln.suggested_focus else ""
            lines.append(f"- iter {ln.iteration}: {ln.diagnosis_summary}{focus}")

    errors = [ln for ln in report.lines if ln.error]
    if errors:
        lines += ["", "Errors:", ""]
        for ln in errors:
            lines.append(f"- iter {ln.iteration}: {ln.error}")

    caveats = _caveats(report)
    if caveats:
        lines += ["", "Caveats:", ""]
        lines += [f"- {c}" for c in caveats]

    lines += ["", f"Recommendation: {_recommendation(report)}", ""]
    return "\n".join(lines)


def write_report(report: RunReport, run_dir: Path) -> tuple[Path, Path]:
    """Write `report.md` + `report.json` into `run_dir`; return both paths.

    The one deliberate I/O helper in this module: the orchestrator's finalize
    block and `agent report --out` both need this exact pair of files in this
    exact place, and one shared idiom beats two drifting copies.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    md_path = run_dir / REPORT_MD_NAME
    json_path = run_dir / REPORT_JSON_NAME
    md_path.write_text(render_markdown(report))
    json_path.write_text(json.dumps(report.to_json(), indent=2, default=str))
    _log.debug("wrote run report for %s to %s", report.run_id, run_dir)
    return md_path, json_path


# ----------------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------------

def _normalize_record(record: Any, target_metric: str) -> dict[str, Any]:
    """Flatten an orchestrator IterationRecord into the fields a line needs."""
    metrics = record.metrics if isinstance(record.metrics, dict) else {}
    action = record.action if isinstance(record.action, dict) else None
    diagnosis = record.diagnosis if isinstance(record.diagnosis, dict) else None
    status = record.action_status or _UNKNOWN_STATUS
    return {
        "iteration": int(record.iteration),
        "metrics": metrics,
        "action": action,
        "diagnosis": diagnosis,
        "action_status": status,
        "change_desc": getattr(record, "change_desc", None),
        "carry_forward": status in CARRY_FORWARD_STATUSES,
        "target_value": record.target_value,
        "target_metric": target_metric,
    }


def _normalize_row(row: dict[str, Any], target_metric: str) -> dict[str, Any]:
    """Flatten a RunTracker iteration row into the fields a line needs.

    `action_status` and `change_desc` are not persisted, so the status is
    `unknown` and no row can be identified as carry-forward. The report names
    that in `Caveats:` instead of pretending otherwise.
    """
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    metrics = metrics or {}
    action = row.get("action") if isinstance(row.get("action"), dict) else None
    diagnosis = row.get("diagnosis") if isinstance(row.get("diagnosis"), dict) else None
    denormalized = row.get(target_metric)
    target = denormalized if isinstance(denormalized, (int, float)) else None
    if target is None:
        target = _extract_target(metrics, target_metric)
    return {
        "iteration": int(row.get("iteration", 0)),
        "metrics": metrics,
        "action": action,
        "diagnosis": diagnosis,
        "action_status": _UNKNOWN_STATUS,
        "change_desc": None,
        "carry_forward": False,
        "target_value": target,
        "target_metric": target_metric,
    }


def _build_lines(raws: list[dict[str, Any]], target_metric: str) -> list[IterationLine]:
    """Turn normalized iteration dicts into lines carrying direction-aware deltas."""
    lines: list[IterationLine] = []
    prev_value: float | None = None
    best_before: float | None = None

    for raw in raws:
        metrics = raw["metrics"]
        error = None
        if isinstance(metrics, dict) and "error" in metrics and "overall" not in metrics:
            error = str(metrics["error"])

        carry_forward = bool(raw["carry_forward"])
        target = None if error else raw["target_value"]
        scoreable = target is not None and not carry_forward

        delta_prev = None
        delta_best = None
        became_best = False
        if scoreable:
            if prev_value is not None:
                delta_prev = _improvement_pct(target_metric, target, prev_value)
            if best_before is not None:
                delta_best = _improvement_pct(target_metric, target, best_before)
            became_best = _is_better(target_metric, target, best_before)

        action = raw["action"] or {}
        diagnosis = raw["diagnosis"] or {}
        lines.append(IterationLine(
            iteration=raw["iteration"],
            action_name=action.get("name"),
            action_params=dict(action.get("params") or {}),
            action_status=raw["action_status"],
            change_desc=raw["change_desc"],
            rationale=(action.get("rationale") or None),
            expected_effect=(action.get("expected_effect") or None),
            diagnosis_summary=((diagnosis.get("summary") or "").strip() or None),
            suggested_focus=(diagnosis.get("suggested_focus") or None),
            target_value=target,
            delta_vs_prev=delta_prev,
            delta_vs_best_before=delta_best,
            became_best=became_best,
            carry_forward=carry_forward,
            error=error,
            metrics=metrics if isinstance(metrics, dict) else {},
        ))

        if scoreable:
            prev_value = target
            if became_best:
                best_before = target

    return lines


def _best_index(lines: list[IterationLine], target_metric: str) -> int | None:
    """Index into `lines` of the best scoreable iteration, or None if there is none."""
    best_idx: int | None = None
    best_val: float | None = None
    for i, ln in enumerate(lines):
        if ln.target_value is None or ln.carry_forward:
            continue
        if _is_better(target_metric, ln.target_value, best_val):
            best_idx = i
            best_val = ln.target_value
    return best_idx


def _assemble_report(
    *,
    run_id: str,
    target_metric: str,
    stop_reason: str | None,
    lines: list[IterationLine],
    best_index: int | None,
    baseline_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    configs: list[dict[str, Any] | None],
    cutoff_date: str | None,
    source: str,
    degraded_fields: list[str],
) -> RunReport:
    """Shared tail of both builders: derive the headline numbers and pack them."""
    baseline_value = lines[0].target_value if lines else None
    best_value = lines[best_index].target_value if best_index is not None else None
    improvement = None
    if baseline_value is not None and best_value is not None:
        improvement = _improvement_pct(target_metric, best_value, baseline_value)

    window_metrics = best_metrics or baseline_metrics
    overall = window_metrics.get("overall") if isinstance(window_metrics, dict) else None
    overall = overall if isinstance(overall, dict) else {}
    date_range = window_metrics.get("date_range") if isinstance(window_metrics, dict) else None

    return RunReport(
        run_id=run_id,
        target_metric=target_metric,
        stop_reason=stop_reason,
        cutoff_date=cutoff_date,
        model_family=_model_family(configs),
        n_iterations=len(lines),
        best_iteration=(lines[best_index].iteration if best_index is not None else None),
        baseline_value=baseline_value,
        best_value=best_value,
        improvement_frac=improvement,
        lines=lines,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        date_range=dict(date_range) if isinstance(date_range, dict) else {},
        n_locations=_as_int(window_metrics.get("n_locations")),
        n_forecasts=_as_int(overall.get("n_forecasts")),
        source=source,
        degraded_fields=list(degraded_fields),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _model_family(configs: list[dict[str, Any] | None]) -> str | None:
    """Read `config["model"]["family"]` from the first config that has one."""
    for config in configs:
        if not isinstance(config, dict):
            continue
        model = config.get("model")
        if isinstance(model, dict) and model.get("family"):
            return str(model["family"])
    return None


def _cutoff_date(configs: list[dict[str, Any] | None]) -> str | None:
    """Read `config["data"]["cutoff_date"]` from the first config that has one."""
    for config in configs:
        if not isinstance(config, dict):
            continue
        data = config.get("data")
        if isinstance(data, dict) and data.get("cutoff_date"):
            return str(data["cutoff_date"])
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any, digits: int = 1) -> str:
    """Format a number, or `-` when it is missing / not a number."""
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def _metric_num(value: Any, metric: str) -> str:
    return _num(value, _METRIC_DIGITS.get(metric, _DEFAULT_DIGITS))


def _pct(frac: Any) -> str:
    """Render a direction-aware fraction as a signed percentage; `-` if absent."""
    return f"{frac * 100:+.1f}%" if isinstance(frac, (int, float)) else "-"


def _sorted_keys(keys: Any) -> list[str]:
    """Sort segment keys, numerically when they are numeric strings.

    Real metrics dicts come back from JSON with horizon keys as the strings
    "1".."4" and only the phases that actually occurred in the window, so this
    never assumes a key set.
    """
    def order(key: Any) -> tuple[int, int, str]:
        text = str(key)
        if text.lstrip("-").isdigit():
            return (0, int(text), text)
        return (1, 0, text)

    return [str(k) for k in sorted(keys, key=order)]


def _segment(metrics: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    block = metrics.get(section) if isinstance(metrics, dict) else None
    if not isinstance(block, dict):
        return {}
    value = block.get(key)
    return value if isinstance(value, dict) else {}


def _segment_keys(report: RunReport, section: str) -> list[str]:
    keys: set[str] = set()
    for metrics in (report.baseline_metrics, report.best_metrics):
        block = metrics.get(section) if isinstance(metrics, dict) else None
        if isinstance(block, dict):
            keys.update(str(k) for k in block)
    return _sorted_keys(keys)


def _header_block(report: RunReport) -> list[str]:
    metric = report.target_metric
    window = report.date_range or {}
    span = f"{window.get('min', '?')} .. {window.get('max', '?')}"
    n_loc = report.n_locations if report.n_locations is not None else "?"
    n_fc = report.n_forecasts if report.n_forecasts is not None else "?"
    stop = report.stop_reason or "(not persisted for this run)"

    out = [
        f"- run: {report.run_id} (source: {report.source})",
        f"- target metric: {metric}",
        f"- model family: {report.model_family or 'unknown'}",
        f"- cutoff: {report.cutoff_date or 'unknown'}",
        f"- evaluation window: {span} ({n_loc} locations, {n_fc} forecasts)",
        f"- iterations: {report.n_iterations}",
        f"- stop reason: {stop}",
        f"- best iteration: **{report.best_iteration if report.best_iteration is not None else 'none'}**",
    ]
    if report.baseline_value is not None and report.best_value is not None:
        out.append(
            f"- baseline -> best: {_metric_num(report.baseline_value, metric)} -> "
            f"{_metric_num(report.best_value, metric)} ({_pct(report.improvement_frac)})"
        )
    else:
        out.append("- baseline -> best: no scoreable iteration")
    return out


def _tried_table(report: RunReport) -> list[str]:
    metric = report.target_metric
    rows = [
        f"| iter | action | status | {metric} | vs prev | vs best-so-far |",
        "|---|---|---|---|---|---|",
    ]
    if not report.lines:
        rows.append("| - | (no iterations recorded) | - | - | - | - |")
        return rows
    for ln in report.lines:
        status = ln.action_status
        if ln.carry_forward:
            status = f"{status} (carry-forward)"
        if ln.error:
            status = f"{status} (error)"
        if ln.became_best and ln.delta_vs_best_before is not None:
            status = f"{status} (new best)"
        rows.append(
            f"| {ln.iteration} | {ln.action_label()} | {status} "
            f"| {_metric_num(ln.target_value, metric)} "
            f"| {_pct(ln.delta_vs_prev)} | {_pct(ln.delta_vs_best_before)} |"
        )
    return rows


def _phase_table(report: RunReport) -> list[str]:
    return _segment_table(report, "by_phase", "phase")


def _horizon_table(report: RunReport) -> list[str]:
    return _segment_table(report, "by_horizon", "horizon")


def _segment_table(report: RunReport, section: str, label: str) -> list[str]:
    keys = _segment_keys(report, section)
    rows = [
        f"| {label} | wis base | wis best | mape base | mape best | n |",
        "|---|---|---|---|---|---|",
    ]
    if not keys:
        rows.append(f"| - | (no {label} breakdown recorded) | - | - | - | - |")
        return rows
    for key in keys:
        base = _segment(report.baseline_metrics, section, key)
        best = _segment(report.best_metrics, section, key)
        n = best.get("n", base.get("n"))
        rows.append(
            f"| {key} | {_num(base.get('wis'), 2)} | {_num(best.get('wis'), 2)} "
            f"| {_num(base.get('mape'))} | {_num(best.get('mape'))} "
            f"| {n if n is not None else '-'} |"
        )
    return rows


def _location_block(report: RunReport) -> list[str]:
    """The 5 worst locations of the best config, exactly as the metrics dict caps them."""
    worst = report.best_metrics.get("worst_locations") or report.baseline_metrics.get(
        "worst_locations"
    )
    if not isinstance(worst, list) or not worst:
        return []
    out = ["", f"Worst {min(len(worst), _MAX_LOCATIONS_REPORTED)} locations of the best config:", ""]
    for loc in worst[:_MAX_LOCATIONS_REPORTED]:
        if not isinstance(loc, dict):
            continue
        out.append(
            f"- {loc.get('location', '?')}: mape={_num(loc.get('mape'))} "
            f"mae={_num(loc.get('mae'))} bias={_num(loc.get('bias'))}"
        )
    return out


def _identical_metrics_iterations(report: RunReport) -> list[int]:
    """Iterations whose overall metrics byte-match an earlier retrained iteration.

    A retrain that changed the config but produced identical numbers means the
    pipeline did not actually retrain — which is precisely what
    `--fake-pipeline` does. Carry-forward rows are excluded: duplicating the
    previous metrics is their defined behavior, not a symptom. The baseline
    seeds the fingerprints but is never itself reported, since it is the row
    every other iteration is supposed to differ from.
    """
    seen: dict[str, int] = {}
    duplicates: list[int] = []
    for ln in report.lines:
        if ln.carry_forward or ln.error:
            continue
        overall = ln.metrics.get("overall") if isinstance(ln.metrics, dict) else None
        if not isinstance(overall, dict) or not overall:
            continue
        fingerprint = json.dumps(overall, sort_keys=True, default=str)
        if fingerprint in seen:
            if ln.action_name:
                duplicates.append(ln.iteration)
        else:
            seen[fingerprint] = ln.iteration
    return duplicates


def _caveats(report: RunReport) -> list[str]:
    out: list[str] = []

    carried = [ln.iteration for ln in report.lines if ln.carry_forward]
    if carried:
        listed = ", ".join(str(i) for i in carried)
        out.append(
            f"iterations {listed} are carry-forward rows (skipped / no-op / stop): their "
            "metrics duplicate the previous iteration and are excluded from attribution."
        )

    if report.source == "tracker":
        out.append(
            "rebuilt from runs.db, which does not persist "
            f"{', '.join(report.degraded_fields)} — carry-forward iterations cannot be "
            "distinguished from applied ones here. Read the run's own report.md when it exists."
        )

    duplicates = _identical_metrics_iterations(report)
    if duplicates:
        listed = ", ".join(str(i) for i in duplicates)
        out.append(
            f"identical metrics across iterations with different configs (iterations {listed}) "
            "— verify the pipeline actually retrained (`--fake-pipeline` produces exactly this)."
        )

    out.append(
        "rmse is computed overall only; the phase and horizon tables carry "
        "wis/mape/mae/bias/coverage_95."
    )
    if report.best_metrics.get("worst_locations") or report.baseline_metrics.get(
        "worst_locations"
    ):
        out.append(
            f"per-location attribution is capped at the {_MAX_LOCATIONS_REPORTED} worst / "
            f"{_MAX_LOCATIONS_REPORTED} best locations by MAPE, ranked on MAPE even when the "
            f"loop optimized {report.target_metric}."
        )
    out.append(
        "this report describes the search over configs, not the epidemic — it makes no "
        "claim about influenza itself."
    )
    return out


def _recommendation(report: RunReport) -> str:
    metric = report.target_metric
    if report.n_iterations == 0 or report.best_value is None:
        reason = report.stop_reason or "no scoreable iteration was recorded"
        return f"inconclusive — the run produced no scoreable iteration ({reason})"

    frac = report.improvement_frac
    if report.best_iteration == 0 or frac is None or frac <= 0:
        return f"keep the baseline config — no iteration improved {metric} over iteration 0"

    best_line = next(
        (ln for ln in report.lines if ln.iteration == report.best_iteration), None
    )
    if frac < MATERIAL_IMPROVEMENT_FRAC:
        return (
            f"marginal: iteration {report.best_iteration} improved {metric} by "
            f"{frac * 100:+.1f}%, below the "
            f"{MATERIAL_IMPROVEMENT_FRAC * 100:.0f}% materiality bar"
        )
    what = "unknown change"
    if best_line is not None:
        what = best_line.change_desc or best_line.action_label()
    return (
        f"adopt iteration {report.best_iteration}'s config ({what}) — "
        f"{metric} improved {frac * 100:+.1f}%"
    )
