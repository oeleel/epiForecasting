"""Peak rectification experiment harness (knowledge-bank design doc, rectification actions 4 + 5).

Question
--------
XGBoost's worst phase on the pinned 2025-26 split is the peak (WIS 122.4,
bias -52.4 = under-prediction; `outputs/model_selection/demo_2026-09-02.json`).
Does either of the lab's two cheap rectification actions move peak WIS
without hurting overall WIS?

    action 5  loss weight lambda on approaching-peak training rows
              (`sample_weights.approaching_peak = {weeks_before: K, weight: lambda}`)
    action 4  fit on a short rolling window of recent weeks
              (`data.train_window_weeks = N`)

Fixed vs varied
---------------
Fixed (recorded in `manifest.json`): the pinned split and its stride-4 cutoffs,
US excluded from scoring, `XGBOOST_PARAMS_V2` (seed 42), feature version,
quantile levels, log target, floor constraint, git SHA, data-cache mtime, host
and library versions. Everything comes from `src.config.get_default_config()`;
an arm only changes the keys its `config_delta` names.

Varied (one log row per config):
    baseline          stock config
    lambda            approaching_peak weight lambda in DEFAULT_LAMBDAS, K = DEFAULT_APPROACHING_PEAK_WEEKS
    lambda_calendar   existing calendar `by_phase.peak = lambda`, lambda in DEFAULT_CALENDAR_LAMBDAS
                      (does the label definition matter?)
    window            `train_window_weeks = N`, N in DEFAULT_WINDOWS

Oversampling and SMOTE (actions 2 and 3) are planned arms in the design doc
and are NOT run here.

Reuse, not reimplementation
---------------------------
Training goes through `FluForecastAdapter.run_pipeline` (the same path the
improve loop uses), scoring through `FluForecastAdapter.compute_metrics` (the
same function `select-model` and the improve loop use), and the reward
direction through `agent.orchestrator._is_better` / `_improvement_pct` (the
single source of truth `run_report` also imports). There is no private scorer
in this file.

How to run
----------
    # full sweep (9 cutoffs x 10 configs; ~70 s per config per cutoff on a MacBook
    # with OMP_NUM_THREADS=1, so ~1.5-2 h; run it in the background)
    .venv/bin/python scripts/experiments/peak_rectification.py

    # smoke run: first cutoff only, baseline + one lambda
    .venv/bin/python scripts/experiments/peak_rectification.py --cutoffs 1 --arms baseline lambda --lambdas 2

    # regenerate summary.md from the log without training anything
    .venv/bin/python scripts/experiments/peak_rectification.py --summary-only

Where output lands
------------------
`outputs/experiments/peak_rectification/` (gitignored except `summary.md` and
`manifest.json`, which are committed by hand):
    manifest.json             the fixed part of the experiment (written once)
    log.jsonl                 one (state, action, reward) row per config
    forecasts/<config_id>.csv pooled forecasts across cutoffs, for re-scoring
    summary.md                generated from log.jsonl, never edited by hand
A run on a cutoff subset (`--cutoffs N` below the full split) defaults to a
sibling directory `peak_rectification-smoke-<N>cutoffs/` so a smoke run can
never be mistaken for, or block, the real sweep.

Invariants
----------
- Every row in one log shares the same cutoffs; the manifest pins them and a
  run with different cutoffs into the same directory fails loudly.
- The baseline config is always run (or read back from the log) before any
  other arm, because every reward is relative to it.
- A `config_id` already present in the log is skipped (resumable).
- Quantiles must be enabled: WIS needs the quantile columns.
- A missing target phase in a run's forecasts (for example a single October
  cutoff has no peak-phase rows) yields a null reward, not a fabricated one.
- `summary.md` is regenerated from the log after every run, so it can never
  disagree with it.
"""

from __future__ import annotations

import os
import sys

# macOS: xgboost and torch each bundle a libomp; importing both in one process
# segfaults without this (see .claude/CLAUDE.md). Must precede those imports.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import json
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from agent.adapters.flu_forecast import FluForecastAdapter  # noqa: E402
from agent.orchestrator import _improvement_pct, _is_better  # noqa: E402
from src import config as cfg  # noqa: E402
from src.direct_forecast import DEFAULT_APPROACHING_PEAK_WEEKS  # noqa: E402
from src.pipeline import MIN_TRAIN_WINDOW_WEEKS  # noqa: E402

__all__ = [
    "Arm",
    "FIXED",
    "apply_delta",
    "arm_lambda",
    "arm_lambda_calendar",
    "arm_window",
    "build_arms",
    "build_manifest",
    "log_row",
    "main",
    "read_log",
    "reward_for",
    "run_config",
    "score",
    "write_summary",
]

# ---- What is fixed ----------------------------------------------------------

MODEL_FAMILY = "xgboost_direct"
TARGET_PHASE = "peak"
TARGET_METRIC = "wis"
GUARD_METRIC = "wis"  # overall WIS: an arm that wins the peak but loses overall is flagged
DEFAULT_STRIDE_WEEKS = 4  # the 09-02 demo split: 9 cutoffs
EXCLUDE_LOCATIONS = ("US",)  # national aggregate is not a state; excluded from scoring only

# The complete fixed part of the experiment, written to manifest.json.
FIXED: dict[str, Any] = {
    "model_family": MODEL_FAMILY,
    "target_phase": TARGET_PHASE,
    "target_metric": TARGET_METRIC,
    "guard_metric": f"overall_{GUARD_METRIC}",
    "split": {
        "train_start_date": cfg.TRAIN_START_DATE,
        "eval_start_date": cfg.EVAL_START_DATE,
        "eval_end_date": cfg.EVAL_END_DATE,
        "stride_weeks": DEFAULT_STRIDE_WEEKS,
    },
    "exclude_locations": list(EXCLUDE_LOCATIONS),
    "xgboost_params": dict(cfg.XGBOOST_PARAMS_V2),
    "seed": cfg.XGBOOST_PARAMS_V2["random_state"],
    "feature_version": cfg.FEATURE_VERSION,
    "quantile_levels": list(cfg.QUANTILES),
    "target_mode": cfg.TARGET_MODE,
    "forecast_horizon": cfg.FORECAST_HORIZON,
    "approaching_peak_weeks": DEFAULT_APPROACHING_PEAK_WEEKS,
    "min_train_window_weeks": MIN_TRAIN_WINDOW_WEEKS,
}

# ---- What is varied ---------------------------------------------------------

ARM_BASELINE = "baseline"
ARM_LAMBDA = "lambda"
ARM_LAMBDA_CALENDAR = "lambda_calendar"
ARM_WINDOW = "window"
ARM_NAMES = (ARM_BASELINE, ARM_LAMBDA, ARM_LAMBDA_CALENDAR, ARM_WINDOW)

DEFAULT_LAMBDAS = (1.5, 2.0, 3.0, 5.0)
DEFAULT_CALENDAR_LAMBDAS = (2.0, 3.0)
DEFAULT_WINDOWS = (12, 26, 52)
BASELINE_CONFIG_ID = "baseline"

# ---- Output layout ----------------------------------------------------------

EXPERIMENT_NAME = "peak_rectification"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT_NAME
MANIFEST_NAME = "manifest.json"
LOG_NAME = "log.jsonl"
SUMMARY_NAME = "summary.md"
FORECASTS_DIRNAME = "forecasts"
DATA_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "flusight_hospital_admissions.csv"

SUMMARY_MISSING = "n/a"  # rendered when a metric is absent (e.g. no peak rows in a smoke run)

CLI_EPILOG = """\
examples:
  full sweep (9 cutoffs x 10 configs; ~70 s per config per cutoff on a laptop with OMP_NUM_THREADS=1)
      %(prog)s
  smoke run: first cutoff only, baseline + lambda=2 (lands in a -smoke-1cutoffs sibling dir)
      %(prog)s --cutoffs 1 --arms baseline lambda --lambdas 2
  regenerate summary.md from the log without training
      %(prog)s --summary-only
"""


# ---- Arms -------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Arm:
    """One configuration to run.

    `config_delta` maps dotted config paths to the value that replaces the
    default, e.g. {"data.train_window_weeks": 12}. It is recorded verbatim in
    the log row so the row is self-describing without the script.
    """

    name: str
    config_id: str
    config_delta: Mapping[str, Any] = field(default_factory=dict)
    describe: str = ""

    def __post_init__(self) -> None:
        if self.name not in ARM_NAMES:
            raise ValueError(f"Arm.name must be one of {ARM_NAMES}, got {self.name!r}")
        if not self.config_id or any(ch in self.config_id for ch in "/\\ "):
            raise ValueError(f"Arm.config_id must be a non-empty file-safe token, got {self.config_id!r}")


def _fmt_number(value: float) -> str:
    """1.5 -> '1.5', 2.0 -> '2': keeps config ids short and stable."""
    return f"{value:g}"


def arm_baseline() -> Arm:
    return Arm(name=ARM_BASELINE, config_id=BASELINE_CONFIG_ID, describe="stock config")


def arm_lambda(weight: float, weeks_before: int = DEFAULT_APPROACHING_PEAK_WEEKS) -> Arm:
    if weight <= 0:
        raise ValueError(f"lambda weight must be > 0, got {weight}")
    if not isinstance(weeks_before, int) or isinstance(weeks_before, bool) or weeks_before < 1:
        raise ValueError(f"weeks_before must be an int >= 1, got {weeks_before!r}")
    return Arm(
        name=ARM_LAMBDA,
        config_id=f"lambda_{_fmt_number(weight)}",
        config_delta={
            "sample_weights.approaching_peak": {"weeks_before": weeks_before, "weight": weight},
        },
        describe=f"approaching-peak rows (K={weeks_before} weeks before the season max) weighted {weight:g}x",
    )


def arm_lambda_calendar(weight: float) -> Arm:
    if weight <= 0:
        raise ValueError(f"calendar lambda weight must be > 0, got {weight}")
    return Arm(
        name=ARM_LAMBDA_CALENDAR,
        config_id=f"lambda_calendar_{_fmt_number(weight)}",
        config_delta={"sample_weights.by_phase.peak": weight},
        describe=f"calendar peak-phase rows (Dec-Jan) weighted {weight:g}x",
    )


def arm_window(weeks: int) -> Arm:
    if not isinstance(weeks, int) or isinstance(weeks, bool):
        raise ValueError(f"window weeks must be an int, got {weeks!r}")
    if weeks < MIN_TRAIN_WINDOW_WEEKS:
        raise ValueError(
            f"window weeks must be >= {MIN_TRAIN_WINDOW_WEEKS} (src.pipeline.MIN_TRAIN_WINDOW_WEEKS), got {weeks}"
        )
    return Arm(
        name=ARM_WINDOW,
        config_id=f"window_{weeks}",
        config_delta={"data.train_window_weeks": weeks},
        describe=f"fit only on feature rows from the last {weeks} weeks before each cutoff",
    )


def build_arms(
    arm_names: Sequence[str],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    calendar_lambdas: Sequence[float] = DEFAULT_CALENDAR_LAMBDAS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> list[Arm]:
    """Expand arm names into concrete configs. Baseline is always first."""
    unknown = sorted(set(arm_names) - set(ARM_NAMES))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {list(ARM_NAMES)}")
    arms: list[Arm] = [arm_baseline()]
    if ARM_LAMBDA in arm_names:
        arms.extend(arm_lambda(w) for w in lambdas)
    if ARM_LAMBDA_CALENDAR in arm_names:
        arms.extend(arm_lambda_calendar(w) for w in calendar_lambdas)
    if ARM_WINDOW in arm_names:
        arms.extend(arm_window(n) for n in windows)
    ids = [a.config_id for a in arms]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate config ids: {ids}")
    return arms


def apply_delta(config: dict[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of `config` with each dotted path in `delta` set.

    Intermediate dicts are created when absent (`sample_weights.approaching_peak`
    is not in the default config). Only the final leaf is replaced.
    """
    out = copy.deepcopy(config)
    for path, value in delta.items():
        node: Any = out
        *parents, leaf = path.split(".")
        for segment in parents:
            if segment not in node:
                node[segment] = {}
            if not isinstance(node[segment], dict):
                raise ValueError(f"config path {path!r}: {segment!r} is not a mapping")
            node = node[segment]
        node[leaf] = copy.deepcopy(value)
    return out


# ---- Run + score ------------------------------------------------------------

def _base_config() -> dict[str, Any]:
    config = cfg.get_default_config()
    if not config["quantiles"]["enabled"]:
        raise ValueError(
            "quantiles.enabled is False in src.config.get_default_config(); "
            "WIS needs the predicted_q* columns, so this harness cannot run"
        )
    if config["model"]["family"] != MODEL_FAMILY:
        raise ValueError(
            f"default model family is {config['model']['family']!r}; this harness fixes {MODEL_FAMILY!r}"
        )
    return config


def run_config(
    adapter: FluForecastAdapter,
    arm: Arm,
    cutoffs: Sequence[str],
    forecasts_dir: Path,
    say: Any = print,
) -> tuple[pd.DataFrame, float]:
    """Train + forecast `arm` at every cutoff; return (pooled forecasts, seconds).

    Excluded locations are dropped from the pooled frame the same way
    `FluForecastAdapter.load_data` drops them, so scoring matches the loop.
    The pooled frame is written to forecasts/<config_id>.csv for re-scoring.
    """
    if not cutoffs:
        raise ValueError("cutoffs must be non-empty")
    t0 = time.perf_counter()
    frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix=f"{arm.config_id}_") as tmp:
        for cutoff in cutoffs:
            config = apply_delta(_base_config(), arm.config_delta)
            config["data"]["cutoff_date"] = cutoff
            t_cut = time.perf_counter()
            csv_path = adapter.run_pipeline(config, output_path=str(Path(tmp) / f"{cutoff}.csv"))
            frame = pd.read_csv(csv_path)
            frames.append(frame)
            say(f"    {arm.config_id:<22s} {cutoff}  {len(frame):5d} rows  {time.perf_counter() - t_cut:5.1f}s")
    pooled = pd.concat(frames, ignore_index=True)
    if adapter.exclude_locations:
        pooled = pooled[~pooled["location"].astype(str).isin(adapter.exclude_locations)]
    if pooled.empty:
        raise ValueError(f"{arm.config_id}: no forecast rows left after excluding {sorted(adapter.exclude_locations)}")
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(forecasts_dir / f"{arm.config_id}.csv", index=False)
    return pooled, time.perf_counter() - t0


def score(adapter: FluForecastAdapter, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, Any]:
    """The loop's own scorer, trimmed to the three sections the log keeps."""
    metrics = adapter.compute_metrics(forecasts, actuals)
    return {
        "overall": metrics["overall"],
        "by_phase": metrics["by_phase"],
        # json turns int keys into strings anyway; do it here so the in-memory
        # row equals the re-read row
        "by_horizon": {str(h): v for h, v in metrics["by_horizon"].items()},
    }


def _target_value(metrics: Mapping[str, Any]) -> float | None:
    phase = (metrics.get("by_phase") or {}).get(TARGET_PHASE) or {}
    value = phase.get(TARGET_METRIC)
    return None if value is None else float(value)


def reward_for(metrics: Mapping[str, Any], baseline_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Reward = target-phase metric vs the baseline's, using the loop's direction rules.

    A null `value` or `baseline_value` (the phase is absent from the forecasts)
    yields null delta / improvement and better=False rather than a guess.
    """
    value = _target_value(metrics)
    baseline_value = _target_value(baseline_metrics)
    row: dict[str, Any] = {
        "metric": TARGET_METRIC,
        "phase": TARGET_PHASE,
        "baseline_value": baseline_value,
        "value": value,
        "delta": None,
        "improvement_frac": None,
        "better": False,
        "guard": {
            "metric": FIXED["guard_metric"],
            "baseline_value": (baseline_metrics.get("overall") or {}).get(GUARD_METRIC),
            "value": (metrics.get("overall") or {}).get(GUARD_METRIC),
        },
    }
    if value is None or baseline_value is None:
        return row
    row["delta"] = round(value - baseline_value, 2)
    row["improvement_frac"] = round(_improvement_pct(TARGET_METRIC, value, baseline_value), 4)
    row["better"] = _is_better(TARGET_METRIC, value, baseline_value)
    return row


# ---- Log + manifest ---------------------------------------------------------

def read_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path}:{lineno}: malformed log line: {exc}") from exc
    return rows


def log_row(
    log_path: Path,
    arm: Arm,
    cutoffs: Sequence[str],
    metrics: Mapping[str, Any],
    reward: Mapping[str, Any],
    seconds: float,
) -> dict[str, Any]:
    """Append one (state, action, reward) row; returns the row as written."""
    row = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": arm.name,
        "config_id": arm.config_id,
        "config_delta": dict(arm.config_delta),
        "describe": arm.describe,
        "state": {
            "model": MODEL_FAMILY,
            "target_phase": TARGET_PHASE,
            "metric": TARGET_METRIC,
            "split": dict(FIXED["split"]),
            "n_cutoffs": len(cutoffs),
        },
        "metrics": dict(metrics),
        "reward": dict(reward),
        "seconds": round(seconds, 1),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_manifest(cutoffs: Sequence[str]) -> dict[str, Any]:
    cache_mtime = (
        datetime.fromtimestamp(DATA_CACHE_PATH.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        if DATA_CACHE_PATH.exists()
        else None
    )
    import xgboost  # local: keep module import cheap for --summary-only

    config = _base_config()
    return {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "cutoffs": list(cutoffs),
        "n_cutoffs": len(cutoffs),
        "floor": dict(config["floor"]),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "data_cache": {"path": str(DATA_CACHE_PATH.relative_to(PROJECT_ROOT)), "mtime": cache_mtime},
        "host": platform.node(),
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        **FIXED,
    }


def _ensure_manifest(out_dir: Path, cutoffs: Sequence[str]) -> dict[str, Any]:
    """Write the manifest once; on a resumed run, refuse a different split."""
    path = out_dir / MANIFEST_NAME
    if path.exists():
        manifest = json.loads(path.read_text())
        if list(manifest.get("cutoffs", [])) != list(cutoffs):
            raise ValueError(
                f"{path} pins cutoffs {manifest.get('cutoffs')} but this run uses {list(cutoffs)}; "
                "pass --out for a separate directory"
            )
        return manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cutoffs)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# ---- Summary ----------------------------------------------------------------

def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return SUMMARY_MISSING
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: Any, digits: int = 1) -> str:
    if value is None:
        return SUMMARY_MISSING
    return f"{float(value):+.{digits}f}"


def write_summary(out_dir: Path) -> Path:
    """Render summary.md from log.jsonl. Rows are ordered by arm, then log order."""
    log_path = out_dir / LOG_NAME
    rows = read_log(log_path)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    order = {name: i for i, name in enumerate(ARM_NAMES)}
    rows_sorted = sorted(enumerate(rows), key=lambda ir: (order.get(ir[1]["arm"], len(order)), ir[0]))

    lines = [
        f"# {EXPERIMENT_NAME}: results",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{LOG_NAME}` "
        f"({len(rows)} config{'s' if len(rows) != 1 else ''}). Do not edit by hand; rerun the script "
        "or pass `--summary-only`.",
        "",
        f"Target: {TARGET_PHASE} {TARGET_METRIC.upper()} (lower is better). Guard: overall "
        f"{GUARD_METRIC.upper()}. Every delta is vs this run's own `{BASELINE_CONFIG_ID}` row.",
    ]
    if manifest:
        lines += [
            "",
            f"Split: train from {manifest['split']['train_start_date']}, eval "
            f"{manifest['split']['eval_start_date']}..{manifest['split']['eval_end_date']}, "
            f"{manifest['n_cutoffs']} cutoff(s) at stride {manifest['split']['stride_weeks']}: "
            f"{', '.join(manifest['cutoffs'])}. Excluded from scoring: {', '.join(manifest['exclude_locations'])}. "
            f"Git {manifest['git_sha'][:10]}{' (dirty)' if manifest.get('git_dirty') else ''}, "
            f"data cache {manifest['data_cache']['mtime']}.",
        ]
    lines += [
        "",
        "| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for _, row in rows_sorted:
        peak = (row["metrics"].get("by_phase") or {}).get(TARGET_PHASE) or {}
        overall = row["metrics"].get("overall") or {}
        reward = row.get("reward") or {}
        improvement = reward.get("improvement_frac")
        lines.append(
            f"| {row['arm']} | `{row['config_id']}` | {_fmt(peak.get('wis'), 2)} | {_fmt(overall.get('wis'), 2)} "
            f"| {_fmt(peak.get('bias'))} | {_fmt_signed(reward.get('delta'), 2)} "
            f"| {SUMMARY_MISSING if improvement is None else f'{100 * improvement:+.1f}%'} "
            f"| {'yes' if reward.get('better') else 'no'} | {_fmt(row.get('seconds'))} |"
        )
    if not rows:
        lines.append("| (no rows logged yet) | | | | | | | | |")

    lines += ["", "## Configs", ""]
    for _, row in rows_sorted:
        delta = row.get("config_delta") or {}
        lines.append(
            f"- `{row['config_id']}` ({row['arm']}): {row.get('describe') or 'stock config'}"
            f"{'; delta ' + json.dumps(delta, sort_keys=True) if delta else ''}; run {row['run_at']}"
        )
    lines += [
        "",
        "## Reading this table",
        "",
        f"- `{SUMMARY_MISSING}` means the metric is absent from that run's forecasts (for example a "
        f"cutoff subset with no {TARGET_PHASE}-phase target dates), so no reward was computed.",
        "- `better` uses `agent.orchestrator._is_better` on the target metric only; check the "
        "overall WIS column before calling an arm a win.",
        "- Phases are calendar phases (`agent.phase_evaluator.PhaseEvaluator.PHASE_MAP`), the same "
        "labelling every other number in this repo uses.",
        "",
    ]
    summary_path = out_dir / SUMMARY_NAME
    summary_path.write_text("\n".join(lines))
    return summary_path


# ---- CLI --------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peak rectification experiment: lambda / calendar-lambda / window arms vs baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )
    parser.add_argument(
        "--cutoffs", type=int, default=None, metavar="N",
        help="use only the first N of the pinned stride-4 cutoffs (smoke runs). Default: all.",
    )
    parser.add_argument(
        "--arms", nargs="+", default=list(ARM_NAMES), choices=list(ARM_NAMES), metavar="ARM",
        help=f"arms to run; baseline always runs first. Choices: {', '.join(ARM_NAMES)}.",
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS), metavar="L",
        help=f"approaching-peak weights for the lambda arm. Default: {list(DEFAULT_LAMBDAS)}.",
    )
    parser.add_argument(
        "--calendar-lambdas", nargs="+", type=float, default=list(DEFAULT_CALENDAR_LAMBDAS), metavar="L",
        help=f"by_phase.peak weights for the lambda_calendar arm. Default: {list(DEFAULT_CALENDAR_LAMBDAS)}.",
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS), metavar="W",
        help=f"train_window_weeks values for the window arm. Default: {list(DEFAULT_WINDOWS)}.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"output directory. Default: {DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT)} "
             "(a `-smoke-<N>cutoffs` sibling when --cutoffs is a subset).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="regenerate summary.md from the existing log and exit; trains nothing.",
    )
    parser.add_argument("--verbose", action="store_true", help="show the pipeline's training chatter")
    return parser.parse_args(argv)


def _resolve_cutoffs(n: int | None) -> tuple[list[str], bool]:
    """Return (cutoffs, is_subset). `n` beyond the split is a mistake, not a clamp."""
    full = cfg.generate_eval_cutoffs(DEFAULT_STRIDE_WEEKS)
    if n is None:
        return full, False
    if n < 1 or n > len(full):
        raise ValueError(f"--cutoffs must be in [1, {len(full)}] for stride {DEFAULT_STRIDE_WEEKS}, got {n}")
    return full[:n], n < len(full)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cutoffs, is_subset = _resolve_cutoffs(args.cutoffs)
    out_dir: Path = args.out or (
        DEFAULT_OUT_DIR.with_name(f"{EXPERIMENT_NAME}-smoke-{len(cutoffs)}cutoffs") if is_subset else DEFAULT_OUT_DIR
    )
    out_dir = out_dir.resolve()
    log_path = out_dir / LOG_NAME

    if args.summary_only:
        if not log_path.exists():
            raise ValueError(f"--summary-only: nothing logged at {log_path}; run the experiment first")
        path = write_summary(out_dir)
        print(f"summary written: {path} ({len(read_log(log_path))} rows)")
        return 0

    arms = build_arms(args.arms, args.lambdas, args.calendar_lambdas, args.windows)
    _ensure_manifest(out_dir, cutoffs)
    logged = {row["config_id"]: row for row in read_log(log_path)}

    print(f"peak rectification: {len(arms)} config(s) x {len(cutoffs)} cutoff(s) -> {out_dir}")
    print(f"  cutoffs: {', '.join(cutoffs)}")
    if logged:
        print(f"  resuming: {len(logged)} config(s) already logged: {', '.join(sorted(logged))}")

    adapter = FluForecastAdapter(exclude_locations=list(EXCLUDE_LOCATIONS))
    actuals = adapter._load_actuals()
    forecasts_dir = out_dir / FORECASTS_DIRNAME

    baseline_metrics: Mapping[str, Any] | None = None
    for arm in arms:  # baseline is arms[0] by construction
        if arm.config_id in logged:
            row = logged[arm.config_id]
            print(f"  skip {arm.config_id} (logged {row['run_at']})")
        else:
            print(f"  run  {arm.config_id}: {arm.describe}")
            pooled, seconds = run_config(adapter, arm, cutoffs, forecasts_dir)
            metrics = score(adapter, pooled, actuals)
            reference = metrics if arm.config_id == BASELINE_CONFIG_ID else baseline_metrics
            if reference is None:
                raise RuntimeError("baseline metrics missing before a non-baseline arm; this is a bug")
            reward = reward_for(metrics, reference)
            row = log_row(log_path, arm, cutoffs, metrics, reward, seconds)
            print(
                f"       {TARGET_PHASE} {TARGET_METRIC}={_fmt(reward['value'], 2)} "
                f"delta={_fmt_signed(reward['delta'], 2)} better={reward['better']} ({seconds:.0f}s)"
            )
            if reward["value"] is None:
                print(f"       warning: no {TARGET_PHASE}-phase rows in these forecasts; reward is null")
        if arm.config_id == BASELINE_CONFIG_ID:
            baseline_metrics = row["metrics"]
        write_summary(out_dir)

    print(f"done. summary: {out_dir / SUMMARY_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
