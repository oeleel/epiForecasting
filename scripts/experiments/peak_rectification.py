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
Fixed (recorded in `manifest.json`): the pinned split and its cutoffs (stride 4 unless `--stride-weeks`),
US excluded from scoring, `XGBOOST_PARAMS_V2`, feature version, quantile
levels, log target, floor constraint, phase scheme, seed list, git SHA,
data-cache mtime, host and library versions. Everything comes from
`src.config.get_default_config()`; an arm only changes the keys its
`config_delta` names, plus `xgboost.random_state` for its seed.

Varied (one log row per config per seed):
    baseline          stock config
    lambda            approaching_peak weight lambda in DEFAULT_LAMBDAS, K = DEFAULT_APPROACHING_PEAK_WEEKS
    lambda_calendar   existing `by_phase.peak = lambda`, lambda in DEFAULT_CALENDAR_LAMBDAS
                      (does the label definition matter?)
    window            `train_window_weeks = N`, N in DEFAULT_WINDOWS
    oversample        `data.oversample = {phase: peak, factor: k}`, k in DEFAULT_OVERSAMPLE_FACTORS
                      (action 2; A. Adiga, 2026-10-15). Both this arm and
                      lambda_calendar label a row by its TARGET date (origin + h
                      weeks) under DEFAULT_PHASE_SCHEME, the label the evaluator
                      scores. In XGBoost, k copies of a row give the same gradient
                      and hessian sums as weight k, so this arm differs from
                      lambda_calendar = k only through row subsampling
                      (`subsample` draws each copy independently).

SMOTE (action 3) is a planned arm in the design doc and is NOT run here.

Rounds
------
Round 1 (2026-09-30, `outputs/experiments/peak_rectification/`): one seed,
calendar_v1 phases, and lambda_calendar weighted rows by ORIGIN month
(Dec-Jan); round 2's lambda_calendar weights by target date under Adiga's
scheme, so the two rows of the same config id are different treatments. Round 2 (A. Adiga, 2026-10-15) adds the oversample arm,
several seeds (mean, sd and per-seed paired deltas in the summary), phase x
horizon WIS in every metrics block, and `--rescore`. Round 2 writes to its own
directory by default: its config ids would otherwise collide with round 1's
and be skipped as already logged.

Seeds
-----
`--seeds` sets `xgboost.random_state`, which `src.pipeline.run_pipeline`
passes as `xgb_params_override` to every per-horizon, per-quantile
`XGBRegressor`; with `subsample` 0.7 and `colsample_bytree` 0.6 it changes the
fit. Every round-2 config id gets a `__seed<N>` suffix, so per-seed runs
in separate directories never collide when `--merge`d, and each arm's
reward is paired with the baseline of the same seed.

Data vintage
------------
`FluDataLoader` refreshes the CDC cache once it is older than 168 h, and
CDC backfill moves past weeks. A fresh run refreshes a stale cache once,
then pins the cache's sha256 in the manifest (`--pin-data-sha256` skips the
refresh and requires a given snapshot instead, so a new run can share an
older run's data). For the rest of the run the
loader's refresh is switched off (`_frozen_data_cache`), the hash is
re-checked after every config, and a resume or merge against a different
cache fails loudly, so every row and its baseline share one snapshot.

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
    # round 2 sweep: 11 configs (baseline, 4 lambda, 2 lambda_calendar, window_52,
    # 3 oversample) x 9 cutoffs x 3 seeds; window_12/26 lost by ~70 peak WIS in
    # round 1. Timing: the uncontended 2026-10-15 smoke ran ~20 s per config
    # per cutoff (baseline 19.5-19.8 s, oversample_2 21.3-21.4 s). 297
    # config-cutoffs x ~20 s = ~1.65 h, plus margin for later cutoffs (more
    # training rows) and oversample_5 (~1.6x the rows): ~1.5-2.5 h sequential
    # with OMP_NUM_THREADS=1, ~35-50 min per seed. (Round 1's 2856 s is the SUM
    # of per-config seconds of 5 parallel, CPU-contended processes, not wall
    # time; its wall time was ~21 min.) Run it in the background.
    .venv/bin/python scripts/experiments/peak_rectification.py --windows 52 --seeds 42 43 44

    # the same sweep as one process per seed (one core each), then merged;
    # every source must pin the same cutoffs, phase scheme, data sha256, git SHA
    for s in 42 43 44; do
        .venv/bin/python scripts/experiments/peak_rectification.py --windows 52 --seeds $s \
            --out outputs/experiments/pr2-seed$s &
    done; wait
    .venv/bin/python scripts/experiments/peak_rectification.py \
        --merge outputs/experiments/pr2-seed42 outputs/experiments/pr2-seed43 outputs/experiments/pr2-seed44

    # smoke run: first cutoff only, baseline + oversample k=2, two seeds
    .venv/bin/python scripts/experiments/peak_rectification.py --cutoffs 1 --arms baseline oversample \
        --oversample-factors 2 --seeds 42 43

    # regenerate summary.md from the log without training anything
    .venv/bin/python scripts/experiments/peak_rectification.py --summary-only

    # re-score every logged config from its saved forecasts against current
    # actuals and the current phase scheme; trains nothing, writes
    # log.rescored.jsonl + summary.rescored.md beside the originals
    .venv/bin/python scripts/experiments/peak_rectification.py --rescore \
        --out outputs/experiments/peak_rectification

Where output lands
------------------
`outputs/experiments/peak_rectification_round2/` (round 1 stays in
`outputs/experiments/peak_rectification/`; gitignored except `summary.md` and
`manifest.json`, which are committed by hand):
    manifest.json             the fixed part of the experiment (written once)
    log.jsonl                 one (state, action, reward) row per config per seed
    forecasts/<config_id>.csv pooled forecasts across cutoffs, for re-scoring
    summary.md                generated from log.jsonl, never edited by hand
    log.rescored.jsonl        `--rescore` output; log.jsonl is never modified
    summary.rescored.md       generated from log.rescored.jsonl
`--merge` writes manifest.json (seeds = the union, plus `merged_from`),
log.jsonl, forecasts/ and summary.md into `--out` from per-seed directories.
A run on a cutoff subset (`--cutoffs N` below the full split) defaults to a
sibling directory `peak_rectification_round2-smoke-<N>cutoffs/` so a smoke run
can never be mistaken for, or block, the real sweep.

Invariants
----------
- Every row in one log shares the same cutoffs, phase scheme, seed list and
  data-cache sha256; the manifest pins them and a run that differs in any of
  them fails loudly. The loader cannot refresh the cache mid-run.
- For each seed, the baseline config is run (or read back from the log)
  before any other arm, because every reward is relative to it.
- A `config_id` already present in the log is skipped (resumable).
- Quantiles must be enabled: WIS needs the quantile columns.
- A missing target phase in a run's forecasts (for example a single October
  cutoff has no peak-phase rows) yields a null reward, not a fabricated one.
- `summary.md` is regenerated from the log after every run, so it can never
  disagree with it.
- Every metrics block carries `phase_scheme` (rows from before the adapter
  recorded it are calendar_v1) and `by_phase_horizon`, the adapter's own
  scorer run on each horizon's subset of the forecasts.
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
import contextlib
import copy
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from agent.adapters.flu_forecast import FluForecastAdapter  # noqa: E402
from agent.orchestrator import _improvement_pct, _is_better  # noqa: E402
from agent.phase_evaluator import DEFAULT_PHASE_SCHEME  # noqa: E402
from src import config as cfg  # noqa: E402
from src.data_loader import FluDataLoader  # noqa: E402
from src.direct_forecast import DEFAULT_APPROACHING_PEAK_WEEKS  # noqa: E402
from src.pipeline import MIN_TRAIN_WINDOW_WEEKS, OVERSAMPLE_MIN_FACTOR  # noqa: E402

__all__ = [
    "Arm",
    "FIXED",
    "apply_delta",
    "arm_lambda",
    "arm_lambda_calendar",
    "arm_oversample",
    "arm_window",
    "build_arms",
    "build_manifest",
    "log_row",
    "main",
    "merge",
    "read_log",
    "rescore",
    "reward_for",
    "run_config",
    "score",
    "with_seed",
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
ARM_OVERSAMPLE = "oversample"
ARM_NAMES = (ARM_BASELINE, ARM_LAMBDA, ARM_LAMBDA_CALENDAR, ARM_WINDOW, ARM_OVERSAMPLE)

DEFAULT_LAMBDAS = (1.5, 2.0, 3.0, 5.0)
DEFAULT_CALENDAR_LAMBDAS = (2.0, 3.0)
DEFAULT_WINDOWS = (12, 26, 52)
DEFAULT_OVERSAMPLE_FACTORS = (2, 3, 5)
OVERSAMPLE_PHASE = TARGET_PHASE
BASELINE_CONFIG_ID = "baseline"

# Seeds: the stock random_state alone reproduces round 1's single-seed rows.
SEED_CONFIG_PATH = "xgboost.random_state"
STOCK_SEED: int = cfg.XGBOOST_PARAMS_V2["random_state"]
DEFAULT_SEEDS = (STOCK_SEED,)
SEED_SUFFIX = "__seed"

# Metrics blocks logged before the adapter recorded a phase scheme were all
# scored with the month map now called calendar_v1.
LEGACY_PHASE_SCHEME = "calendar_v1"

# ---- Output layout ----------------------------------------------------------

EXPERIMENT_NAME = "peak_rectification"
EXPERIMENT_ROUND = 2
ROUND1_OUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT_NAME
DEFAULT_OUT_DIR = ROUND1_OUT_DIR.with_name(f"{EXPERIMENT_NAME}_round{EXPERIMENT_ROUND}")
MANIFEST_NAME = "manifest.json"
LOG_NAME = "log.jsonl"
SUMMARY_NAME = "summary.md"
RESCORED_LOG_NAME = "log.rescored.jsonl"
RESCORED_SUMMARY_NAME = "summary.rescored.md"
FORECASTS_DIRNAME = "forecasts"
DATA_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "flusight_hospital_admissions.csv"
DATA_CACHE_HASH_KEY = "data_cache.sha256"
# Manifest keys every --merge source must share: rows scored on different
# splits, schemes, data snapshots or code are not comparable.
MERGE_PINNED_KEYS = ("cutoffs", "phase_scheme", DATA_CACHE_HASH_KEY, "git_sha")

SUMMARY_MISSING = "n/a"  # rendered when a metric is absent (e.g. no peak rows in a smoke run)

CLI_EPILOG = """\
examples:
  round 2 sweep (11 configs x 9 cutoffs x 3 seeds; ~1.5-2.5 h sequential with OMP_NUM_THREADS=1)
      %(prog)s --windows 52 --seeds 42 43 44
  the same, one process per seed in parallel, then merged
      %(prog)s --windows 52 --seeds 42 --out outputs/experiments/pr2-seed42   (and 43, 44)
      %(prog)s --merge outputs/experiments/pr2-seed42 outputs/experiments/pr2-seed43 outputs/experiments/pr2-seed44
  smoke run: first cutoff only, baseline + oversample k=2, two seeds (-smoke-1cutoffs sibling dir)
      %(prog)s --cutoffs 1 --arms baseline oversample --oversample-factors 2 --seeds 42 43
  regenerate summary.md from the log without training
      %(prog)s --summary-only
  re-score round 1 under the current phase scheme and actuals (writes *.rescored.*)
      %(prog)s --rescore --out outputs/experiments/peak_rectification
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
    seed: int = STOCK_SEED

    def __post_init__(self) -> None:
        if self.name not in ARM_NAMES:
            raise ValueError(f"Arm.name must be one of {ARM_NAMES}, got {self.name!r}")
        if not self.config_id or any(ch in self.config_id for ch in "/\\ "):
            raise ValueError(f"Arm.config_id must be a non-empty file-safe token, got {self.config_id!r}")

    @property
    def base_config_id(self) -> str:
        """The config id without its `__seed<N>` suffix: the key seeds are pooled by."""
        return _base_config_id(self.config_id)


def _base_config_id(config_id: str) -> str:
    return config_id.split(SEED_SUFFIX, 1)[0]


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
        describe=(
            f"peak-phase rows by target date (origin + h weeks) under {DEFAULT_PHASE_SCHEME} "
            f"weighted {weight:g}x"
        ),
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


def arm_oversample(factor: int, phase: str = OVERSAMPLE_PHASE) -> Arm:
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < OVERSAMPLE_MIN_FACTOR:
        raise ValueError(f"oversample factor must be an int >= {OVERSAMPLE_MIN_FACTOR}, got {factor!r}")
    return Arm(
        name=ARM_OVERSAMPLE,
        config_id=f"oversample_{factor}",
        config_delta={"data.oversample": {"phase": phase, "factor": factor}},
        describe=(
            f"{phase}-phase training rows by target date under {DEFAULT_PHASE_SCHEME} fitted {factor}x "
            f"({factor - 1} extra cop{'y' if factor == 2 else 'ies'} each)"
        ),
    )


def with_seed(arm: Arm, seed: int) -> Arm:
    """`arm` trained with XGBoost random_state `seed`, config id suffixed `__seed<N>`.

    Always suffixed, even for one seed, so per-seed runs in separate
    directories have distinct config ids and forecast files for `--merge`.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an int >= 0, got {seed!r}")
    return Arm(
        name=arm.name,
        config_id=f"{arm.config_id}{SEED_SUFFIX}{seed}",
        config_delta={**arm.config_delta, SEED_CONFIG_PATH: seed},
        describe=arm.describe,
        seed=seed,
    )


def build_arms(
    arm_names: Sequence[str],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    calendar_lambdas: Sequence[float] = DEFAULT_CALENDAR_LAMBDAS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    oversample_factors: Sequence[int] = DEFAULT_OVERSAMPLE_FACTORS,
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
    if ARM_OVERSAMPLE in arm_names:
        arms.extend(arm_oversample(k) for k in oversample_factors)
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

    The pooled frame keeps every location, excluded ones included, and is
    written to forecasts/<config_id>.csv for re-scoring and plotting;
    `score` drops excluded locations, so scoring still matches the loop.
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
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(forecasts_dir / f"{arm.config_id}.csv", index=False)
    return pooled, time.perf_counter() - t0


def score(adapter: FluForecastAdapter, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, Any]:
    """The loop's own scorer, trimmed to the sections the log keeps.

    `by_phase_horizon[phase][horizon]` is the adapter's `by_phase` block
    computed on that horizon's rows alone (same fields as `by_phase`), so a
    pooled peak number can be read back per horizon without a second scorer.
    Excluded locations are dropped first, the same way
    `FluForecastAdapter.load_data` drops them.
    """
    if adapter.exclude_locations:
        forecasts = forecasts[~forecasts["location"].astype(str).isin(adapter.exclude_locations)]
    if forecasts.empty:
        raise ValueError(f"no forecast rows left to score after excluding {sorted(adapter.exclude_locations)}")
    metrics = adapter.compute_metrics(forecasts, actuals)
    by_phase_horizon: dict[str, dict[str, Any]] = {}
    for horizon in sorted(int(h) for h in forecasts["horizon"].unique()):
        subset = adapter.compute_metrics(forecasts[forecasts["horizon"] == horizon], actuals)
        for phase, block in subset["by_phase"].items():
            by_phase_horizon.setdefault(phase, {})[str(horizon)] = block
    return {
        "phase_scheme": metrics.get("phase_scheme", LEGACY_PHASE_SCHEME),
        "overall": metrics["overall"],
        "by_phase": metrics["by_phase"],
        # json turns int keys into strings anyway; do it here so the in-memory
        # row equals the re-read row
        "by_horizon": {str(h): v for h, v in metrics["by_horizon"].items()},
        "by_phase_horizon": by_phase_horizon,
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
        "base_config_id": arm.base_config_id,
        "seed": arm.seed,
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


def _data_cache_mtime() -> str | None:
    if not DATA_CACHE_PATH.exists():
        return None
    return datetime.fromtimestamp(DATA_CACHE_PATH.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def _data_cache_sha256() -> str | None:
    if not DATA_CACHE_PATH.exists():
        return None
    return hashlib.sha256(DATA_CACHE_PATH.read_bytes()).hexdigest()


def _manifest_value(manifest: Mapping[str, Any], key: str) -> Any:
    """Dotted lookup; None when any segment is absent (older manifests lack the data hash)."""
    node: Any = manifest
    for segment in key.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            return None
        node = node[segment]
    return node


def _check_data_vintage(pinned_sha256: str | None) -> None:
    """Raise when the data cache no longer matches the snapshot the manifest pinned."""
    now = _data_cache_sha256()
    if now is None or now != pinned_sha256:
        raise ValueError(
            f"{DATA_CACHE_PATH} has sha256 {now!r} but the manifest pins {pinned_sha256!r}: the CDC "
            "cache changed (refresh or backfill), so new rows would not be comparable with the logged "
            "baseline. Start a fresh --out directory, or restore the pinned cache."
        )


@contextlib.contextmanager
def _frozen_data_cache() -> Iterator[None]:
    """Keep FluDataLoader on the cached CSV for the duration, however old it gets.

    Why: run_pipeline builds a FluDataLoader per cutoff, and the loader
    refetches from GitHub once the cache is older than 168 h. A refresh in
    the middle of a sweep would train later arms on a different snapshot
    from their seed's baseline. A missing cache still fetches.
    """
    with mock.patch.object(FluDataLoader, "is_cache_stale", return_value=False):
        yield


def _require_project_root_cwd() -> None:
    """FluDataLoader's cache path is cwd-relative; the pinned hash is of DATA_CACHE_PATH."""
    loader_cache = FluDataLoader().cache_file.resolve()
    if loader_cache != DATA_CACHE_PATH:
        raise ValueError(
            f"run from {PROJECT_ROOT}: FluDataLoader would read {loader_cache}, "
            f"but the harness pins and scores {DATA_CACHE_PATH}"
        )


def build_manifest(
    cutoffs: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    stride_weeks: int = DEFAULT_STRIDE_WEEKS,
) -> dict[str, Any]:
    cache_mtime = _data_cache_mtime()
    import xgboost  # local: keep module import cheap for --summary-only

    config = _base_config()
    return {
        "experiment": EXPERIMENT_NAME,
        "round": EXPERIMENT_ROUND,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "cutoffs": list(cutoffs),
        "n_cutoffs": len(cutoffs),
        "seeds": list(seeds),
        "phase_scheme": DEFAULT_PHASE_SCHEME,
        "floor": dict(config["floor"]),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "data_cache": {
            "path": str(DATA_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "mtime": cache_mtime,
            "sha256": _data_cache_sha256(),
        },
        "host": platform.node(),
        "python_version": platform.python_version(),
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        **FIXED,
        "split": {**FIXED["split"], "stride_weeks": stride_weeks},
    }


def _ensure_manifest(
    out_dir: Path,
    cutoffs: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    stride_weeks: int = DEFAULT_STRIDE_WEEKS,
) -> dict[str, Any]:
    """Write the manifest once; on a resumed run, refuse a different split, seed list,
    phase scheme or data snapshot.

    A round-1 manifest has no `seeds` / `phase_scheme`; it pinned the stock
    seed and calendar_v1, so those are its values here. A manifest without
    a data hash cannot prove its vintage, so resuming it fails.
    """
    path = out_dir / MANIFEST_NAME
    if path.exists():
        manifest = json.loads(path.read_text())
        pinned = {
            "cutoffs": (list(manifest.get("cutoffs", [])), list(cutoffs)),
            "seeds": (list(manifest.get("seeds", DEFAULT_SEEDS)), list(seeds)),
            "phase_scheme": (manifest.get("phase_scheme", LEGACY_PHASE_SCHEME), DEFAULT_PHASE_SCHEME),
        }
        for key, (was, now) in pinned.items():
            if was != now:
                raise ValueError(
                    f"{path} pins {key} {was!r} but this run uses {now!r}; pass --out for a separate directory"
                )
        _check_data_vintage(_manifest_value(manifest, DATA_CACHE_HASH_KEY))
        return manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cutoffs, seeds, stride_weeks)
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


def _row_seed(row: Mapping[str, Any]) -> int:
    """Rows from before seeds were logged all ran the stock seed."""
    return int(row.get("seed", STOCK_SEED))


def _row_base_id(row: Mapping[str, Any]) -> str:
    return str(row.get("base_config_id") or _base_config_id(row["config_id"]))


def _peak_wis(row: Mapping[str, Any]) -> float | None:
    return _target_value(row["metrics"])


def _overall_wis(row: Mapping[str, Any]) -> float | None:
    value = (row["metrics"].get("overall") or {}).get(GUARD_METRIC)
    return None if value is None else float(value)


def _target_wis_at(row: Mapping[str, Any], horizon: str) -> float | None:
    by_phase_horizon = row["metrics"].get("by_phase_horizon") or {}
    value = ((by_phase_horizon.get(TARGET_PHASE) or {}).get(horizon) or {}).get(TARGET_METRIC)
    return None if value is None else float(value)


def _overall_wis_at(row: Mapping[str, Any], horizon: str) -> float | None:
    value = ((row["metrics"].get("by_horizon") or {}).get(horizon) or {}).get(TARGET_METRIC)
    return None if value is None else float(value)


def _mean(values: Sequence[float | None]) -> float | None:
    """Mean of the present values; None when any is missing, so a gap is never averaged over."""
    if not values or any(v is None for v in values):
        return None
    return statistics.fmean(v for v in values if v is not None)


def _sd(values: Sequence[float | None]) -> float | None:
    """Sample sd across seeds; None below two seeds or when any value is missing."""
    if len(values) < 2 or any(v is None for v in values):
        return None
    return statistics.stdev(v for v in values if v is not None)


def _group_by_config(rows_sorted: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Base config id -> its rows ordered by seed, keeping the arm order of `rows_sorted`."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows_sorted:
        groups.setdefault(_row_base_id(row), []).append(row)
    return {key: sorted(rows, key=_row_seed) for key, rows in groups.items()}


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], value_of: Any,
) -> list[tuple[int, float | None]]:
    """(seed, value - baseline value of the same seed) for each of `rows`."""
    baseline_by_seed = {_row_seed(r): value_of(r) for r in baseline_rows}
    out: list[tuple[int, float | None]] = []
    for row in rows:
        value, base = value_of(row), baseline_by_seed.get(_row_seed(row))
        out.append((_row_seed(row), None if value is None or base is None else value - base))
    return out


def _horizons(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    found = {h for row in rows for h in (row["metrics"].get("by_horizon") or {})}
    return sorted(found, key=int)


def _horizon_table(
    title: str,
    note: str,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    horizons: Sequence[str],
    value_at: Any,
) -> list[str]:
    """One row per config: mean-over-seeds WIS at each horizon, with the delta vs the baseline mean.

    With several seeds each cell also gives the range of the per-seed paired
    deltas, so a reader can tell a horizon effect from seed noise.
    """
    baseline_rows = groups.get(BASELINE_CONFIG_ID, [])
    multi_seed = any(len(rows) > 1 for rows in groups.values())
    lines = [
        "",
        f"## {title}",
        "",
        note,
        "",
        "| config | " + " | ".join(f"h{h} (delta)" for h in horizons) + " |",
        "|---|" + "---:|" * len(horizons),
    ]
    for config_id, rows in groups.items():
        cells = []
        for h in horizons:
            value = _mean([value_at(r, h) for r in rows])
            base = _mean([value_at(r, h) for r in baseline_rows])
            delta = None if value is None or base is None else value - base
            spread = ""
            if multi_seed:
                paired = [d for _, d in _paired_deltas(rows, baseline_rows, lambda r, h=h: value_at(r, h))]
                spread = (
                    f"; paired {SUMMARY_MISSING}" if any(d is None for d in paired)
                    else f"; paired {_fmt_signed(min(paired))}..{_fmt_signed(max(paired))}"
                )
            cells.append(SUMMARY_MISSING if value is None else f"{_fmt(value)} ({_fmt_signed(delta)}{spread})")
        lines.append(f"| `{config_id}` | " + " | ".join(cells) + " |")
    return lines


def write_summary(out_dir: Path, log_name: str = LOG_NAME, summary_name: str = SUMMARY_NAME) -> Path:
    """Render `summary_name` from `log_name`. Rows are ordered by arm, then log order."""
    log_path = out_dir / log_name
    rows = read_log(log_path)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    order = {name: i for i, name in enumerate(ARM_NAMES)}
    rows_sorted = [
        row for _, row in sorted(enumerate(rows), key=lambda ir: (order.get(ir[1]["arm"], len(order)), ir[0]))
    ]
    schemes = sorted({row["metrics"].get("phase_scheme", LEGACY_PHASE_SCHEME) for row in rows})

    lines = [
        f"# {EXPERIMENT_NAME}: results",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{log_name}` "
        f"({len(rows)} config{'s' if len(rows) != 1 else ''}). Do not edit by hand; rerun the script "
        f"or pass `{'--rescore' if log_name == RESCORED_LOG_NAME else '--summary-only'}`.",
        "",
        f"Target: {TARGET_PHASE} {TARGET_METRIC.upper()} (lower is better). Guard: overall "
        f"{GUARD_METRIC.upper()}. Every delta is vs this run's own `{BASELINE_CONFIG_ID}` row "
        "(of the same seed, when several seeds ran).",
        "",
        f"Phase scheme: {', '.join(f'`{s}`' for s in schemes) or SUMMARY_MISSING}.",
    ]
    if manifest:
        lines += [
            "",
            f"Split: train from {manifest['split']['train_start_date']}, eval "
            f"{manifest['split']['eval_start_date']}..{manifest['split']['eval_end_date']}, "
            f"{manifest['n_cutoffs']} cutoff(s) at stride {manifest['split']['stride_weeks']}: "
            f"{', '.join(manifest['cutoffs'])}. Excluded from scoring: {', '.join(manifest['exclude_locations'])}. "
            f"Seeds: {', '.join(str(s) for s in manifest.get('seeds', DEFAULT_SEEDS))}. "
            f"Git {manifest['git_sha'][:10]}{' (dirty)' if manifest.get('git_dirty') else ''}, "
            f"data cache {manifest['data_cache']['mtime']}"
            f"{' (sha256 ' + manifest['data_cache']['sha256'][:12] + ')' if manifest['data_cache'].get('sha256') else ''}.",
        ]
    rescored = [row["rescored"] for row in rows if row.get("rescored")]
    if rescored:
        lines += [
            "",
            f"Re-scored {rescored[0]['at']} from `{FORECASTS_DIRNAME}/` under `{rescored[0]['phase_scheme']}` "
            f"against the data cache of {rescored[0]['data_cache_mtime']}; no model was retrained. The "
            f"original `{LOG_NAME}` (scored under "
            f"{', '.join(sorted({f'`{r['original_phase_scheme']}`' for r in rescored}))}) is untouched.",
        ]
    lines += [
        "",
        "| arm | config | peak WIS | overall WIS | peak bias | delta peak WIS | improvement | better | seconds |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in rows_sorted:
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

    groups = _group_by_config(rows_sorted)
    baseline_rows = groups.get(BASELINE_CONFIG_ID, [])
    if any(len(group) > 1 for group in groups.values()):
        baseline_peak = _mean([_peak_wis(r) for r in baseline_rows])
        lines += [
            "",
            "## Across seeds",
            "",
            "Mean and sample sd over seeds. `delta vs baseline mean` compares means; `paired deltas` "
            "are config minus baseline at the same seed, the comparison that cancels seed noise.",
            "",
            f"| config | seeds | {TARGET_PHASE} WIS mean | sd | overall WIS mean | sd "
            f"| delta {TARGET_PHASE} WIS vs baseline mean | paired {TARGET_PHASE} WIS deltas (seed: delta) |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for config_id, group in groups.items():
            peaks = [_peak_wis(r) for r in group]
            overalls = [_overall_wis(r) for r in group]
            mean_peak = _mean(peaks)
            delta = None if mean_peak is None or baseline_peak is None else mean_peak - baseline_peak
            paired = ", ".join(
                f"{seed}: {_fmt_signed(d, 2)}" for seed, d in _paired_deltas(group, baseline_rows, _peak_wis)
            )
            lines.append(
                f"| `{config_id}` | {len(group)} | {_fmt(mean_peak, 2)} | {_fmt(_sd(peaks), 2)} "
                f"| {_fmt(_mean(overalls), 2)} | {_fmt(_sd(overalls), 2)} | {_fmt_signed(delta, 2)} | {paired} |"
            )

    horizons = _horizons(rows)
    if horizons:
        lines += _horizon_table(
            f"{TARGET_PHASE.capitalize()} WIS by horizon",
            f"{TARGET_PHASE.capitalize()}-phase WIS on each horizon's forecasts alone "
            f"(`by_phase_horizon`), mean over seeds; delta vs `{BASELINE_CONFIG_ID}`, and with "
            "several seeds the min..max of the same-seed paired deltas. "
            f"`{SUMMARY_MISSING}` when the phase is absent from that horizon's forecasts or the row "
            "was logged before phase x horizon was recorded.",
            groups, horizons, _target_wis_at,
        )
        lines += _horizon_table(
            "Overall WIS by horizon",
            f"All scored phases (`by_horizon`), mean over seeds; delta vs `{BASELINE_CONFIG_ID}`. "
            "The pooled WIS above is the mean of these over horizons, so it can hide an arm that "
            "helps h1/h2 and hurts h3/h4.",
            groups, horizons, _overall_wis_at,
        )

    lines += ["", "## Configs", ""]
    for row in rows_sorted:
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
        "- Phases follow the phase scheme named above (`agent.phase_evaluator`); the same phase name "
        "covers different dates in different schemes, so never compare across schemes.",
        "",
    ]
    summary_path = out_dir / summary_name
    summary_path.write_text("\n".join(lines))
    return summary_path


def rescore(out_dir: Path, adapter: FluForecastAdapter, actuals: pd.DataFrame) -> Path:
    """Re-score every logged config from forecasts/<config_id>.csv; trains nothing.

    Writes log.rescored.jsonl (rebuilt from scratch on every call) and
    summary.rescored.md. log.jsonl and summary.md are only read. Rewards
    are recomputed against the re-scored baseline of the same seed. Returns
    the rescored log path.
    """
    rows = read_log(out_dir / LOG_NAME)
    if not rows:
        raise ValueError(f"--rescore: nothing logged at {out_dir / LOG_NAME}")
    forecasts_dir = out_dir / FORECASTS_DIRNAME
    missing = [r["config_id"] for r in rows if not (forecasts_dir / f"{r['config_id']}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"--rescore: no saved forecasts in {forecasts_dir} for {missing}")

    stamp = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase_scheme": DEFAULT_PHASE_SCHEME,
        "data_cache_mtime": _data_cache_mtime(),
        "data_cache_sha256": _data_cache_sha256(),
    }
    metrics_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        # dtype=str keeps zero-padded FIPS ("01") as written.
        forecasts = pd.read_csv(forecasts_dir / f"{row['config_id']}.csv", dtype={"location": str})
        metrics_by_id[row["config_id"]] = score(adapter, forecasts, actuals)

    baseline_by_seed = {
        _row_seed(r): metrics_by_id[r["config_id"]] for r in rows if _row_base_id(r) == BASELINE_CONFIG_ID
    }
    rescored_path = out_dir / RESCORED_LOG_NAME
    with rescored_path.open("w") as fh:
        for row in rows:
            seed = _row_seed(row)
            if seed not in baseline_by_seed:
                raise ValueError(f"--rescore: no {BASELINE_CONFIG_ID} row for seed {seed} (config {row['config_id']})")
            metrics = metrics_by_id[row["config_id"]]
            new_row = {
                **row,
                "metrics": metrics,
                "reward": reward_for(metrics, baseline_by_seed[seed]),
                "rescored": {**stamp, "original_phase_scheme": row["metrics"].get("phase_scheme", LEGACY_PHASE_SCHEME)},
            }
            fh.write(json.dumps(new_row, default=str) + "\n")
    write_summary(out_dir, RESCORED_LOG_NAME, RESCORED_SUMMARY_NAME)
    return rescored_path


def merge(out_dir: Path, sources: Sequence[Path]) -> Path:
    """Merge sweep directories run in parallel (one per seed) into `out_dir`; trains nothing.

    Every source must pin the same MERGE_PINNED_KEYS, and the sources'
    seeds and config ids must be disjoint. Rewards were paired with each
    source's own same-seed baseline, so rows are copied unchanged. Writes
    manifest.json (seeds = the union, `merged_from` = the sources),
    log.jsonl, forecasts/ and summary.md. Returns the merged log path.
    """
    if len(sources) < 2:
        raise ValueError(f"--merge needs at least two source directories, got {[str(s) for s in sources]}")
    if (out_dir / MANIFEST_NAME).exists() or (out_dir / LOG_NAME).exists():
        raise ValueError(f"--merge: {out_dir} already has a manifest or log; merge into a new --out directory")
    manifests: list[dict[str, Any]] = []
    for source in sources:
        if not (source / MANIFEST_NAME).exists():
            raise FileNotFoundError(f"--merge: no {MANIFEST_NAME} in {source}")
        manifests.append(json.loads((source / MANIFEST_NAME).read_text()))
    for key in MERGE_PINNED_KEYS:
        values = {json.dumps(_manifest_value(m, key)) for m in manifests}
        if len(values) != 1 or values == {json.dumps(None)}:
            raise ValueError(f"--merge: sources must pin one non-null {key}, got {sorted(values)}")

    seeds: list[int] = [seed for m in manifests for seed in m.get("seeds", DEFAULT_SEEDS)]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"--merge: sources share seeds {sorted(seeds)}")
    rows = [row for source in sources for row in read_log(source / LOG_NAME)]
    ids = [row["config_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"--merge: duplicate config ids across sources: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    missing = [
        str(source / FORECASTS_DIRNAME / f"{row['config_id']}.csv")
        for source in sources for row in read_log(source / LOG_NAME)
        if not (source / FORECASTS_DIRNAME / f"{row['config_id']}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(f"--merge: saved forecasts missing: {missing}")

    forecasts_dir = out_dir / FORECASTS_DIRNAME
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for row in read_log(source / LOG_NAME):
            shutil.copy2(source / FORECASTS_DIRNAME / f"{row['config_id']}.csv", forecasts_dir)
    manifest = {
        **manifests[0],
        "seeds": sorted(seeds),
        "merged_from": [
            {"dir": str(source), "seeds": m.get("seeds", list(DEFAULT_SEEDS)), "created_at": m.get("created_at")}
            for source, m in zip(sources, manifests)
        ],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    log_path = out_dir / LOG_NAME
    with log_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    write_summary(out_dir)
    return log_path


# ---- CLI --------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peak rectification experiment: lambda / calendar-lambda / window / oversample arms vs baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )
    parser.add_argument(
        "--cutoffs", type=int, default=None, metavar="N",
        help="use only the first N cutoffs at --stride-weeks (smoke runs). Default: all.",
    )
    parser.add_argument(
        "--stride-weeks", type=int, default=DEFAULT_STRIDE_WEEKS, metavar="K",
        help=f"weeks between eval cutoffs on the pinned split. Default: {DEFAULT_STRIDE_WEEKS} "
             "(9 cutoffs); 1 gives a weekly hindcast (35 cutoffs).",
    )
    parser.add_argument(
        "--pin-data-sha256", default=None, metavar="SHA",
        help="fresh run only: train on the data cache as it is, without the usual refresh, and fail "
             "unless its sha256 is SHA (e.g. another run's manifest value, to share its snapshot).",
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
        "--oversample-factors", nargs="+", type=int, default=list(DEFAULT_OVERSAMPLE_FACTORS), metavar="K",
        help=f"{OVERSAMPLE_PHASE}-phase copy factors for the oversample arm. "
             f"Default: {list(DEFAULT_OVERSAMPLE_FACTORS)}.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), metavar="S",
        help=f"XGBoost random_state values; each config runs once per seed and its config id gets "
             f"__seed<N>. Default: {list(DEFAULT_SEEDS)} (round 1's seed).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"output directory. Default: {DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT)} "
             "(a `-smoke-<N>cutoffs` sibling when --cutoffs is a subset).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--summary-only", action="store_true",
        help="regenerate summary.md from the existing log and exit; trains nothing.",
    )
    mode.add_argument(
        "--rescore", action="store_true",
        help=f"re-score every logged config from {FORECASTS_DIRNAME}/ against current actuals and phase "
             f"scheme; writes {RESCORED_LOG_NAME} and {RESCORED_SUMMARY_NAME}, trains nothing.",
    )
    mode.add_argument(
        "--merge", nargs="+", type=Path, default=None, metavar="DIR",
        help="merge per-seed sweep directories (same cutoffs, phase scheme, data sha256, git SHA; "
             "disjoint seeds) into --out; trains nothing.",
    )
    parser.add_argument("--verbose", action="store_true", help="show the pipeline's training chatter")
    return parser.parse_args(argv)


def _resolve_cutoffs(n: int | None, stride_weeks: int = DEFAULT_STRIDE_WEEKS) -> tuple[list[str], bool]:
    """Return (cutoffs, is_subset). `n` beyond the split is a mistake, not a clamp."""
    if stride_weeks < 1:
        raise ValueError(f"--stride-weeks must be >= 1, got {stride_weeks}")
    full = cfg.generate_eval_cutoffs(stride_weeks)
    if n is None:
        return full, False
    if n < 1 or n > len(full):
        raise ValueError(f"--cutoffs must be in [1, {len(full)}] for stride {stride_weeks}, got {n}")
    return full[:n], n < len(full)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cutoffs, is_subset = _resolve_cutoffs(args.cutoffs, args.stride_weeks)
    out_dir: Path = args.out or (
        DEFAULT_OUT_DIR.with_name(f"{DEFAULT_OUT_DIR.name}-smoke-{len(cutoffs)}cutoffs") if is_subset else DEFAULT_OUT_DIR
    )
    out_dir = out_dir.resolve()
    log_path = out_dir / LOG_NAME

    if args.summary_only:
        if not log_path.exists():
            raise ValueError(f"--summary-only: nothing logged at {log_path}; run the experiment first")
        path = write_summary(out_dir)
        print(f"summary written: {path} ({len(read_log(log_path))} rows)")
        return 0

    if args.merge:
        sources = [source.resolve() for source in args.merge]
        path = merge(out_dir, sources)
        print(f"merged {len(read_log(path))} rows from {len(sources)} directories into {out_dir}")
        return 0

    if args.rescore:
        adapter = FluForecastAdapter(exclude_locations=list(EXCLUDE_LOCATIONS))
        t0 = time.perf_counter()
        path = rescore(out_dir, adapter, adapter._load_actuals())
        print(
            f"rescored {len(read_log(path))} rows under {DEFAULT_PHASE_SCHEME} in {time.perf_counter() - t0:.1f}s: "
            f"{path}, {out_dir / RESCORED_SUMMARY_NAME}"
        )
        return 0

    seeds: list[int] = list(args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"--seeds must be distinct, got {seeds}")
    arms = build_arms(args.arms, args.lambdas, args.calendar_lambdas, args.windows, args.oversample_factors)
    # Seed-major order: each seed is a full replicate whose baseline runs first.
    runs = [with_seed(arm, seed) for seed in seeds for arm in arms]
    _require_project_root_cwd()
    if not (out_dir / MANIFEST_NAME).exists():
        if args.pin_data_sha256:
            _check_data_vintage(args.pin_data_sha256)  # share another run's snapshot: no refresh
        else:
            # A fresh run starts from the newest data; a resume must not refresh.
            FluDataLoader().fetch_data()
    manifest = _ensure_manifest(out_dir, cutoffs, seeds, args.stride_weeks)
    pinned_sha256 = _manifest_value(manifest, DATA_CACHE_HASH_KEY)
    _check_data_vintage(pinned_sha256)  # also rejects a fresh manifest with no cache to hash
    logged = {row["config_id"]: row for row in read_log(log_path)}

    print(
        f"peak rectification round {EXPERIMENT_ROUND}: {len(arms)} config(s) x {len(seeds)} seed(s) x "
        f"{len(cutoffs)} cutoff(s) -> {out_dir}"
    )
    print(f"  cutoffs: {', '.join(cutoffs)}; seeds: {', '.join(str(s) for s in seeds)}")
    if logged:
        print(f"  resuming: {len(logged)} config(s) already logged: {', '.join(sorted(logged))}")

    print(f"  data cache sha256 {pinned_sha256[:12]} (refresh disabled for this run)")
    with _frozen_data_cache():
        _run_sweep(runs, cutoffs, logged, log_path, out_dir, pinned_sha256)
    print(f"done. summary: {out_dir / SUMMARY_NAME}")
    return 0


def _run_sweep(
    runs: Sequence[Arm],
    cutoffs: Sequence[str],
    logged: Mapping[str, Mapping[str, Any]],
    log_path: Path,
    out_dir: Path,
    pinned_sha256: str,
) -> None:
    """Run (or read back) every arm in order; the caller has frozen the data cache."""
    adapter = FluForecastAdapter(exclude_locations=list(EXCLUDE_LOCATIONS))
    # Loaded after the vintage check, from the same pinned file training reads.
    actuals = adapter._load_actuals()
    forecasts_dir = out_dir / FORECASTS_DIRNAME

    baseline_metrics: Mapping[str, Any] | None = None
    for arm in runs:  # within each seed, baseline comes first by construction
        if arm.base_config_id == BASELINE_CONFIG_ID:
            baseline_metrics = None
        if arm.config_id in logged:
            row = logged[arm.config_id]
            print(f"  skip {arm.config_id} (logged {row['run_at']})")
        else:
            print(f"  run  {arm.config_id}: {arm.describe}")
            pooled, seconds = run_config(adapter, arm, cutoffs, forecasts_dir)
            _check_data_vintage(pinned_sha256)
            metrics = score(adapter, pooled, actuals)
            reference = metrics if arm.base_config_id == BASELINE_CONFIG_ID else baseline_metrics
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
        if arm.base_config_id == BASELINE_CONFIG_ID:
            baseline_metrics = row["metrics"]
        write_summary(out_dir)


if __name__ == "__main__":
    sys.exit(main())
