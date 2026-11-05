"""Paired significance test of a candidate forecast set against the baseline, per forecast date.

No training. Reads a harness run directory (`peak_rectification.py` output: manifest,
`forecasts/<config>__seed<N>.csv`), builds the candidate (a logged config, or a logged
emphasis config passed through the horizon gate of `gated_emphasis.py`), scores every row
with the harness scorer, and tests whether the candidate's WIS differs from the baseline's.

Unit of analysis (see `agent/significance.py`): for each forecast date, the sum over that
date's in-scope rows of WIS(candidate) - WIS(baseline), divided by the mean number of in-scope
rows per date, then averaged over the run's seeds. Dates have different numbers of in-scope
rows (a peak-phase scope holds only h4 rows at its first date and only h1 rows at its last),
so an unweighted per-date mean is a different estimand from pooled WIS and can even flip its
sign; with this scaling the mean over dates equals the pooled row-level WIS delta.
In-scope rows: every target week (`--phase all`) or the target weeks in one phase of
`adiga_2026_10`. The decision is the pre-registered rule: significant iff p < alpha,
and an improvement iff the mean delta is also negative.

Usage:
    # pre-registered primary test (documentation/experiments/holdout_2024_25/preregistration.md)
    .venv/bin/python scripts/experiments/significance_test.py \\
        --run-dir outputs/experiments/peak_rectification_holdout_2024_25 --candidate oversample_5 --gate horizon
    # in-sample check on the 2025-26 weekly hindcast
    .venv/bin/python scripts/experiments/significance_test.py --candidate oversample_5 --gate horizon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.significance import (  # noqa: E402
    BootstrapInterval,
    DieboldMarianoResult,
    block_bootstrap_interval,
    diebold_mariano,
)
from scripts.experiments.gated_emphasis import (  # noqa: E402
    KEY_COLUMNS,
    PHASE_SCHEME,
    compose,
    horizon_mask,
    score_rows,
)

__all__ = [
    "ALL_PHASES",
    "GATE_HORIZON",
    "GATE_NONE",
    "TestReport",
    "build_candidate",
    "date_deltas",
    "pooled_date_series",
    "load_seed_forecasts",
    "main",
    "run_test",
]

DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / "peak_rectification_weekly"
BASELINE = "baseline"
ALL_PHASES = "all"
GATE_NONE = "none"
GATE_HORIZON = "horizon"
GATES = (GATE_NONE, GATE_HORIZON)
# Pre-registered settings (preregistration.md): forecasts up to 3 weeks apart share target
# weeks, so autocorrelation is allowed to lag 3; bootstrap blocks span 4 adjacent dates.
DEFAULT_MAX_LAG = 3
DEFAULT_BLOCK_LENGTH = 4
DEFAULT_ALPHA = 0.05
DEFAULT_PHASE = "peak"


@dataclass(frozen=True, slots=True)
class TestReport:
    run_dir: str
    eval_window: str
    seeds: tuple[int, ...]
    candidate: str
    gate: str
    phase: str
    n_dates: int
    first_date: str
    last_date: str
    baseline_wis: float
    candidate_wis: float
    delta_by_horizon: dict[str, float]
    dm: DieboldMarianoResult
    bootstrap: BootstrapInterval
    alpha: float
    significant: bool
    improvement: bool


def load_seed_forecasts(run_dir: Path, config: str, seed: int) -> pd.DataFrame:
    path = run_dir / "forecasts" / f"{config}__seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no forecasts for config {config!r} seed {seed} at {path}")
    frame = pd.read_csv(path, dtype={"location": str, "cutoff_date": str})
    frame["location"] = frame["location"].str.zfill(2)
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def build_candidate(baseline: pd.DataFrame, emphasis: pd.DataFrame, gate: str) -> pd.DataFrame:
    """The candidate forecast set: `emphasis` as is, or composed with the baseline by `gate`."""
    if gate == GATE_NONE:
        return emphasis
    if gate == GATE_HORIZON:
        return compose(baseline, emphasis, horizon_mask(baseline.sort_values(list(KEY_COLUMNS))))
    raise ValueError(f"gate must be one of {GATES}, got {gate!r}")


def date_deltas(candidate_scored: pd.DataFrame, baseline_scored: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Per-row WIS deltas joined on the forecast key, restricted to `phase` (or all rows).

    Returns columns cutoff_date, horizon, delta, wis_candidate, wis_baseline.
    """
    keys = list(KEY_COLUMNS)
    joined = candidate_scored[keys + ["wis", "phase"]].merge(
        baseline_scored[keys + ["wis", "phase"]], on=keys, suffixes=("_candidate", "_baseline"), validate="1:1"
    )
    if len(joined) != len(baseline_scored) or len(joined) != len(candidate_scored):
        raise ValueError(
            f"candidate ({len(candidate_scored)}) and baseline ({len(baseline_scored)}) scored rows "
            f"do not share their keys ({len(joined)} matched)"
        )
    if phase != ALL_PHASES:
        joined = joined[joined["phase_baseline"] == phase]
    if joined.empty:
        raise ValueError(f"no scored rows in phase {phase!r}")
    joined["delta"] = joined["wis_candidate"] - joined["wis_baseline"]
    return joined[["cutoff_date", "horizon", "delta", "wis_candidate", "wis_baseline"]]


def pooled_date_series(rows: pd.DataFrame) -> pd.Series:
    """Per-date delta scaled so its mean over dates equals the pooled row-level mean delta.

    `rows` holds one seed's in-scope rows (cutoff_date, delta). Value at date t:
    sum of date t's deltas / (rows / dates).
    """
    if rows.empty:
        raise ValueError("no rows to aggregate")
    sums = rows.groupby("cutoff_date")["delta"].sum().sort_index()
    rows_per_date = len(rows) / len(sums)
    return sums / rows_per_date


def run_test(
    run_dir: Path,
    candidate: str,
    gate: str = GATE_NONE,
    phase: str = DEFAULT_PHASE,
    max_lag: int = DEFAULT_MAX_LAG,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    alpha: float = DEFAULT_ALPHA,
) -> TestReport:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("phase_scheme") != PHASE_SCHEME:
        raise ValueError(f"run phase scheme is {manifest.get('phase_scheme')!r}, this script scores {PHASE_SCHEME!r}")
    cache = PROJECT_ROOT / manifest["data_cache"]["path"]
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != manifest["data_cache"]["sha256"]:
        raise ValueError(f"{cache} has sha256 {digest} but the run pinned {manifest['data_cache']['sha256']}")
    actuals = pd.read_csv(cache, dtype={"location": str})
    seeds = tuple(int(s) for s in manifest["seeds"])

    per_seed: list[pd.DataFrame] = []
    for seed in seeds:
        baseline = load_seed_forecasts(run_dir, BASELINE, seed)
        cand = build_candidate(baseline, load_seed_forecasts(run_dir, candidate, seed), gate)
        rows = date_deltas(score_rows(cand, actuals), score_rows(baseline, actuals), phase)
        per_seed.append(rows.assign(seed=seed))
    rows = pd.concat(per_seed, ignore_index=True)

    by_date_seed = pd.concat(
        {seed: pooled_date_series(seed_rows) for seed, seed_rows in rows.groupby("seed")}, axis=1
    )
    if by_date_seed.isna().any().any():
        raise ValueError("seeds do not cover the same forecast dates; refusing to average across them")
    deltas = by_date_seed.mean(axis=1).sort_index()

    dm = diebold_mariano(deltas.to_list(), max_lag=max_lag)
    interval = block_bootstrap_interval(deltas.to_list(), block_length=block_length)
    significant = dm.p_value < alpha
    return TestReport(
        run_dir=str(run_dir),
        eval_window=manifest.get("split", {}).get("eval_window", "primary"),
        seeds=seeds,
        candidate=candidate,
        gate=gate,
        phase=phase,
        n_dates=len(deltas),
        first_date=str(deltas.index[0]),
        last_date=str(deltas.index[-1]),
        baseline_wis=float(rows["wis_baseline"].mean()),
        candidate_wis=float(rows["wis_candidate"].mean()),
        delta_by_horizon={f"h{int(h)}": float(v) for h, v in rows.groupby("horizon")["delta"].mean().items()},
        dm=dm,
        bootstrap=interval,
        alpha=alpha,
        significant=significant,
        improvement=significant and dm.mean_delta < 0,
    )


def render(report: TestReport) -> str:
    verdict = (
        "significant improvement" if report.improvement
        else "significant harm" if report.significant
        else "not significant"
    )
    horizons = ", ".join(f"{h} {v:+.2f}" for h, v in report.delta_by_horizon.items())
    return "\n".join([
        f"{report.candidate} (gate {report.gate}) vs {BASELINE}, {report.phase} WIS, "
        f"{report.eval_window}, seeds {list(report.seeds)}",
        f"  dates {report.n_dates} ({report.first_date} .. {report.last_date}); "
        f"WIS {report.baseline_wis:.2f} -> {report.candidate_wis:.2f}",
        f"  mean delta per date {report.dm.mean_delta:+.3f}; by horizon {horizons}",
        f"  Diebold-Mariano (HLN, lag {report.dm.max_lag}, {report.dm.variance_kernel}): "
        f"stat {report.dm.statistic:+.3f}, df {report.dm.df}, p {report.dm.p_value:.4f}",
        f"  block bootstrap {report.bootstrap.level:.0%} CI (block {report.bootstrap.block_length}): "
        f"[{report.bootstrap.low:+.3f}, {report.bootstrap.high:+.3f}]",
        f"  verdict at alpha {report.alpha}: {verdict}",
    ])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--candidate", required=True, help="logged config id without the seed suffix")
    parser.add_argument("--gate", choices=GATES, default=GATE_NONE)
    parser.add_argument("--phase", default=DEFAULT_PHASE, help=f"a phase of {PHASE_SCHEME}, or {ALL_PHASES!r}")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)
    report = run_test(
        args.run_dir.resolve(), args.candidate, args.gate, args.phase, args.max_lag, args.block_length, args.alpha
    )
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
