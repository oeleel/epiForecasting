"""Gated peak emphasis: compose new forecast sets from existing hindcast runs and score them.

No training. Every composed forecast set takes each row (location x forecast date x
horizon) either from an emphasis config (oversample_k, lambda_calendar_k) or from
baseline, decided by a gate that reads only what is known at the forecast date:

- `horizon`: emphasis for horizons 1-2, baseline for 3-4 (the round-2 tables show
  emphasis helping h1-h2 and hurting h3-h4).
- `calendar`: emphasis when the forecast date (`cutoff_date`) is in the growth phase of
  `adiga_2026_10` (Nov 1 - Dec 31), baseline otherwise.
- `curve`: emphasis while the location's observed series is still rising at the forecast
  date: value(cutoff) > value(cutoff - K weeks) * (1 + r), baseline otherwise.
- `horizon+curve`: emphasis only where both the horizon and the curve gate say so.

Invariants:
- The two source runs must cover exactly the same (location, cutoff_date, horizon) keys;
  composition refuses otherwise, so no row is silently dropped or duplicated.
- The curve gate looks up observed values at dates <= cutoff_date only. It reads the
  run's pinned data cache (a 2026-10-15 vintage), not the vintage that was live on each
  forecast date. The models themselves were fitted on that same cache, so the gate is as
  "real-time" as the hindcast, no more: reporting backfill is ignored by both.
- A location-week whose current or K-weeks-earlier value is missing is not rising
  (baseline). The count of such pairs is reported.
- Scoring is the harness's: `PhaseEvaluator.merge_forecasts_actuals` against the pinned
  cache (sha256 checked against the run manifest), US excluded, overall = every target
  week, phase = target week's phase. Before any gate is scored, the baseline and every
  emphasis config are re-scored and checked against the run's `log.jsonl`, so the
  scorer is proven to reproduce the logged numbers.

Exactness of the `horizon` gate: the per-horizon models are trained independently
(separate XGBoost objects, per-horizon sample weights and oversampling, identical seed),
so the h1-h2 models of an emphasis run are the models an "emphasis only on h1-h2" run
would fit. The one cross-horizon coupling is the floor constraint at forecast time
(`QuantileDirectForecastEnsemble.generate_forecasts`): horizon h is floored at
0.3 * (1 - 0.05 (h-1)) times horizon h-1's quantiles. `floor_counterfactual` rebuilds
what that single run would have emitted (bracketed where a baseline value sat on its own
floor, so its raw value is unknown), and the report gives the WIS gap between the plain
composition and both ends of that bracket. The calendar and curve gates switch whole
(location, forecast date) forecasts, so they are exact selections with no coupling.

All results are in-sample: the gates were designed after looking at this same season.

Usage:
    .venv/bin/python scripts/experiments/gated_emphasis.py
    .venv/bin/python scripts/experiments/gated_emphasis.py --emphasis oversample_2 --lookback-weeks 2 --rise-threshold 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.phase_evaluator import PhaseEvaluator, assign_phases, phase_order  # noqa: E402

__all__ = [
    "GATES",
    "calendar_mask",
    "compose",
    "curve_mask",
    "floor_counterfactual",
    "horizon_mask",
    "rising_flags",
    "score_rows",
    "summarize",
]

DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / "peak_rectification_weekly"
DEFAULT_OUT_DIR = PROJECT_ROOT / "documentation" / "experiments" / "weekly_hindcast" / "gated"
SEED = 42
BASELINE = "baseline"
DEFAULT_EMPHASIS = ("oversample_2", "oversample_5", "lambda_calendar_3")
EXCLUDED_FROM_SCORING = ("US",)
PHASE_SCHEME = "adiga_2026_10"
FOCUS_PHASE = "peak"
LOG_MATCH_TOLERANCE = 0.005  # log.jsonl rounds WIS to 2 decimals

KEY_COLUMNS = ("location", "cutoff_date", "horizon")
VALUE_COLUMNS = (
    "predicted", "predicted_q05", "predicted_q25", "predicted_q50", "predicted_q75", "predicted_q95",
)
QUANTILE_COLUMNS = VALUE_COLUMNS[1:]

# Horizon gate: the round-2 summary shows emphasis helping h1-h2 and hurting h3-h4.
EMPHASIS_HORIZONS = frozenset({1, 2})
# Calendar gate: emphasis while the forecast date is in this phase of PHASE_SCHEME.
CALENDAR_EMPHASIS_PHASE = "growth"
# Curve gate defaults, fixed before the sensitivity grid was run (not the grid's best).
DEFAULT_LOOKBACK_WEEKS = 2
DEFAULT_RISE_THRESHOLD = 0.10
SENSITIVITY_LOOKBACKS = (1, 2, 3)
SENSITIVITY_THRESHOLDS = (0.0, 0.10, 0.25)

# Floor constraint as run (manifest "floor"): horizon h is floored at
# FLOOR_PCT * (1 - FLOOR_DECAY * (h - 1)) times horizon h-1's quantiles.
FLOOR_PCT = 0.30
FLOOR_DECAY = 0.05
# CSV round trip keeps ~7 significant digits; a floored value equals its floor to this.
FLOOR_BIND_RTOL = 1e-6
RAW_FILLS = ("zero", "floor")

# Forecast dates shown individually: the surge, the turn and the weeks after it.
TURN_WINDOW = ("2025-11-22", "2026-01-24")


# ---------------------------------------------------------------------------
# Gates: each returns a boolean mask over `keys` (True = take the emphasis row).
# ---------------------------------------------------------------------------


def horizon_mask(keys: pd.DataFrame, horizons: frozenset[int] = EMPHASIS_HORIZONS) -> np.ndarray:
    """True where the row's horizon is one of `horizons`."""
    if not horizons:
        raise ValueError(f"horizons must be non-empty, got {horizons!r}")
    return keys["horizon"].astype(int).isin(horizons).to_numpy()


def calendar_mask(
    keys: pd.DataFrame, phase: str = CALENDAR_EMPHASIS_PHASE, scheme: str = PHASE_SCHEME
) -> np.ndarray:
    """True where the FORECAST date (cutoff_date, not the target date) falls in `phase`."""
    if phase not in phase_order(scheme):
        raise ValueError(f"phase must be one of {phase_order(scheme)} for scheme {scheme!r}, got {phase!r}")
    return assign_phases(keys["cutoff_date"], scheme) == phase


def rising_flags(
    actuals: pd.DataFrame,
    pairs: pd.DataFrame,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    rise_threshold: float = DEFAULT_RISE_THRESHOLD,
) -> pd.DataFrame:
    """Per (location, cutoff_date): is value(cutoff) > value(cutoff - K weeks) * (1 + r)?

    Only observations dated cutoff_date and cutoff_date - K weeks are read, so nothing
    after the forecast date can leak in. A missing value on either date gives False.

    Returns `pairs` (deduplicated) with columns last, prior, rising.
    """
    if isinstance(lookback_weeks, bool) or not isinstance(lookback_weeks, int) or lookback_weeks < 1:
        raise ValueError(f"lookback_weeks must be an int >= 1, got {lookback_weeks!r}")
    if not np.isfinite(rise_threshold) or rise_threshold < 0:
        raise ValueError(f"rise_threshold must be a finite float >= 0, got {rise_threshold!r}")
    observed = actuals[["location", "date", "value"]].copy()
    observed["date"] = pd.to_datetime(observed["date"]).dt.date.astype(str)
    lookup = observed.set_index(["location", "date"])["value"]
    if lookup.index.duplicated().any():
        raise ValueError("actuals have duplicate (location, date) rows; the curve gate needs one value per week")

    out = pairs[["location", "cutoff_date"]].drop_duplicates().reset_index(drop=True)
    prior_dates = [
        (date.fromisoformat(str(c)) - timedelta(weeks=lookback_weeks)).isoformat() for c in out["cutoff_date"]
    ]
    out["last"] = lookup.reindex(list(zip(out["location"], out["cutoff_date"].astype(str)))).to_numpy()
    out["prior"] = lookup.reindex(list(zip(out["location"], prior_dates))).to_numpy()
    known = out["last"].notna() & out["prior"].notna()
    out["rising"] = known & (out["last"] > out["prior"] * (1.0 + rise_threshold))
    return out


def curve_mask(keys: pd.DataFrame, flags: pd.DataFrame) -> np.ndarray:
    """True where the row's (location, cutoff_date) is flagged rising in `flags`."""
    merged = keys[["location", "cutoff_date"]].merge(
        flags[["location", "cutoff_date", "rising"]], on=["location", "cutoff_date"], how="left"
    )
    if merged["rising"].isna().any():
        missing = merged.loc[merged["rising"].isna(), ["location", "cutoff_date"]].head(3).to_dict("records")
        raise ValueError(f"curve flags do not cover every forecast key, e.g. {missing}")
    return merged["rising"].astype(bool).to_numpy()


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    describe: str
    mask: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]  # (keys, curve flags) -> mask


GATES: tuple[Gate, ...] = (
    Gate("horizon", "emphasis for h1-h2, baseline for h3-h4", lambda k, f: horizon_mask(k)),
    Gate("calendar", "emphasis when the forecast date is in growth (Nov 1-Dec 31)", lambda k, f: calendar_mask(k)),
    Gate("curve", "emphasis while the location's series is rising at the forecast date", lambda k, f: curve_mask(k, f)),
    Gate("horizon+curve", "emphasis for h1-h2 while rising", lambda k, f: horizon_mask(k) & curve_mask(k, f)),
)


# ---------------------------------------------------------------------------
# Composition and the floor-coupling check
# ---------------------------------------------------------------------------


def _sorted_by_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def compose(baseline: pd.DataFrame, emphasis: pd.DataFrame, use_emphasis: np.ndarray) -> pd.DataFrame:
    """Row-wise pick: emphasis values where `use_emphasis`, baseline values elsewhere.

    `use_emphasis` is aligned with `baseline` sorted by KEY_COLUMNS (what the gates are
    computed on). Adds a boolean `from_emphasis` column.
    """
    base = _sorted_by_key(baseline)
    emph = _sorted_by_key(emphasis)
    if len(base) != len(emph) or not base[list(KEY_COLUMNS)].equals(emph[list(KEY_COLUMNS)]):
        raise ValueError(
            f"baseline ({len(base)} rows) and emphasis ({len(emph)} rows) do not cover the same "
            f"{KEY_COLUMNS} keys; refusing to compose"
        )
    mask = np.asarray(use_emphasis, dtype=bool)
    if mask.shape != (len(base),):
        raise ValueError(f"use_emphasis has shape {mask.shape}, expected ({len(base)},)")
    out = base.copy()
    for column in VALUE_COLUMNS:
        out[column] = np.where(mask, emph[column].to_numpy(), base[column].to_numpy())
    out["from_emphasis"] = mask
    return out


def floor_counterfactual(composed: pd.DataFrame, baseline: pd.DataFrame, raw_fill: str) -> pd.DataFrame:
    """The composition as ONE run with emphasis only in the gated rows would have emitted it.

    Forecast time floors horizon h's quantiles at ratio(h) * horizon h-1's emitted
    quantiles (`QuantileDirectForecastEnsemble.generate_forecasts`), so a baseline row
    after an emphasis row would have been floored against the emphasis values. Both
    the raw quantiles and the floor are sorted, so the emitted value is the elementwise
    max(sorted raw, floor). Where the baseline value is above its floor the raw value is
    known; where it sits on its floor the raw value is only known to lie in
    [0, floor], and `raw_fill` picks an end: "zero" or "floor". The two fills bracket
    the counterfactual values; the change is propagated horizon by horizon. Emphasis
    rows are left as they are (in every gate here they follow emphasis rows only).
    """
    if raw_fill not in RAW_FILLS:
        raise ValueError(f"raw_fill must be one of {RAW_FILLS}, got {raw_fill!r}")
    comp = _sorted_by_key(composed)
    base = _sorted_by_key(baseline)
    if not comp[list(KEY_COLUMNS)].equals(base[list(KEY_COLUMNS)]):
        raise ValueError("composed and baseline do not cover the same keys")
    n_horizons = int(comp["horizon"].max())
    if len(comp) % n_horizons or not (comp["horizon"].to_numpy().reshape(-1, n_horizons) == np.arange(1, n_horizons + 1)).all():
        raise ValueError(f"every (location, cutoff_date) needs horizons 1..{n_horizons} exactly once")
    shape = (-1, n_horizons, len(QUANTILE_COLUMNS))
    base_v = base[list(QUANTILE_COLUMNS)].to_numpy(float).reshape(shape)
    out = comp[list(QUANTILE_COLUMNS)].to_numpy(float).reshape(shape).copy()
    from_emphasis = comp["from_emphasis"].to_numpy(bool).reshape(-1, n_horizons)
    for i in range(1, n_horizons):  # index i is horizon i + 1
        ratio = FLOOR_PCT * (1 - FLOOR_DECAY * i)
        base_floor = ratio * base_v[:, i - 1]
        new_floor = ratio * out[:, i - 1]
        on_floor = (base_floor > 0) & np.isclose(base_v[:, i], base_floor, rtol=FLOOR_BIND_RTOL, atol=0.0)
        raw = np.where(on_floor, 0.0 if raw_fill == "zero" else base_floor, base_v[:, i])
        rows = ~from_emphasis[:, i]
        out[rows, i] = np.maximum(raw, new_floor)[rows]
    result = comp.copy()
    result[list(QUANTILE_COLUMNS)] = out.reshape(-1, len(QUANTILE_COLUMNS))
    result["predicted"] = result["predicted_q50"]
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_rows(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Harness scoring: excluded locations dropped, then merge_forecasts_actuals (WIS per row)."""
    kept = forecasts[~forecasts["location"].isin(EXCLUDED_FROM_SCORING)]
    if kept.empty:
        raise ValueError(f"no rows left after excluding {EXCLUDED_FROM_SCORING}")
    merged = PhaseEvaluator.merge_forecasts_actuals(kept, actuals, phase_scheme=PHASE_SCHEME)
    merged["cutoff_date"] = merged["cutoff_date"].astype(str)
    return merged


def summarize(scored: pd.DataFrame) -> dict[str, float]:
    """Flat WIS summary: overall, per phase (scored phases), per horizon."""
    out = {"overall": float(scored["wis"].mean())}
    for phase in phase_order(PHASE_SCHEME):
        rows = scored[scored["phase"] == phase]
        out[phase] = float(rows["wis"].mean()) if len(rows) else float("nan")
    for horizon, rows in scored.groupby("horizon"):
        out[f"h{int(horizon)}"] = float(rows["wis"].mean())
    return out


def location_counts(scored: pd.DataFrame, baseline_scored: pd.DataFrame, phase: str | None = None) -> dict[str, int]:
    """Locations whose mean WIS (all target weeks, or `phase` only) is lower / higher / equal vs baseline."""
    def per_location(frame: pd.DataFrame) -> pd.Series:
        rows = frame if phase is None else frame[frame["phase"] == phase]
        return rows.groupby("location")["wis"].mean()

    diff = (per_location(scored) - per_location(baseline_scored)).dropna()
    return {
        "better": int((diff < -1e-9).sum()),
        "worse": int((diff > 1e-9).sum()),
        "equal": int((diff.abs() <= 1e-9).sum()),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_run(run_dir: Path, configs: Sequence[str]) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cache = PROJECT_ROOT / manifest["data_cache"]["path"]
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != manifest["data_cache"]["sha256"]:
        raise ValueError(f"{cache} has sha256 {digest} but the run pinned {manifest['data_cache']['sha256']}")
    if manifest.get("phase_scheme") != PHASE_SCHEME:
        raise ValueError(f"run phase scheme is {manifest.get('phase_scheme')!r}, this script scores {PHASE_SCHEME!r}")
    actuals = pd.read_csv(cache, dtype={"location": str})
    forecasts = {}
    for config in configs:
        path = run_dir / "forecasts" / f"{config}__seed{SEED}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no forecasts for config {config!r} at {path}")
        frame = pd.read_csv(path, dtype={"location": str, "cutoff_date": str})
        frame["location"] = frame["location"].str.zfill(2)
        forecasts[config] = _sorted_by_key(frame)
    logged = {}
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        row = json.loads(line)
        logged[row["base_config_id"]] = row["metrics"]
    return manifest, actuals, forecasts, logged


def check_scorer(scored_by_config: Mapping[str, pd.DataFrame], logged: Mapping[str, dict]) -> list[str]:
    """Raise unless re-scored overall and peak WIS match log.jsonl for every source config."""
    lines = []
    for config, scored in scored_by_config.items():
        mine = summarize(scored)
        theirs = logged[config]
        pairs = (("overall", mine["overall"], theirs["overall"]["wis"]),
                 (FOCUS_PHASE, mine[FOCUS_PHASE], theirs["by_phase"][FOCUS_PHASE]["wis"]))
        for label, value, expected in pairs:
            if abs(value - expected) > LOG_MATCH_TOLERANCE:
                raise ValueError(f"{config} {label} WIS re-scored {value:.4f}, log.jsonl has {expected}")
        lines.append(f"{config}: overall {mine['overall']:.2f}, peak {mine[FOCUS_PHASE]:.2f} (matches log.jsonl)")
    return lines


def _fmt(delta: float) -> str:
    return "n/a" if not np.isfinite(delta) else f"{delta:+.2f}"


def run(
    run_dir: Path, out_dir: Path, emphasis: Sequence[str], lookback_weeks: int, rise_threshold: float,
) -> str:
    _, actuals, forecasts, logged = load_run(run_dir, [BASELINE, *emphasis])
    base = forecasts[BASELINE]
    keys = base[list(KEY_COLUMNS)]
    scored_sources = {config: score_rows(frame, actuals) for config, frame in forecasts.items()}
    check_lines = check_scorer(scored_sources, logged)
    base_scored = scored_sources[BASELINE]
    base_summary = summarize(base_scored)
    base_week = base_scored.groupby("cutoff_date")["wis"].mean()

    default_flags = rising_flags(actuals, keys, lookback_weeks, rise_threshold)
    missing_pairs = int((default_flags["last"].isna() | default_flags["prior"].isna()).sum())
    summary_rows, week_rows, coupling_rows = [], [], []
    for config in emphasis:
        variants = [("ungated", np.ones(len(keys), dtype=bool))]
        variants += [(gate.name, gate.mask(keys, default_flags)) for gate in GATES]
        for gate_name, mask in variants:
            composed = compose(base, forecasts[config], mask)
            scored = score_rows(composed, actuals)
            stats = summarize(scored)
            overall_counts = location_counts(scored, base_scored)
            peak_counts = location_counts(scored, base_scored, FOCUS_PHASE)
            summary_rows.append({
                "emphasis": config, "gate": gate_name,
                "share_rows_emphasis": float(mask.mean()),
                **{f"d_{k}": stats[k] - base_summary[k] for k in base_summary},
                "locs_better_overall": overall_counts["better"], "locs_worse_overall": overall_counts["worse"],
                "locs_better_peak": peak_counts["better"], "locs_worse_peak": peak_counts["worse"],
            })
            week = scored.groupby("cutoff_date")["wis"].mean() - base_week
            week_rows += [{"emphasis": config, "gate": gate_name, "cutoff_date": c, "d_wis": float(v)}
                          for c, v in week.items()]
            if gate_name in ("horizon", "horizon+curve"):
                row = {"emphasis": config, "gate": gate_name}
                for fill in RAW_FILLS:
                    exact = floor_counterfactual(composed, base, fill)
                    exact_scored = score_rows(exact, actuals)
                    changed = ~np.isclose(exact_scored["wis"].to_numpy(), scored["wis"].to_numpy(), rtol=0, atol=1e-9)
                    exact_stats = summarize(exact_scored)
                    row[f"rows_changed_{fill}"] = int(changed.sum())
                    row[f"d_overall_{fill}"] = exact_stats["overall"] - stats["overall"]
                    row[f"d_peak_{fill}"] = exact_stats[FOCUS_PHASE] - stats[FOCUS_PHASE]
                coupling_rows.append(row)

    sensitivity_rows = []
    for k in SENSITIVITY_LOOKBACKS:
        for r in SENSITIVITY_THRESHOLDS:
            flags = rising_flags(actuals, keys, k, r)
            curve = curve_mask(keys, flags)
            for config in emphasis:
                for gate_name, mask in (("curve", curve), ("horizon+curve", curve & horizon_mask(keys))):
                    stats = summarize(score_rows(compose(base, forecasts[config], mask), actuals))
                    sensitivity_rows.append({
                        "emphasis": config, "gate": gate_name, "lookback_weeks": k, "rise_threshold": r,
                        "share_rows_emphasis": float(mask.mean()),
                        "d_overall": stats["overall"] - base_summary["overall"],
                        "d_peak": stats[FOCUS_PHASE] - base_summary[FOCUS_PHASE],
                    })

    rising_share = default_flags.groupby("cutoff_date")["rising"].mean()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    weeks = pd.DataFrame(week_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    coupling = pd.DataFrame(coupling_rows)
    summary.round(3).to_csv(out_dir / "gate_summary.csv", index=False)
    weeks.round(3).to_csv(out_dir / "gate_by_week.csv", index=False)
    sensitivity.round(3).to_csv(out_dir / "curve_sensitivity.csv", index=False)
    coupling.to_csv(out_dir / "floor_coupling.csv", index=False)

    md = _markdown(base_summary, check_lines, summary, weeks, base_week, sensitivity, coupling,
                   rising_share, missing_pairs, lookback_weeks, rise_threshold)
    (out_dir / "tables.md").write_text(md)
    return md


def _markdown(
    base_summary: dict[str, float], check_lines: list[str], summary: pd.DataFrame, weeks: pd.DataFrame,
    base_week: pd.Series, sensitivity: pd.DataFrame, coupling: pd.DataFrame, rising_share: pd.Series,
    missing_pairs: int, lookback_weeks: int, rise_threshold: float,
) -> str:
    phases = list(phase_order(PHASE_SCHEME))
    lines = [
        "# Gated emphasis: generated tables",
        "",
        "Generated by `scripts/experiments/gated_emphasis.py`. Do not edit by hand. "
        "IN-SAMPLE: the gates were designed after looking at this same season.",
        "",
        f"Curve gate (default): rising if value(cutoff) > value(cutoff - {lookback_weeks} wk) x "
        f"(1 + {rise_threshold}). Location-weeks with a missing value (treated as not rising): {missing_pairs}.",
        "",
        "Scorer check:",
        "",
        *[f"- {line}" for line in check_lines],
        "",
        "## Pooled WIS, delta vs baseline (negative is better)",
        "",
        f"Baseline: overall {base_summary['overall']:.2f}, "
        + ", ".join(f"{p} {base_summary[p]:.2f}" for p in phases)
        + ", " + ", ".join(f"h{h} {base_summary[f'h{h}']:.2f}" for h in range(1, 5)) + ".",
        "",
        "| emphasis | gate | rows from emphasis | overall | " + " | ".join(phases)
        + " | h1 | h2 | h3 | h4 | locs better/worse (overall) | locs better/worse (peak) |",
        "|---|---|---:|---:|" + "---:|" * len(phases) + "---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        r = row._asdict()
        lines.append(
            f"| `{r['emphasis']}` | {r['gate']} | {r['share_rows_emphasis']:.0%} | {_fmt(r['d_overall'])} | "
            + " | ".join(_fmt(r[f"d_{p}"]) for p in phases) + " | "
            + " | ".join(_fmt(r[f"d_h{h}"]) for h in range(1, 5))
            + f" | {r['locs_better_overall']}/{r['locs_worse_overall']} | {r['locs_better_peak']}/{r['locs_worse_peak']} |"
        )
    lines += ["", "## WIS by forecast date around the turn, delta vs baseline", ""]
    window = [c for c in base_week.index if TURN_WINDOW[0] <= c <= TURN_WINDOW[1]]
    for config in summary["emphasis"].unique():
        gates = list(summary.loc[summary["emphasis"] == config, "gate"])
        lines += [f"### `{config}`", "",
                  "| forecast date | baseline WIS | share rising | " + " | ".join(gates) + " |",
                  "|---|---:|---:|" + "---:|" * len(gates)]
        sub = weeks[weeks["emphasis"] == config].set_index(["gate", "cutoff_date"])["d_wis"]
        for c in window:
            lines.append(f"| {c} | {base_week[c]:.1f} | {rising_share.get(c, float('nan')):.0%} | "
                         + " | ".join(f"{sub[(g, c)]:+.1f}" for g in gates) + " |")
        lines.append("")
    lines += ["## Curve-gate sensitivity (overall / peak WIS delta vs baseline)", "",
              "| emphasis | gate | K (weeks) | r | rows from emphasis | overall | peak |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in sensitivity.itertuples(index=False):
        lines.append(f"| `{row.emphasis}` | {row.gate} | {row.lookback_weeks} | {row.rise_threshold:.2f} | "
                     f"{row.share_rows_emphasis:.0%} | {_fmt(row.d_overall)} | {_fmt(row.d_peak)} |")
    lines += ["", "## Floor coupling (horizon gates)", "",
              "Single-run counterfactual minus the plain composition, at both ends of the raw-value "
              "bracket (`raw_fill` zero / floor), over every scored (non-US) row.", "",
              "| emphasis | gate | scored rows changed (zero / floor) | overall WIS gap (zero / floor) | "
              "peak WIS gap (zero / floor) |",
              "|---|---|---:|---:|---:|"]
    for row in coupling.itertuples(index=False):
        lines.append(f"| `{row.emphasis}` | {row.gate} | {row.rows_changed_zero} / {row.rows_changed_floor} | "
                     f"{row.d_overall_zero:+.1e} / {row.d_overall_floor:+.1e} | "
                     f"{row.d_peak_zero:+.1e} / {row.d_peak_floor:+.1e} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--emphasis", nargs="+", default=list(DEFAULT_EMPHASIS))
    parser.add_argument("--lookback-weeks", type=int, default=DEFAULT_LOOKBACK_WEEKS)
    parser.add_argument("--rise-threshold", type=float, default=DEFAULT_RISE_THRESHOLD)
    args = parser.parse_args(argv)
    print(run(args.run_dir, args.out, args.emphasis, args.lookback_weeks, args.rise_threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
