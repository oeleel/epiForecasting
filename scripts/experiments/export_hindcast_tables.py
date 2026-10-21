"""Export a hindcast run as human-readable tables for the repo (A. Adiga, 2026-10-21).

Reads a peak-rectification run directory (`manifest.json`, `forecasts/`, `summary.md`) and
writes, under `--out`:

- `forecasts/<config>.csv`: one row per location x forecast date x horizon with the five
  quantiles, the observed value, the target week's phase and that row's WIS. Every
  location is kept, US included, so the files also feed `src.forecast_panels`.
- `wis_by_location.csv` / `.md`: WIS per state, pooled over horizons, overall and per phase;
  the markdown shows the peak phase as a change from baseline, best gains first.
- `wis_by_week.csv` / `.md`: WIS per forecast date across the season, as a change from
  baseline.
- `direction_by_horizon.csv` / `.md` and `direction_by_phase.csv` / `.md`: direction
  agreement, the share of scored rows whose median points the same way as the truth;
  `direction_peak_confusion.csv` (shown in `direction_by_phase.md`) breaks the peak phase
  down by actual x predicted direction.
- `summary.md` and `manifest.json` copied from the run (WIS by horizon, provenance).

Scoring matches the harness exactly: rows are merged and scored by
`PhaseEvaluator.merge_forecasts_actuals` against the run's pinned data cache and US is
excluded. As in `FluForecastAdapter.compute_metrics`, overall WIS covers every target week,
off-season included (the last forecast dates reach into June); per-phase WIS groups rows by
their target week's phase, off_season being one of them. `--check` (on by default) recomputes
each config's pooled overall and peak WIS from these rows and fails unless they match the
run's `log.jsonl`, so the tables cannot drift from the logged results.

Direction (A. Adiga, 2026-10-21: CDC reads a forecast for "going up or coming down", and a
model can score well on WIS while flat-lining). Each row is anchored on the last observed
value at its `cutoff_date` in the same data cache. The predicted change is `q50 - anchor`,
the actual change `observed - anchor`; each is `up`, `down` or `flat`. A change is flat when
its magnitude is below `max(DIRECTION_FLAT_RELATIVE * anchor, DIRECTION_FLAT_MIN_ABSOLUTE)`.
ASSUMPTION: the repo documents no trend categories, so the band is +-10% of the anchor with
a floor of 10 admissions, one band for every horizon. The floor keeps small states (anchor
near 0, where a 10% band is a fraction of one admission) from turning every wobble into a
trend, and the band is a multiplication, never a division, so a zero anchor is safe.
FluSight's own rate-trend categories (per-100k thresholds that widen with horizon) are not
reproduced here. Classification runs on the exported (rounded) values, so the CSV columns
can be recomputed from the CSV itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.phase_evaluator import PhaseEvaluator  # noqa: E402

__all__ = [
    "classify_direction", "direction_agreement", "direction_confusion", "export", "score_rows",
    "wis_by_location", "wis_by_week",
]

DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / "peak_rectification_weekly"
DEFAULT_OUT_DIR = PROJECT_ROOT / "documentation" / "experiments" / "weekly_hindcast"
DEFAULT_SEED = 42
DEFAULT_FOCUS_CONFIGS = ("lambda_3", "lambda_calendar_3", "oversample_2", "oversample_5", "window_52")
BASELINE = "baseline"
EXCLUDED_FROM_SCORING = ("US",)
FOCUS_PHASE = "peak"
LOG_MATCH_TOLERANCE = 0.005  # log.jsonl rounds WIS to 2 decimals
VALUE_DECIMALS = 1
WIS_DECIMALS = 2
QUANTILE_COLUMNS = ("predicted_q05", "predicted_q25", "predicted_q50", "predicted_q75", "predicted_q95")
ROW_COLUMNS = (
    "location", "location_name", "cutoff_date", "target_end_date", "horizon", "phase",
    "observed", "q05", "q25", "q50", "q75", "q95", "wis",
    "anchor", "predicted_direction", "actual_direction",
)
DIRECTION_FLAT_RELATIVE = 0.10  # |change| under 10% of the anchor is flat (assumption, see docstring)
DIRECTION_FLAT_MIN_ABSOLUTE = 10.0  # admissions; floor on the flat band for anchors near 0
DIRECTIONS = ("up", "flat", "down")
ALL_GROUPS = "all"
AGREEMENT_DECIMALS = 4
PHASE_ORDER = ("growth", "peak", "decline", "plateau", "off_season")


def score_rows(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per forecast with observed value, target phase and WIS (all locations)."""
    merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)
    rows = merged.rename(columns={"actual": "observed", **{c: c.removeprefix("predicted_") for c in QUANTILE_COLUMNS}})
    rows["target_end_date"] = pd.to_datetime(rows["forecast_date"]).dt.date.astype(str)
    rows["cutoff_date"] = rows["cutoff_date"].astype(str)
    rows = rows.round({c: VALUE_DECIMALS for c in ("observed", "q05", "q25", "q50", "q75", "q95")})
    rows = _with_directions(rows, actuals)
    return rows[list(ROW_COLUMNS)].sort_values(["location", "cutoff_date", "horizon"]).reset_index(drop=True)


def classify_direction(change: pd.Series, anchor: pd.Series) -> pd.Series:
    """`up` / `down` / `flat` for a change from `anchor`; flat inside the band (see module docstring)."""
    if change.isna().any() or anchor.isna().any():
        raise ValueError(f"direction needs complete values; {int(change.isna().sum())} changes and "
                         f"{int(anchor.isna().sum())} anchors are NaN")
    band = np.maximum(DIRECTION_FLAT_RELATIVE * anchor.abs(), DIRECTION_FLAT_MIN_ABSOLUTE)
    labels = np.select([change >= band, change <= -band], ["up", "down"], default="flat")
    return pd.Series(labels, index=change.index)


def _with_directions(rows: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    anchors = actuals.assign(cutoff_date=pd.to_datetime(actuals["date"]).dt.date.astype(str))
    anchors = anchors[["location", "cutoff_date", "value"]].rename(columns={"value": "anchor"})
    rows = rows.merge(anchors, on=["location", "cutoff_date"], how="left", validate="many_to_one")
    missing = rows[rows["anchor"].isna()][["location", "cutoff_date"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(f"no observed value at the cutoff for {len(missing)} location x cutoff pairs, "
                         f"e.g. {missing.head(3).to_dict('records')}")
    return rows.assign(
        predicted_direction=classify_direction(rows["q50"] - rows["anchor"], rows["anchor"]),
        actual_direction=classify_direction(rows["observed"] - rows["anchor"], rows["anchor"]),
    )


def scored(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows the harness scores: every state row (US excluded), all target weeks."""
    return rows[~rows["location"].isin(EXCLUDED_FROM_SCORING)]


def wis_by_location(rows_by_config: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for config, rows in rows_by_config.items():
        kept = scored(rows)
        overall = kept.groupby(["location", "location_name"])["wis"].agg(["mean", "size"]).assign(phase="all")
        by_phase = kept.groupby(["location", "location_name", "phase"])["wis"].agg(["mean", "size"])
        frame = pd.concat([overall.reset_index(), by_phase.reset_index()])
        frames.append(frame.assign(config=config))
    table = pd.concat(frames).rename(columns={"mean": "wis", "size": "n"})
    return table[["config", "location", "location_name", "phase", "wis", "n"]].round({"wis": WIS_DECIMALS})


def wis_by_week(rows_by_config: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for config, rows in rows_by_config.items():
        kept = scored(rows)
        week = kept.groupby("cutoff_date")["wis"].agg(["mean", "size"]).reset_index()
        frames.append(week.assign(config=config))
    table = pd.concat(frames).rename(columns={"mean": "wis", "size": "n"})
    return table[["config", "cutoff_date", "wis", "n"]].round({"wis": WIS_DECIMALS})


def direction_agreement(rows_by_config: dict[str, pd.DataFrame], by: str) -> pd.DataFrame:
    """Share of scored rows whose predicted direction equals the actual one, per config x `by`.

    One extra row per config with `by` = "all" pools every group.
    """
    frames = []
    for config, rows in rows_by_config.items():
        kept = scored(rows).assign(hit=lambda r: r["predicted_direction"] == r["actual_direction"])
        grouped = kept.groupby(by)["hit"].agg(["mean", "size"]).reset_index()
        grouped[by] = grouped[by].astype(str)
        pooled = pd.DataFrame({by: [ALL_GROUPS], "mean": [kept["hit"].mean()], "size": [len(kept)]})
        frames.append(pd.concat([grouped, pooled]).assign(config=config))
    table = pd.concat(frames).rename(columns={"mean": "agreement", "size": "n"})
    return table[["config", by, "agreement", "n"]].round({"agreement": AGREEMENT_DECIMALS}).reset_index(drop=True)


def direction_confusion(rows_by_config: dict[str, pd.DataFrame], phase: str = FOCUS_PHASE) -> pd.DataFrame:
    """Scored rows of one phase counted by actual x predicted direction, with the share of each actual class."""
    frames = []
    for config, rows in rows_by_config.items():
        kept = scored(rows)
        kept = kept[kept["phase"] == phase]
        counts = pd.crosstab(kept["actual_direction"], kept["predicted_direction"])
        counts = counts.reindex(index=list(DIRECTIONS), columns=list(DIRECTIONS), fill_value=0)
        long = counts.stack().rename("n").reset_index()
        totals = long.groupby("actual_direction")["n"].transform("sum")
        frames.append(long.assign(config=config, phase=phase, share_of_actual=(long["n"] / totals.where(totals > 0))))
    table = pd.concat(frames)
    columns = ["config", "phase", "actual_direction", "predicted_direction", "n", "share_of_actual"]
    return table[columns].round({"share_of_actual": AGREEMENT_DECIMALS}).reset_index(drop=True)


def export(
    run_dir: Path, out_dir: Path, seed: int = DEFAULT_SEED,
    focus: Sequence[str] = DEFAULT_FOCUS_CONFIGS, check: bool = True,
) -> list[Path]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    actuals = _pinned_actuals(manifest)
    suffix = f"__seed{seed}.csv"
    configs = sorted((p.name[: -len(suffix)] for p in (run_dir / "forecasts").glob(f"*{suffix}")),
                     key=lambda c: (c != BASELINE, c))
    if BASELINE not in configs:
        raise ValueError(f"no {BASELINE}{suffix} in {run_dir / 'forecasts'}; deltas need it")
    missing = [c for c in focus if c not in configs]
    if missing:
        raise ValueError(f"focus configs {missing} not in this run; available: {configs}")

    rows_by_config = {
        config: score_rows(pd.read_csv(run_dir / "forecasts" / f"{config}{suffix}", dtype={"location": str}), actuals)
        for config in configs
    }
    if check:
        _check_against_log(run_dir, rows_by_config, seed)

    (out_dir / "forecasts").mkdir(parents=True, exist_ok=True)
    written = []
    for config, rows in rows_by_config.items():
        path = out_dir / "forecasts" / f"{config}.csv"
        rows.round({"wis": WIS_DECIMALS}).to_csv(path, index=False)
        written.append(path)
    by_location = wis_by_location(rows_by_config)
    by_week = wis_by_week(rows_by_config)
    by_horizon = direction_agreement(rows_by_config, "horizon")
    by_phase = direction_agreement(rows_by_config, "phase")
    confusion = direction_confusion(rows_by_config)
    for name, table in (
        ("wis_by_location", by_location), ("wis_by_week", by_week), ("direction_by_horizon", by_horizon),
        ("direction_by_phase", by_phase), ("direction_peak_confusion", confusion),
    ):
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written.append(path)
    shown = [BASELINE, *focus]
    written.append(_write(out_dir / "wis_by_location.md", _location_markdown(by_location, shown, manifest)))
    written.append(_write(out_dir / "wis_by_week.md", _week_markdown(by_week, rows_by_config[BASELINE], shown, manifest)))
    flat_share = (scored(rows_by_config[BASELINE])["actual_direction"] == "flat").mean()
    written.append(_write(out_dir / "direction_by_horizon.md",
                          _direction_markdown(by_horizon, "horizon", ["1", "2", "3", "4"], shown, flat_share, manifest)))
    phase_md = _direction_markdown(by_phase, "phase", list(PHASE_ORDER), shown, flat_share, manifest)
    written.append(_write(out_dir / "direction_by_phase.md", phase_md + _confusion_markdown(confusion, shown)))
    for name in ("summary.md", "manifest.json"):
        shutil.copyfile(run_dir / name, out_dir / name)
        written.append(out_dir / name)
    return written


def _pinned_actuals(manifest: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / manifest["data_cache"]["path"]
    pinned = manifest["data_cache"]["sha256"]
    now = hashlib.sha256(path.read_bytes()).hexdigest()
    if now != pinned:
        raise ValueError(f"{path} has sha256 {now} but the run pinned {pinned}; restore the pinned cache")
    return pd.read_csv(path, dtype={"location": str})


def _check_against_log(run_dir: Path, rows_by_config: dict[str, pd.DataFrame], seed: int) -> None:
    logged = {json.loads(line)["config_id"]: json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()}
    for config, rows in rows_by_config.items():
        metrics = logged[f"{config}__seed{seed}"]["metrics"]
        kept = scored(rows)
        pairs = {
            "overall": (kept["wis"].mean(), metrics["overall"]["wis"]),
            FOCUS_PHASE: (kept[kept["phase"] == FOCUS_PHASE]["wis"].mean(), metrics["by_phase"][FOCUS_PHASE]["wis"]),
        }
        for label, (mine, theirs) in pairs.items():
            if abs(mine - theirs) > LOG_MATCH_TOLERANCE:
                raise ValueError(f"{config} {label} WIS {mine:.6f} from the rows != logged {theirs:.6f}")


def _location_markdown(table: pd.DataFrame, shown: Sequence[str], manifest: dict) -> str:
    peak = table[table["phase"] == FOCUS_PHASE].pivot(index=["location", "location_name"], columns="config", values="wis")
    deltas = peak[list(shown[1:])].sub(peak[BASELINE], axis=0)
    out = pd.DataFrame({"baseline WIS": peak[BASELINE]}).join(deltas.add_prefix("Δ "))
    out = out.assign(_best=deltas.min(axis=1)).sort_values("_best").drop(columns="_best")
    gains = (deltas < 0).sum()
    lines = [
        "# Peak-phase WIS by location",
        "",
        _provenance(manifest),
        "",
        f"Peak-phase WIS (target week Jan 1 - end of Feb) per location (50 states, DC, Puerto Rico), "
        f"pooled over horizons 1-4. Δ is the "
        f"config minus baseline: negative means the config beat baseline there. Sorted by the "
        f"largest gain of any config shown. US is not scored (see README).",
        "",
        "Locations where each config beats baseline: "
        + ", ".join(f"`{c}` {int(gains[c])}/{len(out)}" for c in shown[1:]) + ".",
        "",
        "| location | " + " | ".join(out.columns) + " |",
        "|---|" + "---:|" * len(out.columns),
    ]
    for (_code, name), values in out.iterrows():
        cells = [f"{values.iloc[0]:.1f}", *(f"{v:+.1f}" for v in values.iloc[1:])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _week_markdown(table: pd.DataFrame, baseline_rows: pd.DataFrame, shown: Sequence[str], manifest: dict) -> str:
    wide = table.pivot(index="cutoff_date", columns="config", values="wis")
    deltas = wide[list(shown[1:])].sub(wide[BASELINE], axis=0)
    h1_phase = baseline_rows[baseline_rows["horizon"] == 1].groupby("cutoff_date")["phase"].first()
    lines = [
        "# WIS by forecast date",
        "",
        _provenance(manifest),
        "",
        "WIS per forecast date (the last observed week), mean over locations and horizons 1-4. "
        "`phase` is the phase of the 1-week-ahead target. Δ is config minus "
        "baseline; negative is better.",
        "",
        "| forecast date | phase | baseline WIS | " + " | ".join(f"Δ {c}" for c in shown[1:]) + " |",
        "|---|---|---:|" + "---:|" * (len(shown) - 1),
    ]
    for date, base in wide[BASELINE].items():
        cells = [f"{deltas.loc[date, c]:+.1f}" for c in shown[1:]]
        lines.append(f"| {date} | {h1_phase.get(date, '')} | {base:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _direction_markdown(
    table: pd.DataFrame, by: str, groups: Sequence[str], shown: Sequence[str], flat_share: float, manifest: dict,
) -> str:
    wide = table.pivot(index="config", columns=by, values="agreement")
    unknown = [g for g in groups if g not in wide.columns]
    if unknown:
        raise ValueError(f"direction groups {unknown} not in the {by} table; present: {sorted(wide.columns)}")
    columns = [*groups, ALL_GROUPS]
    labels = [f"h{g}" if by == "horizon" else g for g in groups] + [ALL_GROUPS]
    lines = [
        f"# Direction agreement by {by}",
        "",
        _provenance(manifest),
        "",
        "Share of scored rows (US excluded, every target week) where the median points the same way "
        "as what happened: each of `q50 - anchor` and `observed - anchor` is up, down or flat, the "
        "anchor being the last observed value at the forecast date. Flat means a change smaller than "
        f"{DIRECTION_FLAT_RELATIVE:.0%} of the anchor or {DIRECTION_FLAT_MIN_ABSOLUTE:.0f} admissions, "
        "whichever is larger (an assumption; the repo defines no trend categories). Higher is better; "
        "Δ is config minus baseline in percentage points. `phase` is the target week's phase.",
        "",
        f"Reference: a forecast that always says flat (persistence) would agree on {flat_share:.1%} of "
        "rows, the share whose truth is flat.",
        "",
        "| config | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for config in shown:
        if config == BASELINE:
            cells = [f"{wide.loc[config, c]:.1%}" for c in columns]
        else:
            cells = [f"{wide.loc[config, c]:.1%} ({100 * (wide.loc[config, c] - wide.loc[BASELINE, c]):+.1f} pp)"
                     for c in columns]
        lines.append(f"| `{config}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _confusion_markdown(confusion: pd.DataFrame, shown: Sequence[str]) -> str:
    share = confusion.pivot(index=["config", "actual_direction"], columns="predicted_direction", values="share_of_actual")
    n = confusion.groupby(["config", "actual_direction"])["n"].sum()
    lines = [
        "",
        f"## {FOCUS_PHASE.capitalize()}-phase confusion",
        "",
        f"{FOCUS_PHASE.capitalize()}-phase rows by what actually happened, and what share of them the "
        "median called up, flat or down. Actual down with median up is the overshoot after the turn; "
        "actual up with median flat or down is a late call on the rise. Every config is in "
        "`direction_peak_confusion.csv`.",
        "",
        "| config | actual | n | median up | median flat | median down |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in shown:
        for actual in DIRECTIONS:
            cells = [f"{share.loc[(config, actual), p]:.1%}" if n[(config, actual)] else "n/a" for p in DIRECTIONS]
            lines.append(f"| `{config}` | {actual} | {int(n[(config, actual)])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _provenance(manifest: dict) -> str:
    return (
        f"Generated by `scripts/experiments/export_hindcast_tables.py` from a {manifest['n_cutoffs']}-date "
        f"hindcast (stride {manifest['split']['stride_weeks']} wk, seeds {manifest['seeds']}, phase scheme "
        f"`{manifest['phase_scheme']}`, git {manifest['git_sha'][:7]}, data sha256 "
        f"{manifest['data_cache']['sha256'][:12]}). Do not edit by hand."
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--focus", nargs="+", default=list(DEFAULT_FOCUS_CONFIGS),
                        help="configs shown in the markdown tables (the CSVs hold every config)")
    parser.add_argument("--no-check", action="store_true", help="skip the match against log.jsonl")
    args = parser.parse_args(argv)
    written = export(args.run_dir, args.out, args.seed, args.focus, check=not args.no_check)
    total = sum(p.stat().st_size for p in written)
    print(f"wrote {len(written)} files ({total / 1e6:.1f} MB) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
