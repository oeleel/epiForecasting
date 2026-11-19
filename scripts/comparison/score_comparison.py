"""Score TimeSeriesScientist (TSci) and our framework on the same forecast rows.

Question A of the comparison study: does each framework improve the model over its own
baseline? For every (location, cutoff, horizon) both frameworks forecast, this collects
four point forecasts and scores them with MAE against the same truth:

    tsci / baseline   TSci's XGBoost at library defaults, raw series, no LLM
    tsci / final      TSci's XGBoost after its Curator, Planner and Forecaster
    ours / baseline   iteration 0 of `agent improve` (default config)
    ours / final      the best iteration of the same run

MAE, not WIS: TSci returns point forecasts only. Our point forecast is the median.
Only rows present for all four are scored, so the two frameworks are never compared on
different locations or dates.

    .venv/bin/python scripts/comparison/score_comparison.py \\
        --tsci-runs outputs/experiments/tsci_comparison/runs \\
        --series outputs/experiments/tsci_comparison/series \\
        --our-runs outputs/agent_runs/<run_id> [<run_id> ...] \\
        --out outputs/experiments/tsci_comparison/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

__all__ = ["KEY_COLUMNS", "common_rows", "load_our_run", "load_tsci_runs", "main", "summarize"]

KEY_COLUMNS = ["location", "cutoff_date", "horizon"]
FRAMEWORK_STAGES = [("tsci", "baseline"), ("tsci", "final"), ("ours", "baseline"), ("ours", "final")]


def load_tsci_runs(runs_dir: Path, series_dir: Path) -> pd.DataFrame:
    """Long frame of TSci forecasts: one row per (stage, location, cutoff, horizon)."""
    manifest = json.loads((series_dir / "manifest.json").read_text())
    rows = []
    for entry in manifest["files"]:
        run_dir = runs_dir / Path(entry["file"]).stem
        if not (run_dir / "results.json").exists():
            continue
        baseline = json.loads((run_dir / "baseline.json").read_text())
        results = json.loads((run_dir / "results.json").read_text())
        final = results["aggregated_results"]["individual_predictions"][baseline["model"]]
        if len(final) != entry["horizon"] or len(baseline["predictions"]) != entry["horizon"]:
            raise ValueError(
                f"{run_dir.name}: expected {entry['horizon']} forecasts, got "
                f"{len(final)} final and {len(baseline['predictions'])} baseline"
            )
        for step in range(entry["horizon"]):
            key = {"location": entry["location"], "cutoff_date": entry["cutoff_date"], "horizon": step + 1}
            actual = baseline["actuals"][step]
            rows.append({**key, "framework": "tsci", "stage": "baseline",
                         "predicted": baseline["predictions"][step], "actual": actual})
            rows.append({**key, "framework": "tsci", "stage": "final",
                         "predicted": final[step], "actual": actual})
    return pd.DataFrame(rows)


def load_our_run(run_dir: Path, truth: pd.DataFrame) -> pd.DataFrame:
    """Long frame of our forecasts for one `agent improve` run: iteration 0 and the best one."""
    frames = []
    for stage, file_name in (("baseline", "iter_0.csv"), ("final", "best_forecast.csv")):
        forecast = pd.read_csv(run_dir / file_name, dtype={"location": str})
        frame = pd.DataFrame(
            {
                "location": forecast["location"],
                "cutoff_date": forecast["cutoff_date"].astype(str).str[:10],
                "horizon": forecast["horizon"].astype(int),
                "target_end_date": forecast["target_end_date"].astype(str).str[:10],
                "framework": "ours",
                "stage": stage,
                "predicted": forecast["predicted"].astype(float),
            }
        )
        frames.append(frame)
    merged = pd.concat(frames).merge(
        truth, left_on=["location", "target_end_date"], right_on=["location", "date"], how="inner"
    )
    return merged.drop(columns=["target_end_date", "date"])


def common_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Keep the (location, cutoff, horizon) keys that every framework and stage forecast."""
    counts = forecasts.groupby(KEY_COLUMNS)[["framework", "stage"]].apply(
        lambda group: len(set(zip(group["framework"], group["stage"], strict=True)))
    )
    complete = counts[counts == len(FRAMEWORK_STAGES)].index
    keyed = forecasts.set_index(KEY_COLUMNS)
    return keyed[keyed.index.isin(complete)].reset_index()


def summarize(forecasts: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """MAE of each framework's baseline and final forecast, and the change between them.

    ``change_pct`` is negative when the framework improved on its own baseline.
    """
    group_columns = (by or []) + ["framework"]
    scored = forecasts.assign(abs_error=(forecasts["predicted"] - forecasts["actual"]).abs())
    table = scored.pivot_table(index=group_columns, columns="stage", values="abs_error", aggfunc="mean")
    table["n"] = scored[scored["stage"] == "baseline"].groupby(group_columns).size()
    table["change_pct"] = 100.0 * (table["final"] - table["baseline"]) / table["baseline"]
    return table.rename(columns={"baseline": "baseline_mae", "final": "final_mae"})[
        ["n", "baseline_mae", "final_mae", "change_pct"]
    ].reset_index()


def _markdown(table: pd.DataFrame) -> str:
    formatted = table.copy()
    for column in ("baseline_mae", "final_mae"):
        formatted[column] = formatted[column].map("{:.1f}".format)
    formatted["change_pct"] = formatted["change_pct"].map("{:+.1f}%".format)
    header = "| " + " | ".join(formatted.columns) + " |"
    rule = "|" + "---|" * len(formatted.columns)
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in formatted.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tsci-runs", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--our-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/raw/flusight_hospital_admissions.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    truth = pd.read_csv(args.data, dtype={"location": str})[["location", "date", "value"]]
    truth = truth.rename(columns={"value": "actual"})
    ours = pd.concat([load_our_run(run_dir, truth) for run_dir in args.our_runs])
    forecasts = pd.concat([load_tsci_runs(args.tsci_runs, args.series), ours], ignore_index=True)
    shared = common_rows(forecasts)
    if shared.empty:
        raise SystemExit("no (location, cutoff, horizon) row is forecast by both frameworks")

    sections = [
        "# TSci vs our framework - XGBoost on the flu data",
        f"Rows scored: {len(shared) // len(FRAMEWORK_STAGES)} (location, cutoff, horizon) keys "
        f"forecast by both frameworks; {shared['location'].nunique()} locations, "
        f"cutoffs {', '.join(sorted(shared['cutoff_date'].unique()))}. Metric: MAE of the point forecast "
        "(weekly hospital admissions). `change_pct` below zero means the framework improved on its own baseline.",
        "## Overall", _markdown(summarize(shared)),
        "## By cutoff", _markdown(summarize(shared, by=["cutoff_date"])),
        "## By horizon", _markdown(summarize(shared, by=["horizon"])),
        "## Our framework on every location it forecast (not comparable with TSci)",
        _markdown(summarize(ours)),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(sections) + "\n")
    shared.to_csv(args.out.with_suffix(".rows.csv"), index=False)
    print(args.out.read_text())


if __name__ == "__main__":
    main()
