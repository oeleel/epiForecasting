"""Score each framework's chosen configuration on a cutoff it was not chosen on.

The first-level comparison has an asymmetry: our loop keeps the iteration that scores
best on the same four weeks the comparison then scores, while TimeSeriesScientist (TSci)
selects on a split inside the history. This script removes it. For a run at cutoff ``c``
it takes the configuration the framework chose there and forecasts from the NEXT pinned
cutoff, ``c + 4 weeks``, next to the framework's untuned baseline from that same cutoff.
Nothing is selected on the weeks that are scored.

Output mimics the layouts ``score_comparison.py`` already reads, so scoring is unchanged:

    ours  ->  <out>/ours/<run_id>/{iter_0.csv, best_forecast.csv}
    tsci  ->  <out>/tsci_runs/<location>_<next_cutoff>/{baseline.json, results.json}

    # our env
    .venv/bin/python scripts/comparison/heldout_rescore.py ours --runs-dir outputs/agent_runs \\
        --run-ids <id> ... --out <out>
    # TSci's env (it calls TSci's own model function)
    <tsci>/.venv/bin/python scripts/comparison/heldout_rescore.py tsci --tsci-root <tsci> \\
        --tsci-runs <runs> --series <series at the next cutoffs> --out <out>
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CUTOFF_STRIDE_WEEKS = 4
DATE_FORMAT = "%Y-%m-%d"


def next_cutoff(cutoff_date: str, stride_weeks: int = CUTOFF_STRIDE_WEEKS) -> str:
    """The pinned cutoff that follows ``cutoff_date``."""
    cutoff = dt.datetime.strptime(cutoff_date, DATE_FORMAT)
    return (cutoff + dt.timedelta(weeks=stride_weeks)).strftime(DATE_FORMAT)


def best_iteration_config(runs_dir: Path, run_id: str) -> tuple[dict, str, int]:
    """Return (config of the best iteration, the run's cutoff, the best iteration number)."""
    report = json.loads((runs_dir / run_id / "report.json").read_text())
    best = int(report["best_iteration"])
    with sqlite3.connect(runs_dir / "runs.db") as connection:
        row = connection.execute(
            "SELECT config_json FROM iterations WHERE run_id = ? AND iteration = ?", (run_id, best)
        ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"run {run_id}: no stored config for best iteration {best}")
    return json.loads(row[0]), report["cutoff_date"], best


def rescore_ours(runs_dir: Path, run_ids: list[str], out: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from src.config import get_default_config
    from src.pipeline import run_pipeline

    for run_id in run_ids:
        chosen, cutoff_date, best = best_iteration_config(runs_dir, run_id)
        held_out = next_cutoff(cutoff_date)
        run_out = out / "ours" / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        baseline = get_default_config()
        baseline["data"]["cutoff_date"] = held_out
        tuned = copy.deepcopy(chosen)
        tuned["data"]["cutoff_date"] = held_out

        run_pipeline(config=baseline, output_path=run_out / "iter_0.csv")
        run_pipeline(config=tuned, output_path=run_out / "best_forecast.csv")
        record = {"run_id": run_id, "chosen_at": cutoff_date, "scored_at": held_out, "best_iteration": best}
        (run_out / "heldout.json").write_text(json.dumps(record, indent=2))
        print(f"ours {run_id}: config of iteration {best} chosen at {cutoff_date}, forecast from {held_out}", flush=True)


def rescore_tsci(tsci_root: Path, tsci_runs: Path, series_dir: Path, out: Path) -> None:
    import pandas as pd

    sys.path.insert(0, str(tsci_root.resolve() / "time_series_agent"))
    from utils.model_library import get_model_function

    manifest = json.loads((series_dir / "manifest.json").read_text())
    by_key = {(entry["location"], entry["cutoff_date"]): entry for entry in manifest["files"]}
    for run_dir in sorted(path for path in tsci_runs.iterdir() if (path / "results.json").exists()):
        location, cutoff_date = run_dir.name.split("_")
        held_out = next_cutoff(cutoff_date)
        entry = by_key.get((location, held_out))
        if entry is None:
            raise ValueError(f"{run_dir.name}: no series for {location} at the next cutoff {held_out} in {series_dir}")
        model = json.loads((run_dir / "baseline.json").read_text())["model"]
        results = json.loads((run_dir / "results.json").read_text())
        chosen = results["all_results"][0]["best_hyperparameters"][model]

        values = pd.read_csv(series_dir / entry["file"])["OT"].to_numpy(dtype=float)
        window, truth = values[: entry["input_length"]], values[entry["input_length"] :]
        predict = get_model_function(model)
        default_forecast = predict({"value": window}, {}, entry["horizon"])
        tuned_forecast = predict({"value": window}, dict(chosen), entry["horizon"])

        target = out / "tsci_runs" / f"{location}_{held_out}"
        target.mkdir(parents=True, exist_ok=True)
        baseline = {"model": model, "hyperparameters": {}, "predictions": [float(v) for v in default_forecast],
                    "actuals": [float(v) for v in truth]}
        (target / "baseline.json").write_text(json.dumps(baseline, indent=2))
        tuned = {"chosen_at": cutoff_date, "scored_at": held_out, "hyperparameters": chosen,
                 "aggregated_results": {"individual_predictions": {model: [float(v) for v in tuned_forecast]}}}
        (target / "results.json").write_text(json.dumps(tuned, indent=2))
        print(f"tsci {run_dir.name}: hyperparameters {chosen} forecast from {held_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    ours = commands.add_parser("ours")
    ours.add_argument("--runs-dir", type=Path, required=True, help="directory with runs.db and the run folders")
    ours.add_argument("--run-ids", nargs="+", required=True)
    ours.add_argument("--out", type=Path, required=True)
    tsci = commands.add_parser("tsci")
    tsci.add_argument("--tsci-root", type=Path, required=True)
    tsci.add_argument("--tsci-runs", type=Path, required=True)
    tsci.add_argument("--series", type=Path, required=True, help="series cut at the NEXT cutoffs")
    tsci.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "ours":
        rescore_ours(args.runs_dir, args.run_ids, args.out)
    else:
        rescore_tsci(args.tsci_root, args.tsci_runs, args.series, args.out)


if __name__ == "__main__":
    main()
