"""Plot one location's forecasts from one date, one panel per config (visual inspection).

    .venv/bin/python scripts/plot_forecasts.py --location Virginia --date 2026-01-10
    .venv/bin/python scripts/plot_forecasts.py --location 06 --date 2025-12-13 \\
        --configs baseline lambda_calendar_3 oversample_5 --lookback-weeks 8 --out ca.png
    .venv/bin/python scripts/plot_forecasts.py --list-dates

Reads a hindcast's saved forecasts (default: the tracked weekly export in
documentation/experiments/weekly_hindcast/) and the data cache the run pinned. See
`src/forecast_panels.py` for what the figure shows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # noqa: E402

from src.forecast_panels import (  # noqa: E402
    DEFAULT_LOOKBACK_WEEKS,
    available_cutoffs,
    load_hindcast,
    plot_forecast_panels,
)

# The tracked export, so the script works from a fresh clone; --run-dir also takes a raw
# harness run directory under outputs/.
DEFAULT_RUN_DIR = PROJECT_ROOT / "documentation" / "experiments" / "weekly_hindcast"
DEFAULT_CONFIGS = ("baseline", "lambda_calendar_3", "oversample_2", "oversample_5", "window_52")
DEFAULT_SEED = 42


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--location", help="FIPS code (06, US) or name (California)")
    parser.add_argument("--date", help="forecast date: the last observed week, YYYY-MM-DD")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                        help="config ids in panel order; 'all' for every saved config")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lookback-weeks", type=int, default=DEFAULT_LOOKBACK_WEEKS)
    parser.add_argument("--out", type=Path, default=None, help="default: <location>_<date>.png here")
    parser.add_argument("--list-dates", action="store_true", help="print the forecast dates and exit")
    args = parser.parse_args(argv)

    configs = None if args.configs == ["all"] else args.configs
    forecasts = load_hindcast(args.run_dir, configs, args.seed)
    if args.list_dates:
        print("\n".join(available_cutoffs(forecasts)))
        return 0
    if not args.location or not args.date:
        parser.error("--location and --date are required (or pass --list-dates)")

    observed = _pinned_observed(args.run_dir)
    out = args.out or Path(f"{args.location.replace(' ', '_')}_{args.date}.png")
    try:
        plot_forecast_panels(args.location, args.date, forecasts, observed, args.lookback_weeks, save_path=out)
    except ValueError as exc:  # bad location, date or lookback: a usage error, not a crash
        parser.error(str(exc))
    print(f"wrote {out}")
    return 0


def _pinned_observed(run_dir: Path) -> pd.DataFrame:
    """The data cache the run trained on; refuses a refreshed cache (backfill moves past weeks)."""
    cache = json.loads((run_dir / "manifest.json").read_text())["data_cache"]
    path = PROJECT_ROOT / cache["path"]
    now = hashlib.sha256(path.read_bytes()).hexdigest()
    if now != cache["sha256"]:
        raise ValueError(f"{path} has sha256 {now} but {run_dir} pinned {cache['sha256']}")
    return pd.read_csv(path, dtype={"location": str})


if __name__ == "__main__":
    raise SystemExit(main())
