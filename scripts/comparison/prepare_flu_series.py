"""Cut the CDC flu cache into the single-series files TimeSeriesScientist (TSci) reads.

TSci forecasts one univariate series per run from a CSV with a ``date`` and an ``OT``
column. With one slice it uses the first ``input_length`` rows as its input window and
the next ``horizon`` rows as the test truth, so each file here holds exactly

    TRAIN_START_DATE .. cutoff            (input window, what our pipeline trains on)
    cutoff + 1 week .. cutoff + H weeks   (test truth, our horizons 1..H)

for one location and one cutoff of the pinned split. ``manifest.json`` records the
``input_length`` to pass to ``run_tsci.py`` for every file.

    .venv/bin/python scripts/comparison/prepare_flu_series.py \\
        --locations 51 06 48 36 12 --cutoffs 2025-11-29 2025-12-27 2026-01-24 \\
        --out outputs/experiments/tsci_comparison/series
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import TRAIN_START_DATE  # noqa: E402

__all__ = ["HORIZON_WEEKS", "build_series", "main"]

HORIZON_WEEKS = 4
DATA_CACHE = REPO_ROOT / "data" / "raw" / "flusight_hospital_admissions.csv"
TSCI_DATE_COLUMN = "date"
TSCI_VALUE_COLUMN = "OT"


def build_series(
    data: pd.DataFrame,
    location: str,
    cutoff_date: str,
    horizon_weeks: int = HORIZON_WEEKS,
    train_start_date: str = TRAIN_START_DATE,
) -> pd.DataFrame:
    """Return the ``date``/``OT`` frame for one location: input window then test truth.

    Raises when the location is absent, when a week is missing (TSci indexes by row
    position, so a gap would shift every later horizon), or when the cache does not
    reach ``cutoff + horizon_weeks``.
    """
    cutoff = pd.Timestamp(cutoff_date)
    start = pd.Timestamp(train_start_date)
    end = cutoff + dt.timedelta(weeks=horizon_weeks)

    rows = data[data["location"].astype(str) == location]
    if rows.empty:
        raise ValueError(f"location {location!r} is not in the data cache")
    dates = pd.to_datetime(rows["date"])
    window = rows[(dates >= start) & (dates <= end)].assign(date=dates).sort_values("date")

    expected = pd.date_range(start, end, freq="7D")
    if len(window) != len(expected) or not (window["date"].to_numpy() == expected.to_numpy()).all():
        raise ValueError(
            f"location {location!r}, cutoff {cutoff_date}: expected {len(expected)} consecutive "
            f"weeks from {start.date()} to {end.date()}, found {len(window)}"
        )
    if window["value"].isna().any():
        raise ValueError(f"location {location!r}, cutoff {cutoff_date}: the window has missing values")

    return pd.DataFrame(
        {
            TSCI_DATE_COLUMN: window["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            TSCI_VALUE_COLUMN: window["value"].to_numpy(dtype=float),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--locations", nargs="+", required=True, help="FIPS codes, e.g. 51 06")
    parser.add_argument("--cutoffs", nargs="+", required=True, help="cutoff dates, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(DATA_CACHE, dtype={"location": str})
    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for location in args.locations:
        for cutoff_date in args.cutoffs:
            series = build_series(data, location, cutoff_date)
            path = args.out / f"{location}_{cutoff_date}.csv"
            series.to_csv(path, index=False)
            files.append(
                {
                    "file": path.name,
                    "location": location,
                    "cutoff_date": cutoff_date,
                    "input_length": len(series) - HORIZON_WEEKS,
                    "horizon": HORIZON_WEEKS,
                    "target_end_dates": series[TSCI_DATE_COLUMN].tail(HORIZON_WEEKS).tolist(),
                }
            )
    manifest = {
        "data_sha256": hashlib.sha256(DATA_CACHE.read_bytes()).hexdigest(),
        "train_start_date": TRAIN_START_DATE,
        "files": files,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(files)} series to {args.out}")


if __name__ == "__main__":
    main()
