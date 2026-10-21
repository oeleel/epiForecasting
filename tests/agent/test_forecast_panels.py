"""Tests for src/forecast_panels.py and scripts/experiments/export_hindcast_tables.py."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.experiments.export_hindcast_tables import score_rows, scored, wis_by_location  # noqa: E402
from src.forecast_panels import (  # noqa: E402
    FAMILY_COLORS,
    UNKNOWN_FAMILY_COLOR,
    config_color,
    load_hindcast,
    plot_forecast_panels,
    resolve_location,
)

CUTOFF = "2026-01-10"
WEEKS = pd.date_range("2025-11-29", periods=11, freq="7D")  # 6 weeks before CUTOFF, 4 after


def _observed() -> pd.DataFrame:
    rows = []
    for code, name, scale in (("51", "Virginia", 1.0), ("US", "US", 20.0)):
        for i, date in enumerate(WEEKS):
            rows.append({"date": date.date().isoformat(), "location": code, "location_name": name,
                         "value": scale * (100.0 + 10 * i)})
    return pd.DataFrame(rows)


def _forecasts(cutoff: str = CUTOFF, median: float = 200.0) -> pd.DataFrame:
    rows = []
    for code in ("51", "US"):
        for horizon in (1, 2, 3, 4):
            target = (pd.Timestamp(cutoff) + timedelta(weeks=horizon)).date().isoformat()
            rows.append({
                "location": code, "cutoff_date": cutoff, "target_end_date": target,
                "forecast_date": target, "horizon": horizon, "predicted": median,
                "predicted_q05": median - 40, "predicted_q25": median - 10, "predicted_q50": median,
                "predicted_q75": median + 10, "predicted_q95": median + 40,
            })
    return pd.DataFrame(rows)


def test_plot_forecast_panels_draws_one_panel_per_config():
    forecasts = {"baseline": _forecasts(), "oversample_5": _forecasts(median=210.0), "window_52": _forecasts()}

    fig = plot_forecast_panels("Virginia", CUTOFF, forecasts, _observed())

    assert [ax.get_title() for ax in fig.axes] == ["baseline", "oversample_5", "window_52"]
    plt.close(fig)


def test_plot_forecast_panels_rejects_a_date_without_forecasts():
    forecasts = {"baseline": _forecasts()}

    try:
        plot_forecast_panels("51", "2026-01-03", forecasts, _observed())
    except ValueError as exc:
        assert "2026-01-03" in str(exc) and CUTOFF in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_panels_rejects_lookback_below_one():
    try:
        plot_forecast_panels("51", CUTOFF, {"baseline": _forecasts()}, _observed(), lookback_weeks=0)
    except ValueError as exc:
        assert "lookback_weeks" in str(exc) and "0" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plot_forecast_panels_writes_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "va.png"

        fig = plot_forecast_panels("51", CUTOFF, {"baseline": _forecasts()}, _observed(), save_path=path)

        assert path.stat().st_size > 0
        plt.close(fig)


def test_resolve_location_accepts_code_and_case_insensitive_name():
    observed = _observed()

    assert (resolve_location("51", observed), resolve_location("virginia", observed)) == ("51", "51")


def test_resolve_location_rejects_unknown_name():
    try:
        resolve_location("Narnia", _observed())
    except ValueError as exc:
        assert "Narnia" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_config_color_follows_the_family_not_the_parameter():
    assert config_color("oversample_2") == config_color("oversample_5__seed42") == FAMILY_COLORS["oversample"]
    assert config_color("lambda_calendar_3") == FAMILY_COLORS["lambda_calendar"]
    assert config_color("mystery") == UNKNOWN_FAMILY_COLOR


def test_load_hindcast_reads_the_exported_layout_baseline_first():
    with tempfile.TemporaryDirectory() as tmp:
        forecasts_dir = Path(tmp) / "forecasts"
        forecasts_dir.mkdir()
        exported = _forecasts().rename(columns=lambda c: c.removeprefix("predicted_"))
        for config in ("window_52", "baseline"):
            exported.to_csv(forecasts_dir / f"{config}.csv", index=False)

        loaded = load_hindcast(Path(tmp))

        assert list(loaded) == ["baseline", "window_52"]
        assert "predicted_q50" in loaded["baseline"].columns


def test_score_rows_keeps_us_but_scored_drops_it():
    rows = score_rows(_forecasts(), _observed())

    assert set(rows["location"]) == {"51", "US"}
    assert set(scored(rows)["location"]) == {"51"}


def test_wis_by_location_pools_horizons_per_location():
    rows = score_rows(_forecasts(), _observed())

    table = wis_by_location({"baseline": rows})

    overall = table[(table["phase"] == "all") & (table["location"] == "51")]
    assert overall["n"].item() == 4
    assert overall["wis"].item() == round(scored(rows)["wis"].mean(), 2)


ALL = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
