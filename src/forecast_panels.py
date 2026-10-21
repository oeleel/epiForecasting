"""Forecast panels for visual inspection: one location, one forecast date, one panel per config.

Modelled on the lab's inspection plots (A. Adiga, 2026-10-21): observed series in black
up to the date the data was available, then for each model the median and two central
intervals over horizons 1-4, every panel on the same y-scale so the models compare at a
glance. Two deliberate differences from the lab's figure:

- The outer band is 90% (q0.05-q0.95), not 95%. The pipeline emits five quantiles
  (`src.config.QUANTILES`), so q0.025/q0.975 do not exist. The legend says 90%.
- These are hindcasts on test data, so what actually happened after the forecast date is
  known; it is drawn as hollow black markers. Without it the plot cannot show whether a
  forecast turned with the curve.

The lab's grey last-season curve and blue guardrail bounds are left out (Adiga: "you don't
need that").

Dates follow the harness's forecast CSVs: `cutoff_date` is the last observed week (the
lab's `avl_date`), horizon h targets `cutoff_date + h` weeks, and the lab's `ref_date` is
`cutoff_date + 1` week.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from agent.phase_evaluator import _compute_wis_vector  # the evaluator's own WIS

__all__ = [
    "DEFAULT_LOOKBACK_WEEKS",
    "FAMILY_COLORS",
    "available_cutoffs",
    "config_color",
    "load_hindcast",
    "plot_forecast_panels",
    "resolve_location",
]

DEFAULT_LOOKBACK_WEEKS = 6  # Adiga: "prior six, seven weeks", adjustable
DEFAULT_SEED = 42
MAX_PANEL_COLUMNS = 4
PANEL_WIDTH_IN = 3.4
PANEL_HEIGHT_IN = 3.6
OUTER_BAND_ALPHA = 0.18
INNER_BAND_ALPHA = 0.38
LINE_WIDTH = 2.0
MARKER_SIZE = 5
OBSERVED_COLOR = "#111111"
MUTED_TEXT_COLOR = "#5f5e5a"

# One hue per arm family, in the validated default categorical order; every config of a
# family shares it, so a config keeps its colour in every figure regardless of which
# other configs are plotted beside it.
FAMILY_COLORS: dict[str, str] = {
    "baseline": "#2a78d6",  # blue
    "lambda": "#eb6834",  # orange
    "lambda_calendar": "#1baf7a",  # aqua
    "oversample": "#e87ba4",  # magenta
    "window": "#6250d6",  # violet
}
UNKNOWN_FAMILY_COLOR = "#5f5e5a"

QUANTILE_COLUMNS = ("predicted_q05", "predicted_q25", "predicted_q50", "predicted_q75", "predicted_q95")
REQUIRED_FORECAST_COLUMNS = ("location", "cutoff_date", "target_end_date", "horizon", *QUANTILE_COLUMNS)
REQUIRED_OBSERVED_COLUMNS = ("date", "location", "location_name", "value")


def load_hindcast(
    run_dir: Path, configs: Sequence[str] | None = None, seed: int = DEFAULT_SEED
) -> dict[str, pd.DataFrame]:
    """Read a run's forecasts into {config: frame}, baseline first.

    Two layouts: a harness run directory (`forecasts/<config>__seed<seed>.csv`, untracked
    under `outputs/`) or an export from `scripts/experiments/export_hindcast_tables.py`
    (`forecasts/<config>.csv`, quantiles named `q05`..`q95`, tracked in git). `configs`
    are config ids without the seed suffix; None loads every saved config.
    """
    forecasts_dir = Path(run_dir) / "forecasts"
    if not forecasts_dir.is_dir():
        raise FileNotFoundError(f"no forecasts/ directory under {run_dir}")
    seeded = f"__seed{seed}.csv"
    if any(forecasts_dir.glob(f"*{seeded}")):
        files = {p.name[: -len(seeded)]: p for p in forecasts_dir.glob(f"*{seeded}")}
    else:
        files = {p.stem: p for p in forecasts_dir.glob("*.csv") if "__seed" not in p.name}
    if not files:
        raise FileNotFoundError(f"no forecasts for seed {seed} in {forecasts_dir}")
    saved = sorted(files, key=lambda c: (c != "baseline", c))
    wanted = list(configs) if configs else saved
    missing = [c for c in wanted if c not in files]
    if missing:
        raise ValueError(f"configs {missing} not saved in {forecasts_dir}; available: {saved}")
    exported_names = {c.removeprefix("predicted_"): c for c in QUANTILE_COLUMNS}
    return {
        config: pd.read_csv(files[config], dtype={"location": str}).rename(columns=exported_names)
        for config in wanted
    }


def available_cutoffs(forecasts: Mapping[str, pd.DataFrame]) -> list[str]:
    """Forecast dates present in every config (the dates a figure can be drawn for)."""
    if not forecasts:
        raise ValueError("forecasts is empty")
    common = set.intersection(*(set(frame["cutoff_date"].astype(str)) for frame in forecasts.values()))
    return sorted(common)


def resolve_location(location: str, observed: pd.DataFrame) -> str:
    """Accept a FIPS code (`51`, `US`) or a name (`Virginia`, case-insensitive)."""
    names = observed[["location", "location_name"]].drop_duplicates()
    codes = set(names["location"].astype(str))
    if location in codes:
        return location
    by_name = {str(name).casefold(): str(code) for code, name in names.itertuples(index=False)}
    code = by_name.get(location.casefold())
    if code is None:
        raise ValueError(f"unknown location {location!r}; use a FIPS code or a name from the data")
    return code


def plot_forecast_panels(
    location: str,
    cutoff_date: str,
    forecasts: Mapping[str, pd.DataFrame],
    observed: pd.DataFrame,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    show_truth: bool = True,
    save_path: Path | None = None,
) -> plt.Figure:
    """Draw one panel per config for `location` forecast from `cutoff_date`.

    Args:
        location: FIPS code or location name.
        cutoff_date: last observed week (YYYY-MM-DD); must be a forecast date in every config.
        forecasts: {config id: forecast frame} in panel order, e.g. from `load_hindcast`.
        observed: the data cache (`date, location, location_name, value`).
        lookback_weeks: weeks of observed history before `cutoff_date`.
        show_truth: draw what was observed after `cutoff_date` as hollow markers.
        save_path: write the figure there (PNG/SVG/PDF by extension) when given.

    Raises:
        ValueError: on an unknown location, a date not forecast in every config, or
            lookback_weeks < 1.
    """
    if lookback_weeks < 1:
        raise ValueError(f"lookback_weeks must be >= 1, got {lookback_weeks}")
    _require_columns(observed, REQUIRED_OBSERVED_COLUMNS, "observed")
    for config, frame in forecasts.items():
        _require_columns(frame, REQUIRED_FORECAST_COLUMNS, f"forecasts[{config!r}]")
    dates = available_cutoffs(forecasts)
    if cutoff_date not in dates:
        raise ValueError(f"no forecast from {cutoff_date} in every config; available: {', '.join(dates)}")
    code = resolve_location(location, observed)

    cutoff = pd.Timestamp(cutoff_date)
    series = observed[observed["location"].astype(str) == code].assign(date=lambda d: pd.to_datetime(d["date"]))
    series = series.sort_values("date")
    start = cutoff - timedelta(weeks=lookback_weeks)
    history = series[(series["date"] >= start) & (series["date"] <= cutoff)]
    if history.empty or history["date"].max() != cutoff:
        raise ValueError(f"no observed value for {code} on {cutoff_date}; cannot anchor the forecast")
    anchor_value = float(history["value"].iloc[-1])

    panels = {config: _panel_rows(frame, code, cutoff_date) for config, frame in forecasts.items()}
    empty = [config for config, rows in panels.items() if rows.empty]
    if empty:
        raise ValueError(f"no forecast rows for location {code} on {cutoff_date} in {empty}")
    last_target = max(rows["target_end_date"].max() for rows in panels.values())
    truth = series[(series["date"] > cutoff) & (series["date"] <= last_target)] if show_truth else series.iloc[0:0]

    n_cols = min(len(panels), MAX_PANEL_COLUMNS)
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(PANEL_WIDTH_IN * n_cols, PANEL_HEIGHT_IN * n_rows + 0.6),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax in axes.flat[len(panels):]:
        ax.remove()  # no empty panels
    for index in range(len(panels)):
        if index + n_cols >= len(panels):  # nothing below it: it carries the date labels
            axes.flat[index].xaxis.set_tick_params(labelbottom=True)

    for ax, (config, rows) in zip(axes.flat, panels.items()):
        color = config_color(config)
        _draw_observed(ax, history, truth)
        _draw_forecast(ax, rows, cutoff, anchor_value, color)
        # The WIS line sits between the title and the plot, so it never covers data.
        ax.set_title(config, fontsize=11, pad=16 if not truth.empty else 6)
        if not truth.empty:
            ax.text(0.5, 1.015, _wis_note(rows, truth), transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, color=MUTED_TEXT_COLOR)
        _style_axes(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("weekly hospital admissions")
    # Once, after every panel is drawn: on shared axes an earlier call would freeze the top
    # at what the panels drawn so far needed and clip the later ones.
    axes.flat[0].set_ylim(bottom=0)

    name = str(series["location_name"].iloc[0])
    label = name if name == code else f"{name} ({code})"
    ref_date = (cutoff + timedelta(weeks=1)).date().isoformat()
    fig.suptitle(f"{label}  ||  ref_date = {ref_date}  ||  avl_date = {cutoff_date}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # Anchored below the axes; savefig's tight bbox pulls it in without a fixed gap.
    fig.legend(handles=_legend_handles(not truth.empty), loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=6, frameon=False, fontsize=9)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def config_color(config: str) -> str:
    """Colour of a config's family: the config id minus its trailing numeric parameter."""
    family = config.split("__seed")[0]
    head, _, tail = family.rpartition("_")
    if head and _is_number(tail):
        family = head
    return FAMILY_COLORS.get(family, UNKNOWN_FAMILY_COLOR)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns {missing}")


def _panel_rows(frame: pd.DataFrame, code: str, cutoff_date: str) -> pd.DataFrame:
    rows = frame[(frame["location"].astype(str) == code) & (frame["cutoff_date"].astype(str) == cutoff_date)]
    return rows.assign(target_end_date=lambda d: pd.to_datetime(d["target_end_date"])).sort_values("horizon")


def _draw_observed(ax: plt.Axes, history: pd.DataFrame, truth: pd.DataFrame) -> None:
    ax.plot(history["date"], history["value"], "-o", color=OBSERVED_COLOR, linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE, zorder=4)
    if not truth.empty:
        ax.plot(truth["date"], truth["value"], ":o", color=OBSERVED_COLOR, linewidth=1.0,
                markersize=MARKER_SIZE + 1, markerfacecolor="white", markeredgewidth=1.4, zorder=5)


def _draw_forecast(ax: plt.Axes, rows: pd.DataFrame, cutoff: pd.Timestamp, anchor: float, color: str) -> None:
    # Every band and the median start at the last observed value, so the forecast reads
    # as a continuation of the curve.
    dates = [cutoff, *rows["target_end_date"]]

    def path(column: str) -> list[float]:
        return [anchor, *rows[column].astype(float)]

    ax.fill_between(dates, path("predicted_q05"), path("predicted_q95"), color=color,
                    alpha=OUTER_BAND_ALPHA, linewidth=0, zorder=1)
    ax.fill_between(dates, path("predicted_q25"), path("predicted_q75"), color=color,
                    alpha=INNER_BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(dates, path("predicted_q50"), "--o", color=color, linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE - 1, zorder=3)
    ax.plot([cutoff], [anchor], "o", markersize=MARKER_SIZE + 3, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2, zorder=6)


def _wis_note(rows: pd.DataFrame, truth: pd.DataFrame) -> str:
    joined = rows.merge(truth[["date", "value"]], left_on="target_end_date", right_on="date", how="inner")
    if joined.empty:
        return ""
    wis = _compute_wis_vector(joined["value"], *(joined[c] for c in QUANTILE_COLUMNS))
    by_h = "  ".join(f"h{int(h)} {w:.0f}" for h, w in zip(joined["horizon"], wis))
    return f"WIS {by_h}"


def _style_axes(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.SA))
    ax.grid(True, which="major", axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def _legend_handles(with_truth: bool) -> list:
    handles = [
        Line2D([], [], color=OBSERVED_COLOR, marker="o", linewidth=LINE_WIDTH, label="observed"),
        Line2D([], [], color=MUTED_TEXT_COLOR, marker="o", linestyle="--", label="median"),
        Patch(color=MUTED_TEXT_COLOR, alpha=INNER_BAND_ALPHA + 0.1, label="50% interval"),
        Patch(color=MUTED_TEXT_COLOR, alpha=OUTER_BAND_ALPHA, label="90% interval"),
    ]
    if with_truth:
        handles.insert(1, Line2D([], [], color=OBSERVED_COLOR, marker="o", linestyle=":", linewidth=1.0,
                                 markerfacecolor="white", label="observed after forecast date"))
    return handles
