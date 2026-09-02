"""Run one bank family end to end: CDC data -> fit -> predict -> forecast CSV.

This is the family-agnostic twin of the legacy XGBoost path in
`src.pipeline.run_pipeline`. The pipeline dispatches here for every family
except "xgboost_direct", so the orchestrator, the CLI, and the selection
stage all drive any model through the same three calls:

    history = load_history(cutoff_date, train_start_date, locations)
    csv_frame, meta = fit_predict(family, params, history, ...)
    csv_frame.to_csv(output_path)

`fit_predict` is pure with respect to disk (no I/O) so the selection stage
can call it in a rolling-origin loop and keep everything in memory.

Invariant: `history` passed to a model contains only rows with ds <= cutoff.
Leakage is prevented here, once, not in every model.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import config as repo_config
from src.model_bank.contract import (
    DEFAULT_QUANTILE_LEVELS,
    ID_COL,
    TIME_COL,
    ForecastModel,
    ModelBankError,
)
from src.model_bank.data_bridge import to_forecast_csv, to_long
from src.model_bank.registry import resolve_family

__all__ = ["FitPredictMeta", "fit_predict", "load_history", "run_bank_model"]

DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/_adhoc")


@dataclass(frozen=True, slots=True)
class FitPredictMeta:
    family: str
    cutoff_date: str
    n_series: int
    n_history_rows: int
    fit_seconds: float
    predict_seconds: float
    n_quantile_crossings_repaired: int


def load_history(
    cutoff_date: str,
    train_start_date: str = repo_config.TRAIN_START_DATE,
    locations: Sequence[str] | None = None,
    cdc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format history with train_start <= ds <= cutoff, optionally filtered by series.

    Pass `cdc` to reuse an already-loaded CDC frame (the selection stage
    loads once and slices per cutoff); otherwise FluDataLoader is used.
    """
    if cdc is None:
        from src.data_loader import FluDataLoader

        with contextlib.redirect_stdout(io.StringIO()):
            cdc = FluDataLoader().fetch_data()
    start = pd.to_datetime(train_start_date)
    end = pd.to_datetime(cutoff_date)
    if start > end:
        raise ModelBankError(f"train_start_date {train_start_date} is after cutoff {cutoff_date}")
    dates = pd.to_datetime(cdc["date"])
    window = cdc[(dates >= start) & (dates <= end)]
    if locations is not None:
        wanted = {str(x).zfill(2) for x in locations}
        window = window[window["location"].astype(str).str.zfill(2).isin(wanted)]
    if window.empty:
        raise ModelBankError(
            f"no history rows in [{train_start_date}, {cutoff_date}]"
            + (f" for locations {list(locations)}" if locations else "")
        )
    return to_long(window)


def fit_predict(
    family: str,
    params: dict[str, Any] | None,
    history: pd.DataFrame,
    cutoff_date: str,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    horizon: int = repo_config.FORECAST_HORIZON,
) -> tuple[pd.DataFrame, FitPredictMeta]:
    """Instantiate `family`, fit on `history`, predict, and return the repo forecast CSV frame."""
    if history[TIME_COL].max() > pd.to_datetime(cutoff_date):
        raise ModelBankError(
            f"history extends past cutoff {cutoff_date} "
            f"(max ds {history[TIME_COL].max().date()}); refusing to fit on leaked rows"
        )
    model_cls = resolve_family(family)
    model: ForecastModel = model_cls(
        params=params, quantile_levels=quantile_levels, horizon=horizon
    )

    t0 = time.perf_counter()
    model.fit(history)
    t1 = time.perf_counter()
    prediction = model.predict(history)
    t2 = time.perf_counter()

    expected_ids = history[ID_COL].unique()
    csv_frame, n_crossing = to_forecast_csv(
        prediction, cutoff_date, model.family, model.quantile_levels, model.horizon,
        expected_ids=expected_ids,
    )
    meta = FitPredictMeta(
        family=model.family,
        cutoff_date=cutoff_date,
        n_series=int(len(expected_ids)),
        n_history_rows=int(len(history)),
        fit_seconds=t1 - t0,
        predict_seconds=t2 - t1,
        n_quantile_crossings_repaired=n_crossing,
    )
    return csv_frame, meta


def run_bank_model(
    config: dict[str, Any],
    output_path: str | Path | None = None,
    verbose: bool = False,
) -> Path:
    """Config dict -> forecast CSV on disk, for any family. Mirrors src.pipeline.run_pipeline."""
    model_cfg = config.get("model") or {}
    family = model_cfg.get("family")
    if not family:
        raise ModelBankError("config['model']['family'] is required for the model bank path")
    params = dict(model_cfg.get("params") or {})

    data_cfg = config.get("data") or {}
    cutoff_date = data_cfg.get("cutoff_date")
    if not cutoff_date:
        raise ModelBankError("config['data']['cutoff_date'] is required")
    horizon = int(data_cfg.get("forecast_horizon", repo_config.FORECAST_HORIZON))
    train_start = data_cfg.get("train_start_date", repo_config.TRAIN_START_DATE)
    locations = data_cfg.get("locations")

    quantiles_cfg = config.get("quantiles") or {}
    levels = list(quantiles_cfg.get("levels", DEFAULT_QUANTILE_LEVELS))

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"forecast_{family}_{cutoff_date}_{stamp}.csv"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = load_history(cutoff_date, train_start_date=train_start, locations=locations)
    csv_frame, meta = fit_predict(family, params, history, cutoff_date, levels, horizon)
    if verbose:
        print(
            f"[model_bank] {meta.family} @ {meta.cutoff_date}: {meta.n_series} series, "
            f"fit {meta.fit_seconds:.1f}s, predict {meta.predict_seconds:.1f}s, "
            f"{meta.n_quantile_crossings_repaired} quantile crossings repaired"
        )
    csv_frame.to_csv(output_path, index=False)
    return output_path
