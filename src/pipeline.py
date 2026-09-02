"""Callable forecasting pipeline for the agent improvement loop.

This module exposes a single function — `run_pipeline(config)` — that the
agent orchestrator calls to train a fresh model and generate forecasts
from a (possibly mutated) configuration dict. It is the bridge between
the dict-based config from `src.config.get_default_config()` and the
existing XGBoost-based forecasting infrastructure in `src/`.

Design:
    - Pure callable: takes a dict, returns a path. No CLI arguments,
      no module-level state, no side effects beyond writing the output
      CSV (and the loader's data cache).
    - Honors every config field that maps to an Agent 2 action:
        xgboost.*               -> XGBRegressor params
        target.mode             -> log/raw/ratio target transform
        floor.enabled,          -> post-prediction floor constraint
        floor.floor_pct
        sample_weights.*        -> XGBoost sample_weight via the helper
                                   in src.direct_forecast
        model.family            -> "xgboost_direct" (this path) or any
                                   model-bank family (src.model_bank.runner)
        data.cutoff_date        -> training/forecast cutoff
        data.train_start_date   -> pinned training window start
        data.forecast_horizon   -> 1..H weeks ahead
        data.locations          -> optional location filter (None = all)
    - Returns the absolute path to a forecast CSV with the columns the
      flu adapter expects: location, cutoff_date, forecast_date,
      forecast, horizon, target_mode (plus a few extras from the
      ensemble's native output).
    - When verbose=False (the default for agent loops), suppresses the
      noisy stdout from the underlying training methods so the
      orchestrator can render its own progress.

The existing `scripts/pipeline.py` CLI is left untouched.
"""

from __future__ import annotations

import contextlib
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src import config as default_config_module
from src.data_loader import FluDataLoader
from src.direct_forecast import DirectForecastEnsemble, QuantileDirectForecastEnsemble
from src.feature_engineering import FeatureEngineer


DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/_adhoc")

# The one family that still runs through this module's own code path.
LEGACY_XGBOOST_FAMILY = "xgboost_direct"


def run_pipeline(
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Path:
    """Train a forecasting model from a config dict and write forecasts to disk.

    Args:
        config: Pipeline config dict (shape from `src.config.get_default_config()`).
            If None, uses the default config. Mutated copies returned by
            `agent.apply_action` are valid input.
        output_path: Destination CSV path. If None, writes to
            outputs/agent_runs/_adhoc/forecast_<cutoff>_<timestamp>.csv.
        verbose: If False (default), suppresses the per-horizon training
            chatter from the underlying ensemble.

    Returns:
        Absolute Path to the written forecast CSV. The CSV has columns:
            location, cutoff_date, forecast_date, forecast, horizon,
            forecast_week, target_mode, raw_model_output, reference_date,
            target_end_date

    Raises:
        FileNotFoundError: if the CDC data cannot be loaded.
        ValueError: if the resulting training set is empty.
    """
    if config is None:
        config = default_config_module.get_default_config()

    # ---- Model-bank dispatch --------------------------------------------------
    # Any family other than the legacy XGBoost ensemble is trained through
    # src.model_bank.runner, which speaks the ForecastModel contract. The
    # legacy path below stays authoritative for "xgboost_direct" because it
    # honors every Agent 2 action (feature toggles, reweighting, floor, ...).
    family = (config.get("model") or {}).get("family", default_config_module.DEFAULT_MODEL_FAMILY)
    if family != LEGACY_XGBOOST_FAMILY:
        from src.model_bank.runner import run_bank_model

        return run_bank_model(config, output_path=output_path, verbose=verbose)

    # ---- Extract config values with safe fallbacks --------------------------
    data_cfg = config.get("data", {})
    cutoff_date: str = data_cfg.get(
        "cutoff_date", default_config_module.DEFAULT_CUTOFF_DATE
    )
    forecast_horizon: int = int(
        data_cfg.get("forecast_horizon", default_config_module.FORECAST_HORIZON)
    )
    locations: Optional[list] = data_cfg.get("locations")  # None = all

    xgb_params: Dict[str, Any] = dict(
        config.get("xgboost", default_config_module.XGBOOST_PARAMS)
    )

    target_cfg = config.get("target", {})
    target_mode: str = target_cfg.get("mode", default_config_module.TARGET_MODE)

    floor_cfg = config.get("floor", {})
    floor_enabled: bool = bool(floor_cfg.get("enabled", True))
    floor_pct: float = float(floor_cfg.get("floor_pct", 0.30))

    quantiles_cfg = config.get("quantiles", {})
    use_quantiles: bool = bool(quantiles_cfg.get("enabled", True))
    quantile_levels: list = list(quantiles_cfg.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95]))

    sample_weights_cfg: Optional[Dict[str, Any]] = config.get("sample_weights")

    # ---- Resolve output path ------------------------------------------------
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"forecast_{cutoff_date}_{ts}.csv"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load + feature engineer --------------------------------------------
    stdout_buf = io.StringIO()
    capture = contextlib.redirect_stdout(stdout_buf) if not verbose else contextlib.nullcontext()

    with capture:
        loader = FluDataLoader()
        data = loader.load_and_preprocess(cutoff_date)

        train_start = data_cfg.get("train_start_date")
        if train_start is not None:
            data = data[pd.to_datetime(data["date"]) >= pd.to_datetime(train_start)].copy()

        if locations is not None:
            data = data[data["location"].isin(locations)].copy()

        if data.empty:
            raise ValueError(
                f"No training data after loading for cutoff_date={cutoff_date}"
                + (f", locations={locations}" if locations else "")
            )

        features_cfg = config.get("features", {})
        groups_enabled = features_cfg.get("groups_enabled")
        engineer = FeatureEngineer(groups_enabled=groups_enabled)
        features_df = engineer.create_all_features(data)

        # ---- Build + train ensemble -----------------------------------------
        # Quantile ensemble produces the q05/q25/q50/q75/q95 columns the
        # PhaseEvaluator needs to compute WIS + 95% coverage. Point ensemble
        # only produces the 'forecast' column (MAPE/MAE/bias only).
        if use_quantiles:
            ensemble = QuantileDirectForecastEnsemble(
                forecast_horizon=forecast_horizon,
                target_mode=target_mode,
                quantiles=quantile_levels,
                xgb_params_override=xgb_params,
            )
        else:
            ensemble = DirectForecastEnsemble(
                forecast_horizon=forecast_horizon,
                target_mode=target_mode,
                xgb_params_override=xgb_params,
            )

        cutoff_dt = pd.to_datetime(cutoff_date)
        training_data = features_df[features_df["date"] <= cutoff_dt].copy()

        if training_data.empty:
            raise ValueError(
                f"No training rows at or before cutoff_date={cutoff_date}"
            )

        ensemble.train(training_data, sample_weights=sample_weights_cfg)

        # ---- Generate forecasts ---------------------------------------------
        forecasts = ensemble.generate_forecasts(
            data=features_df,
            cutoff_date=cutoff_date,
            locations=locations,
            use_floor_constraint=floor_enabled,
            floor_ratio=floor_pct,
        )

    if forecasts.empty:
        raise ValueError(
            f"Ensemble produced no forecasts for cutoff_date={cutoff_date}"
        )

    forecasts.to_csv(output_path, index=False)
    return output_path
