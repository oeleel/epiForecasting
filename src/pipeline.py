"""Callable forecasting pipeline for the agent improvement loop.

This module exposes a single function - `run_pipeline(config)` - that the
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
        data.train_window_weeks -> optional: fit only on the last N weeks of
                                   feature-engineered rows before the cutoff
                                   (date > cutoff - N weeks). None/absent =
                                   every row from train_start_date. Applied
                                   AFTER feature engineering on purpose: the
                                   lag / rolling / seasonal features still see
                                   the full history, only the rows the model
                                   fits on shrink (the lab's "fit on the
                                   last 12 weeks" knob; knowledge-bank design
                                   doc, rectification action 4). N must be
                                   >= MIN_TRAIN_WINDOW_WEEKS or ValueError.
        data.oversample         -> optional {"phase": P, "factor": k}: every
                                   training row whose TARGET date (origin +
                                   h weeks) falls in phase P under the
                                   evaluator's default phase scheme is
                                   fitted k times (k-1 extra copies), the
                                   same rows `sample_weights.by_phase`
                                   weights and the evaluator scores as P. Only on
                                   the xgboost_direct quantile path; see
                                   PhaseOversampledQuantileEnsemble for why the
                                   copies are made after the target shift and
                                   never touch the validation rows.
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
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src import config as default_config_module
from src.data_loader import FluDataLoader
from src.direct_forecast import (
    DirectForecastEnsemble,
    QuantileDirectForecastEnsemble,
    _compute_sample_weights,
)
from src.feature_engineering import FeatureEngineer


__all__ = [
    "run_pipeline",
    "validate_train_window_weeks",
    "MIN_TRAIN_WINDOW_WEEKS",
    "OVERSAMPLE_MIN_FACTOR",
    "OversampleSpec",
    "PhaseOversampledQuantileEnsemble",
    "known_phases",
    "oversample_positions",
    "validate_oversample",
]

_log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/_adhoc")

# The one family that still runs through this module's own code path.
LEGACY_XGBOOST_FAMILY = "xgboost_direct"

# Smallest admissible data.train_window_weeks. The horizon-h model drops the
# last h rows per location (no target yet), so a window of FORECAST_HORIZON
# weeks would leave the h=FORECAST_HORIZON model with zero rows; +4 keeps at
# least four origin weeks per location for every horizon.
MIN_TRAIN_WINDOW_WEEKS = default_config_module.FORECAST_HORIZON + 4

DAYS_PER_WEEK = 7

# data.oversample: a factor of 1 is the stock fit, so the smallest real
# oversampling is one extra copy per matching row.
OVERSAMPLE_MIN_FACTOR = 2
OVERSAMPLE_KEYS = frozenset({"phase", "factor"})
# A leap year, so every (month, day) any phase scheme can name is visited.
_PHASE_PROBE_YEAR = 2024
_DAYS_IN_LEAP_YEAR = 366


@dataclass(frozen=True, slots=True)
class OversampleSpec:
    """Validated `data.oversample`: fit rows of `phase` `factor` times each."""

    phase: str
    factor: int


@lru_cache(maxsize=1)
def known_phases() -> FrozenSet[str]:
    """Every label `PhaseEvaluator.assign_phase` can return under its default scheme.

    Derived by labelling each day of a leap year rather than read from a
    constant, so it follows whatever scheme the evaluator scores with.
    """
    # Lazy import: agent/ imports src.pipeline through the flu adapter.
    from agent.phase_evaluator import PhaseEvaluator

    start = date(_PHASE_PROBE_YEAR, 1, 1)
    return frozenset(
        PhaseEvaluator.assign_phase((start + timedelta(days=i)).isoformat())
        for i in range(_DAYS_IN_LEAP_YEAR)
    )


def validate_oversample(value: Any) -> OversampleSpec:
    """Return `data.oversample` as an OversampleSpec, or raise a specific ValueError."""
    if not isinstance(value, dict):
        raise ValueError(
            'data.oversample must be a dict {"phase": str, "factor": int}, '
            f"got {type(value).__name__}: {value!r}"
        )
    if set(value) != OVERSAMPLE_KEYS:
        raise ValueError(
            f"data.oversample must have exactly the keys {sorted(OVERSAMPLE_KEYS)}, "
            f"got {sorted(value)}"
        )
    phase, factor = value["phase"], value["factor"]
    phases = known_phases()
    if not isinstance(phase, str) or phase not in phases:
        raise ValueError(
            f"data.oversample.phase must be one of {sorted(phases)}, got {phase!r}"
        )
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise ValueError(
            "data.oversample.factor must be an int, "
            f"got {type(factor).__name__}: {factor!r}"
        )
    if factor < OVERSAMPLE_MIN_FACTOR:
        raise ValueError(
            f"data.oversample.factor must be >= {OVERSAMPLE_MIN_FACTOR}, got {factor}"
        )
    return OversampleSpec(phase=phase, factor=factor)


def oversample_positions(
    origin_dates: Sequence[Any], horizon: int, spec: OversampleSpec
) -> np.ndarray:
    """Row positions 0..n-1, each position whose target is in `spec.phase` repeated `spec.factor` times.

    A row is labelled by its TARGET date, origin + `horizon` weeks, under
    the evaluator's default scheme, because the evaluator scores each
    forecast by its target date and `sample_weights.by_phase` labels rows
    the same way. Origin-date labels would copy February origins whose h4
    target is in March and skip December origins whose h4 target is in
    January.

    np.repeat keeps every copy directly after its original, so on a
    (location, date)-sorted frame the copies stay inside their own
    location's block and in date order.
    """
    from agent.phase_evaluator import DEFAULT_PHASE_SCHEME, assign_phases

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be an int >= 1, got {horizon!r}")
    target_dates = pd.to_datetime(pd.Series(origin_dates)) + pd.to_timedelta(
        horizon * DAYS_PER_WEEK, unit="D"
    )
    phases = assign_phases(target_dates, DEFAULT_PHASE_SCHEME)
    counts = np.where(phases == spec.phase, spec.factor, 1)
    return np.repeat(np.arange(len(target_dates)), counts)


class PhaseOversampledQuantileEnsemble(QuantileDirectForecastEnsemble):
    """QuantileDirectForecastEnsemble whose training rows of one phase are fitted k times.

    Why the copies are made here and not on the feature frame: the target is
    built inside `prepare_horizon_data` as a per-location `shift(-h)` over
    (location, date)-sorted rows. A duplicated feature row would become its
    own original's h-step-ahead neighbour, so the target of the original
    would be a copy of the same week. Copies are therefore taken from the
    already-shifted (X, y, meta) frame.

    Why `train` is overridden instead of only `prepare_horizon_data`: the
    parent splits train/validation positionally at int(n * (1 - split)),
    and early stopping runs on the validation rows. Adding rows anywhere
    moves that index, which would move rows of the boundary location (about
    split * extra rows) from training into validation. This override takes
    the parent's split index on the ORIGINAL rows, oversamples only the
    training side, and passes the validation rows through unchanged, so the
    oversample arm differs from the stock fit only by the copies. It mirrors
    the parent's fit loop (same prepare_horizon_data, split rule, sample
    weights, model.train call) and skips only the parent's validation-MAE
    bookkeeping, which nothing downstream reads.
    """

    def __init__(self, *, oversample: OversampleSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.oversample = oversample

    def train(
        self,
        data: pd.DataFrame,
        validation_split: float = 0.2,
        sample_weights: Optional[Dict] = None,
    ) -> Dict:
        results: Dict[str, Any] = {
            "training_date": datetime.now().isoformat(),
            "target_mode": self.target_mode,
            "quantiles": self.quantiles,
            "oversample": {"phase": self.oversample.phase, "factor": self.oversample.factor},
            "horizons": {},
            "sample_weights_applied": bool(sample_weights),
        }
        self._training_data = data.copy()

        for horizon in range(1, self.forecast_horizon + 1):
            for q in self.quantiles:
                X, y, _, meta = self.prepare_horizon_data(
                    data, horizon, quantile=q, return_metadata=True
                )
                # The parent's positional split, on the original rows.
                split_idx = int(len(X) * (1 - validation_split))
                take = oversample_positions(meta["date"].iloc[:split_idx], horizon, self.oversample)
                weights = _compute_sample_weights(meta, sample_weights)
                self.models[horizon][q].train(
                    X.iloc[take],
                    y.iloc[take],
                    (X.iloc[split_idx:], y.iloc[split_idx:]),
                    sample_weight=weights[:split_idx][take] if weights is not None else None,
                )
            results["horizons"][horizon] = {
                "n_train_rows": split_idx,
                "n_train_samples": len(take),
                "n_val_samples": len(X) - split_idx,
            }

        self.is_trained = True
        self.training_info = results
        return results


def validate_train_window_weeks(value: Any) -> int:
    """Return `data.train_window_weeks` as an int, or raise a specific ValueError.

    Accepts a positive integer >= MIN_TRAIN_WINDOW_WEEKS. Rejects bools,
    floats, strings and anything below the minimum, naming the value, so a
    mis-typed experiment arm fails before any training starts.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "data.train_window_weeks must be an int number of weeks, "
            f"got {type(value).__name__}: {value!r}"
        )
    if value < MIN_TRAIN_WINDOW_WEEKS:
        raise ValueError(
            f"data.train_window_weeks={value} is below the minimum "
            f"{MIN_TRAIN_WINDOW_WEEKS} (FORECAST_HORIZON + 4)"
        )
    return value


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
        ValueError: if the resulting training set is empty, or if
            data.train_window_weeks is set below MIN_TRAIN_WINDOW_WEEKS.
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
        if (config.get("data") or {}).get("oversample") is not None:
            raise ValueError(
                f"data.oversample is implemented only for model.family="
                f"{LEGACY_XGBOOST_FAMILY!r}, got family={family!r}"
            )
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
    train_window_weeks_raw = data_cfg.get("train_window_weeks")
    train_window_weeks: Optional[int] = (
        validate_train_window_weeks(train_window_weeks_raw)
        if train_window_weeks_raw is not None
        else None
    )
    oversample_raw = data_cfg.get("oversample")
    oversample: Optional[OversampleSpec] = (
        validate_oversample(oversample_raw) if oversample_raw is not None else None
    )

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
    if oversample is not None and not use_quantiles:
        raise ValueError(
            "data.oversample requires quantiles.enabled=True (the point "
            f"ensemble has no oversampling path), got data.oversample={oversample_raw!r}"
        )

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
        if oversample is not None:
            ensemble = PhaseOversampledQuantileEnsemble(
                oversample=oversample,
                forecast_horizon=forecast_horizon,
                target_mode=target_mode,
                quantiles=quantile_levels,
                xgb_params_override=xgb_params,
            )
        elif use_quantiles:
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

        # Optional rolling fit window (data.train_window_weeks). Applied to the
        # feature-engineered frame so lags/seasonal features keep full history.
        if train_window_weeks is not None:
            # Explicit-unit Timedelta: keyword form warns under numpy 2.x.
            window_start = cutoff_dt - pd.Timedelta(DAYS_PER_WEEK * train_window_weeks, unit="D")
            training_data = training_data[training_data["date"] > window_start].copy()
            if training_data.empty:
                raise ValueError(
                    f"No training rows in the last {train_window_weeks} weeks before "
                    f"cutoff_date={cutoff_date} (window start {window_start.date()})"
                )

        # data.oversample is applied here in effect, but the copies are made
        # inside PhaseOversampledQuantileEnsemble.train (after the target
        # shift, training side of the split only); see its docstring.
        _log.info(
            "pipeline training rows: %d (cutoff=%s, train_window_weeks=%s, "
            "oversample=%s, date range %s..%s)",
            len(training_data),
            cutoff_date,
            train_window_weeks,
            oversample,
            pd.Timestamp(training_data["date"].min()).date(),
            pd.Timestamp(training_data["date"].max()).date(),
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
