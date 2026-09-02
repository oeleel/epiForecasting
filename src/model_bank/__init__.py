"""Model bank: one contract, many forecasting models, zero per-model plumbing.

Public surface (see each module's docstring for the why):
    contract     ForecastModel ABC, ParamSpec, frame validators, column names
    registry     @register, resolve_family("name" | "pkg.mod:Class"), list_families
    data_bridge  CDC <-> long format; validated prediction -> repo forecast CSV
    runner       run_bank_model(config, output_path): load -> fit -> predict -> CSV
    baselines    persistence / seasonal_naive (no extra deps)
    legacy       xgboost_direct / nn_quantile (the in-repo ensembles)
    nixtla       statsforecast / mlforecast / neuralforecast examples (optional deps)
"""

from __future__ import annotations

from src.model_bank.contract import (
    DEFAULT_QUANTILE_LEVELS,
    HORIZON_COL,
    ID_COL,
    TARGET_COL,
    TIME_COL,
    ForecastModel,
    ModelBankError,
    ParamSpec,
    quantile_column,
    quantile_columns,
    validate_forecast_frame,
    validate_history_frame,
    validate_params,
)
from src.model_bank.data_bridge import from_long, to_forecast_csv, to_long
from src.model_bank.registry import FamilyInfo, list_families, register, resolve_family

__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "HORIZON_COL",
    "ID_COL",
    "TARGET_COL",
    "TIME_COL",
    "FamilyInfo",
    "ForecastModel",
    "ModelBankError",
    "ParamSpec",
    "from_long",
    "list_families",
    "quantile_column",
    "quantile_columns",
    "register",
    "resolve_family",
    "to_forecast_csv",
    "to_long",
    "validate_forecast_frame",
    "validate_history_frame",
    "validate_params",
]
