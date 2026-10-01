"""Flu forecasting adapter for the agentic framework.

Wraps the existing XGBoost-based influenza hospitalization forecasting pipeline
to provide domain-specific data loading, metrics computation, and LLM context.
"""

import functools
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.domain_adapter import DomainAdapter
from agent.knowledge import KnowledgeBank, RetrievalContext, render_known_facts
from agent.phase_evaluator import PhaseEvaluator

# The family whose knobs live in the legacy config sections (xgboost.*,
# floor.*, target.*, features.*, sample_weights.*) and whose action catalog
# is hand-written below. Every other family gets a catalog generated from
# its ForecastModel.param_space(), and its knobs live in model.params.*.
LEGACY_MODEL_FAMILY = "xgboost_direct"

# Upper guardrail for set_training_window. 104 weeks = 2 seasons, the long
# "lull" window the lab pairs with 52 (meetings 09-03/09-09/09-24). The
# lower bound is src.pipeline.MIN_TRAIN_WINDOW_WEEKS (FORECAST_HORIZON + 4),
# read from the pipeline so the catalog and the pipeline can never disagree.
DEFAULT_MAX_TRAIN_WINDOW_WEEKS = 104


@functools.lru_cache(maxsize=1)
def training_window_guardrail() -> Tuple[int, int]:
    """(min, max) admissible `weeks` for the set_training_window action.

    Resolved lazily: src.pipeline imports xgboost (~2s, and the libomp clash
    on macOS), and this module keeps src/* out of its import time so the CLI
    and the knowledge tooling stay light.
    """
    from src.pipeline import MIN_TRAIN_WINDOW_WEEKS

    return (MIN_TRAIN_WINDOW_WEEKS, DEFAULT_MAX_TRAIN_WINDOW_WEEKS)


class FluForecastAdapter(DomainAdapter):
    """Adapter for the CDC FluSight influenza hospitalization forecasting model.

    Wraps:
        - src.data_loader.FluDataLoader (actuals)
        - src.evaluate.ModelEvaluator (baseline metrics)
        - agent.phase_evaluator.PhaseEvaluator (phase-aware metrics)
    """

    def __init__(
        self,
        project_root: str = None,
        exclude_locations: List[str] = None,
        knowledge_bank: KnowledgeBank | None = None,
        knowledge_enabled: bool = True,
    ):
        if not isinstance(knowledge_enabled, bool):
            raise ValueError(
                f"knowledge_enabled must be a bool, got {type(knowledge_enabled).__name__}: "
                f"{knowledge_enabled!r}"
            )
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self._location_names = None  # lazy-loaded FIPS-to-name mapping
        self.exclude_locations = set(exclude_locations) if exclude_locations else set()
        # Opened lazily by `knowledge_bank` on the first get_domain_context call, so
        # constructing an adapter never touches knowledge/knowledge.db. Tests pass a
        # bank built in a temp dir; the CLI and the improve loop get the repo bank.
        self._knowledge_bank = knowledge_bank
        # `improve --no-knowledge` is an A/B control against the bank-aware default:
        # with knowledge_enabled=False the KNOWN FACTS block is left out of every
        # prompt and the bank is never opened, so the run reads no bank at all.
        self.knowledge_enabled = knowledge_enabled

    @property
    def knowledge_bank(self) -> KnowledgeBank:
        """The knowledge bank behind get_domain_context (repo bank unless injected)."""
        if self._knowledge_bank is None:
            self._knowledge_bank = KnowledgeBank.open()
        return self._knowledge_bank

    @property
    def location_names(self) -> Dict[str, str]:
        """FIPS code to state name mapping, built from actuals data."""
        if self._location_names is None:
            self._location_names = self._build_location_map()
        return self._location_names

    def _build_location_map(self) -> Dict[str, str]:
        """Build FIPS-to-name mapping from the actuals data."""
        try:
            data_path = self.project_root / "data" / "raw" / "flusight_hospital_admissions.csv"
            if data_path.exists():
                df = pd.read_csv(data_path, usecols=["location", "location_name"])
                df["location"] = df["location"].astype(str).str.zfill(2)
                return dict(zip(df["location"], df["location_name"]))
        except Exception:
            pass

        # Fallback: try loading via FluDataLoader
        try:
            from src.data_loader import FluDataLoader
            loader = FluDataLoader()
            data = loader.fetch_data()
            data["location"] = data["location"].astype(str).str.zfill(2)
            return dict(zip(data["location"], data["location_name"]))
        except Exception:
            return {}

    def load_data(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load forecast CSV and actual hospitalization data.

        Config keys (provide one of):
            forecast_csv: Explicit path to forecast CSV file
            cutoff_date: Find forecast files matching this date

        Returns:
            (forecasts_df, actuals_df)
        """
        # Load forecasts
        if "forecast_csv" in config and config["forecast_csv"]:
            forecast_path = Path(config["forecast_csv"])
            if not forecast_path.is_absolute():
                forecast_path = self.project_root / forecast_path
            if not forecast_path.exists():
                raise FileNotFoundError(f"Forecast CSV not found: {forecast_path}")
            forecasts = pd.read_csv(forecast_path)
        elif "cutoff_date" in config:
            forecasts = self._find_forecasts_for_date(config["cutoff_date"])
        else:
            raise ValueError("Config must include 'forecast_csv' or 'cutoff_date'")

        # Load actuals
        actuals = self._load_actuals()

        # Filter out excluded locations
        if self.exclude_locations:
            forecasts = forecasts[
                ~forecasts["location"].astype(str).isin(self.exclude_locations)
            ]

        return forecasts, actuals

    def _find_forecasts_for_date(self, cutoff_date: str) -> pd.DataFrame:
        """Search outputs/ for forecast files matching the given cutoff date."""
        search_dirs = [
            self.project_root / "outputs" / "quantile_hindcasts",
            self.project_root / "outputs" / "forecasts",
            self.project_root / "outputs",
        ]

        # Look for CSV files containing forecasts
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for csv_path in sorted(search_dir.glob("*.csv"), reverse=True):
                try:
                    df = pd.read_csv(csv_path, nrows=5)
                    if "forecast" in df.columns and "location" in df.columns:
                        full_df = pd.read_csv(csv_path)
                        if "cutoff_date" in full_df.columns:
                            if cutoff_date in full_df["cutoff_date"].astype(str).values:
                                return full_df
                except Exception:
                    continue

        raise FileNotFoundError(
            f"No forecast CSV found for cutoff_date={cutoff_date}. "
            f"Searched: {[str(d) for d in search_dirs]}. "
            f"Use --forecast-csv to specify the file directly."
        )

    def _load_actuals(self) -> pd.DataFrame:
        """Load actual hospitalization data via FluDataLoader or from cache."""
        # Try direct CSV read first (faster, no dependency on FluDataLoader working)
        data_path = self.project_root / "data" / "raw" / "flusight_hospital_admissions.csv"
        if data_path.exists():
            return pd.read_csv(data_path)

        # Fallback to FluDataLoader
        try:
            from src.data_loader import FluDataLoader
            loader = FluDataLoader()
            return loader.fetch_data()
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load actuals data. Ensure data/raw/flusight_hospital_admissions.csv "
                f"exists or run: python -m src.data_loader update\n"
                f"Error: {e}"
            )

    def compute_metrics(
        self, forecasts: pd.DataFrame, actuals: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics with phase breakdown.

        Returns a structured dict ready for LLM consumption:
            overall, by_horizon, by_phase, worst_locations, best_locations,
            worst_segments, date_range, n_locations
        """
        # Merge forecasts with actuals
        merged = PhaseEvaluator.merge_forecasts_actuals(forecasts, actuals)

        # Overall metrics
        overall = PhaseEvaluator.compute_overall_metrics(merged)

        # By horizon
        by_horizon = PhaseEvaluator.evaluate_by_horizon(merged)

        # By phase
        by_phase = PhaseEvaluator.evaluate_by_phase(merged)

        # By location
        location_df = PhaseEvaluator.evaluate_by_location(merged)

        # Map FIPS to names
        loc_names = self.location_names

        # Worst locations (top 5 by MAPE)
        worst_locs = location_df.head(5)
        worst_locations = []
        for _, row in worst_locs.iterrows():
            fips = str(row["location"])
            worst_locations.append({
                "location": row.get("location_name", loc_names.get(fips, fips)),
                "fips": fips,
                "mape": round(row["mape"], 1),
                "mae": round(row["mae"], 1),
                "bias": round(row["bias"], 1),
                "n": int(row["n"]),
            })

        # Best locations (bottom 5 by MAPE)
        best_locs = location_df.tail(5).iloc[::-1]
        best_locations = []
        for _, row in best_locs.iterrows():
            fips = str(row["location"])
            best_locations.append({
                "location": row.get("location_name", loc_names.get(fips, fips)),
                "fips": fips,
                "mape": round(row["mape"], 1),
                "mae": round(row["mae"], 1),
                "bias": round(row["bias"], 1),
                "n": int(row["n"]),
            })

        # Worst individual predictions
        worst_segments = PhaseEvaluator.identify_worst_segments(merged, n=10)

        # Date range
        dates = pd.to_datetime(merged["forecast_date"])
        date_range = {
            "min": str(dates.min().date()),
            "max": str(dates.max().date()),
        }

        return {
            "overall": overall,
            "by_horizon": by_horizon,
            "by_phase": by_phase,
            "worst_locations": worst_locations,
            "best_locations": best_locations,
            "worst_segments": worst_segments,
            "date_range": date_range,
            "n_locations": merged["location"].nunique(),
        }

    # ------------------------------------------------------------------ M2

    # Action catalog (Agent 2's tool box). Each entry includes a JSON
    # schema for the params plus guardrails enforced before any change
    # touches the pipeline. The LLM may only emit actions whose `name`
    # appears here.
    ACTION_CATALOG: List[Dict[str, Any]] = [
        {
            "name": "adjust_hyperparameter",
            "description": (
                "Modify a single XGBoost hyperparameter on the active model. "
                "Use this to tune capacity (max_depth, n_estimators, learning_rate) "
                "or regularization (reg_alpha, reg_lambda, subsample, min_child_weight)."
            ),
            "params_schema": {
                "name": {
                    "type": "string",
                    "enum": [
                        "max_depth", "learning_rate", "n_estimators",
                        "subsample", "colsample_bytree", "min_child_weight",
                        "reg_alpha", "reg_lambda", "gamma",
                    ],
                },
                "value": {"type": "number"},
            },
            "guardrails": {
                "max_depth": (2, 10),
                "learning_rate": (0.005, 0.5),
                "n_estimators": (50, 3000),
                "subsample": (0.3, 1.0),
                "colsample_bytree": (0.3, 1.0),
                "min_child_weight": (1, 100),
                "reg_alpha": (0.0, 20.0),
                "reg_lambda": (0.0, 20.0),
                "gamma": (0.0, 10.0),
            },
        },
        {
            "name": "reweight_training_samples",
            "description": (
                "Upweight a slice of the training data by phase, horizon, location, "
                "or approaching_peak. Use this when one segment is consistently "
                "underperforming and you want the next training run to pay more "
                "attention to it. dimension='approaching_peak' upweights the rows in "
                "the K weeks before each past season's observed peak (value = K, an "
                "integer number of weeks; weight = lambda). Use it when the model "
                "under-predicts the peak; the current season is never labeled."
            ),
            "params_schema": {
                "dimension": {
                    "type": "string",
                    "enum": ["phase", "horizon", "location", "approaching_peak"],
                },
                "value": {"type": "string"},
                "weight": {"type": "number"},
            },
            "guardrails": {
                "weight": (1.0, 5.0),
                "phase_values": ["onset", "peak", "decline"],
                "horizon_values": ["1", "2", "3", "4"],
                # value = weeks_before for dimension='approaching_peak'
                "approaching_peak_weeks": (2, 12),
            },
        },
        {
            "name": "toggle_feature",
            "description": (
                "Enable or disable an entire feature group at training time. "
                "Useful when a group seems to add noise (disable) or when an "
                "important signal is missing (enable a previously-disabled group)."
            ),
            "params_schema": {
                "feature_group": {
                    "type": "string",
                    "enum": [
                        "lag", "rolling", "yoy", "national_context", "interactions",
                    ],
                },
                "enabled": {"type": "boolean"},
            },
            "guardrails": {},
        },
        {
            "name": "adjust_floor_constraint",
            "description": (
                "Change the post-prediction floor percentage. Lowering it lets "
                "the model predict steeper drops; raising it stabilizes during "
                "noisy periods."
            ),
            "params_schema": {
                "floor_pct": {"type": "number"},
            },
            "guardrails": {
                "floor_pct": (0.0, 0.6),
            },
        },
        {
            "name": "change_target_transform",
            "description": (
                "Switch the target transform between log/raw/sqrt. log compresses "
                "the upper tail (good for skewed counts but can under-predict peaks); "
                "raw is uncompressed; sqrt is a middle ground."
            ),
            "params_schema": {
                "transform": {"type": "string", "enum": ["log", "raw", "sqrt"]},
            },
            "guardrails": {},
        },
        {
            "name": "set_training_window",
            "description": (
                "Restrict training to the last N weeks of ROWS per location, applied "
                "after feature engineering so lag/rolling/yoy features stay intact "
                "(they still look back over the full history). Rule: ~12 weeks on a "
                "sharp takeoff (rely less on seasonality, stay agile); 52 or 104 weeks "
                "in a lull (draw on seasonal history). weeks=null restores the default "
                "of training on all rows."
            ),
            "params_schema": {
                "weeks": {"type": "integer|null"},
            },
            "guardrails": {
                # (min, max) weeks, filled by get_available_actions() from
                # training_window_guardrail(); None here because the min is
                # the pipeline's rule and src.pipeline is a lazy import.
                "weeks": None,
            },
        },
        {
            "name": "stop",
            "description": (
                "Declare convergence. Use this when no further action is "
                "expected to improve the target metric."
            ),
            "params_schema": {},
            "guardrails": {},
        },
    ]

    @staticmethod
    def model_family_of(config: Optional[Dict[str, Any]]) -> str:
        """Active model family in a config (legacy default when absent)."""
        from src.config import DEFAULT_MODEL_FAMILY

        return ((config or {}).get("model") or {}).get("family") or DEFAULT_MODEL_FAMILY

    def get_available_actions(
        self, config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Return the action catalog the LLM is allowed to choose from.

        Each entry has: name, description, params_schema, guardrails.
        For the legacy XGBoost family this is the hand-written catalog
        above. For any model-bank family the catalog is generated from the
        family's `param_space()`: one `adjust_hyperparameter` action whose
        allowed names and guardrails are exactly what the model declared,
        plus `stop`. The pipeline-specific actions (feature toggles,
        reweighting, floor, target transform) only apply to the legacy path.
        """
        family = self.model_family_of(config)
        if family == LEGACY_MODEL_FAMILY:
            return [self._resolve_catalog_entry(a) for a in self.ACTION_CATALOG]

        from src.model_bank.registry import resolve_family

        model_cls = resolve_family(family)
        space = model_cls.param_space()
        catalog: List[Dict[str, Any]] = []
        if space:
            guardrails: Dict[str, Any] = {}
            for name, spec in space.items():
                if spec.kind in ("int", "float"):
                    guardrails[name] = (spec.low, spec.high)
                elif spec.kind == "categorical":
                    guardrails[name] = list(spec.choices or ())
                else:
                    guardrails[name] = [True, False]
            catalog.append({
                "name": "adjust_hyperparameter",
                "description": (
                    f"Modify a single hyperparameter of the active {family} model "
                    f"({model_cls.description}). Allowed names and bounds come from "
                    f"the model's declared parameter space."
                ),
                "params_schema": {
                    "name": {"type": "string", "enum": list(space)},
                    "value": {"type": "number|string|boolean"},
                },
                "guardrails": guardrails,
            })
        catalog.append(next(a for a in self.ACTION_CATALOG if a["name"] == "stop"))
        return catalog

    @staticmethod
    def _resolve_catalog_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Copy a catalog entry with lazily-resolved guardrails filled in."""
        if entry["name"] != "set_training_window":
            return entry
        resolved = dict(entry)
        resolved["guardrails"] = {"weeks": training_window_guardrail()}
        return resolved

    def apply_action(
        self,
        action: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply one validated action to a config dict.

        Pure function: returns a *new* config dict (does not mutate input).
        Validates the action's name + params against the catalog and the
        guardrails. Raises ValueError on any violation so the orchestrator's
        validate node can route to the repair path.

        Args:
            action: dict with shape {"name": str, "params": dict, ...}
            config: starting config dict; uses get_default_config() if None

        Returns:
            (new_config, human_readable_change_description)
        """
        # Lazy import to avoid pulling src/* into module load time
        from src.config import (
            get_default_config,
            set_config_value,
            get_config_value,
        )

        if config is None:
            config = get_default_config()

        if not isinstance(action, dict) or "name" not in action:
            raise ValueError(f"Invalid action shape: {action!r}")

        name = action["name"]
        params = action.get("params") or {}

        catalog = {a["name"]: a for a in self.ACTION_CATALOG}
        if name not in catalog:
            raise ValueError(
                f"Unknown action: {name!r}. Allowed: {sorted(catalog)}"
            )
        spec = catalog[name]

        # ---- per-action handling ------------------------------------------------
        if name == "stop":
            return config, "stop (no change)"

        family = self.model_family_of(config)
        if family != LEGACY_MODEL_FAMILY:
            return self._apply_bank_action(name, params, config, family)

        if name == "adjust_hyperparameter":
            self._require_keys(params, ["name", "value"])
            hp_name = params["name"]
            hp_value = params["value"]
            allowed = spec["params_schema"]["name"]["enum"]
            if hp_name not in allowed:
                raise ValueError(
                    f"adjust_hyperparameter: '{hp_name}' not in allowed set {allowed}"
                )
            lo, hi = spec["guardrails"][hp_name]
            if not (lo <= hp_value <= hi):
                raise ValueError(
                    f"adjust_hyperparameter: {hp_name}={hp_value} outside guardrail [{lo}, {hi}]"
                )
            # n_estimators must be int
            if hp_name == "n_estimators":
                hp_value = int(hp_value)
            old = get_config_value(config, f"xgboost.{hp_name}")
            new_cfg = set_config_value(config, f"xgboost.{hp_name}", hp_value)
            return new_cfg, f"xgboost.{hp_name}: {old} -> {hp_value}"

        if name == "reweight_training_samples":
            self._require_keys(params, ["dimension", "value", "weight"])
            dim = params["dimension"]
            val = str(params["value"])
            weight = float(params["weight"])

            allowed_dims = set(spec["params_schema"]["dimension"]["enum"])
            if dim not in allowed_dims:
                raise ValueError(
                    f"reweight: dimension must be one of {sorted(allowed_dims)}, got {dim!r}"
                )
            wlo, whi = spec["guardrails"]["weight"]
            if not (wlo <= weight <= whi):
                raise ValueError(f"reweight: weight={weight} outside [{wlo}, {whi}]")

            if dim == "approaching_peak":
                # value = K weeks before the season peak; stored as a dict so
                # src.direct_forecast._compute_sample_weights reads it directly.
                try:
                    weeks_before = int(val)
                except ValueError:
                    raise ValueError(
                        f"reweight: approaching_peak value must be an integer number "
                        f"of weeks, got {params['value']!r}"
                    ) from None
                klo, khi = spec["guardrails"]["approaching_peak_weeks"]
                if not (klo <= weeks_before <= khi):
                    raise ValueError(
                        f"reweight: approaching_peak weeks_before={weeks_before} "
                        f"outside [{klo}, {khi}]"
                    )
                section = "sample_weights.approaching_peak"
                new_cfg = set_config_value(
                    config, section, {"weeks_before": weeks_before, "weight": weight}
                )
                return new_cfg, f"{section} = {{weeks_before: {weeks_before}, weight: {weight}}}"
            if dim == "phase" and val not in spec["guardrails"]["phase_values"]:
                raise ValueError(
                    f"reweight: phase value must be one of {spec['guardrails']['phase_values']}"
                )
            if dim == "horizon" and val not in spec["guardrails"]["horizon_values"]:
                raise ValueError(
                    f"reweight: horizon value must be one of {spec['guardrails']['horizon_values']}"
                )

            section = f"sample_weights.by_{dim}"
            existing = dict(get_config_value(config, section) or {})
            existing[val] = weight
            new_cfg = set_config_value(config, section, existing)
            return new_cfg, f"{section}[{val}] = {weight}"

        if name == "toggle_feature":
            self._require_keys(params, ["feature_group", "enabled"])
            group = params["feature_group"]
            enabled = bool(params["enabled"])
            allowed = spec["params_schema"]["feature_group"]["enum"]
            if group not in allowed:
                raise ValueError(f"toggle_feature: group must be in {allowed}")
            old = get_config_value(config, f"features.groups_enabled.{group}")
            new_cfg = set_config_value(
                config, f"features.groups_enabled.{group}", enabled
            )
            return new_cfg, f"features.groups_enabled.{group}: {old} -> {enabled}"

        if name == "adjust_floor_constraint":
            self._require_keys(params, ["floor_pct"])
            val = float(params["floor_pct"])
            lo, hi = spec["guardrails"]["floor_pct"]
            if not (lo <= val <= hi):
                raise ValueError(f"floor_pct={val} outside [{lo}, {hi}]")
            old = get_config_value(config, "floor.floor_pct")
            new_cfg = set_config_value(config, "floor.floor_pct", val)
            return new_cfg, f"floor.floor_pct: {old} -> {val}"

        if name == "change_target_transform":
            self._require_keys(params, ["transform"])
            t = params["transform"]
            allowed = spec["params_schema"]["transform"]["enum"]
            if t not in allowed:
                raise ValueError(f"change_target_transform: must be in {allowed}")
            old = get_config_value(config, "target.mode")
            new_cfg = set_config_value(config, "target.mode", t)
            return new_cfg, f"target.mode: {old} -> {t}"

        if name == "set_training_window":
            self._require_keys(params, ["weeks"])
            if params["weeks"] is None:
                old = get_config_value(config, "data.train_window_weeks")
                new_cfg = set_config_value(config, "data.train_window_weeks", None)
                return new_cfg, f"data.train_window_weeks: {old} -> None (all rows)"
            weeks = self._coerce_weeks(params["weeks"])
            lo, hi = training_window_guardrail()
            if not (lo <= weeks <= hi):
                raise ValueError(
                    f"set_training_window: weeks={weeks} outside guardrail [{lo}, {hi}] "
                    f"(minimum {lo} = FORECAST_HORIZON + 4, maximum {hi} = 2 seasons)"
                )
            new_cfg = set_config_value(config, "data.train_window_weeks", weeks)
            return new_cfg, f"data.train_window_weeks = {weeks}"

        # Defensive - should be unreachable due to catalog check above
        raise ValueError(f"Unhandled action: {name}")

    def _apply_bank_action(
        self,
        name: str,
        params: Dict[str, Any],
        config: Dict[str, Any],
        family: str,
    ) -> Tuple[Dict[str, Any], str]:
        """apply_action for model-bank families: only param edits are meaningful."""
        from src.config import get_config_value, set_config_value
        from src.model_bank.contract import ModelBankError
        from src.model_bank.registry import resolve_family

        if name != "adjust_hyperparameter":
            raise ValueError(
                f"action {name!r} is only available for the {LEGACY_MODEL_FAMILY} family; "
                f"active family {family!r} supports: adjust_hyperparameter, stop"
            )
        self._require_keys(params, ["name", "value"])
        space = resolve_family(family).param_space()
        hp_name = params["name"]
        if hp_name not in space:
            raise ValueError(
                f"adjust_hyperparameter: {hp_name!r} not tunable for {family}; "
                f"allowed: {sorted(space)}"
            )
        try:
            hp_value = space[hp_name].validate(params["value"])
        except ModelBankError as e:
            raise ValueError(f"adjust_hyperparameter: {e}") from e
        model_params = dict(get_config_value(config, "model.params") or {})
        old = model_params.get(hp_name, "<default>")
        model_params[hp_name] = hp_value
        new_cfg = set_config_value(config, "model.params", model_params)
        return new_cfg, f"model.params.{hp_name}: {old} -> {hp_value}"

    @staticmethod
    def _require_keys(params: Dict[str, Any], keys: List[str]) -> None:
        missing = [k for k in keys if k not in params]
        if missing:
            raise ValueError(f"Action missing required params: {missing}")

    @staticmethod
    def _coerce_weeks(value: Any) -> int:
        """`weeks` for set_training_window as an int, from the shapes an LLM emits.

        Accepts an int, or a float / string that is exactly an integer ("12",
        12.0), the same leniency `reweight_training_samples` gives its `value`.
        Rejects bool (True is an int in Python but never a week count),
        non-integer floats and non-numeric strings, naming the value.
        """
        if isinstance(value, bool):
            raise ValueError(
                f"set_training_window: weeks must be an int number of weeks or null, got {value!r}"
            )
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(
                f"set_training_window: weeks must be a whole number of weeks, got {value!r}"
            )
        if isinstance(value, str):
            try:
                as_float = float(value.strip())
            except ValueError:
                raise ValueError(
                    f"set_training_window: weeks must be an int number of weeks or null, "
                    f"got {value!r}"
                ) from None
            if as_float.is_integer():
                return int(as_float)
            raise ValueError(
                f"set_training_window: weeks must be a whole number of weeks, got {value!r}"
            )
        raise ValueError(
            "set_training_window: weeks must be an int number of weeks or null, "
            f"got {type(value).__name__}: {value!r}"
        )

    def run_pipeline(
        self,
        config: Dict[str, Any],
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> str:
        """Train + forecast using the agent loop's callable pipeline.

        Args:
            config: Pipeline config dict (shape from src.config.get_default_config())
            output_path: Where to write the forecast CSV. None -> default location.
            verbose: Pass-through to src.pipeline.run_pipeline.

        Returns:
            String path to the written forecast CSV.
        """
        from src.pipeline import run_pipeline as _run_pipeline

        out = _run_pipeline(config=config, output_path=output_path, verbose=verbose)
        return str(out)

    # ------------------------------------------------------------------

    _XGBOOST_MODEL_CONTEXT = (
        "You are analyzing an XGBoost-based influenza hospitalization forecasting model. "
        "The model uses a Direct Forecasting Ensemble: 4 independent XGBoost models, "
        "one per forecast horizon (1-4 weeks ahead). It predicts weekly hospital "
        "admissions for all US states and territories using CDC FluSight surveillance data.\n\n"
        "Key model details:\n"
        "- 59 engineered features: temporal indicators, lag values (1-52 weeks), "
        "rolling statistics, year-over-year comparisons, season severity, "
        "rate of change, US national context, and feature interactions\n"
        "- Target transform: log1p(hospitalizations) during training, expm1() at inference\n"
        "- Post-prediction floor constraint: predictions can't drop below 30% of last known value "
        "(decays 5 percentage points per horizon)\n"
        "- Regularized XGBoost: max_depth=3, subsample=0.7, L1/L2 regularization\n\n"
    )

    def _bank_model_context(self, family: str, config: Dict[str, Any]) -> str:
        from src.model_bank.registry import resolve_family

        model_cls = resolve_family(family)
        space = model_cls.param_space()
        active = {**model_cls.default_params(), **((config.get("model") or {}).get("params") or {})}
        lines = [
            f"You are analyzing the '{family}' influenza hospitalization forecasting model: "
            f"{model_cls.description}. It predicts weekly hospital admissions 1-4 weeks ahead "
            "for all US states and territories using CDC FluSight surveillance data.",
            "",
            "Tunable hyperparameters (current value, allowed range):",
        ]
        if not space:
            lines.append("- (none declared: this model exposes no knobs; only `stop` is useful)")
        for name, spec in space.items():
            bound = (
                f"[{spec.low}, {spec.high}]" if spec.kind in ("int", "float")
                else f"one of {list(spec.choices or ())}" if spec.kind == "categorical"
                else "true/false"
            )
            desc = f" - {spec.description}" if spec.description else ""
            lines.append(f"- {name} = {active.get(name, '?')} ({spec.kind}, {bound}){desc}")
        lines.append("")
        return "\n".join(lines)

    def get_domain_context(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Return flu forecasting domain context for the LLM prompt.

        The model-specific paragraph depends on the active family; everything
        shared by every model (epidemic phases, performance conventions, the
        lab's guardrails) comes from the knowledge bank as a KNOWN FACTS block,
        filtered to entries that apply to the active family. The query limit
        (`DEFAULT_QUERY_LIMIT`) keeps the block bounded as the bank grows; when
        it truncates, the block ends with an explicit omitted-count line. The
        phase/metric-aware retrieval at each proposal step lives in the
        orchestrator (design section 6, unit 4), not here.

        With `knowledge_enabled=False` (`improve --no-knowledge`) only the model
        paragraph is returned and the bank is never opened.
        """
        family = self.model_family_of(config)
        if family == LEGACY_MODEL_FAMILY:
            model_context = self._XGBOOST_MODEL_CONTEXT
        else:
            model_context = self._bank_model_context(family, config or {})
        if not self.knowledge_enabled:
            return model_context
        # The former "Epidemic phases" / "Performance context" bullets now live in
        # knowledge/curated/domain-context.yaml (migrated 2026-09-30).
        ctx = RetrievalContext(model=family)
        facts = self.knowledge_bank.query(ctx)
        omitted = self.knowledge_bank.count_matching(ctx) - len(facts)
        return model_context + render_known_facts(facts, omitted=omitted)
