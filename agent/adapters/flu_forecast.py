"""Flu forecasting adapter for the agentic framework.

Wraps the existing XGBoost-based influenza hospitalization forecasting pipeline
to provide domain-specific data loading, metrics computation, and LLM context.
"""

import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.domain_adapter import DomainAdapter
from agent.phase_evaluator import PhaseEvaluator


class FluForecastAdapter(DomainAdapter):
    """Adapter for the CDC FluSight influenza hospitalization forecasting model.

    Wraps:
        - src.data_loader.FluDataLoader (actuals)
        - src.evaluate.ModelEvaluator (baseline metrics)
        - agent.phase_evaluator.PhaseEvaluator (phase-aware metrics)
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self._location_names = None  # lazy-loaded FIPS-to-name mapping

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

    def get_domain_context(self) -> str:
        """Return flu forecasting domain context for the LLM prompt."""
        return (
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
            "Epidemic phases:\n"
            "- Onset (Oct-Nov): flu activity begins rising\n"
            "- Peak (Dec-Jan): highest hospitalization rates\n"
            "- Decline (Feb-Apr): activity decreasing\n\n"
            "Performance context:\n"
            "- MAPE < 40% is good, 40-60% is acceptable, > 60% indicates problems\n"
            "- Error typically increases with horizon (Week 1 best, Week 4 worst)\n"
            "- Peak periods are hardest to forecast due to rapid changes\n"
            "- Positive bias = model over-predicts; negative bias = model under-predicts"
        )
