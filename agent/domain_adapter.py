"""Abstract base class for domain-specific adapters.

Each domain (flu forecasting, finance, sales, etc.) implements this interface
to plug into the agentic framework. The framework's orchestrator, LLM client,
and evaluation tools are domain-agnostic — all domain knowledge lives here.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import pandas as pd


class DomainAdapter(ABC):
    """Interface that each domain must implement to use the agentic framework.

    Milestone 1 methods (abstract — must implement):
        load_data: Load forecasts and ground truth
        compute_metrics: Compute domain-specific evaluation metrics
        get_domain_context: Provide LLM prompt context about the domain

    Milestone 2+ methods (concrete stubs — override when ready):
        get_available_actions: Define what the LLM can suggest
        apply_action: Programmatically apply an LLM suggestion
        run_pipeline: Retrain and re-forecast
    """

    @abstractmethod
    def load_data(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load model forecasts and ground truth actuals.

        Args:
            config: Dict with keys like 'forecast_csv', 'cutoff_date', etc.

        Returns:
            Tuple of (forecasts_df, actuals_df)
        """
        ...

    @abstractmethod
    def compute_metrics(
        self, forecasts: pd.DataFrame, actuals: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute domain-specific evaluation metrics.

        Should return a structured dict with at minimum:
            - "overall": dict with aggregate metrics
            - "worst_locations": list of worst-performing entities
            - "best_locations": list of best-performing entities

        The returned dict is serialized to JSON and sent to the LLM.
        Pre-compute everything in Python — don't rely on the LLM to do math.

        Args:
            forecasts: Forecast DataFrame
            actuals: Ground truth DataFrame

        Returns:
            Dict of structured metrics (must be JSON-serializable)
        """
        ...

    @abstractmethod
    def get_domain_context(self) -> str:
        """Return a string describing the domain for LLM prompt context.

        Should include: what the model predicts, how it works, key features,
        known limitations, and what "good" vs "bad" performance looks like.
        """
        ...

    # --- Milestone 2+ stubs (override when implementing improvement loop) ---

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Return the constrained set of actions the LLM can suggest.

        Each action is a dict with at minimum:
            - "name": str (action identifier)
            - "description": str (what it does)
            - "params": list of parameter names

        The LLM can ONLY suggest actions from this list.
        """
        raise NotImplementedError(
            "get_available_actions() is not implemented yet (Milestone 2). "
            "Override this method in your adapter to enable the improvement loop."
        )

    def apply_action(self, action: Dict[str, Any]) -> str:
        """Programmatically apply one LLM-suggested action.

        Args:
            action: Dict with "name" and action-specific parameters

        Returns:
            Human-readable description of what was changed
        """
        raise NotImplementedError(
            "apply_action() is not implemented yet (Milestone 2). "
            "Override this method in your adapter to enable the improvement loop."
        )

    def run_pipeline(self, config: Dict[str, Any]) -> str:
        """Retrain the model and generate new forecasts.

        Args:
            config: Pipeline configuration (may have been modified by apply_action)

        Returns:
            Path to the new forecast output file
        """
        raise NotImplementedError(
            "run_pipeline() is not implemented yet (Milestone 2). "
            "Override this method in your adapter to enable the improvement loop."
        )
