"""
Agentic Forecasting Framework

An LLM-powered system that analyzes forecasting model outputs,
diagnoses performance issues, and suggests improvements.

Usage:
    python -m agent check-data --cutoff-date 2024-11-02 --dry-run
    python -m agent summarize --forecast-csv outputs/quantile_hindcasts/point_forecasts_nov_apr.csv --dry-run
    python -m agent improve --cutoff-date 2024-11-02 --auto-apply
    python -m agent history
"""

from agent.domain_adapter import DomainAdapter
from agent.phase_evaluator import PhaseEvaluator
from agent.llm_client import LLMClient
from agent.data_quality import DataQualityChecker, DataQualityReport

__all__ = ["DomainAdapter", "PhaseEvaluator", "LLMClient", "DataQualityChecker", "DataQualityReport"]
