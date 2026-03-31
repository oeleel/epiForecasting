"""
Agentic Forecasting Framework

An LLM-powered system that analyzes forecasting model outputs,
diagnoses performance issues, and suggests improvements.

Usage:
    python -m agent summarize --forecast-csv outputs/quantile_hindcasts/point_forecasts_nov_apr.csv --dry-run
    python -m agent summarize --cutoff-date 2024-11-02 --base-url http://localhost:11434/v1
"""

from agent.domain_adapter import DomainAdapter
from agent.phase_evaluator import PhaseEvaluator
from agent.llm_client import LLMClient

__all__ = ["DomainAdapter", "PhaseEvaluator", "LLMClient"]
