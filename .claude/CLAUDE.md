# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agentic AI framework for epidemiological forecasting** — an LLM-powered agent system that autonomously analyzes forecast outputs, diagnoses performance issues, and iteratively improves model quality. Built on top of an existing XGBoost-based influenza hospitalization forecasting pipeline.

The key idea: replace the manual "run model → inspect CSVs → tweak parameters → re-run" loop with an LLM agent that reasons over structured metrics and takes constrained improvement actions.

### Two Layers
1. **Domain-agnostic orchestration layer** — LangGraph state machine, LLM client, prompt templates, run tracker (reusable across any forecasting domain)
2. **Domain-specific adapter** — Flu forecasting adapter wrapping the existing XGBoost pipeline (`src/`)

## Commands

### Setup
```bash
pip install -r documentation/requirements.txt
```

### Agent Framework (primary focus)
```bash
# Compute metrics only (no LLM needed)
python -m agent summarize --forecast-csv <path> --dry-run

# Full analysis with LLM
python -m agent summarize --forecast-csv <path>

# Specify LLM server (vLLM on Rivanna or Ollama locally)
python -m agent summarize --forecast-csv <path> --base-url http://localhost:8247/v1

# Debug: see the prompt sent to the LLM
python -m agent summarize --forecast-csv <path> --verbose
```

### Forecasting Pipeline (underlying model)
```bash
# Full pipeline (train + forecast)
python scripts/pipeline.py --cutoff-date 2024-11-02

# Specific locations only
python scripts/pipeline.py --cutoff-date 2024-11-02 --locations US 06 12 48

# Hyperparameter optimization (Optuna)
python scripts/optimize.py --n-trials 50

# Batch hindcast Nov 2024-Apr 2025
python scripts/generate_forecasts_nov_apr.py

# Refresh local CDC data cache
python -m src.data_loader update
```

### Testing
No test suite exists yet. `tests/test.py` is a placeholder.

### Key Config Details
- Active XGBoost params: `XGBOOST_PARAMS_V2` (regularized) in `src/config.py`
- Feature version: `v3` — pruned via SHAP analysis; removed/added features listed in `src/config.py`
- Quantile levels: `[0.05, 0.25, 0.5, 0.75, 0.95]`
- Location clustering: 5 clusters, min 3 locations per cluster
- Agent dependencies (langchain-openai) are not in `documentation/requirements.txt` — install separately for agent work

## Architecture

### Agent Framework (`agent/`)

```
agent/
├── __init__.py              # Package exports
├── __main__.py              # Entry point: python -m agent
├── domain_adapter.py        # Abstract base class (DomainAdapter ABC)
├── phase_evaluator.py       # Phase-aware evaluation engine (stateless class methods)
├── prompt_templates.py      # LLM prompt templates + formatter
├── llm_client.py            # Provider-agnostic LLM wrapper (OpenAI-compatible API)
├── cli.py                   # CLI: summarize command (improve command planned)
└── adapters/
    ├── __init__.py
    └── flu_forecast.py      # Concrete adapter for flu forecasting
```

**DomainAdapter ABC** (`domain_adapter.py`) — every domain implements:
| Method | Status | Purpose |
|--------|--------|---------|
| `load_data(config)` | Implemented | Load forecast outputs + ground truth |
| `compute_metrics(forecasts, actuals)` | Implemented | Domain-specific evaluation metrics |
| `get_domain_context()` | Implemented | Provide LLM with domain knowledge |
| `get_available_actions()` | Stub (Milestone 2) | Define what the LLM can suggest |
| `apply_action(action)` | Stub (Milestone 2) | Programmatically apply a suggestion |
| `run_pipeline(config)` | Stub (Milestone 2) | Retrain and re-forecast |

**PhaseEvaluator** (`phase_evaluator.py`) — computes metrics by epidemic phase:
- Phases: Onset (Oct-Nov), Peak (Dec-Jan), Decline (Feb-Apr), Off-season (May-Sep, skipped)
- Key addition over `src/evaluate.py`: **signed bias** (over vs. under-prediction)
- Methods: `assign_phase()`, `merge_forecasts_actuals()`, `compute_overall_metrics()`, `evaluate_by_phase()`, `evaluate_by_horizon()`, `evaluate_by_location()`, `identify_worst_segments()`

**LLM Client** (`llm_client.py`) — uses `langchain-openai`'s `ChatOpenAI(base_url=...)`:
- Defaults to Ollama on localhost for development
- Override via `--base-url` or `LLM_BASE_URL` env var for vLLM on Rivanna
- Graceful degradation: prints raw metrics if LLM unreachable

### LLM Infrastructure
| Environment | Tool | Model | Purpose |
|---|---|---|---|
| Laptop (dev) | Ollama | Qwen 2.5 7B | Fast iteration, testing |
| UVA Rivanna | vLLM on A100 GPU | Qwen 2.5 72B (AWQ quantized) | Full-quality analysis |

Both expose an OpenAI-compatible API (`/v1/chat/completions`). Swapping environments = changing one URL.

### Milestone Status

**Milestone 1: LLM Summarization** — COMPLETE
- `python -m agent summarize` computes phase-aware metrics and generates natural language performance report

**Milestone 2: Improvement Loop** — NEXT
- Iterative loop: evaluate -> LLM diagnoses -> suggest action -> apply -> retrain -> re-evaluate
- New components needed:
  - `agent/orchestrator.py` — LangGraph state machine with evaluate/suggest/apply nodes
  - `agent/run_tracker.py` — SQLite-backed iteration history
  - Complete adapter stubs: `get_available_actions()`, `apply_action()`, `run_pipeline()`
  - Add `sample_weight` support to `src/model.py` and `src/direct_forecast.py`
  - New CLI command: `python -m agent improve --max-iterations 3`
- Constrained action space (LLM can only suggest from predefined list):
  - `adjust_hyperparameter` — change XGBoost params
  - `resample_by_phase` — upweight training samples from specific phases
  - `toggle_feature` — enable/disable features
  - `adjust_floor_constraint` — change post-prediction floor %
  - `change_target_transform` — switch between log/raw/ratio

**Milestone 3: Tracking & History** — PLANNED
- CLI commands: `history`, `compare`, `status`
- Boosting-style sample reweighting based on underperforming segments

**Milestone 4: Generalization** — PLANNED
- Template adapter for new domains (finance, sales, etc.)

### Underlying Forecasting Pipeline (`src/`)

The agent framework wraps this existing pipeline. It should be treated as stable infrastructure — changes here are driven by agent actions (Milestone 2), not manual edits.

```
CDC GitHub CSV -> FluDataLoader -> FeatureEngineer (59+ features) -> DirectForecastEnsemble -> Forecasts
                  (1-week cache)   (10 ordered passes)               (4 horizon-specific models)
```

- **`src/config.py`** — All hyperparameters, feature settings, validation cutoffs
- **`src/data_loader.py`** — CDC data fetch with local file cache (168h TTL)
- **`src/feature_engineering.py`** — 10 ordered feature creation passes + trend features
- **`src/direct_forecast.py`** — `DirectForecastEnsemble` (4 models, 1 per horizon), `QuantileDirectForecastEnsemble`, `ClusteredDirectForecastEnsemble`
- **`src/model.py`** — XGBoost wrapper with monotonic constraints, label encoding, quantile support
- **`src/evaluate.py`** — `ModelEvaluator`, `QuantileEvaluator`, `ClusteredEvaluator`
- **`src/train.py`** — Walk-forward validation and final model training
- **`src/predict.py`** — `FluForecastGenerator`: prediction orchestration
- **`src/nn_model.py`** — `NNQuantileDirectForecastEnsemble`: PyTorch-based quantile regression alternative
- **`src/location_clustering.py`** — `LocationClusterer`: groups similar states for cluster-specific models
- **`src/visualization.py`** — Plotting utilities (Plotly, Matplotlib)

Key details:
- `TARGET_MODE = "log"` — targets are `log1p()` transformed; `expm1()` at inference
- Floor constraint: predictions can't drop below 30% of last known value (decays per horizon)
- Feature engineering: 10 passes in fixed order, forward-fill then zero-fill for missing values

### Other Directories

- **`scripts/evaluation/`** — Feature importance and regularization evaluation scripts (`evaluate_features.py`, `evaluate_regularization.py`, `evaluate_target_transform.py`)
- **`analysis/`** — SHAP feature importance analysis (`shap_analysis.py`) and Plotly performance dashboard (`performance_dashboard.py`)
- **`tests/`** — Placeholder only; no real test suite exists

## Data Source

CDC FluSight Repository: weekly influenza hospitalization data fetched from GitHub CSV URL defined in `CDC_DATA_URL` in `src/config.py`. Cached locally at `data/raw/` with a 1-week TTL.

## Technology Stack

| Component | Technology |
|---|---|
| Forecasting model | XGBoost (primary), PyTorch NN (alternative) |
| Agent orchestration | LangGraph (Milestone 2+) |
| LLM serving | vLLM (cluster) / Ollama (laptop) |
| LLM model | Qwen 2.5 (7B dev / 72B AWQ prod) |
| LLM integration | langchain-openai (`ChatOpenAI`) |
| Hyperparameter tuning | Optuna |
| Feature importance | SHAP |
| Visualization | Plotly, Matplotlib |
| Run tracking | SQLite (Milestone 3) |
| Data source | CDC FluSight GitHub repo |
| Python | 3.11 |
