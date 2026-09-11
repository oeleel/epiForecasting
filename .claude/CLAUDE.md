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
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r documentation/requirements.txt pytest langchain-openai
uv pip install --python .venv/bin/python -r documentation/requirements-nixtla.txt   # optional example families

# macOS only: xgboost + torch bundle separate libomp.dylib copies; running both
# in one process (agent select-model / improve does) segfaults without this
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
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

# Model bank (Stage 1): list families, compare them on the pinned split, refine the winner
python -m agent list-models
python -m agent select-model --stride-weeks 4 --exclude-locations US --json outputs/model_selection/run.json
python -m agent select-model --families persistence my_lab.models:FluLSTM --metric wis --phase peak
python -m agent improve --cutoff-date 2025-12-06 --model-family mlf_lightgbm --auto-apply

# Goal in plain English (LOW priority per advisor 09-10 - researchers write the JSON spec directly)
python -m agent select-model --goal "which model is best at the peak" --explain-goal

# End-of-run report: written automatically after every improve run; regenerate for a past run
python -m agent report <run_id> [--rebuild] [--out report.md]
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
```bash
.venv/bin/python -m pytest tests/agent/ -q -W error::DeprecationWarning   # 186 tests
PYTHONPATH=. .venv/bin/python tests/agent/run_all.py                        # no-pytest fallback
```
Covers: adapter actions (legacy + bank families), config, data quality, feature toggle, orchestrator, prompts, run tracker, sample weights, WIS scoring, model bank (contract/registry/bridge/baselines/runner), model selection, run report, goal parser, phase segmentation.

### Key Config Details
- Active XGBoost params: `XGBOOST_PARAMS_V2` (regularized) in `src/config.py`
- Feature version: `v3` — pruned via SHAP analysis; removed/added features listed in `src/config.py`
- Quantile levels: `[0.05, 0.25, 0.5, 0.75, 0.95]`
- Pinned split (2026-09-02): `TRAIN_START_DATE` 2022-02-05, eval `EVAL_START_DATE`..`EVAL_END_DATE` = 2025-10-01..2026-05-31; `generate_eval_cutoffs(stride_weeks)`
- Model family: `config["model"]["family"]` (default `xgboost_direct` = legacy path); family params in `config["model"]["params"]`
- Location clustering: 5 clusters, min 3 locations per cluster
- Agent dependencies (langchain-openai) are not in `documentation/requirements.txt` — install separately for agent work

## Architecture

### Agent Framework (`agent/`)

```
agent/
├── __init__.py              # Package exports
├── __main__.py              # Entry point: python -m agent
├── domain_adapter.py        # Abstract base class (DomainAdapter ABC)
├── data_quality.py          # Pre-training data quality checks + LLM prompt
├── orchestrator.py          # Two-agent improvement loop (evaluate → diagnose → act → retrain)
├── run_tracker.py           # SQLite-backed run history (outputs/agent_runs/runs.db)
├── phase_evaluator.py       # Phase-aware evaluation engine (stateless class methods)
├── prompt_templates.py      # LLM prompt templates + formatter + validators
├── llm_client.py            # Provider-agnostic LLM wrapper (OpenAI-compatible API)
├── model_selection.py       # Stage 1: rolling warm-up of bank families + incumbent selection
├── run_report.py            # End-of-run report (report.md + report.json per improve run; `agent report`)
├── goal_parser.py           # Natural-language goal -> SelectionGoal (`select-model --goal`); low priority
├── phase_segmentation.py    # Curve-based phase labels (Adiga surge/plateau/decline), alternative to calendar phases
├── cli.py                   # CLI: check-data, summarize, improve, history, status, compare, report, list-models, select-model
└── adapters/
    ├── __init__.py
    └── flu_forecast.py      # Concrete adapter for flu forecasting
```

**DomainAdapter ABC** (`domain_adapter.py`) — every domain implements:
| Method | Status | Purpose |
|--------|--------|---------|
| `load_data(config)` | Implemented | Load forecast outputs + ground truth |
| `compute_metrics(forecasts, actuals)` | Implemented | Domain-specific evaluation metrics |
| `get_domain_context(config)` | Implemented | Provide LLM with domain knowledge (describes the active model family) |
| `get_available_actions(config)` | Implemented | Constrained action catalog; generated from `param_space()` for bank families |
| `apply_action(action)` | Implemented | Apply + validate an LLM-suggested action |
| `run_pipeline(config)` | Implemented | Retrain and re-forecast from a config dict |

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
| Laptop (dev) | Ollama | Qwen 3 8B (`qwen3:8b`, the code default) | Fast iteration, testing |
| UVA Rivanna | vLLM on A100 GPU | Qwen 2.5 72B (AWQ quantized) | Full-quality analysis |

Both expose an OpenAI-compatible API (`/v1/chat/completions`). Swapping environments = changing one URL.

### Milestone Status

**Milestone 1: LLM Summarization** — COMPLETE
- `python -m agent summarize` computes phase-aware metrics and generates natural language performance report

**Milestone 2: Improvement Loop** — COMPLETE
- Two-agent loop: evaluate -> Agent 1 diagnoses -> Agent 2 proposes action -> validate -> apply -> retrain -> re-evaluate
- `agent/orchestrator.py` — plain Python orchestrator (no LangGraph dependency)
- `agent/run_tracker.py` — SQLite-backed iteration history
- `src/pipeline.py` — callable pipeline (config dict in, forecast CSV out)
- All adapter methods implemented: `get_available_actions()`, `apply_action()`, `run_pipeline()`
- Sample weight support wired through all ensemble types
- Feature group toggling wired through `FeatureEngineer`
- 6 constrained actions: `adjust_hyperparameter`, `reweight_training_samples`, `toggle_feature`, `adjust_floor_constraint`, `change_target_transform`, `stop`

**Data Quality Agent** — COMPLETE
- `python -m agent check-data` runs pre-training checks (missing weeks, nulls, spikes, zero-reporting, coverage)
- LLM interprets findings in epidemiological context

**Milestone 3: Tracking & History** — COMPLETE
- CLI commands: `history`, `status`, `compare`
- Sample reweighting by phase/horizon/location via `reweight_training_samples` action

**Stage 1: Model bank + selection (TS-Agent Stage 1)** — COMPLETE (2026-09-02)
- `src/model_bank/` — `ForecastModel` contract, registry (`@register` or dotted path `pkg.mod:Class`), CDC<->long data bridge, runner
- 9 families: 2 baselines, the in-repo XGBoost + PyTorch ensembles, 5 Nixtla examples (optional deps)
- `agent/model_selection.py` + `select-model` CLI: rolling-origin warm-up on the pinned split, phase-aware WIS, incumbent by `SelectionGoal(metric, phase)`
- `improve --model-family X`: the loop refines any family; `adjust_hyperparameter` comes from the family's `param_space()`
- **In-house lab models are the real bank; Nixtla is example scaffolding.** See `documentation/MODEL_BANK.md` for the 30-line plug-in recipe.

**Run report, NL goal, curve-based phases** — LANDED (2026-09-11)
- `agent/run_report.py` (roadmap 6.1): every `improve` run writes `report.md` + `report.json`; `agent report <run_id>` regenerates. Never claims an improvement across different evaluation windows.
- `agent/goal_parser.py` (roadmap 2.3): `select-model --goal "..."`. Advisor (09-10) rated the English front end cosmetic - do not invest further; the structured spec still matters.
- `agent/phase_segmentation.py`: Adiga-style surge/plateau/decline segmentation from the curve, alongside the calendar phases.
- Open review findings on the report + goal parser: `documentation/handoff-2026-09-11-desktop.md` §3.
- **Current priority (advisor 09-10): knowledge bank first** - `documentation/meeting-notes/2026-09-10-knowledge-bank-first.md`.

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
- **`src/model_bank/`** — model contract + registry + runner; `src/pipeline.run_pipeline` dispatches here for every family except `xgboost_direct`
- **`src/location_clustering.py`** — `LocationClusterer`: groups similar states for cluster-specific models
- **`src/visualization.py`** — Plotting utilities (Plotly, Matplotlib)

Key details:
- `TARGET_MODE = "log"` — targets are `log1p()` transformed; `expm1()` at inference
- Floor constraint: predictions can't drop below 30% of last known value (decays per horizon)
- Feature engineering: 10 passes in fixed order, forward-fill then zero-fill for missing values

### Other Directories

- **`scripts/evaluation/`** — Feature importance and regularization evaluation scripts (`evaluate_features.py`, `evaluate_regularization.py`, `evaluate_target_transform.py`)
- **`analysis/`** — SHAP feature importance analysis (`shap_analysis.py`) and Plotly performance dashboard (`performance_dashboard.py`)
- **`tests/agent/`** — the real test suite (pytest or `run_all.py`); `tests/test.py` is a legacy placeholder

## Data Source

CDC FluSight Repository: weekly influenza hospitalization data fetched from GitHub CSV URL defined in `CDC_DATA_URL` in `src/config.py`. Cached locally at `data/raw/` with a 1-week TTL.

## Technology Stack

| Component | Technology |
|---|---|
| Forecasting model | XGBoost (primary), PyTorch NN (alternative) |
| Agent orchestration | LangGraph (Milestone 2+) |
| LLM serving | vLLM (cluster) / Ollama (laptop) |
| LLM model | Qwen 3 8B (dev) / Qwen 2.5 72B AWQ (prod) |
| LLM integration | langchain-openai (`ChatOpenAI`) |
| Hyperparameter tuning | Optuna |
| Feature importance | SHAP |
| Visualization | Plotly, Matplotlib |
| Run tracking | SQLite (Milestone 3) |
| Data source | CDC FluSight GitHub repo |
| Python | 3.11 |
