# Agentic Forecasting Framework — Research Meeting Report

> Status Report | March 2026

---

## 1. Project Vision

We are building an **LLM-powered agent system** that autonomously analyzes forecasting model outputs, diagnoses performance issues, and iteratively improves model quality — replacing the manual inspection loop that currently requires a human analyst to review forecast CSVs across 50+ states and multiple time periods.

### How It Differs from Existing Tools (e.g., Claude Code)

| | Claude Code | Our Framework |
|---|---|---|
| **What it fixes** | Code-level errors (syntax, crashes) | Output-level quality (forecasts are wrong, not broken) |
| **Reasoning domain** | General programming | Domain-aware (epidemic phases, peak sensitivity, bias patterns) |
| **Feedback loop** | One-shot fix | Iterative: evaluate → suggest → retrain → re-evaluate |
| **Decision basis** | "Does the code run?" | "Is the MAPE < 40%? Is there phase-specific bias?" |
| **Data access** | Web search, code repos | Local data only (security/privacy) |

### Why This Matters

- Manual review of forecast outputs across all states is time-consuming and error-prone
- The model may run successfully but produce poor forecasts in specific phases or locations
- Automated diagnosis enables faster iteration cycles (weekly retraining + evaluation)
- The framework is **domain-agnostic** — the same architecture works for finance, sales, or any time-series forecasting domain

---

## 2. Architecture

The system has two layers: a **domain-agnostic orchestration layer** and a **domain-specific adapter**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Domain-Agnostic Layer                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ LangGraph    │   │ LLM Client   │   │ Run Tracker       │   │
│  │ Orchestrator │   │ (vLLM/Ollama)│   │ (SQLite history)  │   │
│  └──────┬───────┘   └──────┬───────┘   └───────┬───────────┘   │
│         │                  │                    │               │
│         ▼                  ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DomainAdapter (ABC)                         │   │
│  │  load_data() | compute_metrics() | get_domain_context() │   │
│  │  get_available_actions() | apply_action() | run_pipeline│   │
│  └─────────────────────────┬───────────────────────────────┘   │
└────────────────────────────┼───────────────────────────────────┘
                             │
┌────────────────────────────┼───────────────────────────────────┐
│                    Flu Forecast Adapter                         │
│                                                                 │
│  Wraps: FluDataLoader, FeatureEngineer, DirectForecastEnsemble │
│         ModelEvaluator, PhaseEvaluator, config.py               │
│                                                                 │
│  Available Actions (Milestone 2):                               │
│  - adjust_hyperparameter (learning_rate, max_depth, etc.)       │
│  - resample_by_phase (upweight peak/takeoff/decline)            │
│  - toggle_feature (enable/disable specific features)            │
│  - adjust_floor_constraint (change post-prediction floor %)     │
│  - change_target_transform (log, raw, ratio)                    │
└─────────────────────────────────────────────────────────────────┘
```

**Key design principle**: Adding a new domain (e.g., financial forecasting) requires writing only one file — a new adapter. The orchestrator, LLM client, prompt templates, and run tracker are all reusable.

---

## 3. LLM Infrastructure

We use **self-hosted open-source models** — no cloud APIs, no API costs, full data privacy.

| Environment | Tool | Model | Purpose |
|---|---|---|---|
| Laptop (dev) | **Ollama** | Qwen 2.5 7B | Fast iteration, testing |
| UVA Rivanna | **vLLM** on A100 GPU | Qwen 2.5 72B (AWQ quantized) | Full-quality analysis |

Both expose an **OpenAI-compatible API** (`/v1/chat/completions`), so the framework code is identical regardless of where the LLM runs. Swapping from laptop to cluster = changing one URL.

**Why Qwen 2.5**: Best-in-class at structured output, tabular data analysis, and JSON generation among open-source models. The 72B AWQ-quantized variant fits on a single A100 80GB GPU.

---

## 4. What Has Been Implemented (Milestone 1)

Milestone 1 delivers **LLM-based summarization** of forecast performance. The system computes phase-aware metrics from existing forecast outputs and generates a natural language analysis report.

### Implemented Files

```
agent/
├── __init__.py              # Package exports
├── __main__.py              # Entry point: python -m agent
├── domain_adapter.py        # Abstract base class (6 methods: 3 active, 3 stubs)
├── phase_evaluator.py       # Phase-aware evaluation engine
├── prompt_templates.py      # LLM prompt template + formatter
├── llm_client.py            # Provider-agnostic LLM wrapper
├── cli.py                   # CLI: summarize command
└── adapters/
    ├── __init__.py
    └── flu_forecast.py      # Concrete adapter for flu forecasting
```

### File-by-File Details

#### `agent/domain_adapter.py` — The Core Abstraction

Defines the `DomainAdapter` abstract base class that every domain must implement:

| Method | Type | Purpose |
|---|---|---|
| `load_data(config)` | Abstract | Load forecast outputs + ground truth |
| `compute_metrics(forecasts, actuals)` | Abstract | Compute domain-specific evaluation metrics |
| `get_domain_context()` | Abstract | Provide LLM with domain knowledge |
| `get_available_actions()` | Stub (M2) | Define what the LLM can suggest |
| `apply_action(action)` | Stub (M2) | Programmatically apply a suggestion |
| `run_pipeline(config)` | Stub (M2) | Retrain and re-forecast |

Milestone 2+ methods raise `NotImplementedError` — they don't need to be implemented until the improvement loop is built.

#### `agent/phase_evaluator.py` — Phase-Aware Evaluation Engine

The central diagnostic tool. All methods are stateless class methods.

**Phase definitions** (calendar-based, adapted from existing `scripts/optimize.py:get_season_phase()`):
- **Onset**: October-November
- **Peak**: December-January
- **Decline**: February-April
- **Off-season**: May-September (skipped in evaluation)

**Key methods**:

| Method | Input | Output |
|---|---|---|
| `assign_phase(date_str)` | Date string | Phase name |
| `merge_forecasts_actuals(forecasts, actuals)` | Two DataFrames | Merged DF with error columns (signed error, abs error, pct error) + phase labels |
| `compute_overall_metrics(merged)` | Merged DF | `{mape, mae, rmse, bias, n_forecasts}` |
| `evaluate_by_phase(merged)` | Merged DF | `{onset: {mape, mae, bias, n}, peak: {...}, decline: {...}}` |
| `evaluate_by_horizon(merged)` | Merged DF | `{1: {mape, mae, bias, n}, 2: {...}, ...}` |
| `evaluate_by_location(merged)` | Merged DF | DataFrame sorted by MAPE descending |
| `identify_worst_segments(merged, n)` | Merged DF | List of N worst individual predictions |

**Critical addition over existing evaluation**: **Signed bias** (`mean(predicted - actual)`). The existing `src/evaluate.py` only computes unsigned MAE/MAPE. Knowing the model *over-predicts during peaks* vs. *under-predicts during onset* is the most actionable diagnostic signal.

#### `agent/adapters/flu_forecast.py` — Flu Forecasting Adapter

Concrete implementation connecting the abstract framework to our existing XGBoost pipeline.

**`load_data(config)`**:
- Accepts either `--forecast-csv` (explicit path) or `--cutoff-date` (auto-discovers forecast files in `outputs/`)
- Loads actuals from `data/raw/flusight_hospital_admissions.csv` (fast path) or via `FluDataLoader` (fallback)
- Includes a **FIPS-to-state-name mapping** so the LLM sees "Ohio" not "39"

**`compute_metrics(forecasts, actuals)`**:
- Merges forecasts with actuals via PhaseEvaluator
- Returns a structured dict with: overall metrics, per-horizon, per-phase, top 5 worst/best locations (with names), top 10 worst predictions, date range
- All metrics pre-computed in Python — the LLM just reasons over the results

**`get_domain_context()`**:
- Returns a description of the model for the LLM: architecture, features, target transform, floor constraints, what "good" performance looks like

#### `agent/prompt_templates.py` — Prompt Engineering

Contains the `SUMMARY_PROMPT` template and `format_summary_prompt()` function.

The prompt instructs the LLM to produce 5 sections:
1. **Overall Assessment** — 2-3 sentence summary
2. **Geographic Patterns** — which states perform well vs. poorly
3. **Temporal Patterns** — error by phase and horizon
4. **Bias Analysis** — systematic over/under-prediction patterns
5. **Areas for Investigation** — 2-3 concrete suggestions

The `format_summary_prompt()` function converts the raw metrics dict into human-readable text with proper formatting (rounded values, +/- for bias direction, etc.).

#### `agent/llm_client.py` — LLM Wrapper

Provider-agnostic wrapper using `langchain-openai`'s `ChatOpenAI(base_url=...)`.

- Defaults to Ollama on localhost (easiest development path)
- Override via `--base-url` flag or `LLM_BASE_URL` env var for vLLM on Rivanna
- **Graceful degradation**: if the LLM server is unreachable, the CLI prints raw metrics instead of crashing
- No API keys needed for local models

#### `agent/cli.py` — Engineer Interface

```bash
# Compute metrics only (no LLM needed)
python -m agent summarize --forecast-csv <path> --dry-run

# Full analysis with LLM
python -m agent summarize --forecast-csv <path>

# Specify LLM server
python -m agent summarize --forecast-csv <path> --base-url http://localhost:8247/v1

# Debug: see the prompt sent to the LLM
python -m agent summarize --forecast-csv <path> --verbose
```

---

## 5. Demo Output

Running against our existing Nov 2024 - Apr 2025 hindcast data (`--dry-run`, no LLM):

```
======================================================================
FORECAST PERFORMANCE SUMMARY
======================================================================

Evaluation period: 2024-11-09 to 2025-05-24
Locations: 53  |  Forecasts: 5512

--- Overall ---
  MAPE: 57.1%  |  MAE: 540.5  |  RMSE: 3340.7  |  Bias: -449.6 (under-predicting)

--- By Horizon ---
  Horizon        MAPE      MAE       Bias      N
  Week 1        30.6%    382.1     -323.6   1378
  Week 2        49.3%    538.8     -440.5   1378
  Week 3        67.6%    605.5     -488.8   1378
  Week 4        80.8%    635.6     -545.3   1378

--- By Epidemic Phase ---
  Phase          MAPE      MAE       Bias      N
  Onset         63.7%     64.5      -45.7    530
  Peak          47.3%    690.7     -684.3   1696
  Decline       63.3%    633.1     -459.2   2756

--- Worst 5 Locations (by MAPE) ---
  Iowa                      MAPE=  96.2%  MAE=   106.8  Bias=   -23.7
  District of Columbia      MAPE=  91.7%  MAE=    38.9  Bias=   -19.5
  South Dakota              MAPE=  89.1%  MAE=    32.2  Bias=    -6.0
  Oklahoma                  MAPE=  88.1%  MAE=   209.4  Bias=   -12.1
  US                        MAPE=  83.2%  MAE= 18016.2  Bias=-18016.2

--- Best 5 Locations (by MAPE) ---
  Nevada                    MAPE=  31.9%  MAE=    39.0  Bias=   -15.4
  Florida                   MAPE=  34.3%  MAE=   626.7  Bias=  -568.1
  Louisiana                 MAPE=  36.0%  MAE=    98.6  Bias=   -61.1
  New York                  MAPE=  36.4%  MAE=   592.1  Bias=  -556.2
  Texas                     MAPE=  37.5%  MAE=   661.2  Bias=  -542.1
```

**Key findings from this output**:
- The model **under-predicts** overall (bias -449.6), especially during **peak** (bias -684.3)
- Peak phase has the *lowest* MAPE (47.3%) but *highest* absolute error — because hospitalization counts are large
- Error grows from 30.6% (Week 1) to 80.8% (Week 4) as expected
- The **US national aggregate** is catastrophically wrong (MAE 18,016) — the model likely isn't designed for national-level totals
- Best-performing states (Nevada, Florida, Louisiana) have MAPE in the 31-37% range

---

## 6. Roadmap

### Milestone 1: LLM Summarization --- COMPLETE

`python -m agent summarize` computes phase-aware metrics and generates a natural language performance report.

### Milestone 2: Improvement Loop (Next)

The core capability: an iterative loop where the LLM diagnoses problems and suggests fixes.

```
evaluate → [LLM: "peak bias is -684, try upweighting peak training data"]
    → suggest → [LLM proposes: resample_by_phase(peak, weight=2.0)]
    → apply → [retrain XGBoost with sample weights]
    → evaluate → [check if MAPE improved]
    → repeat or stop
```

**New components**:
- `agent/orchestrator.py` — LangGraph state machine with evaluate/suggest/apply nodes
- `agent/run_tracker.py` — SQLite-backed iteration history
- Complete the adapter's `get_available_actions()`, `apply_action()`, `run_pipeline()`
- Add `sample_weight` support to `src/model.py` and `src/direct_forecast.py`
- New CLI command: `python -m agent improve --max-iterations 3`

**Constrained action space** — the LLM can only suggest actions from a predefined list:

| Action | What it does |
|---|---|
| `adjust_hyperparameter` | Change XGBoost params (learning_rate, max_depth, etc.) |
| `resample_by_phase` | Upweight training samples from specific phases |
| `toggle_feature` | Enable/disable features from the FEATURES_REMOVED list |
| `adjust_floor_constraint` | Change the post-prediction floor percentage |
| `change_target_transform` | Switch between log/raw/ratio target modes |

This prevents the LLM from suggesting impossible or vague actions.

### Milestone 3: Tracking & History

Engineers can review what the agent did across runs:
```bash
python -m agent history                    # list recent runs
python -m agent compare --run-id <id>      # compare iterations within a run
python -m agent status                     # latest run status
```

Also adds **boosting-style sample reweighting**: automatically upweight training samples from phases/locations where the model underperforms, analogous to boosting in ensemble methods.

### Milestone 4: Generalization

- Template adapter for new domains
- Documentation for plugging in financial, sales, or other forecasting models

---

## 7. Technology Stack

| Component | Technology | Why |
|---|---|---|
| Forecasting model | XGBoost (DirectForecastEnsemble) | Existing, proven pipeline |
| Agent orchestration | LangGraph (Milestone 2+) | Graph-based state machine with checkpointing |
| LLM serving | vLLM (cluster) / Ollama (laptop) | OpenAI-compatible API, self-hosted |
| LLM model | Qwen 2.5 72B AWQ | Best open-source model for structured analysis |
| LLM integration | langchain-openai (`ChatOpenAI`) | Provider-agnostic, same code for all backends |
| Run tracking | SQLite (Milestone 3) | Lightweight, no server needed |
| Data source | CDC FluSight GitHub repo | Weekly influenza hospitalization data |

---

## 8. Open Questions

1. **Loop termination**: What MAPE improvement threshold should stop the improvement loop? (Currently planned: < 1% improvement)
2. **Held-out evaluation**: Should the agent be evaluated on a held-out test period it never sees during improvement?
3. **Scenario modeling**: How should the framework extend to "what-if" analysis (e.g., given a vaccination rate, how will the epidemic evolve)?
4. **Multi-model comparison**: Should the loop compare XGBoost vs. neural network models in parallel?
