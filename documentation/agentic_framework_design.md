# Agentic Forecasting Framework — Conceptual Design

> Prepared for Friday meeting discussion | March 2026

---

## 1. Goal & Scope

### Primary Objective
Automate the **weekly influenza hospitalization forecasting workflow** end-to-end — from CDC data ingestion through model training, forecast generation, evaluation, and reporting — with minimal human intervention.

### Scope Boundaries

| In Scope | Out of Scope (for now) |
|----------|----------------------|
| Weekly automated data fetch from CDC FluSight GitHub | Real-time (sub-weekly) forecasting |
| Feature engineering (59+ features, 10 passes) | Multi-pathogen forecasting (COVID, RSV) |
| DirectForecastEnsemble training (4 horizon models) | Interactive dashboard with live updates |
| 1–4 week ahead predictions for all 58 US locations | Automatic FluSight Hub submission |
| Quantile forecast generation (5 quantiles) | LLM-based interpretation of results |
| Evaluation with WIS, coverage, MAPE metrics | Automatic hyperparameter re-optimization |
| Quality gating and human notification | |
| Weekly report generation | |

### Success Criteria
- Pipeline runs autonomously every week after CDC data release (typically Saturdays)
- Failures are caught, logged, and escalated — never silently ignored
- Human review is required only for quality gate failures
- End-to-end runtime < 15 minutes per weekly cycle

---

## 2. Agent Definitions

### 2.1 Scheduler (Trigger)
**Role**: Initiates the weekly pipeline at the right time.

| Field | Detail |
|-------|--------|
| **Trigger** | Cron job (Saturday morning) or CDC data release webhook |
| **Input** | Current date |
| **Logic** | Compute cutoff date (most recent Saturday); check if this week's run already completed |
| **Output** | `cutoff_date` string → Orchestrator |
| **Existing Code** | None — new component needed |

### 2.2 Orchestrator Agent
**Role**: Central coordinator that sequences the pipeline and handles failures.

| Field | Detail |
|-------|--------|
| **Input** | `cutoff_date` from Scheduler |
| **Logic** | Manages the DAG of agent calls; retries failed steps (max 2); routes to quality gate |
| **Output** | Pipeline status (success/failure), artifact paths |
| **State** | Tracks which steps have completed, intermediate artifacts |
| **Existing Code** | Partially maps to `FluForecastingPipeline.run_full_pipeline()` in `scripts/pipeline.py` |

### 2.3 Data Fetcher Agent
**Role**: Retrieves fresh CDC data and validates it.

| Field | Detail |
|-------|--------|
| **Input** | `cutoff_date` |
| **Tasks** | 1. Check cache staleness via `FluDataLoader.get_data_status()` |
|  | 2. Fetch new data if stale via `FluDataLoader.update_data()` |
|  | 3. Apply temporal cutoff via `load_and_preprocess(cutoff_date)` |
|  | 4. Validate: required columns, no negative values, expected location count |
| **Output** | Validated `DataFrame` (date, location, value, weekly_rate) |
| **Failure Modes** | Network timeout → retry 2x; Corrupt data → fallback to stale cache + alert; Missing locations → warn |
| **Existing Code** | `src/data_loader.py` → `FluDataLoader` (fully implemented) |

### 2.4 Feature Engineer Agent
**Role**: Transforms raw data into model-ready features.

| Field | Detail |
|-------|--------|
| **Input** | Validated DataFrame from Data Fetcher |
| **Tasks** | 1. Run 10 ordered feature passes (temporal → lags → rolling → YoY → ...) |
|  | 2. Create trend features |
|  | 3. Handle missing values (ffill within location, then zero-fill) |
|  | 4. Remove low-importance features (SHAP-guided, `FEATURES_REMOVED` list) |
| **Output** | Feature DataFrame (59+ columns per row) + feature column list |
| **Validation** | Check no NaN in feature columns; verify feature count matches expected |
| **Existing Code** | `src/feature_engineering.py` → `FeatureEngineer` (fully implemented) |

### 2.5 Model Trainer Agent
**Role**: Trains 4 horizon-specific XGBoost models.

| Field | Detail |
|-------|--------|
| **Input** | Feature DataFrame from Feature Engineer |
| **Tasks** | 1. For each horizon h=1..4: shift target by h weeks, apply `log1p` |
|  | 2. Temporal 80/20 train/val split |
|  | 3. Train XGBRegressor with early stopping (50 rounds) |
|  | 4. Compute train/val metrics (MAE, RMSE, MAPE) |
|  | 5. Check overfitting ratio (val_mae / train_mae) |
|  | 6. Save models to `models/direct_forecast_models/` |
| **Output** | 4 trained models + training metrics dict + feature importance CSV |
| **Quality Checks** | Overfitting ratio < 10x; val MAPE < 70% on all horizons |
| **Existing Code** | `src/direct_forecast.py` → `DirectForecastEnsemble.train()` |

### 2.6 Forecaster Agent
**Role**: Generates 1–4 week ahead predictions for all locations.

| Field | Detail |
|-------|--------|
| **Input** | Trained models + latest feature data + cutoff_date |
| **Tasks** | 1. For each location: get last known row at cutoff |
|  | 2. For each horizon: predict → `expm1` → floor constraint (30% decay) |
|  | 3. Optionally run `QuantileDirectForecastEnsemble` for interval forecasts |
|  | 4. Save forecast CSV to `outputs/forecasts_{cutoff}_{ts}.csv` |
| **Output** | Forecast DataFrame (location, forecast_date, forecast, quantiles) |
| **Validation** | No negative predictions; all 58 locations present; 4 horizons per location |
| **Existing Code** | `DirectForecastEnsemble.generate_forecasts()` |

### 2.7 Evaluator Agent
**Role**: Scores forecasts against ground truth (when available) and applies quality gates.

| Field | Detail |
|-------|--------|
| **Input** | Forecast DataFrame + actual data (from previous weeks' ground truth) |
| **Tasks** | 1. Merge forecasts with actuals on (location, forecast_date) |
|  | 2. Compute MAE, RMSE, MAPE, SMAPE, R² overall and by horizon/location |
|  | 3. For quantile forecasts: coverage, calibration, pinball loss, CRPS |
|  | 4. Generate evaluation report with recommendations |
|  | 5. Apply quality gate thresholds |
| **Output** | Evaluation JSON + pass/fail quality gate decision |
| **Quality Gate Thresholds** | MAPE > 70% on any horizon → FAIL; Overfitting > 10x → FAIL; Coverage < 50% or > 95% for 90% interval → WARN |
| **Existing Code** | `src/evaluate.py` → `ModelEvaluator`, `QuantileEvaluator` |

### 2.8 Reporter Agent
**Role**: Consolidates results and notifies stakeholders.

| Field | Detail |
|-------|--------|
| **Input** | All pipeline artifacts: forecasts, evaluation, training metrics |
| **Tasks** | 1. Generate summary report (text or HTML) |
|  | 2. Create key plots (actual vs predicted, MAE by horizon, MAE by location) |
|  | 3. Send notification (email/Slack) with status and key metrics |
|  | 4. Archive artifacts with timestamped paths |
| **Output** | Report file + notification sent |
| **Existing Code** | Partial — `ModelEvaluator.create_evaluation_plots()`, `generate_evaluation_report()` |

---

## 3. Workflow Sequence

```
┌─────────────┐
│  Scheduler   │──── Saturday cron / CDC webhook
└──────┬──────┘
       │ cutoff_date
       ▼
┌──────────────┐
│ Orchestrator │──── Central state machine
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────┐
│ Data Fetcher │────▶│ CDC GitHub │  (external)
└──────┬───────┘     └───────────┘
       │ validated DataFrame
       ▼
┌──────────────────┐
│ Feature Engineer │
└──────┬───────────┘
       │ 59+ feature columns
       ▼
┌───────────────┐
│ Model Trainer │
└──────┬────────┘
       │ 4 trained models
       ▼
┌────────────┐
│ Forecaster │
└──────┬─────┘
       │ forecast CSV
       ▼
┌───────────┐
│ Evaluator │
└──────┬────┘
       │ quality gate
       ▼
   ◇ Pass?
  / \
 Y   N
 │   └──▶ Flag for human review
 ▼
┌──────────┐
│ Reporter │──── Email/Slack notification
└──────────┘
```

### Data Flow Between Agents

| From → To | Artifact | Format |
|-----------|----------|--------|
| Scheduler → Orchestrator | `cutoff_date` | `str` ("YYYY-MM-DD") |
| Orchestrator → Data Fetcher | `cutoff_date` | `str` |
| Data Fetcher → Feature Engineer | Raw data | `pd.DataFrame` (date, location, value, weekly_rate) |
| Feature Engineer → Model Trainer | Feature matrix | `pd.DataFrame` (59+ feature columns + target) |
| Model Trainer → Forecaster | Trained models | `DirectForecastEnsemble` (in-memory) or model files on disk |
| Model Trainer → Evaluator | Training metrics | `Dict` (MAE, RMSE, MAPE per horizon) |
| Forecaster → Evaluator | Predictions | `pd.DataFrame` (location, forecast_date, forecast, quantiles) |
| Data Fetcher → Evaluator | Ground truth | `pd.DataFrame` (actuals for previous forecast dates) |
| Evaluator → Reporter | Evaluation results | `Dict` (metrics, quality gate pass/fail) |
| All agents → Reporter | Artifacts | File paths (CSVs, JSONs, model files) |

---

## 4. State Management

The Orchestrator maintains a **pipeline state object** that tracks progress:

```python
{
    "run_id": "2026-03-21_weekly",
    "cutoff_date": "2026-03-21",
    "status": "in_progress",  # pending | in_progress | completed | failed
    "steps": {
        "data_fetch":       {"status": "completed", "artifact": "data/raw/...csv"},
        "feature_engineer": {"status": "completed", "artifact": None},
        "model_train":      {"status": "in_progress", "artifact": None},
        "forecast":         {"status": "pending", "artifact": None},
        "evaluate":         {"status": "pending", "artifact": None},
        "report":           {"status": "pending", "artifact": None}
    },
    "errors": [],
    "started_at": "2026-03-21T08:00:00Z",
    "completed_at": null
}
```

This enables:
- **Resumability**: If the pipeline crashes mid-training, restart from that step
- **Observability**: Query current state at any time
- **Auditability**: Full history of runs stored as JSON

---

## 5. Error Handling & Quality Gates

### Retry Policy
| Error Type | Strategy |
|-----------|----------|
| Network failure (CDC fetch) | Retry 2x with 60s backoff; fallback to stale cache |
| Training divergence (NaN loss) | Retry with default params; if still fails → alert |
| Forecast validation failure | Re-run forecaster; if still fails → alert |
| Evaluation data missing | Skip evaluation, still produce forecasts, warn |

### Quality Gate Criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| MAPE (any horizon) | > 70% | FAIL → human review required |
| Overfitting ratio | > 10x | FAIL → suggest re-optimization |
| Coverage (90% interval) | < 50% or > 95% | WARN → log, don't block |
| Missing locations | > 5 locations | WARN → log, don't block |
| Negative predictions | Any | FAIL → re-run with floor constraint |

---

## 6. Framework Evaluation

### Option A: LangGraph (Recommended)

**Why it fits:**
- **Graph-based orchestration** matches our sequential pipeline with conditional branches (quality gate)
- **Persistent state** — built-in checkpointing enables resumability after failures
- **Conditional edges** — natural fit for quality gate routing (pass → report, fail → human review)
- **Human-in-the-loop** — pause-and-resume for manual approval when quality gate fails
- **Production-tested** — used by Klarna, Replit, Elastic

**Architecture mapping:**
```
StateGraph nodes: data_fetch → feature_engineer → model_train → forecast → evaluate
Conditional edge from evaluate: quality_pass → report | quality_fail → human_review
State: pipeline_state dict persisted across nodes
```

**Considerations:**
- Steeper learning curve (graph/state-machine thinking)
- We don't need LLM routing — our agents are deterministic Python functions
- Could be overkill if we just need sequential execution with error handling

### Option B: CrewAI

**Why it could work:**
- **Role-based** — intuitive mapping to our agent roles
- **Beginner-friendly** — easier to set up quickly
- **Built-in observability**

**Concerns:**
- Designed for LLM-driven agent collaboration — our pipeline is deterministic
- Less fine-grained control over state transitions
- Role metaphor doesn't add value when agents are fixed Python functions

### Option C: Lightweight Custom Orchestrator (Python)

**Why to consider:**
- Our pipeline is **deterministic** — no LLM decision-making needed between steps
- Existing code already does 90% of what we need (`pipeline.py`)
- A simple state machine + cron + error handling may be sufficient
- No framework dependency; full control

**Implementation sketch:**
```python
class PipelineOrchestrator:
    def run_weekly(self, cutoff_date: str):
        state = PipelineState(cutoff_date)
        try:
            data = self.run_step("data_fetch", self.fetch_data, cutoff_date)
            features = self.run_step("feature_eng", self.engineer_features, data)
            models = self.run_step("train", self.train_models, features)
            forecasts = self.run_step("forecast", self.generate_forecasts, models, data)
            eval_result = self.run_step("evaluate", self.evaluate, forecasts, data)
            if eval_result["quality_gate"] == "pass":
                self.run_step("report", self.send_report, eval_result, forecasts)
            else:
                self.alert_human(eval_result)
        except PipelineError as e:
            state.mark_failed(e)
            self.alert_human(e)
```

### Recommendation

**Start with Option C (custom orchestrator)**, then migrate to LangGraph if we need:
- Multi-model comparison (parallel training of XGBoost vs. NN)
- LLM-powered anomaly interpretation
- Complex branching (auto-reoptimization, A/B testing)

The current pipeline is sequential and deterministic. Adding a framework adds complexity without proportional benefit *today*. However, LangGraph is the clear choice once the system grows.

---

## 7. Implementation Roadmap

### Phase 1: Automated Weekly Runner (1-2 weeks)
- [ ] Add cron-compatible `run_weekly.py` entry point
- [ ] Implement `PipelineState` with JSON persistence
- [ ] Add retry logic and error notification (email or Slack webhook)
- [ ] Quality gate checks integrated into `evaluate.py`

### Phase 2: Monitoring & Observability (1 week)
- [ ] Pipeline run history dashboard (simple HTML or Streamlit)
- [ ] Alert thresholds configurable in `config.py`
- [ ] Forecast performance tracking over time (week-over-week comparison)

### Phase 3: LangGraph Migration (2-3 weeks, when needed)
- [ ] Refactor agents into LangGraph `StateGraph` nodes
- [ ] Add human-in-the-loop approval for quality gate failures
- [ ] Enable parallel execution (e.g., point + quantile forecasts simultaneously)
- [ ] Add LLM-powered report generation and anomaly interpretation

---

## 8. Mapping to Existing Codebase

| Agent | Existing Module | Readiness |
|-------|----------------|-----------|
| Data Fetcher | `src/data_loader.py` → `FluDataLoader` | Ready — fully implemented with cache, validation |
| Feature Engineer | `src/feature_engineering.py` → `FeatureEngineer` | Ready — 10 passes + trend features |
| Model Trainer | `src/direct_forecast.py` → `DirectForecastEnsemble` | Ready — train + save/load models |
| Forecaster | `src/direct_forecast.py` → `generate_forecasts()` | Ready — point + quantile + floor constraints |
| Evaluator | `src/evaluate.py` → `ModelEvaluator`, `QuantileEvaluator` | Ready — metrics + plots + report |
| Orchestrator | `scripts/pipeline.py` → `FluForecastingPipeline` | Partial — needs state persistence, retry, notification |
| Scheduler | None | New — cron job or scheduled task |
| Reporter | Partial in `evaluate.py` | New — needs notification integration |

**Key insight**: ~80% of agent logic already exists. The primary new work is orchestration (state, retry, scheduling) and notification.

---

## Discussion Questions for Friday Meeting

1. **Notification channel**: Email, Slack, or both for pipeline alerts?
2. **Quality gate thresholds**: Are MAPE > 70% and overfitting > 10x the right defaults?
3. **FluSight submission**: Should we add auto-submission to CDC FluSight Hub as a future agent?
4. **Model comparison**: Do we want to run XGBoost and neural network models in parallel?
5. **Timeline**: Should we start with Phase 1 (custom orchestrator) or jump to LangGraph?
