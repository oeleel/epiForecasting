# epiForecasting

**Agentic AI framework for epidemiological forecasting.** An LLM-powered agent system that autonomously analyzes forecast outputs from an XGBoost-based influenza hospitalization model, diagnoses performance issues, and iteratively improves model quality through a two-agent loop.

The project has two layers:

1. **Forecasting pipeline** (`src/`) — an XGBoost Direct Forecast Ensemble that predicts weekly hospital admissions for all US states using CDC FluSight data.
2. **Agent framework** (`agent/`) — a domain-agnostic orchestration layer plus a flu-forecasting adapter that wraps the pipeline. Two cooperating LLM agents (Analyst + Engineer) read structured metrics, propose constrained improvement actions, retrain the model, and loop until convergence.

---

## Table of contents

1. [What you'll be able to do after setup](#what-youll-be-able-to-do-after-setup)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Clone the repository](#step-1--clone-the-repository)
4. [Step 2 — Create a Python environment](#step-2--create-a-python-environment)
5. [Step 3 — Install Python dependencies](#step-3--install-python-dependencies)
6. [Step 4 — Install and run Ollama (the local LLM server)](#step-4--install-and-run-ollama-the-local-llm-server)
7. [Step 5 — Pull the language model](#step-5--pull-the-language-model)
8. [Step 6 — Get the forecast and ground-truth data](#step-6--get-the-forecast-and-ground-truth-data)
9. [Step 7 — Run Milestone 1 (single-shot LLM report)](#step-7--run-milestone-1-single-shot-llm-report)
10. [Step 8 — Run Milestone 2 (the two-agent improvement loop)](#step-8--run-milestone-2-the-two-agent-improvement-loop)
11. [Inspecting past runs](#inspecting-past-runs)
12. [Running the tests](#running-the-tests)
13. [Configuration via environment variables](#configuration-via-environment-variables)
14. [Project layout](#project-layout)
15. [Troubleshooting](#troubleshooting)

---

## What you'll be able to do after setup

- Compute phase-aware evaluation metrics (overall, per horizon, per epidemic phase, per state) on any forecast CSV — including **WIS** and 95% interval coverage when quantile forecasts are provided.
- Generate a natural-language diagnostic report from a local LLM that interprets those metrics (Milestone 1).
- Run the **two-agent improvement loop**: Agent 1 (Analyst) diagnoses model weaknesses, Agent 2 (Engineer) picks an action from a constrained catalog, the system applies it, retrains, re-evaluates, and iterates (Milestone 2).
- Browse and compare past improvement runs from a SQLite tracker.
- Re-train the underlying XGBoost forecasting model on fresh CDC data.
- Run unit + integration tests for the entire agent framework in under one second (no LLM, no retraining required).

---

> **Note:** This guide is written for **macOS** (Apple Silicon or Intel). Linux and Windows instructions will be added later.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Operating system** | macOS 12 (Monterey) or later | Apple Silicon recommended; Intel Macs also work |
| **Python** | 3.11 | Other 3.x versions may work but are untested |
| **Disk space** | ~10 GB | ~5 GB for the LLM model, ~1 GB for Python deps, the rest for data |
| **RAM** | 16 GB recommended | The 8B LLM uses ~6 GB while running |
| **GPU** | Built-in | Ollama uses the Apple Silicon GPU automatically — no setup needed |
| **Git** | Any recent version | Comes with Xcode Command Line Tools |
| **Homebrew** | Latest | The easiest way to install Ollama and Python on macOS |

If you don't have Homebrew yet, install it from <https://brew.sh>:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

If you don't have Python 3.11:

```bash
brew install python@3.11
```

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/<your-org>/epiForecasting.git
cd epiForecasting
```

(Replace `<your-org>` with the actual GitHub organization.)

---

## Step 2 — Create a Python environment

We strongly recommend using a virtual environment so the project's dependencies don't conflict with your system Python.

**Option A — `venv` (built into Python):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Option B — `conda`:**

```bash
conda create -n epiforecast python=3.11 -y
conda activate epiforecast
```

After activation your shell prompt should show `(.venv)` or `(epiforecast)`.

---

## Step 3 — Install Python dependencies

The project has two dependency groups:

### 3a. Core forecasting pipeline

```bash
pip install -r documentation/requirements.txt
```

This installs XGBoost, pandas, numpy, scikit-learn, scipy, matplotlib, plotly, optuna, shap, torch, and the rest of the forecasting stack.

### 3b. Agent framework

The agent framework needs `langchain-openai` to talk to the LLM. It is **not** in `requirements.txt` because the forecasting pipeline can run without it.

```bash
pip install langchain-openai
```

Verify your install:

```bash
python -c "import xgboost, pandas, langchain_openai; print('OK')"
```

You should see `OK`.

---

## Step 4 — Install and run Ollama (the local LLM server)

Ollama is a small native server that runs open-source language models locally and exposes an OpenAI-compatible HTTP API. We use it so the framework can stay completely offline and free.

### Install Ollama

```bash
brew install ollama
```

### Start the Ollama server

The easiest way is to launch Ollama as a background service that auto-starts on login:

```bash
brew services start ollama
```

Alternatively, run it manually in its own terminal window:

```bash
ollama serve
```

You should see something like `Listening on 127.0.0.1:11434`. The framework expects to find Ollama at `http://localhost:11434` by default.

Verify it's running:

```bash
curl http://localhost:11434/api/tags
```

You should get a JSON response (likely an empty model list until step 5).

---

## Step 5 — Pull the language model

With `ollama serve` running, open a **second** terminal and pull the Qwen 3 8B model:

```bash
ollama pull qwen3:8b
```

This downloads about 5 GB and may take a few minutes depending on your connection. Verify it landed:

```bash
ollama list
```

You should see `qwen3:8b` in the list. To do a quick smoke test:

```bash
ollama run qwen3:8b "Say hello in one short sentence."
```

If you get a coherent reply, the LLM side is working.

> **Why Qwen 3 8B?** It's small enough to run on a laptop, supports structured-output / JSON mode (which the two-agent loop relies on), and has strong performance on tabular reasoning tasks compared to other models in its size class. You can use a different model by setting `LLM_MODEL` (see [Configuration](#configuration-via-environment-variables) below).

---

## Step 6 — Get the forecast and ground-truth data

The agent framework reads two things:

1. **A forecast CSV** — the model's predictions
2. **The CDC ground-truth file** — actual hospital admissions

### 6a. Ground-truth data

The CDC FluSight data lives at `data/raw/flusight_hospital_admissions.csv`. If your clone already includes it, you're done. Otherwise, fetch the latest copy from CDC:

```bash
python -m src.data_loader update
```

This downloads from the CDC FluSight GitHub repo and caches the result for one week.

### 6b. Forecast outputs

For a quick start, the repo ships with two pre-computed forecast batches:

```
outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv  ← preferred (has q05/q25/q50/q75/q95)
outputs/quantile_hindcasts/point_forecasts_nov_apr.csv     ← legacy (point forecasts only)
```

The quantile file is what most demos default to, because the WIS metric and 95% interval coverage require quantile data. Both cover November 2024 to April 2025 across all US states.

If you want to generate your own forecasts (for example to evaluate a new cutoff date), run the pipeline:

```bash
# Single cutoff date
python scripts/pipeline.py --cutoff-date 2024-11-02

# Full season hindcast (slow — trains the model many times)
python scripts/generate_forecasts_nov_apr.py
```

The output CSV will be saved under `outputs/`.

---

## Step 7 — Run Milestone 1 (single-shot LLM report)

Milestone 1 is a single LLM pass: load a forecast, compute metrics, send them to the model, get back a natural-language diagnostic report. Useful as a quick sanity check that everything is wired up.

### Option A — The Milestone 1 demo script (recommended for first-time users)

```bash
python scripts/demo_milestone1.py
```

This walks through the entire pipeline with friendly section headers:

1. Loads the pre-computed forecast CSV and CDC ground truth
2. Computes phase-aware metrics (overall, per horizon, per epidemic phase, top/bottom locations)
3. Sends them to `qwen3:8b` via Ollama and prints the LLM's diagnostic report

To see the metrics without calling the LLM (no Ollama needed):

```bash
python scripts/demo_milestone1.py --no-llm
```

### Option B — The `summarize` CLI command

```bash
# Metrics only — no LLM needed
python -m agent summarize \
  --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv \
  --dry-run

# Full analysis with LLM
python -m agent summarize \
  --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv

# Show the prompt being sent to the LLM (useful for debugging)
python -m agent summarize \
  --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv \
  --verbose

# Point at a different LLM endpoint (e.g. a remote vLLM server)
python -m agent summarize \
  --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv \
  --base-url http://my-server:8000/v1 \
  --model qwen2.5:72b-awq
```

If everything is wired up correctly you'll see:

1. A formatted metrics table (overall / by horizon / by phase / top and bottom states / worst individual predictions)
2. A "Sending to LLM..." line
3. A multi-paragraph natural-language analysis of the model's behavior

---

## Step 8 — Run Milestone 2 (the two-agent improvement loop)

Milestone 2 is the headline feature. Each iteration runs two LLM calls and one model retrain:

1. **Agent 1 (Analyst)** reads the latest metrics and emits a structured JSON diagnosis: weak segments by phase / horizon / location, and hypothesized causes.
2. **Agent 2 (Engineer)** reads that diagnosis plus the action history and chooses ONE action from a constrained catalog of six (hyperparameter tweak, sample reweighting, feature toggle, floor adjust, target transform change, or stop).
3. The action is validated against numeric guardrails and applied to a config dict.
4. The pipeline retrains XGBoost with the mutated config and writes a new forecast CSV.
5. The system re-evaluates, logs the iteration to SQLite, and decides whether to loop or stop.

The LLM is in the **reasoning role only**. Python computes metrics, validates JSON, and applies actions with hard guardrails. The LLM cannot push `max_depth=50` even if it wants to — that gets rejected before retraining.

### Option A — The Milestone 2 demo script (recommended)

```bash
# Real LLM, fake pipeline (file copy instead of XGBoost retrain). Fast, ~30-60s total.
python scripts/demo_milestone2.py --fake-pipeline --auto-apply

# Same as above but interactive (asks y/n/s before each action)
python scripts/demo_milestone2.py --fake-pipeline

# Real loop with actual XGBoost retraining (~3-10 minutes for 3 iterations)
python scripts/demo_milestone2.py --auto-apply

# Demo against a harder cutoff (peak-week start gives the loop more to fix)
python scripts/demo_milestone2.py --auto-apply --cutoff-date 2024-12-21
```

The demo prints each iteration's full reasoning trace: Agent 1's diagnosis (summary, focus, weak segments, hypotheses), then Agent 2's action with its rationale and expected effect.

### Option B — The `improve` CLI command

```bash
# Default: regenerates baseline at the cutoff, runs 3 iterations, optimizes WIS
python -m agent improve \
  --cutoff-date 2024-11-02 \
  --max-iterations 3 \
  --target-metric wis \
  --auto-apply

# With --fake-pipeline (skip retraining, exercise LLM end-to-end fast)
python -m agent improve \
  --cutoff-date 2024-11-02 \
  --max-iterations 3 \
  --fake-pipeline \
  --auto-apply

# Optimize a different metric
python -m agent improve --cutoff-date 2024-11-02 --target-metric mape --auto-apply
python -m agent improve --cutoff-date 2024-11-02 --target-metric coverage_95 --auto-apply

# Interactive shadow mode (default — asks before each action)
python -m agent improve --cutoff-date 2024-11-02 --max-iterations 3

# Use a pre-existing baseline file as iteration 0 instead of regenerating it
# (only safe if that CSV was made with the default config at this exact cutoff)
python -m agent improve \
  --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv \
  --cutoff-date 2024-11-02 \
  --no-regenerate-baseline \
  --auto-apply
```

### What happens during a run

```
=== run_id: 20260411-103022-a3f2 ===

[iteration 0/3] baseline (regenerating with default config @ 2024-11-02)
  retraining... -> outputs/agent_runs/20260411-103022-a3f2/iter_0.csv
  wis=58.230

[iteration 1/3]

  Agent 1 (Analyst):
    Model under-predicts during peak weeks with worst errors at horizon 4.
    Focus: peak_underprediction
    Weak segments:
      [high  ] phase=peak (wis delta_vs_overall=+0.32)
      [medium] horizon=4 (wis delta_vs_overall=+0.18)
    Hypotheses:
      (0.75) Log target compression suppresses upper tail at peak weeks.

  Agent 2 (Engineer):
    Action: reweight_training_samples(dimension=peak, weight=2.5)
    Rationale: Upweighting peak samples will counteract the log-target
    compression and let the model fit the upper tail more aggressively.
    Expected: Reduce peak WIS by 10-15%.
  applied: sample_weights.by_phase[peak] = 2.5
  retraining... -> outputs/agent_runs/20260411-103022-a3f2/iter_1.csv
  wis=53.180
  improved by +8.7% on wis

[iteration 2/3]
  ...
```

### Stop conditions

The loop halts on **any** of:

- `iteration >= --max-iterations` (default 5)
- Agent 2 emits the `stop` action
- The target metric regresses in two consecutive iterations
- Improvement falls below `--no-improvement-threshold` (default 1%) for two consecutive iterations
- A pipeline / LLM error in the current iteration

The **best** forecast across all iterations is always retained — never the latest. It's stamped as `best_forecast.csv` in the run directory.

### Run output layout

Each `improve` run produces a directory under `outputs/agent_runs/<run_id>/`:

```
outputs/agent_runs/
├── runs.db                                # SQLite — every run ever
└── 20260411-103022-a3f2/                  # this run
    ├── iter_0.csv                         # baseline forecast
    ├── iter_1.csv                         # forecast after action 1
    ├── iter_2.csv                         # forecast after action 2
    ├── iter_3.csv                         # forecast after action 3
    └── best_forecast.csv                  # copy of the winning iter
```

Every iteration's full metrics, diagnosis, action, and config are JSON-serialized into the SQLite tracker, so any iteration can be reconstructed exactly later.

---

## Inspecting past runs

```bash
# Show the 10 most recent runs (no LLM, no retraining)
python -m agent history

# Show more
python -m agent history --limit 25

# Side-by-side comparison of two runs
python -m agent compare 20260411-103022-a3f2 20260411-091844-7d10

# Compare on a different metric
python -m agent compare <run_id_a> <run_id_b> --metric mape
python -m agent compare <run_id_a> <run_id_b> --metric coverage_95
```

`history` prints a table:

```
run_id                  started_at              iters  best_wis  status
20260411-103022-a3f2    2026-04-11T10:30:22+0000     4    46.690  completed
20260410-133935-77dc    2026-04-10T13:39:35+0000     3    58.230  completed
```

`compare` prints baseline vs. best for each run plus the delta.

---

## Running the tests

The agent framework has a complete unit + integration test suite that runs in under one second. **No LLM, no Ollama, no XGBoost retraining required** — the orchestrator is tested with a fake LLM client and a scripted pipeline runner.

```bash
# Run all 77 tests
PYTHONPATH=. python tests/agent/run_all.py

# Run a single test module
PYTHONPATH=. python tests/agent/test_orchestrator.py
PYTHONPATH=. python tests/agent/test_run_tracker.py
PYTHONPATH=. python tests/agent/test_wis.py
```

Coverage:

| Module | Tests | What it covers |
|---|---:|---|
| `test_run_tracker.py` | 8 | SQLite roundtrip, best-iteration logic, status transitions |
| `test_config.py` | 8 | Dict helpers, immutability, JSON serialization |
| `test_sample_weights.py` | 7 | Per-row weight math, multiplicative combination |
| `test_adapter_actions.py` | 17 | 6 valid actions + 10 guardrail violations + catalog shape |
| `test_prompts.py` | 17 | JSON extraction, validators, prompt builders |
| `test_wis.py` | 12 | WIS row math, vectorized, end-to-end through evaluator |
| `test_orchestrator.py` | 8 | Stop conditions, repair retry, state persistence (uses fake harness) |

---

## Configuration via environment variables

You can set these in your shell (or in a `.env` file you source) to avoid passing flags every time:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Where the OpenAI-compatible LLM API lives |
| `LLM_MODEL` | `qwen3:8b` | Model identifier to send in API requests |

Example:

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen3:8b
python -m agent summarize --forecast-csv outputs/quantile_hindcasts/point_forecasts_nov_apr.csv
```

---

## Project layout

```
epiForecasting/
├── agent/                          # Agent framework (Milestones 1 + 2 complete)
│   ├── __main__.py                 # Entry point: python -m agent
│   ├── cli.py                      # CLI: summarize / improve / history / compare
│   ├── domain_adapter.py           # Abstract base class for domain adapters
│   ├── phase_evaluator.py          # Phase-aware metric computation + WIS
│   ├── prompt_templates.py         # Diagnosis + action prompts and validators
│   ├── llm_client.py               # OpenAI-compatible LLM wrapper
│   ├── orchestrator.py             # Two-agent loop (Milestone 2)
│   ├── run_tracker.py              # SQLite run history + comparison
│   └── adapters/
│       └── flu_forecast.py         # Flu adapter + 6-action catalog
│
├── src/                            # Forecasting pipeline (XGBoost)
│   ├── config.py                   # Hyperparameters + dict-based config helpers
│   ├── pipeline.py                 # Callable run_pipeline(config) for the loop
│   ├── data_loader.py              # CDC FluSight data fetch + cache
│   ├── feature_engineering.py      # 59-feature engineering pipeline
│   ├── direct_forecast.py          # Direct + Quantile + Clustered ensembles
│   ├── model.py                    # XGBoost wrapper (with sample_weight support)
│   ├── nn_model.py                 # PyTorch quantile NN alternative
│   ├── train.py                    # Walk-forward training
│   ├── predict.py                  # Forecast generation
│   ├── evaluate.py                 # Baseline evaluation
│   └── visualization.py            # Plotting utilities
│
├── scripts/
│   ├── demo_milestone1.py          # End-to-end Milestone 1 demo
│   ├── demo_milestone2.py          # End-to-end Milestone 2 demo
│   ├── pipeline.py                 # Legacy CLI pipeline (not used by the loop)
│   ├── optimize.py                 # Optuna hyperparameter search
│   ├── generate_forecasts_nov_apr.py  # Batch hindcast Nov 2024–Apr 2025
│   └── evaluation/                 # Feature/regularization studies
│
├── tests/
│   └── agent/                      # 77 unit + integration tests, <1s to run
│       ├── run_all.py              # Single-process runner (no pytest dep)
│       ├── fakes.py                # FakeLLM, ScriptedPipeline, FakeAdapter
│       ├── test_run_tracker.py
│       ├── test_config.py
│       ├── test_sample_weights.py
│       ├── test_adapter_actions.py
│       ├── test_prompts.py
│       ├── test_wis.py
│       └── test_orchestrator.py
│
├── data/
│   ├── raw/                        # Cached CDC FluSight CSVs
│   ├── processed_features/         # Engineered feature outputs
│   └── feature_dictionary.csv      # Feature documentation
│
├── outputs/
│   ├── agent_runs/                 # Per-run directories + runs.db (gitignored)
│   ├── quantile_hindcasts/         # Pre-computed Nov–Apr forecast batches
│   ├── forecasts/                  # Generated forecasts
│   ├── forecast_plots/             # Visualizations
│   └── shap_analysis/              # SHAP feature importance
│
├── analysis/                       # SHAP and dashboard scripts
├── documentation/
│   ├── requirements.txt            # Python dependencies
│   ├── meeting-notes/              # Research meeting summaries
│   └── *.md                        # Design docs
└── README.md                       # This file
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_openai'`

You skipped step 3b. Run:

```bash
pip install langchain-openai
```

### "LLM server not reachable at ..."

Ollama isn't running. Start it:

```bash
brew services start ollama
```

Then verify with:

```bash
curl http://localhost:11434/api/tags
```

You should get a JSON response listing your installed models.

### "Model 'qwen3:8b' not installed on the server"

Ollama is running but the model the framework is asking for isn't pulled. Either pull it:

```bash
ollama pull qwen3:8b
```

Or pass `--model <name>` with a model that *is* installed:

```bash
ollama list   # see what you have
python -m agent improve --cutoff-date 2024-11-02 --model <one-of-yours> --auto-apply
```

### `! Agent 1 output didn't validate (...). Retrying with error feedback`

Not actually an error — you're seeing the orchestrator's repair logic. Qwen3 8B occasionally returns prose without JSON. The system appends the error to the prompt and re-invokes the LLM once. If the retry succeeds (it almost always does), the iteration continues normally. Two consecutive failures abort the iteration cleanly.

If you see this on every iteration, consider:
- A bigger model: `--model qwen3:14b` or larger
- A different model with stricter JSON adherence

### `FileNotFoundError: ... outputs/agent_runs/runs.db`

The SQLite tracker is auto-created on first use. If you see this, you probably ran `agent history` or `agent compare` from a directory other than the project root. `cd` into the project root and try again.

### `FileNotFoundError: ... quantile_forecasts_nov_apr.csv` or `point_forecasts_nov_apr.csv`

The pre-computed forecast batch is missing. Either pull it from the repo with `git lfs` if you use LFS, or regenerate it:

```bash
python scripts/generate_forecasts_nov_apr.py
```

This is slow (it trains the ensemble multiple times across the season). Note that for Milestone 2 you don't actually *need* a pre-existing baseline CSV — `agent improve --cutoff-date <date>` will regenerate one as iteration 0.

### `FileNotFoundError: ... flusight_hospital_admissions.csv`

You're missing the ground-truth file. Fetch it:

```bash
python -m src.data_loader update
```

### `xgboost.core.XGBoostError: ... GPU ...`

XGBoost is trying to use a GPU it can't find. The pipeline runs fine on CPU; this is usually a config drift issue. Edit `src/config.py` and ensure `XGBOOST_PARAMS_V2` does not set `tree_method="gpu_hist"` or `device="cuda"`.

### Ollama is too slow

The 8B model is the recommended baseline. If responses are slow you can try a smaller model:

```bash
ollama pull qwen2.5:3b
LLM_MODEL=qwen2.5:3b python -m agent summarize --forecast-csv ...
```

Quality will drop but iteration speed improves.

### "I want to use a different LLM provider"

Any OpenAI-compatible endpoint works. Set `LLM_BASE_URL` to your provider's URL and `LLM_MODEL` to the model name. Examples:

- Local vLLM server: `http://localhost:8000/v1`
- LM Studio: `http://localhost:1234/v1`
- A remote OpenAI-compatible service: `https://api.example.com/v1` (you may also need to set an API key in `agent/llm_client.py`)

---

## Where to go next

- Read [`documentation/agentic_framework_design.md`](documentation/agentic_framework_design.md) for the framework's architectural rationale.
- Read [`documentation/agent_framework_report.md`](documentation/agent_framework_report.md) for a deeper look at Milestone 1.
- See [`.claude/CLAUDE.md`](.claude/CLAUDE.md) for the development guide and milestone roadmap.
- See `agent/orchestrator.py` and `agent/adapters/flu_forecast.py` for the Milestone 2 implementation. The action catalog at the top of `flu_forecast.py` is the easiest entry point — adding a new action is one entry there plus a clause in `apply_action`.
- Run `python -m agent --help` to see all CLI commands at a glance.
