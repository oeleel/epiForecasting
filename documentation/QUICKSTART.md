# Agent Framework Quickstart

## Setup

```bash
# Install dependencies (uv; plain pip works too)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r documentation/requirements.txt pytest langchain-openai
uv pip install --python .venv/bin/python -r documentation/requirements-nixtla.txt   # optional example models
source .venv/bin/activate

# Start Ollama (local LLM server)
brew install ollama            # one-time
ollama serve                   # start server
ollama pull qwen3:8b           # download model (~5GB)
```

For Rivanna (vLLM), set the environment variable instead:
```bash
export LLM_BASE_URL=http://localhost:8247/v1
export LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
```

## Verify your setup

```bash
# Run tests (no LLM needed)
python -m pytest tests/agent/ -v

# Check that the agent CLI loads
python -m agent --help
```

## Researcher quickstart: pick a model, then refine it

The framework has two stages. **Stage 1 (model selection)** answers "which
model should I start from?" by warming up every candidate on the pinned
evaluation window and ranking them by the goal metric. **Stage 2 (improvement
loop)** hands that incumbent to the two LLM agents for constrained refinement.

The fastest way to see both, with no LLM needed for the first six steps:

```bash
python scripts/demo_stage1.py            # <1 min: 6 locations, 3 cutoffs, 4 models
python scripts/demo_stage1.py --full     # ~10 min: all locations, every 4th week
python scripts/demo_stage1.py --improve --auto-apply   # + hand-off to the agents (needs an LLM)
```

The same thing as individual commands:

```bash
# What models can I try here? (in-house models: pass pkg.module:ClassName, no registration)
python -m agent list-models

# Which should I start from? Pinned window, phase-aware WIS, incumbent by goal.
python -m agent select-model --stride-weeks 4 --exclude-locations US \
    --json outputs/model_selection/run.json --report outputs/model_selection/run.md

# Quick mode for a live demo (a few states, 3 cutoffs)
python -m agent select-model --locations US 06 48 12 36 17 --max-cutoffs 3 --exclude-locations US

# Same run, different objective (the goal is a parameter)
python -m agent select-model --stride-weeks 4 --exclude-locations US --metric wis --phase peak

# Compare your own model against the bank
python -m agent select-model --families xgboost_direct my_lab.models:FluLSTM --stride-weeks 4

# Refine the winner with the agents (any family)
python -m agent improve --cutoff-date 2025-12-06 --model-family nf_nhits --auto-apply
```

What the pinned window is and why: `documentation/MODEL_BANK.md`. Adding a
model takes two methods and a list of tunable parameters; the same document has
the recipe.

## Workflow

### 1. Check data quality

Inspect the raw CDC data for issues before training:

```bash
python -m agent check-data --cutoff-date 2024-11-02 --dry-run
```

- `--dry-run` runs checks without the LLM (shows raw issues)
- Remove `--dry-run` to get an LLM-powered interpretation of the findings

### 2. Run the improvement loop

The agent evaluates the current model, diagnoses problems, and iteratively improves it:

```bash
# Interactive mode (asks before applying each action)
python -m agent improve --cutoff-date 2024-11-02

# Auto mode (applies all actions without asking)
python -m agent improve --cutoff-date 2024-11-02 --auto-apply

# Test mode (uses a stub pipeline — no XGBoost training, just tests the LLM)
python -m agent improve --cutoff-date 2024-11-02 --auto-apply --fake-pipeline \
    --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv \
    --no-regenerate-baseline
```

Options:
- `--max-iterations N` — limit iterations (default: 5)
- `--target-metric wis` — metric to optimize (wis, mape, mae, rmse, coverage_95, bias)

### 3. Review results

```bash
# List all past runs
python -m agent history

# Detailed view of a specific run (iterations + actions taken)
python -m agent status <run_id>

# Compare two runs side by side
python -m agent compare <run_id_a> <run_id_b>
```

### 4. Analyze a specific forecast (no improvement loop)

Generate a one-time performance report:

```bash
# Metrics only (no LLM)
python -m agent summarize --forecast-csv outputs/.../forecast.csv --dry-run

# Full LLM analysis
python -m agent summarize --forecast-csv outputs/.../forecast.csv
```

## What the agent can do

The improvement loop has two LLM agents:

- **Agent 1 (Analyst)** reads metrics and identifies weak spots (phases, horizons, locations)
- **Agent 2 (Engineer)** picks an action from a constrained menu:

| Action | What it does |
|---|---|
| `adjust_hyperparameter` | Tune one hyperparameter. For XGBoost: max_depth, learning_rate, etc. For any model-bank family: whatever its `param_space()` declares |
| `reweight_training_samples` | Upweight training data by phase/horizon/location |
| `toggle_feature` | Enable/disable feature groups (lag, rolling, yoy, etc.) |
| `adjust_floor_constraint` | Change post-prediction floor percentage |
| `change_target_transform` | Switch between log/raw/sqrt target transforms |
| `stop` | Declare convergence |

All actions have guardrails — the LLM cannot set values outside safe ranges.
The pipeline-specific actions (reweight, feature toggle, floor, target
transform) apply to the legacy XGBoost family; other families get
`adjust_hyperparameter` + `stop`, generated from their own declared knobs.

## File structure

```
agent/
  cli.py              # All CLI commands
  orchestrator.py     # Two-agent improvement loop (Stage 2)
  model_selection.py  # Warm-up + incumbent selection (Stage 1)
  data_quality.py     # Pre-training data checks
  adapters/
    flu_forecast.py   # CDC FluSight adapter (actions, metrics, pipeline)
  llm_client.py       # Ollama/vLLM connection
  run_tracker.py      # SQLite history (outputs/agent_runs/runs.db)
src/model_bank/
  contract.py         # ForecastModel: what any model implements
  registry.py         # named families + dotted-path in-house models
  runner.py           # load -> fit -> predict -> forecast CSV
scripts/demo_stage1.py   # the researcher walkthrough
```

## Troubleshooting

**LLM not connecting**: Make sure `ollama serve` is running and the model is pulled.
```bash
curl http://localhost:11434/api/tags   # should list models
```

**No forecast data**: Refresh the CDC data cache:
```bash
python -m src.data_loader update
```

**Tests fail**: Check Python version (requires 3.11+) and dependencies.

**A family shows `NO` in `list-models`**: its optional dependency is missing;
the reason column names the package. Install `requirements-nixtla.txt` for the
Nixtla examples.

**`select-model` is slow**: use `--locations ... --max-cutoffs 3` for a demo,
`--stride-weeks 4` for a full run. `sf_autoarima` is ~1 min per 3 series; leave
it off the default lineup.
