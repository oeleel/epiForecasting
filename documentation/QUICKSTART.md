# Agent Framework Quickstart

## Setup

```bash
# Install dependencies
pip install -r documentation/requirements.txt
pip install langchain-openai   # required for LLM features

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
| `adjust_hyperparameter` | Tune XGBoost params (max_depth, learning_rate, etc.) |
| `reweight_training_samples` | Upweight training data by phase/horizon/location |
| `toggle_feature` | Enable/disable feature groups (lag, rolling, yoy, etc.) |
| `adjust_floor_constraint` | Change post-prediction floor percentage |
| `change_target_transform` | Switch between log/raw/sqrt target transforms |
| `stop` | Declare convergence |

All actions have guardrails — the LLM cannot set values outside safe ranges.

## File structure

```
agent/
  cli.py              # All CLI commands
  orchestrator.py     # Two-agent improvement loop
  data_quality.py     # Pre-training data checks
  adapters/
    flu_forecast.py   # CDC FluSight adapter (actions, metrics, pipeline)
  llm_client.py       # Ollama/vLLM connection
  run_tracker.py      # SQLite history (outputs/agent_runs/runs.db)
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
