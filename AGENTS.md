# epiForecasting

Agentic framework for epidemiological forecasting: an LLM agent reads structured forecast
metrics, diagnoses weak segments, and takes constrained improvement actions, replacing the
manual "run model, inspect CSVs, tweak, re-run" loop. Built on an XGBoost flu-hospitalization
pipeline (CDC FluSight data).

Two layers:
1. Domain-agnostic orchestration (`agent/`): plain-Python orchestrator (no LangGraph), LLM
   client, prompts, run tracker, knowledge bank.
2. Flu adapter (`agent/adapters/flu_forecast.py`) wrapping the pipeline in `src/`.

Project status, milestones, and open items: `documentation/STATUS.md`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r documentation/requirements.txt pytest langchain-openai
uv pip install --python .venv/bin/python -r documentation/requirements-nixtla.txt   # optional families

# macOS only: xgboost and torch bundle separate libomp.dylib copies; running both in one
# process (select-model, improve) segfaults without these
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

`langchain-openai` is not in `requirements.txt`; install it for any agent work.

## Commands

```bash
# Tests (the real suite is tests/agent/)
.venv/bin/python -m pytest tests/agent/ -q -W error::DeprecationWarning
PYTHONPATH=. .venv/bin/python tests/agent/run_all.py          # no-pytest fallback

# Agent CLI: python -m agent <cmd>
#   check-data, summarize, improve, history, status, compare, report,
#   list-models, select-model, knowledge
python -m agent summarize --forecast-csv <path> --dry-run     # metrics only, no LLM
python -m agent summarize --forecast-csv <path> --base-url http://localhost:8247/v1 --verbose
python -m agent select-model --stride-weeks 4 --exclude-locations US --json outputs/model_selection/run.json
python -m agent select-model --families persistence my_lab.models:FluLSTM --metric wis --phase peak
python -m agent improve --cutoff-date 2025-12-06 --model-family mlf_lightgbm --auto-apply [--no-knowledge]
python -m agent report <run_id> [--rebuild] [--out report.md]

# Knowledge bank: curated YAML in knowledge/curated/ -> SQLite knowledge/knowledge.db (gitignored)
python -m agent knowledge {validate,rebuild,list,query,import-experiment,remove} ...
python scripts/demo_knowledge_bank.py [--live] [--json PATH]

# Peak-rectification experiment harness (slow: run in the background)
.venv/bin/python scripts/experiments/peak_rectification.py --seeds 42 43 44
.venv/bin/python scripts/experiments/peak_rectification.py --cutoffs 1 --arms baseline   # smoke
.venv/bin/python scripts/experiments/peak_rectification.py --summary-only            # rebuild summary.md
.venv/bin/python scripts/experiments/peak_rectification.py --rescore --out <dir>       # re-score saved forecasts
.venv/bin/python scripts/experiments/peak_rectification.py --merge <d42> <d43> --out <merged>

# Underlying pipeline
python scripts/pipeline.py --cutoff-date 2024-11-02 [--locations US 06 12 48]
python scripts/optimize.py --n-trials 50
python -m src.data_loader update                              # refresh the CDC cache
```

The harness flags and design are in its module docstring (`--arms`, `--lambdas`,
`--calendar-lambdas`, `--windows`, `--oversample-factors`).

## Gotchas

- **Phase schemes.** Phases are named calendar schemes in `PHASE_SCHEMES`
  (`agent/phase_evaluator.py`). Default `adiga_2026_10`: Growth Nov 1-Dec 31, Peak Jan 1-end
  Feb, Decline Mar 1-Apr 15, Plateau Apr 16-May 31, Off-season Jun 1-Oct 31 (skipped).
  October = off-season is an assumption awaiting A. Adiga (`ADIGA_OCTOBER_PHASE`). Legacy
  `calendar_v1` stays selectable.
- **Never compare metrics across phase schemes.** Every metrics dict records `phase_scheme`;
  a missing value means `calendar_v1` (every output before 2026-10-15, including round 1 of
  the peak-rectification sweep). `run_report` and `select_incumbent` refuse cross-scheme
  comparisons. A run report never claims an improvement across different evaluation windows.
- **`by_phase` sample weights label the TARGET date** (origin + horizon weeks), as the
  evaluator does, and raise on a phase outside the scheme (e.g. legacy `onset`).
- **Pinned split** (`src/config.py`): `TRAIN_START_DATE` 2022-02-05, eval
  `EVAL_START_DATE`..`EVAL_END_DATE` = 2025-10-01..2026-05-31, cutoffs from
  `generate_eval_cutoffs(stride_weeks)`. Do not move it; every family comparison depends on it.
- **Harness resumes are pinned.** The manifest pins cutoffs, seeds, phase scheme, and the
  data-cache sha256; a mismatched resume fails. Config ids end in `__seed<N>`; logged ids
  are skipped.
- **`sample_weights.approaching_peak`** (`src/direct_forecast.py::_compute_sample_weights`)
  only labels seasons complete relative to the data in hand with >= 40 observed weeks, so the
  in-progress season is never labelled. XGBoost's validation split is positional (last ~20%
  of locations), so weights on those rows are unused.
- **`data.train_window_weeks`** (`src/pipeline.py::run_pipeline`) fits on the last N weeks of
  feature-engineered rows with lags intact; minimum 8. Not the same as
  `data.train_start_date`, which truncates history before feature engineering.
- **`data.oversample`** works for `xgboost_direct` only.
- **Model family** is `config["model"]["family"]`; default `xgboost_direct` is the legacy
  path, every other family dispatches through `src/model_bank/`. In-house lab models are the
  real bank; the Nixtla families are example scaffolding (`documentation/MODEL_BANK.md`).
- **Targets are `log1p`** (`TARGET_MODE = "log"`), inverted with `expm1` at inference.
  Floor constraint: predictions cannot drop below 30% of the last known value (decays per
  horizon).
- **`src/` is stable infrastructure.** Change it through agent actions, not manual edits.
- **`scrap/`** (demoted NL goal parser) is never imported, tested, or maintained.
- **Curated knowledge files starting with `_`** (templates, intake stubs) are skipped. To
  delete a curated entry, edit its YAML and `knowledge rebuild`; `knowledge remove` is for
  derived and experiential entries.
- **LLM endpoint:** Ollama on localhost with `qwen3:8b` by default; vLLM on Rivanna (Qwen
  2.5 72B AWQ) via `--base-url` or `LLM_BASE_URL`. If the LLM is unreachable, the agent
  prints raw metrics instead.

## Map

| Path | What |
|---|---|
| `agent/` | orchestrator, LLM client, prompts, run tracker (`outputs/agent_runs/runs.db`), phase evaluator, model selection, run report, CLI |
| `agent/knowledge/` | knowledge bank code (design: `documentation/knowledge-bank-design.md`) |
| `knowledge/` | bank content: `curated/*.yaml`, `README.md` for the lab's authoring guide |
| `src/` | pipeline: data loader (1-week cache in `data/raw/`), feature engineering, direct-forecast ensembles, NN alternative |
| `src/model_bank/` | `ForecastModel` contract, registry (`@register` or `pkg.mod:Class`), data bridge, runner |
| `scripts/experiments/` | controlled experiments whose logs feed the bank |
| `tests/agent/` | the test suite |
| `documentation/` | roadmap, design docs, reports, `STATUS.md`; `archive/` is superseded |
