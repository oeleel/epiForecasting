# Status 2026-09-03 - what works, how to demo it, what is left

Verified against `main` at `26cc786` on 2026-09-03. Every claim under "What
works" was executed, not read from docs. Companion to
`fall-2026-roadmap.md` (the plan) and `MODEL_BANK.md` (the contract).

## What works (verified)

- **Tests:** `tests/agent/` - 138 passed in <1 s (`-W error::DeprecationWarning`).
- **Data:** cache refreshed 2026-09-02, covers 2022-02-05 .. 2026-07-04, i.e.
  the whole eval window. Roadmap 0.3 done.
- **Pinned split:** `TRAIN_START_DATE`, `EVAL_START_DATE`, `EVAL_END_DATE`,
  `generate_eval_cutoffs()` in `src/config.py`. Roadmap 0.4 done.
- **Stage 1 model bank + selection (Workstream 1, the Thursday goal):**
  `scripts/demo_stage1.py` quick mode runs end to end in ~13 s. 9 families
  load (2 baselines, xgboost_direct, nn_quantile, 5 Nixtla). Warm-up,
  phase-aware WIS, re-ranking by goal without refitting, JSON + markdown
  report all work. Full run (all states, 9 cutoffs, 6 families) is committed
  at `outputs/model_selection/demo_2026-09-02.json`:

  | rank | family | overall WIS |
  |---|---|---|
  | 1 | xgboost_direct | 50.6 |
  | 2 | nf_nhits | 65.6 |
  | 3 | sf_autoets | 82.2 |
  | 4 | persistence | 90.5 |
  | 5 | mlf_lightgbm | 138.7 |
  | 6 | seasonal_naive | 153.2 |

  Quick mode (6 states, 3 cutoffs) ranks sf_autoets first. Rankings flip at
  small scale; cite the full run, never the smoke test.
- **Milestone 1 summarize:** `scripts/demo_milestone1.py --no-llm` runs -
  phase / horizon / location metrics, worst segments, signed bias.
- **Data quality agent:** `python -m agent check-data --dry-run` runs; flags
  ~180 spikes plus gaps and zero-reporting.
- **Milestones 2-3 (improvement loop + tracking):** code, tests, and the
  `history` / `status` / `compare` CLI are in place. NOT exercised live on
  2026-09-03: no LLM reachable (no Ollama on the desktop, `runs.db` empty).
  This is the one path without fresh end-to-end evidence.

## Demo runbook for the researcher

Order it as the loop the lab runs by hand: is the data clean -> which model
do I start from -> refine it -> what did we learn. Steps 1-4 need no LLM.

```bash
cd ~/repos/epiForecasting && source .venv/bin/activate

# 1. Is the data trustworthy?  (~10 s)
python -m agent check-data --cutoff-date 2026-01-24 --dry-run

# 2. Which model do I start from?  Quick smoke live (~15 s); cite the full run
python scripts/demo_stage1.py
python -m agent select-model --stride-weeks 4 --exclude-locations US \
    --json outputs/model_selection/thursday.json \
    --report outputs/model_selection/thursday.md        # ~10 min, run the night before

# 3. "The goal is a parameter" (advisor's explicit ask) - same warm-up, no refit
python -m agent select-model --stride-weeks 4 --exclude-locations US --metric wis --phase peak

# 4. Plugging in a lab model = two methods, no registration
python -m agent list-models
#    walk through documentation/MODEL_BANK.md (the FluLSTM recipe)

# 5. Hand the incumbent to the agents  (needs an LLM, see below)
python -m agent improve --cutoff-date 2026-01-24 --model-family nf_nhits \
    --auto-apply --max-iterations 3

# 6. What happened
python -m agent history
python -m agent status <run_id>
```

Talking points mapped to the 09-02 decisions:
- Nixtla as the template -> step 2.
- Templates over code generation -> step 5: Agent 2 can only tune what the
  family's `param_space()` declares.
- Goal as a parameter -> step 3.
- In-house models are the real bank, Nixtla is scaffolding -> step 4.

### Prep before Thursday

1. **Get an LLM reachable.** Either `ollama serve` + `ollama pull qwen3:8b`
   locally, or tunnel to the Rivanna vLLM and set `LLM_BASE_URL` /
   `LLM_MODEL`. Without it step 5 is a code walkthrough only.
2. **Dry-run step 5 with `--fake-pipeline`** first. The family-aware action
   catalog has no live LLM evidence yet; make sure prompts still parse.
   ```bash
   python -m agent improve --cutoff-date 2026-01-24 --auto-apply --fake-pipeline \
       --forecast-csv outputs/quantile_hindcasts/quantile_forecasts_nov_apr.csv \
       --no-regenerate-baseline
   ```
3. **Run the full `select-model` the night before** and keep the report open.

## What still needs implementing

Roadmap order. Items 0.3, 0.4, and Workstream 1 are done and ticked in
`fall-2026-roadmap.md`.

1. **3.1 Revert-on-regression.** `agent/orchestrator.py` (~line 488) counts a
   regression streak and stops after 2 but keeps the regressed config. Must
   restore the previous best. Small; do first.
2. **2.3 Natural-language goal.** Selection takes `--metric --phase` flags
   only. The "say in English what to optimize" layer does not exist.
3. **3.2 Case retrieval over `runs.db`.** Nothing in the prompts consults past
   runs.
4. **Workstream 4, warm start (paper centerpiece).** `get_state` / `set_state`
   hooks exist on `ForecastModel` but nothing calls them. No `fine_tune` vs
   `retrain_from_scratch` action, no regime-change flag, no experiment harness.
5. **Workstream 5, knowledge bank.** No heuristics file, no prompt injection.
6. **Workstream 6, run reporting.** Selection writes a report; the improvement
   loop has no end-of-run "what was tried / where the benefit came from"
   report and there is no cross-run view. Advisor called this the most
   important output.
7. **Workstream 7, paper.** Outline by November, results freeze December.
