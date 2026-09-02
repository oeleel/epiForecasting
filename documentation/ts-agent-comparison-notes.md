# TS-Agent vs. our agent framework — meeting notes

> Ref: Ang et al., *Structured Agentic Workflows for Financial Time-Series Modeling
> with LLMs and Reflective Feedback* (TS-Agent), NeurIPS 2025.
> Focus: Stage 1 (model pre-selection) + Stage 2 (code refinement) vs. our
> `agent/` improvement loop (Milestones 1–3).

---

## What we already have (shared ground, briefly)

- Iterative refine → retrain → evaluate feedback loop with in-loop memory (`agent/orchestrator.py`) — same shape as their Stage 2.
- Full audit trail — every iteration's metrics/diagnosis/action/config in SQLite (`runs.db`); we already meet their "auditability" claim.
- Domain-specific evaluation bank analog — PhaseEvaluator (WIS, coverage, signed bias by phase/horizon/location) is *more* specialized than their finance measures.
- Hard numeric guardrails + validated action catalog — stricter than their heuristic-guided free edits.
- Human-in-the-loop shadow mode + LLM output repair-retry — they have no interactive confirmation.

## Gaps — what TS-Agent has that we don't

1. **A model selection stage (their Stage 1).** They shortlist top-k candidates from a vetted Model Bank via case retrieval, then warm-up and pick an incumbent. Our loop is hardwired to the XGBoost `DirectForecastEnsemble` even though real competitors already exist in-repo (`src/nn_model.py`, `ClusteredDirectForecastEnsemble`). Biggest structural gap.
2. **Accept/revert semantics.** They keep an edit only if the loss improves, else revert. We commit `config = new_config` even on regression (orchestrator step 8) and keep searching from the worse config — best forecast is retained, but the search degrades.
3. **Case Bank / cross-run learning.** They retrieve similar past tasks to warm-start new ones. Our `runs.db` already collects everything needed but past runs never inform new runs — only within-run history reaches Agent 2's prompt.
4. **Refinement Knowledge Bank.** Their refinement agent is guided by curated best-practice heuristics linked to logged outcomes. Our Engineer gets only generic domain context (`get_domain_context()`) — for an 8B model that's a real handicap.
5. **Parallel round-robin warm-up.** Candidates get a few refine–tune–execute cycles each before committing to one. We're serial: one action, one full retrain, per iteration.
6. **Code-level refinement.** They edit `train.py` directly; we mutate a config dict. *Caveat: this is a deliberate mismatch, not a deficiency — their code-editing assumes GPT-4o/Claude-class backbones, while our constrained config-space keeps small local LLMs (Qwen 8B on Ollama/Rivanna) safe and debuggable. Not proposing we adopt it.*

## Things we liked / want to steal

- **The two-stage formalization** (selection → refinement) — the right mental model; makes explicit that we skipped selection entirely.
- **Revert-on-regression greedy acceptance** — smallest, highest-value-per-line port into our orchestrator.
- **"Refine vetted code, never synthesize from scratch"** — refining = bounded edits to the training harness around a known-good model (e.g., add early stopping / an LR schedule, change normalization or lookback window, switch to walk-forward validation), never generating model code. They credit their 100% success rate to this; independently validates our constrained-catalog philosophy.
- **Read-only knowledge resources** (case / code / refinement banks) as a clean separation from the loop — knowledge lives outside the agent, gets versioned and curated like data.
- **Warm-up before committing to an incumbent** — cheap short runs to de-risk the expensive optimization phase.

## Proposed next steps (ranked)

1. **Revert-on-regression** — on a regressed iteration, restore the previous config instead of committing. ~Few lines in `orchestrator.py`; immediate search-quality win.
2. **Model-selection warm-up (mini Stage 1)** — add `model.family` to the config; run XGBoost / clustered / NN briefly at the cutoff, pick the incumbent by WIS, then refine. `run_pipeline(config)` already takes a config dict, so plumbing is cheap.
3. **Epi refinement-heuristics bank** — small curated file (peak underprediction → target transform or peak reweight; low coverage → widen quantiles; …) injected into Agent 2's prompt. Lowest effort, likely large lift for a small LLM.
4. **Case retrieval over `runs.db`** — surface "what worked at similar cutoffs/phases in past runs" in the proposal prompt. Data is already collected; needs only a retrieval + formatting layer.
