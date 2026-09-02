# 2026-09-02 — TS-Agent review + Nixtla direction

Weekly advisor sync. Reviewed the TS-Agent paper notes
(`../ts-agent-comparison-notes.md`) and set the fall direction.

## Decisions

- **Scope locked: internal exploration tool, not real-time.** The framework
  mimics the lab's manual "new model → try best practices → evaluate" loop to
  speed up how many candidate models get vetted. Real-time operation would be
  plain automation, not agentic decision-making.
- **Out of scope:** the forecast-summarization agent (LLM summarizing final
  submitted forecasts across states). Interesting, but explicitly deferred —
  focus stays on model training + model exploration.
- **Nixtla is the model bank.** Instead of hand-building competitor models, use
  the Nixtla suites (statsforecast / mlforecast / neuralforecast) as the uniform
  template: data in one long format, hyperparameters in one config shape, many
  models deployable with no per-model formatting work. This is the answer to the
  missing Stage-1 model-selection gap. (DARTS mentioned as an alternative; stick
  with Nixtla for now.)
- **Templates over code generation, confirmed.** TS-Agent credits its success to
  refining templates rather than having the LLM write model code from scratch —
  validates our constrained-catalog approach. Nixtla *is* the template.
- **Data window pinned:** train **2022 → 2025**; evaluate **Oct 2025 → May 2026**.
- **Optimization goal stays a parameter.** Advisor deliberately did not fix the
  objective (best-at-peak vs. best weekly average vs. best overall). The
  framework should accept the goal — ideally as a natural-language command — and
  train/select accordingly. Concrete goals will be assigned once the framework
  is set up.
- **Logging is the most important output.** Every run must produce a summary of
  what was tried, what it did, and where the benefits were. End state: assess
  the logs to decide which models enter the lab's own software.

## Research questions raised

- **Warm start vs. retrain-from-scratch.** Several in-house models (LSTM, GNN)
  can resume from last week's weights. But entering a new regime (e.g. flu
  season onset after a low period) may invalidate the warm start. The framework
  should evaluate *a priori* up to what point incremental fine-tuning beats a
  full retrain, per user-specified investigation, and log it.
- **Knowledge bank.** Advisor has ~9 years of best practices (e.g. "entering the
  season → window ≈ 12 weeks"). These belong in a curated, retrievable knowledge
  bank the agent consults — sourced from the lab's practices plus literature.

## Logistics

- Weekly sync: **Thursdays 2:30–3:00 PM**, via **Teams** (chat for async
  questions; may shuffle around school pickup / class conflicts; longer
  sessions scheduled separately when needed).
- **Publication target: ~January submission.** Explore + publish this framework.
- Next meeting goal: **Nixtla set up and working** toward the
  epi-optimization loop.
