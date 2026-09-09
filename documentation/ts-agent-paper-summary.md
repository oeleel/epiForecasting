# TS-Agent — detailed paper summary and what we take from it

> Ang, Bao, Jiang, Tao, Tung, Szpruch, Ni. *Structured Agentic Workflows for
> Financial Time-Series Modeling with LLMs and Reflective Feedback.*
> NeurIPS 2025. NUS / UCL / Edinburgh. PDF: `33_Structured_Agentic_Workflow.pdf`.
>
> Read in full 2026-09-09. This supersedes the second-hand notes in
> `ts-agent-comparison-notes.md`, which were written from the advisor's email
> summary rather than the paper. Where the two disagree, this file is correct.

---

## 1. What the paper is actually claiming

**The gap they target.** AutoML (AutoGluon, Optuna, Auto-WEKA) automates pipeline
construction but its search is static and optimizes generic statistical losses —
no domain alignment, no adaptivity. LLM agents (AutoGPT, DS-Agent, ResearchAgent)
automate end-to-end workflows but are not *robust, auditable, or compliance-ready*.
For finance specifically, practitioners need the decision trace, not just the number.

**Their answer.** TS-Agent formalizes time-series modeling as a **structured,
iterative decision process** over three stages (model selection → code refinement →
fine-tuning), where a planner agent is guided by **curated read-only knowledge banks**
rather than by free-form reasoning, and every decision plus its rationale is logged.

**The headline design commitment**, and the one that matters most for us:
they *refine vetted implementations* rather than synthesize model code from scratch.
They credit this for their 100% execution success rate.

---

## 2. Formalization (§3)

A task is `T = (desc, D, L)` — a **description**, data `D = (D_train, D_test)`, and an
evaluation criterion `L`. The agent produces an executable `train.py` that minimizes
`L` on `D_test` **while logging all decisions**.

For forecasting: windowed pairs `(X_{t-p+1:t}, X_{t+1:t+q})`, model
`f_θ: R^{d×p} → R^{d×q}`, typically MSE loss "often augmented with finance-aware metrics."

Note `desc` is a first-class part of the task tuple. **We have no equivalent** — our
runs are defined entirely by CLI flags, with no task description conditioning anything.
That is the formal gap behind the advisor's "dynamic commands" ask (§6.1 below).

### Action space and factorization

| Action | Meaning |
|---|---|
| `A_model` | choose models / evaluation measures |
| `A_refine` | insert training strategies |
| `A_tune` | tune hyperparameters |
| `A_log` | execute and record |

Edits are factorized as a **chain of code edits**:

```
π(A_code | C_t) = π(A_model | C_t) · π(A_refinement | A_model, C_t) · π(A_tune | A_model, A_refinement, C_t)
```

i.e. *pick the model first, then the training strategy, then the hyperparameters* —
each conditioned on the previous choice. Ours collapses all of this into one flat
`adjust_hyperparameter` action per iteration.

**Memory and context.** At step `t`, memory `M_t = (I_v, S_v)_{v≤t}` stores logs `I_v`
and **code states** `S_v`; context `C_t = M_t ∪ E ∪ T` conditions decisions. Because
`A_refine` can introduce bugs, the agent applies **iterate–fix–rerun debugging**.

---

## 3. The three read-only external resources (§3, Fig 1c)

This is the architectural heart of the paper — knowledge lives *outside* the agent,
curated and versioned like data.

**(1) Case Bank (`E_case`, text).** Curated financial TS forecasting and generation
tasks with concise reports, retrieved to guide new tasks. Sources: benchmarks,
competitions, peer-reviewed studies; data spanning equities, exchange, crypto,
synthetic. **Entries map tasks → effective model families.**

**(2) Refinement Knowledge Bank (`E_refinement`, text).** Best practices linked to
logged outcomes, in three categories:

| Preprocessing | Training Optimization | Tuning & Evaluation |
|---|---|---|
| Scaling | Early Stopping | Systematic Tuning |
| Normalization | Learning Rate Scheduling | Walk-Forward |
| Feature Extraction | Batching & Shuffling | Validation Monitoring |
| Leakage Handling | Weight Decay | Check for Overfitting |
| Augmentation | Dropout | |

**(3) Code Base (`E_code`).** Read-only repo containing a **Model Bank**
(forecasting: DLinear, Autoformer, PatchTST, TimesNet, TimeMixer, iTransformer;
generation: TimeGAN, RCGAN, PCFGAN, TimeVAE, CVAE, DDPM) and an **Evaluation Measure
Bank** (RMSE/MAE/MAPE/sMAPE; Sharpe/VaR/ES; MDS/Corr/AS/Cov).

---

## 4. The two stages (§4, Algorithm 1)

```
Initialize context C_1
for t = 1 to T_max:
    Conduct  Model Selection | Refinement | Fine-tuning
    Conduct  A_logging, record log I_t
    Update   memory   M_{t+1} ← M_t ∪ I_t
    Update   context  C_{t+1} ← (M_{t+1}, E, T)
```

**Stage 1 — Model Pre-selection.** Case-based retrieval from the Case Bank shortlists
the **top-k** candidates, which are instantiated in `train.py`.

**Stage 2 — Code Refinement**, run as a **two-phase round-robin search**:
- **Warm-up phase:** a *short, parallel* round-robin over the shortlisted candidates,
  to pick an incumbent cheaply before committing to expensive optimization.
- **Optimization phase:** iteratively apply Refinement and Fine-tuning, **accepting
  edits only if the loss improves — else revert.**

---

## 5. Results (§5)

**Setup.** Three financial datasets: Crypto (hourly, 20 USDT pairs, 2024), Exchange
(daily FX, 1990–2010), Stock (daily US equities, 2020–2024). Baselines: DS-Agent,
ResearchAgent, AutoGluon (forecasting), Optuna (generation). Four LLM backbones:
GPT-3.5, GPT-4o, Claude Sonnet 4, Nova Pro.

**Forecasting (Table 1).** Reduces RMSE by **>20% on Exchange** and **~8% on Crypto**
vs AutoGluon, and by **up to 30% vs DS-Agent**, while maintaining a **100% success rate**
across 5 runs. On risk fidelity (Table 2), ~**20% lower Sharpe/VaR differences** with
competitive ES.

**Generation (Tables 3–4).** Matches or exceeds Optuna, again at 100% success rate with
markedly lower error dispersion, while the generic agent baselines trail.

**The backbone-agnosticism finding — most important result for us.** GPT-4o is generally
strongest, *but the margins between backbones are smaller for TS-Agent than for the
baselines*, which the authors attribute to "backbone-agnostic resilience from refining
vetted Financial TS Code Base implementations rather than synthesizing models from
scratch." **This is published evidence that our constrained-action design lets a small
local model (qwen3:8b) compete on a task where a naive code-writing agent would need
a frontier model.** It is the strongest external justification for our architecture,
and belongs in our own paper's related-work framing.

### Case study (Fig 3) — the clearest picture of the loop

Task: predict the next three trading-day closes for ten US stocks from a 60-day
window; metric average MAPE.

| Step | Outcome |
|---|---|
| Stage 1 case-based pre-selection | shortlists **Autoformer** and **PatchTST** |
| Warm-up (best per model) | Autoformer **3.41** vs PatchTST **4.17** → Autoformer incumbent |
| Optimization (accept only loss-reducing edits) | Autoformer **1.86** |

Refinement alone took the incumbent from 3.41 → 1.86 MAPE, a **~45% improvement after
model selection had already finished**. That is the quantitative case for why the
refinement loop is worth having at all — and it is the number to beat/quote when we
argue our own loop earns its keep.

---

## 6. Where we stand against it

| TS-Agent component | Our status |
|---|---|
| Code Base → Model Bank | ✅ `src/model_bank/`, 9 families, one `ForecastModel` contract, in-house models via dotted path |
| Code Base → Evaluation Measure Bank | ✅ `PhaseEvaluator` (WIS, coverage, signed bias, by phase/horizon/location) — *more* domain-specialized than theirs |
| Stage 1 model pre-selection | ✅ `agent/model_selection.py` — rolling-origin warm-up, `SelectionGoal(metric, phase)` |
| Stage 2 accept-or-revert | ✅ landed 2026-09-03 (`f327adc`) — compares against best-so-far, reverts on regression |
| Full audit trail / logging | ✅ `runs.db` + (new) `agent/run_report.py` end-of-run report |
| Task `desc` conditioning the run | ⚠️ partial — `--goal "english"` exists for Stage 1 only (`agent/goal_parser.py`) |
| **Case Bank / cross-run retrieval** | ❌ nothing; `runs.db` has the data but never informs a new run |
| **Refinement Knowledge Bank** | ❌ nothing; Agent 2 gets only generic domain context |
| **Parallel round-robin warm-up** | ❌ serial (`model_selection.py:184` loops families, `:190` loops cutoffs) |
| Chain-of-code-edits factorization | ❌ flat single action per iteration |
| Code-level editing of `train.py` | 🚫 **deliberately rejected** — their code editing assumes frontier backbones; our constrained config-space is what keeps qwen3:8b safe. Their own backbone-agnosticism result supports this. |

---

## 7. Proposals for the next phase

Ranked by (advisor priority × leverage) ÷ effort.

### 7.1 A task spec — dynamic commands, and commands from a file

**This is one feature, not two.** The paper's `T = (desc, D, L)` is exactly what we
lack: a single object naming what to investigate, on what data, judged how.

Design:
- A `TaskSpec` dataclass: `description`, `families`, `cutoffs`/window, `goal`
  (metric+phase), `max_iterations`, `train_mode` (retrain vs fine-tune), `constraints`.
- **Two front doors to the same object:**
  - *Natural language* for exploration —
    `agent run --task "compare NHITS and XGBoost over the 2025-26 season, optimize for
    the peak, and tell me whether fine-tuning beats retraining"`. Extends the existing
    `goal_parser.py` ladder (keywords → LLM → repair retry → fallback) from just
    metric+phase to the whole spec.
  - *A YAML file* for reproducibility —
    `agent run --spec experiments/peak-vs-average.yaml`.
- **The two compose, and that is the point:** NL parsing *emits* the YAML. You state
  the goal in English once, the framework writes the spec file, and every later run
  is byte-reproducible. That directly fixes the reproducibility hole I flagged when
  we set `temperature=0.0` — an LLM in the loop is fine for authoring, not for
  re-running an experiment we will cite in January.
- A spec directory also gives us **batch execution**: queue ten specs, run them
  overnight, and the January results table writes itself from the run reports.

Effort: moderate. `goal_parser.py`, `run_report.py`, and the CLI already exist;
this is mostly a dataclass, a YAML loader, and a `run` subcommand.

### 7.2 An epi Refinement Knowledge Bank

Their Fig 1c categories, translated to epidemic forecasting. The advisor has ~9 years
of these and named one unprompted ("entering the season → window ≈ 12 weeks").

Seed entries, in their three-category shape:

| Preprocessing | Training Optimization | Tuning & Evaluation |
|---|---|---|
| backfill/revision handling (CDC revises recent weeks) | window matched to phase (12wk at onset) | walk-forward by epiweek |
| log1p target transform | sample reweighting by phase | per-phase WIS, not just overall |
| leakage: never use post-cutoff data | early stopping on a held-out season | coverage calibration check |

Delivery: a curated, version-controlled markdown/YAML file, retrieved and injected
into Agent 2's proposal prompt. **Highest expected lift per line of code for a small
LLM** — today's live runs showed qwen3:8b proposing three plausible-but-wrong
hyperparameter edits in a row precisely because it had no priors to draw on.

### 7.3 Phase-based training (not just phase-based evaluation)

Today we *evaluate* by phase but *train* uniformly; the only phase-aware lever is
`reweight_training_samples`. Proposals, in increasing ambition:

1. **Phase-matched training window** — the advisor's 12-week rule as an action:
   `set_training_window(weeks=12)`, selected by the phase at the cutoff.
2. **Phase-conditioned model selection** — already possible via
   `SelectionGoal(phase=...)`; surface it as "which model do I switch to at each
   phase boundary?" The committed Stage-1 run already shows the answer is not
   constant (XGBoost wins overall, NHITS wins the decline phase).
3. **Regime-change-aware warm start** — the advisor's actual research question and
   our Workstream 4: at a phase boundary, is last week's model still a valid warm
   start, or does the regime change invalidate it? `ForecastModel.get_state`/
   `set_state` hooks already exist and are unused. This is the paper-differentiating
   experiment: TS-Agent has no notion of regime at all.

### 7.4 Cheaper wins worth queueing

- **Parallel warm-up** (their Stage 2 phase 1). Ours is serial; the full `select-model`
  run takes ~10 min and is embarrassingly parallel across families.
- **Case retrieval over `runs.db`** (their Case Bank). Data is already collected;
  needs a retrieval + formatting layer into Agent 2's prompt.
- **Chain-of-edits factorization** — condition the hyperparameter proposal on a chosen
  *strategy* rather than proposing a raw knob. Likely helps a small model a lot.

---

## 8. Open questions for the advisor

1. Does he want the task spec to also carry the **investigation** ("find where
   fine-tuning stops being enough"), or just the objective? The former makes the
   framework an experiment runner, not just a tuner — closer to what he described.
2. Which of his best practices should seed the Refinement Knowledge Bank first?
   This is the one input only he can supply, and it gates 7.2.
3. For the January paper: is the contribution framed as *epi-specialization of
   TS-Agent* (phase-aware evaluation + regime-aware warm start + small-LLM safety),
   or as a general framework that happens to be demonstrated on flu?
