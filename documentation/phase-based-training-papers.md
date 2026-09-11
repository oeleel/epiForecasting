# Phase-based training — the two Adiga papers, and what they change for us

> **P1** — Adiga, Kaur, Wang, Hurt, Porebski, Venkatramanan, Lewis, Marathe.
> *Phase-Informed Bayesian Ensemble Models Improve Performance of COVID-19
> Forecasts.* **AAAI-23 / IAAI-23**, pp. 15647–15653.
> `03953-IAAI23.AdigaA-APP.pdf`
>
> **P2** — Adiga, Kaur, Hurt, Wang, Porebski, Venkatramanan, Lewis, Marathe.
> *Enhancing COVID-19 Ensemble Forecasting Model Performance Using Auxiliary
> Data Sources.* **2022 IEEE Big Data**, pp. 1594–1602.
> `Enhancing_COVID-19_Ensemble_Forecasting_Model_Performance_Using_Auxiliary_Data_Sources.pdf`
>
> Both read in full 2026-09-09. **These are the advisor's own group** —
> Biocomplexity Institute, UVA; Aniruddha Adiga is first author on both.
> P2 is the Big Data paper he mentioned as a Best Paper winner. This is the
> deployed UVA-ENSEMBLE system that submits to the CDC COVID-19 ForecastHub.

---

## 1. Headline: our phase definition disagrees with theirs

This is the most important thing to come out of reading these, and it is worth
raising first on Wednesday.

| | Their phases | Ours (`agent/phase_evaluator.py`) |
|---|---|---|
| Names | **Surge (growth)**, **Plateau**, **Decline** | Onset, Peak, Decline, Off-season |
| Defined by | **rate of change of the curve** | **calendar month** |
| How assigned | piecewise-linear fit → breakpoints → ±10% rule | Oct–Nov, Dec–Jan, Feb–Apr, May–Sep |
| Re-estimated | **every week**, as new data arrives | never — fixed by the calendar |

Two mismatches, not one:

1. **We have no `plateau`; they have no `peak`.** For them the peak is the
   instant between surge and decline, not a phase you train on. Their third
   phase is the flat stretch, which is exactly where they find AR/Kalman methods
   win — a regime we currently cannot even name.
2. **Calendar months are a proxy for a thing they measure directly.** The
   advisor said it out loud in the meeting: *"We generally know that during
   October things start to grow, but the growth rate is not the same every
   season."* Our `assign_phase()` hard-codes the assumption his own paper's
   method exists to avoid.

Consequence: every per-phase number we have produced — including the Stage-1
finding that NHITS wins the "decline" phase — is computed against phase labels
the lab would not accept. The finding may well survive re-labelling, but right
now it is measured with the wrong ruler.

---

## 2. Their phase segmentation, concretely (P2 Algorithms 1 and 3)

Fully specified in the papers and cheap to implement.

**Algorithm 1 — recursive piecewise-linear fit.** Approximate the case curve
with a piecewise-linear function and extract breakpoints `{b_1 … b_m}`
(R package `segmented`, Muggeo 2008). Start with a window `t_0 = 15`, fit, then
walk forward; at each step re-fit only from the **most recent two breakpoints**
onward. That last detail is what makes it real-time-safe: new data refines the
recent phases without silently rewriting history.

**Algorithm 3 — phase classification**, with threshold `δ` (they use **10%**):

```
for each interval (b_i, b_{i+1}]:
    if   y[b_{i+1}] > (1 + δ)·y[b_i]:   phase = Surge
    elif y[b_{i+1}] < (1 − δ)·y[b_i]:   phase = Decline
    else:                                phase = Plateau
```

They are explicit that phase definitions are **subjective** and may be
user-annotated or produced by standard change-point detection — so `δ` and the
segmentation method are legitimately tunable knobs, i.e. exactly the sort of
thing our agent could be allowed to select.

---

## 3. Phase-based training = selective sampling of history

This is the mechanism, and it is *not* what our `reweight_training_samples`
action currently does.

Standard BMA (their prior deployed system) estimates ensemble weights over the
**previous N contiguous weeks**: `T = {T−1, T−2, …, T−N}`. Recency captures
non-stationarity, but it introduces a **latency in picking the best model at a
phase change** — the ensemble must watch the newly-best model win for several
weeks before its weight rises, and by then the phase may have turned.

Phase-Informed BMA (PI-BMA) replaces that window with a **phase-matched set**:
let `T_S`, `T_D`, `T_P` be all historical week indices in surge / decline /
plateau. For the current phase `r`, estimate weights and variances over `T_r`
only (their Eq. 3):

```
w_{k,r} = (1/|T_r|) Σ_{t ∈ T_r} z_{k,t}
σ²_{k,r} = Σ_{t ∈ T_r} z_{k,t}(y_t − f_{k,t})² / Σ_{t ∈ T_r} z_{k,t}
```

In the advisor's words in the meeting: *"we'll only pick all the growth phases
that you've seen over multiple seasons … where you overemphasize the growth
phase and underemphasize the other phases."*

So the training set is **selected by phase across all seasons**, not weighted
within a recent window. Our current action can approximate the emphasis but not
the selection.

---

## 4. Which model class wins in which phase (P1 ablation)

They measure each model's marginal contribution with a Shapley-value-style
ablation over ensemble subsets. Findings, quoted in substance:

- **Compartmental (SEIR)** — useful during growth and decline, but **over-predict
  during the surge and decline phases**. SEIR matches early exponential growth,
  then overestimates in subsequent weeks; because past performance was strong it
  still receives high weight, *dragging the ensemble down*.
- **Purely data-driven (LSTM)** — **latency** in picking up a phase change, but
  learns the new pattern quickly once it does.
- **Statistical AR / Kalman filter** — superior during the **relatively steady
  (plateau)** phase.

This is direct, published support for **phase-conditioned model selection**, which
our Stage 1 already supports through `SelectionGoal(metric, phase)`. Our own
committed run found the same shape of result — XGBoost best overall, NHITS best
in the "decline" phase — so we have independently reproduced their central
observation on flu data with a different model bank. That is a genuinely useful
thing to say on Wednesday.

---

## 5. Results worth quoting

Both papers use **WIS** — the same metric we optimize — so the comparison is
apples-to-apples with our work.

**P1 (IAAI-23).** Median ranking among *The Hub* models, 4-week ahead:
PI-BMA ranks ~6–7 overall, but **4 of 9 during the Delta surge and 1st during the
Omicron surge** — "regime where the performance of most models from the CDC
forecasting hub dropped significantly." It beats `COVIDhub_ensemble` and
`COVIDhub-trained_ensemble` in those critical phases despite those aggregating
highly tuned individual models.

**P2 (Big Data 2022).** Same story with the auxiliary-data machinery added:
4-week ahead, **median ranking 2 of 9 (Delta surge)** and **2 of 6 (Omicron
surge)**.

Their own summary of why it matters: *"This validates the use of selective
sampling of training data by ensembling methods."*

Note the shape of the claim — **overall performance is merely comparable; the win
is concentrated in the critical phases.** That is a template for how our January
paper should frame its own results, and a caution against reporting only
season-aggregate WIS.

---

## 6. P2's extra machinery: leading indicators and phase *prediction*

P2 adds two components beyond P1.

**Transfer entropy for leading indicators.** Auxiliary signals — doctor visits
for COVID-like illness (DV-CLI), Facebook-survey CLI (FB-CLI), antigen test
positivity, Google Health Trends (GHT), hospitalizations — are screened by
**transfer entropy** rather than Granger causality, because the signals rarely
have a linear dependence on the target (computed with the `IDTxL` toolbox).
Critically, **which signals lead changes over time**: hospitalizations and DV-CLI
led during Delta; only GHT stood out during Omicron. So leading indicators must
be **re-identified every week**.

**Phase prediction.** Knowing today's phase is not enough — you want next week's.
They model the phase series `P(t)` as a Markov process depending on its own past
`τ` values *and* the past phases of the leading indicators, then predict `P(t+1)`
by querying historically observed sequences that match the current one and taking
empirical frequencies (their Fig. 5).

**Relevance to us, honestly assessed:** the CDC FluSight target data we use is a
single signal, so the transfer-entropy layer has nothing to chew on today. It
becomes relevant only if we add auxiliary streams (ILINet, Google Trends, wastewater).
Worth noting as a direction, not a near-term build. They also flag that these
streams undergo **substantial backfill/revision** and that they deliberately do
*not* nowcast — using unrevised data as observed on the forecast date. That is a
leakage discipline we should copy explicitly, and it belongs in the knowledge bank.

---

## 7. What this changes in our plan

Ranked.

1. **Re-implement phase assignment as curve-based, not calendar-based.**
   Port Algorithms 1 + 3. Keep the calendar labels available for backward
   comparison, but make the segmented phases the default for both evaluation and
   training. This is the single highest-value item: it aligns us with the lab's
   own published method, adds the missing `plateau`, and it is a contained,
   testable module.
2. **Add phase-based training as a first-class action** — `select_training_phase`,
   selecting *all historical weeks of the current phase* rather than a recent
   window. This is the paper's actual mechanism and it is not what our existing
   `reweight_training_samples` does.
3. **Seed the Refinement Knowledge Bank from these two papers.** They supply
   ready-made, citable entries — the model/phase affinities in §4, the ±10% rule,
   the no-nowcasting leakage discipline, and the short-window rule (see §8).
4. **Re-run the Stage-1 comparison under the new phase labels** and see whether
   "NHITS wins the decline phase" survives. If it does, we have replicated their
   finding on a new disease and a new model bank — a real result for January.
5. **Report per-phase, not just aggregate.** Their headline is that the win lives
   in the critical phases. Our run report already breaks out by phase; the paper
   framing should follow theirs.

---

## 8. The short-window rule — and the fact that our agent already rediscovered it

The advisor explained why long lookbacks hurt: with ~52 weeks of history the model
*"will just infer, based on seasonality … it'll probably underpredict and just
give you what it saw last season,"* so they deliberately use a short window to
stay agile.

**Our own live run on 2026-09-03 produced exactly this failure.** With `nf_nhits`
at the 2026-01-24 cutoff, Agent 2 proposed `input_size: default → 52` — a
one-year lookback — and WIS went from **101.67 to 533.06**, five times worse. It
then tried `input_size = 104` and still sat at 300.52, never recovering.

That is a small but genuine result: the framework independently reproduced a
failure mode the lab already knows about, and the revert-on-regression fix caught
it. It is also the perfect first knowledge-bank entry — had the bank existed,
Agent 2 would have been told not to reach for a 52-week lookback, and the run
would not have burned three iterations discovering it.

---

## 9. Open items from the meeting, beyond the papers

- **Knowledge-bank override.** He asked directly: what if he wants to intervene
  *on the fly* — *"don't do any of those, I would like to suggest certain
  things"*? So the design needs precedence: **user directive > knowledge bank >
  model's own priors**, settable per run.
- **A file with a list of commands.** He described pointing at a file holding
  *a list* of commands that the framework "goes through" — a queue of tasks, not
  just one spec. Batch execution should be in the first cut.
- **He will supply a list of things to try** once we agree the scope — that list
  is the knowledge bank's seed content and is the one input only he can provide.
- **Compute.** He is restarting the Rivanna/NSAC access thread himself (Dustin is
  on vacation, plus HR/Beth for enrollment) — **we should not send that email**;
  he said he would handle it. Meanwhile: plan the port off the laptop. The AMD GPU
  would need ROCm; the biocomplexity student queue has NVIDIA cards and is the
  better target. He offered their (possibly outdated) setup documentation.
- **Multi-model routing** (our idea, he did not object): a stronger reasoning model
  for the Analyst, a cheap one for the Engineer. Worth a line in the framework
  design; it also makes the Rivanna port more valuable.
- **Still unanswered, and worth asking plainly:** what the day-to-day use actually
  looks like for them. He described exploration ("somebody comes up with a model,
  I want to evaluate how good it is") but the loop between that and their weekly
  operational cadence is still fuzzy on our side.
