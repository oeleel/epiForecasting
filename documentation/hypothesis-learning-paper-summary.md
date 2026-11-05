# Hypothesis learning (Ziatdinov et al. 2021) - summary and mapping to this project

Paper: M. Ziatdinov, Y. Liu, A. N. Morozovska, E. A. Eliseev, X. Zhang, I. Takeuchi,
S. V. Kalinin, "Hypothesis learning in automated experiment: application to combinatorial
materials libraries", arXiv 2112.06649 (2021). Local copy:
`documentation/papers/2112.06649.pdf`. Code: `github.com/ziatdinovmax/hypoAL` (p18).

Page numbers are PDF page numbers. The main text prints the same numbers (p1 is the DOE
notice, the abstract is p2). The supplementary material is PDF p25-31 and has no printed
page numbers.

Why this paper: the PI named it as the model for a "Bayesian abductive loop" in this project.
Section 6 maps it onto our framework and section 7 proposes the smallest build. Statements
marked **(inference)** are my reading, not claims the paper makes.

---

## 1. The problem

- Grid search of composition spaces is impractical. A quaternary alloy with a property that
  peaks within 1% of a phase boundary needs about 10^6 grid samples (p3).
- GP-based active learning (GP-AL) and Bayesian optimization (GP-BO) are purely data-driven.
  The mean function is almost always zero and all structure lives in a generic kernel (RBF,
  Matern). The kernel is "not directly linked to the physics of the system" (p5). Such GPs
  are prone to trivial solutions, need many steps, and are limited to low dimensions (p3).
- Reinforcement learning needs far too much data for an experiment of tens of points (p3).
- Classical Bayesian inference fits a known physical functional form with priors, but does
  not choose where to measure next (p5).
- Goal: "co-navigation of the hypothesis and experimental spaces" (p2, Fig 1 p4). Several
  competing physical models with adjustable parameters are tested during one experiment.
  The loop both reconstructs the measured function from few points and identifies which
  model describes the system. The authors call this "hypothesis learning" (p3, p5).

## 2. The method

### 2.1 Objects

- **Data** `D = {(x_i, y_i)}`: measured points. Here `x` is 1-D (concentration, or a
  synthetic control variable) and `y` is a scalar property (p5, p8).
- **Unmeasured points** `X_*`: a finite candidate list. Each step moves one point from
  `X_*` to `D` (Algorithm 1, p6).
- **Hypotheses** `M`: a fixed list of probabilistic models of `y(x)`. Each is a parametric
  function with priors on its parameters (p5, p8). The method assumes that exactly one model
  in the list is correct, without knowing which one (p5).
- **Two forms of hypothesis** (p5, p9-10):
  1. *Standalone*: `y = m(x; theta) + noise`, with Bayesian priors on `theta`.
  2. *Structured GP (sGP)*: `f ~ MVN(m(x; theta), K(x, x'; psi))`, `y = f(x) + eps`. The
     physical model replaces the GP's zero prior mean. The kernel is Matern. NUTS infers the
     mean-function parameters `theta` and the kernel hyperparameters `psi` jointly (p9-10).
     The mean function "is expected to capture general or even partial trends in the data,
     but it does not have to be precise" (p10). The kernel absorbs what the model misses.

### 2.2 Inference

- Per model, Bayes' rule `P(theta | D) = P(D | theta) P(theta) / P(D)` (Eq 1, p6).
- Sampling uses NUTS in NumPyro (p6). The supplement reports 5000 NUTS warm-up steps and
  5000 samples (p30).
- Posterior predictive mean at each `x_*` is the average over posterior draws (Eq 2a, p6).
  Posterior predictive variance is `V[f_*] = (1/N) sum_n (f_*^n - f_hat_*)^2` (Eq 2b, p7).
- **Total uncertainty** `V_tot = sum over X_* of V[f_*]`, the sum of predictive variances
  over every unmeasured point (Algorithm 1, p6). The real experiment uses the median
  instead of the sum (p7, Fig 7a p14).

### 2.3 Priors

- Synthetic example: uniform prior on the transition point `x_c`, normal priors on slopes
  and power-law coefficients (p8).
- Supplement: `Uniform(0, 1)` on `x_c`, `Normal(0, 1)` on the amplitude parameters (p25).
  The "wrong" control model `a x^2 + b` also has `Normal(0, 1)` priors (p27).
- Real experiment: uniform prior on the transition location, log-normal priors on every
  other model parameter and on the GP kernel hyperparameters. The priors "do not change
  during the experiment" (p14).
- No prior probability is placed on the models themselves. Fig 1 shows "Model n, prob_n"
  (p4), but Algorithm 1 never computes a model probability (p6) **(inference from
  Algorithm 1 and the text on p6-8)**.

### 2.4 The loop (Algorithm 1, p6)

Inputs: seed measurements `D`, candidates `X_*`, model list `M`, sample-averaged rewards
`R_a` (initialized to zero), `N_warmup`, `N_steps`, reward function `R`, and epsilon.

**Warm-up phase** (repeat `N_warmup` times):
1. For every model in `M`: run NUTS on `D`, compute `V[f_*]` over `X_*`, store `V_tot`.
2. Reward the model with the lowest `V_tot`.
3. That model's uncertainty map picks the next point: `x_next = argmax V[f_*]`.
4. Measure at `x_next`; move it from `X_*` to `D`.

Then average the warm-up rewards into `R_a`. The warm-up gives the exploration phase
"momentum" (p5). It is expensive because every model runs full Bayesian inference, so the
authors use about 1-5 warm-up steps (p7).

**Exploration phase** (repeat `N_steps` times):
1. Epsilon-greedy choice: with probability epsilon a random model, else the model with the
   highest `R_a` (p6-7). Epsilon = 0.4 in every experiment, chosen for exploration over
   tens of points. Annealing epsilon or Thompson sampling would work but are not tested (p8).
2. Run NUTS for that one model only. Compute `V_tot^i`.
3. Reward (Eq 4, p8): `R = +1` if `V_tot^i < V_tot^(i-1)`, else `-1`. The comparison is
   with "the total uncertainty recorded in the previous step" (p7). That value can come from
   a different model **(inference: the text does not say that it is the same model's
   previous value)**.
4. Update the selected model's sample-averaged reward `R_a`.
5. `x_next = argmax V[f_*]` of the selected model; measure; update `D` and `X_*`.

### 2.5 What decides which hypothesis wins

The winner is the model with the highest **sample-averaged reward `R_a`**. `R_a` is the
mean of +1/-1 rewards, and each reward records whether that model's posterior predictive
uncertainty over the unmeasured points fell relative to the previous step (p6-8). The
paper reports `R_a` per model as the outcome (Fig 2 p7, Fig 4 p10, Fig 7 text p15).

The deciding quantity is therefore **reduction of predictive variance**, not fit, not a
marginal likelihood, not a Bayes factor, and not a posterior model probability. The paper
computes none of the latter three **(inference: none appears in Algorithm 1 or the text)**.
`R_a` lies in [-1, 1] and is not a probability.

---

## 3. Experiments and results

### 3.1 Synthetic 1-D phase transition (p7-11)

- Ground truth: a piecewise function with one discontinuity (Eq 3, Fig 2a, p7-8). The
  algorithm starts from 4 seed points (p8).
- Three hypotheses based on Eq 3: (1) power law before `x_c`, linear after; (2) linear
  before and after; (3) power law before and after. Model 3 is the true form (p8).
- **Standalone models**, 15 steps (3 warm-up + 12 exploration): `R_a` = 0.0, -0.4 and
  +0.82 for models 1-3. The highest-reward model gives the best fit (Fig 2b-d, p7-8).
- Over 30 random seed initializations, model 3 wins most rewards. Three warm-up steps
  separate it more cleanly than one (Fig 3, p9).
- **Failure mode**: a standalone model list without the correct model fails (p9).
- **sGP models**, same settings: `R_a` = -0.17, -0.25 and +0.82 (Fig 4a-c, p10). Wrong
  models still reconstruct the data reasonably because the kernel compensates. A vanilla
  GP with zero mean fails at the discontinuity (Fig 4d, p10-11).
- **New failure mode of sGP**: with a wrong mean function, total uncertainty still falls as
  points are added. One warm-up step can therefore "lock up" on a wrong model (Fig 5a,
  p10-11). Several warm-up steps fix this because they compare absolute uncertainties across
  models (Fig 5b-d, p10-11).

### 3.2 Supplementary synthetic cases (p25-28, p31)

- Eq S1, a pole-type discontinuity. A Lorentzian-peak sGP gets `R_a` = 0.0 and the correct
  form gets 0.38 after 20 steps (3 + 17) (Supp Fig 1-2, p25-26).
- Two transitions, none of the three models correct. The two "linear after the transition"
  models get 0.68 and 0.57. The power-law model gets -0.40 (Supp Fig 4, p27). The caption
  says "25 steps (3 warm-up + 12 exploration)", which does not add up.
- A partially correct model against a wrong `a x^2 + b` model: 0.55 against 0.25. The
  algorithm prefers the partially correct one (Supp Fig 5, p27).
- **Diagnostic**: a partially correct model converges slowly in MCMC. It shows low `n_eff`
  and `r_hat` above 1.05 (p28). A completely wrong model converges well because the kernel
  takes over, but it gives a trivial solution (p31). The authors propose these diagnostics
  as a signal that "the model needs to be updated" (p28).

### 3.3 Real experiment: Sm-doped BiFeO3 (p11-15, p18)

- A thin-film combinatorial library with Sm from 0% to 20%. 56 locations are marked along
  the gradient (p18). Band-excitation piezoresponse spectroscopy measures hysteresis loops
  (Fig 6, p12-13). The scalar `y` is the loop area `S` as a function of concentration `x`.
- Hypotheses come from a Ginzburg-Landau model (Eq 5-6, p13-14; Supp II, p28-30):
  model 1 is continuous at the transition (Eq 7); model 2 has a jump (Eq 8) (p14).
- Two seed points at the extremes of the concentration axis, 5 warm-up steps and 7
  exploration steps, median uncertainty (p14-15).
- Final `R_a`: 0.13 for model 1, 0.56 for model 2 (p15).
- The authors do not accept the winner at face value. Model 2 had MCMC convergence problems
  at each step (Supp III, p30-31). The posterior samples show a peak-like feature before the
  transition at about 17% Sm, which no model contains (Fig 7b-c, p14-15). They read this
  misfit as a possible discovery of new (morphotropic) behaviour that needs a new model
  (p15).

### 3.4 Outlook (p15-16)

- "Automated scientist" (Fig 8, p15): generate random symbolic hypotheses, filter them with
  hard physical constraints, assign probabilities and priors from prior knowledge
  (databases, literature mining, prior experiments), then run hypothesis learning.
- The size of the hypothesis list should match the experimental budget. Cheap experiments
  allow a wide list; expensive experiments justify more effort to pre-select hypotheses
  (p16).
- They claim, without demonstration, that the method extends to higher-dimensional spaces
  (p16).

---

## 4. Limitations

### 4.1 Stated by the authors

- Standalone models fail when the correct model is not in the list (p9).
- sGP can lock up on a wrong model after a bad initialization (p10).
- Bayesian inference for every model at every warm-up step is costly, so warm-up is capped
  at about 1-5 steps (p7).
- The real-data winner may not be correct (convergence problems, p15). There is no ground
  truth for the real experiment (p14).
- Demonstrated only in 1-D (p16).

### 4.2 Ones I see

1. **Variance reduction is not evidence of correctness.** A model gets +1 whenever its
   predictive variance falls. Variance falls for almost any model as data accumulate. The
   authors see this for sGP (p10) **(inference: it also applies to standalone models)**.
   An overconfident model with tight priors would also earn rewards **(inference)**.
2. **`V_tot` is a sum over a shrinking set.** `X_*` loses one point per step, so the sum
   loses one term per step, and that alone biases the reward towards +1 **(inference from
   the definition in Algorithm 1, p6)**. The median variant (p7) avoids this.
3. **The reward compares across models.** If `V_tot^(i-1)` comes from a different model
   (p7), a model's reward partly reflects which model ran before it **(inference)**.
4. **No calibrated confidence.** `R_a` is not a probability. The paper reports no model
   probability, Bayes factor, credible interval on an effect, or significance test. The
   evidence across 30 initializations is histograms (Fig 3, Fig 5).
5. **Fixed epsilon = 0.4** and fixed step counts, with no sensitivity analysis (p8).
6. **A small real experiment**: 12 measured points, 2 hypotheses (p14-15).

---

## 5. What is worth keeping from the paper

- A **fixed, explicit list of competing parametric hypotheses with priors**, declared before
  the experiment. The priors "do not change during the experiment" (p14).
- The **Bayesian posterior over each hypothesis's parameters**, not a point estimate.
- **The experiment is chosen by the current uncertainty**, so each new measurement goes to
  where the hypotheses are least certain.
- **Misfit is a reported outcome.** When no hypothesis fits (convergence diagnostics, a
  feature no model contains), the method flags the hypothesis set as inadequate. It does
  not quietly declare the best of the bad models the winner (p15, p28, p31).
- **Hypothesis generation is separated from hypothesis testing** (Fig 8, p15). Prior
  knowledge supplies the list and the priors. The testing loop does not invent hypotheses.

---

## 6. Mapping to our framework

### 6.1 The epistemic object is different

The paper's hypotheses are competing functional forms of **one measured curve** `y(x)`.
The PI's hypotheses are claims that **an intervention causes a gain**: "a longer training
window helps", "a higher learning rate helps". These are causal effect claims about
retraining under a changed config, not shape claims about one response **(inference)**.

The translation that keeps the paper's structure: the measured response is the **paired
WIS change caused by an action**, as a function of the action's parameter and of the
evaluation coordinates (forecast date, phase, horizon). A hypothesis is then a parametric
model of that response. This is a well-defined Bayesian object, and an experiment here is
a randomized, paired comparison: same data snapshot, same seed, same cutoff, only the
config delta differs. That pairing is what licenses the word "caused" **(inference)**.

### 6.2 Term by term

| Paper term | Our equivalent | Notes |
|---|---|---|
| Control variable `x` | `(action, parameter value)` plus evaluation coordinates `(cutoff, horizon, phase, location)` | Discrete action catalog, few parameter levels: `param_space()` and the `ACTION_CATALOG` in `agent/adapters/flu_forecast.py`. |
| Measured property `y` | **Paired per-row WIS delta**: `WIS(candidate) - WIS(incumbent)` at the same `(seed, cutoff, location, horizon)` | Per-row WIS already exists: `score_rows` in `scripts/experiments/export_hindcast_tables.py`. |
| Hypothesis `m(x; theta)` | A **parametric expected effect** of one action. Examples below. | Always include the **null** `delta = 0`. |
| Priors on `theta` | **Skeptical priors** centred on zero effect, scale set from past experiments or a knowledge-bank entry | Fixed before the run, as on p14. |
| Experiment / measurement | **One retrain** of one config at one cutoff with one seed, then scoring | About 20 s per config per cutoff (harness docstring). 9 cutoffs x 3 seeds is about 9 min per config. |
| Observation | The **208 paired deltas** per (seed, cutoff): 52 locations x 4 horizons | 1872 rows / 9 cutoffs in the round-2 log = 208. |
| Posterior | `P(theta | deltas)` per hypothesis, and **`P(effect < -delta_min | data)`** for the decision | The paper has no decision probability. We add it (section 6.4). |
| Reward | Two different quantities. See section 6.4. | |
| `X_*` (unmeasured points) | Configs and cutoffs not yet run | |
| Hypothesis generator (Fig 8) | **The LLM proposer + the knowledge bank** | The bank already holds curated priors ("prior knowledge", Fig 8 p15). |

Example hypotheses for the peak-emphasis action (`sample_weights.by_phase.peak = lambda`),
written as models of the peak-phase paired delta `d`:

- `H0` (null): `d = 0 + noise`.
- `H1` (constant effect): `d = beta * (lambda - 1) + noise`.
- `H2` (horizon trade-off): `d = (beta_0 + beta_1 * (h - 1)) * (lambda - 1) + noise`.

Round 2 already contains the evidence that `H2` describes peak weighting better than `H1`.
Peak weight 3 moves peak WIS by -7.2, -12.0, +1.2 and +20.1 at horizons 1-4, and the 1-,
2- and 4-week signs hold in all 3 seeds
(`documentation/experiments/2026-10-peak-rectification-round2-results.md`). The pooled
effect is near zero because the horizon effects cancel. A test of `H1` against `H0` alone
would report "no effect" and miss a real, structured effect. This is the strongest
argument for competing parametric hypotheses over a single pooled point estimate
**(inference)**.

For a window hypothesis the parameter is `N` weeks, for example
`d = beta * log(N_ref / N)`. Round 2's 52-week window costs +68.5 peak WIS. A shape
hypothesis would need a window sweep to be identifiable.

### 6.3 Cheap and expensive, against the paper

- **In the paper**, a measurement is a lab measurement and Bayesian inference is the
  bottleneck. NUTS runs per model per step, which is why warm-up is capped at 1-5 steps and
  epsilon-greedy picks one model per exploration step (p7).
- **For us the cost runs the other way.** A retrain costs about 20 s per cutoff per seed.
  A conjugate Bayesian model of a few thousand paired deltas costs milliseconds. So every
  hypothesis can be scored at every step, and the epsilon-greedy model sampler has no
  purpose here **(inference)**.
- The **expensive axis is replication**. Rows within one fit are not independent: 52
  locations share one model fit, the four horizons share the same data, and states are
  correlated in time. The units of replication are `(cutoff, seed)` fits, not rows
  **(inference)**. Treating 1872 rows as independent would give overconfident posteriors.
- **The target phase is thin.** At stride 4, peak-phase rows (468 per config per seed) come
  from only a few cutoffs whose targets land in Jan-Feb. The weekly hindcast (35 cutoffs,
  `outputs/experiments/peak_rectification_weekly`, seed 42 only) has more peak cutoffs, but
  adjacent weekly cutoffs share most of their training data **(inference)**.
- **Seed noise is the floor.** Round 2 baseline peak WIS is 107.28 with seed sd 1.24
  (round-2 results doc). An effect smaller than that is not resolvable with 3 seeds. It
  sets a natural minimum practical effect `delta_min`.

### 6.4 Reward and posterior: two quantities, not one

The paper uses one quantity, variance reduction, both to choose the next experiment and to
pick the winner. We need to split them **(inference / recommendation)**:

1. **Decision quantity (the PI's "posterior / confidence that the change caused the
   gain")**: `P(mean paired delta < -delta_min | data, H)` on the target metric, plus the
   posterior model probabilities `P(H_k | data)` from each hypothesis's marginal likelihood
   with fixed prior model weights. This is a standard Bayesian model comparison and is not
   in the paper.
2. **Experiment-choice quantity (the paper's reward)**: which next retrain most reduces the
   uncertainty of that decision. Example: choose the candidate whose `P(improve)` is closest
   to 0.5, or the cutoff or seed that most shrinks the posterior on the deciding parameter.
   This is the analogue of `x_next = argmax V[f_*]`.

**Comparison to the baseline and to accepted changes.** Every delta is paired against the
**current incumbent** (the last accepted config), not the original baseline. This matches
how `agent/orchestrator.py` reverts to the best-known config. Each accepted change also
keeps its own delta against the original baseline so that interactions between accepted
changes stay visible. Effects need not be additive **(inference)**.

### 6.5 "Must not keep re-hypothesizing to force a win"

Today the loop has exactly this risk. `_is_better` in `agent/orchestrator.py` (line 275)
returns `current < best` for WIS. Any lower point estimate is accepted, at a single cutoff
(`improve --cutoff-date`) and a single seed. Over several iterations the best-of-k selection
picks up noise (winner's curse). Round 2 shows the scale: no peak arm beats the baseline
pooled by more than seed noise, yet several have lower point estimates (peak weight 2:
-1.86).

The paper's design supplies the guardrails, and we add two of our own:

- **A pre-registered hypothesis list** (paper: fixed `M`, p5-6), with fixed priors (p14),
  a fixed `delta_min`, a fixed acceptance threshold and a fixed retrain budget. Hash these
  into the run's manifest, as the harness already does for cutoffs and data.
- **The null is always in the list.** The paper assumes the correct model is in the list
  (p5). Including `H0` makes "no effect" a hypothesis that can win.
- **New hypotheses mid-run are logged as post hoc** (the analogue of p15, where misfit leads
  to a new model). A post-hoc winner is accepted only after confirmation on cutoffs it was
  not selected on (held-out cutoffs, or the next season) **(recommendation)**.
- **Every tested hypothesis counts.** The run report lists all hypotheses tested and their
  posteriors, not only the winner. Skeptical shared priors give some protection against
  multiplicity, but not full protection **(inference)**.

### 6.6 Narrowing the search space

The PI's end goal is to find which of many parameters matter. In this framing a parameter
"does not matter" when its posterior effect concentrates inside `[-delta_min, +delta_min]`
for every hypothesis, so `H0` keeps most of the posterior mass. That action is then pruned
from the proposer's catalog for this model and phase, and the result is written back as an
experiential knowledge-bank entry (the `knowledge import-experiment` path already exists)
**(recommendation)**. Screening many parameters at once is a classical design-of-experiments
problem (for example a fractional factorial over actions). The paper does not address it,
because it varies one coordinate (p16) **(inference)**.

### 6.7 What does not transfer

- **The structured GP over a continuous coordinate.** Our `x` is a discrete action catalog
  with a handful of levels per parameter, and the evaluation coordinates are dates and
  horizons. A GP with a physical mean function has little to interpolate over. A Bayesian
  linear or hierarchical model of the deltas does the same job in closed form
  **(inference)**.
- **NUTS and epsilon-greedy over models.** Both exist because inference is costly in the
  paper (p7). Ours is not.
- **The +1/-1 variance-reduction reward as a decision rule.** It is biased (section 4.2) and
  is not a probability. Use posterior probabilities for decisions.
- **Physics priors.** There is no first-principles model of "how WIS responds to a training
  window". Our priors come from past experiments and curated knowledge entries, which are
  weaker. Priors must stay skeptical (centred on zero).
- **Active choice of evaluation dates for the decision.** Choosing which cutoffs to evaluate
  based on past results biases the estimate of the season-average effect. The decision
  needs a fixed evaluation design. Active selection is safe for *which config to run next*,
  not for *where to score it* **(inference)**.

---

## 7. Proposed minimal version

Goal: a posterior that a change caused the gain, against the incumbent, with a
pre-registered hypothesis list. Build on existing code; no new training path, no MCMC.

### 7.1 Step 1 - offline posterior over existing runs (no training)

New module `agent/effect_posterior.py`, pure functions:

- Input: per-row WIS of the incumbent and the candidate, from forecast CSVs at matching
  `(seed, cutoff, location, horizon)`. Reuse `score_rows` from
  `scripts/experiments/export_hindcast_tables.py` (or move it into `agent/` first).
- Aggregate paired deltas to one mean per `(seed, cutoff, horizon)` cell on the target
  phase. That makes the unit of replication explicit and avoids treating 52 correlated
  states as independent.
- Fit each hypothesis as a Bayesian linear model `d = X beta + eps` with a conjugate
  Normal-Inverse-Gamma prior. Prior mean of `beta` is zero; prior scale comes from config.
  `H0` has an empty `X`, `H1` an intercept, `H2` intercept plus horizon slope. The marginal
  likelihood is closed form, so posterior model probabilities are exact under the model.
- Output per candidate: `P(H_k | data)`, the posterior of the pooled target-phase effect,
  `P(effect < -delta_min)`, a 90% credible interval, and the number of cells used.

Wire it into the harness as a read-only mode, in the style of `--summary-only` and
`--rescore`: `peak_rectification.py --posterior --out <dir>` reads
`forecasts/<config_id>.csv` for every arm and its same-seed baseline, writes
`posterior.md` and `posterior.json`, and trains nothing.

Note: `log.jsonl` stores metrics per config per seed **pooled over cutoffs**, not per
cutoff. The per-cutoff and per-row values come from the saved forecast CSVs, which keep
every row.

Success checks on the round-2 data:
- The 52-week window gets `P(harm) > 0.99`.
- Pooled peak-weight and oversample arms come out "inconclusive" under `H1`, consistent
  with the seed-noise finding.
- For peak weight 3, `H2` (horizon trade-off) gets more posterior mass than `H1` and `H0`.
- Unit tests on synthetic deltas with a known effect: the posterior recovers the sign and
  covers the true value; pure noise leaves `H0` with the most mass.

### 7.2 Step 2 - acceptance gate in the improve loop

- Add an evaluation design to `improve`: K cutoffs x S seeds per iteration instead of one
  cutoff and one seed. With K = 4 and S = 2 an iteration costs about 8 retrains, about
  3 min.
- Replace the acceptance call to `_is_better` (orchestrator line 580) with a gate:
  accept only if `P(effect < -delta_min) >= threshold` against the incumbent. Otherwise
  record "inconclusive" and revert. Keep `_is_better` for direction-awareness and for the
  report.
- `delta_min`, `threshold`, the prior scale and the hypothesis forms live in validated
  config, not literals. The values are Leo's or the PI's decision.
- Record the posterior and the hypothesis probabilities in `runs.db` and the run report.

### 7.3 Step 3 - pre-registration and the hypothesis list

- The proposer's action becomes a hypothesis: action + parameter + effect form. The null is
  added automatically.
- Write the list, priors, `delta_min`, threshold and budget to the run manifest before the
  first retrain, hashed.
- A hypothesis added after the first result is tagged `post_hoc` and needs confirmation on
  held-out cutoffs before acceptance.

### 7.4 Deliberately left out of the minimal version

- Active choice of the next experiment (the paper's `argmax V` step). Add it after the gate
  works. The first version is "choose the undecided candidate with `P(improve)` nearest
  0.5".
- GP or MCMC models, interaction models between accepted changes, and factorial screening.

### 7.5 Open decisions for Leo / the PI

- `delta_min`: the seed sd (about 1.2 peak WIS) or a clinically motivated value.
- The acceptance threshold (for example 0.9 or 0.95).
- Which metric and phase decide: pooled peak WIS, or per-horizon. Round 2 shows that this
  choice changes the answer.
- Whether evaluation cutoffs for the gate must be disjoint from the cutoffs used to pick
  candidates. Recommendation: yes, for anything reported in the paper.
