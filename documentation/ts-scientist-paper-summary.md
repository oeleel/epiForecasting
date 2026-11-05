# TimeSeriesScientist (TSci) - detailed paper summary and what the project takes from it

> Zhao, Zhang, Wei, Xu, He, Sun, You. *TimeSeriesScientist: A General-Purpose AI Agent for
> Time Series Analysis.* arXiv 2510.01538v2, 6 Oct 2025. Stony Brook / UCSD / UBC / Zhejiang /
> UCLA / CWRU / Fudan. PDF: `documentation/papers/2510.01538.pdf` (34 pages: 10 main, 4
> references, 20 appendix).
>
> Read in full 2026-11-04, every page including the appendix and the prompts. Page numbers
> below are the printed page numbers of the PDF. Statements marked **(inference)** are mine,
> not the paper's. Numbers read off a figure, not a table, are marked **(read from figure)**
> and are approximate.

---

## 1. What the paper is actually claiming

**The gap they target (§1, p1-2).** Organizations manage thousands of short, noisy series
with irregular sampling, missing values and shifting horizons. The paper argues the dominant
cost is not model fitting but the manual preprocessing, validation and ensembling around it.
Domain-tuned expert models degrade under distribution shift; AutoML for forecasting (they cite
AutoGluon-TimeSeries) "centers on model selection and ensembling, but with limited attention to
data quality" and cannot reason about temporal structure or justify choices in natural language
(p1-2); Time-LLM and similar foundation-style methods target the model, not the pipeline (p2).

**Their answer.** An LLM-driven, four-agent pipeline that "automate[s] the entire workflow a
human scientist would follow for univariate time series forecasting" (p2):
Curator (diagnose and clean) -> Planner (pick candidate models and tune them) -> Forecaster
(choose an ensemble strategy, forecast) -> Reporter (write a report). Fig 2 (p3) is the overview.

**Headline numbers (abstract, p1; §4.1, p7).** Across eight public benchmarks in five domains,
TSci reduces forecast error by an average of **10.4%** versus statistical baselines and
**38.2%** versus LLM-based baselines. The introduction states the second figure as **38.3%**
(p2); the abstract and §4.1 say 38.2%. The paper does not show where the 10.4% is computed.

**Stated contributions (p2).**
1. "The first end-to-end, agentic framework for univariate time series forecasting with
   tool-augmented LLM reasoning."
2. "Plot-informed multimodal diagnostics, where a lightweight vision encoder converts plots into
   descriptors" for preprocessing, analysis and model selection.
3. Outperforms statistical and LLM-driven baselines.
4. An evaluation of the generated reports for technical rigor and communication quality.

**The headline design commitment** is opposite to TS-Agent's: TSci treats the problem as
choosing from a fixed library of 21 off-the-shelf models and ensembling them, with the LLM
proposing the search space and the ensemble strategy. It does not refine an existing model,
it has no feedback loop across iterations, and it has no external knowledge bank (it lists
"incorporating external knowledge" as future work, §5, p10).

---

## 2. Formalization (§3.1, p4)

A univariate series `x = {x_{t-T+1}, ..., x_t} in R^{1xT}`. The task is to estimate
`y_{t+i}`, `i = 1..H`. The objective is stated as minimizing the mean absolute error

```
(1/H) * sum_{i=1..H} || y_{t+i} - yhat_{t+i} ||
```

The text calls this "mean average errors (MAE)" (p4). The system output is a report `R`
holding input statistics, visualizations, "proposed model combinations that best fit the data,
and the final forecasting result" (p4).

Observations on the formalization:
- **Univariate only.** No covariates, no cross-series pooling, no hierarchy. Each series (each
  slice, see §7) is processed independently.
- **Point forecast only.** The objective is MAE; validation selection uses MAPE (Eq. 2-3, p5-6).
  The report shows "confidence intervals" (Fig 4, p7; Fig 14 shows 80/90/95% bands, p25), but
  the paper never says how intervals are produced and never scores them (no coverage, no
  interval or quantile score, no CRPS).
- **No task description tuple.** Unlike TS-Agent's `T = (desc, D, L)`, there is no user goal or
  description input; the pipeline is fixed and the metric is fixed (MAE/MAPE).
- **No action space or policy factorization.** The paper formalizes each agent as an operator
  (`A_f`, `A_v`, `A_c`, `Select`), not as a sequential decision process with state, memory and
  accept/revert. The run is a single forward pass (§3, Fig 2) **(inference from the absence of
  any loop in Fig 2, §3 or the prompts)**.

---

## 3. The four agents (§3.2-3.5, Fig 2-4, Appendix A, B, E, H)

### 3.1 Curator (§3.2, Fig 3, p4-5; Appendix A, p15-17; Appendix E, p19-21; prompts p26-31)

Three steps.

**(a) Quality diagnostics and preprocessing (Eq. 1, p5).** The LLM outputs
`Q = A_f(D) = (S, M, O, pi)`:
- `S = (mu, sigma, x_min, x_max, tau_trend)` basic statistics,
- `M` missing-value information, `O` outlier information,
- `pi = (m*, h*)` the LLM-recommended missing-value and outlier-handling strategies.

A transformation `phi` applied per `pi` yields the processed series `D~ = phi(D)`.

The strategy menu is closed and enumerated in the prompts (DATA_ANALYSIS_PROMPT, p30):
- missing values: `interpolate`, `forward_fill`, `backward_fill`, `mean`, `median`, `drop`,
  `zero`;
- outlier detection: `iqr`, `zscore`, `percentile`, `none`;
- outlier handling: `clip`, `drop`, `interpolate`, `ffill`, `bfill`, `mean`, `median`, `smooth`.

Appendix A (p15-17) defines each as a "leakage-safe toolkit", with statistics "estimated on
rolling (local) windows to accommodate non-stationarity":
- detection: rolling IQR with `alpha = 1.5` (Eq. 4-5), window one or two seasonal cycles under
  strong seasonality; rolling z-score with `alpha in [3, 4]`, median/MAD variant for heavy
  tails (Eq. 6-7); percentile rule, e.g. (1%, 99%) (Eq. 8);
- handling: clipping/winsorization (Eq. 9), segment-aware linear interpolation (Eq. 10),
  forward/backward fill (Eq. 11), local mean/median (Eq. 12), light causal smoothing (Eq. 13,
  "use small w to limit lag and peak attenuation");
- missing values: segment-aware interpolation (Eq. 14), fill (Eq. 15), local mean/median
  (Eq. 16, "prefer causal in evaluation"), zero fill "used only when zero has a clear meaning"
  (Eq. 17).

The DATA_PREPROCESS_PROMPT (p27) puts the raw series into the prompt as a Python dict
(`data.to_dict(orient='list')`). With `T = 512` (Appendix D.1, p18), the LLM sees all 512 raw
numbers.

**(b) Visualization generation (p5; Appendix E, p19-21).** `V = A_v(D~)`. Three primary plot
types: overview (raw series, rolling mean and std, Fig 8), additive seasonal decomposition
`X_t = T_t + S_t + R_t` (Fig 9, statsmodels `seasonal_decompose`), and ACF/PACF with 40 lags
(Fig 10). A VISUALIZATION_DECISION_PROMPT (p31) lets the LLM choose among `time_series`,
`distribution`, `rolling_stats`, `autocorrelation`, `seasonal_decomposition`, and set titles and
parameters. Plots are Matplotlib/seaborn PDFs at 300 DPI (§E.2, p20-21).

**(c) Temporal structure profiling (p5).** `A = A_c(D~, V) = {t, s, u}` - trend, seasonality,
stationarity - by prompting over the processed data and visualizations. Output of Curator:
the analysis summary `C = {Q, V, A}`. An example summary for ECL is on p23-24 (JSON with trend,
seasonality, stationarity, issues, actionable insights).

**How the plots reach the LLM is unclear.** Contribution 2 (p2) says "a lightweight vision
encoder converts plots into descriptors"; §3.3 says Planner "extracts visual features from
visualizations V via lightwise pattern recognition and LLM reasoning" (p5); the prompts
interpolate `{{visualizations}}` and `{{viz_info}}` (p27, p32) without saying whether these are
images or text. No vision encoder is named or described.

### 3.2 Planner (§3.3, p5-6; Algorithm 1, p6; prompts p31-33)

"Narrows the hypothesis space of model configurations by reasoning on the analysis summary C.
Rather than exhaustively trying all candidates, it prioritizes models that are most consistent
with data characteristics" (p5). Three steps.

**(a) Model selection.** The LLM maps recognized patterns to model families and forms a
candidate pool `M_p = Select(M; n_p)` from the 21-model library `M` (p5). Example given: choose
Prophet "when recognizing a weak trend with a long seasonal span." For each `m_i` the agent
writes a rationale `r_i`. The Planner prompt (p32) fixes `n_p` in practice: "select the top 3
most suitable forecasting models from the provided list", "Consider diversity in model
selection to maximize ensemble robustness", "Return EXACTLY n_candidates models".

**(b) Hyperparameter optimization (Eq. 2-3, Algorithm 1, p5-6).** For each candidate the LLM
proposes a hyperparameter space `Theta_i` (`PROPOSEHYPERPARAMS(m_i)`). The prompt asks for
"2-3 parameter search spaces as arrays", e.g. ARIMA `p: [0,1,2], d: [0,1], q: [0,1,2]` (p33).
Up to `N` configurations are sampled and the one with the lowest validation MAPE is kept:

```
theta_i* = argmin_{theta in C_i} MAPE_val(m_i(theta))
```

`MAE_val` is also computed "for robustness checks" (p6). `N` is not reported. How `D_val` is
cut from the 512-point input (size, single holdout vs rolling origin) is not described.

**(c) Model ranking (p6).** Tuned candidates are ranked by validation MAPE and the top `k` are
kept as `M_selected` with `Theta*` and `S_val`. `k` is not reported in the main text; with
`n_p = 3` from the prompt, `k <= 3` **(inference)**.

### 3.3 Forecaster (§3.4, p6; Appendix B, p17; prompts p33-34)

Takes `M_selected` and `S_val`. "The agent leverages an LLM-guided policy to select an ensemble
strategy from among three families: single-best selection, performance-aware averaging, or
robust aggregation. The ensemble strategy and (if applicable) weights are fixed before touching
the test set" (p6). It then forecasts the held-out horizon `H` and reports `S_test`.

The §3.4 motivation for ensembling cites Adiga et al. 2023 (the lab's phase-informed Bayesian
ensemble paper) for "maintaining robustness across epidemic phases" (p6, reference on p13).
That is the only place epidemic phases appear in the paper.

Appendix B (p17) formalizes the three families:
- **Validation score.** `s_i = sum_m alpha_m * norm(l_{i,m})` over normalized validation
  metrics (Eq. 18), default `alpha_MAE = alpha_MAPE = 0.5`, min-max normalization across the
  `k` candidates.
- **(A) Single-best** (Eq. 19), preferred when the relative gap
  `(s_(2) - s_(1)) / s_(1) >= delta`, `delta = 0.05` (Eq. 20).
- **(B) Performance-aware averaging** (Eq. 21-23): tempered inverse-loss weights
  `w~_i = (s_i + eps)^(-beta)`, softmax with temperature `tau`, clipped to
  `[w_min, w_max]`, then shrunk toward uniform:
  `w_i = (1 - lambda) clip(w_i^perf) + lambda / k`, defaults `beta = 1, tau = 1,
  lambda = 0.1, w_min = 0.02, w_max = 0.80, eps = 1e-8`.
- **(C) Robust aggregation** (Eq. 24-25): per-horizon-step median or trimmed mean
  (`rho = 0.1`).
- Implementation notes: weights constant across the horizon "to avoid step-wise overfitting";
  ensembling in the scaled space when Curator scaled the data; cost `O(kH)`; `k = 1` forces
  single-best.

Who decides the strategy is ambiguous. Appendix B reads as a deterministic rule (Eq. 20 with a
default `delta`); the prompts (p33-34) ask the LLM to "Decide the best ensemble integration
strategy (choose from: best_model, weighted_average, trimmed_mean, median, custom_weights)" and,
"if using weights, specify the weights for each model." The prompt menu includes
`custom_weights`, which Appendix B does not define. So the LLM can set the weights directly,
and Eq. 21-23 may or may not be applied **(inference: the paper does not reconcile the two)**.
The ENSEMBLE_DECISION_PROMPT takes `json.dumps(individual_forecasts)`; the paper states weights
are fixed on validation data only, but the prompt does not show which forecasts are passed.

### 3.4 Reporter (§3.5, Fig 4, p6-7; example G.3, p25-26)

Produces a report `R` of five parts (Fig 4, p7):
1. ensemble forecast with confidence intervals;
2. performance summary of each model and the ensemble (test metrics);
3. interpretability report: model-selection rationale, ensemble-weight derivation, the
   system's confidence, assumptions and limitations;
4. visualization suite;
5. workflow documentation of every decision (input length, horizon, outlier handling,
   normalization, selected models).

The example report (G.3, p25-26) lists data characteristics, ensemble weights, per-model MSE /
MAE / MAPE, issues and limitations, and recommendations.

---

## 4. Model library (Appendix C, Table 3, p18)

21 models in `model_library.py`, five categories:

| Category | Models |
|---|---|
| Statistical (7) | ARIMA, RandomWalk, ExponentialSmoothing, MovingAverage, TBATS, Theta, Croston |
| ML regression (6) | LinearRegression, PolynomialRegression, RidgeRegression, LassoRegression, ElasticNet, SVR |
| Tree-based (4) | RandomForest, GradientBoosting, XGBoost, LightGBM |
| Neural networks (2) | NeuralNetwork, LSTM |
| Specialized (2) | Prophet, Transformer |

Each is a `predict_<name>` function. The library is general-purpose and off-the-shelf: no
foundation models (Chronos, TimesFM, Lag-Llama are discussed in §2 but are not in the library),
no probabilistic models, no domain models (no compartmental or epidemic models). The library is
fixed by the authors; the paper gives no mechanism for adding a user's own model **(inference
from the absence of any such mechanism in the text)**.

---

## 5. How hypotheses and model choices are narrowed

The paper's phrase is "narrows the hypothesis space of model configurations" (§3, p4; §3.3,
p5). The "hypothesis" is which model family and configuration fits the series. The funnel:

| Stage | Narrowing | Who decides | Evidence used |
|---|---|---|---|
| 21-model library | fixed | authors | - |
| Candidate pool `M_p` | 21 -> `n_p` (3 in the prompt, p32) | LLM | Curator summary `C` (statistics, plots, trend/seasonality/stationarity) |
| Hyperparameter space `Theta_i` | open -> "2-3 parameter search spaces as arrays" (p33) | LLM | same |
| Configuration `theta_i*` | `N` sampled -> 1 | argmin validation MAPE | `D_val` |
| `M_selected` | `n_p` -> top `k` | ranking by validation MAPE | `S_val` |
| Ensemble | `k` -> one strategy and weights | LLM (and/or Eq. 18-25) | `S_val`, individual forecasts, plots |

Points worth stating plainly:
- The LLM's prior reasoning (from plots and summary statistics) does the big cut, 21 -> 3,
  **before any model is fitted**. The 18 rejected models are never evaluated, so a wrong prior
  cannot be detected or corrected within the run **(inference)**.
- After that, every decision is a point comparison of validation MAPE. There is no hypothesis
  test, no confidence interval on the validation difference, no repeated runs, and no seeds
  (nothing in §3, §4 or Appendix D mentions any).
- The only threshold is the single-best gap `delta = 0.05` (Eq. 20, p17), a fixed 5% relative
  margin, not a statistical criterion.
- One pass only: no reflection on test results, no retry, no accept/revert.

---

## 6. Validation and ensembling (summary)

- **Validation.** Single validation set `D_val` per series slice, MAPE as the selection score,
  MAE as a secondary check (Eq. 2-3, p5-6). Split size and method not described.
- **Leakage claims.** Preprocessing uses rolling/causal windows (Appendix A, p15-16); ensemble
  weights "depend only on S_val and pre-specified hyperparameters; no test data is touched"
  (Appendix B, p17). The Curator prompt receives the full input series (p27), so the LLM sees
  the whole 512-point window, which includes the validation part **(inference: there is no
  statement that validation points are hidden from Curator)**.
- **Ensembling.** Three families (Appendix B). Horizon-constant weights. The case study and the
  generated report disagree on what was ensembled (see §9, my limitation 5).

---

## 7. Benchmarks and metrics (§4.1, p7-9; Appendix D, p18-19; Appendix F, p21-24)

**Datasets (Table 4, p19).** The eight long-horizon benchmarks from the TimesNet collection
(Wu et al. 2023b):

| Dataset | Domain | Length | Frequency | Span |
|---|---|---|---|---|
| ETTh1, ETTh2 | Electricity | 17,420 | 1 hour | 2016-07 - 2018-06 |
| ETTm1, ETTm2 | Electricity | 69,680 | 15 min | 2016-07 - 2018-06 |
| Weather | Environment | 52,696 | 10 min | 2020-01 - 2021-01 |
| Electricity (ECL) | Electricity | 26,304 | 1 hour | 2016-07 - 2019-07 |
| Exchange | Economic | 7,588 | 1 day | 1990-01 - 2010-10 |
| ILI | Health | 966 | 1 week | 2002-01 - 2020-06 |

**Protocol (Appendix D.1, p18).** GPT-4o as default backbone. "Due to a limited budget," each
dataset is divided into **25 slices** and results are averaged over the slices. Input length
`T = 512`; horizons `H in {96, 192, 336, 720}`, and `{24, 36, 48, 60}` for ILI (Table 6, p23).
Metrics: MAE and MAPE.

Details the paper does not give: which column is forecast in these multivariate datasets
(univariate setting, so presumably the standard target `OT` **(inference)**); how the slices are
cut (ILI has 966 points, so 25 slices of `512 + 60` points must overlap heavily
**(inference)**); the train/validation/test cut inside a slice.

**Baselines (Appendix D.3, p18).** Five LLMs used as forecasters: GPT-4o, Gemini-2.5 Flash,
Qwen-Plus, DeepSeek-v3, Claude-3.7. The paper does not describe how these baselines were
prompted (zero-shot numbers in, numbers out? with tools?). Statistical baselines: Linear
Regression, ARIMA, Prophet (Fig 6, p9; Fig 11, p22). There is no agentic baseline (no
TS-Agent, DS-Agent, ResearchAgent), no AutoML baseline (no AutoGluon, although the
introduction criticizes it), no deep-learning or foundation-model baseline (no PatchTST,
TimesNet, Chronos), and no "fit all 21 library models and pick the best on validation"
baseline.

**Results versus LLM baselines (Table 1, p8; Tables 5-6, p22-23).** TSci is best on all eight
dataset averages by MAE (Table 1, "1st Count" 8 of 8) and best in 35 of the per-horizon cells
(Table 6, p23). Averages, MAE:

| Dataset | Best LLM baseline (MAE) | TSci MAE |
|---|---|---|
| ETTh1 | Gemini 5.20 | 2.02 |
| ETTh2 | Gemini 11.0 | 4.91 |
| ETTm1 | Qwen 5.09 | 2.73 |
| ETTm2 | Claude 6.94 | 4.87 |
| Weather | Qwen 42.9 | 29.1 |
| ECL | Claude 844 | 667 |
| Exchange | Claude 0.073 | 0.045 |
| ILI | Claude 1.79e5 | 1.41e5 |

On ECL by MAPE, Claude-3.7 (32.2) beats TSci (40.2) (Table 1). Baseline MAPEs reach 100-264%
on ETT, which suggests plain LLM forecasting is a weak baseline here **(inference)**.

**Results versus statistical baselines (Fig 6, p9; Fig 11, p22).** Only figures, no table.
The text concedes that at short horizons "simple linear models (e.g., linear regression),
match or slightly exceed TSci" (p7). Reading Fig 11 **(read from figure)**:
- **ECL:** TSci is worse than all three statistical baselines at every horizon, and much
  worse at 336 and 720 (about 950 and 850 MAE versus about 350-700).
- **ILI:** Prophet is clearly better than TSci at every horizon (about 65-80k versus about
  135-150k MAE). ILI is the only health dataset and the closest to this project's task.
- ETTh2-720 and ETTm1-720: Linear Regression is at or below TSci.

The 10.4% average gain over statistical baselines (abstract) is therefore an average that
hides datasets where TSci loses, including the one epidemiological dataset **(inference from
Fig 11; the paper does not discuss this)**.

**Dispersion (Fig 12-13, p24-25).** Box plots of slice-level log10 MAE and MAPE per dataset and
horizon. These are the only spread information. No standard deviations in tables, no
confidence intervals, no significance tests, no repeated runs with different seeds or LLM
samples.

**Cost and runtime.** Not reported (tokens, wall time, LLM calls per series). The budget
statement in D.1 is the only reference to cost.

---

## 8. Ablations (§4.3, Fig 7, p9-10)

Three variants, each removing one module: without data preprocessing (Curator step a),
without data analysis (Curator step c), without parameter optimization (Planner step b).
Fig 7 covers eight settings: ETTh1 and ETTm1 at the four horizons **(read from the figure's
x-axis; the text does not name them)**. The full system's bars match Table 5 for those
datasets.

| Removed module | Average MAE change (text) |
|---|---|
| Data preprocessing | +41.80%, largest; degradation grows with horizon |
| Data analysis | +28.3%; two minute-level cases improve (ETTm1-96, ETTm1-720) |
| Parameter optimization | the text says "a 36.2% MAE drop on average", which from context means a 36.2% increase |

TSci has the lowest MAE in six of eight settings (Fig 7 caption). At ETTm1-96 the variants are
1.61 / 1.57 / 1.69 versus TSci 1.68; at ETTm1-720, 3.86 / 4.61 / 3.75 versus 4.10
**(read from figure)**.

Not ablated: the Planner's LLM model selection (versus random or all-21), the Forecaster's
ensembling (versus single-best), the visualizations specifically, the Reporter, and the LLM
backbone. TSci is only ever run on GPT-4o; Fig 1a compares TSci against five LLMs as
forecasters, not TSci across five backbones. The ablation uses two of eight datasets.

---

## 9. Report evaluation (§4.2, p8-9; Table 2, Fig 1b)

Five rubrics (p8): **Analysis Soundness (AS)** - rigor of exploratory analysis;
**Model Justification (MJ)** - whether chosen models fit the data and the choice is justified;
**Interpretive Coherence (IC)**; **Actionability Quotient (AQ)**; **Structural Clarity (SC)**.
AS and MJ measure technical rigor; IC, AQ and SC measure communication.

Metric: **win rate**, "the proportion of pairwise comparisons in which our framework's report
is judged superior to a baseline, excluding ties" (p9).

| TSci vs | AS | MJ | IC | AQ | SC |
|---|---|---|---|---|---|
| GPT-4o | 80.8 | 84.6 | 80.8 | 76.9 | 71.4 |
| Gemini-2.5 Flash | 81.8 | 81.8 | 63.6 | 68.2 | 53.8 |
| Qwen-Plus | 83.3 | 83.3 | 79.2 | 75.0 | 75.0 |
| DeepSeek-v3 | 92.3 | 84.6 | 80.8 | 76.9 | 76.9 |
| Claude-3.7 | 84.7 | 87.5 | 84.6 | 80.8 | 53.8 |

Missing from the description: **who judges** (human experts, an LLM judge, or which LLM),
**how many** comparisons, the judging prompt or protocol, inter-rater agreement, and what the
baseline "reports" are (the baselines are plain LLMs; what report they were asked to write is
not stated). Several values match small denominators (80.8 = 21/26, 53.8 = 7/13, 63.6 = 7/11,
81.8 = 9/11) **(inference)**, which suggests tens of comparisons, not hundreds. The rubrics
judge whether the report reads as rigorous, not whether its claims are true.

---

## 10. Limitations

### 10.1 Theirs (§5, p10)

The paper has no limitations section. §5 lists only future work: multimodal settings,
"incorporating external knowledge", and efficiency-oriented designs. The generated report
template has an "Assumptions and Limitations" field (Fig 4), but that is per forecast, not about
the method.

### 10.2 Ones I see

1. **No statistical evidence for any comparison.** Single runs, no seeds, no variance across
   LLM samples, no confidence intervals, no tests (Diebold-Mariano or otherwise) on the
   headline gains. Box plots (Fig 12-13) are the only spread shown.
2. **Weak and under-specified baselines.** The LLM baselines are bare LLMs whose prompting is
   not described; there is no AutoML, agentic, deep-learning, or "all 21 models plus
   validation selection" baseline. The last one is the key missing control: without it, the
   paper cannot show that the LLM's reasoning adds anything over brute-force validation over its
   own library **(inference)**.
3. **It loses to Prophet on ILI and to all statistical baselines on ECL** (Fig 11, read from
   figure). The paper does not discuss either, and the ECL case study (§4.4) only compares
   against LLM baselines.
4. **Point forecasts only, scored only by MAE/MAPE.** Intervals are displayed (Fig 14) but
   their construction is not described and they are never scored. In Fig 14 the test series
   falls well below the 95% band for most of the horizon **(read from figure)**.
5. **Internal inconsistencies in the case study.** §4.4 says Planner chose ARIMA, Prophet and
   Exponential Smoothing (p10), and Fig 14 shows those three; the generated report says the
   ensemble used Exponential Smoothing 35%, ARIMA 40% and Random Forest 25% (p26). §4.4 says
   the ECL slice has "no persistent long-term trend"; the Curator summary says "an overall
   upward trend" and "moderate to strong" (p23-24). In the report's own numbers, the ensemble's
   test MAE (393.83) is worse than Exponential Smoothing alone (377.15) and ARIMA alone (390.10)
   (p26), which is not what §4.4 claims. Fig 4 says weights came from "simple averaging", while
   the case study shows a weighted average. Also: 38.2% vs 38.3% (p1 vs p2), and the
   "36.2% MAE drop" wording in the ablation (p9).
6. **LLM-assigned ensemble weights.** The prompt lets the LLM write `custom_weights` from
   forecasts and plots (p34). That is an unvalidated numeric decision by the LLM, and it is
   unclear whether Eq. 21-23 constrain it.
7. **Preprocessing that is risky for epidemic data.** IQR or z-score outlier detection with
   clipping or smoothing (Appendix A) would flag a real epidemic peak as an outlier and flatten
   it. The appendix warns about "peak attenuation" only for the smoothing window
   **(inference about flu data, not tested in the paper)**.
8. **Model selection before evaluation.** The LLM rejects 18 of 21 models from plot-level
   cues; those models are never scored, so the selection step cannot be audited against what it
   discarded **(inference)**.
9. **Univariate, per-slice, no memory.** No cross-series learning, no covariates, no
   reuse of past runs. Every slice starts from zero.
10. **Report evaluation protocol undisclosed** (judge, count, prompt), and the rubrics reward
    plausible prose.
11. **Backbone dependence untested.** Only GPT-4o drives TSci. Whether the design works on a
    small local model is unknown.
12. **No cost reporting**, although the authors cut each dataset to 25 slices for budget
    reasons.
13. **Long-horizon framing on ILI.** Horizons of 24-60 weeks on national ILI are far outside
    the 1-4 week horizons used operationally (FluSight) **(inference about relevance)**.

---

## 11. What this project takes from it / how we differ

### 11.1 Differentiators, checked against the paper

| Dimension | TSci (verified in paper) | This project (verified in repo) |
|---|---|---|
| **What the LLM may do** | Free choice inside closed menus: preprocessing strategies (p27, p30), any 3 of 21 models (p32), **its own hyperparameter search space** (p32-33), ensemble strategy and **custom weights** (p34). No numeric guardrails on proposed spaces or weights are described. | Constrained, validated action catalog (`FluForecastAdapter.ACTION_CATALOG` in `agent/adapters/flu_forecast.py`): seven actions, schema and range validation, unknown action raises. The LLM edits configs, never code or numbers outside ranges. |
| **External knowledge** | None. "Incorporating external knowledge" is future work (§5, p10). Knowledge lives only in the backbone and the system prompts. | Curated, provenance-typed knowledge bank (`agent/knowledge/`, `knowledge/curated/*.yaml`); deterministic retrieval by phase, model, metric; entries shown with ids; `cited_entries` recorded and unretrieved citations dropped; `--no-knowledge` ablation switch. |
| **Evaluation surface** | MAE and MAPE pooled over the horizon and averaged over 25 slices; per-horizon tables only. No bias, no coverage, no regime/phase breakdown. Epidemic phases appear only as a citation of Adiga et al. 2023 (p6). | `PhaseEvaluator` (`agent/phase_evaluator.py`): WIS, 95% coverage, MAE, MAPE and **signed bias** by phase, horizon and location; named phase schemes recorded in every metrics dict; cross-scheme comparisons refused. |
| **Object of the work** | Picks and ensembles off-the-shelf library models per series. No mechanism to add a user model. | Refines an existing in-house model (`xgboost_direct`, the lab's pipeline in `src/`) and any family plugged into `src/model_bank/` by dotted path. Stage 1 selection exists, but the main loop is refinement of one incumbent. |
| **Series structure** | Univariate, each slice independent (§3.1, p4). | Multi-location: one model trained jointly across the state-level series with location clustering (`src/location_clustering.py`), 1-4 week horizons; national `US` excluded from scoring. |
| **Probabilistic output** | Point forecast; intervals shown but not described or scored. | Quantile forecasts (0.05-0.95), scored by WIS and coverage. |
| **Loop** | Single forward pass; no accept/revert, no iteration on results. | Iterative two-agent loop with accept-or-revert against best-so-far and stop rules (`agent/orchestrator.py`), every iteration in `runs.db`. |
| **Statistical significance** | None anywhere. | Also **no formal test**. Round 2 uses 3 seeds and the outline's rule is "treat differences inside the seed spread as no effect" (`documentation/paper/outline.md` §7). Better than TSci, but not a significance test. No Diebold-Mariano, bootstrap, or permutation test exists in `agent/`, `src/` or `scripts/` (grep, 2026-11-04). |
| **Human in the loop** | None described; fully automatic. | Shadow mode by default with y / n / stop per action; `--auto-apply` to skip (`agent/cli.py`). |
| **Backbone** | GPT-4o only. | Qwen 3 8B locally (Ollama), Qwen 2.5 72B planned on Rivanna. |

### 11.2 What TSci does that this framework lacks

1. **Agentic data preprocessing.** Our `check-data` agent (`agent/data_quality.py`) diagnoses
   missing weeks, nulls, spikes, zero reporting and coverage, and the LLM interprets the
   findings, but nothing transforms the data. TSci diagnoses and then applies a chosen repair.
   Their largest ablation effect is preprocessing (+41.8% MAE when removed, §4.3). Caution: for
   flu, clipping outliers would erase peaks (limitation 7); backfill/revision handling is the
   more relevant preprocessing for us and is still missing.
2. **Cross-family ensembling.** We select one incumbent (`agent/model_selection.py`); TSci
   combines the top `k`. Appendix B's tempered inverse-loss weights with shrinkage and clipping
   (Eq. 21-23) are a concrete, cheap recipe. For quantile forecasts this would need a quantile
   (Vincentization) or WIS-weighted combination, which is closer to the lab's own ensemble work
   **(inference)**. Our bank already has 9 families on the same contract, so this is a
   moderate addition.
3. **Plot-based diagnostics given to the LLM.** We feed the LLM tables of metrics, not plots.
   The lab-style inspection figure exists (`src/forecast_panels.py`) but is for humans. With a
   text-only Qwen 3 8B this is not directly usable; the text descriptors TSci derives
   (trend, seasonality, stationarity) could be computed deterministically and put in the prompt.
4. **LLM-proposed hyperparameter search spaces plus a small grid per candidate.** We take one
   hyperparameter step per iteration from `param_space()`. TSci evaluates a small grid per
   proposal. A "propose a 2-3 value grid, the code evaluates it" action would cut iterations
   without loosening guardrails if the grid stays inside `param_space()` bounds.
5. **A per-run narrative report with rationale for each choice.** Our `run_report.py` already
   covers most of Fig 4 (per-iteration deltas, phase and horizon tables, citations, caveats).
   What TSci adds is an explicit "assumptions and limitations" and "confidence" section.
   Their rubric set (AS, MJ, IC, AQ, SC) is a ready-made template if we ever evaluate our
   reports, but with a disclosed judge and count.

### 11.3 What to take for the paper

- **Related-work positioning.** TSci is the "general-purpose, pick-and-ensemble from a library,
  one pass, frontier LLM, point forecasts" end of the design space. TS-Agent is the "refine
  vetted implementations, accept-or-revert" end. This project sits with TS-Agent and adds
  epidemic-phase evaluation, probabilistic scoring and a cited knowledge bank. TSci makes a
  useful contrast because it shows what is lost without those: on the one health dataset in
  its own benchmark, a single Prophet baseline beats it (Fig 11, read from figure).
- **Evaluation hygiene to claim.** We already refuse cross-window and cross-scheme
  comparisons, record the data hash, and run seeds. TSci shows none of this. We should not
  overclaim either: we also have no significance test. Adding a paired test across cutoffs and
  locations (for example a sign test or Diebold-Mariano on per-row WIS) before the January
  write-up would make this a real differentiator **(recommendation)**.
- **Baseline lesson.** TSci's missing "brute-force over the library" control is the same
  control our paper needs: the refinement loop must beat an Optuna or grid search over the same
  action space at equal compute, or the LLM's contribution is not shown **(recommendation)**.
- **Citation link.** TSci cites Adiga et al. 2023 for ensemble robustness across epidemic
  phases (p6). That is a natural hook: a general-purpose agent paper acknowledges phase
  robustness as a motivation and then never measures it.
