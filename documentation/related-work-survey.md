# Related-work survey (2023-2026)

Prepared 2026-11-04 for section 4 of `documentation/paper/outline.md`. Every paper below was checked
against the arXiv API (title, authors, date) or, for the three non-arXiv items, against doi.org metadata.
Summaries come from the arXiv abstracts; statements about internals that the abstract does not state are
marked "(check full text)". Nothing here is cited from memory without that check.

Our system, for reference: an LLM agent (Qwen 3 8B, local) that refines an existing XGBoost influenza
hospitalization model through a validated config-space action catalog, retrieves and cites entries from a
provenance-typed knowledge bank, scores WIS by epidemic phase and horizon, runs controlled ablations in an
experiment harness, and is planned to frame each change as a hypothesis with a posterior that the change
caused the gain.

Already cited in the outline and not repeated as new findings: TS-Agent, TimeSeriesScientist, Ziatdinov et
al. 2021, Adiga et al. (AAAI-23, IEEE Big Data 2022). They appear below only where the comparison needs them.

---

## 1. LLM agent frameworks for time-series forecasting and AutoML

### 1.1 Forecasting-specific agents

**TS-Agent.** Y. Ang, Y. Bao, L. Jiang, J. Tao, A. K. H. Tung, L. Szpruch, H. Ni. *Structured Agentic
Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback.* NeurIPS 2025.
[arXiv:2508.13915](https://arxiv.org/abs/2508.13915)
- Summary: A planner agent runs model selection, code refinement and fine-tuning as a structured iterative
  process, guided by curated knowledge banks of models and refinement strategies. Evaluated on financial
  forecasting and synthetic data generation against AutoML and agentic baselines.
- Overlap: selection then refinement stages, curated knowledge banks, audit trail. The closest prior design.
- Difference: TS-Agent edits code with a frontier LLM; we restrict the LLM to validated config actions so a
  small local model stays safe. Our bank is provenance-typed with checked citations and experiential
  entries; our evaluation is disaggregated by epidemic phase and horizon.

**TimeSeriesScientist (TSci).** H. Zhao, X. Zhang, J. Wei, Y. Xu, Y. He, S. Sun, et al. *TimeSeriesScientist:
A General-Purpose AI Agent for Time Series Analysis.* 2025.
[arXiv:2510.01538](https://arxiv.org/abs/2510.01538)
- Summary: Four agents (Curator, Planner, Forecaster, Reporter) handle diagnostics and preprocessing, narrow
  the model space, fit and ensemble, and write a transparent report. Reports 10.4% and 38.2% average error
  reduction against statistical and LLM baselines on eight benchmarks.
- Overlap: model selection on diagnostics, natural-language reports, white-box workflow.
- Difference: TSci builds a forecast from scratch per series and is domain-agnostic; we refine one
  existing domain model and evaluate it on epidemic phases. No knowledge bank or statistical test of
  changes is described in the abstract.

**TimeCopilot.** A. Garza, R. Rosillo. *TimeCopilot.* 2025.
[arXiv:2509.00616](https://arxiv.org/abs/2509.00616)
- Summary: Open-source agentic forecasting framework that combines time-series foundation models with LLMs
  behind one API. Automates feature analysis, model selection, cross-validation and forecast generation with
  natural-language explanations; reports state-of-the-art probabilistic results on GIFT-Eval.
- Overlap: LLM-agnostic orchestration over a bank of forecasting families, cross-validated selection
  (our Stage 1).
- Difference: selection and explanation only, no iterative refinement of a chosen model, no domain bank.

**FLAIRR-TS.** G. Jalori, P. Verma, S. O. Arik. *FLAIRR-TS: Forecasting LLM-Agents with Iterative Refinement
and Retrieval for Time Series.* Findings of EMNLP 2025.
[arXiv:2508.19279](https://arxiv.org/abs/2508.19279)
- Summary: A Forecaster agent produces forecasts, a Refiner agent rewrites the prompt from past outputs and
  retrieved context, and a Retrieval agent supplies similar historical segments. Test-time prompt
  optimization without model training.
- Overlap: iterative refinement plus retrieval of past cases.
- Difference: the LLM is the forecaster and the prompt is the refined object; in our system the LLM never
  forecasts and the refined object is the training configuration of a separate model.

**Last-mile forecasting agent.** Y. Liao, Z. Wang, Q. Nie, Z. Zhang. *Bridging the Last Mile of Time Series
Forecasting with LLM Agents.* 2026. [arXiv:2606.02497](https://arxiv.org/abs/2606.02497)
- Summary: An LLM agent sits on top of a forecasting backbone and turns retrieved business context into
  explicit forecast revision actions under structural safety constraints, with a memory bank for post-hoc
  reflection. Shown through real-world case studies.
- Overlap: explicit, constrained revision actions over an existing backbone; memory bank; auditability.
- Difference: it revises forecast outputs; we change how the model is trained. Case-study evaluation, no
  controlled ablation or significance testing per the abstract.

**Context is Key (CiK).** A. R. Williams, A. Ashok, E. Marcotte, V. Zantedeschi, et al. *Context is Key: A
Benchmark for Forecasting with Essential Textual Information.* ICML 2025.
[arXiv:2410.18959](https://arxiv.org/abs/2410.18959)
- Summary: A benchmark of forecasting tasks that pair numeric series with textual context that is necessary
  to forecast well, used to test LLM-based forecasters.
- Overlap: motivation that domain knowledge in text changes forecasts.
- Difference: a benchmark for LLM forecasters, not a refinement framework. Cite as motivation only.

### 1.2 General ML engineering and AutoML agents

**DS-Agent.** S. Guo, C. Deng, Y. Wen, H. Chen, Y. Chang, J. Wang. *DS-Agent: Automated Data Science by
Empowering Large Language Models with Case-Based Reasoning.* ICML 2024.
[arXiv:2402.17453](https://arxiv.org/abs/2402.17453)
- Summary: Retrieves and adapts past expert cases (Kaggle solutions) with case-based reasoning, then revises
  plans from execution feedback; a lighter deployment stage reuses past successful solutions.
- Overlap: case retrieval as the knowledge source for proposals; a TS-Agent baseline.
- Difference: generates full pipelines in code; cases are external, not the framework's own experiments;
  no domain-specific segment evaluation.

**AutoKaggle.** Z. Li, Q. Zang, D. Ma, J. Guo, T. Zheng, M. Liu, et al. *AutoKaggle: A Multi-Agent Framework
for Autonomous Data Science Competitions.* 2024. [arXiv:2410.20424](https://arxiv.org/abs/2410.20424)
- Summary: Multi-agent, phase-based workflow for tabular Kaggle tasks with iterative debugging and unit
  testing and a library of ML tools; supports user intervention.
- Overlap: phase-structured workflow, validated tool library, human can intervene.
- Difference: end-to-end from raw data; no refinement of an incumbent model or significance test.

**AIDE.** Z. Jiang, D. Schmidt, D. Srikanth, D. Xu, I. Kaplan, D. Jacenko, et al. *AIDE: AI-Driven
Exploration in the Space of Code.* 2025. [arXiv:2502.13138](https://arxiv.org/abs/2502.13138)
- Summary: Frames ML engineering as code optimization and runs tree search over candidate solution scripts,
  keeping and refining the best nodes.
- Overlap: greedy accept-the-best refinement over trials.
- Difference: unconstrained code space; search over a single validation score.

**AutoML-Agent.** P. Trirat, W. Jeong, S. J. Hwang. *AutoML-Agent: A Multi-Agent LLM Framework for
Full-Pipeline AutoML.* 2024. [arXiv:2410.02958](https://arxiv.org/abs/2410.02958)
- Summary: Multi-agent system that plans the full AutoML pipeline from a task description with
  retrieval-augmented planning, splits plans into sub-tasks for specialized agents, and verifies results.
- Overlap: retrieval-augmented planning and verification steps.
- Difference: builds pipelines from scratch, generic metrics.

**SELA.** Y. Chi, Y. Lin, S. Hong, D. Pan, et al. *SELA: Tree-Search Enhanced LLM Agents for Automated
Machine Learning.* 2024. [arXiv:2410.17238](https://arxiv.org/abs/2410.17238)
- Summary: Represents pipeline configurations as a tree and uses Monte Carlo Tree Search with LLM agents to
  explore and refine them from experimental feedback; 20 datasets.
- Overlap: search over a configuration space rather than free code.
- Difference: generic tabular AutoML; no domain bank or significance test.

**MLAgentBench.** Q. Huang, J. Vora, P. Liang, J. Leskovec. *MLAgentBench: Evaluating Language Agents on
Machine Learning Experimentation.* ICML 2024. [arXiv:2310.03302](https://arxiv.org/abs/2310.03302)
- Summary: Benchmark of ML experimentation tasks in which an agent edits files, runs experiments and must
  improve a baseline.
- Overlap: "improve an existing baseline" task framing.
- Difference: benchmark, not a method; unconstrained file edits.

**MLE-bench.** J. S. Chan, N. Chowdhury, O. Jaffe, J. Aung, et al. *MLE-bench: Evaluating Machine Learning
Agents on Machine Learning Engineering.* ICLR 2025. [arXiv:2410.07095](https://arxiv.org/abs/2410.07095)
- Summary: 75 Kaggle competitions as a benchmark for ML engineering agents, scored against human
  leaderboards.
- Overlap: evaluation setting used by AIDE, AutoMind and others.
- Difference: benchmark only. Useful as context for why our domain bench is narrower and deeper.

**R&D-Agent.** X. Yang, X. Yang, S. Fang, Y. Zhang, et al. *R&D-Agent: An LLM-Agent Framework Towards
Autonomous Data Science.* 2025. [arXiv:2505.14738](https://arxiv.org/abs/2505.14738)
- Summary: Splits data-science automation into a Researcher that proposes ideas and a Developer that
  implements them, iterating on feedback (check full text for details).
- Overlap: two-role propose/implement loop, like our diagnose/propose pair.
- Difference: code generation, general domain.

**Agent K.** A. Grosnit, A. Maraval, Refinath S N, Z. Zhao, et al. *Kolb-Based Experiential Learning for
Generalist Agents with Human-Level Kaggle Data Science Performance.* 2024.
[arXiv:2411.03562](https://arxiv.org/abs/2411.03562)
- Summary: An agent that learns from experience across Kaggle tasks via a Kolb-style cycle with memory,
  reaching human-level Kaggle performance.
- Overlap: experiential memory that grows from the agent's own runs.
- Difference: generalist, code-level; no curated domain bank or phase evaluation.

**CAAFE.** N. Hollmann, S. Mueller, F. Hutter. *Large Language Models for Automated Data Science: Introducing
CAAFE for Context-Aware Automated Feature Engineering.* NeurIPS 2023.
[arXiv:2305.03403](https://arxiv.org/abs/2305.03403)
- Summary: An LLM proposes semantically motivated features from dataset descriptions; each feature is kept
  only if it improves cross-validated performance.
- Overlap: domain context drives proposals; accept-if-better gating; one narrow action type.
- Difference: feature generation only; no knowledge bank.

**Empirical Research Assistance (ERA), general paper.** E. Aygun, A. Belyaeva, G. Comanici, M. Coram, et al.
*An AI system to help scientists write expert-level empirical software.* 2025.
[arXiv:2509.06503](https://arxiv.org/abs/2509.06503)
- Summary: LLM plus tree search writes scientific software that maximizes a quality metric, integrating
  research ideas from external sources. Among other domains, it generated 14 models that beat the CDC
  ensemble for COVID-19 hospitalization forecasting.
- Overlap: epidemic hospitalization forecasting judged by a hub metric. See 2.1 for the respiratory paper.
- Difference: code search with frontier models, large compute; not a test bench for an existing lab model.

### 1.3 LLM-guided hyperparameter optimization

**LLM for HPO.** M. R. Zhang, N. Desai, J. Bae, J. Lorraine, J. Ba. *Using Large Language Models for
Hyperparameter Optimization.* 2023. [arXiv:2312.04528](https://arxiv.org/abs/2312.04528)
- Summary: Prompts an LLM with the search space and trial history to propose the next configuration; matches
  or beats random search and Bayesian optimization under small budgets.
- Overlap: LLM proposes bounded numeric changes from history, as our `adjust_hyperparameter` does.
- Difference: no diagnosis, no domain knowledge, single aggregate metric.

**AgentHPO.** S. Liu, C. Gao, Y. Li. *Large Language Model Agent for Hyper-Parameter Optimization.* 2024.
[arXiv:2402.01881](https://arxiv.org/abs/2402.01881)
- Summary: An agent reads the task, runs trials and iterates on hyperparameters from historical trials,
  matching or beating human trials on 12 tasks with fewer evaluations and more interpretable logs.
- Overlap: interpretable, history-driven proposals.
- Difference: hyperparameters only; no training-data actions (reweighting, windows) and no segment metrics.

**LLAMBO.** T. Liu, N. Astorga, N. Seedat, M. van der Schaar. *Large Language Models to Enhance Bayesian
Optimization.* ICLR 2024. [arXiv:2402.03921](https://arxiv.org/abs/2402.03921)
- Summary: Integrates an LLM into Bayesian optimization as warm-starter, surrogate and candidate sampler;
  strongest gains early, when observations are sparse.
- Overlap: Bayesian framing of LLM proposals, relevant to the PI's planned Bayesian loop.
- Difference: optimizes an objective; it does not estimate whether a change caused an improvement.

---

## 2. LLM agents for epidemiological and public-health forecasting

**ERA for respiratory forecasting.** S. Martinson, M. P. Brenner, M. Plomecka, B. P. Williams, et al.
(including N. Reich and Z. Shamsi). *Prospective multi-pathogen disease forecasting using autonomous
LLM-guided tree search.* 2026. [arXiv:2605.16238](https://arxiv.org/abs/2605.16238)
- Summary: LLM-guided tree search generates, scores (rolling-origin WIS) and optimizes forecasting code for
  influenza, COVID-19 and RSV, mostly prompted with existing methods to implement. In the prospective
  2025-26 US season, an ensemble of its models ranked first in the CDC post-season evaluation and beat the
  hub ensembles for all three pathogens. Ablations study the optimization target and LLM tier.
- Overlap: same task (FluSight-style weekly hospitalization quantiles), same metric (WIS), same season.
  The most important comparison for reviewers.
- Difference: ERA writes new forecasting code with frontier models and large search; we refine an existing
  lab model through a small, validated action set on a local 8B model, with per-phase evaluation, a cited
  knowledge bank and (planned) a causal posterior per change. Our claim must be "auditable refinement of
  the lab's own model", not "better forecasts than ERA".

**EpiEvolve.** Y. Lu, S. Zeng, Z. Tang, M. Lau, et al. *EpiEvolve: Self-Evolving Agents for Streaming
Pandemic Forecasting under Regime Shifts.* 2026. [arXiv:2606.05513](https://arxiv.org/abs/2606.05513)
- Summary: Wraps a frozen LLM forecaster of weekly COVID-19 hospitalization trends in a self-evolving agent
  with hierarchical episodic memory, reflection on delayed labels, regime-aware retrieval and distilled
  strategic rules. Raises streaming accuracy from 0.561 to 0.629 and cuts recovery lag after regime shifts
  from 5 to 2 weeks; ablations credit each component.
- Overlap: experiential memory, regime (phase) awareness, chronological no-leakage protocol.
- Difference: predicts trend classes with an LLM; we forecast quantiles with XGBoost and the agent only
  changes training. Their regime-aware retrieval is a direct analogue of our phase-keyed retrieval.

**PandemicLLM.** H. Du, J. Zhao, Y. Zhao, S. Xu, et al. *Advancing Real-time Pandemic Forecasting Using Large
Language Models: A COVID-19 Case Study.* 2024 (later in Nature Computational Science).
[arXiv:2404.06962](https://arxiv.org/abs/2404.06962)
- Summary: Recasts real-time forecasting of COVID-19 hospitalization trends as text reasoning over
  multimodal inputs (policy, genomic, spatial, epidemiological series).
- Overlap: LLM reasoning in hospitalization forecasting.
- Difference: LLM as forecaster, not as model engineer.

**EpiLLM.** C. Gong, R. Sun, Y. Zheng, J. Zhang, et al. *EpiLLM: Unlocking the Potential of Large Language
Models in Epidemic Forecasting.* 2025. [arXiv:2505.12738](https://arxiv.org/abs/2505.12738)
- Summary: An LLM-based framework for spatio-temporal epidemic forecasting that aligns disease and mobility
  data with the language model.
- Overlap: domain-specific LLM forecasting.
- Difference: LLM as forecaster; no agent loop.

**Structured LLM agents for HFMD.** J. Chae, R. Wang, C. Xiong, G. Yunhan, et al. *Auditable Context-Aware
HFMD Forecasting with Structured LLM Agents.* 2025 (v1 title: *Beyond Curve Fitting: Neuro-Symbolic Agents
for Context-Aware Epidemic Forecasting*). [arXiv:2511.23276](https://arxiv.org/abs/2511.23276)
- Summary: An LLM Event Interpreter turns school calendars, weather and reports into a scalar
  transmission-impact signal; a Forecast Generator combines it with case history into probabilistic
  one-week forecasts with auditable rationales.
- Overlap: auditability and epidemiological context as requirements for public-health use.
- Difference: the LLM injects a covariate; it does not refine a model.

**EpidemIQs.** M. H. Samaei, F. D. Sahneh, L. W. Cohnstaedt, C. Scoglio. *EpidemIQs: Prompt-to-Paper LLM
Agents for Epidemic Modeling and Analysis.* 2025. [arXiv:2510.00024](https://arxiv.org/abs/2510.00024)
- Summary: Scientist and task-expert agents run literature review, network and mechanistic modeling,
  simulation, analysis and write-up through five predefined phases; 79% average task success.
- Overlap: multi-agent automation in epidemiology with predefined phases of work.
- Difference: mechanistic modeling studies, not forecast refinement.

---

## 3. Hypothesis-driven, Bayesian and statistically rigorous experimentation

**Hypothesis learning.** M. Ziatdinov, Y. Liu, A. N. Morozovska, E. A. Eliseev, X. Zhang, I. Takeuchi, et
al. *Hypothesis Learning in Automated Experiment: Application to Combinatorial Materials Libraries.* 2021.
[arXiv:2112.06649](https://arxiv.org/abs/2112.06649)
- Summary: Competing probabilistic physical models act as hypotheses inside an active-learning loop; the
  experiment selects measurements and rewards the hypothesis that best explains the data.
- Overlap: the PI's template for posterior-over-hypotheses experimentation.
- Difference: no LLM; hypotheses are a fixed set of physical models.

**POPPER.** K. Huang, Y. Jin, R. Li, M. Y. Li, et al. *Automated Hypothesis Validation with Agentic
Sequential Falsifications.* ICML 2025. [arXiv:2502.09858](https://arxiv.org/abs/2502.09858)
- Summary: LLM agents design falsification experiments for the measurable implications of a hypothesis and
  combine the dependent test results with e-values under strict Type-I error control.
- Overlap: the closest statistical template for "did this agent-proposed change really help": sequential
  tests with error control rather than a single best-score comparison.
- Difference: validates free-text scientific hypotheses on data; does not refine a model.

**E-valuator.** S. Sadhuka, D. Prinster, C. Fannjiang, G. Scalia, et al. *E-valuator: Reliable Agent
Verifiers with Sequential Hypothesis Testing.* 2025. [arXiv:2512.03109](https://arxiv.org/abs/2512.03109)
- Summary: Converts any verifier score into a sequential test with false-alarm control that stays valid at
  every step of an agent trajectory, via e-processes.
- Overlap: anytime-valid testing fits our iterative loop, where the number of trials is not fixed.
- Difference: monitors trajectory success, not the effect of a model change.

**BED-LLM.** D. Choudhury, S. Williamson, A. Golinski, N. Miao, et al. *BED-LLM: Intelligent Information
Gathering with LLMs and Bayesian Experimental Design.* ICLR 2026.
[arXiv:2508.21184](https://arxiv.org/abs/2508.21184)
- Summary: Chooses each next query by expected information gain under an LLM-built joint model and updates
  beliefs from responses; large gains on 20 Questions and preference elicitation.
- Overlap: a worked recipe for sequential Bayesian experimental design with an LLM, applicable to choosing
  the next ablation.
- Difference: question answering, not ML experiments.

**AutoDiscovery.** D. Agarwal, B. P. Majumder, R. Adamson, M. Chakravorty, et al. *AutoDiscovery: Open-ended
Scientific Discovery via Bayesian Surprise.* NeurIPS 2025. [arXiv:2507.00310](https://arxiv.org/abs/2507.00310)
- Summary: Treats the LLM as a Bayesian observer, measures the shift from prior to posterior belief about a
  hypothesis after an experiment, and uses that surprise as the reward for MCTS over hypotheses.
- Overlap: explicit LLM prior and posterior over hypotheses, the shape of the PI's proposed loop.
- Difference: open-ended data analysis; the posterior is the LLM's belief, not a statistical posterior on
  a measured effect. Our loop should compute the posterior from forecast scores, not elicit it.

**BoxLM.** M. Y. Li, E. B. Fox, N. D. Goodman. *Automated Statistical Model Discovery with Language Models.*
ICML 2024. [arXiv:2402.17879](https://arxiv.org/abs/2402.17879)
- Summary: Runs Box's loop with an LM that alternately proposes probabilistic programs and critiques them as
  a domain expert, including improving expert models under natural-language constraints.
- Overlap: model criticism loop; improving an existing expert model under constraints.
- Difference: generates probabilistic programs; no constrained action catalog.

**Curie.** P. T. J. Kon, J. Liu, Q. Ding, Y. Qiu, et al. *Curie: Toward Rigorous and Automated Scientific
Experimentation with AI Agents.* 2025. [arXiv:2502.16069](https://arxiv.org/abs/2502.16069)
- Summary: Embeds rigor in agent experimentation with intra-agent and inter-agent rigor modules (methodical
  control) and an experiment knowledge module; 3.4x better than the strongest baseline on 46 experimental
  questions.
- Overlap: controlled experiments and an experiment knowledge store, like our harness and experiential
  entries.
- Difference: general computer-science questions; rigor is procedural, not a quantified posterior.

**The AI Scientist.** C. Lu, C. Lu, R. T. Lange, J. Foerster, et al. *The AI Scientist: Towards Fully
Automated Open-Ended Scientific Discovery.* 2024. [arXiv:2408.06292](https://arxiv.org/abs/2408.06292)
- Summary: End-to-end pipeline that generates ideas, writes code, runs experiments and writes and reviews
  papers in ML subfields.
- Overlap: automated propose-run-evaluate cycle.
- Difference: full autonomy; we keep the researcher in charge of which hypotheses are admissible.

**AI co-scientist.** J. Gottweis, W.-H. Weng, A. Daryin, T. Tu, et al. *Accelerating scientific discovery
with Co-Scientist.* 2025. [arXiv:2502.18864](https://arxiv.org/abs/2502.18864)
- Summary: Multi-agent hypothesis generation, debate and ranking that works with scientists, validated in
  biomedical studies (check full text for evaluation detail).
- Overlap: human-in-the-loop hypothesis generation.
- Difference: hypotheses are tested in the wet lab, not by the system.

**Hypothesis Search.** R. Wang, E. Zelikman, G. Poesia, Y. Pu, et al. *Hypothesis Search: Inductive
Reasoning with Language Models.* ICLR 2024. [arXiv:2309.05660](https://arxiv.org/abs/2309.05660)
- Summary: The LLM proposes natural-language hypotheses, which are turned into programs and checked against
  examples.
- Overlap: separate hypothesis proposal from verification.
- Difference: inductive puzzles, no statistics.

**IDEA.** K. He, M. Zhang, S. Yan, P. Wu, et al. *IDEA: Enhancing the Rule Learning Ability of Large Language
Model Agent through Induction, Deduction, and Abduction.* ACL 2025 Findings.
[arXiv:2408.10455](https://arxiv.org/abs/2408.10455)
- Summary: An agent forms a hypothesis from limited observations by abduction, tests it with planned
  actions, and revises it from evidence.
- Overlap: the abduction-deduction-induction cycle that the PI's "abductive loop" names.
- Difference: rule-learning games; no causal attribution of model gains.

**Error bars for evals.** E. Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model
Evaluations.* 2024. [arXiv:2411.00640](https://arxiv.org/abs/2411.00640)
- Summary: Statistical recipes (standard errors, clustered errors, paired differences, power) for reporting
  evaluation results.
- Overlap: paired comparison of two configs on shared test items maps directly to paired WIS differences
  over (location, date, horizon) rows.
- Difference: method note, no agent.

**Variance in ML benchmarks.** X. Bouthillier, P. Delaunay, M. Bronzi, A. Trofimov, et al. *Accounting for
Variance in Machine Learning Benchmarks.* MLSys 2021. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Summary: Shows that seed and data-split variance can exceed reported gains and recommends randomizing more
  sources of variation.
- Overlap: justifies the multi-seed sweep in our harness.
- Difference: pre-LLM; outside the 2023-2026 window, cited for method.

**Diebold-Mariano test.** F. X. Diebold, R. S. Mariano. *Comparing Predictive Accuracy.* Journal of Business
& Economic Statistics, 1995. [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- The standard test for equal predictive accuracy of two forecasters under serially correlated losses.
  Applies to paired WIS differences across forecast dates. Outside the window, cited for method.

**WIS.** J. Bracher, E. L. Ray, T. Gneiting, N. G. Reich. *Evaluating epidemic forecasts in an interval
format.* PLOS Computational Biology, 2021.
[doi:10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)
- Fills the outline's "WIS for quantile forecasts [cite]".

---

## 4. Knowledge-bank, memory and retrieval-guided agents for ML engineering

**AutoMind.** Y. Luo, Y. Ou, J. Zheng, L. Wei, et al. *AutoMind: Adaptive Knowledgeable Agent for Automated
Data Science.* 2025. [arXiv:2506.10974](https://arxiv.org/abs/2506.10974)
- Summary: A curated expert knowledge base (papers and top competition solutions), knowledge-aware tree
  search and adaptive coding; beats prior agents on MLE-bench and other data-science benchmarks.
- Overlap: curated expert knowledge grounding each proposal; the closest analogue to our curated entries.
- Difference: generic data science; knowledge retrieval is not keyed on a diagnosed failure segment and
  citations are not checked (check full text).

**MLCopilot.** L. Zhang, Y. Zhang, K. Ren, D. Li, et al. *MLCopilot: Unleashing the Power of Large Language
Models in Solving Machine Learning Tasks.* EACL 2024. [arXiv:2304.14979](https://arxiv.org/abs/2304.14979)
- Summary: Stores past ML experiences and distilled knowledge, retrieves the relevant ones, and reasons over
  them to propose solutions (configurations) for new tasks.
- Overlap: experience store plus distilled knowledge feeding configuration proposals; very close to our
  experiential and derived provenance types.
- Difference: cross-task configuration suggestion, no iterative refinement or domain-segment evaluation.

**ExpeL.** A. Zhao, D. Huang, Q. Xu, M. Lin, et al. *ExpeL: LLM Agents Are Experiential Learners.* AAAI 2024.
[arXiv:2308.10144](https://arxiv.org/abs/2308.10144)
- Summary: The agent gathers experience across tasks, extracts natural-language insights, and recalls
  successful trajectories at inference, without weight updates.
- Overlap: the template for our deferred end-of-run experiential bank-writer.
- Difference: decision-making benchmarks, not ML.

**Reflexion.** N. Shinn, F. Cassano, E. Berman, A. Gopinath, et al. *Reflexion: Language Agents with Verbal
Reinforcement Learning.* NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Summary: Agents write verbal reflections on feedback into an episodic memory that guides later trials.
- Overlap: "RL memory" framing from the 2026-09-09 sync; fills the outline's experience-memory `[cite]`.
- Difference: reflections are unstructured and unverified; our entries carry provenance and confidence.

**Agent Workflow Memory.** Z. Z. Wang, J. Mao, D. Fried, G. Neubig. *Agent Workflow Memory.* 2024.
[arXiv:2409.07429](https://arxiv.org/abs/2409.07429)
- Summary: Induces reusable workflows from past trajectories and supplies them to the agent on later tasks.
- Overlap: reusable procedures distilled from experience.
- Difference: web navigation.

**Voyager.** G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, et al. *Voyager: An Open-Ended Embodied Agent with
Large Language Models.* 2023. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- Summary: An ever-growing skill library of verified code, retrieved for new tasks.
- Overlap: only verified items enter the library.
- Difference: embodied agent.

DS-Agent (case bank), Agent K (experiential memory), Curie (experiment knowledge module) and EpiEvolve
(hierarchical episodic memory with regime-aware retrieval) also belong to this group; see above.

---

## Comparison with the closest works

Legend: Yes / Partial / No. "?" means the abstract does not settle it; check the full text before the paper
uses the cell. "Ours" marks PLANNED where the feature is designed but not built.

| Work | Constrained actions | Knowledge bank | Domain / phase-aware eval | Refines existing model | Statistical significance of gains | Human-in-the-loop hypotheses |
|---|---|---|---|---|---|---|
| **Ours** | Yes (validated config actions, guardrails) | Yes (curated / derived / experiential, checked citations) | Yes (WIS by phase, horizon, location) | Yes (lab XGBoost; any bank family) | PLANNED (posterior that a change caused the gain; multi-seed harness exists) | Yes (researcher sets phases, admissible strategies, approves actions) |
| TS-Agent (2508.13915) | Partial (staged workflow, code edits) | Yes (curated, read-only) | No (finance, aggregate loss) | Partial (refines vetted implementations) | ? | ? |
| ERA respiratory (2605.16238) | No (free code via tree search) | Partial (prompted with methods) | Partial (WIS, rolling origins, no phase split per abstract) | Partial (implements known methods as new code) | Partial (prospective hub evaluation, ablations) | Partial (humans choose method prompts) |
| TimeSeriesScientist (2510.01538) | No | No | No (generic benchmarks) | No (from scratch) | ? | No |
| DS-Agent (2402.17453) | No | Yes (external case bank) | No | No | ? | No |
| AutoMind (2506.10974) | No | Yes (curated expert base) | No | No | ? | No |
| EpiEvolve (2606.05513) | No | Yes (episodic and strategic memory) | Yes (variant regimes) | Partial (adapts a frozen LLM forecaster by context) | Partial (component ablations) | No |
| POPPER (2502.09858) | Partial (falsification tests) | No | No | No (validates hypotheses) | Yes (Type-I error control, e-values) | Partial (human states the hypothesis) |
| Curie (2502.16069) | Partial (rigor modules) | Partial (experiment knowledge module) | No | No | Partial (procedural rigor) | ? |

## The gap this paper fills

Recent agents either build forecasting pipelines from scratch with free code generation (TSci, AutoKaggle,
AIDE, ERA) or use the LLM as the forecaster itself (PandemicLLM, EpiLLM, EpiEvolve, FLAIRR-TS). ERA shows
that LLM search can produce winning influenza forecasts. However, it does so by writing new code with
frontier models, and it does not say which change caused which gain. The work on rigorous agent
experimentation (POPPER, E-valuator, Curie, AutoDiscovery) supplies error-controlled or Bayesian tests, but
none of it is applied to refining a forecasting model. No verified paper combines four things for an
existing operational epidemic model: (1) refinement through a small validated action set that a local 8B
model can drive safely; (2) a provenance-typed knowledge bank whose citations are checked; (3) evaluation
disaggregated by epidemic phase and horizon, where peak failures hide in pooled scores; and (4) a
statistical account of whether each accepted change caused the gain. Point (4) is still PLANNED, so the
paper can claim it only after the Bayesian/abductive loop exists. Until then, the safe claim is (1) to (3)
plus the multi-seed ablation harness. ERA is the paper reviewers will compare against. Frame our work as
auditable, phase-aware refinement of a lab's own model, not as a competitor on hub leaderboard accuracy.
