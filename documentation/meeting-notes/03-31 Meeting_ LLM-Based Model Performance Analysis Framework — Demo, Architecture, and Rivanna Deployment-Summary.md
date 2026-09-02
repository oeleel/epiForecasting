# 03-31 Meeting: LLM-Based Model Performance Analysis Framework — Demo, Architecture, and Rivanna Deployment

## Meeting Information
> **Date:** 2026-03-31 13:08:34
> **Location:** [Insert Location]
> **Participants:** [Speaker 1] [Speaker 2] [Speaker 3]
---
## Meeting Notes
### LLM-Based Model Performance Analysis Framework — Current Status
- **Speaker 2** built a working demo using **Ollama** with **Qwen3 8B** (local, ~5 GB) integrated with the existing **XGBoost** model.
- Demo script:
  - Loads CSV data and extracts XGBoost output metrics.
  - Computes phase-aware metrics and key observations via simple calculations.
  - Sends these details with prompt context to the LLM, which produces a performance analysis summary.
- The output is a **first-step demonstration**, not a production-level result.
- The framework currently supports **state-level spatial analysis** (e.g., identifying poor performance in regions like the Midwest).
**Conclusion:** The demo is a solid initial step aligned with the team’s vision. Speaker 1 confirmed it is useful and directionally correct.
---
### Framework Architecture — Modular and Domain-Agnostic Design
- Design uses a **two-layer structure**:
  - **Domain-agnostic layer**: General functions for obtaining context, metrics, data, and available actions.
  - **Domain-specific adapters**: Tailored for specific fields (e.g., flu forecasting with hyperparameter tuning, feature toggling, floor constraint adjustment; finance would have different metrics/diagnostics).
- This modularity allows adding new domains or models via new adapters.
- Evaluation logic currently resides in the domain-specific layer.
**Conclusion:** Speaker 1 validated the modular architecture as a good direction.
---
### Historical Run Tracking — SQLite Database Plan
- **Speaker 2** plans a **SQLite-based run tracker** to store metrics from previous runs.
- Planned CLI commands: list recent agent runs and compare specific run IDs.
- Enables referencing historical performance and comparing current results to benchmarks or prior weeks.
- This feature is **not yet implemented**.
---
### Integrating New Models into the Framework
- The framework currently **reads pre-existing model output** (CSV); it does not train models.
- Speaker 1 asked about plugging in a new model alongside XGBoost.
- Speaker 2 said it should be feasible with the current design but is not fully developed.
- Increasing **modularity** is the key requirement for easy model integration.
---
### LLM Model Selection and Scalability Considerations
- Currently using **Qwen3 8B** locally via Ollama; sufficient for summarization.
- Using **different models for different agents** was discussed:
  - Lightweight models for formatting-focused performance summaries.
  - More advanced models for heavy reasoning and improvement suggestions.
- Model upgrades can be evaluated later as needed.
---
### Spatiotemporal Forecast Summarization Capability
- Speaker 1 raised querying performance across **multiple states and time points** (e.g., “Which states are doing well? At which time points?”).
- Speaker 2 confirmed existing state-level analysis from CSV data.
- Speaker 1 suggested enabling **state-specific queries** (e.g., “How are forecasts performing for Florida?”).
- The system would need simple filtering and scoring across time horizons.
---
### Flu Forecasting Phase Definitions and Context Injection
- The LLM receives **prompt-injected context** (e.g., acceptable MAPE range of 40–60) to frame analysis.
- Speaker 1 noted the summary stated “error highest during onset and decline,” which may differ from typical observations (peak often has highest error).
- **Peak, onset, and decline definitions are subjective** — even CDC lacked a standard.
  - Pre-2019 CDC: 2–3 weeks of ground truth above a certain percentile = onset.
- Team agreed **user-supplied domain definitions** are acceptable and beneficial; changing definitions should change the summary.
**Conclusion:** The framework should support user-specified domain knowledge as context input; no universal standard is required.
---
### Deployment on UVA Rivanna HPC System
- Framework currently runs locally; the team aims to deploy on **UVA Rivanna** (HPC).
- Speaker 2 has **no prior Rivanna experience** (only CS server) and lacks confirmed allocation.
- Speaker 1 demonstrated **Open OnDemand** and indicated Speaker 2 may already have a class-based allocation (possibly from an AI course under Yen Ling).
- Key challenges:
  - Rivanna **cannot download from the internet** directly; Ollama must be downloaded locally, compressed, uploaded, and installed manually.
  - GPU access may require joining the **NSEC student allocation group**.
- Speaker 1 offered to:
  - Share step-by-step Ollama setup instructions for Rivanna.
  - Provide a repository link with setup details.
  - Contact IT (Dustin) to add Speaker 2 to the NSEC student allocation.
**Conclusion:** Deploying on Rivanna is the immediate priority so Speaker 1 can test against live forecast models and provide rapid feedback.
---
### GitHub Repository and README Documentation
- Code pushed to GitHub under **“OEL”** (Leo inverted: L-E-O → O-E-L).
- Repo shared with Speaker 1 and includes:
  - `documentation/` folder with the framework report.
  - Implementation files describing each component.
- **README not yet written**; Speaker 2 will add:
  - Local setup steps.
  - Model version (Qwen3 8B) and download link.
  - Homebrew-based installation steps (used for Ollama on Mac).
- Speaker 1 will review the repo and expects an email notification once the README is updated.
---
### Next Steps for LLM Reasoning and Iteration Loop
- After Rivanna deployment:
  - Have the LLM **identify root causes** of performance issues (beyond description).
  - Suggest **categorical next-best actions** (e.g., adjust hyperparameters, redefine feature windows).
  - Build the **improvement iteration loop**: one agent identifies issues; another modifies code or parameters.
- Speaker 1 confirmed these goals are valid but deprioritized until Rivanna setup is complete.
---
### User Interface Preferences — Terminal vs. Chat
- Speaker 2 asked about **terminal-based CLI** vs. **natural language chatbot**.
- Speaker 1: for now, **terminal/report-based output** is sufficient (e.g., weekly summary by state and horizon).
- A **chatbot interface** could be useful later for real-time model development (e.g., querying best models, requesting retraining).
- Current priority: keep it simple — agent processes in the background and delivers a final report.
---
## Next Arrangements
- [ ] Speaker 2 to verify Rivanna access and confirm if an existing class allocation is usable.
- [ ] Speaker 1 to contact IT (Dustin) to add Speaker 2 to the NSEC student allocation group on Rivanna.
- [ ] Speaker 1 to share Rivanna setup instructions and a repository link for installing Ollama (zip/tar upload method).
- [ ] Speaker 2 to write and push a README covering local setup steps, model version (Qwen3 8B), and download link.
- [ ] Speaker 2 to notify Speaker 1 via email once the README is updated.
- [ ] Speaker 2 to port the framework to Rivanna and get it running there.
- [ ] Speaker 2 to improve modularity to integrate a new model alongside XGBoost.
- [ ] Speaker 1 to review the OEL GitHub repository.
- [ ] Team to meet again on **Friday**.
---
## AI Suggestions
> **AI Suggestions**
> AI has identified the following issues that were not concluded in the meeting or lack clear action items; please pay attention:
> 1. **Rivanna GPU access is unresolved**: Speaker 2 does not currently have confirmed GPU access on Rivanna, which is required to run LLM models (Ollama). The NSEC allocation request to IT (Dustin) was discussed but not yet initiated — this is a potential blocker for deployment.
> 2. **SQLite run tracker is unimplemented**: The historical run tracking feature (storing previous run metrics, enabling run comparison and benchmark referencing) was described as planned but not started. Without it, the framework cannot support week-over-week performance comparison, which Speaker 1 identified as a key use case.
> 3. **LLM improvement iteration loop is not yet built**: The core agentic loop — where one agent identifies model issues and another agent acts on them (e.g., adjusting hyperparameters or retraining) — has not been implemented. This is the primary differentiator of the framework and currently only the summarization portion exists.
> 4. **Phase/peak definitions are not formalized as input**: The discussion confirmed that flu phase definitions (onset, peak, decline) are user-subjective and should be injectable via prompt context, but no structured mechanism for users to supply these definitions has been designed or implemented yet.
> 5. **Framework modularity for new model integration is incomplete**: Speaker 1 asked how easy it would be to add a new model to the system. Speaker 2 acknowledged the framework is "not fully fleshed out" in this regard. No concrete plan or timeline for improving modularity was established during the meeting.