# Agent restructure - design note

2026-11-19. Design only. The advisor ranked this below the comparison study, so nothing here is
implemented and nothing should be this week. The note exists so the idea is written down while the
TimeSeriesScientist (TSci) code is fresh.

**Assumption to confirm.** I did not have the meeting transcript, only the todo line: "curator, planner,
forecaster, parallel reviewer agents with weights, standard report output". The reading of "weights" below
(each reviewer's vote is weighted when a change is accepted or rejected) is my interpretation.

## Where we are

Two LLM agents inside one loop, with deterministic code around them:

```
evaluate (code) -> Agent 1 diagnose -> Agent 2 propose one action -> validate (code)
      ^                                                                    |
      +--------------- retrain (code) <- apply (code) <--------------------+
```

A change is kept when one number, the target metric, improves by more than a threshold. Data quality
(`check-data`), model selection (`select-model`) and the run report are separate commands, not agents in
the loop.

## Proposed roles

| Role | Job | What exists today | What is new |
|---|---|---|---|
| Curator | Check the data before training; pick the knowledge-bank entries that apply to this model, phase and season week | `agent/data_quality.py`, the KNOWN FACTS retrieval in the orchestrator | Runs once at the start of every `improve` run instead of on request; its findings travel with the run |
| Planner | Read the phase-aware metrics, state where the model fails, choose the family and the phase to target | Agent 1 (diagnosis), `select-model`, `SelectionGoal` | One agent owns both "which model" and "what is wrong with it" |
| Forecaster | Propose one constrained action, apply it, retrain | Agent 2, the action catalog, `run_pipeline` | No change in substance |
| Reviewers (parallel) | Each judges the new forecast from one angle and votes keep or revert | Nothing; today one threshold decides | New. See below |
| Reporter | Write the standard report | `agent/run_report.py` | Report gains the reviewers' votes and reasons |

The loop stays. This is the point that separates us from TSci, whose four agents run once in a line.

## Parallel reviewers with weights

After each retrain, several reviewers score the same candidate independently. A first set:

| Reviewer | Looks at | Votes revert when |
|---|---|---|
| Target | The target metric in the target phase | It did not improve beyond the threshold |
| Collateral | Every other phase and horizon | Another segment got worse by more than a tolerance |
| Calibration | 50% and 90% interval coverage, signed bias | Coverage moved away from nominal |
| Knowledge | The cited bank entries against the action taken | The action contradicts a curated entry, or cites nothing |
| Significance | The paired test across seeds and cutoffs | The gain is inside seed noise |

Each reviewer returns `{vote, score, reason}`. The decision is the weighted sum of the scores against a
threshold. The weights live in validated config, start equal, and are a research question in their own
right: they can later be fitted from the experiential entries in the knowledge bank (which reviewer's
objections predicted a regression on held-out cutoffs).

Four of the five need no LLM: target, collateral, calibration and significance are arithmetic on the
metrics the evaluator already produces. Only the knowledge reviewer, and an optional free-text critic, are
LLM calls. That keeps the cost of "parallel reviewers" close to zero on a local 8B model and keeps the
decision reproducible.

## Hand-off between agents: TSci and ours

**TSci (from its released code, commit `41b3963`).** One mutable Python dict is the LangGraph state. Each
node reads the keys it needs and adds its own: `preprocessed_data`, `analysis_result`,
`selected_models`, `best_hyperparameters`, `forecast_result`, `report`. The LLM is asked for JSON in the
prompt text. The output is parsed, and when parsing fails the code falls back silently to a default (for
model selection: the first `k` models in the list, with empty hyperparameters). The analysis is handed to
the next agent as the raw string, pasted into the next prompt. Nothing checks a hand-off against a schema,
so a malformed analysis degrades the run without stopping it, and the report does not say so.

**Ours.** Agent 1 returns a diagnosis object that is checked against `DIAGNOSIS_SCHEMA`
(`summary`, `headline_metrics`, `weak_segments` with `dimension` / `value` / `metric` /
`delta_vs_overall` / `severity`, `hypotheses`, `suggested_focus`). Agent 2 returns
`{name, params, rationale, expected_effect, cited_entries}`, and the action is validated against the
catalog and the parameter bounds before anything is applied. Citations that were not retrieved are dropped
and logged. Each iteration is stored with its status (`applied`, `skipped`, `no_op`, `stop`).

**The difference.** TSci passes loosely typed text forward and tolerates failure; we pass a small validated
object and record what happened to it. Ours is stricter, but it is still two ad hoc JSON shapes, one per
agent pair.

**Proposal for the restructure.** One envelope for every hand-off, as a frozen dataclass:

```
Handoff(sender, receiver, iteration, kind, payload, evidence, cited_entries, status)
```

`payload` is the role-specific object (a diagnosis, an action, a review). `evidence` is the metric rows
the sender based it on, so a reviewer or a human can check a claim against numbers without re-running
anything. Every envelope is written to the run tracker, and the report is generated from the envelopes
alone. This is the "standard report output": the report is a rendering of the hand-off log, not a second
source of truth.

## Open decisions (advisor / Leo)

1. Is the weighted-vote reading of "reviewer agents with weights" the intended one?
2. Should the Curator be allowed to change the data (drop a location, impute), or only to report?
3. Should the Planner choose the model family inside the loop, or does selection stay a separate stage?
4. Fixed weights, or weights learned from the bank? Learned weights need a held-out season to avoid
   fitting the reviewers to the same windows they judge.

## What not to do

- Do not add agents that only relay text. Every role above either calls code that already exists or
  produces a decision that is recorded.
- Do not let reviewers run as LLM debates. Arithmetic reviewers first; add LLM critics only if they
  catch something the arithmetic ones miss.
