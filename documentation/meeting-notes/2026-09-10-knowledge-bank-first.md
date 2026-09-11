# Meeting (week of 2026-09-08) — knowledge bank first, RL formalization ahead

Advisor sync after the TS-Agent + phase-papers review. Decisions below
reprioritize the roadmap.

## Decisions

- **No code-writing agents — confirmed and closed.** He agreed investing in
  agents that write model code is wrong for us ("we don't need to invest time on
  the agents writing the code"). The pipeline assumes a model arrives with code
  written; **assume standardized model input/output for now**. A code-porting /
  cleanup agent is a maybe-later, not now.
- **The research focus is agent communication + optimization**, not software
  polish: "how agents communicate with each other and come up with an optimized
  forecasting model — that should be our goal."
- **Priority order set: knowledge bank FIRST.** Of the three candidate tracks
  (knowledge bank, NL→structured commands, phase-based training): start the
  knowledge bank, *finish it*, then move on. All three eventually — they
  interlink — but one at a time.
- **The natural-language front end is explicitly LOW priority — "cosmetic."**
  Users are researchers who can write the commands in JSON directly. Do not
  invest further in the English-parsing layer for now. (The structured
  spec format itself still matters — it is what the researcher writes.)
- **New direction for the paper: a formal RL framing of the agent loop.** He has
  a write-up he will share: agents take actions, the reward is the change in the
  evaluation score (e.g. MAE decrease = positive reward), memory stores
  state→action→reward so that under the same conditions the agent picks the
  known-good action. His argument: TS-Agent does nothing formal; **quantified
  definitions of good vs. bad actions are how our paper distinguishes itself.**
  The knowledge bank is the prerequisite — it is what constrains the otherwise
  infinite action space.
- **Phase/regime sampling endorsed in passing**: "take information from this
  particular regime and only sample data from that regime... explicitly having
  these things will be useful for the agent." The calendar-vs-curve disparity I
  raised was acknowledged; no explicit ruling on δ or on replacing the labels —
  instead he gave a **third phase rule** to consider (below).

## Knowledge bank — his guidance

- **Sub-banks, not one bank.** His suggested categories: (1) model
  characteristics / which models do well, (2) input data, (3) forecasts.
  Mine (compatible): background/domain information vs. previous-runs takeaways.
- **The lab will supply their information** ("we can give you that") — but the
  *structure* is my task to propose.
- **Explore a knowledge graph** as an alternative/complement to flat banks —
  cursory research first, then decide.
- **Storage should be queryable** — "a database that you create which can be
  queried"; formalize how it is stored and retrieved.
- **Concrete seed example — an onset definition:** e.g. *three consecutive weeks
  of increase above a threshold = onset*. Derive per season from the time series
  (2022 onset ≈ Oct wk2, 2023 ≈ Oct wk3, ...), keep the variance, and as the
  expected onset approaches, the bank tells the model change is coming and an
  action may be needed. Definitions are subjective — "whatever works towards
  improving our model" — but must be principled.
  *(Note: this is a different rule from his papers' ±10% breakpoint
  classification — a candidate bank entry alongside it, and worth reconciling.)*

## Deliverables before next meeting (next Thursday; first cut to him ~Fri/Mon via Teams)

1. **Architecture diagram** of the overall system.
2. **Knowledge-bank design document**: sub-bank structure/categories, storage
   (queryable DB vs files), retrieval path into the agents, implementation notes.
3. **Knowledge-graph vs. flat-bank** cursory research with a recommendation.
4. Share intermediates over Teams for feedback before the meeting.

## Logistics

- **Rivanna**: Dustin is back; advisor will ping him. Porting is a *secondary*
  effort behind the knowledge-bank work.
- Iterate over Teams between meetings; intermediate outputs welcome there.
