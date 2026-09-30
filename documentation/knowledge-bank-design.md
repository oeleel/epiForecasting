# Knowledge Bank Design

**Deliverable for the 2026-09-17 advisor meeting.** Covers the three asks from
the 09-10 sync (`meeting-notes/2026-09-10-knowledge-bank-first.md`): the
knowledge-bank design, the overall architecture diagram (§8), and the
knowledge-graph vs flat-bank recommendation (§7).

Status: **v1 core + curated intake implemented 2026-09-30** (`agent/knowledge/`,
`knowledge/curated/`; §9 units 1-2). Unit 3 (derived jobs), unit 4 (loop
injection point) and unit 5 (experiential bank-writer) are not started.

---

## 0. Decisions at a glance

| Decision | Choice | Why (one line) |
|---|---|---|
| Primary partition | **Provenance**: curated / derived / experiential | Every content category exists on all three sides; provenance decides trust, refresh, and write path |
| Advisor's categories | Tags on entries (+ new `domain_dynamics`) | Preserved as the query dimension he thinks in; onset rule fits none of the original three |
| Entry schema | One uniform table, typed entity refs, JSON payload | One retrieval path for all knowledge; rows are graph-ready edges |
| Storage | SQLite (`knowledge.db`); curated side authored as YAML files, auto-compiled at startup | "A database that you create which can be queried" - serverless, joins native, lab edits text files via PR, zero sync step |
| Write timing | Read at every proposal step; **write once at end of run** | Mid-run "lessons" can be falsified by later iterations; end-of-run rewards are final and aggregatable |
| Bank-writer | Deterministic core fields + optional LLM gloss (marked as interpretation) | Queryable fields can never be hallucinated; small local LLM must not write causal claims into permanent memory |
| Retrieval | Deterministic, context-keyed (phase, season week, model, metric) | We always know the context at proposal time; no embeddings or LLM-composed queries needed at this scale |
| Guardrails | **Advisory + override log** (no hard blocks in v1) | The override log is itself paper evidence (does memory change behavior?); premature blocks turn one bad entry into a blind spot |
| KG vs flat | **Flat, graph-ready; defer the KG** until a real multi-hop query exists | Deferral costs zero: typed entity refs mean a graph is materializable from the flat bank at any time (§7) |

---

## 1. What the bank is, and its role in the RL framing

The framework's agent loop already produces raw experience: every iteration of
an improve run is logged to `runs.db` as (config, action, metric before/after) -
literally a state -> action -> reward tuple. What is missing is **memory that
crosses runs**: the agent that proposed a 52-week input window and regressed
WIS 4x will happily propose it again next week, because nothing carries the
lesson forward.

The knowledge bank is that memory, plus the domain knowledge the lab already
holds. In the RL formalization the advisor is drafting:

- **State** = forecasting context: current phase (surge/plateau/decline),
  week-of-season, model family, target metric, recent config.
- **Action** = an adapter action (change input window, swap loss, re-window
  training data, switch family).
- **Reward** = change in the evaluation score vs best-so-far (WIS decrease =
  positive).
- **The bank = the memory** that stores distilled (state, action, reward)
  evidence so that under similar conditions the agent prefers known-good
  actions and is warned off known-bad ones. It is what constrains the
  otherwise infinite action space.

Two layers, deliberately distinct:

| Layer | Granularity | Written | Exists today? |
|---|---|---|---|
| **Ledger** (`runs.db`) | Every iteration, raw | During the run, automatically | Yes (run tracker) |
| **Bank** (`knowledge.db`) | Distilled entries | Once, at end of run + curated/derived inputs | This design |

The ledger is evidence; the bank is what the evidence taught us.

## 2. Taxonomy: provenance first, categories as tags

The load-bearing split is **who produced the knowledge**, because that decides
how much to trust it, how it is refreshed, and who may write it:

- **`curated`** - a human wrote it. Lab expertise, findings from the papers,
  definitions. Slow-changing, git-reviewed, read-only to the system.
  *Example: "Season onset = 3 consecutive weeks of increase above threshold T."*
- **`derived`** - deterministic code computed it from data. Recomputed by a job
  when new data arrives; no human wrote it, no agent run produced it.
  *Example: per-season onset dates (2022 ~ Oct wk2, 2023 ~ Oct wk3, ...) and
  their variance, derived by applying the curated onset rule to the CDC series.*
- **`experiential`** - the agent loop produced it. Written by the end-of-run
  bank-writer; the RL memory.
  *Example: "input_size=52 on NHITS regressed WIS ~4x vs 26 in decline phase
  (2 observations, runs X, Y)."*

Note the advisor's own seed example straddles the split: the onset *rule* is
curated, the per-season onset *dates* are derived. That is why provenance must
be the partition - his example is two entries with different lifecycles.

The advisor's three categories survive as the **`category` tag** - the
dimension a query filters on:

- `model_characteristics` - which models do well, where, and with what settings
- `input_data` - data source properties, availability, leading-indicator value
- `forecasts` - properties of past forecasts and their errors
- `domain_dynamics` **(new)** - epidemiological regularities: onset/peak
  timing, phase transition behavior, season shape. The onset rule fits none of
  the original three; rather than shoehorn it, we extend the taxonomy by one.

## 3. Entry schema

One schema for all provenances. Curated entries are authored in YAML exactly
in this shape; derived and experiential entries are the same rows written by
code.

```yaml
id: onset-definition-v1            # stable slug; versioned on change
provenance: curated                # curated | derived | experiential
category: domain_dynamics          # model_characteristics | input_data |
                                   #   forecasts | domain_dynamics
statement: >                       # one-sentence human-readable fact
  Season onset = 3 consecutive weeks of increase above threshold T.

entities:                          # typed refs - the graph-ready part (§7)
  phase: onset                     # any of: model, phase, data_source,
                                   #   season, metric (all optional)

context:                           # retrieval keys - WHEN this is relevant
  season_week: [38, 46]            # inclusive week-of-season window
  # phase: [decline]               # applicable phases (omit = all)
  # model: [nhits]                 # applicable models (omit = all)

payload:                           # structured data, JSON; shape varies by
  rule:                            #   entry type, validated per-type
    consecutive_weeks: 3
    threshold: TBD                 # open question §10

evidence:
  source: "advisor, meeting 2026-09-10"   # citation, run_id(s), or job name
  n_observations: null             # experiential: how many runs support this
  reward_delta: null               # experiential: mean delta on target metric

confidence: high                   # high | medium | low
llm_gloss: null                    # optional LLM one-liner, ALWAYS marked as
                                   #   interpretation, never a queryable field
created_at: 2026-09-17
```

**Worked example 2 - derived (the onset dates):**

```yaml
id: onset-dates-2022-2025
provenance: derived
category: domain_dynamics
statement: "Observed onsets: 2022 wk41, 2023 wk42, 2024 wk41, 2025 wk43; std ~0.9 wk."
entities: {phase: onset}
context: {season_week: [36, 46]}     # retrievable as expected onset approaches
payload: {onsets: {"2022": 41, "2023": 42, "2024": 41, "2025": 43}, std_weeks: 0.9}
evidence: {source: "job derive_onsets vs data/raw CDC series", n_observations: 4}
confidence: high
created_at: 2026-09-20                  # required for every provenance; jobs fill it
```

This is exactly the advisor's "as the expected onset approaches, the bank tells
the model change is coming": the `context.season_week` window makes it
retrievable at precisely the right moment, and the refresh job keeps it current.

**Worked example 3 - experiential (the short-window lesson the framework
already rediscovered live):**

```yaml
id: exp-nhits-input52-decline-001
provenance: experiential
category: model_characteristics
statement: "input_size=52 on NHITS regressed WIS ~4x vs input_size=26 (decline phase)."
entities: {model: nhits, phase: decline}
context: {model: [nhits], phase: [decline]}
payload:
  state: {phase: decline, model: nhits, config: {input_size: 26}}
  action: {type: set_param, param: input_size, value: 52}
  reward: {metric: wis, delta: +301.4, direction: worse}
evidence: {source: "run 20260903-xxxxxx", n_observations: 1, reward_delta: 301.4}
confidence: low                     # single observation; rises with replication
created_at: 2026-09-03              # required; the bank-writer fills it from the run
```

One schema, three provenances, uniform retrieval. Repeated observations do not
create duplicate rows: the bank-writer merges into the existing entry,
incrementing `n_observations` and updating `reward_delta` and `confidence`.

## 4. Storage

**One SQLite database, `knowledge/knowledge.db`**, alongside the existing
`runs.db`. Rationale against the alternatives:

- Loose markdown/files only: not queryable - fails the advisor's explicit ask.
- Vector store: solves fuzzy similarity we do not have; adds an embedding
  dependency; at a few hundred entries SQL filters are exact and instant.
- Graph database: §7.
- Client/server DB (Postgres): operational weight with zero benefit at this
  scale; SQLite already ships with the project.

**Authoring split (least-friction rule):**

- **Curated** entries live as YAML files in `knowledge/curated/*.yaml` -
  git-tracked, PR-reviewed, browsable on GitHub as self-documentation. On
  startup the framework rebuilds the curated table from the files (hundreds of
  rows, milliseconds), so there is **no manual compile step and no
  stale-sync failure mode**. Lab members contribute knowledge by editing a
  text file with a documented template, never by writing SQL.
- **Derived** entries are written by refresh jobs (e.g. `derive_onsets`)
  directly to SQLite, tagged with the job name and input-data version.
- **Experiential** entries are written by the end-of-run bank-writer directly
  to SQLite.

## 5. Write path

**Reads happen at every proposal step; writes happen once, at end of run.**

Why not write mid-loop: at iteration 2 you do not yet know whether the
takeaway is true - iteration 4's revert can falsify it; the reward is only
final relative to best-so-far at run end; end-of-run writing aggregates
repeated signals within the run ("52-week window regressed twice" = one strong
entry, not two weak ones); and a run must never retrieve its own half-formed
conclusions.

**The bank-writer** is a post-run step in the orchestrator's finalize path,
sourced from what already exists - the run tracker rows and the run report:

1. For each iteration, extract (state, action, reward) deterministically from
   the `IterationRecord` - these fields *cannot* be hallucinated because no
   LLM touches them.
2. Keep only decisive signals: |reward| above a noise floor
   (`DEFAULT_MIN_REWARD_FRAC`, calibratable), or actions the loop reverted.
3. Merge into existing entries on (entities, action) match: bump
   `n_observations`, update mean `reward_delta`, recompute `confidence`
   (single observation = low; consistent replication = medium/high -
   thresholds are named constants, advisor-calibratable, §10).
4. Optionally ask the local LLM for a one-line `llm_gloss` - stored in a
   dedicated field, rendered with an "interpretation" marker, never used as a
   retrieval key. A wrong gloss cannot corrupt retrieval or the RL record.

## 6. Retrieval and injection into the agent loop

**Mechanism: deterministic, context-keyed.** At proposal time the framework
*knows* the context - current phase (from `phase_segmentation`), week-of-season,
model family, target metric. Retrieval is a SQL filter on the `context` keys
plus a relevance ordering. As implemented (`agent/knowledge/store.py`,
`_ORDER_BY_SQL`): provenance trust desc (curated 3 > derived 2 > experiential
1), confidence desc (high > medium > low), `n_observations` desc (NULL last),
`updated_at` desc, `id` asc - total and deterministic, so ties between curated
entries loaded in one rebuild fall to the id. No embeddings; no LLM-composed
queries - qwen3:8b writing its own queries is a reliability hole we do not
need at this scale.

Retrieved entries are rendered into a compact prompt block:

```
KNOWN FACTS (knowledge bank; [C]=curated [D]=derived [E]=experiential):
- [C, high] SEIR-family models over-predict during surge phases.
- [D, high] Expected onset ~week 42 (observed 41-43 across 2022-2025); 2 weeks out.
- [E, low, 1 run] input_size=52 on NHITS regressed WIS ~4x vs 26 in decline.
```

**Three injection points** (all spec'd; v1 builds B):

- **A - Stage 1, model selection.** Retrieve `model_characteristics` entries
  for the current phase/context into the selection prompt. (TS-Agent's
  case-bank move, on our bank.)
- **B - Stage 2, every proposal step (v1 scope).** Before the agent proposes a
  config action, retrieve entries matching current model + phase + metric.
  This is the RL memory read - the mechanism that makes the loop smarter than
  a memoryless search, and the paper's core claim.
- **C - diagnosis enrichment (fast-follow).** Derived `domain_dynamics`
  entries ("expected onset in ~2 weeks") into the diagnose step so error
  attribution can distinguish model failure from regime change.

**Guardrails: advisory + override log.** When a proposed action matches a
negative experiential entry, the framework does not block it - it verifies the
warning was in the prompt and logs an **override record** (entry id, proposed
action, outcome). Rationale: a hard block from one bad early entry becomes a
permanent blind spot, and the override log is itself experimental evidence -
*does memory change behavior, and is the agent right when it overrides?* -
which feeds directly into the RL evaluation section of the paper. Escalation
to hard blocks is a v2 decision, taken only once entries carry enough
`n_observations` to justify it.

## 7. Knowledge graph vs flat bank - research summary and recommendation

**Flat bank**: knowledge as rows; each entry self-contained; retrieval is
filtering. **Knowledge graph**: knowledge as nodes and typed edges
(`(NHITS) -[degrades_in]-> (decline)`); retrieval can *traverse* -
chaining relations answers questions nobody stored as a single fact:
*"entering decline -> which models degrade there -> which data sources help
those models?"* is three edges walked, and that inference across relations is
the KG's genuine selling point.

**What a KG costs here**: graph storage or a graph layer, a node/edge-type
schema designed before we know our query patterns, traversal query logic, and
a retrieval path complicated enough that either we hand-code every traversal
or let a small local LLM compose graph queries - the exact reliability trade
we rejected for plain SQL.

**The observation that decides it**: in our schema every entry already carries
typed entity references. A row with `entities: {model: nhits, phase: decline}`
and a negative reward *is* the edge `(nhits) -[degrades_in]-> (decline)`,
stored flat. The flat bank is a graph in edge-list form. Materializing an
actual graph (networkx) from it is ~50 lines and zero migration - so deferring
the KG costs nothing, while building it now buys nothing our v1 queries use:
**every v1 retrieval (§6) is a single-hop lookup.**

**Recommendation**: flat, graph-ready, with an explicit adoption trigger -
*the day an agent needs a multi-hop question answered* (e.g. cross-referencing
data sources with model-phase weaknesses in one retrieval), we materialize the
graph from the same rows and evaluate it against the flat baseline. Offer: a
small networkx prototype over the seeded bank, with one worked multi-hop
query, as a demo the following week.

## 8. Architecture diagram

```mermaid
flowchart TB
  subgraph DATA["Data layer"]
    CDC["CDC / FluSight data"] --> BRIDGE["data bridge (CDC to Nixtla long format)"]
  end

  subgraph BANKS["Model bank"]
    BRIDGE --> MB["9 model families - ForecastModel contract + registry"]
  end

  subgraph KB["Knowledge bank (this design)"]
    YAML["curated YAML (lab-authored, git-reviewed)"] -->|compiled at startup| KDB[("knowledge.db (SQLite)")]
    JOBS["derived-knowledge jobs (onset dates, phase stats)"] --> KDB
  end

  subgraph LOOP["Agent loop (orchestrator)"]
    GOAL["researcher goal (JSON spec)"] --> S1["Stage 1 - model selection"]
    S1 --> S2["Stage 2 - propose / evaluate / revert-on-regression"]
    S2 --> EVAL["evaluation (WIS, phase- and horizon-segmented)"]
    EVAL --> S2
  end

  KDB -->|"A: selection context"| S1
  KDB -->|"B: known outcomes at each proposal (v1)"| S2
  KDB -.->|"C: domain context (fast-follow)"| EVAL

  MB --> S2
  S2 --> LEDGER[("runs.db - per-iteration state/action/reward ledger")]
  EVAL --> REPORT["run report (report.md / report.json)"]
  LEDGER --> WRITER["bank-writer (end of run, deterministic + LLM gloss)"]
  REPORT --> WRITER
  WRITER -->|experiential entries| KDB
  S2 --> OVR["override log (advisory guardrail evidence)"]
```

The loop reads the bank at every proposal; the bank grows only between runs.
`runs.db` (raw experience) and `knowledge.db` (distilled memory) are the two
halves of the RL formalization's memory.

## 9. Implementation plan (v1)

| # | Unit | Contents |
|---|---|---|
| 1 | `agent/knowledge_bank.py` | `KnowledgeEntry` (frozen dataclass), SQLite store, YAML compiler, `query(context)` with trust/confidence ordering |
| 2 | `knowledge/curated/*.yaml` | Seed entries: onset rule; short-window rule; model-phase affinities from the two Adiga papers (SEIR/surge, LSTM/transitions, AR-Kalman/plateau); auxiliary-data findings |
| 3 | `agent/derived_knowledge.py` | Onset-dates job (curated rule + `phase_segmentation` over the CDC series) |
| 4 | Orchestrator integration | Injection point B: retrieval + prompt block at each proposal; override log |
| 5 | `agent/bank_writer.py` | End-of-run distiller from tracker rows + report; merge/confidence logic |
| 6 | Stage 1 integration (A) | Selection-prompt retrieval |
| 7 | Evaluation | Re-run a recorded regression scenario with the bank seeded; measure whether the known-bad action is avoided (first behavioral evidence for the paper) |

Tests ship with each unit (`tests/agent/test_<module>.py`, both gates). Units
1-5 are the Thursday-to-Thursday scope; 6-7 stretch.

## 10. Open questions for the advisor

1. **Onset threshold T** - what value (or per-season quantile) for "increase
   above a threshold"? And should the 3-consecutive-weeks rule and the papers'
   ±10% breakpoint classification coexist as *two* curated entries (our
   assumption), or be reconciled into one definition?
2. **Lab content intake** - we supply the YAML template + 3 worked examples;
   does the lab author entries directly, or hand us prose to transcribe?
3. **Confidence calibration** - proposed defaults: 1 observation = low,
   3+ consistent = medium, 5+ consistent = high (named constants,
   calibratable). Reasonable starting points?
4. **KG prototype** - want the networkx materialization + one multi-hop demo
   query next week, or park it until a real multi-hop need appears?
