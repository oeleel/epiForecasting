# Knowledge bank

The knowledge bank is the memory the agent loop reads before it proposes a change: the
lab's domain rules and guardrails, facts derived from the data, and lessons from past runs,
stored as one queryable SQLite table. It is not a document store and not a place for
free-form notes; every entry is a single validated fact with retrieval keys, and the loop
cites the entry it acts on but is still free to explore past it (advisor, 2026-09-24).

Design: `documentation/knowledge-bank-design.md`. Code: `agent/knowledge/` (read
`schema.py` first). CLI: `python -m agent knowledge {validate,rebuild,list,query}`.

## Three provenances

| provenance | who writes it | where it lives | trust |
|---|---|---|---|
| `curated` | a human (this directory, PR-reviewed YAML) | `knowledge/curated/*.yaml` | highest |
| `derived` | a deterministic job over the CDC data (not in v1) | `knowledge.db` only | middle |
| `experiential` | the end-of-run bank-writer (not in v1) | `knowledge.db` only | lowest, rises with `n_observations` |

`knowledge.db` is gitignored and rebuilt from the YAML files every time the bank opens, so
there is no compile step and the files here are the source of truth for curated rows.

## The YAML shape (one annotated example)

```yaml
- id: onset-definition-v1          # slug, unique across all files; version on meaning change
  provenance: curated
  category: domain_dynamics        # model_characteristics | input_data | forecasts | domain_dynamics
  statement: >                     # ONE sentence; exactly what the LLM sees
    Season onset = 3 consecutive weeks of increase above threshold T.
  entities: {phase: onset}         # typed refs (model, phase, data_source, season, metric)
  context: {season_week: [38, 46]} # WHEN it applies: phase/model/metric lists, season_week [lo, hi]
  payload: {rule: {consecutive_weeks: 3}}      # free JSON; see recommendation convention below
  evidence: {source: "advisor (A. Adiga), meeting 2026-09-24", n_observations: null, reward_delta: null}
  confidence: medium               # high | medium | low
  llm_gloss: null                  # reserved for an LLM's interpretation; never a retrieval key
  created_at: 2026-09-30
```

`context` keys are the retrieval keys; omitting one means "applies always". `season_week`
is the epi week of the year (week 40 is early October) and cannot wrap the year boundary.
"What to do" entries add `payload.recommendation: {action, params, note?}` where `action`
is an adapter action name (`reweight_training_samples`, ...) or `not_yet_available`; the
loop renders it as `-> try action(k=v)` or `-> no adapter action yet`.

## How to add an entry

1. Copy `curated/_TEMPLATE.yaml` to `curated/<topic>.yaml` (no leading underscore; `_` files
   are skipped) or append to an existing file. Every file is a YAML list.
2. Fill every field. Cite the source (person + date, paper, or job name) in `evidence.source`.
3. `python -m agent knowledge validate` - it prints per-file counts or every error at once.
4. Open a PR. The bank picks the entry up on the next open; no other step.

## How the loop sees entries

Matching entries are rendered into one block that is appended to the model description in
every diagnose/propose prompt (`agent/adapters/flu_forecast.py::get_domain_context`):

```
KNOWN FACTS (knowledge bank; [C]=curated [D]=derived [E]=experiential):
- [C, high] Season onset = 3 consecutive weeks of increase above threshold T.
- [C, high] Otherwise reweight the loss: ... -> try reweight_training_samples(dimension=approaching_peak, ...)
- [E, low, 1 run] input_size=52 on NHITS regressed WIS ~4x vs 26 (decline).
```

Order is provenance trust (curated > derived > experiential), then confidence, then
`n_observations` (most first), then `updated_at` (newest first), then `id`; the
`ORDER BY` lives in `agent/knowledge/store.py`. `llm_gloss` is always
rendered as `[interpretation: ...]`: it is an LLM's reading of an entry, never the fact.

## Files in `curated/`

- `rectification-actions.yaml` - the advisor's five actions for a model that fails at the peak.
- `training-strategy.yaml` - window-length rules, the seasonality data requirement, onset definition.
- `domain-context.yaml` - calendar phases and performance conventions migrated from the adapter.
- `_TEMPLATE.yaml` - annotated entry to copy. `_training-strategy-by-model-class.yaml` - intake stub.

## What we are asking the lab to write next

Per-model-class training strategies (advisor, 2026-09-24): statistical/autoregressive
models and ML models get different rules for window length, seasonal history, and when to
retrain. The skeleton is `curated/_training-strategy-by-model-class.yaml`: replace the TODO
statements, list the families each rule covers in `context.model`, drop the underscore from
the filename, validate, PR.
