# goal_parser - natural-language `select-model --goal` (demoted 2026-09-30)

## What this was

`python -m agent select-model --goal "which model is best at the peak"` mapped
one English sentence onto `SelectionGoal(metric, phase)` through a ladder:
deterministic keyword table -> LLM JSON parse -> one repair retry -> fallback
to `DEFAULT_GOAL`. Roadmap item 2.3; landed 2026-09-09.

## Why it was demoted

- Advisor, meeting 2026-09-10: the English front end is "cosmetic". Researchers
  write the structured spec (`--metric`, `--phase`, or a JSON/YAML goal) directly.
  Do not invest further.
- The 2026-09-11 desktop review left 10 open findings on this module
  (bare-substring keyword matches such as "wis" inside "Wisconsin", the
  `--metric/--phase` override bypassing the rmse/phase guard, first-in-tuple
  metric precedence, and so on). None will be fixed because the feature will
  not ship. Findings are preserved in
  `documentation/handoff-2026-09-11-desktop.md` section 3.
- It conflicts with the knowledge-bank direction (design doc
  `documentation/knowledge-bank-design.md`): the loop's inputs are structured
  entries, not free text, and a second free-text parser is one more thing to
  explain in January.

## What stayed in `agent/`

`SelectionGoal`, `VALID_METRICS`, `VALID_PHASES`, `DEFAULT_GOAL` in
`agent/model_selection.py`, and the `select-model --metric/--phase` flags in
`agent/cli.py`. `extract_json_from_response` in `agent/prompt_templates.py`
was shared with the improvement loop and stays there.

## Files

| File | Origin |
|---|---|
| `goal_parser.py` | `agent/goal_parser.py`, verbatim except the import of `prompt.py` |
| `prompt.py` | `GOAL_PARSE_PROMPT`, `REQUIRED_GOAL_KEYS`, `format_goal_parse_prompt`, `validate_goal` cut out of `agent/prompt_templates.py` |
| `test_goal_parser.py` | `tests/agent/test_goal_parser.py` plus the two goal tests that lived in `tests/agent/test_prompts.py` |

## Running the scrap tests by hand (not part of any suite)

```bash
PYTHONPATH=. .venv/bin/python scrap/goal_parser/test_goal_parser.py
```

They import `agent.model_selection` and `tests.agent.fakes`, so they only run
from the repo root with the project venv.
