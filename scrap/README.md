# scrap/ - demoted code, kept for reference only

Everything under this directory has been **demoted**: it is not imported by
`agent/` or `src/`, not collected by the documented test commands
(`pytest tests/agent/`, `tests/agent/run_all.py`), and not maintained. It is
kept so the reasoning and the code survive in one place instead of a git
archaeology dig.

Rules:

- Nothing in the live packages may import from here. This is a convention
  enforced by review, not by the interpreter: `scrap/` has no `__init__.py`
  but Python 3 still imports it as a namespace package
  (`import scrap.goal_parser.prompt` works from the repo root). `pytest.ini`
  lists `scrap` under `norecursedirs` so a bare `pytest` never collects it.
- Each subdirectory has its own `README.md` stating what it is, why it was
  demoted, and the date.
- To revive something, move it back under `agent/` or `src/`, restore its
  tests under `tests/agent/`, and re-add the module to `run_all.MODULES`.
  Do not import it from `scrap/` in place.

| Directory | Demoted | Why |
|---|---|---|
| `goal_parser/` | 2026-09-30 | Natural-language `select-model --goal` front end. Advisor (09-10): cosmetic, do not invest; 10 open review findings never worth fixing. The structured `SelectionGoal(metric, phase)` + `--metric/--phase` flags stay in `agent/`. |
