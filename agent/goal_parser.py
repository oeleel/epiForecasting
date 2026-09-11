"""Natural-language front end for `SelectionGoal` (TS-Agent Stage 1, roadmap 2.3).

`agent select-model --goal "which model is best at the peak"` has to become
`SelectionGoal(metric="wis", phase="peak")` before a multi-minute rolling
warm-up starts. This module is the whole mapping, and nothing else: it takes
one English sentence and returns one `ParsedGoal`. It never scores, never
fits, never touches disk.

The mapping is a *degradation ladder*, not an LLM call:

    1. deterministic keyword tables  — if they resolve BOTH fields, the LLM is
       never invoked at all
    2. one LLM call                  — only for the field(s) the tables missed
    3. exactly one repair retry      — the orchestrator's idiom, verbatim
    4. the caller's fallback goal    — with an honest warning attached

Design notes
------------
- **Determinism outranks the model.** A keyword hit wins over the LLM for that
  field, and a disagreement is recorded as a warning rather than silently
  resolved the model's way. English is ambiguous, but the model is not the
  authority on a word it can read verbatim; and the January paper's experiment
  log needs the same sentence to map the same way on every run. (The CLI pins
  `temperature=0.0` for the same reason.)
- **`parse_goal` never raises because of the LLM.** A junk response, a refused
  connection, a missing `langchain-openai` — all degrade to the fallback goal
  with a warning the CLI prints to stderr. It raises only on genuinely bad user
  input: empty text, an off-season request, or `rmse` scoped to a phase.
- **The `rmse` guard is the point of `_guard`.** `PhaseEvaluator._row_metrics`
  emits `mape/mae/bias/n` plus optional `wis/coverage_95` and no `rmse`, so
  `SelectionGoal(metric="rmse", phase="peak")` is individually legal, passes
  `__post_init__`, and then yields `incumbent=None` — but only *after* the full
  warm-up has burned several minutes. That silent trap is caught here, at
  parse time, where it costs nothing.
- **No pre-flight health check.** `LLMClient.is_available()` and `.diagnose()`
  each perform a real `invoke("Hi")` round-trip, so calling one to decide
  whether to call `invoke` doubles the network cost and doubles the failure
  surface. We wrap the single real invoke instead.
- Keyword tables are inline module constants rather than a config file. They
  are a curated vocabulary tied to this repo's metric names, not a knob; when
  they grow past a screen, that is the signal to externalize them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from agent.model_selection import (
    ALL_PHASES,
    DEFAULT_GOAL,
    VALID_METRICS,
    VALID_PHASES,
    SelectionGoal,
)
from agent.prompt_templates import (
    extract_json_from_response,
    format_goal_parse_prompt,
    validate_goal,
)

__all__ = [
    "METRIC_KEYWORDS",
    "PHASE_KEYWORDS",
    "REJECTED_PHRASES",
    "LLMLike",
    "ParsedGoal",
    "keyword_goal",
    "parse_goal",
]

_log = logging.getLogger(__name__)

# Substring -> value, scanned in order against text.lower(); FIRST hit wins, so
# the more specific phrase must come first ("weighted interval" before "wis",
# "calibrat" before "bias"). Substring matching is deliberate: it absorbs
# "calibration"/"calibrated" and "over-predicting"/"over-predicts" for free.
METRIC_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("coverage", "coverage_95"),
    ("calibrat", "coverage_95"),
    ("prediction interval", "coverage_95"),
    ("interval coverage", "coverage_95"),
    ("over-predict", "bias"),
    ("overpredict", "bias"),
    ("under-predict", "bias"),
    ("underpredict", "bias"),
    ("bias", "bias"),
    ("weighted interval", "wis"),
    ("wis", "wis"),
    ("probabilistic", "wis"),
    ("quantile", "wis"),
    ("mape", "mape"),
    ("percentage error", "mape"),
    ("rmse", "rmse"),
    ("root mean squared", "rmse"),
    ("mae", "mae"),
    ("absolute error", "mae"),
)

# The phases are calendar segments: onset Oct-Nov, peak Dec-Jan, decline
# Feb-Apr. Month names map onto their phase so "how is January" resolves
# without an LLM call.
PHASE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("peak", "peak"),
    ("mid-season", "peak"),
    ("midseason", "peak"),
    ("december", "peak"),
    ("january", "peak"),
    ("onset", "onset"),
    ("early season", "onset"),
    ("start of the season", "onset"),
    ("october", "onset"),
    ("november", "onset"),
    ("ramp", "onset"),
    ("decline", "decline"),
    ("downslope", "decline"),
    ("tail of the season", "decline"),
    ("february", "decline"),
    ("march", "decline"),
    ("april", "decline"),
    ("overall", ALL_PHASES),
    ("on average", ALL_PHASES),
    ("across the season", ALL_PHASES),
    ("whole season", ALL_PHASES),
    ("entire season", ALL_PHASES),
    ("average", ALL_PHASES),
)

# May-Sep is dropped by PhaseEvaluator before scoring, so a goal aimed at it can
# never be answered. Refuse loudly rather than quietly re-aim at "all".
REJECTED_PHRASES: tuple[str, ...] = ("off-season", "off season", "offseason", "summer")

# Per-phase rows carry no rmse (PhaseEvaluator._row_metrics), so this pairing is
# the one legal-but-unanswerable combination.
_PHASE_METRICS = ("wis", "mape", "mae", "bias", "coverage_95")

_REPAIR_SUFFIX = (
    "Please re-emit a valid JSON object matching the schema exactly. "
    "Output ONLY the JSON object — no prose, no fences."
)


class LLMLike(Protocol):
    """Structural type for the LLM handle `parse_goal` accepts.

    Duck-typed on purpose so `tests.agent.fakes.FakeLLM` and the real
    `agent.llm_client.LLMClient` are interchangeable with no adapter.
    `is_available` is part of the shape both objects have, but `parse_goal`
    never calls it — see the module docstring on pre-flight cost.
    """

    def invoke(self, prompt: str) -> str: ...

    def is_available(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ParsedGoal:
    """One English sentence, resolved to a `SelectionGoal`, with its provenance.

    `source` is the rung of the degradation ladder that produced `goal`:
    "keyword" (deterministic, fully or partly), "llm", "llm_repair", or
    "fallback" (the LLM was unusable). `warnings` is everything the user
    should see before committing to a multi-minute run.
    """

    goal: SelectionGoal
    source: str
    raw_text: str
    rationale: str | None = None
    warnings: tuple[str, ...] = ()

    def describe(self) -> str:
        return f'"{self.raw_text}" -> {self.goal.describe()} [{self.source}]'


def keyword_goal(text: str) -> tuple[str | None, str | None]:
    """Resolve (metric, phase) from `text` deterministically; None where unmatched.

    Pure and side-effect free — this is the rung the LLM never gets to skip.
    """
    if not isinstance(text, str):
        raise ValueError(f"goal text must be a string, got {type(text).__name__}")
    lowered = text.lower()
    metric = _match_keyword(lowered, METRIC_KEYWORDS)
    phase = _match_keyword(lowered, PHASE_KEYWORDS)
    return (metric[1] if metric else None, phase[1] if phase else None)


def parse_goal(
    text: str,
    llm: LLMLike | None = None,
    fallback: SelectionGoal = DEFAULT_GOAL,
) -> ParsedGoal:
    """Map one English sentence onto a `SelectionGoal`, degrading rather than failing.

    Args:
        text: The user's sentence, verbatim.
        llm: Anything with `invoke(prompt) -> str`. `None` means keyword-only.
        fallback: The goal whose fields fill anything left unresolved.

    Returns:
        A `ParsedGoal` carrying the goal, how it was derived, and any warnings.

    Raises:
        ValueError: On bad *user input* only — empty text, an off-season
            request, or `rmse` scoped to a single phase. Never on LLM trouble.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"goal text must be a non-empty string, got {text!r}")

    lowered = text.lower()
    for phrase in REJECTED_PHRASES:
        if phrase in lowered:
            raise ValueError(
                f"the May-Sep off-season is never scored, so {phrase!r} is not a goal; "
                f"legal phases are {list(VALID_PHASES)} "
                f"(onset Oct-Nov, peak Dec-Jan, decline Feb-Apr, 'all' = whole window)"
            )

    kw_metric_hit = _match_keyword(lowered, METRIC_KEYWORDS)
    kw_phase_hit = _match_keyword(lowered, PHASE_KEYWORDS)
    kw_metric = kw_metric_hit[1] if kw_metric_hit else None
    kw_phase = kw_phase_hit[1] if kw_phase_hit else None

    # Rung 1: fully deterministic. The LLM is not invoked, not constructed, not
    # even checked for reachability.
    if kw_metric is not None and kw_phase is not None:
        _guard(kw_metric, kw_phase)
        return ParsedGoal(
            goal=SelectionGoal(metric=kw_metric, phase=kw_phase),
            source="keyword",
            raw_text=text,
        )

    if llm is None:
        return _without_llm(text, kw_metric, kw_phase, fallback)

    prompt = format_goal_parse_prompt(text, list(VALID_METRICS), list(VALID_PHASES))

    # Rung 2: one real invoke. Anything the transport raises (ConnectionError
    # from LLMClient, ImportError for a missing langchain-openai, an HTTP 500)
    # is a degradation, not a user error.
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # error boundary: the LLM must never sink a run
        return _degraded(text, kw_metric, kw_phase, fallback, f"{type(exc).__name__}: {exc}")

    try:
        goal, rationale, warnings = _resolve(response, kw_metric_hit, kw_phase_hit)
        return ParsedGoal(
            goal=goal, source="llm", raw_text=text, rationale=rationale, warnings=warnings
        )
    except ValueError as exc:
        # Rung 3: exactly one repair retry, quoting the validator's own message
        # back at the model. Idiom copied from agent.orchestrator._diagnose.
        _log.info("goal parse didn't validate (%s); retrying with error feedback", exc)
        repair = (
            prompt
            + "\n\n## Previous attempt failed validation\n"
            + f"Error: {exc}\n"
            + _REPAIR_SUFFIX
        )
        first_error = str(exc)

    try:
        response = llm.invoke(repair)
    except Exception as exc:
        return _degraded(text, kw_metric, kw_phase, fallback, f"{type(exc).__name__}: {exc}")

    try:
        goal, rationale, warnings = _resolve(response, kw_metric_hit, kw_phase_hit)
    except ValueError as exc:
        # Rung 4: two strikes. Degrade honestly rather than raise — the user
        # asked a legal question and the model failed to answer it.
        return _degraded(
            text, kw_metric, kw_phase, fallback, f"{first_error}; on retry: {exc}"
        )
    return ParsedGoal(
        goal=goal, source="llm_repair", raw_text=text, rationale=rationale, warnings=warnings
    )


# ---- private helpers -------------------------------------------------------


def _match_keyword(
    lowered: str, table: tuple[tuple[str, str], ...]
) -> tuple[str, str] | None:
    """First (phrase, value) in `table` whose phrase is a substring of `lowered`."""
    for phrase, value in table:
        if phrase in lowered:
            return (phrase, value)
    return None


def _guard(metric: str, phase: str) -> None:
    """Reject the legal-but-unanswerable pairings before the warm-up runs.

    Today that is exactly one: rmse is computed only on the overall section,
    so any non-'all' phase yields `incumbent=None` several minutes later.
    """
    if metric == "rmse" and phase != ALL_PHASES:
        raise ValueError(
            f"rmse is only computed overall; per-phase metrics are "
            f"{', '.join(_PHASE_METRICS)} — got phase={phase!r}. "
            f"Ask for rmse overall, or pick a per-phase metric."
        )


def _resolve(
    response: str,
    kw_metric_hit: tuple[str, str] | None,
    kw_phase_hit: tuple[str, str] | None,
) -> tuple[SelectionGoal, str | None, tuple[str, ...]]:
    """Parse+validate one LLM response, apply keyword overrides, guard the result.

    Raises ValueError on anything the caller should retry: unparseable JSON, a
    schema violation, an out-of-enum value, or a guarded pairing.
    """
    obj = extract_json_from_response(response)
    validate_goal(obj, list(VALID_METRICS), list(VALID_PHASES))
    metric = str(obj["metric"])
    phase = str(obj["phase"])

    warnings: list[str] = []
    metric, warn = _override(metric, kw_metric_hit, "metric")
    warnings += warn
    phase, warn = _override(phase, kw_phase_hit, "phase")
    warnings += warn

    _guard(metric, phase)
    return SelectionGoal(metric=metric, phase=phase), str(obj["rationale"]), tuple(warnings)


def _override(
    llm_value: str, kw_hit: tuple[str, str] | None, field_name: str
) -> tuple[str, list[str]]:
    """A deterministic keyword hit beats the LLM for its field; disagreement warns."""
    if kw_hit is None:
        return llm_value, []
    phrase, kw_value = kw_hit
    if kw_value == llm_value:
        return kw_value, []
    return kw_value, [
        f"parsed goal said {field_name}={llm_value!r} but your text contains "
        f"{phrase!r}; using {field_name}={kw_value!r}"
    ]


def _without_llm(
    text: str, kw_metric: str | None, kw_phase: str | None, fallback: SelectionGoal
) -> ParsedGoal:
    """Keyword hits plus the fallback's fields, with a warning per defaulted field."""
    warnings: list[str] = []
    metric = kw_metric
    if metric is None:
        metric = fallback.metric
        warnings.append(f"no LLM: defaulted metric to {metric!r}")
    phase = kw_phase
    if phase is None:
        phase = fallback.phase
        warnings.append(f"no LLM: defaulted phase to {phase!r}")

    _guard(metric, phase)  # genuine user input — raising here is correct
    return ParsedGoal(
        goal=SelectionGoal(metric=metric, phase=phase),
        source="keyword" if (kw_metric or kw_phase) else "fallback",
        raw_text=text,
        warnings=tuple(warnings),
    )


def _degraded(
    text: str,
    kw_metric: str | None,
    kw_phase: str | None,
    fallback: SelectionGoal,
    reason: str,
) -> ParsedGoal:
    """The LLM was unusable: keep every keyword hit, fill the rest, say so out loud."""
    metric = kw_metric or fallback.metric
    phase = kw_phase or fallback.phase
    _guard(metric, phase)  # only reachable via user text or a caller's fallback
    goal = SelectionGoal(metric=metric, phase=phase)
    _log.warning("goal parsing degraded to %s (%s)", goal.describe(), reason)
    return ParsedGoal(
        goal=goal,
        source="fallback",
        raw_text=text,
        warnings=(f"LLM goal parsing failed ({reason}); fell back to {goal.describe()}",),
    )
