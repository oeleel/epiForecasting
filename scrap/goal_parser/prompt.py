"""Goal-parsing prompt + validator (DEMOTED 2026-09-30, kept for reference).

These three pieces (`GOAL_PARSE_PROMPT`, `format_goal_parse_prompt`,
`validate_goal`, plus `REQUIRED_GOAL_KEYS`) were cut verbatim out of
`agent/prompt_templates.py` when the natural-language goal front end was
demoted to `scrap/`. See `scrap/goal_parser/README.md` for why.

Not imported by anything under `agent/`; not collected by the documented test
commands. `scrap/goal_parser/goal_parser.py` is the only consumer.
"""

from __future__ import annotations

from typing import Any, List

__all__ = [
    "REQUIRED_GOAL_KEYS",
    "GOAL_PARSE_PROMPT",
    "format_goal_parse_prompt",
    "validate_goal",
]


# ============================================================================
# Goal parsing (Stage 1 natural-language front end)
# ============================================================================
#
# `agent select-model --goal "which model is best at the peak"` maps one
# English sentence onto a SelectionGoal(metric, phase). The deterministic
# keyword table in agent.goal_parser runs first; this prompt is only reached
# when the keywords leave a field unresolved.
#
# The legal metric/phase sets are passed IN rather than imported, so this
# module keeps its zero-import property (agent.model_selection pulls pandas).

REQUIRED_GOAL_KEYS: List[str] = ["metric", "phase", "rationale"]

# Deliberately short. LLMClient caps max_tokens at 4096 *including* qwen3
# reasoning tokens, so a long <think> block truncates the JSON into an
# unrecoverable parse failure. Every line here has to earn its place.
GOAL_PARSE_PROMPT = """\
You map one English sentence about forecast quality onto a machine-readable goal.

## The sentence

{goal_text}

## Legal metrics (pick exactly one)

{metric_block}

Direction: wis, mape, mae and rmse are better when lower. coverage_95 targets
0.95 exactly (not higher, not lower). bias targets 0 (signed over/under-prediction).

## Legal phases (pick exactly one)

{phase_block}

Phases are calendar segments of the flu season: Onset is Oct-Nov, Peak is
Dec-Jan, Decline is Feb-Apr. "all" means the whole scored window at once.
The May-Sep off-season is NOT a legal phase and is never scored - if the
sentence asks for summer or the off-season, choose "all" and say so in the
rationale.

Note: rmse is computed only overall. If the sentence asks for rmse, phase
must be "all".

## Output schema

{{"metric": "...", "phase": "...", "rationale": "<one sentence, plain English>"}}

The rationale is shown to the user before a multi-minute warm-up starts, so
say which words in the sentence drove the mapping.

Do not think out loud. Output ONLY the JSON object. No code fences, no commentary, no preamble.
"""


def format_goal_parse_prompt(
    text: str,
    valid_metrics: List[str],
    valid_phases: List[str],
) -> str:
    """Build the goal-parsing prompt for one English sentence.

    Args:
        text: The user's raw sentence, verbatim.
        valid_metrics: The metric names the caller will accept.
        valid_phases: The phase names the caller will accept.

    Returns:
        Fully formatted prompt string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"goal text must be a non-empty string, got {text!r}")
    if not valid_metrics:
        raise ValueError("valid_metrics must be a non-empty sequence")
    if not valid_phases:
        raise ValueError("valid_phases must be a non-empty sequence")

    metric_block = "\n".join(f"  - {m}" for m in valid_metrics)
    phase_block = "\n".join(f"  - {p}" for p in valid_phases)
    return GOAL_PARSE_PROMPT.format(
        goal_text=text.strip(),
        metric_block=metric_block,
        phase_block=phase_block,
    )


def validate_goal(obj: Any, valid_metrics: List[str], valid_phases: List[str]) -> None:
    """Validate a parsed goal dict against REQUIRED_GOAL_KEYS + the legal enums.

    Raises ValueError with a field-level message on the first violation, echoing
    the offending value so the repair retry can quote it back to the model.
    Returns None on success.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Goal must be a JSON object, got {type(obj).__name__}")

    missing = [k for k in REQUIRED_GOAL_KEYS if k not in obj]
    if missing:
        raise ValueError(f"Goal missing required keys: {missing}")

    if obj["metric"] not in valid_metrics:
        raise ValueError(
            f"goal 'metric' must be one of {list(valid_metrics)}, got {obj['metric']!r}"
        )

    if obj["phase"] not in valid_phases:
        raise ValueError(
            f"goal 'phase' must be one of {list(valid_phases)}, got {obj['phase']!r}"
        )

    if not isinstance(obj["rationale"], str) or not obj["rationale"].strip():
        raise ValueError("Goal 'rationale' must be a non-empty string")
