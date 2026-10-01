"""KNOWN FACTS prompt block (design doc `knowledge-bank-design.md` section 6).

Retrieved entries reach the LLM as one compact, deterministic text block. This
module is the only place that block is formatted, so the improve loop, the
adapter's domain context, and the `knowledge query` CLI render byte-identical
text for the same entries.

Format
------
    KNOWN FACTS (knowledge bank; [C]=curated [D]=derived [E]=experiential):
    - [C, high] Season onset = 3 consecutive weeks of increase above threshold T.
    - [C, high] Loss weight lambda>1 ... -> try reweight_training_samples(dimension=peak)
    - [E, low, 1 run] input_size=52 on NHITS regressed WIS ~4x vs 26 (decline).

Invariants
----------
- One line per entry, in the order given (the store's query already ranks).
- The tag is `[<provenance letter>, <confidence>]`; experiential entries with
  a known `n_observations` add `, N run(s)`, because for them the count is the
  number of runs behind the claim. Curated and derived counts are not runs
  and are not shown.
- `payload.recommendation` renders as ` -> try <action>(<k>=<v>, ...)` with
  every param in authoring order, or ` -> no adapter action yet` for
  `not_yet_available`, so the LLM sees a concrete move or an explicit gap.
  A string param value that is not a plain token (`PLAIN_TOKEN_PATTERN`:
  letters, digits, `_ . -`) is quoted with `repr`, so a value such as
  `">1, to be learned"` cannot read as an operator plus a dangling param.
- `omitted > 0` appends a final `- (N more matching entries omitted; ...)`
  line: a truncated block must say so, never silently drop the tail.
- `llm_gloss` is appended as ` [interpretation: ...]`: always marked, never a
  bare claim (design section 5).
- The empty case renders `EMPTY_BLOCK`, never an empty string, so a prompt
  can never silently lose its knowledge section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from agent.knowledge.schema import PROVENANCES, RECOMMENDATION_NOT_YET_AVAILABLE, KnowledgeEntry

__all__ = [
    "DEFAULT_HEADER",
    "EMPTY_BLOCK",
    "GLOSS_MARKER",
    "PLAIN_TOKEN_PATTERN",
    "PROVENANCE_TAGS",
    "render_entry_line",
    "render_known_facts",
]

PROVENANCE_TAGS: Mapping[str, str] = {"curated": "C", "derived": "D", "experiential": "E"}
assert tuple(PROVENANCE_TAGS) == PROVENANCES, "PROVENANCE_TAGS must cover every provenance"

DEFAULT_HEADER = (
    "KNOWN FACTS (knowledge bank; "
    + " ".join(f"[{tag}]={prov}" for prov, tag in PROVENANCE_TAGS.items())
    + "):"
)
EMPTY_BLOCK = "(knowledge bank: no matching entries)"
GLOSS_MARKER = "interpretation"
_NOT_YET_AVAILABLE_TEXT = "no adapter action yet"

# A string param value rendered bare; anything else (spaces, commas, `>`,
# `=`, quotes) is repr-quoted so the `k=v, k=v` list stays unambiguous.
PLAIN_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _format_value(value: Any) -> str:
    if isinstance(value, str) and not PLAIN_TOKEN_PATTERN.match(value):
        return repr(value)
    return str(value)


def _format_params(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={_format_value(value)}" for key, value in params.items())


def _omitted_line(omitted: int) -> str:
    noun = "entry" if omitted == 1 else "entries"
    return f"- ({omitted} more matching {noun} omitted; narrow the retrieval context)"


def _render_recommendation(rec: Mapping[str, Any]) -> str:
    action = rec["action"]
    if action == RECOMMENDATION_NOT_YET_AVAILABLE:
        return f" -> {_NOT_YET_AVAILABLE_TEXT}"
    return f" -> try {action}({_format_params(rec['params'])})"


def render_entry_line(entry: KnowledgeEntry) -> str:
    """One `- [tag] statement` line for an entry."""
    tag_parts = [PROVENANCE_TAGS[entry.provenance], entry.confidence]
    n_obs = entry.evidence.get("n_observations")
    if entry.provenance == "experiential" and n_obs is not None:
        tag_parts.append(f"{n_obs} run" if n_obs == 1 else f"{n_obs} runs")
    line = f"- [{', '.join(tag_parts)}] {entry.statement}"
    rec = entry.recommendation
    if rec is not None:
        line += _render_recommendation(rec)
    if entry.llm_gloss is not None:
        line += f" [{GLOSS_MARKER}: {entry.llm_gloss}]"
    return line


def render_known_facts(
    entries: Iterable[KnowledgeEntry], *, header: str = DEFAULT_HEADER, omitted: int = 0
) -> str:
    """The KNOWN FACTS block for `entries`, or `EMPTY_BLOCK` when there are none.

    `omitted` is how many matching entries the caller's query limit cut off;
    when > 0 the block ends with an explicit omitted-count line.
    """
    if not isinstance(omitted, int) or isinstance(omitted, bool) or omitted < 0:
        raise ValueError(f"render_known_facts: omitted must be an integer >= 0, got {omitted!r}")
    lines = [render_entry_line(entry) for entry in entries]
    if not lines:
        return EMPTY_BLOCK
    if omitted > 0:
        lines.append(_omitted_line(omitted))
    return "\n".join([header, *lines])
