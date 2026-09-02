"""Prompt templates for the agentic forecasting framework.

Templates are parameterized strings — the framework fills in metrics
computed by the adapter before sending to the LLM.

Two prompt families:

    SUMMARY_PROMPT             — Milestone 1: free-text diagnostic report.
                                  Used by `agent summarize`.

    STRUCTURED_DIAGNOSIS_PROMPT — Milestone 2 (Agent 1, the Analyst):
                                  same metric input but asks the LLM to
                                  emit JSON matching DIAGNOSIS_SCHEMA.
                                  This is the structured handoff that
                                  Agent 2 (the Engineer) consumes.

Validation helpers (`validate_diagnosis`, `extract_json_from_response`)
live alongside the templates so the orchestrator can repair malformed
LLM output without pulling in extra dependencies.
"""

import json
import re
from typing import Any, Dict, List


# ----------------------------------------------------------------------------
# Diagnosis schema (Agent 1 output)
# ----------------------------------------------------------------------------
#
# We deliberately use a hand-rolled validator instead of the `jsonschema`
# package so that the agent framework has zero new runtime dependencies.
# The schema below is the *contract* — every key listed in REQUIRED_KEYS
# must be present, and `weak_segments` / `hypotheses` are list-of-dict.

DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "summary":          {"type": "string"},
    "headline_metrics": {"type": "object"},
    "weak_segments":    {"type": "list_of_objects",
                         "item_keys": ["dimension", "value", "metric",
                                       "delta_vs_overall", "severity"]},
    "hypotheses":       {"type": "list_of_objects",
                         "item_keys": ["id", "text", "confidence"]},
    "suggested_focus":  {"type": "string"},
}
REQUIRED_DIAGNOSIS_KEYS: List[str] = list(DIAGNOSIS_SCHEMA.keys())


SUMMARY_PROMPT = """\
{domain_context}

## Current Evaluation Results

**Overall Performance** (across all locations and horizons):
{overall_metrics}

**Performance by Forecast Horizon**:
{horizon_metrics}

**Performance by Epidemic Phase**:
{phase_metrics}

**Bias Summary**:
{bias_summary}

**Top 5 Worst-Performing Locations** (by MAPE):
{worst_locations}

**Top 5 Best-Performing Locations** (by MAPE):
{best_locations}

**Top 10 Worst Individual Predictions** (by absolute error):
{worst_segments}

**Evaluation Period**: {date_range}
**Locations Evaluated**: {n_locations}

## Your Task

Write a concise performance analysis report for the engineering team. Structure it as:

1. **Overall Assessment** (2-3 sentences): How is the model performing overall? Is it acceptable?

2. **Geographic Patterns**: Which states/regions perform well vs. poorly? Are there regional clusters of high error?

3. **Temporal Patterns**: Which epidemic phases have the highest error? Does error grow significantly with forecast horizon?

4. **Bias Analysis**: Is the model systematically over-predicting or under-predicting? Does the bias direction change by phase (e.g., over-predicting during peaks, under-predicting during onset)?

5. **Areas for Investigation** (2-3 bullet points): Based on the patterns above, what specific aspects of the model should the team investigate to improve performance? Be concrete — reference specific phases, locations, or horizons.

Keep the report direct and actionable. Use specific numbers from the data. Do not repeat raw data tables — synthesize and interpret.
"""


def format_summary_prompt(metrics: dict, domain_context: str) -> str:
    """Build a complete SUMMARY_PROMPT from computed metrics.

    Args:
        metrics: Output of DomainAdapter.compute_metrics()
        domain_context: Output of DomainAdapter.get_domain_context()

    Returns:
        Fully formatted prompt string ready to send to the LLM
    """
    overall = metrics["overall"]
    overall_str = (
        f"- MAPE: {overall['mape']}%\n"
        f"- MAE: {overall['mae']} admissions\n"
        f"- RMSE: {overall['rmse']} admissions\n"
        f"- Bias: {overall['bias']:+.1f} admissions "
        f"({'over-predicting' if overall['bias'] > 0 else 'under-predicting'})\n"
        f"- Total forecasts evaluated: {overall['n_forecasts']}"
    )

    # Horizon metrics
    horizon_lines = []
    for h in sorted(metrics["by_horizon"].keys()):
        m = metrics["by_horizon"][h]
        horizon_lines.append(
            f"- Week {h}: MAPE={m['mape']}%, MAE={m['mae']}, "
            f"Bias={m['bias']:+.1f}, n={m['n']}"
        )
    horizon_str = "\n".join(horizon_lines)

    # Phase metrics
    phase_order = ["onset", "peak", "decline"]
    phase_lines = []
    for phase in phase_order:
        if phase in metrics["by_phase"]:
            m = metrics["by_phase"][phase]
            phase_lines.append(
                f"- {phase.capitalize()}: MAPE={m['mape']}%, MAE={m['mae']}, "
                f"Bias={m['bias']:+.1f}, n={m['n']}"
            )
    phase_str = "\n".join(phase_lines) if phase_lines else "No phase data available"

    # Bias summary
    bias_lines = [
        f"Overall bias: {overall['bias']:+.1f} "
        f"({'over-predicting' if overall['bias'] > 0 else 'under-predicting'})"
    ]
    for phase in phase_order:
        if phase in metrics["by_phase"]:
            b = metrics["by_phase"][phase]["bias"]
            direction = "over-predicting" if b > 0 else "under-predicting"
            bias_lines.append(f"  {phase.capitalize()}: {b:+.1f} ({direction})")
    bias_str = "\n".join(bias_lines)

    # Worst locations
    worst_lines = []
    for loc in metrics["worst_locations"]:
        worst_lines.append(
            f"- {loc['location']} (FIPS {loc['fips']}): "
            f"MAPE={loc['mape']}%, MAE={loc['mae']}, Bias={loc['bias']:+.1f}"
        )
    worst_str = "\n".join(worst_lines) if worst_lines else "No data"

    # Best locations
    best_lines = []
    for loc in metrics["best_locations"]:
        best_lines.append(
            f"- {loc['location']} (FIPS {loc['fips']}): "
            f"MAPE={loc['mape']}%, MAE={loc['mae']}, Bias={loc['bias']:+.1f}"
        )
    best_str = "\n".join(best_lines) if best_lines else "No data"

    # Worst segments
    seg_lines = []
    for seg in metrics["worst_segments"]:
        pct = f", PctError={seg['pct_error']}%" if seg.get("pct_error") is not None else ""
        seg_lines.append(
            f"- {seg['location']} on {seg['date']} (H{seg['horizon']}): "
            f"predicted={seg['predicted']}, actual={seg['actual']}, "
            f"error={seg['error']:+.1f}{pct}"
        )
    seg_str = "\n".join(seg_lines) if seg_lines else "No data"

    # Date range
    dr = metrics["date_range"]
    date_range_str = f"{dr['min']} to {dr['max']}"

    return SUMMARY_PROMPT.format(
        domain_context=domain_context,
        overall_metrics=overall_str,
        horizon_metrics=horizon_str,
        phase_metrics=phase_str,
        bias_summary=bias_str,
        worst_locations=worst_str,
        best_locations=best_str,
        worst_segments=seg_str,
        date_range=date_range_str,
        n_locations=metrics["n_locations"],
    )


# ============================================================================
# Milestone 2: Structured Diagnosis (Agent 1, the Analyst)
# ============================================================================

STRUCTURED_DIAGNOSIS_PROMPT = """\
{domain_context}

## Current Evaluation Results

**Overall Performance**:
{overall_metrics}

**By Forecast Horizon**:
{horizon_metrics}

**By Epidemic Phase**:
{phase_metrics}

**Top 5 Worst Locations** (by MAPE):
{worst_locations}

**Top 5 Best Locations** (by MAPE):
{best_locations}

**Top 10 Worst Individual Predictions**:
{worst_segments}

**Evaluation Period**: {date_range}
**Locations Evaluated**: {n_locations}

## Your Task

You are Agent 1 (the Analyst) in a two-agent improvement loop. Your job is
to read these metrics and produce a STRUCTURED DIAGNOSIS that Agent 2 (the
Engineer) will consume to choose an improvement action. You do NOT propose
actions yourself — you describe what is wrong and why.

Output a single JSON object (no surrounding prose, no markdown fences) that
matches this schema EXACTLY:

{{
  "summary": "<2-4 sentence natural-language overview of the model's behavior>",
  "headline_metrics": {{
    "mape": <float, percent>,
    "mae": <float>,
    "bias": <float, signed>,
    "n_forecasts": <int>
  }},
  "weak_segments": [
    {{
      "dimension": "<phase|horizon|location>",
      "value": "<the segment label, e.g. 'peak' or '4' or '06'>",
      "metric": "<mape|mae|bias>",
      "delta_vs_overall": <float, positive = worse than overall>,
      "severity": "<low|medium|high>"
    }}
    // Include 2-5 entries, ranked worst-first.
  ],
  "hypotheses": [
    {{
      "id": "h1",
      "text": "<one-sentence causal guess about WHY a weak segment is weak>",
      "confidence": <float in [0,1]>
    }}
    // Include 1-3 hypotheses. These are guesses, not certainties.
  ],
  "suggested_focus": "<a short snake_case label for the most pressing issue, e.g. 'peak_underprediction' or 'horizon4_drift' or 'midwest_bias'>"
}}

Rules:
- Output ONLY the JSON object. No code fences, no commentary, no preamble.
- Every key shown above is REQUIRED.
- `weak_segments` must be ranked worst-first.
- Severity buckets: high = clearly the dominant issue; medium = noticeable; low = minor.
- `hypotheses` should connect a weak segment to a plausible cause (e.g. log-target compression, over-regularization, missing feature, distribution shift). Do NOT list actions — that is Agent 2's job.
- If the model is uniformly healthy, return an empty `weak_segments` array, a single hypothesis stating "no significant issues identified", and `suggested_focus: "none"`.
"""


def format_structured_diagnosis_prompt(metrics: dict, domain_context: str) -> str:
    """Build the structured-diagnosis prompt from computed metrics.

    Reuses the same metric formatters as the free-text SUMMARY_PROMPT
    (overall, horizon, phase, locations, segments) so Agent 1 sees the
    same numbers in both modes.

    Args:
        metrics: Output of DomainAdapter.compute_metrics()
        domain_context: Output of DomainAdapter.get_domain_context()

    Returns:
        Fully formatted prompt string ready to send to the LLM in JSON mode.
    """
    overall = metrics["overall"]
    overall_str = (
        f"- MAPE: {overall['mape']}%\n"
        f"- MAE: {overall['mae']}\n"
        f"- RMSE: {overall['rmse']}\n"
        f"- Bias: {overall['bias']:+.1f}\n"
        f"- N: {overall['n_forecasts']}"
    )

    horizon_lines = [
        f"- Week {h}: MAPE={metrics['by_horizon'][h]['mape']}%, "
        f"MAE={metrics['by_horizon'][h]['mae']}, "
        f"Bias={metrics['by_horizon'][h]['bias']:+.1f}, "
        f"n={metrics['by_horizon'][h]['n']}"
        for h in sorted(metrics["by_horizon"].keys())
    ]
    horizon_str = "\n".join(horizon_lines)

    phase_order = ["onset", "peak", "decline"]
    phase_lines = [
        f"- {p.capitalize()}: MAPE={metrics['by_phase'][p]['mape']}%, "
        f"MAE={metrics['by_phase'][p]['mae']}, "
        f"Bias={metrics['by_phase'][p]['bias']:+.1f}, "
        f"n={metrics['by_phase'][p]['n']}"
        for p in phase_order if p in metrics["by_phase"]
    ]
    phase_str = "\n".join(phase_lines) if phase_lines else "(no phase data)"

    worst_lines = [
        f"- {loc['location']} (FIPS {loc['fips']}): MAPE={loc['mape']}%, "
        f"MAE={loc['mae']}, Bias={loc['bias']:+.1f}"
        for loc in metrics["worst_locations"]
    ]
    worst_str = "\n".join(worst_lines) if worst_lines else "(none)"

    best_lines = [
        f"- {loc['location']} (FIPS {loc['fips']}): MAPE={loc['mape']}%, "
        f"MAE={loc['mae']}, Bias={loc['bias']:+.1f}"
        for loc in metrics["best_locations"]
    ]
    best_str = "\n".join(best_lines) if best_lines else "(none)"

    seg_lines = []
    for seg in metrics["worst_segments"]:
        pct = f", PctError={seg['pct_error']}%" if seg.get("pct_error") is not None else ""
        seg_lines.append(
            f"- {seg['location']} on {seg['date']} (H{seg['horizon']}): "
            f"predicted={seg['predicted']}, actual={seg['actual']}, "
            f"error={seg['error']:+.1f}{pct}"
        )
    seg_str = "\n".join(seg_lines) if seg_lines else "(none)"

    dr = metrics["date_range"]
    return STRUCTURED_DIAGNOSIS_PROMPT.format(
        domain_context=domain_context,
        overall_metrics=overall_str,
        horizon_metrics=horizon_str,
        phase_metrics=phase_str,
        worst_locations=worst_str,
        best_locations=best_str,
        worst_segments=seg_str,
        date_range=f"{dr['min']} to {dr['max']}",
        n_locations=metrics["n_locations"],
    )


# ----------------------------------------------------------------------------
# JSON extraction + validation
# ----------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a possibly noisy LLM response.

    Tries, in order:
        0. Strip <think>...</think> blocks (Qwen3 reasoning tokens)
        1. Parse the entire response as JSON
        2. Extract a fenced ```json``` block
        3. Greedy match the first {...} substring
    Raises ValueError if no parseable JSON is found.
    """
    # Strip Qwen3-style thinking blocks
    text = _THINK_RE.sub("", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        try:
            return json.loads(block_match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Found JSON-like block but failed to parse: {e}") from e

    raise ValueError("No JSON object found in LLM response")


def validate_diagnosis(obj: Any) -> None:
    """Validate a parsed diagnosis dict against DIAGNOSIS_SCHEMA.

    Raises ValueError with a field-level message on the first violation.
    Returns None on success.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Diagnosis must be a JSON object, got {type(obj).__name__}")

    missing = [k for k in REQUIRED_DIAGNOSIS_KEYS if k not in obj]
    if missing:
        raise ValueError(f"Diagnosis missing required keys: {missing}")

    if not isinstance(obj["summary"], str) or not obj["summary"].strip():
        raise ValueError("Diagnosis 'summary' must be a non-empty string")

    if not isinstance(obj["headline_metrics"], dict):
        raise ValueError("Diagnosis 'headline_metrics' must be an object")

    if not isinstance(obj["suggested_focus"], str):
        raise ValueError("Diagnosis 'suggested_focus' must be a string")

    # weak_segments: list of dicts with required item keys
    if not isinstance(obj["weak_segments"], list):
        raise ValueError("Diagnosis 'weak_segments' must be a list")
    weak_required = DIAGNOSIS_SCHEMA["weak_segments"]["item_keys"]
    for i, seg in enumerate(obj["weak_segments"]):
        if not isinstance(seg, dict):
            raise ValueError(f"weak_segments[{i}] must be an object")
        # Patch defaults for commonly omitted numeric fields
        if "delta_vs_overall" not in seg:
            seg["delta_vs_overall"] = 0.0
        seg_missing = [k for k in weak_required if k not in seg]
        if seg_missing:
            raise ValueError(f"weak_segments[{i}] missing keys: {seg_missing}")
        if seg.get("severity") not in {"low", "medium", "high"}:
            raise ValueError(
                f"weak_segments[{i}].severity must be low/medium/high, got {seg.get('severity')!r}"
            )
        if seg.get("dimension") not in {"phase", "horizon", "location"}:
            raise ValueError(
                f"weak_segments[{i}].dimension must be phase/horizon/location, got {seg.get('dimension')!r}"
            )

    # hypotheses: list of dicts with required item keys
    if not isinstance(obj["hypotheses"], list):
        raise ValueError("Diagnosis 'hypotheses' must be a list")
    hyp_required = DIAGNOSIS_SCHEMA["hypotheses"]["item_keys"]
    for i, h in enumerate(obj["hypotheses"]):
        if not isinstance(h, dict):
            raise ValueError(f"hypotheses[{i}] must be an object")
        h_missing = [k for k in hyp_required if k not in h]
        if h_missing:
            raise ValueError(f"hypotheses[{i}] missing keys: {h_missing}")
        try:
            conf = float(h["confidence"])
        except (TypeError, ValueError):
            raise ValueError(f"hypotheses[{i}].confidence must be numeric")
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"hypotheses[{i}].confidence must be in [0,1], got {conf}")


# ============================================================================
# Milestone 2: Action Proposal (Agent 2, the Engineer)
# ============================================================================

ACTION_PROPOSAL_PROMPT = """\
{domain_context}

You are Agent 2 (the Engineer) in a two-agent improvement loop. Agent 1
(the Analyst) has already diagnosed the model's current behavior. Your job
is to choose ONE action from the catalog below that is most likely to
improve the target metric on the next iteration.

## Diagnosis from Agent 1

{diagnosis_block}

## Iteration History (most recent last)

{history_block}

## Action Catalog

You may ONLY emit an action whose `name` appears in this list. Each
action's params are constrained by the listed schema and guardrails.

{action_catalog_block}

## Current Config Values

These are the CURRENT settings. Do NOT propose a value that matches
the current value — that would be a no-op.

{current_config_block}

## Target Metric

You are optimizing **{target_metric}** (lower is better, except
coverage_95 which should be close to 0.95).

## Your Task

Choose ONE action from the catalog that best addresses the highest-severity
weak segment in the diagnosis. Consider:

- What did Agent 1 hypothesize about the cause?
- Have you already tried this action in a previous iteration? If a previous
  attempt failed or barely helped, try a different action or different
  parameter rather than repeating yourself.
- Stay within the guardrail ranges. The system will reject out-of-range
  values and force a retry.
- If you believe no further action is likely to improve the metric, emit
  the `stop` action.

Output a single JSON object (no surrounding prose, no markdown fences):

{{
  "name": "<one of: {action_names}>",
  "params": {{ ... action-specific params ... }},
  "rationale": "<one sentence: why this action and why now>",
  "expected_effect": "<one sentence: what you expect to happen to {target_metric}>"
}}

Output ONLY the JSON object. No code fences, no commentary.
"""


# Validation rules for Agent 2's output. Like the diagnosis schema, this is
# a hand-rolled validator — no jsonschema dependency.
REQUIRED_ACTION_KEYS: List[str] = ["name", "params", "rationale", "expected_effect"]


def format_action_proposal_prompt(
    diagnosis: Dict[str, Any],
    history: List[Dict[str, Any]],
    action_catalog: List[Dict[str, Any]],
    domain_context: str,
    target_metric: str = "wis",
    current_config: Dict[str, Any] = None,
) -> str:
    """Build Agent 2's action-proposal prompt.

    Args:
        diagnosis: A validated diagnosis dict from Agent 1.
        history: Compact list of prior iterations. Each entry should have
            shape {"iteration": int, "action": dict|None, "metrics": dict,
            "delta_vs_baseline": float|None}.
        action_catalog: Output of adapter.get_available_actions().
        domain_context: Output of adapter.get_domain_context().
        target_metric: Name of the metric being optimized (for the prompt).
        current_config: Optional pipeline config dict. If provided, key
            values are shown so the LLM avoids no-op proposals.

    Returns:
        Fully formatted prompt string.
    """
    # ---- Diagnosis block ----------------------------------------------------
    diag_lines = [
        f"**Summary**: {diagnosis['summary']}",
        f"**Suggested focus**: `{diagnosis['suggested_focus']}`",
        "",
        "**Headline metrics**:",
    ]
    for k, v in diagnosis.get("headline_metrics", {}).items():
        diag_lines.append(f"  - {k}: {v}")

    diag_lines.append("")
    diag_lines.append("**Weak segments** (worst first):")
    if diagnosis.get("weak_segments"):
        for seg in diagnosis["weak_segments"]:
            diag_lines.append(
                f"  - [{seg['severity']}] {seg['dimension']}={seg['value']}, "
                f"{seg['metric']} delta_vs_overall={seg['delta_vs_overall']:+.3f}"
            )
    else:
        diag_lines.append("  (none — model appears healthy)")

    diag_lines.append("")
    diag_lines.append("**Hypotheses**:")
    if diagnosis.get("hypotheses"):
        for h in diagnosis["hypotheses"]:
            diag_lines.append(
                f"  - [{h['id']} conf={h['confidence']:.2f}] {h['text']}"
            )
    else:
        diag_lines.append("  (none)")

    diagnosis_block = "\n".join(diag_lines)

    # ---- History block ------------------------------------------------------
    if not history:
        history_block = "(no prior iterations — this is the first action)"
    else:
        hist_lines = []
        for h in history:
            it = h.get("iteration", "?")
            action = h.get("action")
            metrics = h.get("metrics") or {}
            delta = h.get("delta_vs_baseline")
            target_val = metrics.get(target_metric)

            if action is None:
                action_str = "baseline (no action)"
            else:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in (action.get("params") or {}).items()
                )
                action_str = f"{action.get('name', '?')}({params_str})"

            if delta is None:
                delta_str = ""
            else:
                sign = "+" if delta > 0 else ""
                delta_str = f"  delta={sign}{delta:.3f}"

            metric_str = (
                f"{target_metric}={target_val}" if target_val is not None else ""
            )
            hist_lines.append(f"- [iter {it}] {action_str}  {metric_str}{delta_str}")
        history_block = "\n".join(hist_lines)

    # ---- Action catalog block ----------------------------------------------
    cat_lines = []
    for action in action_catalog:
        cat_lines.append(f"### `{action['name']}`")
        cat_lines.append(action.get("description", ""))
        params_schema = action.get("params_schema") or {}
        if params_schema:
            cat_lines.append("Params:")
            for p_name, p_spec in params_schema.items():
                p_type = p_spec.get("type", "?")
                if "enum" in p_spec:
                    p_desc = f"{p_type}, one of {p_spec['enum']}"
                else:
                    p_desc = p_type
                cat_lines.append(f"  - `{p_name}`: {p_desc}")
        else:
            cat_lines.append("Params: (none)")
        guardrails = action.get("guardrails") or {}
        if guardrails:
            cat_lines.append("Guardrails:")
            for g_name, g_val in guardrails.items():
                cat_lines.append(f"  - {g_name}: {g_val}")
        cat_lines.append("")
    action_catalog_block = "\n".join(cat_lines).rstrip()

    # ---- Current config block -------------------------------------------------
    if current_config:
        cfg_lines = []
        model_cfg = current_config.get("model") or {}
        family = model_cfg.get("family")
        is_bank_family = bool(family) and family != "xgboost_direct"
        if family:
            cfg_lines.append(f"Model family: {family}")
        if is_bank_family:
            model_params = model_cfg.get("params") or {}
            if model_params:
                cfg_lines.append("Model hyperparameters (overrides of the family defaults):")
                for k, v in sorted(model_params.items()):
                    cfg_lines.append(f"  - {k}: {v}")
            else:
                cfg_lines.append("Model hyperparameters: (family defaults)")
        xgb = current_config.get("xgboost", {})
        if xgb and not is_bank_family:
            cfg_lines.append("XGBoost hyperparameters:")
            for k, v in sorted(xgb.items()):
                cfg_lines.append(f"  - {k}: {v}")
        target = current_config.get("target", {})
        if target:
            cfg_lines.append(f"Target transform: {target.get('mode', '?')}")
        floor = current_config.get("floor", {})
        if floor:
            cfg_lines.append(f"Floor constraint: floor_pct={floor.get('floor_pct', '?')}")
        sw = current_config.get("sample_weights", {})
        active_weights = {k: v for k, v in sw.items() if v} if sw else {}
        if active_weights:
            cfg_lines.append(f"Sample weights: {active_weights}")
        else:
            cfg_lines.append("Sample weights: (none active)")
        fg = current_config.get("features", {}).get("groups_enabled", {})
        disabled = [g for g, v in fg.items() if not v] if fg else []
        if disabled:
            cfg_lines.append(f"Disabled feature groups: {disabled}")
        current_config_block = "\n".join(cfg_lines)
    else:
        current_config_block = "(not available)"

    # The schema example lists exactly the catalog's names so a bank family
    # (whose catalog is generated from its param_space) is never told about
    # legacy-only actions it cannot take.
    action_names = " | ".join(a["name"] for a in action_catalog)

    return ACTION_PROPOSAL_PROMPT.format(
        action_names=action_names,
        domain_context=domain_context,
        diagnosis_block=diagnosis_block,
        history_block=history_block,
        action_catalog_block=action_catalog_block,
        current_config_block=current_config_block,
        target_metric=target_metric,
    )


def validate_action_proposal(
    obj: Any,
    action_catalog: List[Dict[str, Any]],
) -> None:
    """Validate Agent 2's output against the schema + the action catalog.

    Performs only structural validation (required keys, action name in
    catalog, params is a dict). The deeper guardrail check happens
    inside `adapter.apply_action`, which is the single source of truth
    for what params are legal. This split keeps the validator lightweight
    and avoids duplicating guardrail logic in two places.

    Raises ValueError on the first violation.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Action proposal must be a JSON object, got {type(obj).__name__}")

    missing = [k for k in REQUIRED_ACTION_KEYS if k not in obj]
    if missing:
        raise ValueError(f"Action proposal missing required keys: {missing}")

    name = obj["name"]
    if not isinstance(name, str):
        raise ValueError(f"Action 'name' must be a string, got {type(name).__name__}")

    allowed_names = {a["name"] for a in action_catalog}
    if name not in allowed_names:
        raise ValueError(
            f"Action name {name!r} not in catalog. "
            f"Allowed: {sorted(allowed_names)}"
        )

    if not isinstance(obj["params"], dict):
        raise ValueError("Action 'params' must be an object (dict)")

    if not isinstance(obj["rationale"], str) or not obj["rationale"].strip():
        raise ValueError("Action 'rationale' must be a non-empty string")

    if not isinstance(obj["expected_effect"], str) or not obj["expected_effect"].strip():
        raise ValueError("Action 'expected_effect' must be a non-empty string")
