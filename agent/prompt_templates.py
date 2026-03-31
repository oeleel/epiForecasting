"""Prompt templates for the agentic forecasting framework.

Templates are parameterized strings — the framework fills in metrics
computed by the adapter before sending to the LLM.
"""


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
