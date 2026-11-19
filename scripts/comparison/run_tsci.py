"""Drive TimeSeriesScientist (TSci) on one univariate series and save its raw result.

TSci's own ``main.py`` has no command-line flags and a hard-coded data path, so this
wrapper builds the config and calls its graph directly. Run it with TSci's interpreter,
not this repo's: the two environments pin different LangChain versions.

    OPENAI_API_KEY=ollama OPENAI_API_BASE=http://localhost:11434/v1 \\
    <tsci>/.venv/bin/python scripts/comparison/run_tsci.py \\
        --tsci-root <tsci> --data-path <tsci>/dataset/ETTh1.csv \\
        --input-length 512 --horizon 96 --llm-model qwen3:8b --out <dir>

TSci takes the LAST ``input_length + horizon`` rows only when the file has exactly that
many rows; with ``num_slices=1`` it takes the FIRST ones. Pass a file already cut to the
window you want (``prepare_flu_series.py`` does that).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tsci-root", type=Path, required=True, help="TimeSeriesScientist checkout")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV with a date and a value column")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--value-column", default="OT")
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=None, help="restrict TSci's model list (default: all 21)")
    parser.add_argument("--k-models", type=int, default=None, help="models kept after validation (default: TSci's 3)")
    parser.add_argument("--n-candidates", type=int, default=None, help="models the LLM proposes (default: TSci's 5)")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="also forecast with this TSci model at its library defaults (no LLM), e.g. XGBoost",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.tsci_root.resolve() / "time_series_agent"
    if not package_dir.is_dir():
        raise SystemExit(f"not a TimeSeriesScientist checkout: {args.tsci_root}")
    sys.path.insert(0, str(package_dir))

    from config.default_config import DEFAULT_CONFIG
    from graph.agent_graph import TimeSeriesAgentGraph
    from utils.model_library import get_model_function

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # deepcopy: DEFAULT_CONFIG nests the shared MODEL_CONFIG dict, which is edited below.
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        num_slices=1,
        input_length=args.input_length,
        horizon=args.horizon,
        data_path=str(args.data_path.resolve()),
        date_column=args.date_column,
        value_column=args.value_column,
        llm_model=args.llm_model,
        output_dir=str(out_dir / "tsci_results"),
    )
    # The validation agent reads these keys from the top level, the graph from "models".
    overrides = {"available_models": args.models, "k_models": args.k_models, "n_candidates": args.n_candidates}
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
            config["models"][key] = value

    if args.baseline_model is not None:
        # TSci's own model function with empty hyperparameters, on the raw input window:
        # what the model does before any agent touches the data or the settings.
        series = pd.read_csv(args.data_path)[args.value_column].to_numpy(dtype=float)
        window, truth = series[: args.input_length], series[args.input_length : args.input_length + args.horizon]
        forecast = get_model_function(args.baseline_model)({"value": window}, {}, args.horizon)
        baseline = {
            "model": args.baseline_model,
            "hyperparameters": {},
            "predictions": [float(value) for value in forecast],
            "actuals": [float(value) for value in truth],
        }
        (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

    start = time.time()
    results = TimeSeriesAgentGraph(config=config, model=config["llm_model"], debug=False).run()
    elapsed_seconds = round(time.time() - start, 1)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    run_record = {"elapsed_seconds": elapsed_seconds, "args": {k: str(v) for k, v in vars(args).items()}}
    (out_dir / "run.json").write_text(json.dumps(run_record, indent=2))
    print(f"elapsed_seconds={elapsed_seconds}")
    print(f"results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
