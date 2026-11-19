"""Run TimeSeriesScientist on every series in a ``prepare_flu_series.py`` manifest.

One ``run_tsci.py`` subprocess per (location, cutoff), in TSci's own interpreter. A run
whose ``results.json`` already exists is skipped, so an interrupted batch resumes. A run
that fails is logged and the batch continues; the exit code is 1 if any run failed.

    .venv/bin/python scripts/comparison/run_tsci_batch.py \\
        --tsci-root ../TimeSeriesScientist --series outputs/experiments/tsci_comparison/series \\
        --out outputs/experiments/tsci_comparison/runs --models XGBoost --llm-model qwen3:8b

The LLM endpoint comes from the environment (OPENAI_API_BASE, OPENAI_API_KEY), which
``scripts/slurm/llm.sbatch`` sets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TSCI = REPO_ROOT / "scripts" / "comparison" / "run_tsci.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tsci-root", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True, help="directory with manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="TSci model list, e.g. XGBoost")
    parser.add_argument("--llm-model", required=True)
    args = parser.parse_args()

    tsci_root = args.tsci_root.resolve()
    tsci_python = tsci_root / ".venv" / "bin" / "python"
    if not tsci_python.exists():
        raise SystemExit(f"no TSci environment at {tsci_python}; run scripts/comparison/setup_tsci.sh")
    manifest = json.loads((args.series / "manifest.json").read_text())

    failed = []
    for entry in manifest["files"]:
        name = Path(entry["file"]).stem
        run_dir = (args.out / name).resolve()
        if (run_dir / "results.json").exists():
            print(f"skip {name}: done", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(tsci_python), str(RUN_TSCI),
            "--tsci-root", str(tsci_root),
            "--data-path", str((args.series / entry["file"]).resolve()),
            "--input-length", str(entry["input_length"]),
            "--horizon", str(entry["horizon"]),
            "--models", *args.models,
            # One model in the list: the LLM tunes it, there is nothing to select or ensemble.
            "--k-models", str(len(args.models)),
            "--n-candidates", str(len(args.models)),
            "--baseline-model", args.models[0],
            "--llm-model", args.llm_model,
            "--out", str(run_dir),
        ]
        print(f"run  {name}", flush=True)
        with (run_dir / "run.log").open("w") as log:
            # cwd: TSci writes relative paths (logs, cache) wherever it is started.
            completed = subprocess.run(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            failed.append(name)
            print(f"FAIL {name}: exit {completed.returncode}, see {run_dir / 'run.log'}", flush=True)

    print(f"done: {len(manifest['files']) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
