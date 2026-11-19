#!/bin/bash
# Check out TimeSeriesScientist at the commit the comparison uses, apply the two-call-site
# fix its released graph needs to run, and build its own environment.
#   scripts/comparison/setup_tsci.sh [target-dir]      (default: ../TimeSeriesScientist)
# Needs git and uv on PATH. Safe to run again: an existing checkout is reused.
set -euo pipefail

TSCI_REPOSITORY=https://github.com/Y-Research-SBU/TimeSeriesScientist.git
TSCI_COMMIT=41b3963
TSCI_PYTHON_VERSION=3.12

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
target="${1:-$repo_root/../TimeSeriesScientist}"

if [ ! -d "$target/.git" ]; then
    git clone -q "$TSCI_REPOSITORY" "$target"
fi
cd "$target"
git checkout -q "$TSCI_COMMIT"
# Their repo tracks __pycache__ files, which change on every run and block a clean check.
git ls-files '*.pyc' | xargs -r git update-index --assume-unchanged
patch_file="$repo_root/scripts/comparison/tsci_minimal_fixes.patch"
if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "patch already applied"
else
    git apply "$patch_file"
fi

[ -d .venv ] || uv venv -q --python "$TSCI_PYTHON_VERSION" .venv
uv pip install -q --python .venv/bin/python -r "$repo_root/scripts/comparison/tsci-requirements.txt"
(cd time_series_agent && ../.venv/bin/python -c "from graph.agent_graph import TimeSeriesAgentGraph; print('TimeSeriesScientist ready at $target')")
