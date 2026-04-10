"""Run every test module in tests/agent/ in a single process.

This is the no-pytest fallback test runner. Each test_*.py file
exposes a `main()` that exits 0 on success / 1 on any failure. We
import each, dispatch their ALL list, and tally totals.

Usage:
    PYTHONPATH=. python tests/agent/run_all.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULES = [
    "tests.agent.test_run_tracker",
    "tests.agent.test_config",
    "tests.agent.test_sample_weights",
    "tests.agent.test_adapter_actions",
    "tests.agent.test_prompts",
    "tests.agent.test_wis",
    "tests.agent.test_orchestrator",
]


def main():
    total = 0
    failed = 0
    for mod_name in MODULES:
        print(f"\n=== {mod_name} ===")
        mod = importlib.import_module(mod_name)
        for fn in getattr(mod, "ALL_TESTS", getattr(mod, "ALL", [])):
            total += 1
            try:
                fn()
                print(f"  PASS  {fn.__name__}")
            except Exception as e:
                print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
                failed += 1
    print()
    print("=" * 50)
    print(f"  {total - failed}/{total} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
