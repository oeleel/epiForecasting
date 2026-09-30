"""Curated YAML intake (design doc `knowledge-bank-design.md` section 4).

Curated entries are the lab's human-authored guardrails. They live as YAML
files in `knowledge/curated/*.yaml`, git-tracked and PR-reviewed, and the bank
rebuilds its curated rows from these files on open, so there is no compile
step and no stale-sync failure mode. This module is the only reader of that
directory: it turns the files into validated `KnowledgeEntry` records or
raises one error that lists everything wrong.

Rules
-----
- Every file is a YAML **list** of entries in the exact `KnowledgeEntry`
  shape. A file whose top level is not a list (including an empty file) is an
  error.
- Files whose name starts with `SKIP_PREFIX` ("_") are skipped: the template
  and intake stubs live there and are not knowledge yet.
- Only the exact `.yaml` suffix is knowledge. A `.yml` or `.YAML` file that
  is not `_`-prefixed raises (`MISNAMED_SUFFIXES`) instead of being silently
  ignored: an author who mistypes the suffix must hear about it.
- Every schema error is reported as `"<file>[<index>] <id or ?>: <reason>"`,
  all files are read before raising, and the single `ValueError` carries the
  full list. Authors fix everything in one pass.
- Duplicate ids across files are an error (the id is the primary key).
- A directory with no curated files, or files with no entries, raises: a bank
  with no curated knowledge is a misconfiguration, not a valid state.
- Paths are anchored on the repo root computed from `__file__`, never the
  current working directory, so the CLI and tests agree on where the bank is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agent.knowledge.schema import KnowledgeEntry

__all__ = [
    "CURATED_FILE_GLOB",
    "DEFAULT_CURATED_DIR",
    "MISNAMED_SUFFIXES",
    "REPO_ROOT",
    "SKIP_PREFIX",
    "load_curated_dir",
    "list_curated_files",
]

_log = logging.getLogger(__name__)

# agent/knowledge/curated.py -> parents[0]=knowledge, [1]=agent, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURATED_DIR = REPO_ROOT / "knowledge" / "curated"

CURATED_FILE_GLOB = "*.yaml"
SKIP_PREFIX = "_"
# Lower-cased suffixes that look like YAML but are not loaded; their presence is an error.
MISNAMED_SUFFIXES = (".yml", ".yaml")


def list_curated_files(curated_dir: str | Path = DEFAULT_CURATED_DIR) -> list[Path]:
    """Sorted `*.yaml` files in `curated_dir`, minus the `_`-prefixed ones.

    Raises when a non-skipped file carries a YAML-looking suffix other than
    the exact `.yaml` (e.g. `foo.yml`, `foo.YAML`), so a misnamed file is
    never silently left out of the bank.
    """
    directory = Path(curated_dir)
    if not directory.is_dir():
        raise ValueError(f"curated dir does not exist or is not a directory: {directory}")
    misnamed = sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file()
        and not p.name.startswith(SKIP_PREFIX)
        and p.suffix.lower() in MISNAMED_SUFFIXES
        and p.suffix != ".yaml"
    )
    if misnamed:
        raise ValueError(
            f"curated dir {directory} has YAML-looking files that are not loaded "
            f"(only {CURATED_FILE_GLOB} is read): {misnamed}; rename them to .yaml"
        )
    return sorted(
        p for p in directory.glob(CURATED_FILE_GLOB) if not p.name.startswith(SKIP_PREFIX)
    )


def _read_yaml_list(path: Path, errors: list[str]) -> list[Any]:
    """Parse one file; append to `errors` and return [] when it is unusable."""
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        errors.append(f"{path.name}: YAML parse error: {exc}")
        return []
    if not isinstance(loaded, list):
        got = "empty file" if loaded is None else type(loaded).__name__
        errors.append(f"{path.name}: top level must be a list of entries, got {got}")
        return []
    return loaded


def load_curated_dir(curated_dir: str | Path = DEFAULT_CURATED_DIR) -> list[KnowledgeEntry]:
    """Load every curated entry under `curated_dir`, or raise listing every problem."""
    files = list_curated_files(curated_dir)
    if not files:
        raise ValueError(
            f"no curated knowledge files ({CURATED_FILE_GLOB}, not starting with "
            f"{SKIP_PREFIX!r}) in {Path(curated_dir)}"
        )

    entries: list[KnowledgeEntry] = []
    errors: list[str] = []
    first_seen: dict[str, str] = {}  # id -> "<file>[<index>]"
    for path in files:
        for index, raw in enumerate(_read_yaml_list(path, errors)):
            location = f"{path.name}[{index}]"
            raw_id = raw.get("id", "?") if isinstance(raw, dict) else "?"
            try:
                entry = KnowledgeEntry.from_dict(raw)
            except ValueError as exc:
                errors.append(f"{location} {raw_id}: {exc}")
                continue
            if entry.id in first_seen:
                errors.append(
                    f"{location} {entry.id}: duplicate id (first seen in {first_seen[entry.id]})"
                )
                continue
            first_seen[entry.id] = location
            entries.append(entry)

    if errors:
        raise ValueError(
            f"{len(errors)} curated knowledge error(s) in {Path(curated_dir)}:\n  "
            + "\n  ".join(errors)
        )
    if not entries:
        raise ValueError(f"curated knowledge files in {Path(curated_dir)} contain no entries")
    _log.info(
        "Loaded %d curated entries from %d file(s) in %s", len(entries), len(files), curated_dir
    )
    return entries
