"""SQLite-backed run tracker for the agent improvement loop.

Stores the history of every `agent improve` run so we can:
  - Resume / inspect prior runs
  - Compare week-over-week performance
  - Recover the best forecast across all iterations of a run
  - Recreate any iteration's config exactly (config_json is persisted)

The tracker is intentionally framework-agnostic — it doesn't know about
LangGraph, LLMs, or XGBoost. The orchestrator passes plain dicts/JSON in.

Schema:

    runs
        run_id          TEXT PRIMARY KEY    e.g. 20260410-153022-a3f2
        started_at      TEXT (ISO8601)
        finished_at     TEXT (ISO8601, nullable)
        initial_forecast TEXT               path to seed forecast CSV
        cutoff_date     TEXT (nullable)
        target_metric   TEXT                e.g. "wis"
        status          TEXT                running | completed | failed | stopped
        notes           TEXT (nullable)

    iterations
        run_id          TEXT
        iteration       INTEGER             0 = baseline (no action applied yet)
        created_at      TEXT (ISO8601)
        metrics_json    TEXT                full metrics dict (JSON)
        diagnosis_json  TEXT (nullable)     Agent 1 output (JSON)
        action_json     TEXT (nullable)     Agent 2 output (JSON); null at iter 0
        config_json     TEXT (nullable)     Pipeline config snapshot (JSON)
        forecast_path   TEXT                path to forecast CSV for this iteration
        wis             REAL (nullable)     denormalized for fast queries
        mape            REAL (nullable)
        coverage_95     REAL (nullable)
        bias            REAL (nullable)
        PRIMARY KEY (run_id, iteration)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


DEFAULT_DB_PATH = Path("outputs/agent_runs/runs.db")


def _now_iso() -> str:
    """ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _generate_run_id() -> str:
    """Sortable, human-readable run id: YYYYMMDD-HHMMSS-<4hex>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{ts}-{suffix}"


def _extract_headline(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull denormalized columns out of a metrics dict.

    Resilient to missing keys — any unavailable metric is stored as NULL.
    Looks first at metrics["overall"], then at top level.
    """
    overall = metrics.get("overall", metrics) if isinstance(metrics, dict) else {}

    def _get(*keys: str) -> Optional[float]:
        for k in keys:
            if k in overall and overall[k] is not None:
                try:
                    return float(overall[k])
                except (TypeError, ValueError):
                    return None
        return None

    return {
        "wis": _get("wis", "WIS"),
        "mape": _get("mape", "MAPE"),
        "coverage_95": _get("coverage_95", "coverage95", "coverage"),
        "bias": _get("bias"),
    }


class RunTracker:
    """Persistent storage for agent improvement runs.

    Usage:
        tracker = RunTracker()  # uses outputs/agent_runs/runs.db
        run_id = tracker.start_run(initial_forecast="outputs/.../forecast.csv",
                                    target_metric="wis")
        tracker.log_iteration(run_id, iteration=0, metrics=baseline_metrics,
                              forecast_path="outputs/.../forecast.csv")
        # ... loop ...
        tracker.log_iteration(run_id, iteration=1, metrics=new_metrics,
                              diagnosis=diag_dict, action=action_dict,
                              config=config_dict, forecast_path=new_path)
        tracker.finish_run(run_id, status="completed")
    """

    def __init__(self, db_path: Union[str, Path, None] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ schema

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id           TEXT PRIMARY KEY,
                    started_at       TEXT NOT NULL,
                    finished_at      TEXT,
                    initial_forecast TEXT NOT NULL,
                    cutoff_date      TEXT,
                    target_metric    TEXT NOT NULL DEFAULT 'wis',
                    status           TEXT NOT NULL DEFAULT 'running',
                    notes            TEXT
                );

                CREATE TABLE IF NOT EXISTS iterations (
                    run_id          TEXT NOT NULL,
                    iteration       INTEGER NOT NULL,
                    created_at      TEXT NOT NULL,
                    metrics_json    TEXT NOT NULL,
                    diagnosis_json  TEXT,
                    action_json     TEXT,
                    config_json     TEXT,
                    forecast_path   TEXT,
                    wis             REAL,
                    mape            REAL,
                    coverage_95     REAL,
                    bias            REAL,
                    PRIMARY KEY (run_id, iteration),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_started_at
                    ON runs(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_iterations_run
                    ON iterations(run_id, iteration);
                """
            )

    # ----------------------------------------------------------------- writes

    def start_run(
        self,
        initial_forecast: Union[str, Path],
        cutoff_date: Optional[str] = None,
        target_metric: str = "wis",
        notes: Optional[str] = None,
    ) -> str:
        """Create a new run row and return its run_id."""
        run_id = _generate_run_id()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, initial_forecast,
                                  cutoff_date, target_metric, status, notes)
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    _now_iso(),
                    str(initial_forecast),
                    cutoff_date,
                    target_metric,
                    notes,
                ),
            )
        return run_id

    def log_iteration(
        self,
        run_id: str,
        iteration: int,
        metrics: Dict[str, Any],
        diagnosis: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        forecast_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Append one iteration's record. Iteration 0 should be the baseline."""
        headline = _extract_headline(metrics)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO iterations (
                    run_id, iteration, created_at,
                    metrics_json, diagnosis_json, action_json, config_json,
                    forecast_path, wis, mape, coverage_95, bias
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    iteration,
                    _now_iso(),
                    json.dumps(metrics, default=str),
                    json.dumps(diagnosis, default=str) if diagnosis is not None else None,
                    json.dumps(action, default=str) if action is not None else None,
                    json.dumps(config, default=str) if config is not None else None,
                    str(forecast_path) if forecast_path is not None else None,
                    headline["wis"],
                    headline["mape"],
                    headline["coverage_95"],
                    headline["bias"],
                ),
            )

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as finished. Valid statuses: completed, failed, stopped."""
        if status not in {"completed", "failed", "stopped"}:
            raise ValueError(f"Invalid run status: {status}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, _now_iso(), run_id),
            )

    # ------------------------------------------------------------------ reads

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a run plus all its iterations, or None if not found."""
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                return None

            iter_rows = conn.execute(
                """
                SELECT * FROM iterations
                WHERE run_id = ?
                ORDER BY iteration ASC
                """,
                (run_id,),
            ).fetchall()

        run = dict(run_row)
        run["iterations"] = [self._iteration_row_to_dict(r) for r in iter_rows]
        return run

    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent `limit` runs (without iterations)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT runs.*,
                       (SELECT COUNT(*) FROM iterations
                        WHERE iterations.run_id = runs.run_id) AS n_iterations,
                       (SELECT MIN(wis) FROM iterations
                        WHERE iterations.run_id = runs.run_id) AS best_wis
                FROM runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_best_iteration(
        self, run_id: str, metric: str = "wis"
    ) -> Optional[Dict[str, Any]]:
        """Return the iteration with the lowest value of `metric`.

        Lower is better for wis/mape/bias-magnitude. For coverage_95 we
        want closest to 0.95, not lowest.
        """
        if metric not in {"wis", "mape", "coverage_95", "bias"}:
            raise ValueError(f"Unsupported metric: {metric}")

        with self._connect() as conn:
            if metric == "coverage_95":
                # closest to 0.95
                row = conn.execute(
                    """
                    SELECT * FROM iterations
                    WHERE run_id = ? AND coverage_95 IS NOT NULL
                    ORDER BY ABS(coverage_95 - 0.95) ASC
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
            elif metric == "bias":
                # smallest absolute bias
                row = conn.execute(
                    """
                    SELECT * FROM iterations
                    WHERE run_id = ? AND bias IS NOT NULL
                    ORDER BY ABS(bias) ASC
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT * FROM iterations
                    WHERE run_id = ? AND {metric} IS NOT NULL
                    ORDER BY {metric} ASC
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()

        return self._iteration_row_to_dict(row) if row else None

    def compare_runs(
        self, run_id_a: str, run_id_b: str, metric: str = "wis"
    ) -> Dict[str, Any]:
        """Side-by-side comparison of two runs' best iterations."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)
        if run_a is None or run_b is None:
            missing = run_id_a if run_a is None else run_id_b
            raise ValueError(f"Run not found: {missing}")

        best_a = self.get_best_iteration(run_id_a, metric=metric)
        best_b = self.get_best_iteration(run_id_b, metric=metric)

        return {
            "metric": metric,
            "a": {
                "run_id": run_id_a,
                "started_at": run_a["started_at"],
                "n_iterations": len(run_a["iterations"]),
                "best": best_a,
            },
            "b": {
                "run_id": run_id_b,
                "started_at": run_b["started_at"],
                "n_iterations": len(run_b["iterations"]),
                "best": best_b,
            },
            "delta": self._compute_delta(best_a, best_b, metric),
        }

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _iteration_row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        for json_field in ("metrics_json", "diagnosis_json", "action_json", "config_json"):
            raw = d.pop(json_field)
            key = json_field.replace("_json", "")
            d[key] = json.loads(raw) if raw else None
        return d

    @staticmethod
    def _compute_delta(
        a: Optional[Dict[str, Any]],
        b: Optional[Dict[str, Any]],
        metric: str,
    ) -> Optional[Dict[str, Any]]:
        if a is None or b is None:
            return None
        va = a.get(metric)
        vb = b.get(metric)
        if va is None or vb is None:
            return None
        return {
            "absolute": vb - va,
            "relative_pct": ((vb - va) / va * 100.0) if va else None,
        }
