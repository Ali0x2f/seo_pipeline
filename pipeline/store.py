"""Run persistence.

Streamlit keeps state in memory only, so a browser refresh or a rerun loop can throw
away an expensive run. Every stage writes the full RunState to disk, so a run can
always be reopened, inspected, or resumed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import RUNS_DIR
from pipeline.models import RunState


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def save_run(state: RunState) -> Path:
    state.touch()
    p = run_path(state.run_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def load_run(run_id: str) -> RunState:
    return RunState.model_validate_json(run_path(run_id).read_text(encoding="utf-8"))


def list_runs(limit: int = 50) -> list[dict]:
    """Lightweight index for the UI: newest first, tolerant of malformed files."""
    rows: list[dict] = []
    for p in sorted(RUNS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "run_id": raw.get("run_id", p.stem),
                "schema": raw.get("schema_name", ""),
                "stage": raw.get("stage", "?"),
                "model": raw.get("model", ""),
                "created_at": raw.get("created_at", ""),
                "products": len({i.get("product", "") for i in raw.get("inputs", [])}),
                "urls": len(raw.get("inputs", [])),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def delete_run(run_id: str) -> bool:
    p = run_path(run_id)
    if p.exists():
        p.unlink()
        return True
    return False
