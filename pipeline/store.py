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
    """Index for the UI: newest first, tolerant of malformed files.

    Reading every run to count its contents is the only way to show what a run actually
    holds, so callers that render this on every interaction should cache it against
    `runs_signature()`.

    A file that cannot be parsed is reported with `broken=True` rather than skipped, so a
    truncated or hand-edited run is visible instead of silently disappearing.
    """
    rows: list[dict] = []
    for p in sorted(
        RUNS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    ):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("not a JSON object")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            rows.append(
                {
                    "run_id": p.stem,
                    "broken": True,
                    "error": str(e),
                    "stage": "unreadable",
                    "schema": "",
                    "model": "",
                    "provider": "",
                    "products": 0,
                    "urls": 0,
                    "pages_ok": 0,
                    "pages_failed": 0,
                    "extractions": 0,
                    "results": 0,
                    "warnings": 0,
                    "created_at": "",
                    "updated_at": "",
                    "size_kb": p.stat().st_size / 1000,
                }
            )
            if len(rows) >= limit:
                break
            continue

        pages = raw.get("pages") or []
        inputs = raw.get("inputs") or []
        extractions = raw.get("extractions") or []
        rows.append(
            {
                "run_id": raw.get("run_id", p.stem),
                "broken": False,
                "error": "",
                "schema": raw.get("schema_name", ""),
                "stage": raw.get("stage", "?"),
                "model": raw.get("model", ""),
                "provider": raw.get("provider", ""),
                "products": len({i.get("product", "") for i in inputs}),
                "urls": len(inputs),
                "pages_ok": sum(1 for x in pages if x.get("success")),
                "pages_failed": sum(1 for x in pages if not x.get("success")),
                "extractions": len(extractions),
                "results": len(raw.get("results") or []),
                "warnings": len(raw.get("warnings") or []),
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
                "size_kb": p.stat().st_size / 1000,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def runs_signature() -> tuple:
    """Cheap fingerprint of the runs directory, for cache invalidation.

    Changes whenever a run is added, removed or rewritten, without parsing any JSON.
    """
    return tuple(
        sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in RUNS_DIR.glob("*.json")
        )
    )


def delete_run(run_id: str) -> bool:
    p = run_path(run_id)
    if p.exists():
        p.unlink()
        return True
    return False
