"""Run persistence.

Streamlit keeps state in memory only, so a browser refresh or a rerun loop can throw
away an expensive run. Every stage writes the full RunState, so a run can always be
reopened, inspected, or resumed.

Two backends sit behind the same functions: JSON files under `runs/`, and a SQL database
(local SQLite file or a shared server) via `pipeline.db`. The backend is chosen in the UI
or by environment variable, so the rest of the pipeline never knows which one is active.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from config import (
    DATABASE_URL,
    DEFAULT_SQLITE_PATH,
    RUNS_DIR,
    STORAGE_BACKEND,
    STORAGE_SETTINGS_FILE,
)
from pipeline.models import RunState

BACKENDS = ("files", "db", "both")


# ---------------------------------------------------------------------- settings


@dataclass
class StorageSettings:
    backend: str = "files"
    db_url: str = ""

    @property
    def uses_db(self) -> bool:
        return self.backend in ("db", "both")

    @property
    def uses_files(self) -> bool:
        return self.backend in ("files", "both")

    def resolved_url(self) -> str:
        from pipeline.db import sqlite_url

        return self.db_url or sqlite_url(DEFAULT_SQLITE_PATH)


_settings: StorageSettings | None = None


def get_settings() -> StorageSettings:
    """Settings from disk, falling back to the environment on first use."""
    global _settings
    if _settings is None:
        backend = STORAGE_BACKEND if STORAGE_BACKEND in BACKENDS else "files"
        _settings = StorageSettings(backend=backend, db_url=DATABASE_URL)
        try:
            raw = json.loads(STORAGE_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            if raw.get("backend") in BACKENDS:
                _settings.backend = raw["backend"]
            if isinstance(raw.get("db_url"), str) and raw["db_url"]:
                _settings.db_url = raw["db_url"]
    return _settings


def set_settings(
    backend: str, db_url: str = "", persist: bool = True
) -> StorageSettings:
    global _settings
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}.")
    _settings = StorageSettings(backend=backend, db_url=db_url.strip())
    if persist:
        try:
            STORAGE_SETTINGS_FILE.write_text(
                json.dumps(asdict(_settings), indent=2), encoding="utf-8"
            )
        except OSError:
            pass
    return _settings


def get_db(settings: StorageSettings | None = None):
    """The configured RunDB, or None when only files are in use."""
    s = settings or get_settings()
    if not s.uses_db:
        return None
    from pipeline.db import RunDB

    return RunDB(s.resolved_url())


# -------------------------------------------------------------------- file store


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def save_run_file(state: RunState) -> Path:
    p = run_path(state.run_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def load_run_file(run_id: str) -> RunState:
    return RunState.model_validate_json(run_path(run_id).read_text(encoding="utf-8"))


def load_all_run_files() -> tuple[list[RunState], list[str]]:
    """Every readable run on disk, plus errors for the ones that could not be parsed."""
    states: list[RunState] = []
    errors: list[str] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            states.append(RunState.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001 - reported back to the caller
            errors.append(f"{p.name}: {e}")
    return states, errors


def list_runs_files(limit: int = 50) -> list[dict]:
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


def files_signature() -> tuple:
    """Cheap fingerprint of the runs directory, for cache invalidation.

    Changes whenever a run is added, removed or rewritten, without parsing any JSON.
    """
    return tuple(
        sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in RUNS_DIR.glob("*.json")
        )
    )


def delete_run_file(run_id: str) -> bool:
    p = run_path(run_id)
    if p.exists():
        p.unlink()
        return True
    return False


# --------------------------------------------------------------- backend router


def save_run(state: RunState) -> Path | None:
    """Persist a run to every configured backend.

    A database outage must not lose the run, so when files are also enabled the JSON copy
    is written first and a database error is only re-raised if the database is the sole
    backend.
    """
    state.touch()
    s = get_settings()
    path = save_run_file(state) if s.uses_files else None

    if s.uses_db:
        try:
            db = get_db(s)
            if db is not None:
                db.save(state)
        except Exception:
            if not s.uses_files:
                raise
            # The run is safe on disk; DB health is surfaced in the Storage tab.
    return path


def load_run(run_id: str) -> RunState:
    s = get_settings()
    if s.uses_db:
        try:
            db = get_db(s)
            if db is not None:
                return db.load(run_id)
        except Exception:
            if not s.uses_files:
                raise
    return load_run_file(run_id)


def list_runs(limit: int = 50) -> list[dict]:
    """Merged index across the active backends, newest first.

    Under `both`, a run held in each backend is listed once; the database copy wins
    because it is the one the flattened tables belong to.
    """
    s = get_settings()
    rows: list[dict] = []
    seen: set[str] = set()

    if s.uses_db:
        try:
            db = get_db(s)
            if db is not None:
                for r in db.list(limit=limit):
                    r["source"] = "db"
                    rows.append(r)
                    seen.add(r["run_id"])
        except Exception as e:  # noqa: BLE001 - the file list must still render
            rows.append(_unreachable_row(e))

    if s.uses_files:
        for r in list_runs_files(limit=limit):
            if r["run_id"] in seen:
                continue
            r["source"] = "file"
            rows.append(r)

    rows.sort(
        key=lambda r: (r.get("updated_at") or r.get("created_at") or ""), reverse=True
    )
    return rows[:limit]


def _unreachable_row(error: Exception) -> dict:
    return {
        "run_id": "(database unreachable)",
        "broken": True,
        "error": str(error),
        "source": "db",
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
        "size_kb": 0.0,
    }


def runs_signature() -> tuple:
    s = get_settings()
    parts: list = [s.backend, s.db_url]
    if s.uses_files:
        parts.append(files_signature())
    if s.uses_db:
        db = get_db(s)
        parts.append(db.signature() if db is not None else ())
    return tuple(parts)


def delete_run(run_id: str) -> bool:
    s = get_settings()
    deleted = False
    if s.uses_files:
        deleted = delete_run_file(run_id) or deleted
    if s.uses_db:
        try:
            db = get_db(s)
            if db is not None:
                deleted = db.delete(run_id) or deleted
        except Exception:
            if not s.uses_files:
                raise
    return deleted
