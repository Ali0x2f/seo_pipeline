"""SQL persistence for runs.

JSON files are fine for one machine, but they cannot be queried, shared between
instances, or survive a container with ephemeral disk. Any SQLAlchemy URL is accepted
here, so the same code covers a local SQLite file and a shared Postgres/MySQL server.

Every run is stored twice: `runs.payload` holds the exact RunState JSON so loading is
always lossless even if the model gains fields, and the child tables hold the same data
flattened so it can be read with plain SQL by anything else (BI tools, notebooks, a
teammate's psql session).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import DEFAULT_SQLITE_PATH
from pipeline.models import RunState

# MySQL's TEXT tops out at 64 kB, which a single scraped page can exceed.
LongText = Text().with_variant(mysql.LONGTEXT(), "mysql")

metadata = MetaData()

runs = Table(
    "runs",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("schema_name", String(255), default=""),
    Column("stage", String(32), default=""),
    Column("provider", String(64), default=""),
    Column("model", String(128), default=""),
    Column("created_at", String(64), default=""),
    Column("updated_at", String(64), default="", index=True),
    Column("products", Integer, default=0),
    Column("urls", Integer, default=0),
    Column("pages_ok", Integer, default=0),
    Column("pages_failed", Integer, default=0),
    Column("extractions", Integer, default=0),
    Column("results", Integer, default=0),
    Column("warnings", Integer, default=0),
    Column("size_bytes", Integer, default=0),
    Column("payload", LongText, nullable=False),
)

run_inputs = Table(
    "run_inputs",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("product", String(255), default=""),
    Column("node", String(255), default=""),
    Column("url", Text, default=""),
)

run_pages = Table(
    "run_pages",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("product", String(255), default=""),
    Column("node", String(255), default=""),
    Column("url", Text, default=""),
    Column("success", Boolean, default=False),
    Column("title", Text, default=""),
    Column("text", LongText, default=""),
    Column("char_count", Integer, default=0),
    Column("fetch_method", String(64), default=""),
    Column("truncated", Boolean, default=False),
    Column("error", Text, default=""),
    Column("fetched_at", String(64), default=""),
)

run_extractions = Table(
    "run_extractions",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("product", String(255), default=""),
    Column("node", String(255), default=""),
    Column("url", Text, default=""),
    Column("model", String(128), default=""),
    Column("error", Text, default=""),
    Column("from_cache", Boolean, default=False),
    Column("prompt_tokens", Integer, default=0),
    Column("completion_tokens", Integer, default=0),
)

run_claims = Table(
    "run_claims",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("extraction_idx", Integer, primary_key=True),
    Column("field_key", String(128), primary_key=True),
    Column("product", String(255), default=""),
    Column("url", Text, default=""),
    Column("found", Boolean, default=False),
    Column("value_text", LongText, default=""),
    Column("values_json", JSON, default=list),
    Column("quote", LongText, default=""),
    Column("quote_verified", Boolean, default=False),
)

run_results = Table(
    "run_results",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("product", String(255), primary_key=True),
    Column("pages_used", JSON, default=list),
    Column("pages_failed", JSON, default=list),
)

run_fields = Table(
    "run_fields",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("product", String(255), primary_key=True),
    Column("field_key", String(128), primary_key=True),
    Column("value", LongText, default=""),
    Column("items", JSON, default=list),
    Column("sources", JSON, default=list),
    Column("source_count", Integer, default=0),
    Column("conflict", Boolean, default=False),
    Column("conflict_note", Text, default=""),
    Column("method", String(32), default=""),
)

run_warnings = Table(
    "run_warnings",
    metadata,
    Column("run_id", String(128), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("message", Text, default=""),
)

# Order matters: children are cleared before the parent row is replaced.
CHILD_TABLES = (
    run_inputs,
    run_pages,
    run_extractions,
    run_claims,
    run_results,
    run_fields,
    run_warnings,
)


def sqlite_url(path: str | Path) -> str:
    """SQLAlchemy URL for a SQLite file, with Windows paths made URL-safe."""
    return "sqlite:///" + Path(path).resolve().as_posix()


DEFAULT_SQLITE_PATH_URL = sqlite_url(DEFAULT_SQLITE_PATH)


def sqlite_path(url: str) -> Path | None:
    """Filesystem path behind a SQLite URL, or None for other backends."""
    if not url.startswith("sqlite"):
        return None
    raw = url.split(":///", 1)[-1] if ":///" in url else ""
    if not raw or raw == ":memory:":
        return None
    return Path(raw)


def redact(url: str) -> str:
    """Connection string with any password removed, safe to show in the UI or logs."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.password:
        return url
    return url.replace(f":{parsed.password}@", ":***@")


_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def _make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # Pipeline stages save from worker threads, and a stage can take minutes, so the
        # writer lock has to be waited on rather than failing instantly.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):  # pragma: no cover - driver callback
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    return engine


def get_engine(url: str) -> Engine:
    """Engines are pooled per URL; creating one per save would leak connections."""
    with _engines_lock:
        engine = _engines.get(url)
        if engine is None:
            engine = _make_engine(url)
            _engines[url] = engine
        return engine


def dispose_engine(url: str) -> None:
    with _engines_lock:
        engine = _engines.pop(url, None)
    if engine is not None:
        engine.dispose()


class RunDB:
    """Run storage backed by any SQLAlchemy-supported database."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.engine = get_engine(url)

    # ---------------------------------------------------------------- schema

    def ensure_schema(self) -> None:
        path = sqlite_path(self.url)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        metadata.create_all(self.engine)

    def check(self) -> tuple[bool, str]:
        """Connect and create tables, reporting the failure instead of raising."""
        try:
            self.ensure_schema()
            with self.engine.connect() as conn:
                n = conn.execute(select(func.count()).select_from(runs)).scalar_one()
            return True, f"Connected — {n} run(s) stored."
        except SQLAlchemyError as e:
            return False, str(getattr(e, "orig", None) or e)
        except Exception as e:  # noqa: BLE001 - surfaced in the UI
            return False, f"{type(e).__name__}: {e}"

    # ----------------------------------------------------------------- write

    def save(self, state: RunState) -> None:
        """Replace the stored copy of one run, in a single transaction."""
        self.ensure_schema()
        payload = state.model_dump_json()
        summary = {
            "run_id": state.run_id,
            "schema_name": state.schema_name,
            "stage": state.stage,
            "provider": state.provider,
            "model": state.model,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "products": len({i.product for i in state.inputs}),
            "urls": len(state.inputs),
            "pages_ok": sum(1 for p in state.pages if p.success),
            "pages_failed": sum(1 for p in state.pages if not p.success),
            "extractions": len(state.extractions),
            "results": len(state.results),
            "warnings": len(state.warnings),
            "size_bytes": len(payload.encode("utf-8")),
            "payload": payload,
        }

        with self.engine.begin() as conn:
            for table in CHILD_TABLES:
                conn.execute(delete(table).where(table.c.run_id == state.run_id))
            conn.execute(delete(runs).where(runs.c.run_id == state.run_id))
            conn.execute(runs.insert(), summary)

            self._insert(
                conn,
                run_inputs,
                [
                    {
                        "run_id": state.run_id,
                        "idx": i,
                        "product": row.product,
                        "node": row.node,
                        "url": row.url,
                    }
                    for i, row in enumerate(state.inputs)
                ],
            )

            self._insert(
                conn,
                run_pages,
                [
                    {
                        "run_id": state.run_id,
                        "idx": i,
                        "product": p.product,
                        "node": p.node,
                        "url": p.url,
                        "success": p.success,
                        "title": p.title,
                        "text": p.text,
                        "char_count": p.char_count,
                        "fetch_method": p.fetch_method,
                        "truncated": p.truncated,
                        "error": p.error or "",
                        "fetched_at": p.fetched_at,
                    }
                    for i, p in enumerate(state.pages)
                ],
            )

            ex_rows: list[dict] = []
            claim_rows: list[dict] = []
            for i, ex in enumerate(state.extractions):
                ex_rows.append(
                    {
                        "run_id": state.run_id,
                        "idx": i,
                        "product": ex.product,
                        "node": ex.node,
                        "url": ex.url,
                        "model": ex.model,
                        "error": ex.error or "",
                        "from_cache": ex.from_cache,
                        "prompt_tokens": ex.prompt_tokens,
                        "completion_tokens": ex.completion_tokens,
                    }
                )
                for key, claim in ex.claims.items():
                    claim_rows.append(
                        {
                            "run_id": state.run_id,
                            "extraction_idx": i,
                            "field_key": key,
                            "product": ex.product,
                            "url": ex.url,
                            "found": claim.found,
                            "value_text": claim.as_text,
                            "values_json": list(claim.values),
                            "quote": claim.quote,
                            "quote_verified": claim.quote_verified,
                        }
                    )
            self._insert(conn, run_extractions, ex_rows)
            self._insert(conn, run_claims, claim_rows)

            result_rows: list[dict] = []
            field_rows: list[dict] = []
            for res in state.results:
                result_rows.append(
                    {
                        "run_id": state.run_id,
                        "product": res.product,
                        "pages_used": list(res.pages_used),
                        "pages_failed": list(res.pages_failed),
                    }
                )
                for key, f in res.fields.items():
                    field_rows.append(
                        {
                            "run_id": state.run_id,
                            "product": res.product,
                            "field_key": key,
                            "value": f.value,
                            "items": list(f.items),
                            "sources": list(f.sources),
                            "source_count": f.source_count,
                            "conflict": f.conflict,
                            "conflict_note": f.conflict_note,
                            "method": f.method,
                        }
                    )
            self._insert(conn, run_results, result_rows)
            self._insert(conn, run_fields, field_rows)

            self._insert(
                conn,
                run_warnings,
                [
                    {"run_id": state.run_id, "idx": i, "message": w}
                    for i, w in enumerate(state.warnings)
                ],
            )

    @staticmethod
    def _insert(conn, table: Table, rows: list[dict]) -> None:
        if rows:
            conn.execute(table.insert(), rows)

    # ------------------------------------------------------------------ read

    def load(self, run_id: str) -> RunState:
        self.ensure_schema()
        with self.engine.connect() as conn:
            payload = conn.execute(
                select(runs.c.payload).where(runs.c.run_id == run_id)
            ).scalar_one_or_none()
        if payload is None:
            raise FileNotFoundError(f"Run {run_id!r} is not in the database.")
        return RunState.model_validate_json(payload)

    def list(self, limit: int = 50) -> list[dict]:
        """Index rows shaped exactly like the file store's, so the UI is unchanged."""
        self.ensure_schema()
        cols = [c for c in runs.c if c.name != "payload"]
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(*cols)
                    .order_by(runs.c.updated_at.desc(), runs.c.created_at.desc())
                    .limit(limit)
                )
                .mappings()
                .all()
            )

        out: list[dict] = []
        for r in rows:
            out.append(
                {
                    "run_id": r["run_id"],
                    "broken": False,
                    "error": "",
                    "schema": r["schema_name"] or "",
                    "stage": r["stage"] or "?",
                    "model": r["model"] or "",
                    "provider": r["provider"] or "",
                    "products": r["products"] or 0,
                    "urls": r["urls"] or 0,
                    "pages_ok": r["pages_ok"] or 0,
                    "pages_failed": r["pages_failed"] or 0,
                    "extractions": r["extractions"] or 0,
                    "results": r["results"] or 0,
                    "warnings": r["warnings"] or 0,
                    "created_at": r["created_at"] or "",
                    "updated_at": r["updated_at"] or "",
                    "size_kb": (r["size_bytes"] or 0) / 1000,
                }
            )
        return out

    def run_ids(self) -> list[str]:
        self.ensure_schema()
        with self.engine.connect() as conn:
            return [
                r[0]
                for r in conn.execute(
                    select(runs.c.run_id).order_by(runs.c.updated_at.desc())
                )
            ]

    def signature(self) -> tuple:
        """Cheap fingerprint for cache invalidation, without reading any payload."""
        try:
            self.ensure_schema()
            with self.engine.connect() as conn:
                row = conn.execute(
                    select(
                        func.count(),
                        func.max(runs.c.updated_at),
                        func.sum(runs.c.size_bytes),
                    ).select_from(runs)
                ).one()
            return ("db", self.url, row[0], row[1] or "", int(row[2] or 0))
        except SQLAlchemyError as e:
            # A signature must never break rendering; an unreachable DB just looks empty.
            return ("db-error", self.url, str(e)[:200])

    def stats(self) -> dict[str, int]:
        self.ensure_schema()
        counts: dict[str, int] = {}
        with self.engine.connect() as conn:
            for table in (runs, *CHILD_TABLES):
                counts[table.name] = conn.execute(
                    select(func.count()).select_from(table)
                ).scalar_one()
        return counts

    # ---------------------------------------------------------------- delete

    def delete(self, run_id: str) -> bool:
        self.ensure_schema()
        with self.engine.begin() as conn:
            for table in CHILD_TABLES:
                conn.execute(delete(table).where(table.c.run_id == run_id))
            res = conn.execute(delete(runs).where(runs.c.run_id == run_id))
        return bool(res.rowcount)

    # ----------------------------------------------------------------- bulk

    def import_states(
        self, states: Iterable[RunState], overwrite: bool = False
    ) -> dict:
        existing = set(self.run_ids())
        report = {"imported": 0, "skipped": 0, "failed": 0, "errors": []}
        for state in states:
            if state.run_id in existing and not overwrite:
                report["skipped"] += 1
                continue
            try:
                self.save(state)
                report["imported"] += 1
            except Exception as e:  # noqa: BLE001 - collected and shown to the user
                report["failed"] += 1
                report["errors"].append(f"{state.run_id}: {e}")
        return report

    def export_states(self, run_ids: list[str] | None = None) -> list[RunState]:
        ids = run_ids if run_ids is not None else self.run_ids()
        return [self.load(rid) for rid in ids]

    def dump_json(self, run_ids: list[str] | None = None) -> str:
        """Every run as one JSON document, for backup or moving to another database."""
        self.ensure_schema()
        ids = run_ids if run_ids is not None else self.run_ids()
        with self.engine.connect() as conn:
            payloads = [
                json.loads(
                    conn.execute(
                        select(runs.c.payload).where(runs.c.run_id == rid)
                    ).scalar_one()
                )
                for rid in ids
            ]
        return json.dumps({"runs": payloads}, ensure_ascii=False, indent=2)

    def file_bytes(self) -> bytes | None:
        """Raw SQLite file, so the whole database can be downloaded from the UI."""
        path = sqlite_path(self.url)
        if path is None or not path.exists():
            return None
        with self.engine.connect() as conn:
            # WAL contents live outside the main file until a checkpoint.
            conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        return path.read_bytes()
