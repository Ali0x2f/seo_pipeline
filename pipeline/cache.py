"""Content-addressed disk cache.

Prompt tuning means running the pipeline many times over the same URLs. Without a
cache you re-fetch every page and re-pay for every token on each iteration. Keys are
derived from everything that could change the result, so a changed prompt or model
correctly misses the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from config import CACHE_DIR


def make_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:32]


class DiskCache:
    def __init__(self, namespace: str, enabled: bool = True) -> None:
        self.namespace = namespace
        self.enabled = enabled
        self.dir = CACHE_DIR / namespace
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            self.misses += 1
            return None
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.hits += 1
            return data
        except (json.JSONDecodeError, OSError):
            # Corrupt entry: drop it rather than crashing the run.
            try:
                p.unlink()
            except OSError:
                pass
            self.misses += 1
            return None

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        # Stages write from a thread pool, so the temp name carries the writer's id:
        # a shared one would let two threads clobber each other's partial file.
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)  # atomic, so a crash mid-write cannot corrupt the entry
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    def clear(self) -> int:
        if not self.dir.exists():
            return 0
        n = len(list(self.dir.glob("*.json")))
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        return n

    def size(self) -> tuple[int, int]:
        """(entry count, bytes on disk)"""
        if not self.dir.exists():
            return 0, 0
        files = list(self.dir.glob("*.json"))
        return len(files), sum(f.stat().st_size for f in files)


def clear_all() -> dict[str, int]:
    out: dict[str, int] = {}
    if not CACHE_DIR.exists():
        return out
    for d in CACHE_DIR.iterdir():
        if d.is_dir():
            out[d.name] = DiskCache(d.name).clear()
    return out


def total_size() -> tuple[int, int]:
    if not CACHE_DIR.exists():
        return 0, 0
    files = list(CACHE_DIR.rglob("*.json"))
    return len(files), sum(f.stat().st_size for f in files)
