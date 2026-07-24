"""Background job execution with a pollable progress bus.

Streamlit runs the script top to bottom on one thread and widgets may only be touched
from that thread. The pipeline, meanwhile, does its slow work in worker threads (and an
asyncio loop for the browser). Writing progress straight from those threads is unsafe.

So workers only ever write to a lock-protected bus, and the Streamlit thread polls it
and re-renders. The job survives reruns by living in session state.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StageProgress:
    done: int = 0
    total: int = 0
    message: str = ""
    finished: bool = False

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.done / self.total))


class ProgressBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, StageProgress] = {}
        self._order: list[str] = []

    def update(self, stage: str, done: int, total: int, message: str = "") -> None:
        with self._lock:
            if stage not in self._stages:
                self._stages[stage] = StageProgress()
                self._order.append(stage)
            sp = self._stages[stage]
            sp.done, sp.total, sp.message = done, total, message
            if total and done >= total:
                sp.finished = True

    def snapshot(self) -> list[tuple[str, StageProgress]]:
        with self._lock:
            return [
                (
                    name,
                    StageProgress(
                        self._stages[name].done,
                        self._stages[name].total,
                        self._stages[name].message,
                        self._stages[name].finished,
                    ),
                )
                for name in self._order
            ]


@dataclass
class Job:
    bus: ProgressBus
    thread: threading.Thread
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    @property
    def done(self) -> bool:
        return bool(self.result.get("_done"))

    @property
    def error(self) -> str | None:
        return self.result.get("error")


def start_job(fn: Callable[..., Any], **kwargs) -> Job:
    """Run `fn(progress=bus.update, **kwargs)` on a daemon thread."""
    bus = ProgressBus()
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn(progress=bus.update, **kwargs)
        except BaseException as e:                       # noqa: BLE001 - reported to UI
            result["error"] = f"{type(e).__name__}: {e}"
            result["traceback"] = traceback.format_exc()
        finally:
            result["_done"] = True

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return Job(bus=bus, thread=t, result=result)
