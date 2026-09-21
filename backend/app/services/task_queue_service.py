"""Background execution for pipeline steps (Phase 5).

Replaces FastAPI `BackgroundTasks` for the orchestrator with a small, observable executor:

* a **bounded** worker pool (`TASK_WORKERS`, default 4) so a burst of runs can't spawn unbounded
  threads or exhaust DB connections;
* a **per-key lock** (the pipeline run id): two resumes of the same run (e.g. a double-clicked
  approve button) execute one after the other instead of concurrently, which would otherwise race
  the LangGraph checkpoint and insert duplicate decisions;
* **failure capture**: an exception in a task is logged and counted instead of vanishing;
* `TASK_BACKEND=inline` runs tasks synchronously (used by tests and handy for debugging).

Scope note: this is an in-process queue. It bounds and serializes work on ONE server; it does
not survive a process restart or spread work across machines. A distributed broker (Celery/RQ +
Redis) can be dropped in behind `submit()` without touching the routers.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

logger = logging.getLogger(__name__)

BACKEND = os.environ.get("TASK_BACKEND", "thread")  # "thread" | "inline"
MAX_WORKERS = max(1, int(os.environ.get("TASK_WORKERS", "4")))

_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="pipeline")
_key_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_guard = threading.Lock()
_stats = {"submitted": 0, "running": 0, "completed": 0, "failed": 0}


def _bump(**deltas: int) -> None:
    with _guard:
        for name, delta in deltas.items():
            _stats[name] += delta


def _lock_for(key: str | None):
    if key is None:
        return threading.Lock()  # private lock: no serialization requested
    with _guard:
        return _key_locks[key]


def submit(fn: Callable, *args, key: str | None = None) -> None:
    """Run `fn(*args)` in the background; tasks sharing a `key` never overlap."""
    _bump(submitted=1)

    def runner() -> None:
        with _lock_for(key):
            _bump(running=1)
            try:
                fn(*args)
                _bump(completed=1)
            except Exception:  # noqa: BLE001
                _bump(failed=1)
                logger.exception("Background task %s%s failed", getattr(fn, "__name__", fn), args)
            finally:
                _bump(running=-1)

    if BACKEND == "inline":
        runner()
    else:
        _pool.submit(runner)


def stats() -> dict:
    with _guard:
        return {**_stats, "backend": BACKEND, "workers": MAX_WORKERS}
