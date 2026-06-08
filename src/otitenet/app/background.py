"""Small background task helpers for the Streamlit app.

Streamlit reruns the script for ordinary widget changes. Expensive work that is
safe to compute ahead of time can use this module so tab changes and parameter
edits do not need to synchronously rebuild every artifact.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, Hashable


_MAX_WORKERS = 4
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="otitenet-bg")
_lock = Lock()
_futures: dict[Hashable, Future] = {}


def submit_once(key: Hashable, fn: Callable[..., Any], *args, **kwargs) -> Future:
    """Submit ``fn`` once for ``key`` and return the existing future on repeats."""
    with _lock:
        future = _futures.get(key)
        if future is not None and not future.cancelled():
            return future
        future = _executor.submit(fn, *args, **kwargs)
        _futures[key] = future
        return future


def get_completed_result(key: Hashable, default: Any = None) -> Any:
    """Return a completed background result, or ``default`` without blocking."""
    with _lock:
        future = _futures.get(key)
    if future is None or not future.done() or future.cancelled():
        return default
    try:
        return future.result()
    except Exception:
        return default


def is_running(key: Hashable) -> bool:
    """Return True when a submitted task is still running."""
    with _lock:
        future = _futures.get(key)
    return bool(future is not None and not future.done() and not future.cancelled())


def clear_completed() -> None:
    """Drop completed futures from the registry while leaving running work alone."""
    with _lock:
        for key, future in list(_futures.items()):
            if future.done() or future.cancelled():
                _futures.pop(key, None)


def _clear_for_tests() -> None:
    with _lock:
        _futures.clear()
