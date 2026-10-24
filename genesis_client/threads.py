"""Named threads — Python ``name`` plus kernel ``comm`` name.

Python's ``Thread.name`` never reaches ``/proc/<pid>/task/*/comm``:
from outside the process every one of the mind's threads shows the
process name, which is why "which organ is this hot thread?" was
undiagnosable (thirty-plus threads all named ``genesis``). The kernel
name is set by ``prctl(PR_SET_NAME)`` and must be invoked *by* the
thread itself, so :func:`named_thread` wraps the target — on start it
sets the comm name (≤15 bytes), then runs the real target.

Comm names carry a ``g-`` prefix so her threads are recognisable in
``ps -eL`` output at a glance.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from typing import Any

from .swallow import note_swallowed

_PR_SET_NAME = 15
_PR_GET_NAME = 16
# Kernel comm limit excludes the NUL terminator.
_COMM_MAX = 15

_libc: Any = None


def _prctl() -> Any:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    return _libc.prctl


def _set_comm(name: str) -> None:
    try:
        _prctl()(_PR_SET_NAME, name.encode("utf-8", "replace")[:_COMM_MAX], 0, 0, 0)
    except Exception as e:  # noqa: BLE001
        # Naming is diagnostic only — a failure must never break the thread.
        note_swallowed("genesis_client.threads._set_comm", e)


def comm_name() -> str:
    """This thread's kernel name (used by tests to verify naming landed)."""
    buf = ctypes.create_string_buffer(_COMM_MAX + 1)
    _prctl()(_PR_GET_NAME, buf, 0, 0, 0)
    return buf.value.decode("utf-8", "replace")


def named_thread(
    name: str,
    target: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    daemon: bool = True,
    comm: str | None = None,
) -> threading.Thread:
    """Return a daemon :class:`threading.Thread` that also names itself to the kernel.

    ``name`` is the Python thread name (stack dumps, ``threading.enumerate``);
    ``comm`` is the ≤15-byte kernel name (``/proc``, ``ps -L``) — when omitted
    the truncated Python name is used.
    """

    def _named() -> None:
        _set_comm(comm if comm is not None else name)
        target(*args, **(kwargs or {}))

    return threading.Thread(target=_named, name=name, daemon=daemon)


def call_on_named_thread(
    name: str,
    fn: Callable[[], Any],
    *,
    comm: str | None = None,
    timeout: float = 30.0,
) -> Any:
    """Run *fn* on a short-lived named thread and return its result.

    C libraries spawn workers with ``clone()`` from inside their open
    or start calls — PortAudio's stream callback thread is born inside
    ``Pa_OpenStream`` or ``Pa_StartStream`` depending on the backend.
    ``clone()`` copies the creating thread's comm name and those
    workers are never renamed, so a stream opened or started on a
    ``g-*`` cognitive thread leaves a permanent, mislabeled "second
    worker" in ``ps -L`` (the ``g-inner-life`` duplicate seen in the
    field was the field-audio capture stream opened lazily on the
    inner-life thread). Callers should do both open and start inside
    *fn* — running only one leaves the worker labeled after the caller.

    The exception *fn* raises is re-raised on the calling thread; a
    ``TimeoutError`` bounds the wait when a device open wedges.
    """
    box: dict[str, Any] = {}

    def _go() -> None:
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001 — ferried to the caller
            box["error"] = e

    thread = named_thread(name, _go, comm=comm)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"{name}: device open did not finish within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")
