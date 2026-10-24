"""Region-to-region messages, carried on the module telemetry.

This software bus abstracts both direct cortical and transthalamic
communication; it is not a model of mandatory thalamic routing for
all neural signals. A region posts a message onto its
own module entry in the runtime manifest — the ``task_id`` field the
heartbeat already writes and ``GET_SUBSYSTEM_TELEMETRY`` already
returns — and the channel delivers the payload to whichever regions
projected onto that module.

The manifest carries the *fact* of the message (which module spoke,
what kind) — the same record an EEG-visible trace would leave. The
payload travels the axon: ``post`` dispatches it synchronously to
in-process subscribers, because a conversational reply cannot wait
for the next telemetry sample. If a region ever lives in another
process, the same task_id arrives through ``deliver`` instead —
the wire format does not change.

``post(source_module, kind, payload)`` publishes from the source.
Receivers ``register`` against the module they listen to — a
projection, not a broadcast: only subscribers of that source hear it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

# kind is the low 8 bits of the module's task_id. 0 means the module
# has nothing to say. The remaining bits are a per-post salt so two
# identical messages in a row still register as a change on the wire.
KIND_HEARD = 1  # afferent: words reached the ear (sensory -> anyone)
KIND_SPOKE = 2  # efferent: words left the mouth (language -> anyone)
KIND_RECALLED = 3  # memory surfaced a trace
KIND_FELT = 4  # limbic state shifted

_MASK = 0xFF

# signature: (kind, payload) -> None
Handler = Callable[[int, object], None]

# signature for once(): (kind, payload) -> bool — True ends the
# projection, False leaves it listening.
OnceHandler = Callable[[int, object], bool]


def pack(kind: int, salt: int) -> int:
    """Pack a message kind into the task_id field."""
    return (int(salt) << 8) | (int(kind) & _MASK)


def unpack(task_id: int) -> int:
    """Return the message kind carried by a task_id, or 0 if none."""
    return int(task_id) & _MASK


class Thalamus:
    """Posts messages through module telemetry and delivers them back.

    ``post`` records the message a module publishes, fires the local
    axons (subscribers run synchronously on the caller's thread), and
    leaves the task_id for the next heartbeat to write into the
    manifest. ``deliver`` is the other direction: it is called with
    the module section of a telemetry report and hands any message
    this process did *not* post itself — one that arrived over the
    wire — to the subscribers of that module.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[int, object]] = {}
        self._seen: dict[int, int] = {}
        self._handlers: dict[int, list[Handler]] = {}
        self._salt = 1

    def register(self, module_id: int, handler: Handler) -> Handler:
        """Project onto ``module_id`` — hear every message it publishes."""
        with self._lock:
            self._handlers.setdefault(module_id, []).append(handler)
        return handler

    def unregister(self, module_id: int, handler: Handler) -> None:
        """Stop hearing ``module_id`` — drop the projection."""
        with self._lock:
            handlers = self._handlers.get(module_id)
            if handlers and handler in handlers:
                handlers.remove(handler)

    def once(self, module_id: int, handler: OnceHandler) -> Handler:
        """Hear the first matching message, then drop the projection.

        ``handler`` returns True when it has what it came for; the
        projection is removed only then, so non-matching messages pass
        it by without consuming it.
        """

        def _wrapper(kind: int, payload: object) -> None:
            if handler(kind, payload):
                self.unregister(module_id, _wrapper)

        return self.register(module_id, _wrapper)

    def post(self, module_id: int, kind: int, payload: object) -> int:
        """Publish a message from ``module_id``; returns its task_id.

        Subscribers fire synchronously, in registration order. The
        task_id stays in the mailbox until the heartbeat writes it —
        that is how the manifest comes to show which module spoke.
        """
        with self._lock:
            self._salt = (self._salt + 1) & 0xFFFFFF
            task_id = pack(kind, self._salt)
            self._pending[module_id] = (task_id, payload)
            # Mark delivered-in-process so the telemetry echo does
            # not replay it.
            self._seen[module_id] = task_id
            handlers = list(self._handlers.get(module_id, ()))
        for handler in handlers:
            handler(kind, payload)
        return task_id

    def take(self, module_id: int) -> int | None:
        """Return the task_id a heartbeat should publish, if any.

        The payload stays until a newer post replaces it: a reader
        sampling the manifest between heartbeat and telemetry still
        resolves the message.
        """
        with self._lock:
            pending = self._pending.get(module_id)
            return pending[0] if pending else None

    def deliver(self, modules) -> None:
        """Hand messages that arrived over the wire to subscribers.

        ``modules`` is the module section of a subsystem telemetry
        report. A task_id is delivered at most once, and only if this
        process did not post it — local posts already fired their
        axons in ``post``.
        """
        for module in modules or ():
            module_id = getattr(module, "module_id", None)
            task_id = int(getattr(module, "task_id", 0) or 0)
            if module_id is None or unpack(task_id) == 0:
                continue
            with self._lock:
                if self._seen.get(module_id) == task_id:
                    continue
                self._seen[module_id] = task_id
                posted = self._pending.get(module_id)
                payload = posted[1] if posted and posted[0] == task_id else None
                handlers = list(self._handlers.get(module_id, ()))
            for handler in handlers:
                handler(unpack(task_id), payload)
