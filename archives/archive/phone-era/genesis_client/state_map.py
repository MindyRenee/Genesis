"""Ambient reads of the body's memory-mapped CoreState.

``core_state.bin`` is the extracellular medium: the daemon publishes
the whole neurochemical field into it every tick (~200 ms), and any
process mapping the file can sample the field at its own cadence.
That is how volume transmission works in tissue — neuromodulators
diffuse through the extracellular fluid and every region bathed in
them is modulated locally, whether or not it sent a message. Reading
the field directly makes it ambient: no request/reply pair on the
IPC socket, no frame that can desync, no queueing behind writes.
The same 3480 bytes ``GET_STATE`` ships, read off the page.

Consistency model
-----------------
The daemon publishes non-atomically — ``MmapState::modify`` mutates a
stack copy and publishes it with a single volatile write — so a
reader can observe the file mid-publish (the daemon's own
``read_consistent`` accepts torn reads as organism semantics). The
file's CRC32 is what makes an *un*-torn snapshot detectable: it
covers every byte except the vestigial ``seq_lock`` word (header
40..48), the checksum field itself, and the derived
``inference_signals`` tail, which is recomputed each tick and allowed
to be torn. A copy whose checksum verifies was observed between
publishes — a consistent snapshot — and a torn copy fails
verification and is retried. False-pass odds are ~2**-32 per torn
read, and the retry budget covers a publish window by orders of
magnitude.

Freshness
---------
``header.last_updated`` is stamped on every publish. A field whose
stamp is older than ``max_age_ms`` is treated as unobservable — a
body that stopped publishing stopped living — and the caller falls
back to the addressed channel, preserving the dead-daemon error
contract the IPC path already had.
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
import zlib

from .types import CoreState

# The state file's name inside the data dir (src/bin/genesis-daemon.rs).
STATE_FILENAME = "core_state.bin"

# Schema v5 — src/state/header.rs::SCHEMA_VERSION and the fixed
# GenesisCoreState layout (src/state/core_state.rs).
SCHEMA_VERSION = 5
STATE_SIZE = 3480
_MAGIC = b"GNSX"
_CHECKSUM_OFFSET = 3412
# CRC32 (IEEE 802.3) coverage: [0..40) then [48..checksum). The
# excluded 40..48 is the vestigial seq_lock word; the tail after the
# checksum (inference_signals, 3416..3480) is derived state outside
# the CRC by design. zlib.crc32 is the same polynomial/init/xor as
# the daemon's crc32_ieee.
_CRC_RANGES = ((0, 40), (48, _CHECKSUM_OFFSET))
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

# A publish is a ~3.5 KB volatile copy — microseconds. Eight attempts
# at half-millisecond spacing covers a collision window by orders of
# magnitude while still failing fast on a genuinely dead writer.
_MAX_ATTEMPTS = 8
_RETRY_SECONDS = 0.0005


def _verify(buf: bytes) -> bool:
    """True when *buf* is a whole, schema-v5 state image.

    The checks are ordered cheapest-first: magic, schema version, and
    declared size before the CRC walk. A publish racing the copy makes
    the CRC mismatch; a recreated file makes the identity checks fail.
    """
    if len(buf) < STATE_SIZE:
        return False
    if buf[0:4] != _MAGIC:
        return False
    if _U32.unpack_from(buf, 4)[0] != SCHEMA_VERSION:
        return False
    if _U32.unpack_from(buf, 8)[0] != STATE_SIZE:
        return False
    crc = 0
    for start, end in _CRC_RANGES:
        crc = zlib.crc32(buf[start:end], crc)
    return crc == _U32.unpack_from(buf, _CHECKSUM_OFFSET)[0]


class CoreStateMap:
    """Read-only mapping of the daemon's ``core_state.bin``.

    The field each region bathes in. ``read`` returns the raw
    3480-byte image — the same bytes ``GET_STATE`` would ship — or
    ``None`` when the field can't be observed right now (missing,
    stale, wrong schema, or persistently torn), in which case the
    caller falls back to the IPC path.

    Rebirth is followed automatically: a reincarnated daemon may
    recreate the file, so the path is re-stat'd and the inode
    compared on every read.

    Lock-free: each thread holds its own mapping of the same page —
    a private receptor for the ambient field — so open/close can
    never interleave between threads. A thread's map dies with it;
    ``close`` drops the calling thread's.
    """

    def __init__(self, path: str, max_age_ms: float = 10_000.0) -> None:
        self.path = path
        self.max_age_ms = max_age_ms
        self._axons = threading.local()

    def close(self) -> None:
        """Drop this thread's mapping. The next read reopens lazily."""
        mm = getattr(self._axons, "mm", None)
        fd = getattr(self._axons, "fd", None)
        self._axons.mm = self._axons.fd = self._axons.ino = None
        if mm is not None:
            try:
                mm.close()
            except (OSError, ValueError):
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _reopen(self) -> bool:
        """Ensure the mapping is open on the live inode.

        Compares the path's inode to the mapped one so a reborn
        daemon's recreated file is picked up rather than reading a
        detached inode forever. False when the file is absent or not
        yet fully formed (created but not yet ftruncated).
        """
        try:
            st = os.stat(self.path)
        except OSError:
            self.close()
            return False
        if getattr(self._axons, "mm", None) is not None and self._axons.ino == st.st_ino:
            return True
        self.close()
        if st.st_size < STATE_SIZE:
            return False
        try:
            fd = os.open(self.path, os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            except OSError:
                os.close(fd)
                return False
        except OSError:
            return False
        self._axons.fd, self._axons.mm, self._axons.ino = fd, mm, st.st_ino
        return True

    def read(self, max_age_ms: float | None = None) -> bytes | None:
        """Copy a verified snapshot of the state field.

        Returns the raw 3480-byte image, or ``None`` when no verified
        snapshot could be taken within the retry budget or the newest
        publish is older than *max_age_ms* (default: the map's).
        """
        limit = self.max_age_ms if max_age_ms is None else max_age_ms
        if not self._reopen():
            return None
        mm = self._axons.mm
        assert mm is not None
        for _ in range(_MAX_ATTEMPTS):
            buf = bytes(mm[:STATE_SIZE])
            if not _verify(buf):
                time.sleep(_RETRY_SECONDS)
                continue
            last_updated = _U64.unpack_from(buf, 24)[0]
            if limit is not None and (
                last_updated <= 0 or time.time() * 1000.0 - last_updated > limit
            ):
                return None
            return buf
        return None

    def read_state(self, max_age_ms: float | None = None) -> CoreState | None:
        """Verified snapshot decoded into a ``CoreState``, or None."""
        buf = self.read(max_age_ms)
        return CoreState.unpack(buf) if buf is not None else None
