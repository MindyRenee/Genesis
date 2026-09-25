"""Read-only scanner for the LTM metadata index (``ltm_store.meta``).

The daemon's ``get_recent_episodes`` only reports *active* episodes —
sleep compression marks low-salience episodes archived, which removes
them from ``active_ids`` while keeping their payloads retrievable by
ID (``retrieve_episode``). This module reads the index file directly so
callers can see what the archive holds — e.g. dream insights that no
longer appear in recent-episode scans.

Layout (see ``src/store/ltm.rs`` — ``IndexHeader`` / ``IndexEntry``)::

    header: 64 bytes — magic "LTMM" u32 @0, version u32 @4,
            episode_count u32 @12
    entries: N × 64 bytes —
        0   episode_id        u64
        8   data_offset       u64
        16  timestamp         u64   (Unix ms)
        24  association_hash  u64
        32  compressed_len    u32
        36  uncompressed_len  u32
        40  salience          f32
        44  emotional_tag     f32 × 4
        60  flags             u8    (bit0 deleted, bit1 meta, bit2 archived)
        61  source_module     u8

The index is append-only while the daemon runs; entries are bounded by
the file length, so a partially written tail entry is skipped rather
than misparsed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_HEADER_SIZE = 64
_ENTRY_SIZE = 64
_MAGIC = b"LTMM"
_ENTRY_STRUCT = struct.Struct("<QQQQII f 4f BB2x".replace(" ", ""))
# <QQQQII f 4f BB2x = episode_id, data_offset, timestamp, assoc_hash,
# compressed_len, uncompressed_len, salience, emotional_tag[4],
# flags, source_module, pad[2] → 64 bytes

_FLAG_DELETED = 0x01
_FLAG_META_MEMORY = 0x02
_FLAG_ARCHIVED = 0x04


@dataclass(frozen=True, slots=True)
class LtmIndexEntry:
    """One row of the LTM metadata index."""

    episode_id: int
    timestamp: int
    salience: float
    flags: int
    source_module: int

    @property
    def is_deleted(self) -> bool:
        return bool(self.flags & _FLAG_DELETED)

    @property
    def is_meta_memory(self) -> bool:
        return bool(self.flags & _FLAG_META_MEMORY)

    @property
    def is_archived(self) -> bool:
        """Archived by sleep compression — off the active list but
        still retrievable by ID."""
        return bool(self.flags & _FLAG_ARCHIVED)


def scan_ltm_index(
    meta_path: str | Path, source_module: int | None = None
) -> list[LtmIndexEntry]:
    """Read the LTM index file and return its entries, oldest first.

    Args:
        meta_path: Path to ``ltm_store.meta``.
        source_module: If given, keep only entries from this module.

    Returns:
        Parsed index entries. An unreadable, missing, or foreign file
        (bad magic, trailing partial entry) yields an empty list — this
        is a read-only observer and must never disturb the store.
    """
    try:
        data = Path(meta_path).read_bytes()
    except OSError:
        return []
    if len(data) < _HEADER_SIZE or data[:4] != _MAGIC:
        return []

    # Bound by file length, not the header's episode_count — the daemon
    # checkpoints that field lazily (it can lag far behind the appended
    # entries), and its own rebuild iterates the full entry region.
    available = (len(data) - _HEADER_SIZE) // _ENTRY_SIZE
    entries: list[LtmIndexEntry] = []
    for i in range(available):
        offset = _HEADER_SIZE + i * _ENTRY_SIZE
        fields = _ENTRY_STRUCT.unpack_from(data, offset)
        # fields: id, data_offset, timestamp, assoc_hash, comp_len,
        #         uncomp_len, salience, tag0..3, flags, source_module
        entry = LtmIndexEntry(
            episode_id=fields[0],
            timestamp=fields[2],
            salience=fields[6],
            flags=fields[11],
            source_module=fields[12],
        )
        if source_module is None or entry.source_module == source_module:
            entries.append(entry)
    return entries
