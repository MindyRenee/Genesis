#!/usr/bin/env python3
"""Restore dream-insight episodes that sleep compression archived.

``SleepCompressor._phase_compact_ltm`` used to page *every* active LTM
episode into the salience economy and archive all but the top
``max_ltm_episodes``. Dream insights are stored at a fixed 0.70 salience
(source_module 9), which can never compete in a global top-N once the
store holds more than N higher-salience episodes — so every N3 pass
archived every dream it saw. ``get_recent_episodes`` reads only the
active list, so wake residues, ``/sleep`` status, and dream
notifications all read empty even though the dreams still existed on
disk.

The compressor now exempts module-9 episodes, so the loss cannot recur.
This script repairs the loss that already happened: it clears the
ARCHIVED flag bit on every dream-insight index entry that is not
deleted, returning those episodes to ``active_ids`` — the channel every
dream surface reads.

Mechanism. The index is append-only with 64-byte entries; the flags byte
sits at offset 60 within each entry (see ``src/store/ltm.rs`` —
``IndexEntry`` layout and the writeback at ``open()``). At ``open()``
the daemon rebuilds ``active_ids`` from non-deleted, non-archived
entries, so a cleared flag takes effect on the next daemon start — no
data-file or bundle changes are needed: archived payloads were never
removed, only hidden.

**Run while the daemon is stopped.** The daemon holds its entries in
memory; editing the index under it would be invisible to the running
store and could race its own flag writes. The script refuses to run if
``<data_dir>/genesis.sock`` answers a connection.

Idempotent: entries already unarchived are skipped, so re-running is
safe. Dry-run by default; pass ``--write`` to commit. The meta file is
backed up to ``ltm_store.meta.bak.<unix-ts>`` before the first write.

Usage:
    python3 scripts/restore_archived_dreams.py [--data-dir DIR] [--write]
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent.parent / "python"),
    str(Path(__file__).resolve().parent.parent),
]

from genesis_client.ltm_index import scan_ltm_index  # noqa: E402
from infrastructure.config import default_data_dir  # noqa: E402

_DREAM_MODULE = 9
_FLAG_ARCHIVED = 0x04
_FLAG_DELETED = 0x01
_HEADER_SIZE = 64
_ENTRY_SIZE = 64
_FLAGS_OFFSET = 60  # within an entry


def _daemon_alive(socket_path: Path) -> bool:
    """True if the daemon's unix socket answers a connection."""
    if not socket_path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(str(socket_path))
            return True
    except OSError:
        return False


def restore_archived_dreams(meta_path: Path, *, write: bool) -> tuple[int, int]:
    """Clear ARCHIVED on non-deleted module-9 index entries.

    Returns ``(dream_entries_seen, entries_restored)``. With ``write``
    False it only reports what it would restore.
    """
    data = meta_path.read_bytes()
    # `scan_ltm_index` tolerates foreign/truncated files but reports them
    # as empty; check the magic ourselves so a bad file fails loudly.
    if len(data) < _HEADER_SIZE or data[:4] != b"LTMM":
        raise SystemExit(f"{meta_path}: not an LTM index (bad magic)")

    targets = [
        e
        for e in scan_ltm_index(meta_path, source_module=_DREAM_MODULE)
        if e.is_archived and not e.is_deleted
    ]
    if not targets or not write:
        seen = sum(1 for e in scan_ltm_index(meta_path, source_module=_DREAM_MODULE))
        return seen, len(targets)

    target_ids = {e.episode_id for e in targets}
    buf = bytearray(data)
    restored = 0
    for idx in range((len(buf) - _HEADER_SIZE) // _ENTRY_SIZE):
        base = _HEADER_SIZE + idx * _ENTRY_SIZE
        # Read the entry id/flags/module in place — the layout matches
        # _ENTRY_STRUCT in genesis_client.ltm_index.
        ep_id = int.from_bytes(buf[base : base + 8], "little")
        if ep_id not in target_ids:
            continue
        flags = buf[base + _FLAGS_OFFSET]
        module = buf[base + _FLAGS_OFFSET + 1]
        if module != _DREAM_MODULE or flags & _FLAG_DELETED:
            continue
        buf[base + _FLAGS_OFFSET] = flags & ~_FLAG_ARCHIVED
        restored += 1

    backup = meta_path.with_name(f"{meta_path.name}.bak.{int(time.time())}")
    shutil.copy2(meta_path, backup)
    meta_path.write_bytes(bytes(buf))
    return len(scan_ltm_index(meta_path, source_module=_DREAM_MODULE)), restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Genesis data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="commit the flag changes (default: dry-run report only)",
    )
    args = parser.parse_args()

    meta_path = args.data_dir / "ltm_store.meta"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path}: no LTM index — nothing to restore")

    if _daemon_alive(args.data_dir / "genesis.sock"):
        raise SystemExit(
            "daemon is running — stop it first (./run.sh --stop or "
            "genesis --stop). The index must only be edited offline."
        )

    seen, restored = restore_archived_dreams(meta_path, write=args.write)
    if args.write:
        print(
            f"restored {restored} of {seen} dream episodes to the active "
            "list — they resurface on the daemon's next open()."
        )
    else:
        print(
            f"would restore {restored} of {seen} dream episodes (dry-run — pass --write to commit)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
