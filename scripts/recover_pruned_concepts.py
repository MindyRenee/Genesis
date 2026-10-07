#!/usr/bin/env python3
"""Recover concepts that a pruning pass destroyed, back into the archive.

``clean_noise`` used to delete unconditionally during N3 sleep. It also ran
*before* ``spill_dormant`` in the same pass, so the destructive step could
drop the network under the cap and thereby stop the non-destructive step
from ever running — the mechanism meant to preserve knowledge silencing
the mechanism that preserves it. One pass destroyed ~7.5k concepts with
the archive sitting at zero, and reported it only as a number in a journal
line.

``clean_noise`` now archives when an archive is attached, so the loss
cannot recur. This script repairs the loss that already happened: every
concept present in a backup but absent from the target's working memory and
archive is written back to the archive.

**Why the archive and not working memory.** The recovered concepts are
dormant by definition — they were shed because they were low-confidence
and isolated. Working memory is capped (``max_in_memory``, 15 000) and the
target was already close to that, so re-adding them to RAM would overflow
the cap on the next pass and get them archived again immediately. The
archive is where they belong: recallable by name or id through the normal
``add_concept`` / ``_resolve`` path, occupying no RAM.

Edges are deliberately *not* restored. N3 decayed and pruned the edge set
on purpose, and resurrecting it would undo consolidation work rather than
recover anything. Recovered concepts return isolated, which is precisely
the state they were shed in.

Idempotent: concepts already present in working memory or the archive are
skipped, so re-running is safe. Dry-run by default; pass ``--write`` to
commit. Run while the mind is stopped.

Usage:
    python3 scripts/recover_pruned_concepts.py --from BACKUP [--from BACKUP] --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from genesis_conscious.concepts import ConceptNetwork, open_archive
from genesis_conscious.concepts.edge_log import open_edge_log
from genesis_conscious.infrastructure.config import default_data_dir
from genesis_conscious.infrastructure.persistence import restore_network


def _load_state(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _concepts_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Concept records from a data directory's saved state."""
    state = _load_state(path / "cognitive_state.json")
    return {c["id"]: c for c in state["concept_network"]["concepts"]}


def _resident_ids(data_dir: Path) -> tuple[set[str], set[str]]:
    """(working-memory ids, archived ids) for the target directory.

    Reads both through Genesis's own restore path and archive API rather
    than the raw JSON, so normalization and the archive schema cannot be
    bypassed.
    """
    state = _load_state(data_dir / "cognitive_state.json")
    network = ConceptNetwork()
    network.attach_edge_log(open_edge_log(data_dir))
    # restore_network takes the `concept_network` sub-dict, not the whole
    # state — passing the latter yields an empty network and reports every
    # backed-up concept as "recoverable".
    restore_network(network, state["concept_network"])
    working = set(network._concepts)
    try:
        archive = open_archive(data_dir)
    except Exception as e:
        print(f"error: cannot open archive in {data_dir}: {e}", file=sys.stderr)
        raise
    return working, archive.get_all_ids()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--to",
        default=str(default_data_dir()),
        help="target data directory (default: resolved the way run.sh does)",
    )
    ap.add_argument(
        "--from",
        dest="sources",
        action="append",
        required=True,
        help="a data directory to recover concepts from; repeatable",
    )
    ap.add_argument(
        "--write", action="store_true", help="commit (default: dry run)",
    )
    args = ap.parse_args()

    target = Path(args.to)
    working, archived = _resident_ids(target)

    candidates: dict[str, dict[str, Any]] = {}
    for src in args.sources:
        src_dir = Path(src)
        found = _concepts_by_id(src_dir)
        new = 0
        for cid, record in found.items():
            if cid in candidates:
                continue
            candidates[cid] = record
            new += 1
        print(f"  {src_dir}: {len(found)} concepts ({new} new to this run)")

    recoverable = {
        cid: rec for cid, rec in candidates.items()
        if cid not in working and cid not in archived
    }
    already = len(candidates) - len(recoverable)

    print()
    print(f"target working memory : {len(working)}")
    print(f"target archive        : {len(archived)}")
    print(f"seen across backups   : {len(candidates)}")
    print(f"already resident      : {already}")
    print(f"recoverable           : {len(recoverable)}")
    if recoverable:
        sample = sorted(recoverable)[:5]
        print(f"  sample: {', '.join(sample)}")

    if not args.write:
        print("\ndry run — pass --write to commit")
        return 0

    archive = open_archive(target)
    items = [
        (
            cid,
            {
                k: v
                for k, v in rec.items()
                if k in {
                    "id", "aliases", "activation", "confidence", "origin",
                    "columns", "created_at", "review_count", "last_reviewed",
                    "properties", "category", "modality",
                    "is_animacy_detected", "is_semantic_hub",
                }
            },
            set(rec.get("aliases") or []),
        )
        for cid, rec in recoverable.items()
    ]
    written = archive.archive_concepts_batch(items)
    archive.close()
    print(f"\narchived {written} recovered concepts into {target}")
    print("they are recallable by name or id; no edges were restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
