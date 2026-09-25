#!/usr/bin/env python3
"""Migrate Genesis's edges into the canonical edge log.

Reads ``cognitive_state.json`` (gzip or plain) and
``concept_archive.db`` from a data directory, classifies every edge:

- canonical  — typed relations and earned untyped edges → logged
- derivable  — pipeline-generated geometry → dropped (recomputed by
  the embedding similarity provider at query time, never stored)
- orphan     — an endpoint exists in no concept store → dropped

Writes ``edge_log.jsonl`` as a single snapshot event. Dry-run by
default; pass ``--write`` to commit. Run while the mind is stopped —
the log is folded at next boot and outranks the JSON edge projection.

Usage:
    python3 scripts/migrate_edges_to_log.py [--data-dir DIR] [--write] [--force]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from genesis_cognitive.concepts.edge_log import (
    EdgeLog,
    is_derivable_edge,
)
from genesis_cognitive.concepts.types import RelationType


def _load_cognitive_state(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _archive_rows(db_path: Path) -> tuple[list[dict], set[str]]:
    """Return (edge dicts, archived concept ids) from concept_archive.db."""
    if not db_path.exists():
        return [], set()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        edges = [
            {
                "source": r[0],
                "target": r[1],
                "relation": r[2],
                "weight": r[3],
                "created_at": r[4],
                "origin": r[5],
            }
            for r in con.execute(
                "SELECT source, target, relation, weight, created_at, origin "
                "FROM edges"
            )
        ]
        try:
            concepts = {
                r[0] for r in con.execute("SELECT concept_id FROM concepts")
            }
        except sqlite3.Error:
            concepts = set()
        return edges, concepts
    finally:
        con.close()


def migrate(data_dir: Path, write: bool, force: bool) -> int:
    state_path = data_dir / "cognitive_state.json"
    log_path = data_dir / "edge_log.jsonl"
    if not state_path.exists():
        print(f"no cognitive_state.json in {data_dir}", file=sys.stderr)
        return 2
    if log_path.exists() and log_path.stat().st_size > 0 and not force:
        print(
            f"{log_path} already exists and is non-empty — pass --force to rebuild",
            file=sys.stderr,
        )
        return 2

    state = _load_cognitive_state(state_path)
    cn = state.get("concept_network", {})
    concepts = {c["id"] for c in cn.get("concepts", [])}
    working_edges = cn.get("edges", [])

    archived_edges, archived_concepts = _archive_rows(
        data_dir / "concept_archive.db"
    )
    all_concepts = concepts | archived_concepts

    stats: Counter[str] = Counter()
    kept: dict[str, dict] = {}
    dropped_origins: Counter[str] = Counter()

    def consider(e: dict, store: str) -> None:
        try:
            rel = RelationType(e["relation"])
        except ValueError:
            stats[f"{store}_unknown_relation"] += 1
            return
        src, tgt, origin = e["source"], e["target"], e.get("origin", "")
        if src not in all_concepts or tgt not in all_concepts:
            stats[f"{store}_orphan"] += 1
            dropped_origins[f"orphan:{origin}"] += 1
            return
        if is_derivable_edge(rel, origin):
            stats[f"{store}_derivable"] += 1
            dropped_origins[origin] += 1
            return
        key = f"{src}\t{tgt}\t{rel.value}"
        prev = kept.get(key)
        if prev is None or e.get("weight", 0.5) > prev["weight"]:
            kept[key] = {
                "source": src,
                "target": tgt,
                "relation": rel.value,
                "weight": e.get("weight", 0.5),
                "origin": origin,
                "created_at": e.get("created_at", 0),
            }
        stats[f"{store}_canonical"] += 1

    for e in working_edges:
        consider(e, "json")
    for e in archived_edges:
        consider(e, "archive")

    print(f"working JSON edges:   {len(working_edges)}")
    print(f"archive DB edges:     {len(archived_edges)}")
    print(f"concepts (live+arch): {len(all_concepts)}")
    print("---")
    for k, v in sorted(stats.items()):
        print(f"{k:24s} {v}")
    print("---")
    print(f"canonical edges kept: {len(kept)}")
    if dropped_origins:
        print("dropped by origin:")
        for o, n in dropped_origins.most_common(15):
            print(f"  {o or '?':24s} {n}")

    if not write:
        print("\ndry-run — pass --write to commit")
        return 0

    edges_list = list(kept.values())
    # Write via a temp EdgeLog on a scratch file, then atomic-rename
    # into place — the canonical file is born complete or not at all.
    tmp = data_dir / "edge_log.jsonl.tmp"
    try:
        tmp.unlink(missing_ok=True)
        log = EdgeLog(tmp)
        # Fold the classified set through the same event semantics the
        # runtime uses: snapshot is the migration's birth record.
        from genesis_cognitive.concepts.types import Edge

        log.snapshot([
            Edge(
                source=e["source"],
                target=e["target"],
                relation=RelationType(e["relation"]),
                weight=e["weight"],
                created_at=e["created_at"],
                origin=e["origin"],
            )
            for e in edges_list
        ])
        log.close()
        tmp.replace(log_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    print(f"\nwrote {log_path} ({len(edges_list)} edges)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default=str(Path.home() / ".local/share/genesis"),
        help="Genesis data directory",
    )
    p.add_argument("--write", action="store_true", help="commit the migration")
    p.add_argument("--force", action="store_true", help="overwrite an existing non-empty log")
    args = p.parse_args()
    return migrate(Path(args.data_dir), args.write, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
