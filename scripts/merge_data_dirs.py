#!/usr/bin/env python3
"""Merge a forked Genesis data directory into its canonical one.

A checkout pins its state directory with ``.genesis-data-dir``, but a
launcher that sets ``XDG_DATA_HOME`` (Flatpak, a sandboxed agent host)
used to override that pin, so the mind forked a second, divergent state
directory. Both instances then accumulated their own concepts, edges,
journal entries, and projects, and neither is a superset of the other.

This merges the fork's *lossless* state into the canonical directory:

- **concepts** — union by id. Fields that accumulate (activation,
  confidence, review_count, last_reviewed) take the max; set-valued
  fields (aliases, columns) take the union; properties are merged with
  the canonical side winning conflicts; ``created_at`` takes the min so
  a concept keeps its real age.
- **edges** — each side's ``edge_log.jsonl`` is folded to its live edge
  set (assert/retract replayed, last write wins), the two are unioned,
  and the result is written back as a single snapshot event. Folding
  first means no retract from one side can cancel an assert from the
  other. The log is canonical for edges, so the JSON ``edges`` array is
  only a debugging projection and is left untouched.
- **journal** — ``cognitive_journal.jsonl`` is append-only, so the fork's
  entries are appended verbatim.
- **projects** — copied whole. Aborts if any name collides, because
  merging two same-named project directories is not something this
  script can do safely.

NOT merged, because the formats have no supported union:

- ``ltm_store.*`` (episodic memory) and the derived indices
  ``concept_vectors.npz`` / ``holographic_graph.npz`` /
  ``vq_codebook.npz``. The canonical side keeps its own. Vectors and the
  holographic graph are rebuilt for concepts that lack them; the fork's
  ~110 KB of episodes are reported as a known loss.

Run while the mind is stopped. Dry-run by default; pass ``--write`` to
commit.

Usage:
    python3 scripts/merge_data_dirs.py --from FORK_DIR --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from genesis_conscious.concepts.edge_log import EdgeLog, open_edge_log
from genesis_conscious.concepts.types import Edge


def _load_state(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _write_state(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data).encode()
    tmp = path.with_suffix(path.suffix + ".merge-tmp")
    with open(tmp, "wb") as fh:
        fh.write(gzip.compress(payload))
        fh.flush()
    tmp.replace(path)


def _merge_concept(canonical: dict, incoming: dict) -> dict:
    """Union two records for the same concept id."""
    merged = dict(canonical)
    for key in ("aliases", "columns"):
        merged[key] = sorted(
            set(canonical.get(key) or []) | set(incoming.get(key) or []),
        )
    for key in ("activation", "confidence", "review_count", "last_reviewed"):
        merged[key] = max(
            canonical.get(key) or 0, incoming.get(key) or 0,
        )
    # Earliest creation wins: it is the concept's real age, and a later
    # copy must not make an older idea look newly learned.
    created = [c for c in (canonical.get("created_at"), incoming.get("created_at")) if c]
    merged["created_at"] = min(created) if created else 0
    props = dict(incoming.get("properties") or {})
    props.update(canonical.get("properties") or {})
    merged["properties"] = props
    for key in ("is_animacy_detected", "is_semantic_hub"):
        merged[key] = bool(canonical.get(key)) or bool(incoming.get(key))
    # "unknown" is the absence of a classification, so a real value on
    # either side beats it.
    for key in ("category", "modality"):
        if (canonical.get(key) or "unknown") == "unknown":
            merged[key] = incoming.get(key) or "unknown"
    if (canonical.get("origin") or "conversation") == "conversation":
        merged["origin"] = incoming.get("origin") or "conversation"
    merged["id"] = canonical["id"]
    return merged


def _fold_edges(data_dir: Path) -> dict[str, Edge]:
    """Replay a directory's edge log into its live edge set."""
    log_path = data_dir / "edge_log.jsonl"
    if not log_path.exists():
        return {}
    log = EdgeLog(log_path)
    try:
        return log.fold()
    finally:
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--to", required=True, help="canonical data dir")
    ap.add_argument("--from", dest="src", required=True, help="fork data dir")
    ap.add_argument("--write", action="store_true", help="commit (default: dry run)")
    args = ap.parse_args()

    dst, src = Path(args.to), Path(args.src)
    for d in (dst, src):
        if not (d / "cognitive_state.json").exists():
            print(f"error: no cognitive_state.json in {d}", file=sys.stderr)
            return 1

    dst_state = _load_state(dst / "cognitive_state.json")
    src_state = _load_state(src / "cognitive_state.json")

    # ── concepts ────────────────────────────────────────────────
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for c in dst_state["concept_network"]["concepts"]:
        by_id[c["id"]] = dict(c)
        order.append(c["id"])
    added = merged_n = 0
    for c in src_state["concept_network"]["concepts"]:
        cid = c["id"]
        if cid in by_id:
            by_id[cid] = _merge_concept(by_id[cid], c)
            merged_n += 1
        else:
            by_id[cid] = dict(c)
            order.append(cid)
            added += 1
    print(f"concepts: {len(by_id)} total (+{added} new, {merged_n} merged)")

    # ── edges ───────────────────────────────────────────────────
    edges = _fold_edges(dst)
    dst_edges = len(edges)
    edge_added = edge_kept = 0
    for key, e in _fold_edges(src).items():
        cur = edges.get(key)
        if cur is None:
            edges[key] = e
            edge_added += 1
        else:
            if e.weight > cur.weight:
                edges[key] = e
            edge_kept += 1
    known = set(by_id)
    orphaned = [e for e in edges.values() if e.source not in known or e.target not in known]
    edges = {
        k: e for k, e in edges.items()
        if e.source in known and e.target in known
    }
    print(
        f"edges: {len(edges)} total "
        f"({dst_edges} canonical + {edge_added} from fork, {edge_kept} merged, "
        f"{len(orphaned)} orphans dropped)",
    )

    # ── journal ─────────────────────────────────────────────────
    j_dst = dst / "cognitive_journal.jsonl"
    j_src = src / "cognitive_journal.jsonl"
    j_lines = 0
    if j_src.exists():
        j_lines = sum(1 for line in j_src.open("rb") if line.strip())

    # ── projects ────────────────────────────────────────────────
    p_dst, p_src = dst / "projects", src / "projects"
    clashes = []
    new_projects = []
    if p_dst.exists() and p_src.exists():
        existing = {p.name for p in p_dst.iterdir()}
        new_projects = [p for p in p_src.iterdir() if p.name not in existing]
        clashes = [p.name for p in p_src.iterdir() if p.name in existing]
    print(f"journal: {j_lines} entries to append")
    print(f"projects: {len(new_projects)} to copy, {len(clashes)} name clashes")
    if clashes:
        print(f"error: project name collisions: {clashes}", file=sys.stderr)
        return 1

    ltm = (src / "ltm_store.dat").stat().st_size if (src / "ltm_store.dat").exists() else 0
    print(f"NOT merged: fork ltm_store.dat ({ltm} bytes of episodes) and derived indices")

    if not args.write:
        print("\ndry run — pass --write to commit")
        return 0

    # ── commit ──────────────────────────────────────────────────
    dst_state["concept_network"]["concepts"] = [by_id[cid] for cid in order]
    _write_state(dst / "cognitive_state.json", dst_state)
    print(f"wrote {dst / 'cognitive_state.json'}")

    log = open_edge_log(dst)
    try:
        log.snapshot(list(edges.values()))
        log.close()
    except Exception:
        log.close()
        raise
    print(f"wrote edge snapshot to {dst / 'edge_log.jsonl'}")

    if j_lines:
        with j_dst.open("ab") as out, j_src.open("rb") as inp:
            out.write(inp.read())
        print(f"appended {j_lines} journal entries")

    for p in new_projects:
        shutil.copytree(p, p_dst / p.name)
    if new_projects:
        print(f"copied {len(new_projects)} projects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
