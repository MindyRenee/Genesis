"""One-time prune of dead concepts from an existing cognitive_state.json.

A concept is considered dead if ALL of:
- activation is below ``_DEAD_ACTIVATION_THRESHOLD`` (never activated,
  or fully decayed without reuse — the autonomous learner leaves tiny
  positive values like 0.001 that are functionally dormant)
- review_count is 0 (never reviewed via spaced repetition)
- has no edges (truly orphaned — not connected to the network)

In the default mode, dead concepts are pruned regardless of origin,
EXCEPT for protected origins that represent deliberate seeding
(identity, introspection, foundational, seeded, structural). These
are intentionally placed and should not be removed even if dormant.

In aggressive mode (--aggressive), also prunes dead concepts whose
ONLY edges are weak auto-generated bridges (weight < 0.3, bridge
origins like semantic_bridge/hub_attachment/associative_bridge).
These are concepts that were connected by the sleep bridging system
but never actually used — the bridges are noise, not knowledge.

Protected origins (identity, introspection, foundational, seeded,
structural) are never pruned in aggressive mode — they are the
structural foundation of its mind. Taught/stated/learned concepts are
pruned only when dead AND connected solely by weak auto-generated
bridges; their teaching edges do not by themselves protect a dormant
concept.

The script backs up the original file before modifying it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import lzma
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    orjson = None  # type: ignore
    _HAS_ORJSON = False

# Gzip magic bytes (0x1f 0x8b). The live system saves
# cognitive_state.json with gzip compression (see persistence.py);
# the prune script must auto-detect this to load and save correctly.
_GZIP_MAGIC = b"\x1f\x8b"

logger = logging.getLogger(__name__)

# Origins that represent deliberate seeding — never pruned even if
# dormant. These are the structural foundation of its mind.
_PROTECTED_ORIGINS = frozenset({
    "identity", "introspection", "foundational", "seeded", "structural",
})

# Origins that represent auto-generated bridge edges — in aggressive
# mode, concepts whose ONLY edges are weak bridges are pruned.
_BRIDGE_ORIGINS = frozenset({
    "semantic_bridge", "hub_attachment", "associative_bridge", "bridge",
})

# Weight threshold for "weak" bridge edges in aggressive mode.
_WEAK_BRIDGE_THRESHOLD = 0.3

# Activation threshold below which a concept is considered dormant.
# The autonomous learner leaves tiny positive activation values
# (e.g. 0.001) on concepts it creates — these are functionally dead
# (never used in conversation or reasoning) but would not be caught
# by an exact == 0.0 check. Concepts below this threshold that also
# have zero review_count and no edges are noise from bulk learning.
_DEAD_ACTIVATION_THRESHOLD = 0.01


def _load_json(path: str) -> dict[str, Any]:
    """Load JSON state, auto-detecting gzip compression.

    The live system saves cognitive_state.json with gzip compression.
    Old state files may be raw JSON. We detect gzip by magic bytes so
    both formats work transparently.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    if _HAS_ORJSON:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def _save_json(path: str, data: dict[str, Any]) -> None:
    """Save state with gzip compression (matches the live system's format).

    The live system (persistence.py) saves with gzip at compresslevel=5.
    We match that so the pruned state file is immediately usable by the
    live system without a format-conversion save cycle.
    """
    if _HAS_ORJSON:
        data_bytes = orjson.dumps(data, default=str)
    else:
        data_bytes = json.dumps(data, default=str).encode("utf-8")
    data_bytes = gzip.compress(data_bytes, compresslevel=5)
    with open(path, "wb") as f:
        f.write(data_bytes)


def _build_edge_index(edges: list[Any]) -> dict[str, list[tuple[str, float, str]]]:
    """Build a map from concept ID to list of (neighbor_id, weight, origin)."""
    index: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for e in edges:
        source = e.get("source", "")
        target = e.get("target", "")
        weight = e.get("weight", 0.5)
        origin = e.get("origin", "inferred")
        index[source].append((target, weight, origin))
        index[target].append((source, weight, origin))
    return index


def _is_dead(concept: dict[str, Any]) -> bool:
    """Check if a concept is dead (dormant activation, 0 review_count)."""
    return (
        concept.get("activation", 0.0) < _DEAD_ACTIVATION_THRESHOLD
        and concept.get("review_count", 0) == 0
    )


def _has_only_weak_bridges(
    cid: str,
    edge_index: dict[str, list[tuple[str, float, str]]],
) -> bool:
    """Check if a concept's only edges are weak auto-generated bridges.

    Returns True if the concept has edges, but ALL of them are:
    - origin in _BRIDGE_ORIGINS (auto-generated, not deliberate)
    - weight < _WEAK_BRIDGE_THRESHOLD (weak, never reinforced)

    Returns False if the concept has no edges OR has at least one
    strong/deliberate edge.
    """
    neighbors = edge_index.get(cid, [])
    if not neighbors:
        return False  # no edges — handled by the 0-edge check
    for _neighbor, weight, origin in neighbors:
        if origin not in _BRIDGE_ORIGINS:
            return False  # has a deliberate edge
        if weight >= _WEAK_BRIDGE_THRESHOLD:
            return False  # has a strong bridge
    return True


def _classify_concepts(
    concepts: list[Any],
    edges: list[Any],
    aggressive: bool = False,
    learned_dormant: bool = False,
) -> tuple[set[str], dict[str, int]]:
    """Return (ids to keep, per-origin prune counts).

    Args:
        aggressive: Also prune dead concepts whose only edges are weak
            auto-generated bridges.
        learned_dormant: Prune ALL dormant "learned"-origin concepts
            regardless of edge connectivity. These are bulk-imported
            from Wikipedia/man pages by the autonomous learner and
            form large interconnected noise clusters. Their edges to
            each other are also noise. Edges to kept concepts are
            simply removed (the kept concept retains its other edges).
    """
    edge_index = _build_edge_index(edges)
    keep_ids: set[str] = set()
    prune_by_origin: dict[str, int] = defaultdict(int)

    for c in concepts:
        cid = c["id"]
        origin = c.get("origin", "unknown")

        # Protected origins are never pruned
        if origin in _PROTECTED_ORIGINS:
            keep_ids.add(cid)
            continue

        if not _is_dead(c):
            keep_ids.add(cid)
            continue

        # Dead concept — check edge connectivity
        neighbors = edge_index.get(cid, [])
        n_edges = len(neighbors)

        if n_edges == 0:
            # Truly orphaned — safe to prune regardless of mode
            prune_by_origin[origin] += 1
            continue

        if learned_dormant and origin == "learned":
            # Dormant learned concepts are bulk-learning noise.
            # Prune regardless of edge connectivity — the edges
            # between noise concepts are also noise, and edges to
            # kept concepts are simply removed.
            prune_by_origin[origin] += 1
            continue

        if aggressive and _has_only_weak_bridges(cid, edge_index):
            # All edges are weak auto-generated bridges — noise
            prune_by_origin[origin] += 1
            continue

        # Dead but has real connections — keep it
        keep_ids.add(cid)

    return keep_ids, prune_by_origin


def _filter_kept(
    concepts: list[Any],
    edges: list[Any],
    keep_ids: set[str],
) -> tuple[list[Any], list[Any]]:
    """Filter concepts and edges to those whose endpoints are kept."""
    kept_concepts = [c for c in concepts if c["id"] in keep_ids]
    kept_edges = [e for e in edges if e["source"] in keep_ids and e["target"] in keep_ids]
    return kept_concepts, kept_edges


def _backup_state(state_path: str, keep: int = 5) -> str:
    """Backup the state file with lzma compression, timestamped.

    The state file may be gzip-compressed (live system format) or raw
    JSON (old format). We decompress gzip first, then compress with
    lzma. Backups are write-once-read-rarely, so lzma's slower
    compression is worth the ~3x better ratio over gzip for JSON.

    Backups are timestamped (``.pre-prune-backup.<UTC>.xz``) and the
    newest ``keep`` are retained; older ones are deleted. A legacy
    ``.pre-prune-backup.xz`` (from previous versions) is rotated to
    ``.pre-prune-backup.old.xz`` once, then timestamped backups take
    over — no silent destruction of more than one generation.
    """
    import datetime

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = f"{state_path}.pre-prune-backup.{stamp}.xz"
    legacy = state_path + ".pre-prune-backup.xz"
    legacy_old = state_path + ".pre-prune-backup.old.xz"
    if os.path.exists(legacy) and not os.path.exists(legacy_old):
        shutil.move(legacy, legacy_old)
    # Read the state file, decompress gzip if needed, then compress
    # with lzma. Decompressing first avoids double-compression (gzip
    # → lzma gains almost nothing; raw JSON → lzma gains ~3x).
    with open(state_path, "rb") as f:
        raw = f.read()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    compressed = lzma.compress(raw, preset=9)
    with open(backup_path, "wb") as f:
        f.write(compressed)
    logger.info("Backed up original state to %s", backup_path)
    # Retain newest `keep` timestamped backups.
    backups = sorted(
        str(p) for p in Path(state_path).parent.glob(
            Path(state_path).name + ".pre-prune-backup.*.xz"
        )
    )
    for stale in backups[:-keep] if len(backups) > keep else []:
        try:
            os.remove(stale)
        except OSError as e:
            logger.warning("could not remove old backup %s: %s", stale, e)
    return backup_path


def _build_report(
    original_concepts: int,
    original_edges: int,
    kept_concepts: list[Any],
    kept_edges: list[Any],
    prune_by_origin: dict[str, int],
    aggressive: bool = False,
    learned_dormant: bool = False,
    original_size: int | None = None,
    new_size: int | None = None,
) -> dict[str, Any]:
    """Build the result dict, including size info when available."""
    if learned_dormant:
        mode = "learned-dormant"
    elif aggressive:
        mode = "aggressive"
    else:
        mode = "default"
    report: dict[str, Any] = {
        "mode": mode,
        "original_concepts": original_concepts,
        "original_edges": original_edges,
        "kept_concepts": len(kept_concepts),
        "kept_edges": len(kept_edges),
        "pruned_concepts": original_concepts - len(kept_concepts),
        "pruned_edges": original_edges - len(kept_edges),
        "by_origin": dict(prune_by_origin),
    }
    if original_size is not None and new_size is not None:
        report["original_size"] = original_size
        report["new_size"] = new_size
    return report


def _raw_size_of_backup(backup_path: str, state_path: str) -> int:
    """Best-effort raw (decompressed) size of the pre-prune state."""
    try:
        with open(backup_path, "rb") as f:
            return len(lzma.decompress(f.read()))
    except (OSError, lzma.LZMAError):
        pass
    try:
        return os.path.getsize(state_path)
    except OSError:
        return 0


def _raw_size_of_state(state_path: str) -> int:
    """Best-effort raw (decompressed) size of the pruned state file."""
    try:
        with open(state_path, "rb") as f:
            raw = f.read()
        if raw[:2] == _GZIP_MAGIC:
            raw = gzip.decompress(raw)
        return len(raw)
    except (OSError, gzip.BadGzipFile, EOFError):
        try:
            return os.path.getsize(state_path)
        except OSError:
            return 0


def prune_state(
    data_dir: str,
    dry_run: bool = False,
    aggressive: bool = False,
    learned_dormant: bool = False,
) -> dict[str, Any]:
    """Prune dead concepts from the state file in data_dir.

    Args:
        data_dir: Path to the directory containing cognitive_state.json.
        dry_run: If True, report what would be pruned without writing.
        aggressive: If True, also prune dead concepts whose only edges
            are weak auto-generated bridges.
        learned_dormant: If True, prune ALL dormant "learned"-origin
            concepts regardless of edge connectivity. This removes
            bulk-learning noise from the autonomous learner.
    """
    state_path = os.path.join(data_dir, "cognitive_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"No state file at {state_path}")

    logger.info("Loading %s ...", state_path)
    state = _load_json(state_path)

    if "concept_network" not in state:
        raise ValueError("State file has no concept_network key")

    cn = state["concept_network"]
    concepts = cn.get("concepts", [])
    edges = cn.get("edges", [])

    original_concepts = len(concepts)
    original_edges = len(edges)
    keep_ids, prune_by_origin = _classify_concepts(
        concepts, edges, aggressive, learned_dormant
    )
    kept_concepts, kept_edges = _filter_kept(concepts, edges, keep_ids)

    pruned_concepts = original_concepts - len(kept_concepts)
    pruned_edges = original_edges - len(kept_edges)

    if learned_dormant:
        mode_label = "learned-dormant"
    elif aggressive:
        mode_label = "aggressive"
    else:
        mode_label = "default"
    logger.info(
        "Pruning %d concepts (%s mode) and %d orphaned edges",
        pruned_concepts,
        mode_label,
        pruned_edges,
    )
    if prune_by_origin:
        for origin, count in sorted(prune_by_origin.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d", origin, count)

    if dry_run:
        return _build_report(
            original_concepts, original_edges, kept_concepts, kept_edges,
            prune_by_origin, aggressive, learned_dormant,
        )

    cn["concepts"] = kept_concepts
    cn["edges"] = kept_edges

    backup_path = _backup_state(state_path)
    _save_json(state_path, state)

    # Compare decompressed (raw JSON) sizes apples-to-apples: the
    # backup is lzma and the live file is gzip, so raw file sizes
    # are incomparable across formats. Fall back to file sizes if
    # either side fails to decompress.
    original_size = _raw_size_of_backup(backup_path, state_path)
    new_size = _raw_size_of_state(state_path)
    logger.info(
        "Saved pruned state: %d concepts, %d edges. "
        "Raw JSON size %d -> %d bytes (%.1f%% reduction)",
        len(kept_concepts),
        len(kept_edges),
        original_size,
        new_size,
        (100.0 * (original_size - new_size) / original_size) if original_size else 0.0,
    )

    return _build_report(
        original_concepts,
        original_edges,
        kept_concepts,
        kept_edges,
        prune_by_origin,
        aggressive,
        learned_dormant,
        original_size,
        new_size,
    )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Prune dead concepts from Genesis state.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(
            os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
            "genesis",
        ),
        help=(
            "Path to the data directory "
            "(default: ${XDG_DATA_HOME:-~/.local/share}/genesis, same as run.sh)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be pruned without writing anything.",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Also prune dead concepts whose only edges are weak auto-generated bridges.",
    )
    parser.add_argument(
        "--learned-dormant",
        action="store_true",
        help="Prune ALL dormant learned-origin concepts regardless of edges. "
             "Removes bulk-learning noise from the autonomous learner.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    result = prune_state(
        args.data_dir,
        dry_run=args.dry_run,
        aggressive=args.aggressive,
        learned_dormant=args.learned_dormant,
    )
    logger.info(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
