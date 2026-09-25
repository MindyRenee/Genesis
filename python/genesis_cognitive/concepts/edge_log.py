"""Canonical edge log — the single source of truth for relationships.

The concept network's edge list is a fold over this log, not an
independently persisted structure. Everything else that looks like a
relationship store — the cognitive_state.json edge array, the archive
edge table, the holographic npz — is either a derived view of this
log or a separate tier (cold typed associations) with its own scope.

Semantics: append-only, last-write-wins per edge key.

- ``assert`` — upsert an edge. Carries absolute weight; re-asserting
  an existing edge records the new weight (reinforcement).
- ``retract`` — tombstone an edge. Fold drops it.
- ``snapshot`` — replace the entire fold base. Used by bulk rewrites
  (sleep compression) and by seeding an empty log from a live network.
  Everything before a snapshot line is dead history once compaction
  runs.

Compaction rewrites the file as a single snapshot line of the live
fold, then appends nothing else — history is a debugging aid, not a
requirement.

Derivability: edges whose content is a function of the embedding
geometry (untyped associations created by machine pipelines —
hub_attachment, semantic bridges, co-occurrence) are never logged.
They are recomputed at query time by the similarity provider. The
log holds only earned facts: typed relational claims and untyped
associations with asserted or observed provenance.

Decay is a fold-time policy (``decay_fn``), off by default — the log
stores raw earned weights and lets the reader price them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .types import Edge, RelationType

logger = logging.getLogger(__name__)


# Origins produced by machine pipelines — statistics materialized as
# edges. Combined with an untyped relation these edges carry no
# information the embedding field cannot recompute, so they are
# derivable and never enter the canonical log.
DERIVABLE_ORIGINS: frozenset[str] = frozenset({
    "hub_attachment",
    "co_occurrence",
    "associative_bridge",
    "semantic_bridge",
    "semantic_connect",
    "semantic",
    "bridge_creation",
    "bridge",
    "global_connect",
    "dedupe",
    "inferred",
})

# Relations that are pure geometry — a typed relation is always a
# factual claim (even a weakly provenanced one), but these three only
# assert "these are associated", which is exactly what similarity
# measures. With a derivable origin they are redundant storage.
GEOMETRIC_RELATIONS: frozenset[RelationType] = frozenset({
    RelationType.RELATED_TO,
    RelationType.BRIDGES,
    RelationType.SIMILAR_TO,
})

def is_derivable_edge(relation: RelationType | str, origin: str) -> bool:
    """True if this edge is pipeline-generated geometry, not an earned fact.

    Derivable edges must never be persisted — they are recomputed at
    query time by the similarity provider. The rule is intentionally
    asymmetric:

    - Typed relations are always canonical: ``is_a``, ``causes``,
      ``calls`` are claims that similarity cannot reconstruct, no
      matter which pipeline produced them.
    - Untyped relations (``related_to``, ``bridges``, ``similar_to``)
      are derivable only with a KNOWN pipeline origin. Unlisted
      provenance defaults to canonical — a false positive costs
      decay bandwidth; a false negative loses an irrecoverable
      experiential binding (sleep-replay links, introspection
      associations, sensory co-activation).
    """
    rel_value = relation.value if isinstance(relation, RelationType) else relation
    if rel_value not in {r.value for r in GEOMETRIC_RELATIONS}:
        return False
    return origin in DERIVABLE_ORIGINS


def edge_key(source: str, target: str, relation: str) -> str:
    """Stable string key for an edge triple (used as fold map key)."""
    return f"{source}	{target}	{relation}"


class EdgeLog:
    """Append-only canonical store for concept-network edges.

    The log file is JSONL: one event per line. Append is O(1) and
    crash-safe (a torn last line is detected and skipped on fold).
    ``fold()`` replays the log into the live edge map; ``compact()``
    rewrites it as a single snapshot.

    Thread-safe: all mutating calls serialize on an internal lock.
    """

    def __init__(self, path: Path, decay_fn: Callable[[Edge, int], float] | None = None) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._fh: TextIO | None = None
        # Fold-time decay policy: (edge, now_ms) -> effective weight.
        # None = raw earned weights (default). The mechanism lives
        # here so policy can be activated without touching callers.
        self._decay_fn = decay_fn
        self._open()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_empty(self) -> bool:
        """True when the log holds no events.

        The file itself is created at open time (``a+``), so existence
        is meaningless — only a non-empty log is authoritative.
        """
        try:
            return self._path.stat().st_size == 0
        except OSError:
            return True

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # a+ so fold() can read from the start and append writes at end
        self._fh = open(self._path, "a+", encoding="utf-8")
        self._fh.seek(0, os.SEEK_END)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None

    def __enter__(self) -> EdgeLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ─── Writes ─────────────────────────────────────────────────

    def _write(self, event: dict[str, Any]) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")

    def assert_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float,
        origin: str,
        created_at: int,
    ) -> bool:
        """Append an assert event. Returns False if the edge is derivable.

        Derivable edges are rejected at the boundary — callers can
        pass every edge blindly and the log enforces the rule.
        """
        if is_derivable_edge(relation, origin):
            return False
        with self._lock:
            self._write({
                "op": "assert",
                "source": source,
                "target": target,
                "relation": relation.value,
                "weight": weight,
                "origin": origin,
                "created_at": created_at,
            })
            if self._fh is not None:
                self._fh.flush()
        return True

    def retract_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
    ) -> None:
        """Append a tombstone for an edge triple."""
        with self._lock:
            self._write({
                "op": "retract",
                "source": source,
                "target": target,
                "relation": relation.value,
            })
            if self._fh is not None:
                self._fh.flush()

    def snapshot(self, edges: list[Edge]) -> None:
        """Append a full-state snapshot, resetting the fold base."""
        payload = [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation.value,
                "weight": e.weight,
                "origin": e.origin,
                "created_at": e.created_at,
            }
            for e in edges
            if not is_derivable_edge(e.relation, e.origin)
        ]
        with self._lock:
            self._write({"op": "snapshot", "edges": payload})
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())

    def sync(self) -> None:
        """Flush + fsync pending appends (called at save points)."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())

    # ─── Reads ──────────────────────────────────────────────────

    def fold(self, now_ms: int = 0) -> dict[str, Edge]:
        """Replay the log into the live edge map (last write wins).

        Returns ``{key: Edge}`` for every live (non-retracted,
        non-derivable) edge. ``now_ms`` is passed to the decay policy
        when one is attached; weights are stored raw either way —
        decay, when enabled, prices at fold time rather than mutating
        history.
        """
        live: dict[str, Edge] = {}
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
            try:
                with open(self._path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # torn tail from a crash — skip
                        self._apply(ev, live)
            except FileNotFoundError:
                return {}
        if self._decay_fn is not None and now_ms:
            for e in live.values():
                e.weight = max(0.0, min(1.0, self._decay_fn(e, now_ms)))
        return live

    @staticmethod
    def _apply(ev: dict[str, Any], live: dict[str, Edge]) -> None:
        op = ev.get("op")
        if op == "snapshot":
            live.clear()
            for e in ev.get("edges", []):
                EdgeLog._apply(
                    {"op": "assert", **e}, live
                )
            return
        source, target, relation = ev.get("source"), ev.get("target"), ev.get("relation")
        if not (source and target and relation):
            return
        key = edge_key(source, target, relation)
        if op == "assert":
            try:
                rel = RelationType(relation)
            except ValueError:
                return
            if is_derivable_edge(rel, ev.get("origin", "")):
                live.pop(key, None)
                return
            existing = live.get(key)
            live[key] = Edge(
                source=source,
                target=target,
                relation=rel,
                weight=float(ev.get("weight", 0.5)),
                created_at=int(ev.get("created_at", 0))
                or (existing.created_at if existing else 0),
                origin=ev.get("origin", "inferred"),
            )
        elif op == "retract":
            live.pop(key, None)

    def __len__(self) -> int:
        return len(self.fold())

    # ─── Compaction ─────────────────────────────────────────────

    def compact(self) -> int:
        """Rewrite the log as a single snapshot of the live fold.

        Returns the number of live edges written. Atomic via
        temp-file + rename; the fold is taken under the lock so no
        append can interleave mid-compaction.
        """
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
            live = self.fold()
            edges = [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation.value,
                    "weight": e.weight,
                    "origin": e.origin,
                    "created_at": e.created_at,
                }
                for e in live.values()
            ]
            fd, tmp = tempfile.mkstemp(
                dir=self._path.parent, prefix="edge_log_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"op": "snapshot", "edges": edges}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                # Close before rename so the new file is the one we append to
                if self._fh is not None:
                    self._fh.close()
                os.replace(tmp, self._path)
                self._open()
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                if self._fh is None:
                    self._open()
                raise
        return len(edges)


def open_edge_log(
    data_dir: str | Path,
    decay_fn: Callable[[Edge, int], float] | None = None,
) -> EdgeLog:
    """Open (creating if needed) the canonical edge log for a data dir."""
    return EdgeLog(Path(data_dir) / "edge_log.jsonl", decay_fn=decay_fn)
