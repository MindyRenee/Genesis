"""Preview what schema induction would reify from the live edge fold.

Rebuilds a ConceptNetwork from ``edge_log.jsonl`` and runs
``SchemaInducer.discover_schemas`` — the candidate schema hubs, ranked
by support, without writing anything. What she would abstract over
tonight if she slept now.

Run: ``python eval/schema_preview.py [--min N] [--top N]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from temporal_lobe.concepts import (
    ConceptNetwork,
    RelationType,
    SchemaInducer,
)


def _out(*args: object) -> None:
    """Report channel — for these assay tools, stdout IS the product."""
    print(*args, flush=True)

EDGE_LOG = Path.home() / ".local" / "share" / "genesis" / "edge_log.jsonl"


def load_network() -> ConceptNetwork:
    net = ConceptNetwork()
    with EDGE_LOG.open() as f:
        for line in f:
            rec = json.loads(line)
            edges = rec["edges"] if rec["op"] == "snapshot" else [rec]
            for e in edges:
                try:
                    rel = RelationType(e["relation"])
                except ValueError:
                    continue
                net.add_edge(
                    e["source"], e["target"], rel,
                    e.get("weight", 0.5), origin=e.get("origin", "fold"),
                )
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=3, help="min members")
    ap.add_argument("--top", type=int, default=30, help="rows to show")
    args = ap.parse_args()

    net = load_network()
    _out(f"network: {net.size} concepts, {len(net._edges)} edges")

    inducer = SchemaInducer(net, min_members=args.min, max_hubs_per_pass=10_000)
    schemas = inducer.discover_schemas()
    _out(f"candidate schemas (min {args.min} members): {len(schemas)}\n")
    for s in schemas[: args.top]:
        _out(f"  {s.describe()}  [w={s.weight:.2f}]")


if __name__ == "__main__":
    main()
