"""Abstraction-gradient analysis over the canonical edge log.

Answers one question: is what Genesis assembles rising in abstraction
over time? Buckets edges by ``created_at`` and reports, per day:

- provenance tier mix (seed / experienced / machine / labeling /
  generative) — does self-assembled structure's share rise?
- new concepts born via generative origins — does her ontology fill
  with things she constructed, or things she was told?
- second-order assembly — edges where both endpoints were born via
  generative origins (assembly of assemblies)
- relation sophistication — causal/functional vs bare association

Run: ``python eval/abstraction_gradient.py``
Reads ``<data dir>/edge_log.jsonl`` (the live fold), resolving the
data dir the way run.sh does.
"""

from __future__ import annotations

import collections
import datetime
import json
import math
import os
import statistics
from pathlib import Path


def _out(*args: object) -> None:
    """Report channel — for these assay tools, stdout IS the product."""
    print(*args, flush=True)


GENERATIVE = {
    "swr-replay",
    "spindle",
    "lucid-dream",
    "dream-synthesis",
    "discovered",
    "insight",
    "introspection",
    "analogy",
    "spontaneous",
    "exploration",
    "corrected",
    "reified",
}
SEEDS = {
    "vocabulary_seed",
    "relation_verb_seed",
    "relation_phrase_seed",
    "situation_verb_seed",
}
MACHINE = {"semantic"}  # derivable similarity materialized per edge_log.py
LABELING = {
    "emotion_labeling",
    "mode_labeling",
    "labeling",
    "plasticity_labeling",
    "valence_labeling",
    "emotion_share",
}
CAUSAL = {
    "causes",
    "creates",
    "enables",
    "prevents",
    "leads_to",
    "emerges_from",
    "harms",
    "contradicts",
    "goal_directed",
    "opposite_of",
}
ASSOC = {"related_to", "similar_to"}
TAXONOMIC = {"is_a", "part_of"}
CODE = {"calls", "defines", "depends_on"}


def _default_data_dir() -> str:
    """Resolve the data dir the way run.sh does.

    GENESIS_DATA_DIR, then the checkout's .genesis-data-dir pin, then
    XDG_DATA_HOME, then ~/.local/share. The pin outranks XDG_DATA_HOME
    because the latter is ambient launcher environment, not Genesis
    config — a sandboxed host sets it, and grading a sibling checkout's
    edges would report its drift as this instance's.
    """
    # Blank counts as unset, matching run.sh and config.py.
    explicit = (os.environ.get("GENESIS_DATA_DIR") or "").strip()
    if explicit:
        return explicit
    pin = Path(__file__).resolve().parent.parent / ".genesis-data-dir"
    if pin.is_file():
        pinned = pin.read_text().splitlines()[0].strip()
        if pinned:
            return pinned
    return str(
        Path(
            os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")),
        )
        / "genesis",
    )


EDGE_LOG = Path(_default_data_dir()) / "edge_log.jsonl"


def tier(origin: str) -> str:
    if origin in GENERATIVE:
        return "generative"
    if origin in SEEDS:
        return "seed"
    if origin in MACHINE:
        return "machine"
    if origin in LABELING:
        return "labeling"
    return "experienced"  # observed, web:*, stated, learned, perception…


def is_cat(name: str) -> bool:
    """True for category concepts — the ``_cat:``-prefixed names."""
    return name.startswith("_cat:")


def day_of(edge: dict) -> datetime.date:
    return datetime.date.fromtimestamp(edge["created_at"] / 1000)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for i, j in enumerate(order):
        ranks[j] = float(i)
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def _second_order_table(
    days: list[datetime.date],
    by_day: dict[datetime.date, list[dict]],
    birth: dict[str, tuple[int, str]],
) -> None:
    """Second-order assembly: both endpoints born via generative origins."""
    _out(f"\n{'date':<12}{'n':>5}{'2ndOrd':>7}{'pct':>6}  (both endpoints born generatively)")
    for d in days:
        es = by_day[d]
        n = len(es)
        so = sum(
            1
            for e in es
            if birth.get(e["source"], (0, "x"))[1] == "generative"
            and birth.get(e["target"], (0, "x"))[1] == "generative"
        )
        _out(f"{d} {n:>5} {so:>7} {so / n * 100:>5.1f}")


def main() -> None:
    edges: list[dict] = []
    with EDGE_LOG.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["op"] == "snapshot":
                edges.extend(rec["edges"])
            elif rec["op"] == "assert":
                edges.append(rec)

    days = sorted({day_of(e) for e in edges})
    by_day = {d: [e for e in edges if day_of(e) == d] for d in days}
    _out(f"days: {days[0]} -> {days[-1]} ({len(days)})")
    _out(f"edges: {len(edges)}")

    _out(f"\n{'date':<12}{'n':>5}{'gen%':>7}{'exp%':>7}{'mach%':>7}{'lab%':>7}{'seed%':>7}")
    gen_share: list[tuple[datetime.date, float]] = []
    for d in days:
        es = by_day[d]
        n = len(es)
        c = collections.Counter(tier(e["origin"]) for e in es)
        g = c["generative"] / n * 100
        gen_share.append((d, g))
        _out(
            f"{d} {n:>5} {g:>6.1f} {c['experienced'] / n * 100:>6.1f}"
            f" {c['machine'] / n * 100:>6.1f} {c['labeling'] / n * 100:>6.1f}"
            f" {c['seed'] / n * 100:>6.1f}"
        )

    # Concept births: tier of the first edge that touched the concept.
    birth: dict[str, tuple[int, str]] = {}
    for e in sorted(edges, key=lambda x: x["created_at"]):
        for name in (e["source"], e["target"]):
            birth.setdefault(name, (e["created_at"], tier(e["origin"])))
    _out(f"\nconcepts: {len(birth)}")
    bc = collections.Counter(t for _, (_, t) in birth.items())
    _out(f"birth tiers: {dict(bc)}")

    _out(f"\n{'date':<12}{'newC':>5}{'genC':>5}{'genC%':>7}{'catC':>5}")
    seen: set[str] = set()
    for d in days:
        es = sorted(by_day[d], key=lambda x: x["created_at"])
        new = gen = cat = 0
        for e in es:
            for name in (e["source"], e["target"]):
                if name in seen:
                    continue
                seen.add(name)
                new += 1
                if birth[name][1] == "generative":
                    gen += 1
                if is_cat(name):
                    cat += 1
        pct = gen / new * 100 if new else 0.0
        _out(f"{d} {new:>5} {gen:>5} {pct:>6.1f} {cat:>5}")

    _second_order_table(days, by_day, birth)

    # Relation sophistication.
    def rkind(e: dict) -> str:
        r = e["relation"]
        if r in CAUSAL:
            return "causal"
        if r in TAXONOMIC:
            return "taxon"
        if r in ASSOC:
            return "assoc"
        if r in CODE:
            return "code"
        return "other"

    _out(f"\n{'date':<12}{'n':>5}{'caus%':>7}{'taxon%':>7}{'assoc%':>7}{'code%':>7}")
    for d in days:
        es = by_day[d]
        n = len(es)
        c = collections.Counter(rkind(e) for e in es)
        _out(
            f"{d} {n:>5} {c['causal'] / n * 100:>6.1f}"
            f" {c['taxon'] / n * 100:>6.1f} {c['assoc'] / n * 100:>6.1f}"
            f" {c['code'] / n * 100:>6.1f}"
        )

    xs = [float(i) for i in range(len(gen_share))]
    ys = [g for _, g in gen_share]
    _out(f"\nSpearman(gen_share, day) = {spearman(xs, ys):.3f}")
    first = [g for _, g in gen_share[:7]]
    last = [g for _, g in gen_share[-7:]]
    _out(
        f"week1 mean gen%: {statistics.mean(first):.1f}"
        f"  week3 mean gen%: {statistics.mean(last):.1f}"
    )


if __name__ == "__main__":
    main()
