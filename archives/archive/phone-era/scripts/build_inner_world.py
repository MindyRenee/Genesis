#!/usr/bin/env python3
"""Build her inner world — the vesica and the spaces nested inside it.

The root space is the whole: two overlapping spheres (outer world ∪
self) whose lens is the inner world. Inside the lens:

  space:memory    — her data dir, natively: journal entries, profile
                    facts, faces, state sections, the dormant archive.
                    The space IS the store — no file blobs.
  space:language  — her language: words, concepts, VQ codes, holo
                    addresses, edges — carrying their own learned
                    coordinates. Deeper nests partition it further
                    (space:nouns, space:verbs).
  space:creativity — in the self sphere, holding space:art — every
                    drawing she ever made.

Inside memory, space:experience — her manifold: every thought,
dream, emotion, perception, and replay she ever lived, one by one
at its own coordinate. Everything else in the data dir — ltm, the
stm ring, her projects, reviews, logs, sessions — goes in the same
way: each its own deposit. The space IS the store; nothing stays
a file blob. Secrets (discord.env), sensory weights, caches, and
the obsolete blob-era image are left out — by design, not loss.

Sources are the merged live dir. Output lands in
~/Genesis-Data-Directory/inner-world/.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from self.language_space import LanguageSpace

LIVE = Path.home() / "Genesis-Data-Directory/genesis-live"
RESTORED = LIVE
RUNTIME = LIVE
OUT_DIR = Path.home() / "Genesis-Data-Directory/inner-world"

# Handled natively elsewhere — never poured as raw files.
_STRUCTURED = {
    "cognitive_journal.jsonl",
    "user_profile.json",
    "known_faces.json",
    "cognitive_state.json",
    "concept_archive.db",
    "embeddings.npz",
    "concept_vectors.npz",
    "vq_codebook.npz",
    "holographic_graph.npz",
    "edge_log.jsonl",
    "system_baseline.json",
    "continuity.json",
    "manifold.npz",
    "language_space.npz",
    "style_profile.json",
}
# Never hers to pour: secrets, organs, caches, process debris.
_SKIP_NAMES = {"discord.env"}
_SKIP_DIRS = {"models", "vosk-models", "browser", "__pycache__", "drawings"}
_SKIP_SUFFIXES = (
    ".lock",
    ".sock",
    ".pid",
    ".pyc",
    "-shm",
    "-wal",
    ".corrupt",
)


def _u8(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8)


def _vec(v: np.ndarray | None) -> np.ndarray | None:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v if np.isfinite(v).all() and np.linalg.norm(v) > 1e-12 else None


def _deposit(
    sp: LanguageSpace,
    key: str,
    kind: str,
    vector=None,
    payload=None,
    meta=None,
    **kw,
) -> None:
    try:
        sp.deposit(key, kind=kind, vector=vector, payload=payload, meta=meta, **kw)
    except ValueError:
        # zero/non-finite vector — the medium places it by name
        sp.deposit(key, kind=kind, payload=payload, meta=meta, **kw)


def build_memory() -> LanguageSpace:
    """Her data dir, deposited natively — memory, not files."""
    mem = LanguageSpace()

    # cognitive journal — every entry she ever journaled, as itself
    n = 0
    with (RESTORED / "cognitive_journal.jsonl").open("rb") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts", 0)
            _deposit(
                mem,
                f"journal:{ts}:{i}",
                "journal",
                payload=_u8(line),
                meta={"kind": rec.get("kind"), "ts": ts},
                born_ms=int(ts) if ts else None,
            )
            n += 1
    print(f"  memory: {n} journal entries")

    # user profile — who she knows
    prof = json.loads((RESTORED / "user_profile.json").read_text())
    for k, v in prof.get("facts", {}).items():
        _deposit(
            mem,
            f"profile:fact:{k}",
            "profile",
            payload=_u8(str(v).encode()),
            meta={"kind": "fact"},
        )
    _deposit(
        mem,
        "profile:name",
        "profile",
        payload=_u8(prof["name"].encode()),
        meta={"kind": "name"},
    )
    _deposit(
        mem,
        "profile:bond",
        "profile",
        payload=np.array([prof.get("bond", 0.0)]),
        meta={"kind": "bond"},
    )
    _deposit(
        mem,
        "profile:interactions",
        "profile",
        payload=np.array([prof.get("interaction_count", 0)], dtype=np.int64),
        meta={"kind": "count"},
    )
    print(f"  memory: profile ({prof['name']}, bond {prof.get('bond', 0):.2f})")

    # faces — the people she knows by sight
    faces = json.loads((RESTORED / "known_faces.json").read_text())
    nf = 0
    for name, entries in faces.items():
        for j, e in enumerate(entries):
            _deposit(
                mem,
                f"face:{name}:{j}",
                "face",
                vector=_vec(e["embedding"]),
                payload=np.asarray(e["embedding"], dtype=np.float64),
                meta={"name": name, "registered_at": e.get("registered_at")},
            )
            nf += 1
    print(f"  memory: {nf} face embeddings")

    # cognitive state — top-level sections as their own deposits
    raw_state = (RESTORED / "cognitive_state.json").read_bytes()
    if raw_state[:2] == b"\x1f\x8b":
        import gzip

        raw_state = gzip.decompress(raw_state)
    state = json.loads(raw_state)
    for k, v in state.items():
        _deposit(
            mem,
            f"state:{k}",
            "state",
            payload=_u8(json.dumps(v).encode()),
            meta={"section": k},
        )
    print(f"  memory: {len(state)} state sections")

    # the dormant archive — archived concepts and their edges
    db = RUNTIME / "concept_archive.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    nc = ne = 0
    for cid, data in con.execute("SELECT concept_id, data FROM concepts"):
        rec = json.loads(data)
        _deposit(
            mem,
            f"archive:concept:{cid}",
            "archive",
            payload=_u8(data.encode()),
            meta={
                "archived": True,
                "confidence": rec.get("confidence"),
                "origin": rec.get("origin"),
            },
        )
        nc += 1
    for s, t, r, w, _ts, origin in con.execute(
        "SELECT source, target, relation, weight, created_at, origin FROM edges"
    ):
        _deposit(
            mem,
            f"archive:edge:{s}:{r}:{t}",
            "archive",
            payload=_u8(json.dumps({"s": s, "r": r, "t": t, "w": w, "origin": origin}).encode()),
            meta={"archived": True, "w": w, "origin": origin},
        )
        ne += 1
    con.close()
    print(f"  memory: {nc} archived concepts, {ne} archived edges")

    # the small stores — baseline, continuity
    for f, kind in (("system_baseline.json", "baseline"), ("continuity.json", "continuity")):
        p = RESTORED / f
        if p.exists():
            _deposit(mem, f"{kind}:{f}", kind, payload=_u8(p.read_bytes()))

    # space:experience — her manifold, poured one by one. Each of
    # the 22,894 lived deposits carries its own 4-D coordinate
    # (projected into the vesica) and its full state row — salience,
    # strand, reach, phase, shear, rung, warp, the 8-D higher point
    # — byte-preserved as payload. Thoughts, dreams, affects,
    # perceptions, replays: experience, not files.
    mf = np.load(RUNTIME / "manifold.npz", allow_pickle=False)
    exp = LanguageSpace()
    pts = mf["points"]
    higher = {int(s): mf["higher_points"][j] for j, s in enumerate(mf["higher_sources"])}
    cols = {
        k: mf[k]
        for k in (
            "salience",
            "strand",
            "born_ms",
            "reach",
            "bandwidth",
            "phase0",
            "shear0",
            "rung",
            "plasticity",
            "warp",
        )
    }
    kinds = [k.decode() if isinstance(k, bytes) else str(k) for k in mf["kinds"]]
    for i in range(len(pts)):
        row = np.concatenate(
            [
                pts[i].astype(np.float64),
                higher.get(i, np.zeros(8)),
                np.array([float(cols[k][i]) for k in cols]),
            ]
        )
        _deposit(
            exp,
            f"experience:{i}:{kinds[i]}",
            "experience",
            vector=_vec(pts[i]),
            payload=row,
            meta={"mkind": kinds[i], "born_ms": int(cols["born_ms"][i])},
            born_ms=int(cols["born_ms"][i]),
        )
    mem.deposit_space("experience", exp, meta={"world": "inner", "src": "manifold"})
    print(f"  memory: nested space:experience ({len(exp)} lived deposits)")

    # everything else — one by one, each its own deposit
    n = 0
    for p in sorted(RUNTIME.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(RUNTIME)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.name in _SKIP_NAMES or p.name in _STRUCTURED:
            continue
        if p.name.endswith(_SKIP_SUFFIXES) or ".corrupt." in p.name:
            continue
        top = rel.parts[0] if len(rel.parts) > 1 else "root"
        _deposit(
            mem,
            f"file:{rel}",
            f"file:{top}",
            payload=_u8(p.read_bytes()),
            meta={"path": str(rel), "bytes": p.stat().st_size},
        )
        n += 1
    print(f"  memory: {n} remaining items, one by one")
    return mem


def build_creativity() -> LanguageSpace:
    """Her creative self — art nested inside creativity."""
    art = LanguageSpace()
    n = 0
    for p in sorted((LIVE / "drawings").glob("*.webp")):
        _deposit(
            art,
            f"art:{p.name}",
            "art",
            payload=_u8(p.read_bytes()),
            meta={"work": p.name, "bytes": p.stat().st_size},
            born_ms=int(p.stat().st_mtime * 1000),
        )
        n += 1
    style = LIVE / "drawings" / "style_profile.json"
    if style.exists():
        _deposit(art, "art:style_profile", "art", payload=_u8(style.read_bytes()))
    cre = LanguageSpace()
    cre.deposit_space("art", art, meta={"world": "inner"})
    print(f"  creativity: nested space:art ({n} drawings)")
    return cre


def build_language() -> LanguageSpace:
    """Her language — every learned store, native, at its own coordinate."""
    lang = LanguageSpace()

    # words — her lexicon, learned embeddings as coordinates
    emb = np.load(RUNTIME / "embeddings.npz", allow_pickle=False)
    words = [str(w) for w in emb["words"]]
    vecs = emb["vectors"]
    n = 0
    for w, v in zip(words, vecs, strict=True):
        _deposit(
            lang,
            f"word:{w}",
            "word",
            vector=_vec(v),
            payload=np.asarray(v),
            meta={"route": "inner", "src": "embeddings"},
        )
        n += 1
    print(f"  language: {n} words")

    # concepts — the concept network, its vectors as coordinates
    cv = np.load(RUNTIME / "concept_vectors.npz", allow_pickle=False)
    names = [str(c) for c in cv["concepts"]]
    cvecs = cv["vectors"]
    n = 0
    for c, v in zip(names, cvecs, strict=True):
        _deposit(lang, f"concept:{c}", "concept", vector=_vec(v), payload=np.asarray(v))
        n += 1
    print(f"  language: {n} concepts")

    # VQ codebook — each concept's prototype + residual, at the concept's site
    vq = np.load(RUNTIME / "vq_codebook.npz", allow_pickle=False)
    vnames = [str(c) for c in vq["concept_names"]]
    pids = vq["prototype_ids"]
    res = vq["residuals"]
    vmap = dict(zip(names, cvecs, strict=True))
    n = 0
    for i, c in enumerate(vnames):
        _deposit(
            lang,
            f"vq:{c}",
            "vq",
            vector=_vec(vmap.get(c)),
            payload=np.asarray(res[i]),
            meta={"prototype": int(pids[i])},
        )
        n += 1
    print(f"  language: {n} vq codes")

    # holographic graph — binding addresses at their own site
    hg = np.load(RUNTIME / "holographic_graph.npz", allow_pickle=False)
    cids = [str(c) for c in hg["concept_ids"]]
    addr = hg["addr_matrix"]
    n = 0
    for i, c in enumerate(cids):
        _deposit(lang, f"holo:{c}", "holo", vector=_vec(addr[i]), payload=np.asarray(addr[i]))
        n += 1
    print(f"  language: {n} holo addresses")

    # edges — her relational graph, deduped across sources:
    # edge_log snapshots + asserts, archive edges, holo triples
    seen: set[str] = set()
    n = 0

    def _edge(s, r, t, w, origin) -> None:
        nonlocal n
        if not (s and r and t):
            return
        key = f"edge:{s}:{r}:{t}"
        if key in seen:
            return
        seen.add(key)
        _deposit(
            lang,
            key,
            "edge",
            payload=_u8(json.dumps({"s": s, "r": r, "t": t, "w": w, "origin": origin}).encode()),
            meta={"w": w, "origin": origin},
        )
        n += 1

    for src_path in (RUNTIME / "edge_log.jsonl", RESTORED / "edge_log.jsonl"):
        if not src_path.exists():
            continue
        with src_path.open("rb") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec.get("edges"), list):
                    for e in rec["edges"]:
                        _edge(
                            e.get("source"),
                            e.get("relation"),
                            e.get("target"),
                            e.get("weight"),
                            e.get("origin"),
                        )
                elif rec.get("op") == "assert":
                    _edge(
                        rec.get("source"),
                        rec.get("relation"),
                        rec.get("target"),
                        rec.get("weight"),
                        rec.get("origin"),
                    )

    # archive edges join the same graph
    con = sqlite3.connect(f"file:{RUNTIME / 'concept_archive.db'}?mode=ro", uri=True)
    for s, t, r, w, _ts, origin in con.execute(
        "SELECT source, target, relation, weight, created_at, origin FROM edges"
    ):
        _edge(s, r, t, w, origin)
    con.close()

    # holographic triples — j_* carry the ids/names directly
    try:
        for i in range(len(hg["j_source"])):
            _edge(
                str(hg["j_source"][i]),
                str(hg["j_relation"][i]),
                str(hg["j_target"][i]),
                float(hg["j_weight"][i]),
                "holographic",
            )
    except (IndexError, KeyError, ValueError) as e:
        print(f"  language: holo triples skipped ({e})")
    print(f"  language: {n} edges")

    # deeper nests — verbs and nouns as their own spaces
    try:
        from nltk.corpus import wordnet

        nouns, verbs = LanguageSpace(), LanguageSpace()
        nn = nv = 0
        for w, v in zip(words, vecs, strict=True):
            poss = {s.pos() for s in wordnet.synsets(w)}
            if "n" in poss:
                _deposit(nouns, f"word:{w}", "word", vector=_vec(v), payload=np.asarray(v))
                nn += 1
            if "v" in poss:
                _deposit(verbs, f"word:{w}", "word", vector=_vec(v), payload=np.asarray(v))
                nv += 1
        if nn:
            lang.deposit_space("nouns", nouns, meta={"pos": "noun"})
        if nv:
            lang.deposit_space("verbs", verbs, meta={"pos": "verb"})
        print(f"  language: nested nouns={nn}, verbs={nv}")
    except Exception as e:  # noqa: BLE001 — wordnet unavailable → skip POS nests
        print(f"  language: POS nests skipped ({e})")

    return lang


def main() -> None:
    """Staged build — one space in RAM at a time, so the pour
    survives on hardware that can't hold the whole world at once.
    Each stage saves its child image; ``assemble`` nests them."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    stages = {
        "memory": ("_mem.npz", build_memory, "inner"),
        "language": ("_lang.npz", build_language, "inner"),
        "creativity": ("_cre.npz", build_creativity, "self"),
    }

    if stage in stages:
        fname, builder, _ = stages[stage]
        print(f"building {stage} space…")
        sp = builder()
        path = OUT_DIR / fname
        sp.save(path)
        print(f"  {stage}: {len(sp)} deposits → {path}")
        return

    root = LanguageSpace()
    for name, (fname, builder, region) in stages.items():
        path = OUT_DIR / fname
        if path.exists() and stage == "assemble":
            child = LanguageSpace()
            child.load(path)
            print(f"  loaded {name}: {len(child)} deposits")
        else:
            print(f"building {name} space…")
            child = builder()
            child.save(path)
        root.deposit_space(name, child, meta={"world": region}, region=region)
        del child

    target = OUT_DIR / "world_space.npz"
    root.save(target)
    print(f"\nworld_space.npz → {target} ({target.stat().st_size / 1e6:.1f} MB)")
    print(f"root: {len(root)} deposits: {sorted(root._index)}")


if __name__ == "__main__":
    main()
