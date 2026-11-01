#!/usr/bin/env python3
"""Genesis vitals pulse — periodic measurement dump to Discord.

Posts one compact message per interval: live IPC telemetry (phase,
neurochemistry, generative-model signals, memory counts) plus deltas
folded from the canonical edge log and cognitive journal since the
last post.

Observer-only by construction: it reads state files and the IPC
socket, never writes into them, and every collector fails soft — a
missing daemon or unreadable file just drops that field from the
message rather than killing the loop.

Usage:
    GENESIS_DISCORD_PULSE_WEBHOOK=... python scripts/discord_pulse.py
    GENESIS_DISCORD_PULSE_WEBHOOK=... python scripts/discord_pulse.py --once

Env:
    GENESIS_DISCORD_PULSE_WEBHOOK  webhook for the pulse channel — ops
                                   telemetry only; her public channel
                                   (GENESIS_DISCORD_WEBHOOK) is hers
                                   alone and is never written to here
    GENESIS_DATA_DIR               data dir (default ~/.local/share/genesis)
    GENESIS_PULSE_INTERVAL         seconds between posts (default 2700)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discord_feed import DiscordFeed
from infrastructure.config import default_data_dir

logger = logging.getLogger("discord_pulse")

_STATE_NAME = "discord_pulse_state.json"


def _load_state(data_dir: Path) -> dict:
    try:
        return json.loads((data_dir / _STATE_NAME).read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_state(data_dir: Path, state: dict) -> None:
    try:
        (data_dir / _STATE_NAME).write_text(json.dumps(state))
    except OSError:
        logger.debug("pulse state save failed", exc_info=True)


def _ipc_block(data_dir: Path) -> dict[str, str]:
    """Best-effort IPC telemetry. Any field that fails is simply absent."""
    out: dict[str, str] = {}
    sock = data_dir / "genesis.sock"
    if not sock.exists():
        return out
    try:
        from genesis_client import GenesisClient

        client = GenesisClient(str(sock), request_timeout=5.0)
    except Exception:  # noqa: BLE001
        return out
    try:
        neuro = client.get_neuro_summary(timeout=5.0)
        out["phase"] = neuro.phase_name
        out["neuro"] = (
            f"arousal {neuro.arousal:.2f} · valence {neuro.valence:+.2f} · "
            f"tone {neuro.global_tone:.2f}"
        )
    except Exception:
        logger.debug("neuro summary unavailable", exc_info=True)
    try:
        inf = client.get_inference_summary(timeout=5.0)
        out["model"] = (
            f"FE {inf.free_energy:.3f} · surprise {inf.surprise_ema:.3f} · "
            f"precision {inf.precision:.2f} · allostasis {inf.allostasis_load:.2f} · "
            f"maturity {inf.model_maturity:.2f} · pe-d {inf.prediction_error_dopamine:+.2f}"
        )
        out["ticks"] = str(inf.inference_tick_count)
        out["authority"] = f"{inf.policy_authority:.2f}"
    except Exception:
        logger.debug("inference summary unavailable", exc_info=True)
    try:
        mem = client.get_memory_stats(timeout=5.0)
        out["mem"] = f"stm {mem.stm_count} · ltm {mem.ltm_count} eps"
    except Exception:
        logger.debug("memory stats unavailable", exc_info=True)
    return out


def _edge_log_delta(data_dir: Path, state: dict) -> dict[str, int]:
    """Fold new edge-log ops since the stored byte offset.

    Returns a per-origin delta histogram. First run counts the whole
    log once; afterwards only bytes past the saved offset are read.
    Snapshot ops replace the running total with their edge set.
    """
    path = data_dir / "edge_log.jsonl"
    deltas: Counter[str] = Counter()
    if not path.exists():
        return {}
    try:
        offset = int(state.get("edge_log_offset", 0))
        size = path.stat().st_size
        if offset > size:  # truncated/rotated — recount
            offset = 0
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            state["edge_log_offset"] = fh.tell()
        for raw in chunk.split(b"\n"):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            op = rec.get("op")
            if op in ("assert", "edge_assert"):
                deltas[rec.get("origin") or rec.get("edge", {}).get("origin") or "unknown"] += 1
            elif op in ("retract", "edge_retract"):
                deltas["retracted"] += 1
            elif op == "snapshot":
                state["edge_total"] = len(rec.get("edges", []))
        asserted = sum(v for k, v in deltas.items() if k != "retracted")
        state["edge_total"] = state.get("edge_total", 0) + asserted - deltas.get("retracted", 0)
    except OSError:
        return {}
    return dict(deltas)


def _journal_delta(data_dir: Path, state: dict) -> dict[str, object]:
    """New journal lines since last post: last dream utterance + insight events."""
    path = data_dir / "cognitive_journal.jsonl"
    out: dict[str, object] = {"dreams": 0, "insights": [], "last_dream": None}
    if not path.exists():
        return out
    try:
        offset = int(state.get("journal_offset", 0))
        size = path.stat().st_size
        if offset > size:
            offset = 0
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            state["journal_offset"] = fh.tell()
        last_dream = None
        for raw in chunk.split(b"\n"):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            kind = str(rec.get("kind") or "")
            text = str(rec.get("text") or "")
            if kind.startswith("dream") or "dream" in str(rec.get("trigger", "")):
                out["dreams"] = int(out["dreams"]) + 1
                if text:
                    last_dream = f'{kind}: "{text}"'
            elif text.startswith("insight"):
                out["insights"].append(text)  # type: ignore[attr-defined]
        out["last_dream"] = last_dream
    except OSError:
        logger.debug("journal delta read failed", exc_info=True)
    return out


def _concept_stats(data_dir: Path, state: dict) -> dict[str, int]:
    """Concept count + origin histogram from cognitive_state (mtime-gated)."""
    path = data_dir / "cognitive_state.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if mtime == state.get("state_mtime") and "concepts" in state:
        return {"total": int(state["concepts"])}
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            doc = json.load(fh)
    except Exception:  # noqa: BLE001
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            return {}
    net = doc.get("concept_network") or {}
    concepts = net.get("concepts") or doc.get("concepts") or {}
    if isinstance(concepts, dict):
        total = len(concepts)
        origins = Counter(
            (c.get("origin") if isinstance(c, dict) else None) or "unknown"
            for c in concepts.values()
        )
    elif isinstance(concepts, list):
        total = len(concepts)
        origins = Counter()
    else:
        return {}
    state["state_mtime"] = mtime
    state["concepts"] = total
    out = {"total": total}
    for origin, n in origins.most_common(6):
        out[f"origin:{origin}"] = n
    return out


def _compose(
    ipc: dict[str, str],
    edges: dict[str, int],
    journal: dict,
    concepts: dict,
    prev: dict,
) -> str:
    lines: list[str] = []
    phase = ipc.get("phase", "?")
    ticks = ipc.get("ticks")
    tick_s = ""
    if ticks:
        delta = int(ticks) - int(prev.get("ticks", 0) or 0)
        tick_s = f" (+{delta})" if delta else ""
        tick_s = f" · tick {int(ticks):,}{tick_s}"
    lines.append(f"[pulse] phase={phase}{tick_s}")
    if ipc.get("neuro"):
        lines.append(f"neuro: {ipc['neuro']}")
    if ipc.get("model"):
        lines.append(f"model: {ipc['model']}")
    if "authority" in ipc:
        lines.append(f"policy authority delegated: {ipc['authority']}")
    if ipc.get("mem"):
        lines.append(f"mem: {ipc['mem']}")
    edge_bits = []
    if concepts.get("total"):
        edge_bits.append(f"concepts {concepts['total']:,}")
    edge_total = prev.get("edge_total")
    if edge_total:
        edge_bits.append(f"canonical edges {edge_total:,}")
    if edges:
        delta_s = ", ".join(f"{k}:{v}" for k, v in sorted(edges.items()))
        edge_bits.append(f"new +{sum(v for k, v in edges.items() if k != 'retracted')} ({delta_s})")
    if edge_bits:
        lines.append("graph: " + " · ".join(edge_bits))
    if journal.get("dreams"):
        lines.append(f"dreams this window: {journal['dreams']}")
    if journal.get("last_dream"):
        text = str(journal["last_dream"])
        if len(text) > 160:
            text = text[:157] + "…"
        lines.append(f"last: {text}")
    for ins in journal.get("insights", [])[:3]:
        text = str(ins)
        if len(text) > 160:
            text = text[:157] + "…"
        lines.append(f"insight: {text}")
    return "\n".join(lines)


def run_once(data_dir: Path, feed: DiscordFeed) -> None:
    state = _load_state(data_dir)
    prev_ticks = state.get("ticks", 0)
    ipc = _ipc_block(data_dir)
    edges = _edge_log_delta(data_dir, state)
    journal = _journal_delta(data_dir, state)
    concepts = _concept_stats(data_dir, state)
    prev = {"ticks": prev_ticks, "edge_total": state.get("edge_total")}
    msg = _compose(ipc, edges, journal, concepts, prev)
    if ipc.get("ticks"):
        state["ticks"] = int(ipc["ticks"])
    _save_state(data_dir, state)
    feed.post(msg)
    logger.info("pulse posted (%d chars)", len(msg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="post a single pulse and exit")
    parser.add_argument(
        "--interval", type=int, default=int(os.environ.get("GENESIS_PULSE_INTERVAL", "2700"))
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    url = os.environ.get("GENESIS_DISCORD_PULSE_WEBHOOK", "").strip()
    if not url:
        logger.error(
            "no webhook configured (GENESIS_DISCORD_PULSE_WEBHOOK) — "
            "GENESIS_DISCORD_WEBHOOK is her public channel and is not a fallback"
        )
        return 2
    feed = DiscordFeed(url)
    try:
        if args.once:
            run_once(args.data_dir, feed)
            time.sleep(2)  # let the poster thread flush
            return 0
        logger.info("pulse loop: every %ds → %s", args.interval, args.data_dir)
        while True:
            try:
                run_once(args.data_dir, feed)
            except Exception:  # noqa: BLE001
                logger.exception("pulse cycle failed")
            time.sleep(args.interval)
    finally:
        feed.close()


if __name__ == "__main__":
    sys.exit(main())
