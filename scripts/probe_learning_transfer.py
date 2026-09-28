"""Diagnostic probe: does experience persist and alter later cognition?

This probe does not modify Genesis's cognitive architecture. It runs the real
CognitionEngine in isolated conditions:

1. baseline: target question with no relevant experience
2. relevant: teach a small novel relation, then ask a structurally related query
3. irrelevant: teach unrelated novel relations, then ask the same target

The relevant condition is also round-tripped through Genesis's persistence
layer before the target query. This separates in-process learning from
learned-state survival across a fresh engine/network instance.

The output is intentionally diagnostic rather than a pass/fail intelligence
score. A useful result is a difference between conditions that is traceable
to learned state and survives persistence.
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from genesis_client import GenesisClient
from genesis_cognitive.cognition.engine import CognitionEngine
from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.language import GenerativeEngine
from genesis_cognitive.memory import MemoryEngine
from genesis_cognitive.persistence import load_state, restore_network, save_state
from genesis_cognitive.self import SelfModel
from genesis_cognitive.user_profile import UserProfile


TARGET = "What does nib cause?"
RELEVANT_EXPERIENCE = [
    "Nib causes vesh.",
    "Vesh is a kind of flom.",
]
IRRELEVANT_EXPERIENCE = [
    "Qaz is a kind of lum.",
    "Lum causes tor.",
]


def make_engine(data_dir: str | None = None) -> tuple[CognitionEngine, ConceptNetwork, str]:
    data_dir = data_dir or tempfile.mkdtemp(prefix="genesis_probe_")
    client = GenesisClient(os.path.join(data_dir, "genesis.sock"))
    client.connect()
    self_model = SelfModel()
    network = ConceptNetwork()
    memory = MemoryEngine(client, network=network, get_emotion=lambda: None)
    language = GenerativeEngine(self_model, seed=1, network=network)
    engine = CognitionEngine(
        client=client,
        self_model=self_model,
        memory=memory,
        language=language,
        network=network,
        data_dir=data_dir,
        user_profile=UserProfile(),
    )
    return engine, network, data_dir


def find_daemon_binary() -> str:
    candidates = [
        Path(__file__).resolve().parents[1] / "target" / "release" / "genesis-daemon",
        Path(__file__).resolve().parents[1] / "target" / "debug" / "genesis-daemon",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError("genesis-daemon binary not found; build with cargo build --release")


def wait_for_socket(socket_path: str, timeout: float = 10.0) -> None:
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(socket_path)
                return
            except OSError:
                pass
        time.sleep(0.1)
    raise RuntimeError(f"daemon socket did not become ready: {socket_path}")


def start_daemon(data_dir: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            find_daemon_binary(),
            "--data-dir",
            data_dir,
            "--stm-capacity",
            "64",
            "--ltm-capacity",
            "256",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def stop_daemon(proc: subprocess.Popen[bytes], socket_path: str) -> None:
    try:
        client = GenesisClient(socket_path)
        client.connect()
        client.shutdown()
        client.disconnect()
        proc.wait(timeout=5.0)
    except Exception:  # noqa: BLE001
        proc.kill()
        proc.wait()


def snapshot(network: ConceptNetwork, engine: CognitionEngine) -> dict[str, Any]:
    edges = []
    for edge in getattr(network, "_edges", []):
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        relation = getattr(edge, "relation", None)
        if source is not None and target is not None:
            edges.append({
                "source": str(source),
                "relation": getattr(relation, "value", str(relation)),
                "target": str(target),
                "weight": getattr(edge, "weight", None),
            })

    return {
        "concept_count": network.size,
        "edge_count": network.edge_count,
        "nib_present": network.get_concept("nib") is not None,
        "vesh_present": network.get_concept("vesh") is not None,
        "learned_edges": [
            e for e in edges
            if e["source"] in {"nib", "vesh"} or e["target"] in {"nib", "vesh"}
        ],
        "conversation_turns": len(engine.memory.conversation),
    }


def state_summary(state: Any) -> dict[str, Any]:
    thought = getattr(state, "thought", None)
    return {
        "topics": list(getattr(thought, "topics", []) or []),
        "intent": getattr(thought, "intent", None),
        "confidence": getattr(thought, "confidence", None),
        "prediction_error": getattr(state, "prediction_error", None),
        "attention_foci": list(getattr(state, "attention_foci", []) or []),
        "decision_option": getattr(state, "decision_option", None),
        "decision_confidence": getattr(state, "decision_confidence", None),
        "executive_goal": getattr(state, "executive_goal", None),
        "workspace_items": list(getattr(state, "workspace_items", []) or []),
        "semantic_facts_extracted": getattr(state, "semantic_facts_extracted", None),
        "td_rpe": getattr(state, "td_rpe", None),
    }


def persist_and_reload(
    engine: CognitionEngine,
    network: ConceptNetwork,
    data_dir: str,
) -> tuple[CognitionEngine, ConceptNetwork]:
    """Persist cognitive state, then rebuild cognition around a fresh network."""
    save_state(
        data_dir,
        network,
        engine.reflection,
        engine.narrative,
        engine.self_model,
    )
    saved = load_state(data_dir)
    if saved is None:
        raise RuntimeError("persistence probe wrote no cognitive state")

    reloaded_engine, reloaded_network, _ = make_engine(data_dir)
    restore_network(reloaded_network, saved["concept_network"])
    return reloaded_engine, reloaded_network


def run_condition(
    name: str,
    experience: list[str],
    reload_before_target: bool = False,
) -> dict[str, Any]:
    data_dir = tempfile.mkdtemp(prefix="genesis_probe_")
    proc = start_daemon(data_dir)
    socket_path = os.path.join(data_dir, "genesis.sock")
    try:
        wait_for_socket(socket_path)
        engine, network, data_dir = make_engine(data_dir)
        experience_results = []

        for statement in experience:
            response, state = engine.think(statement)
            experience_results.append({
                "input": statement,
                "response": response,
                "state": state_summary(state),
            })

        before_target = snapshot(network, engine)
        reload_report: dict[str, Any] | None = None

        if reload_before_target:
            engine, network = persist_and_reload(engine, network, data_dir)
            reloaded = snapshot(network, engine)
            reload_report = {
                "concept_count": reloaded["concept_count"],
                "edge_count": reloaded["edge_count"],
                "nib_present": reloaded["nib_present"],
                "vesh_present": reloaded["vesh_present"],
                "learned_edges": reloaded["learned_edges"],
                "survived": (
                    reloaded["nib_present"]
                    and reloaded["vesh_present"]
                    and bool(reloaded["learned_edges"])
                ),
            }

        response, state = engine.think(TARGET)
        after_target = snapshot(network, engine)

        return {
            "condition": name,
            "experience": experience_results,
            "before_target": before_target,
            "persistence_reload": reload_report,
            "target": {
                "input": TARGET,
                "response": response,
                "state": state_summary(state),
            },
            "after_target": after_target,
            "persistent_state_delta": {
                "concept_count": after_target["concept_count"] - before_target["concept_count"],
                "edge_count": after_target["edge_count"] - before_target["edge_count"],
                "conversation_turns": (
                    after_target["conversation_turns"]
                    - before_target["conversation_turns"]
                ),
            },
        }
    finally:
        stop_daemon(proc, socket_path)
        shutil.rmtree(data_dir, ignore_errors=True)


def main() -> None:
    report = {
        "probe": "learning-transfer-v2",
        "target": TARGET,
        "conditions": [
            run_condition("baseline", []),
            run_condition(
                "relevant_experience",
                RELEVANT_EXPERIENCE,
                reload_before_target=True,
            ),
            run_condition("irrelevant_experience", IRRELEVANT_EXPERIENCE),
        ],
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
