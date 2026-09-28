"""Diagnostic probe: does experience persist and alter later cognition?

This probe does not modify Genesis's cognitive architecture. It runs the real
CognitionEngine in three isolated in-memory conditions:

1. baseline: target question with no relevant experience
2. relevant: teach a small novel relation, then ask a structurally related query
3. irrelevant: teach unrelated novel relations, then ask the same target

The output is intentionally diagnostic rather than a pass/fail intelligence
score. A useful result is a difference between conditions that is traceable to
persistent learned state.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from genesis_client import GenesisClient
from genesis_cognitive.cognition.engine import CognitionEngine
from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.language import GenerativeEngine
from genesis_cognitive.memory import MemoryEngine
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


def make_engine() -> tuple[CognitionEngine, ConceptNetwork]:
    data_dir = tempfile.mkdtemp(prefix="genesis_probe_")
    client = GenesisClient(os.path.join(data_dir, "genesis.sock"))
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
    return engine, network


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


def run_condition(name: str, experience: list[str]) -> dict[str, Any]:
    engine, network = make_engine()
    experience_results = []

    for statement in experience:
        response, state = engine.think(statement)
        experience_results.append({
            "input": statement,
            "response": response,
            "state": state_summary(state),
        })

    before = snapshot(network, engine)
    response, state = engine.think(TARGET)
    after = snapshot(network, engine)

    return {
        "condition": name,
        "experience": experience_results,
        "before_target": before,
        "target": {
            "input": TARGET,
            "response": response,
            "state": state_summary(state),
        },
        "after_target": after,
        "persistent_state_delta": {
            "concept_count": after["concept_count"] - before["concept_count"],
            "edge_count": after["edge_count"] - before["edge_count"],
            "conversation_turns": after["conversation_turns"] - before["conversation_turns"],
        },
    }


def main() -> None:
    report = {
        "probe": "learning-transfer-v1",
        "target": TARGET,
        "conditions": [
            run_condition("baseline", []),
            run_condition("relevant_experience", RELEVANT_EXPERIENCE),
            run_condition("irrelevant_experience", IRRELEVANT_EXPERIENCE),
        ],
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
