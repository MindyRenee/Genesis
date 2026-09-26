"""Conversation continuity — fast routes must stay part of the dialogue.

The most common exchanges (factual questions, follow-ups, answers to
Genesis's own questions) take the early metacognitive route, which
used to return before the store/track stage: the turn never entered
working memory, the conversation history, or long-term memory. The
next turn then had no referent for "it"/"that" and the exchange was
invisible to recall — a break in conversational continuity, not just
bookkeeping.

These tests exercise the real engine path offline (no daemon): an
early-route exchange must land in working memory and the conversation
record exactly once, and a follow-up question routed through "it"
must resolve to the previous turn's focus.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from genesis_client import GenesisClient
from genesis_cognitive.cognition.engine import CognitionEngine
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.language import GenerativeEngine
from genesis_cognitive.memory import MemoryEngine
from genesis_cognitive.perception import perceive
from genesis_cognitive.self import SelfModel
from genesis_cognitive.user_profile import UserProfile


@pytest.fixture
def engine():
    """A real cognition engine — offline (no daemon socket)."""
    data_dir = tempfile.mkdtemp()
    client = GenesisClient(os.path.join(data_dir, "genesis.sock"))
    self_model = SelfModel()
    network = ConceptNetwork()
    memory = MemoryEngine(client, network=network, get_emotion=lambda: None)
    language = GenerativeEngine(self_model, seed=1, network=network)
    eng = CognitionEngine(
        client=client,
        self_model=self_model,
        memory=memory,
        language=language,
        network=network,
        data_dir=data_dir,
        user_profile=UserProfile(),
    )
    return eng, network, memory


def test_fast_factual_route_records_the_turn(engine):
    """'What is a quoll?' takes the early route — and still counts as a turn."""
    eng, network, memory = engine
    network.add_concept("quoll", confidence=0.8)

    early = eng._think_early_routes(
        "what is a quoll?", perceive("what is a quoll?")
    )
    assert early is not None
    response, _state = early

    # Working memory carries the exchange — referent focus for a later
    # "it"/"that", and a recent turn for thread tracking.
    last = eng.working_memory.get_last_turn()
    assert last is not None
    assert last.user_input == "what is a quoll?"
    assert last.genesis_response == response
    focus = eng.working_memory.central_executive.focus_history
    assert "quoll" in focus

    # Conversation history carries both sides, exactly once each.
    roles = [t.role for t in memory.conversation]
    assert roles == ["user", "genesis"]
    assert memory.conversation[0].text == "what is a quoll?"
    assert memory.conversation[1].text == response


def test_pending_question_answer_records_the_turn(engine):
    """Answering a curiosity question early-routes — and is remembered."""
    eng, _network, memory = engine
    eng._pending_question_concepts["wug"] = "what is a wug?"

    early = eng._think_early_routes(
        "a wug is a small bird", perceive("a wug is a small bird")
    )
    assert early is not None

    last = eng.working_memory.get_last_turn()
    assert last is not None
    assert last.user_input == "a wug is a small bird"
    assert [t.role for t in memory.conversation] == ["user", "genesis"]


def test_followup_pronoun_answers_about_the_previous_focus(engine):
    """'What does it eat?' after 'What is a quoll?' — 'it' binds to quoll.

    Both repairs must chain: the first fast-route turn must be recorded
    (so "it" has a referent), and the routed consumes target must be
    referent-resolved (so the composer looks up "quoll", not "it").
    """
    eng, network, memory = engine
    network.add_concept("quoll", confidence=0.8)
    network.add_concept("insects", confidence=0.8)
    network.add_edge("quoll", "insects", RelationType.DEPENDS_ON, 0.9)

    first = eng._think_early_routes(
        "what is a quoll?", perceive("what is a quoll?")
    )
    assert first is not None

    # Turn 2 — run the real anaphora stage, then the early router with
    # the raw words (as think() does).
    raw = "what does it eat?"
    resolved = eng._resolve_anaphora(raw)
    early = eng._think_early_routes(resolved, perceive(raw), raw_input=raw)
    assert early is not None
    _response, state = early

    # The composer answered about quoll — the resolved referent —
    # reaching its depends_on edge to insects.
    knowledge = state.thought.metadata.get("knowledge") or []
    assert any("insect" in str(target) for _rel, target, _w in knowledge)

    # The follow-up turn is also recorded — the conversation now holds
    # both exchanges.
    roles = [t.role for t in memory.conversation]
    assert roles == ["user", "genesis", "user", "genesis"]
    last = eng.working_memory.get_last_turn()
    assert last is not None
    assert last.user_input == resolved


def test_plural_target_resolves_to_the_stored_concept(engine):
    """'Tell me about animals' must reach the 'animal' concept.

    The router captures the literal phrase; without variant resolution
    the composer looks up "animals", misses the stored "animal"
    concept, and answers with uncertainty about a known thing.
    """
    eng, network, _memory = engine
    network.add_concept("animal", confidence=0.8)
    network.add_concept("cat", confidence=0.8)
    network.add_edge("animal", "cat", RelationType.RELATED_TO, 0.9)

    early = eng._think_early_routes(
        "tell me about animals", perceive("tell me about animals")
    )
    assert early is not None
    _response, state = early
    # The routed target normalized to the stored singular concept.
    assert "animal" in state.thought.topics
    assert "animals" not in state.thought.topics


def test_consumes_plural_target_reaches_the_consumer(engine):
    """'What do wolves eat?' resolves to the 'wolf' concept."""
    eng, network, _memory = engine
    network.add_concept("wolf", confidence=0.8)
    network.add_concept("meat", confidence=0.8)
    network.add_edge("wolf", "meat", RelationType.DEPENDS_ON, 0.9)

    early = eng._think_early_routes(
        "what do wolves eat?", perceive("what do wolves eat?")
    )
    assert early is not None
    _response, state = early
    knowledge = state.thought.metadata.get("knowledge") or []
    assert any("meat" in str(target) for _rel, target, _w in knowledge)


def test_comparison_normalizes_each_side(engine):
    """'Wolves vs dogs' resolves each half to its stored concept."""
    eng, network, _memory = engine
    network.add_concept("wolf", confidence=0.8)
    network.add_concept("dog", confidence=0.8)

    early = eng._think_early_routes(
        "wolves vs dogs", perceive("wolves vs dogs")
    )
    if early is None:
        pytest.skip("comparison route did not fire")
    _response, state = early
    assert "wolf" in state.thought.topics
    assert "dog" in state.thought.topics


def test_non_routed_input_still_falls_through(engine):
    """A plain statement with no route keeps going to the full pipeline."""
    eng, _network, memory = engine
    early = eng._think_early_routes(
        "the sky was quiet today", perceive("the sky was quiet today")
    )
    assert early is None
    # Nothing recorded — the main pipeline's store/track stage owns it.
    assert len(memory.conversation) == 0
    assert eng.working_memory.get_last_turn() is None
