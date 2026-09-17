"""Tests for bonded-user recognition (public/day-1 semantics).

The public system seeds no creator name. Users are learned from
introductions and grounded as *people*, never as creators. Creator
facts come only from explicit teaching (an X CREATES genesis edge).
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
from genesis_cognitive.self import SelfModel


@pytest.fixture
def engine():
    """A minimal cognition engine — no daemon required."""
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
    )
    return eng, self_model, network, memory


def test_no_seeded_creator_name(engine):
    """Day 1: no person is claimed as creator — only the abstract role."""
    eng, self_model, network, _ = engine
    creator_name = eng.self_composer._discover_creator_name(self_model, network)
    assert creator_name == ""


def test_introduction_learns_user_not_creator(engine):
    """'My name is Sam' grounds Sam as a person, not a creator."""
    eng, self_model, network, memory = engine
    assert eng._recognize_bonded_user("hi, my name is Sam") is True

    assert self_model.self_knowledge["user_name"] == "Sam"
    assert "Sam" in self_model.self_knowledge["known_users"]
    assert memory.get_user_facts()["name"] == "Sam"
    assert any(
        "introduced themselves" in n for n in self_model.relationship_notes
    )

    rels = {e.relation for e in network.get_edges("sam", "out")}
    assert RelationType.IS_A in rels
    assert RelationType.RELATED_TO in rels
    assert RelationType.CREATES not in rels


def test_known_user_recognized_by_name(engine):
    """Once introduced, mentioning the user by name re-triggers the bond."""
    eng, *_ = engine
    eng._recognize_bonded_user("my name is Sam")
    assert eng._recognize_bonded_user("sam is here") is True
    assert eng._recognize_bonded_user("what do you think, Sam?") is True


def test_multiple_users_all_known(engine):
    """Many people can introduce themselves; all are remembered."""
    eng, self_model, *_ = engine
    eng._recognize_bonded_user("my name is Sam")
    eng._recognize_bonded_user("my name is Priya")
    assert set(self_model.self_knowledge["known_users"]) == {"Sam", "Priya"}
    assert eng._recognize_bonded_user("priya asked a question") is True
    assert eng._recognize_bonded_user("sam agrees") is True


def test_non_name_words_not_learned(engine):
    """'I'm happy/tired/fine' must not ground a bogus user."""
    eng, self_model, *_ = engine
    for phrase in ("i am happy", "i'm tired", "i am fine", "i'm not sure"):
        assert eng._recognize_bonded_user(phrase) is False
    assert "user_name" not in self_model.self_knowledge


def test_unknown_person_not_recognized(engine):
    """No bond response for names that were never introduced."""
    eng, *_ = engine
    assert eng._recognize_bonded_user("alice did this") is False


def test_taught_creator_is_discovered(engine):
    """An explicit X-CREATES-genesis edge (from teaching) is discoverable."""
    eng, self_model, network, _ = engine
    network.add_concept("ada")
    network.add_edge("ada", "genesis", RelationType.CREATES, 0.9)
    name = eng.self_composer._discover_creator_name(self_model, network)
    assert name == "ada"
