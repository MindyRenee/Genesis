"""Tests for the QuestionHandler subsystem.

Exercises question_relation_query (wh-question graph traversal),
question_tool_lookup (knowledge-metadata answers), and the
past-marker gate — with a real ConceptNetwork and ToolRegistry so the
traversal, direction, and resolution logic is genuinely exercised.
"""

from unittest.mock import MagicMock

from genesis_cognitive.cognition.question_handler import (
    _PAST_MARKER_RE,
    QuestionHandler,
)
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.memory.engine import MemoryContext
from genesis_cognitive.perception import Intent, Perception, QuestionType
from genesis_cognitive.self import SelfComposer, SelfModel
from genesis_cognitive.tools.framework import ToolRegistry

_VERB_MAP = {
    RelationType.CREATES: "creates",
    RelationType.CAUSES: "causes",
    RelationType.DEPENDS_ON: "depends on",
    RelationType.EMERGES_FROM: "emerges from",
    RelationType.RELATED_TO: "relates to",
    RelationType.ENABLES: "enables",
    RelationType.PART_OF: "is part of",
    RelationType.SPATIAL_RELATION: "is in",
}


def _network() -> ConceptNetwork:
    net = ConceptNetwork()
    for name in (
        "alice", "genesis", "cognition", "sleep", "memory",
        "food", "glip", "dreams", "emptycon",
    ):
        net.add_concept(name, confidence=0.8)
    net.add_edge("alice", "genesis", RelationType.CREATES, 0.95)
    net.add_edge("genesis", "cognition", RelationType.CREATES, 0.7)
    net.add_edge("sleep", "memory", RelationType.CAUSES, 0.8)
    # Stronger than the CAUSES edge but a different relation — used to
    # check that mixed-relation answers keep one dominant relation.
    net.add_edge("sleep", "dreams", RelationType.RELATED_TO, 0.9)
    net.add_edge("glip", "food", RelationType.DEPENDS_ON, 0.9)
    return net


def _handler(net: ConceptNetwork) -> QuestionHandler:
    return QuestionHandler(
        network=net,
        language=MagicMock(),
        composer=MagicMock(),
        self_composer=SelfComposer(),
        self_model=SelfModel(),
        theory_of_mind=MagicMock(),
        tools=ToolRegistry(),
        display_name=lambda cid: cid.replace("_", " "),
        relation_verb=lambda rel: _VERB_MAP.get(rel, str(rel)),
        resolve_topics=lambda topics, _raw: topics,
    )


def _perception(
    text: str, qtype: QuestionType, topics: list[str] | None = None
) -> Perception:
    return Perception(
        raw_text=text,
        intent=Intent.QUESTION,
        question_type=qtype,
        topics=topics or [],
        sentiment=0.0,
        sentiment_label="neutral",
        emotion_word="",
        is_about_genesis=False,
        is_about_user=False,
        is_about_code=False,
        is_about_emotion=False,
        is_about_existence=False,
        entities={},
        key_phrases=[],
        word_count=len(text.split()),
        confidence=0.8,
    )


def _emotion() -> EmotionalState:
    return EmotionalState(label="calm", cognitive_style="reflective")


def _relation_meta(handler: QuestionHandler, text: str, qtype: QuestionType):
    thought = handler.question_relation_query(_perception(text, qtype), _emotion())
    if thought is None:
        return None
    return thought.metadata.get("relation_answer")


# ─── Relation query: direction and verb mapping ──────────────────


def test_who_created_self_is_incoming():
    """'Who created you?' walks incoming CREATES edges to genesis."""
    meta = _relation_meta(_handler(_network()), "Who created you?", QuestionType.WHO)
    assert meta is not None
    assert meta["direction"] == "incoming"
    assert meta["objects"] == ["alice"]


def test_what_created_self_is_incoming():
    """'What created you?' is passive — subject is the object of the relation.

    Regression: this used to walk outgoing edges and answer what
    Genesis creates instead of what created it.
    """
    meta = _relation_meta(_handler(_network()), "What created you?", QuestionType.WHAT)
    assert meta is not None
    assert meta["direction"] == "incoming"
    assert meta["objects"] == ["alice"]


def test_what_did_subject_create_is_outgoing():
    """'What did Alice create?' — agent subject, base-form verb.

    Regression: did-questions produced base-form verbs that were
    missing from the verb table, so they always returned None.
    """
    meta = _relation_meta(_handler(_network()), "What did Alice create?", QuestionType.WHAT)
    assert meta is not None
    assert meta["direction"] == "outgoing"
    assert meta["objects"] == ["genesis"]


def test_who_did_subject_create_is_outgoing():
    meta = _relation_meta(_handler(_network()), "Who did Alice create?", QuestionType.WHO)
    assert meta is not None
    assert meta["direction"] == "outgoing"
    assert meta["objects"] == ["genesis"]


def test_what_does_subject_eat():
    meta = _relation_meta(_handler(_network()), "What does a glip eat?", QuestionType.WHAT)
    assert meta is not None
    assert meta["direction"] == "outgoing"
    assert meta["relation"] == "depends_on"
    assert meta["objects"] == ["food"]


def test_produced_passive_is_incoming():
    meta = _relation_meta(_handler(_network()), "What produced cognition?", QuestionType.WHAT)
    assert meta is not None
    assert meta["direction"] == "incoming"
    assert meta["objects"] == ["genesis"]


def test_negated_question_declined():
    """Negated questions ask about absences its edges can't enumerate."""
    handler = _handler(_network())
    assert _relation_meta(
        handler, "Why does sleep not cause memory?", QuestionType.WHY
    ) is None
    # "doesn't" (no space) must also be detected.
    assert _relation_meta(
        handler, "Why doesn't sleep affect memory?", QuestionType.WHY
    ) is None


def test_mixed_relations_keep_dominant():
    """When a verb maps to several relations, objects share one relation."""
    meta = _relation_meta(_handler(_network()), "What does sleep affect?", QuestionType.WHAT)
    assert meta is not None
    # Strongest edge is RELATED_TO (0.9) — the CAUSES edge must not be
    # mislabeled under the same verb.
    assert meta["relation"] == "related_to"
    assert meta["objects"] == ["dreams"]


def test_unknown_subject_returns_none():
    assert _relation_meta(
        _handler(_network()), "What does a zorblax eat?", QuestionType.WHAT
    ) is None


def test_no_question_type_returns_none():
    handler = _handler(_network())
    thought = handler.question_relation_query(
        _perception("alice created genesis", QuestionType.NONE), _emotion()
    )
    assert thought is None


# ─── Tool lookup: knowledge metadata, not pre-formatted text ─────


def test_tool_lookup_empty_concept_returns_none():
    """A concept with no definition/edges must not produce an empty answer."""
    handler = _handler(_network())
    thought = handler.question_tool_lookup(
        _perception("What is emptycon?", QuestionType.WHAT, ["emptycon"]),
        _emotion(),
    )
    assert thought is None


def test_tool_lookup_returns_knowledge_metadata():
    """The answer flows through the vocabulary's knowledge path."""
    handler = _handler(_network())
    thought = handler.question_tool_lookup(
        _perception("What is a glip?", QuestionType.WHAT, ["glip"]),
        _emotion(),
    )
    assert thought is not None
    assert thought.metadata.get("knowledge") == [("depends_on", "food", 0.9)]
    assert thought.metadata.get("topic") == "glip"


def test_tool_lookup_missing_concept_returns_none():
    handler = _handler(_network())
    thought = handler.question_tool_lookup(
        _perception("What is a zorblax?", QuestionType.WHAT, ["zorblax"]),
        _emotion(),
    )
    assert thought is None


# ─── Past-marker gate ────────────────────────────────────────────


def test_past_marker_regex_word_boundaries():
    """Substrings inside words must not trigger the memory path."""
    assert not _PAST_MARKER_RE.search("what is a dragon")
    assert not _PAST_MARKER_RE.search("tell me beforehand")
    assert _PAST_MARKER_RE.search("do you remember cats")
    assert _PAST_MARKER_RE.search("something from long ago")
    assert _PAST_MARKER_RE.search("what did i tell you")


def test_memory_retrieval_skips_non_memory_questions():
    """'What is a dragon?' must not hijack into episodic recall."""
    handler = _handler(_network())
    memory = MemoryContext(retrieved=[MagicMock()])
    thought = handler.question_memory_retrieval(
        _perception("What is a dragon?", QuestionType.WHAT, ["dragon"]),
        _emotion(),
        memory,
        lambda eid: MagicMock(text="a memory"),
    )
    assert thought is None


# ─── Creator questions ───────────────────────────────────────────


def test_creator_name_derived_from_network():
    """'who is <creator>' matches the discovered name, not a hardcoded one."""
    handler = _handler(_network())
    thought = handler.question_creator_personal("who is alice", _emotion())
    assert thought is not None
    assert thought.metadata.get("field") == "creator"
