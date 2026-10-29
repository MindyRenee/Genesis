"""Tests for the QuestionHandler subsystem.

Exercises question_relation_query (wh-question graph traversal),
question_tool_lookup (knowledge-metadata answers), and the
past-marker gate — with a real ConceptNetwork and ToolRegistry so the
traversal, direction, and resolution logic is genuinely exercised.
"""

from unittest.mock import MagicMock

from frontal_lobe.question_handler import (
    _PAST_MARKER_RE,
    QuestionHandler,
)
from frontal_lobe.reasoning import Judge
from limbic_system.emotion import EmotionalState
from memory.memory import MemoryContext
from occipital_lobe.perception import Intent, Perception, QuestionType
from parietal_lobe.correspondence import CorrespondenceLedger
from parietal_lobe.situation_model import SituationModel
from self import SelfComposer, SelfModel
from temporal_lobe.base import Thought
from temporal_lobe.comprehension import Comprehension
from temporal_lobe.concepts import ConceptNetwork, RelationType
from tools.framework import ToolRegistry

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
        "alice",
        "genesis",
        "cognition",
        "sleep",
        "memory",
        "food",
        "glip",
        "dreams",
        "emptycon",
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


def _perception(text: str, qtype: QuestionType, topics: list[str] | None = None) -> Perception:
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


# ─── Anaphora-resolved text reaches the relation walk ───────────


def test_relation_query_parses_resolved_text():
    """'What does it eat?' — 'it' was bound to the focus upstream.

    Perception stores the user's literal words, so the referent
    exists only in the resolved utterance; the relation query must
    parse that form, not the pronoun.
    """
    handler = _handler(_network())
    thought = handler.question_relation_query(
        _perception("What does it eat?", QuestionType.WHAT),
        _emotion(),
        resolved_text="What does a glip eat?",
    )
    assert thought is not None
    meta = thought.metadata.get("relation_answer")
    assert meta is not None
    assert meta["objects"] == ["food"]


def test_relation_query_unresolved_pronoun_finds_nothing():
    """Without resolved_text, 'it' is not a concept — no relation answer."""
    handler = _handler(_network())
    thought = handler.question_relation_query(
        _perception("What does it eat?", QuestionType.WHAT),
        _emotion(),
    )
    assert thought is None


def test_handle_question_threads_resolved_text():
    """handle_question passes resolved_text through to the relation walk."""
    handler = _handler(_network())
    thought = handler.handle_question(
        _perception("What does it eat?", QuestionType.WHAT),
        _emotion(),
        MemoryContext(),
        resolved_text="What does a glip eat?",
    )
    meta = thought.metadata.get("relation_answer")
    assert meta is not None
    assert meta["objects"] == ["food"]


# ─── Memory enrichment ─────────────────────────────────────────


def _memory_context(ep_text: str):
    """A MemoryContext with one highly-relevant episode + retriever."""
    from types import SimpleNamespace

    from genesis_client.types import SimilarEpisode

    ctx = MemoryContext()
    ctx.retrieved.append(
        SimilarEpisode(episode_id=7, hamming_distance=5, timestamp=1, salience=0.9)
    )
    episode = SimpleNamespace(text=ep_text)
    return ctx, lambda _eid: episode


def _bare_thought() -> Thought:
    return Thought(
        content="mammal",
        intent="inform",
        emotion="calm",
        confidence=0.9,
        topics=["mammal"],
    )


def test_enrich_with_memory_rejects_questions():
    """A recalled question is not knowledge to weave into an answer.

    Live regression: her own "I'm not sure about quoll tell me more?"
    reply was stored, recalled, and spoken back as "I recall ...".
    """
    handler = _handler(_network())
    memory, retrieve = _memory_context(
        "User: what is a quoll? | Genesis: I'm not sure about quoll tell me more?"
    )
    thought = handler._enrich_with_memory(
        _bare_thought(),
        _perception("tell me about mammals", QuestionType.NONE),
        memory,
        retrieve,
    )
    assert "memory" not in thought.metadata


def test_enrich_with_memory_prefers_user_fact():
    """Enrichment recalls what the user taught, not Genesis's reply."""
    handler = _handler(_network())
    memory, retrieve = _memory_context(
        "User: a quoll eats insects | Genesis: Quoll requires insect."
    )
    thought = handler._enrich_with_memory(
        _bare_thought(),
        _perception("tell me about quolls", QuestionType.NONE),
        memory,
        retrieve,
    )
    assert thought.metadata["memory"] == "a quoll eats insects"


def test_enrich_with_memory_named_speaker():
    """A turn recorded under the speaker's name parses like 'User:'.

    Conversation memories are attributed to whoever she believes is
    speaking ("Mindy: ... | Genesis: ..."), so enrichment must lift
    the named speaker's side, not only the generic label's.
    """
    handler = _handler(_network())
    memory, retrieve = _memory_context(
        "Mindy: a quoll eats insects | Genesis: Quoll requires insect."
    )
    thought = handler._enrich_with_memory(
        _bare_thought(),
        _perception("tell me about quolls", QuestionType.NONE),
        memory,
        retrieve,
    )
    assert thought.metadata["memory"] == "a quoll eats insects"


def test_named_speaker_side_not_recalled_as_genesis_reply():
    """The user's named turn must not leak into Genesis's side."""
    handler = _handler(_network())
    _, retrieve = _memory_context(
        "Mindy: tell me everything you know about quolls | Genesis: Quoll requires insect."
    )
    ep = retrieve(7)
    assert handler._extract_genesis_response(ep.text) == "Quoll requires insect."


def test_produced_passive_is_incoming():
    meta = _relation_meta(_handler(_network()), "What produced cognition?", QuestionType.WHAT)
    assert meta is not None
    assert meta["direction"] == "incoming"
    assert meta["objects"] == ["genesis"]


def test_negated_question_declined():
    """Negated questions ask about absences its edges can't enumerate."""
    handler = _handler(_network())
    assert _relation_meta(handler, "Why does sleep not cause memory?", QuestionType.WHY) is None
    # "doesn't" (no space) must also be detected.
    assert _relation_meta(handler, "Why doesn't sleep affect memory?", QuestionType.WHY) is None


def test_mixed_relations_keep_dominant():
    """When a verb maps to several relations, objects share one relation."""
    meta = _relation_meta(_handler(_network()), "What does sleep affect?", QuestionType.WHAT)
    assert meta is not None
    # Strongest edge is RELATED_TO (0.9) — the CAUSES edge must not be
    # mislabeled under the same verb.
    assert meta["relation"] == "related_to"
    assert meta["objects"] == ["dreams"]


def test_unknown_subject_returns_none():
    assert (
        _relation_meta(_handler(_network()), "What does a zorblax eat?", QuestionType.WHAT) is None
    )


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


# ─── Polar questions: judgment against the situation model ──────


def _polar_handler() -> tuple[QuestionHandler, SituationModel, Comprehension]:
    """A handler wired like production: real model, ledger, judge.

    The story is fed through real parses, so the verdict exercises
    the same proposition→claim→bearing chain the live path does.
    """
    model = SituationModel()
    handler = _handler(_network())
    handler._situation = model
    handler._judge = Judge(CorrespondenceLedger(model))
    return handler, model, Comprehension()


def _polar_meta(handler: QuestionHandler, comp: Comprehension, text: str):
    result = comp.comprehend(text)
    thought = handler.question_situation(
        _perception(text, QuestionType.YES_NO),
        _emotion(),
        result.propositions,
    )
    if thought is None:
        return None
    return thought.metadata.get("situation_answer")


def test_polar_supported_claim_hedges_then_asserts():
    """The first check cannot license assertion; the second can."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("Mary is in the kitchen."))

    first = _polar_meta(handler, comp, "is Mary in the kitchen?")
    assert first is not None
    assert first["kind"] == "verdict_hedge"
    assert first["stance"] == "hedge"
    assert first["claim"] == "mary is in the kitchen"

    # Checking again is more evidence — the Beta posterior accumulates
    # per claim, so the second answer commits.
    second = _polar_meta(handler, comp, "is Mary in the kitchen?")
    assert second["kind"] == "verdict_assert"
    assert second["stance"] == "assert"


def test_polar_contradicted_claim_doubts_then_denies():
    """A claim the model affirms the negation of leans false first."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("Mary is in the kitchen."))

    first = _polar_meta(handler, comp, "is Mary in the bathroom?")
    assert first is not None
    assert first["kind"] == "verdict_doubt"

    second = _polar_meta(handler, comp, "is Mary in the bathroom?")
    assert second["kind"] == "verdict_deny"


def test_polar_equivocal_claim_suspends():
    """An either/or discourse is equivocal bearing — judgment holds."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("John is either in the bathroom or the den."))

    meta = _polar_meta(handler, comp, "is John in the bathroom?")
    assert meta is not None
    assert meta["kind"] == "verdict_suspend"


def test_polar_unknown_claim_suspends():
    """An entity the discourse never mentioned gives no bearing."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("Mary is in the kitchen."))

    meta = _polar_meta(handler, comp, "is the dog in the den?")
    assert meta is not None
    assert meta["kind"] == "verdict_suspend"


def test_polar_possession_question_judged():
    """'does John have the ball?' checks the carries claim 'has' settled."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("John has the ball."))

    meta = _polar_meta(handler, comp, "does John have the ball?")
    assert meta is not None
    assert meta["kind"] == "verdict_hedge"
    assert meta["claim"] == "john has ball"


def test_polar_without_judge_falls_through():
    """No assertion gate → no verdict; the question routes onward."""
    model = SituationModel()
    comp = Comprehension()
    handler = _handler(_network())
    handler._situation = model
    model.update(comp.comprehend("Mary is in the kitchen."))

    result = comp.comprehend("is Mary in the kitchen?")
    thought = handler.question_situation(
        _perception("is Mary in the kitchen?", QuestionType.YES_NO),
        _emotion(),
        result.propositions,
    )
    assert thought is None


def test_polar_without_propositions_falls_through():
    """Without the parsed clauses there is no radical to judge."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("Mary is in the kitchen."))

    thought = handler.question_situation(
        _perception("is Mary in the kitchen?", QuestionType.YES_NO),
        _emotion(),
    )
    assert thought is None


def test_wh_question_not_hijacked_by_polar_path():
    """A wh-question carries its own focus — never the whether-gap."""
    handler, model, comp = _polar_handler()
    model.update(comp.comprehend("Mary is in the kitchen."))

    result = comp.comprehend("where is Mary?")
    thought = handler.question_situation(
        _perception("where is Mary?", QuestionType.WHERE),
        _emotion(),
        result.propositions,
    )
    assert thought is not None
    assert thought.metadata["situation_answer"]["kind"] == "location"
