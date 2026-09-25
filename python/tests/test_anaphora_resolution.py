"""Tests for ResponseStyler.resolve_anaphora.

Anaphora resolution rewrites user input before comprehension and
learning — a wrong rewrite is learned verbatim as knowledge. The
live-session regression: "a ferret is a small mammal that likes to
steal shiny objects" was rewritten to "…mammal genesis likes to
steal shiny objects" because the relative pronoun "that" was treated
as anaphoric and replaced with the discourse focus.

These tests pin down the distinction between pronouns that refer to
the focus (demonstrative "that is cool", bare "it"/"they") and
pronouns that do grammatical work inside the sentence (relative
"that"/"which"/"who", complementizer "that", determiner "that dog",
zero-relative "the pet you want", expletive "it is raining").
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_cognitive.cognition.response_styler import ResponseStyler
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.concepts.types import ConceptCategory
from genesis_cognitive.memory import WorkingMemory


def _make_styler(
    focus: str | None = None,
    foci: list[str] | None = None,
    network: ConceptNetwork | None = None,
) -> ResponseStyler:
    """Build a styler with a real WorkingMemory at the given focus."""
    wm = WorkingMemory()
    for f in foci or ([] if focus is None else [focus]):
        wm.central_executive.switch_task(f)
    return ResponseStyler(
        language=None,  # type: ignore[arg-type]
        composer=None,  # type: ignore[arg-type]
        working_memory=wm,
        rng=None,
        response_style_getter=lambda: "neutral",
        network=network,
    )


# ─── Relative clauses must survive untouched ────────────────────


def test_relative_that_subject_of_clause_not_rewritten() -> None:
    """'that' as subject of a relative clause is not anaphoric."""
    styler = _make_styler("genesis")
    out = styler.resolve_anaphora(
        "a ferret is a small mammal that likes to steal shiny objects"
    )
    assert out == "a ferret is a small mammal that likes to steal shiny objects"


def test_relative_that_auxiliary_not_rewritten() -> None:
    """'that' followed by an auxiliary still heads a relative clause."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora(
        "a bird that can fly is neat"
    ) == "a bird that can fly is neat"
    assert styler.resolve_anaphora(
        "a fox is a canine that hunts mice"
    ) == "a fox is a canine that hunts mice"
    assert styler.resolve_anaphora(
        "the cat that lives here is friendly"
    ) == "the cat that lives here is friendly"


def test_relative_which_who_where_when_not_rewritten() -> None:
    """'which'/'who'/'where'/'when' are never focus pronouns."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora(
        "a cat which hunts mice is useful"
    ) == "a cat which hunts mice is useful"
    assert styler.resolve_anaphora(
        "the person who called earlier"
    ) == "the person who called earlier"
    assert styler.resolve_anaphora(
        "the place where we met"
    ) == "the place where we met"
    assert styler.resolve_anaphora(
        "the day when everything changed"
    ) == "the day when everything changed"


# ─── Complementizer 'that' must survive untouched ───────────────


def test_complementizer_that_not_rewritten() -> None:
    """'that' introducing a complement clause is not anaphoric."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora(
        "i think that ferrets are cute"
    ) == "i think that ferrets are cute"
    # "she"/"he" bind only to animate referents — with no animate
    # entity in focus history they stay unresolved. What must survive
    # is the complementizer "that" and the expletive "it was late".
    assert styler.resolve_anaphora(
        "she said that it was late"
    ) == "she said that it was late"
    out = styler.resolve_anaphora("i told you that he left")
    assert " that " in out and "told Genesis that" in out


def test_complementizer_that_after_pronoun_object() -> None:
    """'tell me that it matters' — 'that' stays, clause intact."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora(
        "tell me that it matters"
    ) == "tell me that it matters"


# ─── Determiners and zero-relatives must survive untouched ──────


def test_determiner_that_this_not_rewritten() -> None:
    """'that dog' / 'this concept' — demonstrative determiners."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora("that dog is big") == "that dog is big"
    assert styler.resolve_anaphora(
        "this concept is interesting"
    ) == "this concept is interesting"


def test_zero_relative_you_not_rewritten() -> None:
    """'you' inside a zero-relative clause is generic, not Genesis."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora(
        "a pet you can keep"
    ) == "a pet you can keep"
    assert styler.resolve_anaphora(
        "the thing you said"
    ) == "the thing you said"


# ─── Expletive 'it' must survive untouched ──────────────────────


def test_expletive_it_not_rewritten() -> None:
    """Weather/time/evaluative 'it' refers to nothing."""
    styler = _make_styler("genesis")
    assert styler.resolve_anaphora("it is raining") == "it is raining"
    assert styler.resolve_anaphora("it looks good") == "it looks good"
    assert styler.resolve_anaphora("it was late") == "it was late"


# ─── Genuine anaphora still resolve ─────────────────────────────


def test_demonstrative_that_resolves_to_focus() -> None:
    """Sentence-initial 'that is…' resolves to the focus."""
    styler = _make_styler("ferret")
    assert styler.resolve_anaphora("that is interesting") == (
        "ferret is interesting"
    )


def test_demonstrative_object_position_resolves() -> None:
    """'that' after a verb or preposition resolves to the focus."""
    styler = _make_styler("ferret")
    assert styler.resolve_anaphora("i like that") == "i like ferret"
    assert styler.resolve_anaphora("tell me about that") == (
        "tell me about ferret"
    )


def test_bare_pronouns_resolve_to_focus() -> None:
    """'it'/'they' in argument position resolve to the focus."""
    styler = _make_styler("ferret")
    assert styler.resolve_anaphora("it runs fast") == "ferret runs fast"
    assert styler.resolve_anaphora("i saw it") == "i saw ferret"


def test_you_your_resolve_to_genesis() -> None:
    """'you'/'your' address Genesis directly."""
    styler = _make_styler("ferret")
    assert styler.resolve_anaphora("are you awake") == "are Genesis awake"
    assert styler.resolve_anaphora("what is your name") == (
        "what is Genesis's name"
    )


def test_no_focus_leaves_text_unchanged() -> None:
    """With no attentional focus, nothing is substituted."""
    styler = _make_styler(None)
    assert styler.resolve_anaphora("it is fascinating") == "it is fascinating"


def test_no_pronouns_leaves_text_unchanged() -> None:
    """Plain factual input is passed through byte-identical."""
    styler = _make_styler("genesis")
    text = "ferrets are small mammals"
    assert styler.resolve_anaphora(text) == text


# ─── Referent typing: animacy and focus history ────────────────


def _network_with_animacy() -> ConceptNetwork:
    """ferret → is_a → mammal (LIVING); wheel is NON_LIVING."""
    net = ConceptNetwork()
    net.add_concept("ferret", confidence=0.8)
    net.add_concept("wheel", confidence=0.8)
    net.add_concept("mammal", confidence=0.8)
    net.add_edge("ferret", "mammal", RelationType.IS_A)
    mammal = net.get_concept("mammal")
    wheel = net.get_concept("wheel")
    assert mammal is not None and wheel is not None
    mammal.category = ConceptCategory.LIVING
    wheel.category = ConceptCategory.NON_LIVING
    return net


def test_it_binds_to_newest_non_participant() -> None:
    """'it' resolves to the newest focus that isn't a discourse
    participant — 'i saw it' after focus moved to wheel."""
    styler = _make_styler(foci=["ferret", "wheel"])
    assert styler.resolve_anaphora("i saw it") == "i saw wheel"


def test_it_skips_genesis_focus() -> None:
    """'it' must not resolve to genesis — self is 'I', not 'it'."""
    styler = _make_styler(foci=["ferret", "genesis"])
    assert styler.resolve_anaphora("i saw it") == "i saw ferret"


def test_she_binds_to_animate_referent() -> None:
    """'she' walks history to the first animate entity, skipping
    inanimate foci like 'wheel'."""
    styler = _make_styler(
        foci=["ferret", "wheel"], network=_network_with_animacy()
    )
    assert styler.resolve_anaphora("she left") == "ferret left"


def test_she_unresolved_without_animate() -> None:
    """'she' with only inanimate foci stays unresolved — better than
    binding a gendered pronoun to a wheel."""
    styler = _make_styler(foci=["wheel"], network=_network_with_animacy())
    assert styler.resolve_anaphora("she left") == "she left"


def test_they_accepts_inanimate_fallback() -> None:
    """'they' prefers animate but accepts plural-inanimate referents."""
    styler = _make_styler(foci=["wheel"], network=_network_with_animacy())
    assert styler.resolve_anaphora("they fell off") == "wheel fell off"
