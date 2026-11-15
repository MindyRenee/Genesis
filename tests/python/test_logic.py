"""Formal-logic tests — the model-theoretic engine: deduction via
mental-model enumeration, AGM revision, Ramsey conditionals,
minimal abduction, and categorical syllogisms."""

import pytest
from frontal_lobe.reasoning.logic import (
    MAX_ATOMS,
    Categorical,
    Quantifier,
    all_of,
    and_,
    atom,
    contract,
    entails,
    expand,
    explain,
    iff,
    implies,
    models,
    not_,
    or_,
    parse,
    ramsey,
    revise,
    satisfiable,
    syllogism,
    verify,
)


def P(text: str):
    return parse(text)


# ---------------------------------------------------------------
# Model theory — deduction as model enumeration
# ---------------------------------------------------------------


def test_modus_ponens_valid():
    d = verify([P("p -> q"), P("p")], P("q"))
    assert d.valid and d.countermodel is None


def test_modus_tollens_valid():
    assert verify([P("p -> q"), P("~q")], P("~p")).valid


def test_disjunctive_syllogism_valid():
    assert verify([P("p | q"), P("~p")], P("q")).valid


def test_hypothetical_syllogism_valid():
    assert verify([P("a -> b"), P("b -> c")], P("a -> c")).valid


def test_affirming_consequent_invalid_with_countermodel():
    d = verify([P("p -> q"), P("q")], P("p"))
    assert not d.valid
    assert d.countermodel == {"p": False, "q": True}
    # the countermodel really is one: premises hold, conclusion fails
    assert all(f.evaluate(d.countermodel) for f in d.premises)


def test_denying_antecedent_invalid():
    assert not verify([P("p -> q"), P("~p")], P("~q")).valid


def test_entails_tautology_from_empty_kb():
    assert entails([], P("p | ~p"))
    assert not entails([], P("p"))


def test_satisfiable_and_models_enumeration():
    fs = [P("a | b"), P("a -> b")]
    assert satisfiable(fs)
    found = list(models(fs))
    # models: (F,T), (T,T) — (T,F) violates a->b, (F,F) violates a|b
    assert len(found) == 2
    assert len(list(models(fs, limit=1))) == 1


def test_inconsistent_kb_has_no_models():
    assert not satisfiable([P("a"), P("~a"), P("b")])


def test_atom_bound_raises():
    f = all_of(atom(f"x{i}") for i in range(MAX_ATOMS + 1))
    with pytest.raises(ValueError):
        list(models([f]))


# ---------------------------------------------------------------
# Parser — precedence, associativity, surface forms
# ---------------------------------------------------------------


def test_parse_precedence_and_before_or():
    f = P("a & b | c")
    assert f.evaluate({"a": True, "b": True, "c": False})
    assert not f.evaluate({"a": True, "b": False, "c": False})
    assert f.evaluate({"a": False, "b": False, "c": True})


def test_parse_implication_right_associative():
    f = P("a -> b -> c")
    inner = implies(atom("b"), atom("c"))
    assert f == implies(atom("a"), inner)


def test_parse_iff_and_negation():
    assert P("a <-> ~b").evaluate({"a": True, "b": False})
    assert not P("a <-> ~b").evaluate({"a": True, "b": True})
    assert P("¬a ∨ b").evaluate({"a": True, "b": True})


def test_parse_namespaced_atoms_and_constants():
    f = P("concept:bird & ~exception:penguin -> trait:flies")
    assert f.evaluate({"concept:bird": True, "exception:penguin": False, "trait:flies": True})
    assert P("true").evaluate({})
    assert not P("false").evaluate({})


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        P("a -> ")
    with pytest.raises(ValueError):
        P("(a & b")
    with pytest.raises(ValueError):
        P("a b")


# ---------------------------------------------------------------
# Formula algebra — double negation, constants, flattening
# ---------------------------------------------------------------


def test_double_negation_collapses():
    p = atom("p")
    assert not_(not_(p)) == p


def test_constants_absorb():
    p = atom("p")
    from frontal_lobe.reasoning.logic import BOTTOM, TOP

    assert and_(p, TOP) == p
    assert or_(p, BOTTOM) == p
    assert and_(p, BOTTOM) is BOTTOM
    assert or_(p, TOP) is TOP


def test_iff_semantics():
    f = iff(atom("a"), atom("b"))
    assert f.evaluate({"a": True, "b": True})
    assert f.evaluate({"a": False, "b": False})
    assert not f.evaluate({"a": True, "b": False})


# ---------------------------------------------------------------
# AGM — expansion, contraction, revision
# ---------------------------------------------------------------


def test_expand_adds_belief():
    k = expand([], P("p"))
    assert P("p") in k
    assert entails(k, P("p"))


def test_contract_vacuity_returns_kb_unchanged():
    k = frozenset({P("a")})
    assert contract(k, P("b")) == k  # k never entailed b


def test_contract_removes_minimal_justification():
    k = frozenset({P("p"), P("p -> q")})
    out = contract(k, P("q"))
    assert len(out) == 1  # one minimal surrender suffices
    assert not entails(out, P("q"))


def test_contract_respects_entrenchment():
    k = frozenset({P("p"), P("p -> q")})
    out = contract(k, P("q"), entrenchment={P("p -> q"): 10.0})
    # the entrenched rule survives; the naked fact is surrendered
    assert out == frozenset({P("p -> q")})


def test_contract_tautology_is_impossible():
    k = frozenset({P("a")})
    assert contract(k, P("b | ~b")) == k


def test_revise_levi_identity_consistent_and_successful():
    k = frozenset({P("p"), P("p -> q")})
    out = revise(k, P("~q"))
    assert P("~q") in out  # Success: the input is believed
    assert satisfiable(out)  # Consistency
    assert entails(out, P("~q"))


def test_revise_prefers_surrendering_least_entrenched():
    k = frozenset({P("p"), P("p -> q")})
    out = revise(k, P("~q"), entrenchment={P("p -> q"): 10.0})
    # drops p rather than the rule → modus tollens remains live
    assert P("p -> q") in out
    assert entails(out, P("~p"))


# ---------------------------------------------------------------
# Ramsey — conditionals and counterfactuals via revision
# ---------------------------------------------------------------


def test_ramsey_conditional_acceptance():
    k = frozenset({P("bird -> flies")})
    assert ramsey(k, P("bird"), P("flies"))
    assert not ramsey(k, P("rock"), P("flies"))


def test_ramsey_nonmonotonic_exception():
    # penguins are birds that don't fly — the classic defeater
    k = frozenset({P("bird -> flies"), P("penguin -> bird"), P("penguin -> ~flies")})
    assert ramsey(k, P("penguin"), P("~flies"))
    # note: the bare "bird -> flies" default yields both in the naive
    # set; revision keeps consistency, and ~flies is entailed
    out = revise(k, P("penguin"))
    assert satisfiable(out)


def test_ramsey_counterfactual_on_false_antecedent():
    # "the light is off" believed; "if it were on, the room would be lit"
    k = frozenset({P("~on"), P("on -> lit")})
    assert ramsey(k, P("on"), P("lit"))


# ---------------------------------------------------------------
# Abduction — minimal consistent explanans
# ---------------------------------------------------------------


def test_explain_picks_single_sufficient_cause():
    k = frozenset({P("sprinkler -> wet"), P("rain -> wet")})
    out = explain(k, P("wet"), [P("sprinkler"), P("rain"), P("windy")])
    assert out is not None and len(out) == 1
    assert next(iter(out)) in (P("sprinkler"), P("rain"))


def test_explain_returns_none_when_nothing_explains():
    k = frozenset({P("a -> b")})
    assert explain(k, P("z"), [P("a"), P("b")]) is None


def test_explain_already_entailed_is_empty():
    assert explain([P("q")], P("q"), [P("r")]) == frozenset()


def test_explain_rejects_inconsistent_explanans():
    k = frozenset({P("a -> b"), P("c -> ~b")})
    # {a,c} would entail b∧~b — inconsistent, so a size-2 answer must
    # be one consistent subset… here both singletons suffice anyway
    out = explain(k, P("b"), [P("a"), P("c")])
    assert out == frozenset({P("a")})


# ---------------------------------------------------------------
# Syllogisms — Venn-region semantics
# ---------------------------------------------------------------


def Cat(q, a, b):
    return Categorical(q, a, b)


def test_barbara_valid():
    d = syllogism(
        Cat(Quantifier.ALL, "m", "p"),
        Cat(Quantifier.ALL, "s", "m"),
        Cat(Quantifier.ALL, "s", "p"),
    )
    assert d.valid


def test_celarent_valid():
    assert syllogism(
        Cat(Quantifier.NO, "m", "p"),
        Cat(Quantifier.ALL, "s", "m"),
        Cat(Quantifier.NO, "s", "p"),
    ).valid


def test_particular_form_valid():
    # All M are P ∧ Some S are M ⊨ Some S are P
    assert syllogism(
        Cat(Quantifier.ALL, "m", "p"),
        Cat(Quantifier.SOME, "s", "m"),
        Cat(Quantifier.SOME, "s", "p"),
    ).valid


def test_illicit_middle_invalid():
    # All A are B ∧ All C are B ⊭ All A are C (the classic fallacy)
    d = syllogism(
        Cat(Quantifier.ALL, "a", "b"),
        Cat(Quantifier.ALL, "c", "b"),
        Cat(Quantifier.ALL, "a", "c"),
    )
    assert not d.valid
    assert "inhabited_regions" in d.countermodel


def test_darapti_invalid_without_existential_import():
    # All M are P ∧ All M are S ⊭ Some S are P — valid for Aristotle,
    # invalid in modern semantics since M may be uninhabited
    d = syllogism(
        Cat(Quantifier.ALL, "m", "p"),
        Cat(Quantifier.ALL, "m", "s"),
        Cat(Quantifier.SOME, "s", "p"),
    )
    assert not d.valid


def test_syllogism_requires_three_terms():
    with pytest.raises(ValueError):
        syllogism(
            Cat(Quantifier.ALL, "a", "b"),
            Cat(Quantifier.ALL, "b", "a"),
            Cat(Quantifier.ALL, "a", "b"),
        )
