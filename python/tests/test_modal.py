"""Tests for the Lewisian modal cluster — worlds, counterparts,
supervenience, conceivability."""

from frontal_lobe.reasoning.logic import and_, atom, implies, not_, or_
from frontal_lobe.reasoning.modal import (
    Individual,
    conceivable,
    contingent,
    could,
    counterparts,
    necessarily,
    pluriverse,
    possibly,
    recombine,
    recombining,
    status,
    supervenes,
)


def test_possibility_and_necessity():
    """◇ holds in some world; □ holds in all."""
    worlds = pluriverse(["p", "q"])
    assert possibly(atom("p"), worlds)
    assert necessarily(or_(atom("p"), not_(atom("p"))), worlds)
    assert not necessarily(atom("p"), worlds)
    assert contingent(atom("p"), worlds)


def test_constraints_cut_the_pluriverse():
    """A theory narrows logical space to metaphysical possibility —
    the worlds where its laws hold."""
    law = implies(atom("spark"), atom("fire"))
    all_w = pluriverse(["spark", "fire"])
    lawful = pluriverse(["spark", "fire"], [law])
    assert len(lawful) < len(all_w)
    # The spark-without-fire world is logically possible, not lawful.
    assert possibly(and_(atom("spark"), not_(atom("fire"))), all_w)
    assert not possibly(and_(atom("spark"), not_(atom("fire"))), lawful)


def test_counterparts_select_by_resemblance():
    """The counterpart is what most resembles the original — and
    nothing in that world resembles it more."""
    x = Individual("socrates", frozenset({"snub", "wise", "bald"}), "w0")
    w1 = [
        Individual("twin", frozenset({"snub", "wise", "bald", "ugly"}), "w1"),
        Individual("other", frozenset({"tall", "swift"}), "w1"),
    ]
    cps = counterparts(x, w1)
    assert {c.name for c in cps} == {"twin"}


def test_de_re_possibility_through_counterparts():
    """'x could have been F' iff some world has an F-counterpart."""
    x = Individual("socrates", frozenset({"snub", "wise"}), "w0")
    worlds = [
        [Individual("a", frozenset({"snub", "wise", "snub-nosed"}), "w1")],
        [Individual("b", frozenset({"snub", "wise", "athletic"}), "w2")],
    ]
    assert could(x, "athletic", worlds)
    assert not could(x, "purple", worlds)


def test_counterparthood_is_not_transitive():
    """Resemblance degrades stepwise — the signature of counterpart
    semantics: y like x, z like y, z nothing like x."""
    x = Individual("x", frozenset({"a", "b"}))
    mid = [Individual("y", frozenset({"a", "b", "c", "d"}))]
    far = [Individual("z", frozenset({"c", "d", "e", "f"}))]
    # y counterparts x (shares a,b); z counterparts y (shares c,d);
    # z shares no quality with x — no counterpart there.
    ys = counterparts(x, mid)
    zs = counterparts(next(iter(ys)), far)
    assert zs and next(iter(zs)).name == "z"
    assert "z" not in {c.name for c in counterparts(x, far)}


def test_recombination_patches_worlds():
    """Grafting qualities between worlds yields another world —
    anything can coexist with anything."""
    w1 = {"wet": True, "cold": True}
    w2 = {"wet": False, "cold": False}
    assert recombine(w1, w2, ["wet"]) == {"wet": True, "cold": False}


def test_recombination_closure_is_the_pluriverse():
    """Free recombination of the qualities reaches the space of
    worlds — each atom independently taking a value seen in some
    source. One world alone recombines to nothing new; two worlds
    that differ on every atom generate the full pluriverse."""
    assert len(recombining([{"a": True, "b": False}], ["a", "b"])) == 1
    closed = recombining([{"a": True, "b": False}, {"a": False, "b": True}], ["a", "b"])
    assert len(closed) == 4  # every assignment over {a,b}


def test_supervenience_no_difference_without_base_difference():
    """B supervenes on A iff no B-difference without an A-difference."""
    # target "smile" fully fixed by base "happy"
    worlds = [
        {"happy": True, "smile": True, "weather": "sun"},
        {"happy": False, "smile": False, "weather": "sun"},
    ]
    # 'weather' is not a bool atom — use pure-quality worlds instead.
    worlds = [
        {"happy": True, "smile": True},
        {"happy": False, "smile": False},
    ]
    assert supervenes(worlds, ["happy"], ["smile"])
    # If a world smiles without being happy, supervenience fails.
    broken = [*worlds, {"happy": False, "smile": True}]
    assert not supervenes(broken, ["happy"], ["smile"])


def test_conceivability_transcends_observation():
    """The conceivable reaches worlds never observed: a property
    absent from every observed fact is still possible — free
    recombination constructs it."""
    observed_world = {"feather": True, "heavy": False}
    # Nothing observed is both feathered and heavy; still conceivable.
    f = and_(atom("feather"), atom("heavy"))
    assert not f.evaluate(observed_world)
    assert conceivable(f, {"feather", "heavy"})


def test_status_three_way():
    worlds = pluriverse(["p"])
    assert status(atom("p"), worlds) == "contingent"
    assert status(or_(atom("p"), not_(atom("p"))), worlds) == "necessary"
    assert status(and_(atom("p"), not_(atom("p"))), worlds) == "impossible"
