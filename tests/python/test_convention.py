"""Tests for Lewis's analysis of convention — conformity, mutual
belief, conditional preference, arbitrariness."""

from frontal_lobe.reasoning.convention import Agent, is_convention


def _pop(payoff=(2, 0), alt=(2,)):
    """A population genuinely conventional: all drive right, expect
    it, prefer it given the others do, and would as gladly drive
    left had the others chosen that."""
    return [
        Agent(
            name=f"a{i}",
            action="right",
            expects="right",
            conform_payoff=payoff[0],
            deviate_payoff=payoff[1],
            alt_payoffs={"left": alt[0]},
        )
        for i in range(4)
    ]


def test_genuine_convention():
    """All of Lewis's clauses satisfied — right-side driving is a
    convention: followed, expected, preferred-conditional, and
    arbitrary (left would have served)."""
    report = is_convention(_pop(), "right", alternatives=["left"])
    assert report.is_convention
    assert report.conformity == 1.0
    assert report.failure == "conventional"


def test_habit_is_not_convention():
    """A regularity followed for independent reasons fails
    conditional preference — everyone would do it anyway; there is
    nothing to coordinate on."""
    pop = _pop(payoff=(0, 2))  # deviating pays more — habit has no pull
    report = is_convention(pop, "right", alternatives=["left"])
    assert not report.is_convention
    assert "habit" in report.failure


def test_no_alternative_means_not_conventional():
    """One viable option is forced, not conventional — arbitrariness
    fails: nothing else could have been the rule."""
    report = is_convention(_pop(), "right", alternatives=[])
    assert not report.is_convention
    assert "forced" in report.failure


def test_dead_convention_fails_mutual_belief():
    """Still performed but no longer believed to be performed — the
    convention has lapsed though the motions continue."""
    pop = [
        Agent(
            name="a0",
            action="right",
            expects="left",  # believes the others have switched
            conform_payoff=2,
            deviate_payoff=0,
            alt_payoffs={"left": 2},
        ),
        *_pop()[1:],
    ]
    report = is_convention(pop, "right", alternatives=["left"])
    # 3 of 4 expect conformity — below the almost-all floor.
    assert not report.is_convention
    assert "dead convention" in report.failure


def test_unfollowed_regularity_fails_conformity():
    """A regularity nobody actually keeps is no convention at all."""
    pop = [
        Agent(
            name=f"a{i}",
            action="left",
            expects="left",
            conform_payoff=2,
            deviate_payoff=0,
            alt_payoffs={"right": 2},
        )
        for i in range(4)
    ]
    report = is_convention(pop, "right", alternatives=["left"])
    assert not report.is_convention
    assert report.conformity == 0.0


def test_almost_all_threshold():
    """Lewis's 'almost everyone' — a single dissenter does not kill
    the convention."""
    pop = _pop()
    pop[0] = Agent(
        name="rogue",
        action="left",
        expects="right",
        conform_payoff=2,
        deviate_payoff=0,
        alt_payoffs={"left": 2},
    )
    report = is_convention(pop, "right", alternatives=["left"], threshold=0.7)
    assert report.conformity == 0.75
    assert report.is_convention
