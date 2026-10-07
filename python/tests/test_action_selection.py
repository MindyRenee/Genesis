"""Tests for the reduced basal-ganglia action-selection circuit."""

from __future__ import annotations

import math

import pytest

from genesis_conscious.basal_ganglia import (
    ActionBid,
    BasalGangliaSelector,
)


def test_selector_gates_strongest_action_by_disinhibition() -> None:
    selector = BasalGangliaSelector(hyperdirect_gain=0.3)
    result = selector.select(
        [ActionBid("answer", 0.9), ActionBid("ask", 0.2)],
    )

    assert result.selected == "answer"
    assert result.disinhibition["answer"] > result.disinhibition["ask"]
    assert result.output_inhibition["answer"] < result.output_inhibition["ask"]


def test_selector_raises_threshold_under_high_conflict() -> None:
    selector = BasalGangliaSelector(hyperdirect_gain=0.3)
    result = selector.select(
        [ActionBid("answer", 0.9), ActionBid("ask", 0.85)],
    )

    assert result.conflict > 0.9
    assert result.hyperdirect_brake > 0.0
    assert result.effective_threshold > selector.base_threshold
    # A mild brake narrows the release margin without vetoing, which is
    # the behaviour the decision engine relies on: its mid-salience
    # candidate field must still select something.
    winner = result.selected
    assert winner is not None
    margin = result.disinhibition[winner] - result.effective_threshold
    assert margin > 0.0
    # The margin must be tighter under conflict than for a lopsided field,
    # which is the behavioural point of the hyperdirect pathway.
    lopsided = selector.select([ActionBid("answer", 0.9), ActionBid("ask", 0.2)])
    assert lopsided.selected is not None
    lopsided_margin = (
        lopsided.disinhibition[lopsided.selected] - lopsided.effective_threshold
    )
    assert margin < lopsided_margin


def test_selector_vetoes_when_brake_exceeds_the_winner() -> None:
    """A strong hyperdirect brake blocks release entirely.

    The veto path -- ``selected is None`` -- is what DecisionEngine
    treats as "hold back this turn". It needs a brake large enough to
    push the threshold above the best action's disinhibition, which for
    these bids means ``hyperdirect_gain`` well above the 0.8 default.
    At the default gain the brake narrows the margin (covered by
    ``test_selector_raises_threshold_under_high_conflict``) and the
    winner is still released.
    """
    selector = BasalGangliaSelector(hyperdirect_gain=2.0)
    result = selector.select(
        [ActionBid("answer", 0.9), ActionBid("ask", 0.85)],
    )

    assert result.selected is None
    assert result.effective_threshold > max(result.disinhibition.values())


def test_positive_rpe_favors_direct_drive_without_changing_salience() -> None:
    selector = BasalGangliaSelector(hyperdirect_gain=0.0)
    neutral = selector.select(
        [ActionBid("answer", 0.8), ActionBid("ask", 0.4)],
        dopamine_rpe=0.0,
    )
    positive = selector.select(
        [ActionBid("answer", 0.8), ActionBid("ask", 0.4)],
        dopamine_rpe=1.0,
    )

    assert positive.salience == neutral.salience
    assert positive.d1_drive["answer"] > neutral.d1_drive["answer"]
    assert positive.d2_drive["answer"] < neutral.d2_drive["answer"]
    assert positive.selected == "answer"


def test_negative_rpe_opposes_direct_drive() -> None:
    selector = BasalGangliaSelector(hyperdirect_gain=0.0)
    neutral = selector.select(
        [ActionBid("answer", 0.8), ActionBid("ask", 0.4)],
        dopamine_rpe=0.0,
    )
    negative = selector.select(
        [ActionBid("answer", 0.8), ActionBid("ask", 0.4)],
        dopamine_rpe=-1.0,
    )

    assert negative.d1_drive["answer"] < neutral.d1_drive["answer"]
    assert negative.d2_drive["answer"] > neutral.d2_drive["answer"]


@pytest.mark.parametrize(
    "bids",
    [
        [ActionBid("answer", math.nan)],
        [ActionBid("answer", math.inf)],
    ],
)
def test_selector_rejects_non_finite_salience(bids: list[ActionBid]) -> None:
    with pytest.raises(ValueError):
        BasalGangliaSelector().select(bids)


def test_selector_rejects_duplicate_actions() -> None:
    with pytest.raises(ValueError):
        BasalGangliaSelector().select(
            [ActionBid("answer", 0.5), ActionBid("answer", 0.4)]
        )
