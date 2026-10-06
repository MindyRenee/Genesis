"""Tests for the reduced basal-ganglia action-selection circuit."""

from __future__ import annotations

import math

import pytest

from genesis_cognitive.action_selection import (
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
    assert result.selected is None


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
