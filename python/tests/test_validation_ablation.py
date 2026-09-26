"""Tests for the experimental learning ablation probes."""

from validation.learning_ablation import run_stdp_probe, run_td_probe


def test_td_enabled_changes_value() -> None:
    """TD learning produces a measurable value change."""
    result = run_td_probe(enabled=True)
    assert result.delta != 0.0


def test_td_disabled_preserves_value() -> None:
    """Disabling TD preserves the measured value."""
    result = run_td_probe(enabled=False)
    assert result.delta == 0.0


def test_stdp_enabled_changes_similarity() -> None:
    """Timing-dependent plasticity changes the embedding relationship."""
    result = run_stdp_probe(enabled=True)
    assert result.delta != 0.0


def test_stdp_disabled_preserves_similarity() -> None:
    """Disabling STDP preserves the embedding relationship."""
    result = run_stdp_probe(enabled=False)
    assert result.delta == 0.0
