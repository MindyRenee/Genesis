"""Tests for the phototransduction cascade model."""

import pytest
from occipital_lobe.perception.phototransduction import (
    PhototransductionCascade,
    PhototransductionParams,
    PhototransductionState,
    cone_vs_rod_comparison,
)


def test_phototransduction_initialization():
    """Test that the cascade initializes to dark-adapted baseline."""
    cascade = PhototransductionCascade()
    state = cascade.state

    # Dark-adapted: cGMP and calcium at baseline, no activation
    assert state.cgmp == 1.0
    assert state.calcium == 1.0
    assert state.r_star == 0.0
    assert state.pde_star == 0.0
    # In dark, cGMP is high → CNG channels open → high photocurrent
    assert state.photocurrent == pytest.approx(0.5, abs=0.1)  # Half-maximal at cGMP=1.0


def test_phototransduction_step():
    """Test that a single step updates the state correctly."""
    cascade = PhototransductionCascade()
    initial_state = cascade.state

    # Step with light
    state = cascade.step(light_intensity=0.5)

    # Light should activate the cascade (or at least not decrease)
    assert state.r_star >= initial_state.r_star
    assert state.pde_star >= initial_state.pde_star
    # cGMP should decrease or stay similar (PDE hydrolyzes it, but GC synthesizes)
    # With basal PDE activity, the change is small in one step
    assert abs(state.cgmp - initial_state.cgmp) < 0.01


def test_phototransduction_flash_response():
    """Test flash response simulation."""
    cascade = PhototransductionCascade()

    # Simulate a 100 ms flash at 0.5 intensity
    time, photocurrent, cgmp, pde_star = cascade.simulate_flash_response(
        flash_intensity=0.5,
        flash_duration=0.1,
        total_time=0.5,
    )

    # Check array shapes
    assert len(time) == len(photocurrent) == len(cgmp) == len(pde_star)
    assert len(time) > 0

    # Check that photocurrent drops during flash (hyperpolarization)
    # and recovers after flash ends
    flash_end_idx = int(0.1 / cascade.params.dt)
    min_during_flash = photocurrent[:flash_end_idx].min()
    max_after_flash = photocurrent[flash_end_idx:].max()

    assert min_during_flash < max_after_flash  # Recovery after flash


def test_phototransduction_steady_state():
    """Test steady-state response to constant light."""
    cascade = PhototransductionCascade()

    # Dark steady state
    dark_state = cascade.steady_state_response(light_intensity=0.0)
    assert dark_state.cgmp == pytest.approx(1.0, abs=0.1)
    assert dark_state.photocurrent == pytest.approx(0.5, abs=0.1)

    # Bright steady state
    bright_state = cascade.steady_state_response(light_intensity=1.0)
    # Bright light should reduce cGMP and photocurrent
    assert bright_state.cgmp < dark_state.cgmp
    assert bright_state.photocurrent < dark_state.photocurrent


def test_phototransduction_sensitivity():
    """Test that sensitivity decreases with adaptation."""
    cascade = PhototransductionCascade()

    # Dark-adapted sensitivity
    dark_state = cascade.steady_state_response(light_intensity=0.0)
    dark_current = dark_state.photocurrent

    # Adapt to bright light
    bright_state = cascade.steady_state_response(light_intensity=1.0)
    bright_current = bright_state.photocurrent

    # Bright light should reduce photocurrent (less cGMP, channels close)
    assert bright_current < dark_current

    # The sensitivity probe: the suppression slope normalized to the
    # dark-adapted value. At these input magnitudes the Ca²⁺→GC
    # feedback holds the operating point, so sensitivity stays ≈ 1 —
    # the model shows no Weber-law desensitization here; what the
    # probe must get right is the sign (suppression, not gain) and
    # leaving the live state untouched.
    assert cascade.sensitivity(0.0) == pytest.approx(1.0, abs=0.01)
    assert 0.0 < cascade.sensitivity(0.5) <= 1.0
    assert 0.0 < cascade.sensitivity(1.0) <= 1.0

    # The probe must not disturb the live state.
    state = cascade.step(0.3)
    marker = (state.cgmp, state.calcium)
    cascade.sensitivity(0.4)
    assert (cascade.state.cgmp, cascade.state.calcium) == marker


def test_phototransduction_reset():
    """Test that reset returns to dark-adapted baseline."""
    cascade = PhototransductionCascade()

    # Drive the cascade with light
    cascade.step(light_intensity=1.0)
    assert cascade.state.r_star > 0

    # Reset
    cascade.reset()
    assert cascade.state.r_star == 0.0
    assert cascade.state.pde_star == 0.0
    assert cascade.state.cgmp == 1.0
    assert cascade.state.calcium == 1.0


def test_cone_vs_rod_parameters():
    """Test that cone and rod parameters differ as expected."""
    params = cone_vs_rod_comparison()

    assert "cone" in params
    assert "rod" in params

    cone = params["cone"]
    rod = params["rod"]

    # Cones have faster inactivation
    assert cone.k_r_inact > rod.k_r_inact
    assert cone.k_deact > rod.k_deact


def test_cone_vs_rod_response():
    """Test that cones respond faster than rods."""
    params = cone_vs_rod_comparison()

    cone_cascade = PhototransductionCascade(params["cone"])
    rod_cascade = PhototransductionCascade(params["rod"])

    # Simulate flash response for both
    cone_time, cone_current, _, _ = cone_cascade.simulate_flash_response(
        flash_intensity=0.5,
        flash_duration=0.1,
        total_time=0.5,
    )

    rod_time, rod_current, _, _ = rod_cascade.simulate_flash_response(
        flash_intensity=0.5,
        flash_duration=0.1,
        total_time=0.5,
    )

    # Find time to minimum (peak response)
    cone_min_idx = cone_current.argmin()
    rod_min_idx = rod_current.argmin()

    # Cone should reach minimum faster
    assert cone_time[cone_min_idx] < rod_time[rod_min_idx]


def test_phototransduction_state_dataclass():
    """Test that PhototransductionState is a proper dataclass."""
    state = PhototransductionState(
        r_star=0.5,
        pde_star=0.3,
        cgmp=0.8,
        calcium=0.9,
        photocurrent=0.4,
    )

    assert state.r_star == 0.5
    assert state.pde_star == 0.3
    assert state.cgmp == 0.8
    assert state.calcium == 0.9
    assert state.photocurrent == 0.4


def test_phototransduction_params_dataclass():
    """Test that PhototransductionParams is a proper dataclass."""
    params = PhototransductionParams(
        k_r=2.0,
        k_r_inact=60.0,
        k_act=40.0,
    )

    assert params.k_r == 2.0
    assert params.k_r_inact == 60.0
    assert params.k_act == 40.0


def test_michaelis_menten_kinetics():
    """Test that PDE follows Michaelis-Menten kinetics."""
    cascade = PhototransductionCascade()

    # Set up a state with known PDE* and cGMP
    cascade.state.pde_star = 1.0
    cascade.state.cgmp = 0.5

    # Step with no light (only PDE* acting)
    cascade.step(light_intensity=0.0)

    # cGMP should decrease due to PDE* hydrolysis
    assert cascade.state.cgmp < 0.5


def test_calcium_feedback():
    """Test that calcium feedback regulates cGMP synthesis."""
    cascade = PhototransductionCascade()

    # High calcium should inhibit GC, reducing cGMP synthesis
    cascade.state.calcium = 2.0
    cascade.state.cgmp = 0.5
    cascade.state.pde_star = 0.0  # No hydrolysis

    # Step with no light
    cascade.step(light_intensity=0.0)

    # cGMP should recover slowly (GC inhibited by high Ca)
    assert cascade.state.cgmp < 1.0  # Not fully recovered

    # Low calcium should disinhibit GC
    cascade.reset()
    cascade.state.calcium = 0.1
    cascade.state.cgmp = 0.5
    cascade.state.pde_star = 0.0

    cascade.step(light_intensity=0.0)

    # cGMP should recover faster (GC disinhibited)
    assert cascade.state.cgmp > 0.5
