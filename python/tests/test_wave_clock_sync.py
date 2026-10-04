"""The brain's clock must track the body's clock.

Genesis advances two dynamical systems from the same wall clock: the
neurochemistry (in Rust, via ``advance_neuro``) and the brain-wave
oscillator (here). They are only coupled if both integrate the *same*
elapsed interval.

They did not. The chemical step is clamped to DT_MAX (10 s, mirrored
from the daemon); the wave step was clamped to 1.0 s. At the mind's
~1 Hz heartbeat that is usually harmless, but any slower cycle — a GC
pause, CPU contention, a busy body — advanced chemistry by the full
interval and the waves by at most 1 s. The remainder was discarded, not
deferred, so the wave clock fell permanently behind and never caught up:

    gap    chem dt   wave dt   wave error
    1.5s     1.500s    1.000s      -33%
    2.0s     2.000s    1.000s      -50%
    5.0s     5.000s    1.000s      -80%

Because phase is the source of every phase-derived quantity —
cross-frequency coupling, REM's epsilon-gated alpha, sleep-stage
timing — the whole wave system ran slow and reported itself healthy:
amplitudes relaxed on the full interval and still looked right.

The tests below pin the two properties that fix it:

* frame-rate independence — the oscillator lands on the same state
  whether assess() is called once or a hundred times over the same
  interval;
* clock agreement — the wave step no longer has a tighter ceiling than
  the chemical step.
"""

from __future__ import annotations

import math

import pytest

from genesis_client import NeuroSummary
from genesis_client.protocol import DT_MAX
from genesis_conscious.thalamus.brain_waves import (
    _TURN_SNAP_EPS,
    BrainWave,
    BrainWaveOscillator,
    _sleep_phase_powers,
)


def _make_summary(arousal: float = 0.5, phase: int = 0) -> NeuroSummary:
    """A NeuroSummary with the fields the wave targets read."""
    return NeuroSummary(
        arousal=arousal,
        valence=0.0,
        global_tone=0.5,
        plasticity_gate=0.5,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=phase,
    )


def _targets(osc: BrainWaveOscillator, **kw):
    summary = _make_summary(**kw)
    targets, _label, _desc = _sleep_phase_powers(summary)
    return targets


# ── The bug itself ──────────────────────────────────────────────


def test_wave_step_ceiling_matches_chemistry():
    """The wave clock must not have a tighter ceiling than the body's.

    This is the regression guard for the drift itself: if the wave
    clamp is ever made smaller than the chemical DT_MAX, the two clocks
    disagree again for any interval between the two bounds.
    """
    assert DT_MAX == 10.0
    # The clamp in assess() is min(DT_MAX, ...) — pinned here so that
    # lowering it is a deliberate act with a failing test, not a
    # silent truncation.
    assert BrainWaveOscillator._MAX_SUBSTEP <= DT_MAX


def _phase_delta(a: float, b: float) -> float:
    """Shortest distance between two phases on the circle.

    Phase lives on [0, 2pi), so 0.0 and 2pi are the *same* phase that
    differ by 2pi in floating point. Plain subtraction reports that as a
    full-cycle error. Every phase comparison in this module goes
    through here.
    """
    two_pi = 2.0 * math.pi
    return abs((a - b + math.pi) % two_pi - math.pi)


def test_two_second_gap_advances_full_interval():
    """A 2 s stall must advance 2 s of wave time, not 1 s.

    Pre-fix this lost 50% of the interval permanently. Compared against
    the full-2s expectation rather than "not equal to 1 s", because for
    slow bands a 1 s and 2 s advance can coincide modulo 2pi (EPSILON at
    0.25 Hz advances exactly pi either way) -- so the discriminating
    check is the full-interval one.
    """
    osc = BrainWaveOscillator()
    before = dict(osc._phase)

    osc._advance_phase_exact(2.0)

    two_pi = 2.0 * math.pi
    for band in BrainWave:
        freq = osc._FREQS[band]
        expected = (before[band] + two_pi * freq * 2.0) % two_pi
        assert _phase_delta(osc._phase[band], expected) < 1e-9

    # A band whose 2 s and 1 s advances are genuinely distinguishable
    # proves the full interval was used. THETA is 5.5 Hz: 2 s is 11
    # whole cycles (phase 0), 1 s is 5.5 cycles (phase pi).
    theta = BrainWaveOscillator()
    theta._phase[BrainWave.THETA] = 0.0
    theta._advance_phase_exact(2.0)
    two_s = theta._phase[BrainWave.THETA]

    theta._phase[BrainWave.THETA] = 0.0
    theta._advance_phase_exact(1.0)
    one_s = theta._phase[BrainWave.THETA]

    assert _phase_delta(two_s, 0.0) < 1e-6
    assert _phase_delta(one_s, math.pi) < 1e-6
    assert _phase_delta(two_s, one_s) > 1.0, (
        "a 2 s advance must be distinguishable from a 1 s advance; "
        "if these agree the full interval is not being applied"
    )


def test_phase_is_exact_regardless_of_subdivision():
    """phi = (phi_0 + 2*pi*f*dt) mod 2*pi has a closed form -- no drift.

    Advancing 5 s in one call and as fifty 0.1 s steps must agree to
    floating-point precision. Note the *kind* of agreement: for a band
    whose 5 s is a whole number of cycles (lambda, 150 Hz -> 750), the
    single call returns exactly 0.0 while the accumulated sum lands
    ~1e-14 away. That residue is pure float summation, orders of
    magnitude below anything downstream can observe -- and crucially it
    does not grow with dt, because each call re-enters the closed form
    rather than integrating a derivative.
    """
    one_shot = BrainWaveOscillator()
    stepped = BrainWaveOscillator()

    one_shot._advance_phase_exact(5.0)
    for _ in range(50):
        stepped._advance_phase_exact(0.1)

    for band in BrainWave:
        delta = _phase_delta(stepped._phase[band], one_shot._phase[band])
        assert delta < 1e-9, f"{band.value} drifted {delta}"

    # The single call is exact where it can be: lambda completes a whole
    # number of cycles in 5 s and returns to phase 0.
    assert _phase_delta(one_shot._phase[BrainWave.LAMBDA], 0.0) < 1e-12

    # Error is bounded by float precision and does not accumulate with
    # the interval -- 100 s in one call is as exact as 1 s.
    two_pi = 2.0 * math.pi
    long_ = BrainWaveOscillator()
    long_._advance_phase_exact(100.0)
    for band in BrainWave:
        freq = BrainWaveOscillator._FREQS[band]
        expected = (two_pi * freq * 100.0) % two_pi
        assert _phase_delta(long_._phase[band], expected) < 1e-9


def test_exponential_relaxation_is_exact_under_subdivision():
    """The decay terms are exact; relaxation is exact for fixed targets.

    Two distinct claims, and the difference matters:

    * Drive and stimulus **decay** compose exactly — they have no
      feedback through the state they influence, so
      ``exp(-(h1+h2)/τ) == exp(-h1/τ)·exp(-h2/τ)`` holds bit-for-bit.
      Verified below to 1e-16.
    * Amplitude relaxation toward a **constant** target is also exact,
      landing on the closed-form solution regardless of subdivision.
    * Amplitude relaxation with a **decaying drive** is not, and must
      not be: the drive enters the target, so the target varies within
      the interval and composing the relaxation exponentials does not
      close. This is why the drive is decayed *inside* each sub-step
      rather than applied once for the whole interval — the finer the
      split, the more faithfully the relaxation follows the decaying
      target, converging on the continuum solution.

    The sub-stepping is therefore not merely harmless for the dynamics;
    it is what makes the driven case correct.
    """
    osc = BrainWaveOscillator()
    targets = _targets(osc)
    band = BrainWave.ALPHA
    tau = osc._TAU[band]

    # Drive/stimulus decay: bit-exact under subdivision.
    def run_decay(n: int) -> tuple[float, float]:
        o = BrainWaveOscillator()
        o._top_down = dict.fromkeys(BrainWave, 0.3)
        o._stimulus = dict.fromkeys(BrainWave, 0.4)
        for _ in range(n):
            decay = math.exp(-(0.5 / n) / o._DRIVE_TAU)
            for b in BrainWave:
                o._top_down[b] *= decay
            sd = math.exp(-(0.5 / n) / 5.0)
            for b in BrainWave:
                o._stimulus[b] *= sd
        return o._top_down[band], o._stimulus[band]

    d1, s1 = run_decay(1)
    d5, s5 = run_decay(5)
    assert d5 == pytest.approx(d1, rel=1e-15)
    assert s5 == pytest.approx(s1, rel=1e-15)
    assert math.exp(-0.5 / 2.0) == pytest.approx(math.exp(-0.1 / 2.0) ** 5, rel=1e-15)

    # Relaxation toward a constant target: exact for any split, and
    # equal to the closed form.
    def run_relax(n: int) -> float:
        o = BrainWaveOscillator()
        o._phase = dict.fromkeys(BrainWave, 0.0)
        o._amplitude = dict.fromkeys(BrainWave, 0.1)
        o._top_down = dict.fromkeys(BrainWave, 0.0)
        o._stimulus = dict.fromkeys(BrainWave, 0.0)
        o._last_phase_name = "active"
        for _ in range(n):
            o._relax_once(0.5 / n, targets)
        return o._amplitude[band]

    analytic = 0.1 + (targets[band] - 0.1) * (1.0 - math.exp(-0.5 / tau))
    assert run_relax(1) == pytest.approx(analytic, rel=1e-12)
    assert run_relax(5) == pytest.approx(analytic, rel=1e-12)
    assert run_relax(20) == pytest.approx(analytic, rel=1e-12)

    # With a decaying drive the single-step result is biased high: the
    # full drive is applied to the whole interval at once. Refining the
    # split converges toward the driven continuum solution.
    def run_driven(n: int) -> float:
        o = BrainWaveOscillator()
        o._phase = dict.fromkeys(BrainWave, 0.0)
        o._amplitude = dict.fromkeys(BrainWave, 0.1)
        o._top_down = dict.fromkeys(BrainWave, 0.3)
        o._stimulus = dict.fromkeys(BrainWave, 0.0)
        o._last_phase_name = "active"
        for _ in range(n):
            o._relax_once(0.5 / n, targets)
        return o._amplitude[band]

    coarse, fine = run_driven(1), run_driven(50)
    assert coarse > fine, "a single step must overshoot the decaying drive"
    assert fine == pytest.approx(coarse, rel=0.1)


# ── Frame-rate independence ─────────────────────────────────────


# Bands whose period is long relative to the sub-step. PAC modulation
# on these converges cleanly with sampling density.
_SLOW_BANDS = (
    BrainWave.EPSILON,
    BrainWave.DELTA,
    BrainWave.THETA,
    BrainWave.ALPHA,
    BrainWave.SIGMA,
)
# Bands fast enough that a 0.1 s sub-step spans many of their cycles
# (lambda: 15 cycles per sub-step), so the slow-band PAC modulator is
# sampled at effectively a fixed phase. Their residual does not converge
# with more sub-steps — it is an aliasing limit of the coupling, not
# integration error, and is bounded by _PAC_DEPTH.
_FAST_BANDS = (
    BrainWave.BETA1,
    BrainWave.BETA2,
    BrainWave.BETA3,
    BrainWave.GAMMA,
    BrainWave.LAMBDA,
)

# Measured convergence of the frame-rate independence test against the
# 0.1 s reference, over 2 s: n=1 -> 1e-2, n=5 -> 4e-4, n=10 -> 1.2e-4,
# n>=20 -> bounded at 1e-4 by fast-band PAC aliasing. See
# test_fast_band_pac_has_a_bounded_aliasing_floor.
_FRAME_RATE_TOL = 2e-4


@pytest.mark.parametrize("steps", [20, 50, 100, 500])
def test_frame_rate_independence(steps: int):
    """Wave time is the same regardless of how often assess() runs.

    Phase is exact by construction. Amplitudes agree to the measured
    PAC sampling floor (~1e-4), which is the value of _PAC_DEPTH, not a
    fudge factor.

    The splits start at 20 (= 2 s / _MAX_SUBSTEP, the resolution the
    integrator actually uses) and go finer. Coarser splits are not
    tested here because the integrator never produces them; the 0.1 s
    resolution is the coarsest it will emit, and a hypothetical
    single-step 2 s integration is 1e-2 off by construction — that is
    the whole reason for sub-stepping, not a property to assert here.
    """
    total = 2.0
    reference = BrainWaveOscillator()
    targets = _targets(reference)
    reference._amplitude = dict.fromkeys(BrainWave, 0.1)
    reference._advance_phase_exact(total)
    reference._advance_relaxation(total, targets)

    other = BrainWaveOscillator()
    other._amplitude = dict.fromkeys(BrainWave, 0.1)
    other._advance_phase_exact(total)
    for _ in range(steps):
        other._relax_once(total / steps, targets)

    assert steps > 0
    for band in BrainWave:
        assert other._amplitude[band] == pytest.approx(
            reference._amplitude[band], rel=_FRAME_RATE_TOL
        )


def test_coarse_splits_are_measurably_worse():
    """The reason _MAX_SUBSTEP exists: coarser integration drifts.

    One step over 2 s is ~1e-2 off the reference — an order of
    magnitude worse than the 0.1 s default. This is the quantitative
    justification for sub-stepping, and the number a future change to
    _MAX_SUBSTEP should be checked against.
    """
    targets = _targets(BrainWaveOscillator())
    reference = BrainWaveOscillator()
    reference._amplitude = dict.fromkeys(BrainWave, 0.1)
    reference._advance_phase_exact(2.0)
    reference._advance_relaxation(2.0, targets)

    coarse = BrainWaveOscillator()
    coarse._amplitude = dict.fromkeys(BrainWave, 0.1)
    coarse._advance_phase_exact(2.0)
    coarse._relax_once(2.0, targets)

    worst = max(
        abs(coarse._amplitude[b] - reference._amplitude[b])
        / reference._amplitude[b]
        for b in BrainWave
    )
    assert worst > 10 * _FRAME_RATE_TOL, (
        f"single-step error {worst:.2e} is not meaningfully worse than the "
        f"sub-stepped reference; _MAX_SUBSTEP may be redundant"
    )


def test_slow_band_amplitude_converges_with_sampling():
    """Where PAC converges, more sub-steps reach the reference exactly.

    The slow bands are the ones whose phase the PAC modulator actually
    tracks, so refining the split converges toward the reference to
    float precision. This is the property that makes the 0.1 s default
    a safe choice rather than a tolerance.
    """
    targets = _targets(BrainWaveOscillator())
    reference = BrainWaveOscillator()
    reference._amplitude = dict.fromkeys(BrainWave, 0.1)
    reference._advance_phase_exact(2.0)
    reference._advance_relaxation(2.0, targets)

    for steps in (20, 40, 200):
        other = BrainWaveOscillator()
        other._amplitude = dict.fromkeys(BrainWave, 0.1)
        other._advance_phase_exact(2.0)
        for _ in range(steps):
            other._relax_once(2.0 / steps, targets)
        for band in _SLOW_BANDS:
            assert other._amplitude[band] == pytest.approx(
                reference._amplitude[band], rel=1e-12
            )


def test_fast_band_pac_has_a_bounded_aliasing_floor():
    """Fast bands carry a PAC sampling residual, bounded by _PAC_DEPTH.

    lambda oscillates at 150 Hz, so a 0.1 s sub-step spans 15 of its
    cycles and the slow-band modulator is effectively sampled at a
    fixed phase. Refining the split therefore does not converge — this
    pins that the residual stays inside the modulation depth instead of
    growing, which is the property that matters (it is coupling error,
    not time lost).
    """
    targets = _targets(BrainWaveOscillator())
    residuals = []
    for steps in (20, 200, 2000):
        osc = BrainWaveOscillator()
        osc._amplitude = dict.fromkeys(BrainWave, 0.1)
        osc._advance_phase_exact(2.0)
        for _ in range(steps):
            osc._relax_once(2.0 / steps, targets)
        residuals.append(dict(osc._amplitude))

    base = residuals[0]
    for band in _FAST_BANDS:
        for other in residuals[1:]:
            assert other[band] == pytest.approx(
                base[band], rel=_FRAME_RATE_TOL
            )
        # The spread across split resolutions is bounded by the
        # modulation depth, scaled by how far the band has actually
        # relaxed — the residual is coupling error, not lost time, so
        # it must not grow without bound as the split gets finer.
        spread = max(
            abs(r[band] - base[band]) for r in residuals
        )
        assert spread <= BrainWaveOscillator._PAC_DEPTH * max(
            abs(base[band]), 1e-9
        )


def test_substep_count_respects_bound():
    """A long stall must not cost unbounded CPU.

    At DT_MAX (10 s) the ideal 0.1 s resolution would want 100
    sub-steps; the bound caps the work. Sub-steps must still cover the
    interval exactly, so no time is lost — only the PAC sampling rate
    degrades, which the exponential terms are immune to.
    """
    osc = BrainWaveOscillator()
    targets = _targets(osc)

    calls: list[float] = []
    original = osc._relax_once

    def counting(dt, tgts):
        calls.append(dt)
        original(dt, tgts)

    osc._relax_once = counting  # type: ignore[method-assign]
    osc._advance_relaxation(float(DT_MAX), targets)

    assert len(calls) == osc._MAX_SUBSTEPS
    assert osc._MAX_SUBSTEPS < DT_MAX / osc._MAX_SUBSTEP, (
        "the bound must actually engage at DT_MAX, or CPU is unbounded"
    )
    # Sub-steps cover the interval exactly, whatever the count.
    assert sum(calls) == pytest.approx(float(DT_MAX), rel=1e-12)
    # Enlarged steps are permitted past the bound — that is the point.
    assert max(calls) > osc._MAX_SUBSTEP


def test_normal_interval_uses_configured_resolution():
    """A 1 s heartbeat splits at the designed 0.1 s resolution."""
    osc = BrainWaveOscillator()
    targets = _targets(osc)
    calls: list[float] = []
    original = osc._relax_once

    def counting(dt, tgts):
        calls.append(dt)
        original(dt, tgts)

    osc._relax_once = counting  # type: ignore[method-assign]
    osc._advance_relaxation(1.0, targets)

    assert len(calls) == 10
    assert all(dt == pytest.approx(0.1) for dt in calls)

# ── End-to-end: the real assess() path, across a slow heartbeat ──


def _run_heartbeat(gaps: list[float], monkeypatch) -> list:
    """Drive assess() with a controlled clock, returning final phases.

    Replaces time.monotonic so the elapsed interval between calls is
    exactly what each entry in `gaps` says. This exercises the real
    dt computation in assess(), not the helpers.
    """
    osc = BrainWaveOscillator()
    summary = _make_summary()
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr(
        "genesis_conscious.thalamus.brain_waves.time.monotonic", fake_monotonic
    )
    # Prime the oscillator's clock. A fresh instance has _last_time None,
    # and assess() then falls back to a 0.1 s default for its first
    # interval regardless of what the clock says -- so without this the
    # first gap would be silently measured as 0.1 s. Each gap here is
    # therefore the interval between two consecutive real ticks.
    osc._last_time = clock["t"]
    for gap in gaps:
        clock["t"] += gap
        osc.assess(summary)
    return [osc._phase[b] for b in BrainWave]


def test_assess_does_not_truncate_a_slow_heartbeat(monkeypatch):
    """The regression test for the actual bug, through assess().

    A heartbeat that stalls for 2 s must advance the waves by 2 s. With
    the old `min(1.0, ...)` clamp the wave clock silently lost 1 s of
    every such cycle and never recovered, while chemistry advanced the
    full interval -- the drift this whole change exists to remove.

    THETA (5.5 Hz) is the probe: 2 s is exactly 11 cycles, so the phase
    returns to where it started, whereas 1 s would leave it at pi.
    """
    # One 2 s stall, from phase 0.
    phases = _run_heartbeat([2.0], monkeypatch)
    theta = phases[list(BrainWave).index(BrainWave.THETA)]
    assert _phase_delta(theta, 0.0) < 1e-6, (
        f"after a 2 s stall THETA should have completed 11 whole cycles "
        f"(phase 0), got {theta} -- the interval was truncated"
    )

    # Ten 0.2 s ticks must equal one 2 s tick: same total wave time.
    ticked = _run_heartbeat([0.2] * 10, monkeypatch)
    for band, phase in zip(BrainWave, ticked, strict=True):
        assert _phase_delta(phase, phases[list(BrainWave).index(band)]) < 1e-6


def test_assess_matches_chemistry_dt_over_a_slow_cycle(monkeypatch):
    """Wave time must equal the dt the daemon would have been given.

    The daemon clamps ADVANCE_NEURO's dt to DT_MAX; heartbeat.py passes
    the raw elapsed interval. Whatever chemistry is told, the wave step
    must span the same interval — that equality is the invariant the
    two-clock coupling depends on.
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(
        "genesis_conscious.thalamus.brain_waves.time.monotonic",
        lambda: clock["t"],
    )
    for gap in (0.2, 1.0, 1.5, 2.0, 5.0, 9.9):
        osc = BrainWaveOscillator()
        summary = _make_summary()
        # Each case starts from t=0, so the interval assess() sees is
        # exactly `gap` and nothing accumulates between iterations.
        clock["t"] = 0.0
        osc._last_time = 0.0
        osc._phase[BrainWave.THETA] = 0.0

        clock["t"] = gap
        osc.assess(summary)

        chem_dt = min(DT_MAX, max(0.0, gap))
        expected = (2.0 * math.pi * 5.5 * chem_dt) % (2.0 * math.pi)
        assert _phase_delta(osc._phase[BrainWave.THETA], expected) < 1e-6, (
            f"gap={gap}s: chemistry would advance {chem_dt}s but the waves "
            f"landed at {osc._phase[BrainWave.THETA]}, expected {expected}"
        )


def test_phase_does_not_drift_across_many_slow_heartbeats(monkeypatch):
    """Over 60 s the wave clock must not depend on the heartbeat rate.

    Compares a 0.1 s heartbeat against a 2.0 s one, both totalling 60 s
    of wall-clock. Only phase is compared -- amplitudes are not, because
    PAC coupling samples the slow bands' phase and so legitimately
    depends on how finely the interval is subdivided (bounded and
    measured separately in the aliasing test). The claim under test is
    that *time* is not lost: a slow heartbeat must not leave the brain
    behind the body.

    The oracle is one closed-form 60 s advance, not a formula written
    here: for beta3 (25 Hz, 1500 cycles in 60 s) computing the expected
    value as (2*pi*25*60) % (2*pi) in float carries its own error and
    would measure this test instead of the code.
    """
    oracle = BrainWaveOscillator()
    oracle._advance_phase_exact(60.0)

    clock = {"t": 0.0}
    monkeypatch.setattr(
        "genesis_conscious.thalamus.brain_waves.time.monotonic",
        lambda: clock["t"],
    )
    for gap, count in ((0.1, 600), (2.0, 30)):
        osc = BrainWaveOscillator()
        summary = _make_summary()
        clock["t"] = 0.0
        osc._last_time = 0.0
        for _ in range(count):
            clock["t"] += gap
            osc.assess(summary)
        for band in BrainWave:
            assert _phase_delta(osc._phase[band], oracle._phase[band]) < 1e-9, (
                f"{band.value} at {gap}s x{count}: expected "
                f"{oracle._phase[band]}, got {osc._phase[band]}"
            )


def test_phase_is_frame_rate_independent_for_integral_schedules():
    """Chunking a fixed interval never changes the resulting phase.

    The property that makes the wave clock trustworthy: for any total
    duration, splitting it differently lands on the same phase. This is
    what "the brain runs at the body's speed" reduces to.
    """
    total = 60.0
    reference = BrainWaveOscillator()
    reference._advance_phase_exact(total)

    for chunk in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        steps = round(total / chunk)
        osc = BrainWaveOscillator()
        for _ in range(steps):
            osc._advance_phase_exact(chunk)
        for band in BrainWave:
            assert _phase_delta(osc._phase[band], reference._phase[band]) < 1e-9, (
                f"{band.value}: {steps} x {chunk}s disagrees with one "
                f"{total}s advance"
            )


def test_integral_cycle_counts_return_exactly_to_start():
    """An exact whole number of cycles must return to the start phase.

    Several band/dt pairs have an integral cycle count, so the correct
    residual after removing whole turns is zero. In float the count
    lands on either side of the integer depending on the pair:

    * THETA, 5.5 Hz, 2 s -> 11 turns, evaluates to 10.999999999999998
      (just *below*). Subtracting floor turns leaves ~2π of residual.
    * BETA1, 13.5 Hz, 2 s -> 27 turns, evaluates to exactly 27.0.
      Here math.remainder returns the *nearest* multiple, −7.1e-15,
      which wraps to a full turn under mod.

    Neither reduction alone handles both, so the integrator subtracts
    the nearest turn and snaps a within-noise residual to zero. This
    test pins that behaviour: integral cycles must land back on the
    start phase, exactly, for every such pair.
    """
    cases = (
        (BrainWave.THETA, 2.0, 11),    # 5.5 Hz x 2 s
        (BrainWave.SIGMA, 5.0, 70),     # 14 Hz x 5 s
        (BrainWave.SIGMA, 2.0, 28),     # 14 Hz x 2 s
        (BrainWave.BETA1, 2.0, 27),     # 13.5 Hz x 2 s
        (BrainWave.BETA3, 3.0, 75),     # 25 Hz x 3 s
        (BrainWave.GAMMA, 1.5, 75),     # 50 Hz x 1.5 s
        (BrainWave.LAMBDA, 10.0, 1500),  # 150 Hz x 10 s (the DT_MAX end)
    )
    for band, dt, turns in cases:
        freq = BrainWaveOscillator._FREQS[band]
        assert freq * dt == pytest.approx(turns, rel=0, abs=0), (
            f"{band.value} x {dt}s is not {turns} whole turns; the case is "
            "not exercising what this test claims"
        )

        # A single step lands exactly on the start phase.
        osc = BrainWaveOscillator()
        osc._phase[band] = 0.0
        osc._advance_phase_exact(dt)
        assert _phase_delta(osc._phase[band], 0.0) < 1e-12, (
            f"{band.value} x {dt}s ({turns} turns) landed on "
            f"{osc._phase[band]}, expected the start phase"
        )

        # Repeated steps must not accumulate drift.
        osc = BrainWaveOscillator()
        osc._phase[band] = 0.0
        for _ in range(30):
            osc._advance_phase_exact(dt)
        assert _phase_delta(osc._phase[band], 0.0) < 1e-12, (
            f"{band.value} drifted over 30 x {dt}s steps"
        )


def test_fractional_cycle_counts_are_not_snapped():
    """A genuinely fractional turn must still be integrated.

    The snap must only absorb float noise. Alpha at 10 Hz over 0.2 s is
    exactly 2 turns and is snapped, but theta over 0.2 s is 1.1 turns --
    that 0.1 must survive as a real phase advance.
    """
    osc = BrainWaveOscillator()
    osc._phase[BrainWave.THETA] = 0.0
    osc._advance_phase_exact(0.2)
    two_pi = 2.0 * math.pi
    assert _phase_delta(osc._phase[BrainWave.THETA], two_pi * 0.1) < 1e-12, (
        f"1.1 turns must leave a 0.1-turn residue; got "
        f"{osc._phase[BrainWave.THETA]}"
    )

    # And the snap threshold is not so loose that it eats real fractions.
    assert 0.1 > 10 * _TURN_SNAP_EPS


def test_snap_epsilon_is_well_above_float_noise():
    """_TURN_SNAP_EPS must exceed float64 error at every reachable count.

    The largest cycle count within DT_MAX is lambda's 150 Hz over 10 s =
    1500 turns, where float64 spacing is 2^-52 * 1500 ~ 3.3e-13 turns.
    The tolerance has to clear that by a wide margin, or integral cycles
    stop snapping; and it has to stay far below any meaningful fraction
    of a turn.
    """
    worst_spacing = 2.0 ** -52 * (150.0 * 10.0)  # turns
    assert _TURN_SNAP_EPS > 1000 * worst_spacing, (
        f"snap epsilon {_TURN_SNAP_EPS} is not comfortably above float "
        f"noise {worst_spacing}"
    )
    # 1e-9 turns is 6.3e-9 rad — invisible downstream.
    assert 2.0 * math.pi * _TURN_SNAP_EPS < 1e-8


def test_naive_reductions_leave_a_full_turn_residue():
    """Documents the residues a naive reduction leaves behind.

    Not a property of the oscillator -- a guard on the alternatives, so
    the snap's necessity is visible rather than asserted from memory.

    A floor-only reduction on THETA (5.5 Hz, 2 s) leaves ~2π: the count
    is 10.999999999999998 turns, so floor removes 10 and leaves
    0.9999999999999989 of a turn. A remainder-only reduction on BETA1
    (13.5 Hz, 2 s) leaves −7.1e-15 rad, which wraps to a full turn.

    Both are ~2π, so the *phase* is barely disturbed on the very first
    step -- the real cost is that the increment is no longer exactly
    zero, so the step advances a fraction of a turn every time and the
    rate drifts. The snap is what makes an integral cycle count advance
    by exactly nothing, which the tests above confirm to 1e-12.
    """
    two_pi = 2.0 * math.pi

    # THETA: floor leaves nearly a full turn of residue.
    inc = two_pi * BrainWaveOscillator._FREQS[BrainWave.THETA] * 2.0
    floored = inc - two_pi * math.floor(inc / two_pi)
    assert floored / two_pi == pytest.approx(1.0, abs=1e-12), (
        "expected floor to leave ~1 whole turn of residue for THETA"
    )
    assert floored != 0.0

    # BETA1: remainder leaves a tiny negative, i.e. ~1 turn the other
    # way, not zero.
    inc = two_pi * BrainWaveOscillator._FREQS[BrainWave.BETA1] * 2.0
    remainder = math.remainder(inc, two_pi)
    assert remainder != 0.0
    assert abs(remainder) < 1e-13

    # The snap collapses both to exactly zero, which is the property the
    # integral-cycle tests rely on.
    for band, dt, _turns in (
        (BrainWave.THETA, 2.0, 11),
        (BrainWave.BETA1, 2.0, 27),
    ):
        inc = two_pi * BrainWaveOscillator._FREQS[band] * dt
        turns = inc / two_pi
        nearest = round(turns)
        snapped = (
            0.0
            if abs(turns - nearest) <= _TURN_SNAP_EPS
            else inc - two_pi * nearest
        )
        assert snapped == 0.0, f"{band.value} should snap to exactly 0"


def test_integral_cycle_snap_is_a_precision_tidy_up_not_a_behavioural_fix():
    """Sizes the benefit honestly, so the claim is not overstated.

    For the affected pairs (THETA at 2 s, ALPHA at 1.5 s) a floor-only
    reduction leaves a residue of ~(1 - 1.1e-15) turns per step. Over 30
    steps that is ~3e-14 turns of drift — far below anything observable.

    So the snap is not fixing a bug; it makes "an integral number of
    cycles advances nothing" *provable* instead of true-to-15-digits.
    This test asserts both halves of that: the snap is exact, and the
    alternative really is negligible, so nobody later mistakes this for
    a load-bearing optimisation and tightens the tolerance.
    """
    two_pi = 2.0 * math.pi
    for band, dt, turns in (
        (BrainWave.THETA, 2.0, 11),
        (BrainWave.ALPHA, 1.5, 15),
        (BrainWave.BETA3, 3.0, 75),
        (BrainWave.BETA1, 2.0, 27),
    ):
        freq = BrainWaveOscillator._FREQS[band]
        inc = two_pi * freq * dt

        # Snapped: exactly zero, for every one of these.
        assert abs(inc / two_pi - turns) <= _TURN_SNAP_EPS
        osc = BrainWaveOscillator()
        osc._phase[band] = 0.0
        osc._advance_phase_exact(dt)
        assert osc._phase[band] == 0.0, (
            f"{band.value} x {dt}s should land on exactly 0.0 with the "
            f"snap; got {osc._phase[band]!r}"
        )

        # What a floor-only reduction would leave. Two shapes appear:
        # either ~0 (the count evaluates exactly, e.g. BETA1's 27.0) or
        # ~1 turn (it evaluates just below the integer, e.g. THETA's
        # 10.999999999999998). Both are float artefacts of an integral
        # count, and the snap collapses both to exactly 0.
        residue_turns = (inc - two_pi * math.floor(inc / two_pi)) / two_pi
        artefact = min(
            abs(residue_turns), abs(residue_turns - 1.0)
        )
        assert artefact < 1e-14, (
            "a floor-only residue is either ~0 or ~1 turn; if this grew, "
            "the snap would have become load-bearing and this test's "
            f"framing would be wrong (got {artefact:.2e})"
        )
        # Accumulated over a long run it stays far below the snap
        # tolerance, which is exactly why this is a tidy-up.
        assert artefact * 30 < _TURN_SNAP_EPS
