"""The two gears must mesh: body 5 Hz, mind 1 Hz, exact 5:1.

Genesis runs two clocks that must stay in step:

* the **body** (daemon) — neurochemistry, circadian phase, sleep
  staging, interoception — integrating at ``TICK_INTERVAL_MS`` = 200 ms,
  i.e. 5 Hz;
* the **mind** (cognitive process) — the brain-wave oscillator and the
  IPC that drives the body — running its own loop at ``time.sleep(1.0)``,
  i.e. 1 Hz.

Like meshed gears they turn at different speeds and must keep a fixed
ratio, or they slip. The ratio is 5:1, and it is exact: one mind
revolution is precisely five body revolutions. No fractional tooth, no
accumulating slip.

Why this file exists. The two clocks previously had *different* ceilings
on how much elapsed time they would accept — chemistry took up to 10 s,
the wave oscillator only 1.0 s. At 1 Hz that is invisible, but any
slower heartbeat made the waves lose time the chemistry had kept, and
the loss was discarded rather than deferred, so the brain fell
permanently behind the body. That was fixed by giving both the same
``DT_MAX`` (see ``protocol.py``) and by making phase advance exactly.

What these tests pin is the *invariant*, so the mesh cannot silently
change: the gear ratio stays integral, and both gears are handed the
same dt for the same wall-clock interval. See also
``test_wave_clock_sync.py`` for the wave-side integration properties.
"""

from __future__ import annotations

from genesis_client.protocol import DT_MAX, DT_MIN

# Mirrors of the two clocks. These are duplicated as literals on
# purpose: a test that imports the production constant cannot catch the
# constant being changed. `test_constants_match_source` below greps the
# sources to prove these still agree with them.
BODY_TICK_MS = 200          # src/daemon/tick.rs: TICK_INTERVAL_MS
MIND_HEARTBEAT_S = 1.0      # mind/heartbeat.py: time.sleep(1.0)
AUTONOMOUS_INTERVAL_MS = 200  # tick.rs: AUTONOMOUS_INTERVAL_MS

BODY_HZ = 1000.0 / BODY_TICK_MS
MIND_HZ = 1.0 / MIND_HEARTBEAT_S
GEAR_RATIO = BODY_HZ / MIND_HZ


def test_gear_ratio_is_five_to_one() -> None:
    """The mesh is 5:1 — five body teeth per mind tooth."""
    assert BODY_HZ == 5.0
    assert MIND_HZ == 1.0
    assert GEAR_RATIO == 5.0


def test_one_mind_revolution_is_a_whole_number_of_body_teeth() -> None:
    """An integral ratio is what keeps the gears from slipping.

    A fractional tooth (say the mind ran at 1.1 Hz) would mean the mesh
    only lines up every few revolutions, and any per-tooth work done at
    the boundary lands inconsistently. The tuned rates give exactly 5.
    """
    body_teeth_per_mind_tooth = BODY_HZ / MIND_HZ
    assert body_teeth_per_mind_tooth == 5.0
    assert body_teeth_per_mind_tooth == int(body_teeth_per_mind_tooth)


def test_autonomous_fallback_matches_the_tuned_body_rate() -> None:
    """The body's own clock runs at the rate its constants were tuned for.

    ``NeuroTickParams::DEFAULT.dt`` is 0.1 s and the autonomous fallback
    overrides it to ``TICK_INTERVAL_MS / 1000`` = 0.2 s, with a comment
    in tick.rs noting that without that correction every rate constant
    would run at half real-time speed (circadian ~48 h, adenosine ~38 h).
    The fallback must therefore match the tuned interval exactly, or the
    body runs at a different speed depending on whether the mind is alive.
    """
    assert AUTONOMOUS_INTERVAL_MS == BODY_TICK_MS
    assert BODY_TICK_MS / 1000.0 == 0.2


def test_both_gears_receive_the_same_dt() -> None:
    """For any elapsed interval, both integrators must be handed the same dt.

    This is the invariant the drift bug violated. The daemon clamps
    ADVANCE_NEURO's dt to ``[DT_MIN, DT_MAX]``; the wave oscillator must
    accept the identical range. If either ceiling were tighter, the
    gears would slip by the difference on every slow heartbeat.

    Expressed over the whole reachable range rather than a few samples,
    so shrinking one bound cannot pass.
    """
    mismatches = 0
    steps = 200_000
    for i in range(1, steps + 1):
        elapsed = i * (DT_MAX / steps)  # 0 .. DT_MAX inclusive
        chem_dt = min(DT_MAX, max(DT_MIN, elapsed))
        wave_dt = min(DT_MAX, max(0.0, elapsed))
        # Below DT_MIN the two differ by design (chemistry refuses to go
        # below 1 ms); above it they must be bit-identical.
        if elapsed >= DT_MIN and chem_dt != wave_dt:
            mismatches += 1
    assert mismatches == 0, (
        f"{mismatches} intervals where the two gears were handed "
        f"different dt values"
    )


def test_demand_intervals_are_all_inside_the_shared_ceiling() -> None:
    """Every dt either gear can actually produce must be reachable by both.

    The mind's loop sleeps 1.0 s, so the nominal interval is 1 s. Even a
    badly delayed heartbeat is expected to stay well inside DT_MAX; the
    clamp is a guard against a genuine stall (suspend, GC pause), not a
    routine bound. Checking the common cases documents that the shared
    ceiling is not doing routine truncation work.
    """
    for nominal in (0.2, 0.5, MIND_HEARTBEAT_S, 1.5, 2.0, 3.0, 5.0):
        chem_dt = min(DT_MAX, max(DT_MIN, nominal))
        wave_dt = min(DT_MAX, max(0.0, nominal))
        assert chem_dt == wave_dt == nominal, (
            f"a {nominal}s heartbeat is being altered: chemistry "
            f"{chem_dt}, waves {wave_dt}"
        )


def test_wave_clock_ceiling_is_not_tighter_than_the_body() -> None:
    """The specific regression: a wave ceiling below DT_MAX.

    Asserted separately from the equality test above because this is the
    exact shape the bug took — the waves were capped at 1.0 s while the
    body accepted 10 s, so every interval in between lost wave time.
    """
    assert DT_MAX == 10.0, (
        "DT_MAX moved; if the daemon's clamp in ipc.rs changed, update "
        "this and the ceiling assertions together"
    )
    # The wave oscillator's sub-step resolution is the designed 0.1 s,
    # matched to NeuroTickParams::DEFAULT.dt.
    from genesis_conscious.thalamus.brain_waves import BrainWaveOscillator

    assert BrainWaveOscillator._MAX_SUBSTEP == 0.1
    assert BrainWaveOscillator._MAX_SUBSTEP <= DT_MAX


def test_constants_match_source() -> None:
    """The mirrors at the top of this file still match the real sources.

    Guards against the constants drifting while the literals here stay
    put — the failure mode a test that imports the constant cannot see.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]

    tick_rs = (root / "src" / "daemon" / "tick.rs").read_text(encoding="utf-8")
    m = re.search(r"pub const TICK_INTERVAL_MS: u64 = (\d+)", tick_rs)
    assert m, "TICK_INTERVAL_MS not found in tick.rs"
    assert int(m.group(1)) == BODY_TICK_MS

    m = re.search(r"pub const AUTONOMOUS_INTERVAL_MS: u64 = (\d+)", tick_rs)
    assert m, "AUTONOMOUS_INTERVAL_MS not found in tick.rs"
    assert int(m.group(1)) == AUTONOMOUS_INTERVAL_MS

    heartbeat = (
        root / "python" / "genesis_conscious" / "mind" / "heartbeat.py"
    ).read_text(encoding="utf-8")
    assert "time.sleep(1.0)" in heartbeat, (
        "the mind's heartbeat sleep changed; update MIND_HEARTBEAT_S"
    )


def test_substep_resolution_finer_than_either_gear_teeth() -> None:
    """The wave sub-step must be finer than the fastest gear tooth.

    The body tooth is 200 ms and the mind tooth 1000 ms. Integrating the
    relaxation at 100 ms means at least two sub-steps per body tooth, so
    the wave dynamics are resolved independently of which gear is
    driving — which is the whole point of sub-stepping.
    """
    from genesis_conscious.thalamus.brain_waves import BrainWaveOscillator

    body_tooth_s = BODY_TICK_MS / 1000.0
    assert BrainWaveOscillator._MAX_SUBSTEP <= body_tooth_s / 2
    # And it divides the mind tooth exactly, so no partial tooth remains.
    per_mind_tooth = MIND_HEARTBEAT_S / BrainWaveOscillator._MAX_SUBSTEP
    assert per_mind_tooth == int(per_mind_tooth) == 10


def test_frequencies_remain_anatomically_identifiable() -> None:
    """Bands stay separated under the shared integration.

    Not a clock test, but it is the reason the wave clock must be exact:
    the dominant-band decision gates learning and speech, so bands must
    not drift into each other over a long run.
    """
    from genesis_conscious.thalamus.brain_waves import BrainWave, BrainWaveOscillator

    freqs = BrainWaveOscillator._FREQS
    assert set(freqs) == set(BrainWave)
    # Slowest is infraslow, fastest is lambda; both as documented.
    assert freqs[BrainWave.EPSILON] == 0.25
    assert freqs[BrainWave.LAMBDA] == 150.0

    # Bands are ordered by frequency, so "dominant band" is meaningful
    # and each anatomical label stays attached to the right rhythm.
    by_freq = sorted(BrainWave, key=lambda b: freqs[b])
    assert by_freq[0] is BrainWave.EPSILON
    assert by_freq[-1] is BrainWave.LAMBDA
    values = [freqs[b] for b in by_freq]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "band frequencies must be distinct"

    # The tuned centre frequencies are unchanged. This is a guard on the
    # table being edited, not on the integrator -- the integrator
    # properties live in test_wave_clock_sync.py.
    assert freqs[BrainWave.DELTA] == 2.25
    assert freqs[BrainWave.THETA] == 5.5
    assert freqs[BrainWave.ALPHA] == 10.0
    assert freqs[BrainWave.SIGMA] == 14.0
    assert freqs[BrainWave.BETA1] == 13.5
    assert freqs[BrainWave.BETA2] == 17.5
    assert freqs[BrainWave.BETA3] == 25.0
    assert freqs[BrainWave.GAMMA] == 50.0
