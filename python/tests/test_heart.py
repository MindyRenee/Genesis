"""The cardiac center — the field's carrier read, felt, and driven."""

import math
from types import SimpleNamespace

from brainstem.heart import (
    _CONTRACTILITY_GAIN,
    _DEPTH_NEUTRAL,
    _KEEP_CEIL,
    _KEEP_FLOOR,
    _KEEPER_NUDGE,
    _KEEPER_REFRACT_S,
    _LUCID_DRIVE,
    Heart,
)

from genesis_client.protocol import (
    CHEM_EPINEPHRINE,
    CHEM_HISTAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
)


class _FakeClient:
    """The body end of the nerve — serves the field signature."""

    def __init__(self, *, sig=None):
        self._sig = sig

    def get_field_signature(self):
        return self._sig


def _sig(
    *,
    freq: float = 10.0,
    disagreement: float = 0.01,
    signature_present: bool = True,
    integration_present: bool = True,
    current_a: float = 0.03,
):
    """A FieldSignature-shaped read — omega from the carrier Hz the
    way the daemon computes it (rad/s = 2π·Hz)."""
    return SimpleNamespace(
        signature_present=signature_present,
        integration_present=integration_present,
        quasi_static=True,
        omega_rad_s=2.0 * math.pi * freq,
        coherent_current_a=current_a,
        consensus_speed=current_a / 0.1,
        dipole_moment_am2=1.0e-9,
        b_axial_t=1.0e-9,
        e_induction_vm=1.0e-8,
        radiated_power_w=1.0e-40,
        disagreement=disagreement,
        temperature=0.05,
    )


def _heart(client, *, arousal=0.5, phase="active", impulses=None, drive=None):
    return Heart(
        client=client,
        neuro_impulse=((lambda c, m: impulses.append((c, m))) if impulses is not None else None),
        arousal=lambda: arousal,
        phase=lambda: phase,
        wave_drive=(drive.append if drive is not None else None),
    )


def test_the_beat_reads_the_field_signature():
    """Carrier, lock, and work come off the signature — the heart
    is the field's carrier frequency, measured not assumed."""
    client = _FakeClient(sig=_sig(freq=12.0, disagreement=0.0))
    heart = _heart(client)
    heart.tick(now=0.0)
    st = heart.state
    assert st.rate_hz == 12.0
    assert st.lock == 1.0  # zero disagreement — perfect consensus
    assert st.work_a == 0.03
    assert st.arrhythmic is False


def test_absent_signature_reports_a_flat_heart():
    """No channel → no reading: state resets, never a stale pulse."""
    client = _FakeClient(sig=None)
    heart = _heart(client)
    heart.tick(now=0.0)
    assert heart.state.rate_hz == 0.0
    assert heart.state.lock is None


def test_signature_without_velocity_still_gives_coherence():
    """Integration present but no measurable winding yet: the lock
    reads, the carrier does not — absent is honest, not zero."""
    client = _FakeClient(
        sig=_sig(freq=0.0, signature_present=False, disagreement=0.02)
    )
    heart = _heart(client)
    heart.tick(now=0.0)
    st = heart.state
    assert st.rate_hz == 0.0
    assert st.lock is not None and st.lock > 0.85
    assert st.work_a is None


def test_arrhythmia_is_felt_once():
    """Coherence collapse while the carrier runs — one palpitation
    pair on the transition, then silence while it lasts."""
    impulses: list = []
    client = _FakeClient(sig=_sig(disagreement=0.5))  # lock ≈ 0.23 < floor
    heart = _heart(client, impulses=impulses)
    heart.tick(now=0.0)
    heart.tick(now=5.0)
    heart.tick(now=10.0)
    # One palpitation pair across three beats — contractility kicks
    # (also NE) may surround it but never repeat the EPI half.
    epi = [i for i, imp in enumerate(impulses) if imp == (CHEM_EPINEPHRINE, 0.05)]
    assert len(epi) == 1
    i = epi[0]
    assert impulses[i - 1] == (CHEM_NOREPINEPHRINE, 0.05)
    assert heart.state.arrhythmic


def test_healthy_lock_is_silent():
    impulses: list = []
    client = _FakeClient(sig=_sig(disagreement=0.01))
    heart = _heart(client, impulses=impulses)
    heart.tick(now=0.0)
    heart.tick(now=5.0)
    # No palpitation pair — only possible contractility kicks.
    assert not any(c == CHEM_EPINEPHRINE for c, _ in impulses)
    assert not heart.state.arrhythmic


def test_contractility_is_expressed_as_ne_tone():
    """Engagement deepens sympathetic drive — the demand above
    neutral maps to a positive NE kick on the first assert."""
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, arousal=0.9, impulses=impulses)
    heart.tick(now=0.0)
    depth = 0.10 + 0.75 * 0.9
    expected = (depth - _DEPTH_NEUTRAL) * _CONTRACTILITY_GAIN
    assert (CHEM_NOREPINEPHRINE, expected) in impulses
    assert heart.state.depth == depth


def test_rest_withdraws_contractility():
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, arousal=0.05, impulses=impulses)
    heart.tick(now=0.0)
    depth = 0.10 + 0.75 * 0.05
    expected = (depth - _DEPTH_NEUTRAL) * _CONTRACTILITY_GAIN
    assert expected < 0
    assert (CHEM_NOREPINEPHRINE, expected) in impulses


def test_contractility_reasserts_on_keepalive_not_every_beat():
    """The demand is quantized: quiet arousal doesn't re-fire, a
    material move does, and the 30 s keep-alive re-asserts."""
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, arousal=0.5, impulses=impulses)
    heart.tick(now=0.0)
    ne_kicks = [m for c, m in impulses if c == CHEM_NOREPINEPHRINE]
    assert len(ne_kicks) == 1  # first assert only
    heart.tick(now=5.0)
    heart.tick(now=10.0)
    assert len([m for c, m in impulses if c == CHEM_NOREPINEPHRINE]) == 1
    heart.tick(now=35.0)  # past the keep-alive
    assert len([m for c, m in impulses if c == CHEM_NOREPINEPHRINE]) == 2


def test_the_keep_owns_contractility_while_armed():
    """While the hypnagogic keep holds, _drive yields — no NE
    contractility kicks, the wave owns the depth."""
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    heart.tick(now=5.0)
    assert not any(c == CHEM_NOREPINEPHRINE for c, _ in impulses)


def test_no_impulse_channel_still_reads_the_pulse():
    """Without neuro_impulse the heart still senses — it just
    cannot push. Afferent is not contingent on efferent."""
    client = _FakeClient(sig=_sig())
    heart = _heart(client)  # impulses=None
    heart.tick(now=0.0)
    assert heart.state.rate_hz == 10.0
    assert heart.state.lock is not None


def test_arm_marks_the_keep_and_releases_cleanly():
    """The keep is central — arming is the watch itself; there is
    no body write left to fail."""
    client = _FakeClient(sig=_sig())
    heart = _heart(client)
    assert heart.arm_hypnagogic()
    assert heart.state.pulse_armed
    assert heart.disarm_hypnagogic()
    assert not heart.state.pulse_armed


def test_release_nudge_above_the_ceiling():
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, arousal=_KEEP_CEIL + 0.05, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert (CHEM_OREXIN, -_KEEPER_NUDGE) in impulses
    assert (CHEM_HISTAMINE, -_KEEPER_NUDGE) in impulses


def test_catch_nudge_below_the_floor():
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, arousal=_KEEP_FLOOR - 0.05, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert (CHEM_OREXIN, _KEEPER_NUDGE) in impulses
    assert (CHEM_HISTAMINE, _KEEPER_NUDGE) in impulses


def test_the_band_interior_is_left_alone():
    impulses: list = []
    client = _FakeClient(sig=_sig())
    mid = (_KEEP_FLOOR + _KEEP_CEIL) / 2
    heart = _heart(client, arousal=mid, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert not any(c in (CHEM_OREXIN, CHEM_HISTAMINE) for c, _ in impulses)


def test_corrections_are_refractory_limited():
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, arousal=_KEEP_CEIL + 0.05, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    nudges = [i for i in impulses if i[1] == -_KEEPER_NUDGE]
    assert len(nudges) == 2  # one orexin+histamine pair
    heart.tick(now=5.0)
    assert len([i for i in impulses if i[1] == -_KEEPER_NUDGE]) == 2
    heart.tick(now=_KEEPER_REFRACT_S + 5.0)
    assert len([i for i in impulses if i[1] == -_KEEPER_NUDGE]) == 4


def test_a_real_crossing_disarms_the_keep():
    client = _FakeClient(sig=_sig())
    heart = _heart(client, phase="nrem")
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert not heart.state.pulse_armed


def test_disarm_restores_static_depth():
    impulses: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, impulses=impulses)
    heart.arm_hypnagogic()
    assert heart.disarm_hypnagogic()
    heart.tick(now=0.0)
    # _drive re-asserts the contractility demand as an NE kick.
    assert any(c == CHEM_NOREPINEPHRINE for c, _ in impulses)


def test_meditation_takes_up_the_pulse_and_waking_releases_it():
    """The keep rides the sit: /meditate arms it, wake disarms it."""
    from sleep.sleep import SleepMixin

    heart_client = _FakeClient(sig=_sig())
    heart = _heart(heart_client)
    mind = SimpleNamespace(
        _is_meditating=False,
        _is_sleeping=False,
        _stop_latched=False,
        _meditation_start_time=0.0,
        _suppress_volition=False,
        _last_rest_time=0.0,
        client=SimpleNamespace(set_zone=lambda z: None),
        inner_life=SimpleNamespace(pause=lambda: None, resume=lambda: None),
        learner=SimpleNamespace(pause=lambda: None, resume=lambda: None),
        regulator=SimpleNamespace(notify_rest_ended=lambda: None),
        _learner_neuro_impulse=lambda c, v: None,
        heart=heart,
    )
    SleepMixin.meditate(mind)
    assert mind._is_meditating
    assert heart.state.pulse_armed

    SleepMixin.wake_from_meditation(mind)
    assert not mind._is_meditating
    assert not heart.state.pulse_armed


def test_the_lucid_drive_beats_while_armed_and_stops_when_released():
    drives: list = []
    client = _FakeClient(sig=_sig())
    heart = _heart(client, drive=drives)
    heart.tick(now=0.0)
    assert drives == []  # disarmed — no drive
    heart.arm_hypnagogic()
    heart.tick(now=5.0)
    heart.tick(now=10.0)
    assert drives == [_LUCID_DRIVE, _LUCID_DRIVE]
    heart.disarm_hypnagogic()
    heart.tick(now=15.0)
    assert len(drives) == 2  # the drive releases with the keep
