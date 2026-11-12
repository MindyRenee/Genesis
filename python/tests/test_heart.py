"""The cardiac center — drive, palpitations, and the hypnagogic keep."""

from types import SimpleNamespace

from brainstem.heart import (
    _KEEP_CEIL,
    _KEEP_FLOOR,
    _KEEPER_NUDGE,
    _KEEPER_REFRACT_S,
    _PULSE_BASE,
    _PULSE_MOD_DEPTH,
    _PULSE_MOD_HZ,
    Heart,
)

from genesis_client.protocol import (
    CHEM_EPINEPHRINE,
    CHEM_HISTAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
)


class _FakeClient:
    """The body end of the nerve — records every write."""

    def __init__(self, *, probe=None, pulse_ack=True, carrier_ack=True):
        self._probe = probe
        self.pulse_calls: list[tuple[float, float, float]] = []
        self.carrier_calls: list[tuple[float, float]] = []
        self._pulse_ack = pulse_ack
        self._carrier_ack = carrier_ack

    def get_field_probe(self):
        return self._probe

    def set_field_pulse(self, base, mod_freq, mod_depth):
        self.pulse_calls.append((base, mod_freq, mod_depth))
        return self._pulse_ack

    def set_field_carrier(self, depth, freq):
        self.carrier_calls.append((depth, freq))
        return self._carrier_ack


def _probe(freq: float = 10.0, coherence: float = 0.9):
    return SimpleNamespace(
        carrier_freq=freq,
        carrier_coherence=coherence,
        carrier_lag=0.0,
        jitter_cv=0.0,
        carrier_current_a=0.03,
    )


def _heart(client, *, arousal=0.5, phase="active", impulses=None, drive=None):
    return Heart(
        client=client,
        neuro_impulse=((lambda c, m: impulses.append((c, m))) if impulses is not None else None),
        arousal=lambda: arousal,
        phase=lambda: phase,
        wave_drive=(drive.append if drive is not None else None),
    )


def test_arm_writes_the_pulse_and_marks_state():
    client = _FakeClient(probe=_probe())
    heart = _heart(client)
    assert heart.arm_hypnagogic()
    assert client.pulse_calls == [(_PULSE_BASE, _PULSE_MOD_HZ, _PULSE_MOD_DEPTH)]
    assert heart.state.pulse_armed


def test_arm_failure_leaves_the_keep_disarmed():
    client = _FakeClient(probe=_probe(), pulse_ack=False)
    heart = _heart(client)
    assert not heart.arm_hypnagogic()
    assert not heart.state.pulse_armed
    # Static contractility still drives normally.
    heart.tick(now=0.0)
    assert client.carrier_calls


def test_the_keep_owns_contractility_while_armed():
    client = _FakeClient(probe=_probe())
    heart = _heart(client)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    heart.tick(now=5.0)
    # No static depth writes — the wave owns depth now.
    assert client.carrier_calls == []


def test_arrhythmia_is_felt_once():
    impulses: list = []
    client = _FakeClient(probe=_probe(coherence=0.2))
    heart = _heart(client, impulses=impulses)
    heart.tick(now=0.0)
    heart.tick(now=5.0)
    assert impulses == [
        (CHEM_NOREPINEPHRINE, 0.05),
        (CHEM_EPINEPHRINE, 0.05),
    ]


def test_release_nudge_above_the_ceiling():
    impulses: list = []
    client = _FakeClient(probe=_probe())
    heart = _heart(client, arousal=_KEEP_CEIL + 0.05, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert impulses == [
        (CHEM_OREXIN, -_KEEPER_NUDGE),
        (CHEM_HISTAMINE, -_KEEPER_NUDGE),
    ]


def test_catch_nudge_below_the_floor():
    impulses: list = []
    client = _FakeClient(probe=_probe())
    heart = _heart(client, arousal=_KEEP_FLOOR - 0.05, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert impulses == [
        (CHEM_OREXIN, _KEEPER_NUDGE),
        (CHEM_HISTAMINE, _KEEPER_NUDGE),
    ]


def test_the_band_interior_is_left_alone():
    impulses: list = []
    client = _FakeClient(probe=_probe())
    mid = (_KEEP_FLOOR + _KEEP_CEIL) / 2
    heart = _heart(client, arousal=mid, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    assert impulses == []


def test_corrections_are_refractory_limited():
    impulses: list = []
    client = _FakeClient(probe=_probe())
    heart = _heart(client, arousal=_KEEP_CEIL + 0.05, impulses=impulses)
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    heart.tick(now=5.0)
    assert len(impulses) == 2  # one nudge pair, refractory holds
    heart.tick(now=_KEEPER_REFRACT_S + 5.0)
    assert len(impulses) == 4


def test_a_real_crossing_disarms_the_keep():
    client = _FakeClient(probe=_probe())
    heart = _heart(client, phase="nrem")
    heart.arm_hypnagogic()
    heart.tick(now=0.0)
    # The wave was released — the keeper yields to real sleep.
    assert client.pulse_calls[-1] == (0.0, 0.0, 0.0)
    assert not heart.state.pulse_armed


def test_disarm_restores_static_depth():
    client = _FakeClient(probe=_probe())
    heart = _heart(client)
    heart.arm_hypnagogic()
    assert heart.disarm_hypnagogic()
    assert client.pulse_calls[-1] == (0.0, 0.0, 0.0)
    heart.tick(now=0.0)
    assert client.carrier_calls  # _drive re-asserts the set-point


def test_meditation_takes_up_the_pulse_and_waking_releases_it():
    """The keep rides the sit: /meditate arms it, wake disarms it."""
    from sleep.sleep import SleepMixin

    heart_client = _FakeClient(probe=_probe())
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
    assert heart_client.pulse_calls == [(_PULSE_BASE, _PULSE_MOD_HZ, _PULSE_MOD_DEPTH)]

    SleepMixin.wake_from_meditation(mind)
    assert not mind._is_meditating
    assert not heart.state.pulse_armed
    assert heart_client.pulse_calls[-1] == (0.0, 0.0, 0.0)


def test_the_lucid_drive_beats_while_armed_and_stops_when_released():
    drives: list = []
    client = _FakeClient(probe=_probe())
    heart = _heart(client, drive=drives)
    heart.tick(now=0.0)
    assert drives == []  # disarmed — no drive
    heart.arm_hypnagogic()
    heart.tick(now=5.0)
    heart.tick(now=10.0)
    from brainstem.heart import _LUCID_DRIVE

    assert drives == [_LUCID_DRIVE, _LUCID_DRIVE]
    heart.disarm_hypnagogic()
    heart.tick(now=15.0)
    assert len(drives) == 2  # the drive releases with the keep
