"""Tests for the physical field probe → brain-wave stimulus coupling.

The daemon's field probe measures real substrate dynamics and exposes
them as band energies; ``brainstem.field_probe`` maps the signature
onto the wave oscillator's stimulus pathway. The contract under test:

- wire order is epsilon, delta, theta, alpha, sigma, beta1, beta2,
  beta3, gamma, lambda — NOT BrainWave's declaration order;
- an absent/unwarmed channel injects nothing (never a healthy zero);
- a live signature lands stimulus on the matching bands.
"""

import math
import struct

import pytest
from thalamus.brain_waves import BrainWave

from genesis_client.types import FieldProbeSignature


def _signature(
    throughput=(0.0,) * 10,
    jitter=(0.0,) * 10,
    *,
    samples_ok: bool = True,
    pstate_rate: float = 0.0,
    irq_rate: float = 0.0,
    carrier: float = 10.0,
    jitter_cv: float = 0.5,
) -> FieldProbeSignature:
    return FieldProbeSignature(
        throughput_bands=throughput,
        jitter_bands=jitter,
        throughput_mean=2e7,
        jitter_cv=jitter_cv,
        pstate_rate=pstate_rate,
        irq_rate=irq_rate,
        carrier_freq=carrier,
        samples_ok=samples_ok,
    )


def _wire(sig: FieldProbeSignature) -> bytes:
    out = bytearray([1])
    out += struct.pack("<20f", *sig.throughput_bands, *sig.jitter_bands)
    out += struct.pack(
        "<5f",
        sig.throughput_mean,
        sig.jitter_cv,
        sig.pstate_rate,
        sig.irq_rate,
        sig.carrier_freq,
    )
    out.append(1 if sig.samples_ok else 0)
    return bytes(out)


def test_unpack_wire_layout() -> None:
    sig = _signature(
        throughput=tuple(0.01 * i for i in range(10)),
        pstate_rate=3.0,
        irq_rate=1500.0,
        carrier=6.0,
    )
    rt = FieldProbeSignature.unpack(_wire(sig))
    assert rt.throughput_bands == pytest.approx(sig.throughput_bands)
    assert rt.jitter_bands == pytest.approx(sig.jitter_bands)
    assert rt.pstate_rate == pytest.approx(3.0)
    assert rt.irq_rate == pytest.approx(1500.0)
    assert rt.carrier_freq == pytest.approx(6.0)
    assert rt.samples_ok is True


def test_unpack_rejects_short_and_bad_version() -> None:
    with pytest.raises(ValueError):
        FieldProbeSignature.unpack(b"\x01" * 50)
    with pytest.raises(ValueError):
        FieldProbeSignature.unpack(bytes([9]) + b"\x00" * 101)


def test_wire_order_maps_to_correct_bands() -> None:
    """Wire index 3 is alpha; index 8 is gamma — declaration order
    would get this wrong."""
    from brainstem.field_probe import _signature_stimulus

    t = [0.0] * 10
    t[3] = 0.4  # alpha slot on the wire
    t[8] = 0.3  # gamma slot on the wire
    pairs = dict(_signature_stimulus(_signature(throughput=tuple(t))))
    assert BrainWave.ALPHA in pairs and pairs[BrainWave.ALPHA] > 0.0
    assert BrainWave.GAMMA in pairs and pairs[BrainWave.GAMMA] > 0.0
    # No stimulus lands where nothing was reported.
    assert BrainWave.DELTA not in pairs


def test_slow_channels_map_to_alert_bands() -> None:
    from brainstem.field_probe import _signature_stimulus

    pairs = dict(
        _signature_stimulus(
            _signature(pstate_rate=5.0, irq_rate=2000.0)
        )
    )
    assert pairs.get(BrainWave.BETA3, 0.0) > 0.0
    assert pairs.get(BrainWave.BETA2, 0.0) > 0.0


class _FakeClient:
    def __init__(self, sig):
        self._sig = sig
        self.pushed: list[float] = []

    def get_field_probe(self):
        return self._sig

    def set_field_phase(self, phase: float) -> bool:
        self.pushed.append(phase)
        return True


class _FailingClient:
    def get_field_probe(self):
        raise ConnectionError("daemon down")


@pytest.fixture(autouse=True)
def _no_audio(monkeypatch):
    """Isolate the acoustic channel: apply_field_stimulus tries it
    independently, and on a machine with a live mic it would open
    real hardware mid-test. Stub it absent by default."""
    import sys
    import types

    mod = types.ModuleType("brainstem.field_audio")
    mod.apply_audio_stimulus = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "brainstem.field_audio", mod)
    return mod


def test_absent_channel_injects_nothing() -> None:
    from brainstem.field_probe import apply_field_stimulus

    assert apply_field_stimulus(_FakeClient(None)) is False
    assert apply_field_stimulus(
        _FakeClient(_signature(samples_ok=False))
    ) is False
    assert apply_field_stimulus(_FailingClient()) is False


def test_live_signature_applies_stimulus() -> None:
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    seen: list[tuple[BrainWave, float]] = []
    orig = brain_waves.hold_stimulus
    brain_waves.hold_stimulus = lambda band, level: seen.append(
        (band, level)
    )
    try:
        t = [0.0] * 10
        t[3] = 0.25  # alpha — 0.25 * 1.6 = 0.4, below the 0.5 cap
        ok = field_probe.apply_field_stimulus(
            _FakeClient(_signature(throughput=tuple(t)))
        )
        assert ok is True
        assert (BrainWave.ALPHA, pytest.approx(0.4)) in [
            (b, pytest.approx(s)) for b, s in seen
        ]
    finally:
        brain_waves.hold_stimulus = orig
        brain_waves.reset_oscillator()


def test_channels_apply_independently(monkeypatch, _no_audio) -> None:
    """Each channel is real or absent on its own — either one alone
    is enough to count as applied."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        # Probe live, audio absent → applied.
        assert field_probe.apply_field_stimulus(
            _FakeClient(_signature())
        ) is True
        # Probe absent, audio live → applied.
        _no_audio.apply_audio_stimulus = lambda: True  # type: ignore[attr-defined]
        assert field_probe.apply_field_stimulus(_FakeClient(None)) is True
        assert field_probe.apply_field_stimulus(_FailingClient()) is True
        # Both absent → not applied.
        _no_audio.apply_audio_stimulus = lambda: False  # type: ignore[attr-defined]
        assert field_probe.apply_field_stimulus(_FakeClient(None)) is False
    finally:
        brain_waves.reset_oscillator()


def test_per_band_jitter_step_nudges_phase(monkeypatch, _no_audio) -> None:
    """A ≥3× jump in a band's jitter energy evoked-shifts that band's
    phase by π/2 — the physical-transient channel."""
    import math

    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        base = [0.1] * 10
        # First signature establishes the baseline (prev must exist).
        field_probe._prev_signature = _signature(jitter=tuple(base))
        step = list(base)
        step[5] = 0.4  # wire slot 5 = BETA1: 4× baseline → evoked
        field_probe.apply_field_stimulus(
            _FakeClient(_signature(jitter=tuple(step)))
        )
        phase = brain_waves.wave_phase_of(BrainWave.BETA1)
        assert phase == pytest.approx(math.pi / 2, abs=1e-9)
        # Untouched bands stay at their initial phase.
        assert brain_waves.wave_phase_of(BrainWave.ALPHA) == 0.0
    finally:
        field_probe._prev_signature = None
        brain_waves.reset_oscillator()


def test_global_shudder_resets_alert_band(monkeypatch, _no_audio) -> None:
    """jitter_cv ≥ 1.5 — a whole-scheduler shudder — resets the
    broadband alert band to phase zero."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        brain_waves.perturb_wave_phase(BrainWave.BETA3, 2.5)
        field_probe._prev_signature = _signature()
        field_probe.apply_field_stimulus(
            _FakeClient(_signature(jitter_cv=2.0))
        )
        assert brain_waves.wave_phase_of(BrainWave.BETA3) == 0.0
    finally:
        field_probe._prev_signature = None
        brain_waves.reset_oscillator()


def test_carrier_band_mapping() -> None:
    from brainstem.field_probe import _carrier_band

    # The real carrier setpoints: wake alpha, REM theta, NREM delta —
    # each sitting exactly on its Goertzel bin center.
    assert _carrier_band(10.0) is BrainWave.ALPHA
    assert _carrier_band(5.5) is BrainWave.THETA
    assert _carrier_band(2.25) is BrainWave.DELTA
    assert _carrier_band(0.0) is None  # carrier off → no band
    assert _carrier_band(45.0) is None  # far from any band


def test_phase_lock_syncs_when_coherent(_no_audio) -> None:
    """High measured coherence → the modeled band's phase *is* the
    physical phase (rung 2: hard sync), and it's pushed back."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        # carrier_phase 1.0, lag 0.2 → physical response phase ≈ 0.8.
        sig = _signature()
        object.__setattr__(sig, "carrier_phase", 1.0)
        object.__setattr__(sig, "carrier_lag", 0.2)
        object.__setattr__(sig, "carrier_coherence", 0.5)
        client = _FakeClient(sig)
        field_probe.apply_field_stimulus(client)
        # Sync gate: model alpha phase becomes the physical phase.
        assert brain_waves.wave_phase_of(BrainWave.ALPHA) == pytest.approx(
            0.8, abs=1e-9
        )
        # The physical phase is pushed back — physics follows model.
        assert client.pushed == [pytest.approx(0.8)]
    finally:
        brain_waves.reset_oscillator()


def test_phase_lock_pulls_below_coherence_gate(_no_audio) -> None:
    """Low coherence → soft pull only; the model keeps its own state."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        sig = _signature()
        object.__setattr__(sig, "carrier_phase", 1.0)
        object.__setattr__(sig, "carrier_lag", 0.2)
        object.__setattr__(sig, "carrier_coherence", 0.1)  # below gate
        client = _FakeClient(sig)
        field_probe.apply_field_stimulus(client)
        # Model alpha at phase 0; err = 0.8 → pull = 0.25·0.8 = 0.2.
        assert brain_waves.wave_phase_of(BrainWave.ALPHA) == pytest.approx(
            0.2, abs=1e-9
        )
        assert client.pushed == [pytest.approx(0.2)]
    finally:
        brain_waves.reset_oscillator()


def test_v1_signature_skips_phase_lock(_no_audio) -> None:
    """A v1 signature has no phase fields — absent channel, no pull."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        assert field_probe.apply_field_stimulus(
            _FakeClient(_signature())
        ) is True
        # All phases untouched — the lock leg saw an absent channel.
        assert brain_waves.wave_phase_of(BrainWave.ALPHA) == 0.0
    finally:
        brain_waves.reset_oscillator()


def test_field_phase_lock_metric(_no_audio) -> None:
    from brainstem import field_probe

    field_probe._phase_errs.clear()
    assert field_probe.field_phase_lock()["plv"] == 0.0
    # Constant phase error → rigidly locked, PLV ≈ 1.
    for _ in range(20):
        field_probe._phase_errs.append(0.4)
    assert field_probe.field_phase_lock()["plv"] == pytest.approx(1.0)
    # Uniformly spread errors → no lock, PLV → 0.
    field_probe._phase_errs.clear()
    for i in range(32):
        field_probe._phase_errs.append(2 * math.pi * i / 32)
    assert field_probe.field_phase_lock()["plv"] < 0.05
    field_probe._phase_errs.clear()


def test_unpack_v2_phase_fields() -> None:
    """The v2 wire carries lag/phase/coherence; v1 leaves them None."""
    v1 = FieldProbeSignature.unpack(_wire(_signature()))
    assert v1.carrier_lag is None and v1.carrier_phase is None

    out = bytearray(_wire(_signature()))
    out[0] = 2
    out += struct.pack("<3f", 0.35, 1.2, 0.7)
    v2 = FieldProbeSignature.unpack(bytes(out))
    assert v2.carrier_lag == pytest.approx(0.35)
    assert v2.carrier_phase == pytest.approx(1.2)
    assert v2.carrier_coherence == pytest.approx(0.7)


def test_phase_ops_wrap_modulo_tau(_no_audio) -> None:
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        brain_waves.perturb_wave_phase(BrainWave.THETA, 7.0)
        phase = brain_waves.wave_phase_of(BrainWave.THETA)
        assert phase == pytest.approx(7.0 - 2 * math.pi, abs=1e-9)
        brain_waves.reset_wave_phase(BrainWave.THETA, -0.5)
        phase = brain_waves.wave_phase_of(BrainWave.THETA)
        assert phase == pytest.approx(2 * math.pi - 0.5, abs=1e-9)
    finally:
        brain_waves.reset_oscillator()


def test_assess_without_client_is_unchanged() -> None:
    """No client → no field fetch; the call shape stays pure."""
    from thalamus.brain_waves import assess_brain_waves, reset_oscillator

    from genesis_client.protocol import PHASE_ACTIVE
    from genesis_client.types import NeuroSummary

    reset_oscillator()
    summary = NeuroSummary(
        arousal=0.6,
        valence=0.0,
        global_tone=0.5,
        plasticity_gate=0.5,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=PHASE_ACTIVE,
    )
    state = assess_brain_waves(summary)
    assert state is not None
    reset_oscillator()
