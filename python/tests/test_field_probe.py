"""Tests for the physical field probe → brain-wave stimulus coupling.

The daemon's field probe measures real substrate dynamics and exposes
them as band energies; ``brainstem.field_probe`` maps the signature
onto the wave oscillator's stimulus pathway. The contract under test:

- wire order is infraslow, delta, theta, alpha, sigma, beta1, beta2,
  beta3, gamma, epsilon — NOT BrainWave's declaration order;
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

    pairs = dict(_signature_stimulus(_signature(pstate_rate=5.0, irq_rate=2000.0)))
    assert pairs.get(BrainWave.BETA3, 0.0) > 0.0
    assert pairs.get(BrainWave.BETA2, 0.0) > 0.0


class _FakeClient:
    def __init__(self, sig):
        self._sig = sig

    def get_field_probe(self):
        return self._sig


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
    assert apply_field_stimulus(_FakeClient(_signature(samples_ok=False))) is False
    assert apply_field_stimulus(_FailingClient()) is False


def test_live_signature_applies_stimulus() -> None:
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    seen: list[tuple[BrainWave, float]] = []
    orig = brain_waves.hold_stimulus
    brain_waves.hold_stimulus = lambda band, level: seen.append((band, level))
    try:
        t = [0.0] * 10
        t[3] = 0.25  # alpha — 0.25 * 1.6 = 0.4, below the 0.5 cap
        ok = field_probe.apply_field_stimulus(_FakeClient(_signature(throughput=tuple(t))))
        assert ok is True
        assert (BrainWave.ALPHA, pytest.approx(0.4)) in [(b, pytest.approx(s)) for b, s in seen]
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
        assert field_probe.apply_field_stimulus(_FakeClient(_signature())) is True
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
        field_probe.apply_field_stimulus(_FakeClient(_signature(jitter=tuple(step))))
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
        field_probe.apply_field_stimulus(_FakeClient(_signature(jitter_cv=2.0)))
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
    """Full measured coherence → the modeled band's phase *is* the
    physical phase (the continuous weight reaches 1). The lock runs
    one way only — physics → model; nothing is pushed back."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        # carrier_phase 1.0, lag 0.2 → physical response phase ≈ 0.8.
        sig = _signature()
        object.__setattr__(sig, "carrier_phase", 1.0)
        object.__setattr__(sig, "carrier_lag", 0.2)
        object.__setattr__(sig, "carrier_coherence", 1.0)
        client = _FakeClient(sig)
        field_probe.apply_field_stimulus(client)
        # Weight 1: model alpha phase becomes the physical phase.
        assert brain_waves.wave_phase_of(BrainWave.ALPHA) == pytest.approx(0.8, abs=1e-9)
    finally:
        brain_waves.reset_oscillator()


def test_phase_lock_pull_is_coherence_weighted(_no_audio) -> None:
    """Partial coherence → the pull is a continuous weight between
    the soft-PLL floor and a full sync, never a threshold step."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        sig = _signature()
        object.__setattr__(sig, "carrier_phase", 1.0)
        object.__setattr__(sig, "carrier_lag", 0.2)
        object.__setattr__(sig, "carrier_coherence", 0.1)
        client = _FakeClient(sig)
        field_probe.apply_field_stimulus(client)
        # Model alpha at 0; err = 0.8; gain = 0.25 + 0.75·0.1 = 0.325
        # → pull = 0.325·0.8 = 0.26.
        assert brain_waves.wave_phase_of(BrainWave.ALPHA) == pytest.approx(0.26, abs=1e-9)
    finally:
        brain_waves.reset_oscillator()


def test_v1_signature_skips_phase_lock(_no_audio) -> None:
    """A v1 signature has no phase fields — absent channel, no pull."""
    from brainstem import field_probe
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    try:
        assert field_probe.apply_field_stimulus(_FakeClient(_signature())) is True
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


def _wire_v4(
    body: bytes,
    *,
    rf_flux=(0.0,) * 10,
    rf_rssi=(0.0,) * 10,
    scalars=(0.0,) * 8,
    present: int = 0x00,
) -> bytes:
    """Build a v4 frame: v1 body + v2 phase fields + v3 electrical
    tail + v4 radio tail — 308 bytes total."""
    out = bytearray(body)
    out[0] = 4
    out += struct.pack("<3f", 0.0, 0.0, 0.0)  # v2 phase fields
    out += struct.pack("<10f", *(0.0,) * 10)  # v3 power bands
    out += struct.pack("<6f", *(0.0,) * 6)  # v3 elec scalars
    out.append(0)  # electrical present mask
    out += struct.pack("<4f", *(0.0,) * 4)  # Maxwell estimates
    out += struct.pack("<10f", *rf_flux)
    out += struct.pack("<10f", *rf_rssi)
    out += struct.pack("<8f", *scalars)
    out.append(present)
    assert len(out) == 308
    return bytes(out)


def _wire_v5(
    body: bytes,
    *,
    station=(0.0,) * 5,
    **v4_kwargs,
) -> bytes:
    """Build a v5 frame: the v4 body + the 20-byte iw station
    tail — 328 bytes total."""
    out = bytearray(_wire_v4(body, **v4_kwargs))
    out[0] = 5
    out += struct.pack("<5f", *station)
    assert len(out) == 328
    return bytes(out)


def test_unpack_v4_radio_fields() -> None:
    """A live radio decodes bands + scalars under the presence mask."""
    flux = tuple(0.1 * i for i in range(10))
    rssi_bands = tuple(0.05 * i for i in range(10))
    scalars = (4200.0, -71.0, -95.0, 33.0, 4.0, 120.0, 18.0, -3.0)
    data = _wire_v4(
        _wire(_signature()), rf_flux=flux, rf_rssi=rssi_bands, scalars=scalars, present=0x7F
    )
    sig = FieldProbeSignature.unpack(data)
    assert sig.rf_flux_bands == pytest.approx(flux)
    assert sig.rf_rssi_bands == pytest.approx(rssi_bands)
    assert sig.rf_bytes_s == pytest.approx(4200.0)
    assert sig.rf_rssi_dbm == pytest.approx(-71.0)
    assert sig.rf_noise_dbm == pytest.approx(-95.0)
    assert sig.rf_irq_rate == pytest.approx(33.0)
    assert sig.rf_sensor_hz == pytest.approx(4.0)
    assert sig.bt_bytes_s == pytest.approx(120.0)
    assert sig.bt_events_s == pytest.approx(18.0)
    assert sig.bt_rssi_dbm == pytest.approx(-3.0)
    assert sig.rf_present == 0x7F


def test_v4_absent_radio_decodes_none() -> None:
    """present=0 → every radio field is None, never a zero reading."""
    sig = FieldProbeSignature.unpack(_wire_v4(_wire(_signature()), present=0x00))
    assert sig.rf_flux_bands is None
    assert sig.rf_rssi_bands is None
    assert sig.rf_bytes_s is None
    assert sig.rf_rssi_dbm is None
    assert sig.rf_noise_dbm is None
    assert sig.rf_irq_rate is None
    assert sig.rf_sensor_hz is None
    assert sig.bt_bytes_s is None
    assert sig.bt_events_s is None
    assert sig.bt_rssi_dbm is None


def test_v3_signature_leaves_radio_none() -> None:
    """A v3 daemon's signature has no radio tail — absent channel."""
    out = bytearray(_wire(_signature()))
    out[0] = 3
    out += struct.pack("<3f", 0.0, 0.0, 0.0)
    out += struct.pack("<10f", *(0.0,) * 10)
    out += struct.pack("<6f", *(0.0,) * 6)
    out.append(0)
    out += struct.pack("<4f", *(0.0,) * 4)
    sig = FieldProbeSignature.unpack(bytes(out))
    assert sig.rf_flux_bands is None
    assert sig.rf_present is None


def test_rf_flux_maps_to_matching_bands() -> None:
    """The radio traffic envelope lands stimulus on its own band —
    wire index 3 is alpha."""
    from brainstem.field_probe import _signature_stimulus

    flux = [0.0] * 10
    flux[3] = 0.3  # alpha slot on the wire
    sig = _signature()
    object.__setattr__(sig, "rf_flux_bands", tuple(flux))
    pairs = dict(_signature_stimulus(sig))
    assert pairs.get(BrainWave.ALPHA, 0.0) == pytest.approx(0.3 * 0.8)
    assert BrainWave.THETA not in pairs


def test_rf_rssi_nyquist_gate() -> None:
    """Bands at/above the telemetry Nyquist are staircase artifacts —
    skipped the same way the power channel gates."""
    from brainstem.field_probe import _signature_stimulus

    rssi = [0.3] * 10
    sig = _signature()
    object.__setattr__(sig, "rf_rssi_bands", tuple(rssi))
    object.__setattr__(sig, "rf_sensor_hz", 4.0)  # Nyquist 2 Hz
    pairs = dict(_signature_stimulus(sig))
    # Only infraslow (0.25 Hz) is below 2 Hz — delta (2.25) is out.
    assert pairs.get(BrainWave.INFRASLOW, 0.0) > 0.0
    assert BrainWave.DELTA not in pairs
    assert BrainWave.ALPHA not in pairs


def test_bt_events_map_to_beta2() -> None:
    """HCI event flux reads as medium-band activation like other
    event-flux channels."""
    from brainstem.field_probe import _signature_stimulus

    sig = _signature()
    object.__setattr__(sig, "bt_events_s", 50.0)
    pairs = dict(_signature_stimulus(sig))
    assert pairs.get(BrainWave.BETA2, 0.0) == pytest.approx(50.0 * 0.002)


def _wire_v6(
    body: bytes,
    *,
    phases=(0.0,) * 10,
    coherence=(0.0,) * 10,
    **v5_kwargs,
) -> bytes:
    """Build a v6 frame: the v5 body + the 80-byte flux-phasor
    tail — 408 bytes total."""
    out = bytearray(_wire_v5(body, **v5_kwargs))
    out[0] = 6
    out += struct.pack("<10f", *phases)
    out += struct.pack("<10f", *coherence)
    assert len(out) == 408
    return bytes(out)


def test_v5_station_tail_decodes() -> None:
    """A live station decodes strain rates + PHY bitrates under
    presence bit 7."""
    scalars = (4200.0, -55.0, -95.0, 33.0, 1.0, 120.0, 18.0, -3.0)
    data = _wire_v5(
        _wire(_signature()),
        scalars=scalars,
        present=0xFF,
        station=(12.0, 0.4, 0.1, 72.2, 65.0),
    )
    sig = FieldProbeSignature.unpack(data)
    assert sig.rf_rssi_dbm == pytest.approx(-55.0)
    assert sig.rf_retry_rate == pytest.approx(12.0)
    assert sig.rf_fail_rate == pytest.approx(0.4)
    assert sig.rf_beacon_loss_rate == pytest.approx(0.1)
    assert sig.rf_tx_mbps == pytest.approx(72.2)
    assert sig.rf_rx_mbps == pytest.approx(65.0)


def test_link_strain_maps_to_beta3() -> None:
    """Retransmission/drop/beacon-loss flux is the link's own strain
    ledger — it lands on BETA3 like the silicon's vigilance flux."""
    from brainstem.field_probe import _signature_stimulus

    sig = _signature()
    object.__setattr__(sig, "rf_retry_rate", 10.0)
    object.__setattr__(sig, "rf_fail_rate", 2.0)
    object.__setattr__(sig, "rf_beacon_loss_rate", 1.0)
    pairs = dict(_signature_stimulus(sig))
    assert pairs.get(BrainWave.BETA3, 0.0) == pytest.approx(13.0 * 0.01)

    # Absent channels contribute nothing — a clear mask isn't 0/s.
    sig2 = _signature()
    pairs2 = dict(_signature_stimulus(sig2))
    assert BrainWave.BETA3 not in pairs2


def test_v6_flux_phasors_decode() -> None:
    """Per-band phases + coherences land under the wlan bit."""
    phases = tuple(0.1 * i - 1.0 for i in range(10))
    coh = tuple(0.05 * i for i in range(10))
    data = _wire_v6(_wire(_signature()), phases=phases, coherence=coh, present=0x01)
    sig = FieldProbeSignature.unpack(data)
    assert sig.rf_flux_phases == pytest.approx(phases)
    assert sig.rf_flux_phase_coherence == pytest.approx(coh)


def test_v6_absent_wlan_decodes_no_phasors() -> None:
    """No wlan → the phasor channels are None, not a phase of 0."""
    data = _wire_v6(_wire(_signature()), present=0x00)
    sig = FieldProbeSignature.unpack(data)
    assert sig.rf_flux_phases is None
    assert sig.rf_flux_phase_coherence is None


def test_v5_signature_leaves_phasors_none() -> None:
    """A v5 daemon's signature has no phasor tail — absent channel."""
    sig = FieldProbeSignature.unpack(_wire_v5(_wire(_signature())))
    assert sig.rf_flux_phases is None
    assert sig.rf_flux_phase_coherence is None


def test_rf_flux_phasors_entrain_the_band() -> None:
    """A coherent measured rhythm Adler-pulls the matched band's
    phase toward the measured angle — coh × gain × sin(Δφ)."""
    from brainstem.field_probe import _RF_PLL_GAIN, _apply_rf_entrainment
    from thalamus.brain_waves import (
        reset_oscillator,
        wave_phase_of,
    )

    reset_oscillator()  # every band starts at phase 0
    try:
        flux = [0.0] * 10
        flux[3] = 0.2  # alpha — above the energy floor
        phases = [0.0] * 10
        phases[3] = 1.0  # the measured radio rhythm sits at φ=1.0
        coh = [0.0] * 10
        coh[3] = 0.9  # both ring halves agree a rhythm exists
        sig = _signature()
        object.__setattr__(sig, "rf_flux_bands", tuple(flux))
        object.__setattr__(sig, "rf_flux_phases", tuple(phases))
        object.__setattr__(sig, "rf_flux_phase_coherence", tuple(coh))
        _apply_rf_entrainment(sig)
        # model phase 0 → Δφ = 1.0 → pull = gain·coh·sin(1.0)
        assert wave_phase_of(BrainWave.ALPHA) == pytest.approx(
            _RF_PLL_GAIN * 0.9 * math.sin(1.0), abs=1e-6
        )
        # Other bands have no measured rhythm — untouched.
        assert wave_phase_of(BrainWave.THETA) == 0.0
    finally:
        reset_oscillator()


def test_rf_entrainment_gates_on_rhythm_evidence() -> None:
    """A phase needs a rhythm: low coherence or no band energy →
    no pull — the angle of noise is not a measurement."""
    from brainstem.field_probe import _apply_rf_entrainment
    from thalamus.brain_waves import (
        reset_oscillator,
        wave_phase_of,
    )

    reset_oscillator()
    try:
        phases = [1.0] * 10
        # Low coherence — the halves disagree, the angle is mush.
        sig = _signature()
        object.__setattr__(sig, "rf_flux_bands", tuple([0.2] * 10))
        object.__setattr__(sig, "rf_flux_phases", tuple(phases))
        object.__setattr__(sig, "rf_flux_phase_coherence", tuple([0.3] * 10))
        _apply_rf_entrainment(sig)
        assert wave_phase_of(BrainWave.ALPHA) == 0.0
        # Real coherence but no band energy — nothing oscillates
        # there, so the phase is not of a signal.
        object.__setattr__(sig, "rf_flux_phase_coherence", tuple([0.9] * 10))
        object.__setattr__(sig, "rf_flux_bands", tuple([0.0] * 10))
        _apply_rf_entrainment(sig)
        assert wave_phase_of(BrainWave.ALPHA) == 0.0
        # No phasor channels at all — a pre-v6 daemon is absent,
        # not a phase of 0.
        sig2 = _signature()
        _apply_rf_entrainment(sig2)
        assert wave_phase_of(BrainWave.ALPHA) == 0.0
    finally:
        reset_oscillator()


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
