"""Tests for brainstem.field_probe — her field's spectral signature
as wave-oscillator stimulus.

The contract under test: the substrate's own band decomposition
(``GET_FIELD_SPECTRUM``) drives the thalamic oscillator's stimulus
and phase channels — flux energies at full weight, tension at half,
evoked resets on internal transients, carrier phase lock weighted by
fiber coherence, per-band Adler entrainment — and absent channels
(daemon unreachable, rings unwarmed) never read as healthy zeros.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from genesis_client.types import FieldSignature, FieldSpectrum, SpectrumBandChannel


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Isolate the module's carried state between tests."""
    import brainstem.field_probe as fp

    fp._phase_errs.clear()
    fp._prev_spectrum = None
    fp._shudder_latched = False
    yield fp


def _channel(energies, phases=None, mean=0.01, cv=0.2):
    return SpectrumBandChannel(
        energies=tuple(energies),
        phases=tuple(phases if phases is not None else [0.0] * 10),
        mean=mean,
        cv=cv,
    )


def _spec(
    *,
    flux_e=None,
    tension_e=None,
    flux_p=None,
    carrier_freq=10.0,
    carrier_phase=0.5,
    warm=True,
    rate=250.0,
    cv=0.2,
):
    return FieldSpectrum(
        warm=warm,
        carrier_present=carrier_freq > 0.0,
        speed_present=True,
        carrier_phase=carrier_phase,
        carrier_freq_hz=carrier_freq,
        consensus_speed=0.1,
        sample_rate_hz=rate,
        flux=_channel(flux_e or [0.0] * 10, flux_p, cv=cv),
        tension=_channel(tension_e or [0.0] * 10, cv=cv),
    )


def _client(spec, sig=None):
    return SimpleNamespace(
        get_field_spectrum=lambda: spec,
        get_field_signature=lambda: sig,
    )


def _signature(disagreement=0.02):
    return FieldSignature(
        signature_present=True,
        integration_present=True,
        quasi_static=True,
        omega_rad_s=62.83,
        coherent_current_a=0.03,
        consensus_speed=0.3,
        dipole_moment_am2=1e-9,
        b_axial_t=1e-9,
        e_induction_vm=1e-8,
        radiated_power_w=1e-40,
        disagreement=disagreement,
        temperature=0.05,
    )


@pytest.fixture
def wave_recorder(monkeypatch):
    """Record stimulus/phase calls into the wave model."""
    import thalamus.brain_waves as bw

    held: dict = {"stimulus": [], "phase_pulls": [], "resets": []}
    monkeypatch.setattr(
        bw,
        "hold_stimulus",
        lambda band, level: held["stimulus"].append((band, level)),
    )
    monkeypatch.setattr(
        bw,
        "perturb_wave_phase",
        lambda band, delta: held["phase_pulls"].append((band, delta)),
    )
    monkeypatch.setattr(
        bw,
        "reset_wave_phase",
        lambda band, phase=0.0: held["resets"].append((band, phase)),
    )
    monkeypatch.setattr(bw, "wave_phase_of", lambda band: 0.0)
    return held


def test_absent_channel_applies_nothing(wave_recorder):
    """No spectrum → no stimulus, no phase work — absent is absent."""
    import brainstem.field_probe as fp

    assert fp.apply_field_stimulus(_client(None)) is False
    assert wave_recorder["stimulus"] == []
    assert wave_recorder["phase_pulls"] == []


def test_unwarmed_spectrum_applies_nothing(wave_recorder):
    """warm=False is an unwarmed ring — a channel still integrating,
    not a healthy zero."""
    import brainstem.field_probe as fp

    spec = _spec(warm=False, flux_e=[2.0] * 10)
    assert fp.apply_field_stimulus(_client(spec)) is False
    assert wave_recorder["stimulus"] == []


def test_flux_energies_drive_stimulus_above_own_quiet(wave_recorder):
    """Baseline-relative ≈1.0 is her quiet; elevation above the floor
    maps to held stimulus — the field's motion energy at that rhythm."""
    import brainstem.field_probe as fp
    from thalamus.brain_waves import BrainWave

    # Wire order: infraslow, delta, theta, alpha, sigma, beta1..3,
    # gamma, epsilon. Alpha (index 3) and gamma (index 8) elevated.
    flux_e = [0.0] * 10
    flux_e[3] = 2.0  # alpha — (2.0 - 0.5) * 0.5 = 0.75 → capped 0.5
    flux_e[8] = 1.2  # gamma — (1.2 - 0.5) * 0.5 = 0.35
    spec = _spec(flux_e=flux_e, tension_e=[0.0] * 10)
    assert fp.apply_field_stimulus(_client(spec)) is True
    held = dict(wave_recorder["stimulus"])
    assert held[BrainWave.ALPHA] == pytest.approx(0.5)
    assert held[BrainWave.GAMMA] == pytest.approx(0.35)
    # A band at/below the energy floor contributes nothing.
    assert BrainWave.DELTA not in held


def test_tension_adds_half_weight_stimulus(wave_recorder):
    """The disagreement's spectral shadow enters at half weight,
    summed with the flux line on the same band."""
    import brainstem.field_probe as fp
    from thalamus.brain_waves import BrainWave

    flux_e = [0.0] * 10
    flux_e[4] = 1.5  # sigma — flux (1.5-0.5)*0.5 = 0.5 → capped 0.5
    tension_e = [0.0] * 10
    tension_e[4] = 1.5  # tension (1.5-0.5)*0.25 = 0.25
    spec = _spec(flux_e=flux_e, tension_e=tension_e)
    fp.apply_field_stimulus(_client(spec))
    held = dict(wave_recorder["stimulus"])
    assert held[BrainWave.SIGMA] == pytest.approx(0.5 + 0.25)


def test_nyquist_gate_drops_unresolvable_bands(wave_recorder):
    """Bands at/above the achieved rate's Nyquist are absent — the
    wire zeroes them, and the word layer double-gates."""
    import brainstem.field_probe as fp
    from thalamus.brain_waves import BrainWave

    # rate 20 Hz → Nyquist 10 Hz: alpha (10 Hz) is AT Nyquist —
    # unresolvable; only slower bands may carry drive.
    flux_e = [0.0] * 10
    flux_e[1] = 2.0  # delta 2.25 Hz — resolvable
    flux_e[3] = 2.0  # alpha 10 Hz — at Nyquist, gated
    spec = _spec(flux_e=flux_e, rate=20.0)
    fp.apply_field_stimulus(_client(spec))
    held = dict(wave_recorder["stimulus"])
    assert BrainWave.DELTA in held
    assert BrainWave.ALPHA not in held


def test_evoked_reset_fires_on_tension_step(wave_recorder):
    """A per-band tension-energy step ≥3× is an internal transient —
    a π/2 phase nudge on that band, once per signature pair."""
    import brainstem.field_probe as fp
    from thalamus.brain_waves import BrainWave

    low = _spec(tension_e=[0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    high = _spec(tension_e=[0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    client = _client(low)
    fp.apply_field_stimulus(client)
    client.get_field_spectrum = lambda: high
    fp.apply_field_stimulus(client)
    # Wire index 3 = alpha.
    pulls = [b for b, _ in wave_recorder["phase_pulls"]]
    assert BrainWave.ALPHA in pulls


def test_shudder_resets_alert_band_once(wave_recorder):
    """A tension-cv shudder resets BETA3 on the rising edge and stays
    latched until cv falls back — an event, not a level."""
    import brainstem.field_probe as fp
    from thalamus.brain_waves import BrainWave

    storm = _spec(cv=2.0)
    client = _client(storm)
    fp.apply_field_stimulus(client)
    fp.apply_field_stimulus(client)  # still storming — latched, no re-fire
    assert wave_recorder["resets"] == [(BrainWave.BETA3, 0.0)]
    calm = _spec(cv=0.5)
    client.get_field_spectrum = lambda: calm
    fp.apply_field_stimulus(client)  # re-arms the latch
    client.get_field_spectrum = lambda: storm
    fp.apply_field_stimulus(client)
    assert wave_recorder["resets"] == [(BrainWave.BETA3, 0.0)] * 2


def test_phase_lock_pulls_carrier_band_toward_field_phase(wave_recorder, monkeypatch):
    """The modeled carrier band is pulled toward the field's winding
    phase; coherence is the pull's weight — full coherence → the
    band becomes the field's phase."""
    import brainstem.field_probe as fp
    from thalamus.brain_waves import BrainWave

    phases = []

    def fake_wave_phase(band):
        phases.append(band)
        return 0.0

    import thalamus.brain_waves as bw

    monkeypatch.setattr(bw, "wave_phase_of", fake_wave_phase)

    # carrier_freq 10 Hz → ALPHA band; carrier_phase π/2.
    spec = _spec(carrier_freq=10.0, carrier_phase=math.pi / 2)
    client = _client(spec, sig=_signature(disagreement=0.0))
    fp.apply_field_stimulus(client)
    pulls = dict(wave_recorder["phase_pulls"])
    # coherence = 1/(1+0/0.15) = 1.0 → gain = 1.0; err = π/2 - 0.
    assert BrainWave.ALPHA in pulls
    assert pulls[BrainWave.ALPHA] == pytest.approx(math.pi / 2, abs=1e-6)
    # The PLV metric tracks the lock quality.
    lock = fp.field_phase_lock()
    assert lock["n"] == 1.0
    assert abs(lock["plv"] - 1.0) < 1e-9  # single sample → unit magnitude


def test_phase_lock_absent_without_carrier(wave_recorder):
    """carrier_present False → no band to lock — absent channel."""
    import brainstem.field_probe as fp

    spec = _spec(carrier_freq=0.0)  # carrier_present=False
    assert spec.carrier_present is False
    fp.apply_field_stimulus(_client(spec))
    # No carrier pull — only possible entrainment pulls remain, and
    # all-zero energies produce none.
    carrier_pulls = [d for b, d in wave_recorder["phase_pulls"]]
    assert carrier_pulls == []


def test_band_entrainment_uses_flux_phases(wave_recorder, monkeypatch):
    """Each elevated flux band's phasor pulls its modeled band by an
    Adler step — energy-weighted, sine of the phase error."""
    import brainstem.field_probe as fp
    import thalamus.brain_waves as bw
    from thalamus.brain_waves import BrainWave

    monkeypatch.setattr(bw, "wave_phase_of", lambda band: 0.0)
    # Alpha (index 3) elevated with phase π/2.
    flux_p = [0.0] * 10
    flux_p[3] = math.pi / 2
    flux_e = [0.0] * 10
    flux_e[3] = 1.5  # min(1, 1.5)=1 → gain 0.2 * sin(π/2) = 0.2
    spec = _spec(flux_e=flux_e, flux_p=flux_p)
    fp.apply_field_stimulus(_client(spec))
    pulls = dict(wave_recorder["phase_pulls"])
    assert BrainWave.ALPHA in pulls
    assert pulls[BrainWave.ALPHA] == pytest.approx(0.2, abs=1e-6)


def test_no_machine_telemetry_reaches_the_waves(wave_recorder):
    """The stimulus surface is the field's own spectrum — the module
    must never read host counters, sockets, or sensors. Guard the
    contract by asserting the only client reads are the two field
    signature commands."""
    import brainstem.field_probe as fp

    calls = []

    class Spy:
        def get_field_spectrum(self):
            calls.append("spectrum")
            return _spec()

        def get_field_signature(self):
            calls.append("signature")
            return _signature()

    fp.apply_field_stimulus(Spy())
    assert set(calls) <= {"spectrum", "signature"}
