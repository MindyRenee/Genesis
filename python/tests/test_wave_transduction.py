"""Wave→chemical transduction — release flux, not level tracking.

The band power is the presynaptic rate: UP chemicals get release
flux ``k·P·Δt`` (cadence-invariant — the ticker only delivers), DOWN
chemicals get enhanced first-order clearance ``−k·P·C·Δt``. Levels
are never commanded; the body's own kinetics decide where they settle.
"""

from __future__ import annotations

import pytest
from hypothalamus.transduction import DRIVE_RATE, RELEASE_GAIN, transduce_waves, wave_impulses
from thalamus.brain_waves import BrainWave, BrainWaveState

DT = 1.0


def _state(**powers: float) -> BrainWaveState:
    """A minimal wave state with the given band powers."""
    p = {b: powers.get(b.value, 0.0) for b in BrainWave}
    dom = max(p, key=lambda b: p[b])
    return BrainWaveState(
        dominant=dom,
        secondary=dom,
        powers=p,
        focus=0.5,
        integration=0.5,
        consolidation=0.5,
        label="test",
        description="test",
    )


def _by_chem(impulses):
    return dict(impulses)


class TestReleaseFlux:
    def test_beta_releases_wake_chemicals(self):
        """Sustained beta drive commands release flux on its affiliates."""
        impulses = _by_chem(wave_impulses(_state(beta2=0.5), {}, DT))
        from genesis_client.protocol import CHEM_CRH, CHEM_HISTAMINE, CHEM_OREXIN

        assert impulses[CHEM_OREXIN] > 0
        assert impulses[CHEM_HISTAMINE] > 0
        assert impulses[CHEM_CRH] > 0

    def test_quiet_band_emits_nothing(self):
        """No band power → no release flux. The body's own clearance
        withdraws the hold — the wave never pushes a level down."""
        impulses = _by_chem(wave_impulses(_state(delta=0.8), {}, DT))
        from genesis_client.protocol import CHEM_OREXIN

        assert CHEM_OREXIN not in impulses

    def test_flux_is_rate_times_dt(self):
        """The impulse is release rate × elapsed — cadence-invariant."""
        from genesis_client.protocol import CHEM_OREXIN

        one = _by_chem(wave_impulses(_state(beta2=0.4), {}, 1.0))
        half = _by_chem(wave_impulses(_state(beta2=0.4), {}, 0.5))
        assert one[CHEM_OREXIN] == pytest.approx(2.0 * half[CHEM_OREXIN])

    def test_flux_is_capped_by_authority_rate(self):
        """Beyond the peptide's rate ceiling the flux saturates."""
        from genesis_client.protocol import CHEM_OREXIN

        # orexin cap = max_impulse(0.1) * DRIVE_RATE(0.2) = 0.02/s;
        # RELEASE_GAIN·P = 0.25·0.9 = 0.225 → capped at 0.02·dt.
        impulses = _by_chem(wave_impulses(_state(beta2=0.9), {}, DT))
        assert impulses[CHEM_OREXIN] == pytest.approx(0.1 * DRIVE_RATE * DT)

    def test_release_ignores_current_level(self):
        """Feedforward, not feedback: release flux does not read C —
        the body owns the level, the wave owns the rate."""
        from genesis_client.protocol import CHEM_OREXIN

        low = _by_chem(wave_impulses(_state(beta2=0.3), {"orexin": 0.05}, DT))
        high = _by_chem(wave_impulses(_state(beta2=0.3), {"orexin": 0.9}, DT))
        assert low[CHEM_OREXIN] == pytest.approx(high[CHEM_OREXIN])

    def test_beta_aggregates_sub_bands(self):
        """Declared 'beta' affinity sums all three sub-band powers."""
        from genesis_client.protocol import CHEM_OREXIN

        split = _by_chem(wave_impulses(_state(beta1=0.15, beta2=0.15, beta3=0.15), {}, DT))
        joint = _by_chem(wave_impulses(_state(beta2=0.45), {}, DT))
        assert split[CHEM_OREXIN] == pytest.approx(joint[CHEM_OREXIN])


class TestDownSuppression:
    def test_cortisol_is_never_released(self):
        """THE regression guard: delta dominance suppresses cortisol.

        Cortisol is the DOWN member of the delta pair — the wave can
        only enhance its clearance, never pump it during sleep.
        """
        impulses = _by_chem(wave_impulses(_state(delta=0.9), {"cortisol": 0.3}, DT))
        from genesis_client.protocol import CHEM_CORTISOL

        assert impulses[CHEM_CORTISOL] < 0

    def test_cortisol_suppression_is_floor_aware(self):
        """Suppression is first-order in substrate — it dies with C."""
        impulses = _by_chem(wave_impulses(_state(delta=0.9), {"cortisol": 0.01}, DT))
        from genesis_client.protocol import CHEM_CORTISOL

        # rate = RELEASE_GAIN·P·C = 0.25·0.9·0.01 = 0.00225, under cap.
        assert impulses[CHEM_CORTISOL] == pytest.approx(-0.00225 * DT)

    def test_quiet_delta_leaves_cortisol_alone(self):
        impulses = _by_chem(wave_impulses(_state(beta2=0.6), {"cortisol": 0.3}, DT))
        from genesis_client.protocol import CHEM_CORTISOL

        assert CHEM_CORTISOL not in impulses

    def test_melatonin_is_the_up_member(self):
        """Same band, opposite sign: delta releases melatonin."""
        impulses = _by_chem(wave_impulses(_state(delta=0.5), {}, DT))
        from genesis_client.protocol import CHEM_MELATONIN

        assert impulses[CHEM_MELATONIN] > 0


class TestChannel:
    def test_transduce_emits_only_true_impulses(self):
        class FakeClient:
            def __init__(self):
                self.sent = []

            def neuro_impulse(self, chem, magnitude):
                self.sent.append((chem, magnitude))
                return True

        client = FakeClient()
        n = transduce_waves(client, _state(beta2=0.6), {}, DT)
        assert n == len(client.sent)
        assert all(abs(m) > 0 for _, m in client.sent)

    def test_dead_body_stops_the_wave_command(self):
        class DeadClient:
            def neuro_impulse(self, chem, magnitude):
                raise ConnectionError("gone")

        assert transduce_waves(DeadClient(), _state(beta2=0.6), {}, DT) == 0
