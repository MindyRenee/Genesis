"""Wave→chemical transduction — the hypothalamic release channel.

Covers the control law in ``hypothalamus.transduction``:

- UP peptides (wave_sign +1) track their band power as the commanded
  level — saturated exact tracking, rate-limited by max_impulse.
- DOWN peptides (wave_sign -1, cortisol) are suppressed by their
  band, never released — the cortisol regression guard.
- Band aliases aggregate sub-band power (beta = BETA1+2+3).
- The quiet-band half of the up/down rule: no power → the peptide's
  hold is withdrawn (negative impulse), not merely un-driven.
"""

from __future__ import annotations

import pytest
from hypothalamus.transduction import DRIVE_RATE, transduce_waves, wave_impulses
from thalamus.brain_waves import BrainWave, BrainWaveState


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


class TestUpTracking:
    def test_beta_releases_wake_peptides(self):
        """Sustained beta drive commands orexin/histamine/CRH release."""
        impulses = _by_chem(wave_impulses(_state(beta2=0.5), {}))
        # All three beta peptides get positive impulses when their
        # command (0.5) exceeds the current level (baseline).
        from genesis_client.protocol import CHEM_CRH, CHEM_HISTAMINE, CHEM_OREXIN

        assert impulses[CHEM_OREXIN] > 0
        assert impulses[CHEM_HISTAMINE] > 0
        assert impulses[CHEM_CRH] > 0

    def test_quiet_band_withdraws_hold(self):
        """The down half of every up: no band power → negative impulse.

        Orexin must fall in sleep, not merely stop rising — the wave
        commands the level, so zero command means withdrawal.
        """
        impulses = _by_chem(wave_impulses(_state(delta=0.8), {}))
        from genesis_client.protocol import CHEM_OREXIN

        assert impulses[CHEM_OREXIN] < 0

    def test_tracking_is_exact_within_budget(self):
        """Within the rate cap the impulse is exactly r - C."""
        from genesis_client.protocol import CHEM_OREXIN

        impulses = _by_chem(wave_impulses(_state(beta2=0.4), {"orexin": 0.3}))
        # r - C = 0.1 > cap (0.1 * 0.2 = 0.02) → saturated at cap.
        assert impulses[CHEM_OREXIN] == pytest.approx(0.1 * DRIVE_RATE)

    def test_already_at_command_is_silent(self):
        impulses = _by_chem(wave_impulses(_state(beta2=0.3), {"orexin": 0.3}))
        from genesis_client.protocol import CHEM_OREXIN

        assert CHEM_OREXIN not in impulses

    def test_beta_aggregates_sub_bands(self):
        """Declared 'beta' affinity sums all three sub-band powers."""
        from genesis_client.protocol import CHEM_OREXIN

        split = _by_chem(wave_impulses(_state(beta1=0.15, beta2=0.15, beta3=0.15), {}))
        joint = _by_chem(wave_impulses(_state(beta2=0.45), {}))
        assert split[CHEM_OREXIN] == pytest.approx(joint[CHEM_OREXIN])


class TestDownSuppression:
    def test_cortisol_is_never_released(self):
        """THE regression guard: delta dominance suppresses cortisol.

        Cortisol is the DOWN member of the delta pair — the wave can
        only push it toward its floor, never pump it during sleep.
        """
        impulses = _by_chem(wave_impulses(_state(delta=0.9), {"cortisol": 0.3}))
        from genesis_client.protocol import CHEM_CORTISOL

        assert impulses[CHEM_CORTISOL] < 0

    def test_cortisol_suppression_is_floor_aware(self):
        """Suppression can never drive below zero."""
        impulses = _by_chem(wave_impulses(_state(delta=0.9), {"cortisol": 0.01}))
        from genesis_client.protocol import CHEM_CORTISOL

        assert impulses[CHEM_CORTISOL] >= -0.01

    def test_quiet_delta_leaves_cortisol_alone(self):
        impulses = _by_chem(wave_impulses(_state(beta2=0.6), {"cortisol": 0.3}))
        from genesis_client.protocol import CHEM_CORTISOL

        assert CHEM_CORTISOL not in impulses

    def test_melatonin_is_the_up_member(self):
        """Same band, opposite sign: delta releases melatonin."""
        impulses = _by_chem(wave_impulses(_state(delta=0.5), {}))
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
        n = transduce_waves(client, _state(beta2=0.6), {})
        assert n == len(client.sent)
        assert all(abs(m) > 0 for _, m in client.sent)

    def test_dead_body_stops_the_wave_command(self):
        class DeadClient:
            def neuro_impulse(self, chem, magnitude):
                raise ConnectionError("gone")

        assert transduce_waves(DeadClient(), _state(beta2=0.6), {}) == 0
