"""Sensorimotor prediction-error layer — the cerebellar comparator.

The cerebellum receives an efference copy of every motor command
alongside the reafferent sensory stream, and its output is the
*mismatch* — graded, not binary. A stride that lands ninety degrees
off prediction and a half-step jitter are both "unexpected", but
only one of them reorganizes behavior; the layer reports magnitude
so cortex can tell them apart.

Two kinds of signal leave here:

- **Per-trial error magnitude** — how far the observed reafference
  sat from the predicted one. Strong error is an orienting event:
  a norepinephrine impulse marks it the way locus-coeruleus phasic
  bursts mark violated expectations (Aston-Jones & Cohen, 2005).
- **Per-action calibration** — an error EMA per action. One bump is
  the world's novelty; sustained error on the same action means the
  forward model's entry for it is stale — recalibration territory.
  An action can only earn calibration back by being retried, the
  same way its error accrued.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_CHEM_NOREPINEPHRINE = 2

# Surprise strong enough to orient toward — same operating point as
# the semantic layer's phasic-NE gate.
_MOTOR_NE_THRESHOLD = 0.7
# An action whose running error sits above this has a stale forward
# model — worth re-babbling even though it nominally has trials.
_MISCALIBRATION_THRESHOLD = 0.5


class SensorimotorLayer:
    """Compares predicted and observed reafference for each act.

    ``compare`` is the whole contract: given the action issued, the
    displacement the forward model predicted, and the displacement
    the frame actually carried, it returns a graded error in
    ``[0, 1]`` and folds it into the action's calibration estimate.
    Chemistry coupling is optional — without an impulse sink the
    arithmetic still runs; with one, strong error becomes an
    orienting signal instead of a silent statistic.
    """

    def __init__(
        self,
        on_neuro_impulse: Callable[[int, float], Any] | None = None,
        decay: float = 0.2,
    ) -> None:
        self._on_neuro_impulse = on_neuro_impulse
        self._decay = decay
        self._error_ema: dict[str, float] = {}

    def compare(
        self,
        action: str,
        expected: tuple[float, float],
        observed: tuple[float, float],
    ) -> float:
        """Grade one reafferent mismatch; return its magnitude.

        Error is the miss distance normalized by the predicted
        stride: a full refusal (predicted a step, observed nothing)
        and an orthogonal landing (predicted right, went down) are
        both maximal surprises; a partial slide is partial error.
        """
        miss = math.hypot(
            observed[0] - expected[0], observed[1] - expected[1]
        )
        norm = math.hypot(*expected)
        magnitude = 1.0 if norm == 0 else min(1.0, miss / norm)
        ema = self._error_ema.get(action, 0.0)
        self._error_ema[action] = (
            self._decay * magnitude + (1.0 - self._decay) * ema
        )
        if (
            magnitude >= _MOTOR_NE_THRESHOLD
            and self._on_neuro_impulse is not None
        ):
            try:
                self._on_neuro_impulse(
                    _CHEM_NOREPINEPHRINE,
                    0.02 + (magnitude - _MOTOR_NE_THRESHOLD) * 0.06,
                )
            except (OSError, ConnectionError, RuntimeError):
                logger.debug("motor NE impulse failed", exc_info=True)
        return magnitude

    def miscalibrated(self, action: str) -> bool:
        """The forward model's entry for this action is stale.

        Calibration can only be earned back by acting — the EMA
        moves on trials, not on time — so a flagged action stays a
        recalibration question until the world answers otherwise.
        """
        return (
            self._error_ema.get(action, 0.0)
            >= _MISCALIBRATION_THRESHOLD
        )

    def error_of(self, action: str) -> float:
        """The action's running prediction-error estimate."""
        return self._error_ema.get(action, 0.0)

    def reset(self) -> None:
        """A new world, a new body of physics — drop all estimates."""
        self._error_ema.clear()
