"""LimbicSystem — affect regulation and felt-state machinery.

The emotional regulator owns its neurochemistry: it reads body state,
drives the affect dynamics, and injects neuromodulatory impulses.
Sentiment analysis (``limbic_system/sentiment.py``) and the feeling
reporter live in this package; the reporter is grown by the frontal
projection that feeds it.
"""

from __future__ import annotations

from typing import Any

from limbic_system.emotional_regulator import EmotionalRegulator


class LimbicSystem:
    """The limbic region: the felt side of every state.

    Attributes:
        regulator: affect dynamics — reads the body, drives emotion.
        feeling_reporter: felt-state narration (grown by the frontal
            projection — it reports *frontal's* felt states).
    """

    def __init__(
        self,
        get_emotion,
        neuro_impulse,
        get_state,
        config,
        socket_path: str,
    ) -> None:
        self.regulator = EmotionalRegulator(
            get_emotion=get_emotion,
            neuro_impulse=neuro_impulse,
            get_state=get_state,
            config=config,
            socket_path=socket_path,
        )
        self.feeling_reporter = None

    def grow_feeling_reporter(self, **deps: Any):
        """Grow the felt-state reporter from the frontal projection."""
        from limbic_system.feeling_reporter import FeelingReporter

        self.feeling_reporter = FeelingReporter(**deps)
        return self.feeling_reporter
