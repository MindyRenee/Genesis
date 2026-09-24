"""Mind — Genesis's cognitive mind, decomposed into cohesive mixins.

``Mind`` lives in ``mind/core.py``; its behavior is split across
mixin modules in this package (lifecycle, heartbeat, volition, sleep,
projects, proposals, status, conversation). Threshold constants live
in ``mind/thresholds.py``.
"""

from .core import Mind
from .thresholds import (
    AUTO_SLEEP_ADENOSINE,
    AUTO_SLEEP_CONFIRM_S,
    AUTO_SLEEP_EXIT,
    AUTO_SLEEP_MIN_AWAKE,
    AUTO_SLEEP_MIN_IDLE,
    AUTO_WAKE_ADENOSINE,
    DROWSINESS_ADENOSINE,
    DROWSINESS_CONFIRM_S,
    DROWSINESS_EXIT,
    WARN_ADENOSINE,
    WARN_AROUSAL,
    WARN_CORTISOL,
    WARN_VALENCE,
)

__all__ = [
    "AUTO_SLEEP_ADENOSINE",
    "AUTO_SLEEP_CONFIRM_S",
    "AUTO_SLEEP_EXIT",
    "AUTO_SLEEP_MIN_AWAKE",
    "AUTO_SLEEP_MIN_IDLE",
    "AUTO_WAKE_ADENOSINE",
    "DROWSINESS_ADENOSINE",
    "DROWSINESS_CONFIRM_S",
    "DROWSINESS_EXIT",
    "WARN_ADENOSINE",
    "WARN_AROUSAL",
    "WARN_CORTISOL",
    "WARN_VALENCE",
    "Mind",
]
