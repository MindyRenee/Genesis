"""Sequence — pattern continuation on the competence substrate.

The sixth task family wired into ``TaskCompetence``: where ``sorter``
handles matching under constraints, sequence handles "continue what
repeats" — the bead-string shape. The line's hidden rule is the world;
the agent only perceives a candidate's lookback and learns which
lookback predicts acceptance — discovering the period empirically.
"""

from .agent import PatternAgent, PatternResult
from .puzzle import PatternLine, Placement, Token

__all__ = [
    "PatternAgent",
    "PatternLine",
    "PatternResult",
    "Placement",
    "Token",
]
