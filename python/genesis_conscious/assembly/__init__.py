"""Assembly — piece-arrangement puzzles on the competence substrate.

The fourth task family wired into ``TaskCompetence``: where ``spatial``
handles grids and navigation and ``planning`` handles conceptual
procedures, assembly handles "arrange parts so constraints between
them are satisfied" — the jigsaw shape. The puzzle model is honest
(reports mismatches, never pre-filters), and ``AssemblyAgent`` learns
the placement affordance by trial rather than being told the rule.
"""

from .agent import AssemblyAgent, AssemblyResult
from .puzzle import Piece, PiecePuzzle, Placement

__all__ = [
    "AssemblyAgent",
    "AssemblyResult",
    "Piece",
    "PiecePuzzle",
    "Placement",
]
