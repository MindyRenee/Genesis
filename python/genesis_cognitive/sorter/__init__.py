"""Sorting — matching under conjunctive constraints on the competence
substrate.

The fifth task family wired into ``TaskCompetence``: where ``spatial``
handles grids, ``planning`` handles conceptual procedures, and
``assembly`` handles piece arrangement, sorting handles "select what
satisfies every stated constraint" — the shape-sorter shape. The
world is honest (``insert`` reports how many constraints were met,
never pre-filters), the lid distractor is learnable rather than
forbidden, and ``SorterAgent`` learns the match→cost affordance by
trial instead of being told the rule.

``PerceptualSorter`` is the same physics seen through the visual
cortex: candidates stop leaking the oracle's ``matched`` count and
``PerceptualSorterAgent`` works from rendered views, learning
seen-similarity→cost. ``teach`` lets the world name its own pieces —
"square" becomes a silhouette it recognizes, not just a string.
"""

from .agent import PerceptualSorterAgent, SorterAgent, SorterResult
from .puzzle import (
    Block,
    Insertion,
    PerceptualSorter,
    ShapeSorter,
    Slot,
    render_object,
)

__all__ = [
    "Block",
    "Insertion",
    "PerceptualSorter",
    "PerceptualSorterAgent",
    "ShapeSorter",
    "Slot",
    "SorterAgent",
    "SorterResult",
    "render_object",
]
