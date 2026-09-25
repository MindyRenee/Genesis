"""Relations — arranging named things so stated relations hold.

The seventh task family wired into ``TaskCompetence``: the
"directions" domain of kindergarten — "put the star left of the
moon, the moon next to the sun" — plus seriation ("line the cups
up fewest to most") through the attribute-ordering ``ordered``
relation. ``Arrangement`` is honest: placements always succeed and
goal satisfaction is checked against the board, so the agent
discovers that satisfying *stated* relations is the task.
``RelationsAgent`` learns the agree→cost affordance by trial and
repairs violations by lifting the more suspect of the two parties —
a violation is a property of a pair of placements, not of a thing.
"""

from .agent import RelationsAgent, RelationsResult
from .world import Arrangement, Goal, Placement, Thing

__all__ = [
    "Arrangement",
    "Goal",
    "Placement",
    "RelationsAgent",
    "RelationsResult",
    "Thing",
]
