"""Quantities — composing an exact total under a capacity.

The eighth task family wired into ``TaskCompetence``: the number-sense
domain of kindergarten — "give me 5," "which groups make 10."
``Basket`` is honest: overflow is rejected (the capacity is real),
fitting-but-dead-ending costs a lift later, and ``QuantitiesAgent``
learns the fit→cost affordance by trial — including that "fits but
leaves a remainder nothing can fill" is not the same as "fits."
"""

from .agent import QuantitiesAgent, QuantitiesResult
from .world import Basket, Group, Option

__all__ = [
    "Basket",
    "Group",
    "Option",
    "QuantitiesAgent",
    "QuantitiesResult",
]
