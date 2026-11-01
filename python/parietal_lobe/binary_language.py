"""Binary Mathematical Language (BML): a learnable symbolic substrate for Genesis.

BML borrows the *finite bit-cell idea* of six-dot Braille, not Braille's
human-assigned letter mapping. A cell is a six-bit pattern; typed frames
compose cells into observations, concepts, relations, and utterances.

This module provides an encoding and a small evidence-based association
store. It does not claim that bit patterns alone produce language: meanings
must be grounded in observations and tested against later evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from math import log2


CELL_BITS = 6
CELL_COUNT = 1 << CELL_BITS
CELL_MASK = CELL_COUNT - 1


class Role(IntEnum):
    """Typed roles for compositional frames; values are structural markers."""
    OBSERVATION = 1
    ENTITY = 2
    PROPERTY = 3
    ACTION = 4
    RELATION = 5
    AGENT = 6
    PATIENT = 7
    TIME = 8
    NEGATION = 9
    QUESTION = 10
    UNKNOWN = 11


@dataclass(frozen=True, slots=True)
class BitCell:
    """One six-bit pattern, independent of any preassigned natural-language word."""
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("cell value must be an integer")
        if not 0 <= self.value <= CELL_MASK:
            raise ValueError("six-bit cell must be in range 0..63")

    @classmethod
    def from_dots(cls, raised: Iterable[int]) -> BitCell:
        """Encode Braille-style dot positions numbered 1..6 into a bit cell."""
        value = 0
        for dot in raised:
            if isinstance(dot, bool) or not isinstance(dot, int) or dot not in range(1, 7):
                raise ValueError("dot positions must be integers in 1..6")
            value |= 1 << (dot - 1)
        return cls(value)

    @property
    def raised_dots(self) -> tuple[int, ...]:
        return tuple(i + 1 for i in range(CELL_BITS) if self.value & (1 << i))

    @property
    def bits(self) -> str:
        return format(self.value, "06b")

    @property
    def information_capacity_bits(self) -> int:
        """Maximum raw information per unconstrained six-bit cell."""
        return log2(CELL_COUNT)


@dataclass(frozen=True, slots=True)
class Symbol:
    """A locally assigned symbol; its meaning is an evidence-backed hypothesis."""
    cell: BitCell
    role: Role = Role.UNKNOWN


@dataclass(frozen=True, slots=True)
class Frame:
    """A typed, ordered composition of symbols."""
    role: Role
    symbols: tuple[Symbol, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise TypeError("frame role must be a Role")
        if not all(isinstance(s, Symbol) for s in self.symbols):
            raise TypeError("frame symbols must be Symbol instances")


@dataclass(slots=True)
class Association:
    """Evidence linking a pattern to a grounded referent or observed outcome."""
    pattern: tuple[int, ...]
    referent: str
    positive_evidence: int = 0
    counter_evidence: int = 0
    contexts: set[str] = field(default_factory=set)

    @property
    def support(self) -> float:
        """Smoothed support ratio, not a calibrated probability."""
        return (self.positive_evidence + 1) / (
            self.positive_evidence + self.counter_evidence + 2
        )


@dataclass(slots=True)
class GroundingMemory:
    """Store and revise symbol-grounding hypotheses without hiding uncertainty.

    Evidence should be supplied by Genesis from perception/action outcomes.
    A positive observation supports a hypothesis; a counterexample weakens it.
    The support score is a simple bookkeeping statistic, not proof of meaning.
    """
    associations: dict[tuple[tuple[int, ...], str], Association] = field(default_factory=dict)

    def observe(
        self,
        pattern: Iterable[BitCell],
        referent: str,
        *,
        supported: bool,
        context: str,
    ) -> Association:
        cells = tuple(cell.value for cell in pattern)
        if not cells:
            raise ValueError("pattern must contain at least one cell")
        if not referent.strip():
            raise ValueError("referent must not be empty")
        key = (cells, referent)
        item = self.associations.get(key)
        if item is None:
            item = Association(pattern=cells, referent=referent)
            self.associations[key] = item
        if supported:
            item.positive_evidence += 1
        else:
            item.counter_evidence += 1
        item.contexts.add(context)
        return item

    def hypotheses(self, pattern: Iterable[BitCell]) -> list[Association]:
        """Return known candidate referents, strongest support first."""
        cells = tuple(cell.value for cell in pattern)
        matches = [a for (p, _), a in self.associations.items() if p == cells]
        return sorted(matches, key=lambda a: (-a.support, a.referent))


def encode_bits(value: int, width: int = CELL_BITS) -> tuple[int, ...]:
    """Encode a non-negative integer into fixed-width six-bit cells, big-endian."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("value must be a non-negative integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    bit_string = format(value, "b")
    padded = bit_string.zfill(((len(bit_string) + CELL_BITS - 1) // CELL_BITS) * CELL_BITS)
    cells = tuple(int(padded[i:i + CELL_BITS], 2) for i in range(0, len(padded), CELL_BITS))
    if len(cells) > width:
        raise ValueError("value does not fit requested cell width")
    return (0,) * (width - len(cells)) + cells


def decode_bits(cells: Iterable[BitCell]) -> int:
    """Decode a sequence of cells as one big-endian binary integer."""
    values = tuple(c.value for c in cells)
    if not values:
        raise ValueError("at least one cell is required")
    result = 0
    for value in values:
        result = (result << CELL_BITS) | value
    return result
