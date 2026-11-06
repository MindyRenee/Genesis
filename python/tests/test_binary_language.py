"""Tests for the binary mathematical language substrate."""
import pytest

from parietal_lobe.binary_language import (
    BitCell,
    GroundingMemory,
    Role,
    Symbol,
    Frame,
    decode_bits,
    encode_bits,
)


def test_braille_style_dot_positions_round_trip() -> None:
    cell = BitCell.from_dots([1, 4, 6])
    assert cell.value == 41
    assert cell.raised_dots == (1, 4, 6)
    assert cell.bits == "101001"


def test_cell_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        BitCell(64)
    with pytest.raises(ValueError):
        BitCell.from_dots([7])


def test_binary_integer_round_trip() -> None:
    for number in (0, 1, 63, 64, 4095, 123456):
        cells = tuple(BitCell(v) for v in encode_bits(number))
        assert decode_bits(cells) == number


def test_typed_frames_compose_symbols() -> None:
    frame = Frame(Role.ACTION, (Symbol(BitCell(3), Role.ACTION),))
    assert frame.role is Role.ACTION
    assert frame.symbols[0].cell.bits == "000011"


def test_grounding_evidence_updates_and_counterexamples_reduce_support() -> None:
    memory = GroundingMemory()
    pattern = [BitCell.from_dots([1, 3])]
    first = memory.observe(pattern, "movement", supported=True, context="scene-1")
    assert first.support > 0.5
    memory.observe(pattern, "movement", supported=True, context="scene-2")
    revised = memory.observe(pattern, "movement", supported=False, context="scene-3")
    assert revised.support < 1.0
    assert len(revised.contexts) == 3
    assert memory.hypotheses(pattern)[0].referent == "movement"


def test_grounding_keeps_competing_hypotheses_separate() -> None:
    memory = GroundingMemory()
    pattern = [BitCell(7)]
    memory.observe(pattern, "object", supported=True, context="a")
    memory.observe(pattern, "action", supported=True, context="b")
    assert {item.referent for item in memory.hypotheses(pattern)} == {"object", "action"}
