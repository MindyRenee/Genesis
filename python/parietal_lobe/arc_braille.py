"""Lossless six-dot-cell encoding for complete ARC game records.

This is a Braille-inspired mathematical representation, not standard literary
Braille. It preserves the entire structured game record (observation, legal
actions, decision, transition, terminal outcome) as UTF-8 bytes packed into
six-bit cells. The payload is reversible; the cell values do not themselves
claim to be words or meanings.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from parietal_lobe.binary_language import BitCell

_MAGIC = b"GBR1"
_HEADER_BITS = 32


@dataclass(frozen=True, slots=True)
class BrailleGameRecord:
    """One complete game event encoded as six-dot cells plus exact bit length."""

    cells: tuple[BitCell, ...]
    bit_length: int

    def __post_init__(self) -> None:
        if self.bit_length < 0 or self.bit_length > len(self.cells) * 6:
            raise ValueError("bit_length must fit the supplied cells")


def _plain(value: Any, depth: int = 0) -> Any:
    """Convert common toolkit objects to deterministic JSON-safe structures."""
    if depth > 12:
        return type(value).__name__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value), depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist(), depth + 1)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item, depth + 1)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return type(value).__name__


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    if not isinstance(record, Mapping):
        raise TypeError("game record must be a mapping")
    try:
        payload = json.dumps(
            _plain(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"game record must contain finite JSON-compatible values: {exc}") from exc
    return _MAGIC + len(payload).to_bytes(8, "big") + payload


def encode_game_record(record: Mapping[str, Any]) -> BrailleGameRecord:
    """Encode a complete game record without dropping fields or inventing semantics."""
    raw = _canonical_bytes(record)
    bit_string = "".join(format(byte, "08b") for byte in raw)
    bit_length = len(bit_string)
    bit_string += "0" * ((-bit_length) % 6)
    cells = tuple(
        BitCell(int(bit_string[i:i + 6], 2))
        for i in range(0, len(bit_string), 6)
    )
    return BrailleGameRecord(cells=cells, bit_length=bit_length)


def decode_game_record(encoded: BrailleGameRecord) -> dict[str, Any]:
    """Recover the exact JSON structure represented by the six-dot cells."""
    bits = "".join(cell.bits for cell in encoded.cells)[:encoded.bit_length]
    if len(bits) % 8:
        raise ValueError("record bit length is not byte-aligned")
    raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    if len(raw) < 12 or raw[:4] != _MAGIC:
        raise ValueError("invalid Braille game record header")
    payload_size = int.from_bytes(raw[4:12], "big")
    payload = raw[12:]
    if len(payload) != payload_size:
        raise ValueError("truncated or inconsistent game record")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("game record root must decode to an object")
    return decoded


def observation_record(
    game_id: str,
    observation: Any,
    legal_actions: list[Any],
    *,
    step: int,
) -> dict[str, Any]:
    """Build a lossless, explicitly typed record for one observed game state."""
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("game_id must be a non-empty string")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return {
        "record_type": "arc_observation",
        "game_id": game_id,
        "step": step,
        "observation": observation,
        "legal_actions": legal_actions,
    }


def transition_record(
    game_id: str,
    step: int,
    observation: Any,
    legal_actions: list[Any],
    action: Any,
    next_observation: Any,
    terminal_state: str | None,
) -> dict[str, Any]:
    """Build a complete action/consequence record; terminal labels remain explicit."""
    return {
        "record_type": "arc_transition",
        "game_id": game_id,
        "step": step,
        "observation": observation,
        "legal_actions": legal_actions,
        "action": action,
        "next_observation": next_observation,
        "terminal_state": terminal_state,
    }
