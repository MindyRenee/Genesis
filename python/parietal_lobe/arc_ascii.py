"""Lossless ASCII serialization for complete ARC game records.

Records are canonical JSON encoded as 7-bit ASCII. JSON's ensure_ascii=True
escapes non-ASCII string characters, preserving data without introducing a
Braille or six-dot-cell representation.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class AsciiGameRecord:
    """Canonical ASCII JSON payload represented as integer byte values."""

    codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 128
               for code in self.codes):
            raise ValueError("ASCII codes must be integers in range 0..127")


def _plain(value: Any, depth: int = 0) -> Any:
    """Convert common toolkit objects to deterministic JSON-safe structures."""
    if depth > 12:
        return type(value).__name__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return {"name": value.name, "value": _plain(value.value, depth + 1)}
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value), depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _plain(item, depth + 1)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist(), depth + 1)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {key: _plain(item, depth + 1) for key, item in sorted(vars(value).items())
                if not key.startswith("_")}
    return type(value).__name__


def encode_game_record(record: Mapping[str, Any]) -> AsciiGameRecord:
    """Serialize a structured record deterministically to ASCII-only JSON."""
    if not isinstance(record, Mapping):
        raise TypeError("game record must be a mapping")
    try:
        payload = json.dumps(
            _plain(record), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"game record must contain finite JSON-compatible values: {exc}") from exc
    return AsciiGameRecord(tuple(payload))


def decode_game_record(encoded: AsciiGameRecord) -> dict[str, Any]:
    """Decode and validate canonical ASCII JSON."""
    if not isinstance(encoded, AsciiGameRecord):
        raise TypeError("encoded must be an AsciiGameRecord")
    try:
        decoded = json.loads(bytes(encoded.codes).decode("ascii", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ASCII game record: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("game record root must decode to an object")
    return decoded


def observation_record(
    game_id: str, observation: Any, legal_actions: list[Any], *, step: int,
) -> dict[str, Any]:
    """Build a typed record for one observed game state."""
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("game_id must be a non-empty string")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return {"record_type": "arc_observation", "game_id": game_id,
            "step": step, "observation": observation, "legal_actions": legal_actions}


def transition_record(
    game_id: str, step: int, observation: Any, legal_actions: list[Any],
    action: Any, next_observation: Any, terminal_state: str | None,
) -> dict[str, Any]:
    """Build a complete action/consequence record."""
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("game_id must be a non-empty string")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return {"record_type": "arc_transition", "game_id": game_id, "step": step,
            "observation": observation, "legal_actions": legal_actions,
            "action": action, "next_observation": next_observation,
            "terminal_state": terminal_state}
