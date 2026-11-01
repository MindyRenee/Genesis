"""Tests for lossless ASCII JSON ARC game encoding."""
import pytest
from parietal_lobe.arc_ascii import (
    AsciiGameRecord,
    decode_game_record,
    encode_game_record,
    observation_record,
    transition_record,
)


def test_full_observation_round_trips_without_losing_fields() -> None:
    record = observation_record(
        "game-17",
        {"grid": [[0, 1], [2, 3]], "state": "PLAYING", "score": 4},
        ["UP", "DOWN", "ACTION1"],
        step=3,
    )
    encoded = encode_game_record(record)
    assert all(0 <= code < 128 for code in encoded.codes)
    assert decode_game_record(encoded) == record


def test_non_ascii_values_round_trip_via_ascii_json_escapes() -> None:
    record = {"label": "café → 你好"}
    encoded = encode_game_record(record)
    assert all(code < 128 for code in encoded.codes)
    assert decode_game_record(encoded) == record


def test_full_transition_preserves_action_and_terminal_outcome() -> None:
    record = transition_record(
        "game-17", 4, {"grid": [[0]], "state": "PLAYING"}, ["ACTION1"],
        {"name": "ACTION1", "data": {"x": 1}},
        {"grid": [[1]], "state": "WIN"}, "WIN",
    )
    assert decode_game_record(encode_game_record(record)) == record


def test_encoding_is_deterministic_for_mapping_order() -> None:
    first = encode_game_record({"record_type": "test", "value": {"a": 1, "b": 2}})
    second = encode_game_record({"value": {"b": 2, "a": 1}, "record_type": "test"})
    assert first == second


def test_non_json_game_payload_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError):
        encode_game_record({"bad": object()})


def test_record_rejects_non_ascii_codes() -> None:
    with pytest.raises(ValueError):
        AsciiGameRecord((128,))
