"""Cognitive journal + LTM index scan + archived-dream rendering tests.

Covers the observability layer added for durable inner-life recording:

- ``CognitiveJournal`` — append-only JSONL, never raises, rotates at cap.
- ``record_error`` — module-level error capture for subsystems that
  hold no Mind reference.
- ``scan_ltm_index`` — read-only parse of the daemon's index file,
  including entries ``get_recent_episodes`` cannot see (archived).
- ``_cmd_dreams`` — reports archived dream episodes honestly instead
  of claiming "No dreams stored yet" while the archive holds dozens.
- Dream kind labeling — dream thoughts journaled as kind="dream",
  not indistinguishable "thought".
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import genesis_cli
from genesis_client.ltm_index import scan_ltm_index
from genesis_cognitive.cognitive_journal import (
    JOURNAL_FILENAME,
    CognitiveJournal,
    record_error,
    set_active,
)
from genesis_cognitive.mind import Mind


def _read_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ─── CognitiveJournal ────────────────────────────────────────────────


def test_journal_writes_jsonl_events(tmp_path):
    journal = CognitiveJournal(tmp_path)
    journal.record("thought", "wondering about light", trigger="curiosity")
    journal.record("dream", "an image of the sea", chain_id=7, is_lucid=True)
    journal.close()

    events = _read_events(tmp_path / JOURNAL_FILENAME)
    assert len(events) == 2
    assert events[0]["kind"] == "thought"
    assert events[0]["text"] == "wondering about light"
    assert events[0]["trigger"] == "curiosity"
    assert events[0]["ts"] > 0
    assert events[1]["kind"] == "dream"
    assert events[1]["is_lucid"] is True


def test_journal_appends_across_instances(tmp_path):
    CognitiveJournal(tmp_path).record("thought", "first")
    j2 = CognitiveJournal(tmp_path)
    j2.record("thought", "second")
    j2.close()

    events = _read_events(tmp_path / JOURNAL_FILENAME)
    assert [e["text"] for e in events] == ["first", "second"]


def test_journal_record_never_raises_on_unwritable_dir():
    journal = CognitiveJournal("/nonexistent-dir-for-journal/nope")
    journal.record("thought", "silently dropped")
    journal.close()  # no exception


def test_journal_close_is_idempotent_and_drops_writes(tmp_path):
    journal = CognitiveJournal(tmp_path)
    journal.record("thought", "before close")
    journal.close()
    journal.close()
    journal.record("thought", "after close")

    events = _read_events(tmp_path / JOURNAL_FILENAME)
    assert len(events) == 1


def test_journal_rotates_at_size_cap(tmp_path):
    journal = CognitiveJournal(tmp_path)
    journal._MAX_BYTES = 200
    for i in range(20):
        journal.record("thought", f"event {i} " + "x" * 50)
    journal.close()

    assert (tmp_path / "cognitive_journal.1.jsonl").exists()
    events = _read_events(tmp_path / JOURNAL_FILENAME)
    assert events and events[-1]["text"].startswith("event 19")


def test_journal_record_error(tmp_path):
    journal = CognitiveJournal(tmp_path)
    journal.record_error("volition.bug_scan", ValueError("boom"))
    journal.close()

    (event,) = _read_events(tmp_path / JOURNAL_FILENAME)
    assert event["kind"] == "error"
    assert event["site"] == "volition.bug_scan"
    assert "boom" in event["text"]


def test_module_record_error_noop_without_active(tmp_path):
    set_active(None)
    record_error("test.site", RuntimeError("x"))  # no raise, no write

    journal = CognitiveJournal(tmp_path)
    set_active(journal)
    try:
        record_error("test.site", RuntimeError("captured"))
    finally:
        set_active(None)
        journal.close()

    (event,) = _read_events(tmp_path / JOURNAL_FILENAME)
    assert event["kind"] == "error"
    assert event["site"] == "test.site"


# ─── scan_ltm_index ──────────────────────────────────────────────────

_ENTRY = struct.Struct("<QQQQII f 4f BB2x".replace(" ", ""))
_HEADER = struct.Struct("<4sIIIQ40x".replace(" ", ""))


def _write_meta(path: Path, entries: list[tuple[int, int, float, int, int]]):
    """entries: (episode_id, timestamp, salience, flags, source_module)."""
    body = b"".join(
        _ENTRY.pack(eid, 16, ts, 0, 4, 64, sal, 0, 0, 0, 0, flags, mod)
        for eid, ts, sal, flags, mod in entries
    )
    header = _HEADER.pack(b"LTMM", 2, 0xFFFFFFFF, len(entries), len(entries) + 1)
    path.write_bytes(header + body)


def test_scan_ltm_index_roundtrip(tmp_path):
    meta = tmp_path / "ltm_store.meta"
    _write_meta(
        meta,
        [
            (1, 1000, 0.5, 0, 3),      # plain active episode
            (2, 2000, 0.7, 6, 9),      # meta+archived dream insight
            (3, 3000, 0.9, 2, 9),      # meta-only active dream insight
        ],
    )

    entries = scan_ltm_index(meta)
    assert len(entries) == 3
    assert entries[1].episode_id == 2
    assert entries[1].is_archived and entries[1].is_meta_memory
    assert not entries[1].is_deleted
    assert not entries[0].is_archived

    mod9 = scan_ltm_index(meta, source_module=9)
    assert [e.episode_id for e in mod9] == [2, 3]


def test_scan_ltm_index_handles_missing_and_foreign_files(tmp_path):
    assert scan_ltm_index(tmp_path / "absent.meta") == []
    bad = tmp_path / "bad.meta"
    bad.write_bytes(b"NOPE" + b"\x00" * 200)
    assert scan_ltm_index(bad) == []
    short = tmp_path / "short.meta"
    short.write_bytes(b"LTMM" + b"\x00" * 10)
    assert scan_ltm_index(short) == []


def test_scan_ltm_index_ignores_partial_tail(tmp_path):
    meta = tmp_path / "ltm_store.meta"
    _write_meta(meta, [(1, 1000, 0.5, 6, 9)])
    with meta.open("ab") as f:
        f.write(b"\xde\xad" * 10)  # torn in-flight append
    entries = scan_ltm_index(meta, source_module=9)
    assert [e.episode_id for e in entries] == [1]


# ─── /dreams rendering ───────────────────────────────────────────────


def _ep(ep_id, ts, text, module=9, sal=0.6):
    return SimpleNamespace(
        episode_id=ep_id, timestamp=ts, salience=sal,
        event_type=0, source_module=module, text=text,
    )


def test_cmd_dreams_shows_archived_when_active_empty(tmp_path):
    meta = tmp_path / "ltm_store.meta"
    _write_meta(meta, [(42, 1000, 0.6, 6, 9), (43, 2000, 0.7, 6, 9)])

    mind = Mock()
    mind.data_dir = str(tmp_path)
    mind.client.get_recent_episodes.return_value = []
    mind.client.retrieve_episode.side_effect = lambda eid, **kw: _ep(
        eid, {42: 1000, 43: 2000}[eid], f"[dream-insight] raw {eid}"
    )

    out = genesis_cli._cmd_dreams(mind, "")
    assert "No dreams stored yet" not in out
    assert "0 active, 2 archived" in out
    assert "·archived" in out
    assert "raw 43" in out  # newest first


def test_cmd_dreams_merges_active_and_archived(tmp_path):
    meta = tmp_path / "ltm_store.meta"
    _write_meta(meta, [(10, 1000, 0.6, 6, 9), (11, 2000, 0.7, 2, 9)])

    mind = Mock()
    mind.data_dir = str(tmp_path)
    mind.client.get_recent_episodes.return_value = [
        _ep(11, 2000, "[dream-insight] live one")
    ]
    mind.client.retrieve_episode.side_effect = lambda eid, **kw: _ep(
        eid, 1000, "[dream-insight] old one"
    )

    out = genesis_cli._cmd_dreams(mind, "")
    assert "1 active, 1 archived" in out
    assert "live one" in out and "old one" in out


def test_cmd_dreams_empty_store_still_says_none(tmp_path):
    (tmp_path / "ltm_store.meta").write_bytes(b"")
    mind = Mock()
    mind.data_dir = str(tmp_path)
    mind.client.get_recent_episodes.return_value = []
    assert "No dreams stored yet" in genesis_cli._cmd_dreams(mind, "")


def test_cmd_dreams_daemon_unreachable(tmp_path):
    mind = Mock()
    mind.data_dir = str(tmp_path)
    mind.client.get_recent_episodes.side_effect = OSError("gone")
    assert "cannot reach subcognitive" in genesis_cli._cmd_dreams(mind, "")


# ─── dream kind labeling ─────────────────────────────────────────────


def test_dream_thoughts_journal_as_dream_kind(tmp_path):
    mind = Mind(str(tmp_path / "genesis.sock"))
    try:
        dream = SimpleNamespace(
            content="a corridor of blue rooms",
            trigger="rem",
            intent=None,
            chain_position=0,
            chain_id=3,
            is_dream=True,
            is_lucid=False,
            metadata=None,
        )
        mind._on_spontaneous_thought(dream)

        waking = SimpleNamespace(
            content="the word light again",
            trigger="curiosity",
            intent=None,
            chain_position=0,
            chain_id=4,
            is_dream=False,
            is_lucid=False,
            metadata=None,
        )
        mind._on_spontaneous_thought(waking)
    finally:
        mind.journal.close()
        set_active(None)

    events = _read_events(tmp_path / JOURNAL_FILENAME)
    kinds = {e["text"]: e["kind"] for e in events}
    assert kinds["a corridor of blue rooms"] == "dream"
    assert kinds["the word light again"] == "thought"
    dream_event = next(e for e in events if e["kind"] == "dream")
    assert dream_event["trigger"] == "rem"
    assert dream_event["chain_id"] == 3


# ─── regression: is_connected is a property, not a method ────────────


def test_safeguard_action_does_not_crash_on_property(tmp_path):
    """_perform_safeguard must not raise TypeError — is_connected is a
    property. This bug silently killed every safeguard firing until the
    journal recorded it (kind="error", site="volition.safeguard")."""
    mind = Mind(str(tmp_path / "genesis.sock"))
    try:
        # Drive the action directly; the daemon is absent so the probe
        # path exercises the is_connected check then degrades gracefully.
        mind._perform_safeguard()
    finally:
        mind.journal.close()
        set_active(None)

    # The safeguard error must NOT be in the journal — the check itself
    # used to crash before reaching the probe.
    path = tmp_path / JOURNAL_FILENAME
    events = _read_events(path) if path.exists() else []
    assert not any(
        e["kind"] == "error" and "not callable" in e["text"] for e in events
    ), f"safeguard crashed on is_connected(): {events}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
