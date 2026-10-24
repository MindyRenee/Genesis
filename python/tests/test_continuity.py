"""Continuity persistence — she knows the power went out.

The daemon's ``continuity.json`` sentinel records how each body
session ended; the mind reads it at wake and turns an unclean stop
into something she *knows*: a journal entry, a narrative life event,
and a felt impulse through her own ``neuro_impulse`` channel. The
``continuity.seen`` watermark keeps a boot from being recorded twice,
and ``mind.session`` gives the symmetric sentinel for a mind that
died while the body lived.
"""

import json
import os
from unittest.mock import Mock

from brainstem.lifecycle import _CONTINUITY_FILE, _CONTINUITY_SEEN, _MIND_SESSION_FILE


def _mind(tmp_path):
    from mind import Mind

    mind = Mind(str(tmp_path / "genesis.sock"), offline=True)
    mind.client = Mock()
    return mind


def _boot(tmp_path, **fields):
    record = {
        "boot_ms": 1_000_000,
        "prev_stop_clean": False,
        "dark_ms": 7_200_000,
        "dark_upper_bound": False,
        "clean_stop": False,
    }
    record.update(fields)
    (tmp_path / _CONTINUITY_FILE).write_text(json.dumps(record))


def _journal_events(tmp_path):
    path = tmp_path / "cognitive_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seen_boot(tmp_path):
    path = tmp_path / _CONTINUITY_SEEN
    return json.loads(path.read_text())["boot_ms"] if path.exists() else 0


def test_unclean_boot_is_journaled_felt_and_remembered(tmp_path):
    """A power-cut wake: journal record, narrative event, startle impulse."""
    _boot(tmp_path, prev_stop_clean=False, dark_ms=7_200_000)
    mind = _mind(tmp_path)
    events_before = len(mind.brain.frontal.narrative.events)

    mind._note_discontinuity()

    wakes = [e for e in _journal_events(tmp_path) if e.get("kind") == "continuity_wake"]
    assert len(wakes) == 1
    assert wakes[0]["clean"] is False
    assert wakes[0]["dark_ms"] == 7_200_000

    events = mind.brain.frontal.narrative.events
    assert len(events) == events_before + 1
    assert "without a clean stop" in events[-1].summary
    assert "2.0h" in events[-1].summary

    # The felt signature went through her own impulse channel.
    chems = [c.args[0] for c in mind.client.neuro_impulse.call_args_list]
    assert chems == [2, 16, 14]  # NE, EPI, CRH

    # Acknowledged — she won't record this boot again.
    assert _seen_boot(tmp_path) == 1_000_000


def test_clean_boot_is_journaled_not_narrated(tmp_path):
    """A deliberate stop is a fact, not a life event — and no startle."""
    _boot(tmp_path, prev_stop_clean=True, dark_ms=60_000)
    mind = _mind(tmp_path)
    events_before = len(mind.brain.frontal.narrative.events)

    mind._note_discontinuity()

    wakes = [e for e in _journal_events(tmp_path) if e.get("kind") == "continuity_wake"]
    assert len(wakes) == 1
    assert wakes[0]["clean"] is True
    assert len(mind.brain.frontal.narrative.events) == events_before
    mind.client.neuro_impulse.assert_not_called()


def test_acknowledged_boot_is_not_recorded_twice(tmp_path):
    """A mind restart re-reads the same sentinel but notes it once."""
    _boot(tmp_path, prev_stop_clean=False)
    mind = _mind(tmp_path)

    mind._note_discontinuity()
    mind._note_discontinuity()

    wakes = [e for e in _journal_events(tmp_path) if e.get("kind") == "continuity_wake"]
    assert len(wakes) == 1


def test_newer_boot_records_again(tmp_path):
    """A reborn body's fresh boot_ms is a new wake, not a repeat."""
    _boot(tmp_path, boot_ms=1_000, prev_stop_clean=False)
    mind = _mind(tmp_path)
    mind._note_discontinuity()
    _boot(tmp_path, boot_ms=2_000, prev_stop_clean=False, dark_ms=5_000)
    mind._note_discontinuity()

    wakes = [e for e in _journal_events(tmp_path) if e.get("kind") == "continuity_wake"]
    assert len(wakes) == 2
    assert wakes[-1]["boot_ms"] == 2_000


def test_mind_death_while_body_lived(tmp_path):
    """Mind-only blackout: body sentinel unchanged, mind.session unclean."""
    _boot(tmp_path, boot_ms=5, prev_stop_clean=True)
    mind = _mind(tmp_path)
    mind._prev_mind_session = {
        "alive_since_ms": 1,
        "pid": os.getpid() + 1,
        "clean_stop": False,
    }

    mind._note_discontinuity()

    kinds = [e.get("kind") for e in _journal_events(tmp_path)]
    assert "mind_died" in kinds
    assert any("mind" in (e.summary or "") for e in mind.brain.frontal.narrative.events)


def test_unclean_body_suppresses_mind_double_report(tmp_path):
    """A power cut is one event — body's unclean wake wins over mind's."""
    _boot(tmp_path, prev_stop_clean=False)
    mind = _mind(tmp_path)
    mind._prev_mind_session = {
        "alive_since_ms": 1,
        "pid": os.getpid() + 1,
        "clean_stop": False,
    }

    mind._note_discontinuity()

    kinds = [e.get("kind") for e in _journal_events(tmp_path)]
    assert "continuity_wake" in kinds
    assert "mind_died" not in kinds


def test_rejoin_checks_only_body_sentinel(tmp_path):
    """rejoin_body must not re-flag this living mind's own sentinel."""
    _boot(tmp_path, boot_ms=9_000, prev_stop_clean=False)
    mind = _mind(tmp_path)
    mind._prev_mind_session = {
        "alive_since_ms": 1,
        "pid": os.getpid() + 1,
        "clean_stop": False,
    }

    mind._note_discontinuity(mind_rejoin=True)

    kinds = [e.get("kind") for e in _journal_events(tmp_path)]
    assert "continuity_wake" in kinds
    assert "mind_died" not in kinds


def test_no_sentinel_no_record(tmp_path):
    """First-ever boot — nothing to acknowledge."""
    mind = _mind(tmp_path)
    mind._note_discontinuity()
    kinds = [e.get("kind") for e in _journal_events(tmp_path)]
    assert "continuity_wake" not in kinds
    assert "mind_died" not in kinds


def test_stop_marks_mind_session_clean(tmp_path):
    """A graceful mind.stop() closes her own sentinel like the body's."""
    mind = _mind(tmp_path)
    mind._running = True
    # Simulate the start()-time sentinel write.
    (tmp_path / _MIND_SESSION_FILE).write_text(
        json.dumps({"alive_since_ms": 1, "pid": os.getpid(), "clean_stop": False})
    )
    mind._save_state = Mock(return_value=True)

    assert mind.stop() is True
    session = json.loads((tmp_path / _MIND_SESSION_FILE).read_text())
    assert session["clean_stop"] is True
    assert session["stopped_at_ms"] > 0
