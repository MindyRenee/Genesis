"""World-learning gate tests — how much experience a run carried in.

``EmbodimentMixin._load_world_learning`` is where a result's honesty
is decided: did persisted lessons ride into the run (``loaded``), did
the file simply not exist (``none``), was it refused on purpose
(``suppressed`` — a fresh cold-start probe), or did it fail to read
(``unknown``, the one state that must not masquerade as cold start).

And the other half of the contract: a fresh run must not *overwrite*
banked experience with what the probe learned — the measurement is
isolated in both directions.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from parietal_lobe.embodiment import EmbodimentMixin


def _embodiment(tmp_path, monkeypatch, *, fresh=False, context="game-x"):
    """A bare mixin — just the attributes the gate touches."""
    monkeypatch.setattr("parietal_lobe.embodiment.default_data_dir", lambda: tmp_path)
    m = EmbodimentMixin.__new__(EmbodimentMixin)
    m._world_context = context
    m._fresh_world_run = fresh
    m.imported = []
    m.world_model = SimpleNamespace(
        export_learning=lambda: {"learned": True},
        import_learning=lambda data: m.imported.append(data),
    )
    return m


def _learning_file(tmp_path, game="game-x"):
    return tmp_path / "world_learning" / f"{game}.json"


def test_load_reports_none_when_no_file(tmp_path, monkeypatch):
    m = _embodiment(tmp_path, monkeypatch)
    m._load_world_learning()
    assert m._world_learning_loaded == "none"
    assert m.imported == []


def test_load_reports_loaded_when_file_exists(tmp_path, monkeypatch):
    path = _learning_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"effects": {"1": [0, 1]}}))
    m = _embodiment(tmp_path, monkeypatch)
    m._load_world_learning()
    assert m._world_learning_loaded == "loaded"
    assert m.imported == [{"effects": {"1": [0, 1]}}]


def test_fresh_run_suppresses_load_and_save(tmp_path, monkeypatch):
    """Cold-start evidence: the banked file is unread AND unwritten."""
    path = _learning_file(tmp_path)
    path.parent.mkdir(parents=True)
    original = {"effects": {"old": True}}
    path.write_text(json.dumps(original))
    m = _embodiment(tmp_path, monkeypatch, fresh=True)
    m._load_world_learning()
    assert m._world_learning_loaded == "suppressed"
    assert m.imported == []
    # The probe's new learning must not overwrite the banked file.
    m._save_world_learning()
    assert json.loads(path.read_text()) == original


def test_unreadable_file_is_unknown_not_none(tmp_path, monkeypatch):
    """A corrupt ledger is 'unknown' — never pretend cold start."""
    path = _learning_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{corrupt")
    m = _embodiment(tmp_path, monkeypatch)
    m._load_world_learning()
    assert m._world_learning_loaded == "unknown"


def test_no_context_reports_none(tmp_path, monkeypatch):
    m = _embodiment(tmp_path, monkeypatch, context=None)
    m._load_world_learning()
    assert m._world_learning_loaded == "none"


def test_normal_run_still_saves(tmp_path, monkeypatch):
    """The gate only binds fresh runs — ordinary play still banks."""
    m = _embodiment(tmp_path, monkeypatch)
    m._save_world_learning()
    assert json.loads(_learning_file(tmp_path).read_text()) == {"learned": True}


# ─── Domain-general competence checkpoint ─────────────────────
#
# The per-game file banks one world's physics; the shared
# ``_competence.json`` banks the cross-game abstraction layer at the
# same boundary. On restore the checkpoint wins only when it carries
# strictly more evidence (transition_count is cumulative), and only
# once per session — a mid-run game switch must never clobber live
# schemas with a stale file.


def _competence_file(tmp_path):
    return tmp_path / "world_learning" / "_competence.json"


def _embodiment_with_competence(tmp_path, monkeypatch, *, fresh=False):
    """Bare mixin plus a real, empty TaskCompetence."""
    from frontal_lobe.reasoning.competence import TaskCompetence

    m = _embodiment(tmp_path, monkeypatch, fresh=fresh)
    m.brain = SimpleNamespace(frontal=SimpleNamespace(task_competence=TaskCompetence()))
    return m


def test_episode_checkpoint_writes_competence(tmp_path, monkeypatch):
    m = _embodiment_with_competence(tmp_path, monkeypatch)
    comp = m.brain.frontal.task_competence
    comp.transition_count = 7
    m._save_world_learning()
    data = json.loads(_competence_file(tmp_path).read_text())
    assert data["transition_count"] == 7


def test_checkpoint_restores_when_newer(tmp_path, monkeypatch):
    """A crash-guard file newer than memory is recovered."""
    from frontal_lobe.reasoning.competence import TaskCompetence

    banked = TaskCompetence()
    banked.recognize(domain="world.embodied", state={"x": 1}, roles=())
    banked.transition_count = 50
    path = _competence_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(banked.to_dict()))

    m = _embodiment_with_competence(tmp_path, monkeypatch)
    m._load_world_learning()
    comp = m.brain.frontal.task_competence
    assert m._world_competence_loaded == "loaded"
    assert comp.transition_count == 50
    assert len(comp.schemas) == 1


def test_older_checkpoint_does_not_clobber(tmp_path, monkeypatch):
    """Live learning newer than the file survives — never regress."""
    path = _competence_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schemas": [],
                "skills": [],
                "transition_count": 10,
                "match_threshold": 0.5,
            }
        )
    )
    m = _embodiment_with_competence(tmp_path, monkeypatch)
    m.brain.frontal.task_competence.transition_count = 100
    m._load_world_learning()
    assert m._world_competence_loaded == "none"
    assert m.brain.frontal.task_competence.transition_count == 100


def test_competence_restore_runs_once_per_session(tmp_path, monkeypatch):
    """A second game switch can't re-import over live learning."""
    path = _competence_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schemas": [],
                "skills": [],
                "transition_count": 50,
                "match_threshold": 0.5,
            }
        )
    )
    m = _embodiment_with_competence(tmp_path, monkeypatch)
    m._load_world_learning()
    assert m._world_competence_loaded == "loaded"
    # Live progress since the restore must survive the next switch.
    m.brain.frontal.task_competence.transition_count = 999
    m._load_world_learning()
    assert m.brain.frontal.task_competence.transition_count == 999


def test_fresh_run_suppresses_competence_both_ways(tmp_path, monkeypatch):
    """Cold-start isolation covers the shared ledger too."""
    path = _competence_file(tmp_path)
    path.parent.mkdir(parents=True)
    original = {"schemas": [], "skills": [], "transition_count": 5}
    path.write_text(json.dumps(original))
    m = _embodiment_with_competence(tmp_path, monkeypatch, fresh=True)
    m._load_world_learning()
    assert m._world_competence_loaded == "suppressed"
    m._save_world_learning()
    assert json.loads(path.read_text()) == original
