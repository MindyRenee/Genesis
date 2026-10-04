"""Tests for persistent executive ownership of autonomous actions."""

from pathlib import Path
from tempfile import TemporaryDirectory

from genesis_conscious.concepts import ConceptNetwork
from genesis_conscious.frontal_lobe.executive import ExecutiveFunction


def test_executive_intention_survives_failed_episode() -> None:
    executive = ExecutiveFunction()
    intention = executive.form_intention(
        "learn:photosynthesis",
        reason="curiosity",
        priority=0.8,
        confidence=0.7,
        expected_outcome="knowledge or relationships updated",
    )
    selected = executive.select_intention([intention])
    assert selected is intention
    error = executive.observe_intention(
        intention,
        actual_outcome="network unchanged",
        success=False,
    )
    assert error == 1.0
    executive.revise_intention(intention, reason="learning attempt failed")
    assert intention.status == "pending"
    assert intention.attempts == 1
    assert intention.confidence < 0.7


def test_executive_completes_grounded_success() -> None:
    executive = ExecutiveFunction()
    intention = executive.form_intention(
        "inspect:thing", expected_outcome="code structure observed"
    )
    executive.select_intention([intention])
    error = executive.observe_intention(
        intention,
        actual_outcome="thing.py: 10 lines, 1 class",
        success=True,
    )
    assert error == 0.0
    assert intention.status == "completed"
    assert executive.active_intention is None


def test_acting_loop_registers_with_existing_executive() -> None:
    from genesis_conscious.tools.agency import ActingLoop
    from genesis_conscious.tools.framework import ToolRegistry

    with TemporaryDirectory() as tmp:
        data = Path(tmp) / 'data'
        repo = Path(tmp) / 'repo'
        data.mkdir()
        repo.mkdir()
        executive = ExecutiveFunction()
        loop = ActingLoop(
            network=ConceptNetwork(), tools=ToolRegistry(),
            data_dir=str(data), project_root=str(repo), offline=True,
            get_agency_topic=lambda: "photosynthesis",
            executive=executive,
        )
        result = loop.act_once()
        assert result is not None
        assert result.executive_intention is not None
        assert result.executive_intention.objective == "learn:photosynthesis"
        assert executive.intentions

def test_executive_round_trips_persistent_intentions() -> None:
    executive = ExecutiveFunction()
    intention = executive.form_intention(
        "finish:genesis",
        reason="unfinished objective",
        priority=0.9,
        confidence=0.6,
        expected_outcome="verified completion",
    )
    executive.select_intention([intention])
    executive.observe_intention(
        intention,
        actual_outcome="blocked",
        success=False,
    )
    executive.revise_intention(intention, reason="blocked")

    restored = ExecutiveFunction()
    restored.restore_from_dict(executive.to_dict())

    assert restored.active_intention is None
    assert len(restored.intentions) == 1
    saved = restored.intentions[0]
    assert saved.objective == "finish:genesis"
    assert saved.status == "pending"
    assert saved.attempts == 1
    assert saved.last_observation == "blocked"


def test_acting_loop_arbitrates_existing_objective(tmp_path) -> None:
    """A stronger unfinished objective is selected over a fresh probe."""
    from genesis_conscious.tools.agency import ActingLoop, Intention

    executive = ExecutiveFunction()
    prior = executive.form_intention(
        "learn:prior-topic",
        reason="curiosity",
        priority=0.95,
        confidence=0.95,
        expected_outcome="knowledge or relationships updated",
    )
    loop = ActingLoop(
        network=ConceptNetwork(),
        curiosity=None,
        learner=None,
        data_dir=str(tmp_path),
        project_root=str(tmp_path),
        offline=True,
        executive=executive,
    )

    candidate = Intention("observe", "growth_ledger.jsonl", "wander")
    loop._register_intention(candidate)
    selected = loop._activate_executive_intention(candidate)

    assert selected is prior
    assert selected.status == "active"
    decoded = loop._intention_from_executive(selected)
    assert decoded is not None
    assert decoded.kind == "learn"
    assert decoded.target == "prior-topic"


def test_advisory_internal_signal_is_persistent_but_not_actuator_action() -> None:
    """Internal pressure can inform the executive without becoming a fake file action."""
    executive = ExecutiveFunction()
    items = executive.ingest_candidates([{
        "topic": "body_distress",
        "mode": "distress",
        "salience": 0.9,
        "source": "internal_need",
    }])
    assert len(items) == 1
    assert items[0].actionable is False
    assert items[0] not in executive.actionable_intentions
    assert executive.intentions[0].objective == "observe:body_distress"


def test_world_and_improvement_signals_map_to_real_actuators() -> None:
    executive = ExecutiveFunction()
    items = executive.ingest_candidates([
        {
            "topic": "photosynthesis",
            "mode": "environment",
            "kind": "learn",
            "salience": 0.8,
            "source": "world_event",
        },
        {
            "topic": "python/genesis_conscious/executive.py",
            "mode": "improvement",
            "kind": "inspect",
            "salience": 0.7,
            "source": "self_improvement:review",
        },
    ])
    assert {i.objective for i in items} == {
        "learn:photosynthesis",
        "inspect:python/genesis_conscious/executive.py",
    }
    assert all(i.actionable for i in items)
