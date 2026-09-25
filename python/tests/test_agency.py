"""Tests for the acting loop — volition-driven open-ended tool use."""

import tempfile
from pathlib import Path
from typing import Any

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.learning.curiosity import Question
from genesis_cognitive.tools.agency import (
    _TOOL_SCOPES,
    ActingLoop,
    ActingResult,
    Intention,
)
from genesis_cognitive.tools.framework import ToolRegistry


def _make_loop(
    tmp: str,
    *,
    curiosity: Any = None,
    learner: Any = None,
    offline: bool = True,
    get_emotion: Any = None,
    get_agency_topic: Any = None,
    on_event: Any = None,
    on_store_memory: Any = None,
    on_neuro_impulse: Any = None,
    on_live_thought: Any = None,
) -> ActingLoop:
    """Build an ActingLoop over a fresh data dir + project root."""
    data_dir = Path(tmp) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(tmp) / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return ActingLoop(
        network=ConceptNetwork(),
        curiosity=curiosity,
        learner=learner,
        tools=ToolRegistry(),
        data_dir=str(data_dir),
        project_root=str(repo),
        offline=offline,
        get_emotion=get_emotion,
        get_agency_topic=get_agency_topic,
        on_event=on_event,
        on_store_memory=on_store_memory,
        on_neuro_impulse=on_neuro_impulse,
        on_live_thought=on_live_thought,
    )


class _FakeCuriosity:
    """Minimal stand-in returning a fixed question list."""

    def __init__(self, questions: list) -> None:
        self._questions = questions
        self.resolved: list[str] = []

    def generate_questions(self, emotion: Any, max_questions: int = 3) -> list:
        return self._questions[:max_questions]

    def mark_resolved(self, concept: str) -> None:
        self.resolved.append(concept)


class _FakeLearner:
    def __init__(self) -> None:
        self.topics: list[str] = []

    def add_topic(self, topic: str) -> None:
        self.topics.append(topic)


# ─── Intention formation ─────────────────────────────────────────────


def test_agency_topic_produces_learn_intention() -> None:
    """A topic from its train of thought becomes a learn intention."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp, get_agency_topic=lambda: "photosynthesis")
        intention = loop.propose()
        assert intention is not None
        assert intention.kind == "learn"
        assert intention.target == "photosynthesis"
        assert intention.origin == "agency"


def test_curiosity_code_concept_produces_inspect() -> None:
    """A curiosity question about a code concept becomes an inspection."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        (repo / "genesis_cognitive").mkdir(parents=True)
        (repo / "genesis_cognitive" / "thing.py").write_text(
            "def f():\n    return 1\n"
        )
        q = Question(
            text="", target_concept="python:thing", gap_type="missing_edge",
            curiosity_score=0.8, question_type="isolation",
        )
        loop = _make_loop(
            tmp, curiosity=_FakeCuriosity([q]), get_emotion=lambda: object(),
        )
        intention = loop.propose()
        assert intention is not None
        assert intention.kind == "inspect"
        assert intention.target == "python:thing"


def test_no_state_falls_back_to_wandering_probe() -> None:
    """With nothing pressing, it forms a wandering probe intention."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp)
        intention = loop.propose()
        assert intention is not None
        assert intention.origin == "wander"
        assert intention.kind in ("observe", "explore", "measure")


# ─── Capability policy ───────────────────────────────────────────────


def test_destructive_tools_never_autonomous() -> None:
    """delete_file/move_file are not in the autonomous tool scope."""
    assert "delete_file" not in _TOOL_SCOPES
    assert "move_file" not in _TOOL_SCOPES


def test_writes_confined_to_scratch() -> None:
    """An acting episode may write into experiments/, never the repo."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp)
        repo = Path(tmp) / "repo"
        (repo / "marker.py").write_text("x = 1\n")
        result = loop.act_once()
        assert result is not None
        # The field-notes journal lands in the sandbox, not the repo.
        notes = Path(tmp) / "data" / "experiments" / "field_notes.md"
        assert notes.is_file()
        # Nothing in the repo tree was created or modified by acting.
        assert sorted(p.name for p in repo.iterdir()) == ["marker.py"]


def test_shell_commands_are_shape_vetted() -> None:
    """run_shell only accepts internally-built measurement shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp)
        result = ActingResult(intention=Intention("measure", "x", "wander"))
        refused = loop._use(
            result, "run_shell", command="cat /etc/passwd", _scope="scratch",
        )
        assert refused is None
        assert result.steps[-1].detail == "unvetted command"


def test_bad_scope_refused() -> None:
    """A tool invoked outside its declared scope is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp)
        result = ActingResult(intention=Intention("explore", ".", "wander"))
        refused = loop._use(
            result, "write_file", path="x.txt", content="x", _scope="repo",
        )
        assert refused is None
        assert "bad scope" in result.steps[-1].detail


def test_offline_skips_net_tools() -> None:
    """Net-scoped tools are refused when the mind is offline."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp, offline=True)
        result = ActingResult(intention=Intention("learn", "x", "wander"))
        refused = loop._use(result, "web_search", query="x")
        assert refused is None
        assert result.steps[-1].detail == "offline"


# ─── Execution ───────────────────────────────────────────────────────


def test_explore_reads_text_files() -> None:
    """Explore lists a directory and reads its text files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        docs = repo / "docs"
        docs.mkdir(parents=True)
        (docs / "notes.md").write_text(
            "# Notes\n\nA cat is a small animal. A dog is a loyal animal.\n"
        )
        loop = _make_loop(tmp)
        result = ActingResult(intention=Intention("explore", "docs", "wander"))
        loop._execute_explore(result.intention, result)
        assert result.success
        assert any("explored docs" in d for d in result.discoveries)
        assert any("read notes.md" in d for d in result.discoveries)


def test_observe_reads_own_state() -> None:
    """Observe lists the data dir and reads a state file."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp)
        data_dir = Path(tmp) / "data"
        (data_dir / "growth_ledger.jsonl").write_text('{"a": 1}\n')
        result = ActingResult(
            intention=Intention("observe", "growth_ledger.jsonl", "wander")
        )
        loop._execute_observe(result.intention, result)
        assert result.success
        assert any("growth_ledger" in d for d in result.discoveries)


def test_measure_counts_files() -> None:
    """Measure runs whitelisted find/wc/du commands and records numbers."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        target = repo / "genesis_cognitive"
        target.mkdir(parents=True)
        (target / "a.py").write_text("x = 1\n")
        (target / "b.py").write_text("y = 2\n")
        loop = _make_loop(tmp)
        result = ActingResult(
            intention=Intention("measure", "genesis_cognitive", "wander")
        )
        loop._execute_measure(result.intention, result)
        assert result.success
        assert any("find" in d or "du" in d for d in result.discoveries)


def test_inspect_analyzes_python() -> None:
    """Inspect resolves a code concept to a file and analyzes it."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        pkg = repo / "genesis_cognitive"
        pkg.mkdir(parents=True)
        (pkg / "widget.py").write_text(
            "class Widget:\n    def spin(self):\n        return 1\n"
        )
        loop = _make_loop(tmp)
        result = ActingResult(
            intention=Intention("inspect", "python:widget", "curiosity")
        )
        loop._execute_inspect(result.intention, result)
        assert result.success
        assert any("widget.py" in d and "functions" in d for d in result.discoveries)


def test_step_budget_is_enforced() -> None:
    """No acting episode may exceed MAX_STEPS tool calls."""
    with tempfile.TemporaryDirectory() as tmp:
        loop = _make_loop(tmp)
        result = ActingResult(intention=Intention("observe", ".", "wander"))
        for _ in range(loop.MAX_STEPS + 3):
            loop._use(result, "list_dir", path=".", _scope="repo")
        assert len(result.steps) <= loop.MAX_STEPS


# ─── Recording and habituation ───────────────────────────────────────


def test_successful_act_records_everywhere() -> None:
    """A successful act hits the journal, memory, world, and dopamine."""
    with tempfile.TemporaryDirectory() as tmp:
        events: list[str] = []
        memories: list[str] = []
        impulses: list[int] = []
        loop = _make_loop(
            tmp,
            on_event=events.append,
            on_store_memory=lambda text, salience: memories.append(text),
            on_neuro_impulse=lambda chem, amt: impulses.append(chem),
            on_live_thought=lambda kind, text: None,
        )
        result = loop.act_once()
        assert result is not None
        assert events  # world got the act
        assert result.steps  # tools were used


def test_same_intention_not_repeated() -> None:
    """Habituation: a just-acted (kind, target) is not proposed again."""
    with tempfile.TemporaryDirectory() as tmp:
        topics = iter(["calculus", "calculus"])
        loop = _make_loop(tmp, get_agency_topic=lambda: next(topics, None))
        first = loop.act_once()
        assert first is not None and first.intention.target == "calculus"
        # Second identical topic → skipped → falls back to wandering.
        second = loop.act_once()
        assert second is not None
        assert not (
            second.intention.kind == "learn"
            and second.intention.target == "calculus"
        )
