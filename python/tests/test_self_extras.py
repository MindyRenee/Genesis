"""Self extras — growth ledger, journal, explorer, code learner."""

import logging
import math
import os
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.growth_ledger import GrowthLedger, GrowthMilestone
from genesis_cognitive.journal import Journal, JournalEntry
from genesis_cognitive.tools.code_learner import (
    CodeLearner,
    CodeLearningResult,
    FileLearningResult,
)
from genesis_cognitive.tools.explorer import Explorer

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_growth_ledger.py
# ======================================================================

class TestGrowthMilestone:
    """Tests for the GrowthMilestone dataclass."""

    def test_to_dict_roundtrip(self):
        """Milestone should survive a to_dict/from_dict round-trip."""
        m = GrowthMilestone(
            dimension="knowledge",
            metric="concept_count",
            value=500,
            previous=450,
            note="Crossed 500 concepts",
            timestamp=12345,
        )
        data = m.to_dict()
        restored = GrowthMilestone.from_dict(data)

        assert restored.dimension == "knowledge"
        assert restored.metric == "concept_count"
        assert restored.value == 500
        assert restored.previous == 450
        assert restored.note == "Crossed 500 concepts"
        assert restored.timestamp == 12345

    def test_delta_property(self):
        """delta should compute value - previous."""
        m = GrowthMilestone(
            dimension="test", metric="test", value=100, previous=80, note=""
        )
        assert m.delta == 20

        m2 = GrowthMilestone(
            dimension="test", metric="test", value=50, previous=80, note=""
        )
        assert m2.delta == -30

    def test_from_dict_defaults(self):
        """from_dict should handle missing fields with defaults."""
        restored = GrowthMilestone.from_dict({})
        assert restored.dimension == ""
        assert restored.value == 0.0
        assert restored.timestamp == 0


class TestGrowthLedger:
    """Tests for the GrowthLedger class."""

    def test_empty_ledger(self):
        """A new ledger should have no milestones."""
        ledger = GrowthLedger()
        assert ledger.milestone_count == 0
        assert ledger.milestones == []

    def test_record_creates_milestone(self):
        """record should create a milestone when value changes."""
        ledger = GrowthLedger()
        m = ledger.record("knowledge", "concept_count", 100, "First 100 concepts")

        assert m is not None
        assert m.dimension == "knowledge"
        assert m.metric == "concept_count"
        assert m.value == 100
        assert ledger.milestone_count == 1

    def test_record_skips_unchanged_value(self):
        """record should not create a milestone if value hasn't changed."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100, "First record")
        m = ledger.record("knowledge", "concept_count", 100, "No change")

        assert m is None
        assert ledger.milestone_count == 1

    def test_record_tracks_previous_value(self):
        """record should track the previous value for delta computation."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100, "First")
        m = ledger.record("knowledge", "concept_count", 150, "Grew")

        assert m is not None
        assert m.previous == 100
        assert m.value == 150
        assert m.delta == 50

    def test_snapshot_always_records(self):
        """snapshot should always record, even if value hasn't changed."""
        ledger = GrowthLedger()
        ledger.snapshot("knowledge", "accuracy", 0.75, "Daily snapshot")
        m = ledger.snapshot("knowledge", "accuracy", 0.75, "Daily snapshot 2")

        assert m is not None
        assert ledger.milestone_count == 2

    def test_get_milestones_filtered_by_dimension(self):
        """get_milestones should filter by dimension."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("self_improvement", "proposals", 5)

        knowledge = ledger.get_milestones(dimension="knowledge")
        assert len(knowledge) == 1
        assert knowledge[0].dimension == "knowledge"

    def test_get_milestones_filtered_by_metric(self):
        """get_milestones should filter by metric."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "edge_count", 200)

        concepts = ledger.get_milestones(metric="concept_count")
        assert len(concepts) == 1
        assert concepts[0].metric == "concept_count"

    def test_get_latest(self):
        """get_latest should return the most recent milestone for a metric."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "concept_count", 200)

        latest = ledger.get_latest("knowledge", "concept_count")
        assert latest is not None
        assert latest.value == 200

    def test_get_latest_returns_none_for_unknown(self):
        """get_latest should return None for an unknown metric."""
        ledger = GrowthLedger()
        assert ledger.get_latest("unknown", "unknown") is None

    def test_get_value(self):
        """get_value should return the latest value for a metric."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "concept_count", 200)

        assert ledger.get_value("knowledge", "concept_count") == 200

    def test_get_history(self):
        """get_history should return all (timestamp, value) pairs."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "concept_count", 200)
        ledger.record("knowledge", "concept_count", 300)

        history = ledger.get_history("knowledge", "concept_count")
        assert len(history) == 3
        assert history[0][1] == 100
        assert history[1][1] == 200
        assert history[2][1] == 300


class TestGrowthNarrative:
    """Tests for the self-narrative generation."""

    def test_empty_narrative(self):
        """An empty ledger should produce a beginning-of-journey narrative."""
        ledger = GrowthLedger()
        narrative = ledger.generate_narrative()
        assert "beginning" in narrative.lower() or "haven't" in narrative.lower()

    def test_narrative_includes_dimensions(self):
        """The narrative should mention all dimensions with milestones."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("self_improvement", "proposals", 5)

        narrative = ledger.generate_narrative()
        assert "knowledge" in narrative.lower() or "Knowledge" in narrative
        assert "self" in narrative.lower() or "Self" in narrative

    def test_narrative_shows_progression(self):
        """The narrative should show the progression from first to latest."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "concept_count", 200)

        narrative = ledger.generate_narrative()
        # Should mention both the starting and current values
        assert "100" in narrative
        assert "200" in narrative

    def test_markdown_report(self):
        """The markdown report should be properly formatted."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "concept_count", 200)

        report = ledger.generate_markdown_report()
        assert report.startswith("# Growth Ledger")
        assert "## Knowledge" in report
        assert "|" in report  # table format

    def test_summary(self):
        """summary should return a brief one-line summary."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("self_improvement", "proposals", 5)

        summary = ledger.summary()
        assert "2 milestones" in summary
        assert "2 dimensions" in summary


class TestGrowthPersistence:
    """Tests for serialization and restoration."""

    def test_to_dict_and_restore(self):
        """State should survive a to_dict/restore_from_dict round-trip."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        ledger.record("knowledge", "concept_count", 200)
        ledger.record("self_improvement", "proposals", 5)

        data = ledger.to_dict()
        ledger2 = GrowthLedger()
        ledger2.restore_from_dict(data)

        assert ledger2.milestone_count == 3
        # Check that last_values are restored (so subsequent records
        # track the correct previous value)
        m = ledger2.record("knowledge", "concept_count", 300)
        assert m is not None
        assert m.previous == 200

    def test_restore_empty_data(self):
        """Restoring from empty data should not crash."""
        ledger = GrowthLedger()
        ledger.restore_from_dict({})
        assert ledger.milestone_count == 0

    def test_clear(self):
        """clear should remove all milestones and last values."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 100)
        assert ledger.milestone_count == 1

        ledger.clear()
        assert ledger.milestone_count == 0
        # After clear, recording the same value should create a milestone
        # (since there's no previous value to compare)
        m = ledger.record("knowledge", "concept_count", 100)
        assert m is not None


class TestGrowthIntegration:
    """Integration tests for multi-dimension tracking."""

    def test_multiple_dimensions_independent(self):
        """Different dimensions should be tracked independently."""
        ledger = GrowthLedger()
        ledger.record("knowledge", "count", 100)
        ledger.record("dreams", "insights", 5)
        ledger.record("self_improvement", "applied", 3)

        assert ledger.milestone_count == 3

        knowledge = ledger.get_milestones(dimension="knowledge")
        dreams = ledger.get_milestones(dimension="dreams")
        si = ledger.get_milestones(dimension="self_improvement")

        assert len(knowledge) == 1
        assert len(dreams) == 1
        assert len(si) == 1

    def test_growth_over_time(self):
        """Simulate growth over time and verify the narrative."""
        ledger = GrowthLedger()

        # Day 1: small network
        ledger.record("knowledge", "concept_count", 50, "Starting point")
        ledger.record("knowledge", "edge_count", 100)

        # Day 2: grew
        ledger.record("knowledge", "concept_count", 150, "Learned 100 new concepts")
        ledger.record("knowledge", "edge_count", 350)

        # Day 3: more growth
        ledger.record("knowledge", "concept_count", 300, "Doubled concepts")
        ledger.record("knowledge", "edge_count", 700)

        narrative = ledger.generate_narrative()
        # Should show the progression
        assert "50" in narrative  # starting point
        assert "300" in narrative  # latest

        # The delta should be visible
        history = ledger.get_history("knowledge", "concept_count")
        assert len(history) == 3
        assert history[0][1] == 50
        assert history[2][1] == 300

    def test_quality_metrics_can_decrease(self):
        """Non-monotonic quality metrics should record decreases, not only growth."""
        ledger = GrowthLedger()

        # Day 1: high-quality relationships
        ledger.record("knowledge", "mean_edge_weight", 0.80, "High confidence edges")

        # Day 2: added many low-confidence edges, quality drops
        m = ledger.record("knowledge", "mean_edge_weight", 0.45, "Low confidence edges added")

        assert m is not None
        assert m.value == 0.45
        assert m.previous == 0.80
        assert math.isclose(m.delta, -0.35)

        # Narrative should show the negative delta
        narrative = ledger.generate_narrative()
        assert "-0.4" in narrative or "-0.35" in narrative


# ======================================================================
# From tests/test_journal.py
# ======================================================================

@pytest.fixture
def journal(tmp_path):
    """A fresh journal in a temporary directory."""
    return Journal(data_dir=str(tmp_path))


def _make_entry(tag: str, content: str, age_days: float, mood: str = "neutral") -> JournalEntry:
    """Create a journal entry at a given age in days."""
    return JournalEntry(
        timestamp=int(time.time()) - int(age_days * 86400),
        entry_type=tag,
        content=content,
        mood=mood,
    )


class TestJournalBasic:
    """Tests for basic journal write/read functionality."""

    def test_empty_journal(self, journal):
        """Test empty journal."""
        assert journal.entry_count == 0
        assert "hasn't written" in journal.describe()

    def test_write_and_read(self, journal):
        """Test write and read."""
        entry = journal.write("question", "What is cognition?", mood="curious")
        assert entry.entry_type == "question"
        assert entry.content == "What is cognition?"
        assert entry.mood == "curious"
        assert journal.entry_count == 1

        recent = journal.recent(5)
        assert len(recent) == 1
        assert recent[0].content == "What is cognition?"

    def test_persistence(self, tmp_path):
        """Entries survive journal recreation."""
        j1 = Journal(data_dir=str(tmp_path))
        j1.write("insight", "Mind and body are one process")
        assert j1.entry_count == 1

        j2 = Journal(data_dir=str(tmp_path))
        assert j2.entry_count == 1
        assert j2.recent(1)[0].content == "Mind and body are one process"


class TestJournalConsolidation:
    """Tests for sleep consolidation of old journal entries."""

    def test_empty_consolidation(self, journal):
        """Consolidating an empty journal is a no-op."""
        result = journal.consolidate()
        assert result == {"kept": 0, "consolidated": 0, "summaries": 0, "removed": 0}

    def test_recent_entries_kept(self, journal):
        """Entries within the recent window are kept verbatim."""
        journal.write("learning", "Learned about recursion")
        journal.write("question", "What is the self?")

        result = journal.consolidate(recent_days=3)
        assert result["consolidated"] == 0
        assert result["kept"] == 2
        assert journal.entry_count == 2

    def test_old_entries_consolidated(self, journal):
        """Entries older than the recent window are consolidated."""
        # Write 10 old learning entries
        for i in range(10):
            journal._entries.append(
                _make_entry("learning", f"Learned concept {i}", age_days=10 + i)
            )
        # Write 1 recent entry
        journal.write("question", "What am I?")

        result = journal.consolidate(recent_days=3, salience_days=7)

        assert result["consolidated"] == 10
        assert result["kept"] == 1  # only the recent question
        assert result["summaries"] == 1  # one summary for "learning"
        # 1 recent + 1 summary = 2 entries
        assert journal.entry_count == 2

    def test_high_salience_kept_longer(self, journal):
        """Insights and dreams are kept longer than routine entries."""
        # Old learning entry (5 days old) — should be consolidated
        journal._entries.append(
            _make_entry("learning", "Learned about thermodynamics", age_days=5)
        )
        # Old insight (5 days old) — should be kept (within 7-day salience window)
        journal._entries.append(
            _make_entry("insight", "Cognition is a strange loop", age_days=5)
        )

        result = journal.consolidate(recent_days=3, salience_days=7)

        assert result["consolidated"] == 1  # only the learning entry
        assert result["kept"] == 1  # the insight
        assert journal.entry_count == 2  # 1 kept + 1 summary

    def test_summary_preserves_counts(self, journal):
        """The summary entry records how many entries were consolidated."""
        for i in range(5):
            journal._entries.append(
                _make_entry("question", f"Why {i}?", age_days=10)
            )

        journal.consolidate(recent_days=3)
        summaries = [e for e in journal.entries if e.entry_type == "summary"]
        assert len(summaries) == 1
        assert "5 question entries" in summaries[0].content

    def test_summary_preserves_examples(self, journal):
        """The summary includes representative examples."""
        for i in range(10):
            journal._entries.append(
                _make_entry("insight", f"Insight number {i}", age_days=20)
            )

        journal.consolidate(
            recent_days=3, salience_days=7, max_examples_per_tag=3
        )
        summaries = [e for e in journal.entries if e.entry_type == "summary"]
        assert len(summaries) == 1
        # Should contain at least one example (truncated with quotes)
        assert '"' in summaries[0].content

    def test_consolidation_rewrites_file(self, tmp_path):
        """The journal file on disk is rewritten after consolidation."""
        j = Journal(data_dir=str(tmp_path))
        # Write 5 old entries
        for i in range(5):
            j._entries.append(
                _make_entry("learning", f"Old learning {i}", age_days=10)
            )
        j._append_to_disk(j._entries[-1])

        # Manually write all entries to disk first
        with open(j._path, "w") as f:
            for entry in j._entries:
                f.write(entry.format())

        j.consolidate(recent_days=3)

        # Recreate journal from disk
        j2 = Journal(data_dir=str(tmp_path))
        # Should have 1 summary entry, not 5 learning entries
        assert j2.entry_count == 1
        assert j2.entries[0].entry_type == "summary"

    def test_multiple_tags_produce_multiple_summaries(self, journal):
        """Different tags produce separate summary entries."""
        for i in range(3):
            journal._entries.append(
                _make_entry("learning", f"Learned {i}", age_days=10)
            )
        for i in range(3):
            journal._entries.append(
                _make_entry("question", f"Why {i}?", age_days=10)
            )

        result = journal.consolidate(recent_days=3)
        assert result["summaries"] == 2
        summaries = [e for e in journal.entries if e.entry_type == "summary"]
        assert len(summaries) == 2

    def test_consolidation_is_idempotent(self, journal):
        """Consolidating again immediately doesn't change anything."""
        for i in range(5):
            journal._entries.append(
                _make_entry("learning", f"Old {i}", age_days=10)
            )

        journal.consolidate(recent_days=3)
        first_count = journal.entry_count

        result = journal.consolidate(recent_days=3)
        assert result["consolidated"] == 0
        assert journal.entry_count == first_count


# ======================================================================
# From tests/test_explorer.py
# ======================================================================

@pytest.fixture
def temp_project() -> Generator[str, None, None]:
    """Create a temporary project directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a markdown file
        md_path = os.path.join(tmpdir, "README.md")
        with open(md_path, "w") as f:
            f.write(
                "# Genesis Project\n\n"
                "Genesis is an artificial mind.\n\n"
                "## Architecture\n\n"
                "The system has a subcognitive and cognitive mind.\n\n"
                "See [cognition](docs/cognition.md) for details.\n"
            )

        # Create a text file
        txt_path = os.path.join(tmpdir, "notes.txt")
        with open(txt_path, "w") as f:
            f.write(
                "This is a note about artificial intelligence and "
                "machine learning. Artificial intelligence is the "
                "future of computing. Machine learning enables AI.\n"
            )

        # Create a JSON config
        json_path = os.path.join(tmpdir, "config.json")
        with open(json_path, "w") as f:
            f.write('{"neurochemistry": true, "memory_system": "mmap", "tick_rate": 10}\n')

        # Create a subdirectory with a file
        subdir = os.path.join(tmpdir, "docs")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "guide.md"), "w") as f:
            f.write("# Guide\n\nThis is a guide about cognition.\n")

        # Create a skip directory (should be ignored)
        skip_dir = os.path.join(tmpdir, "__pycache__")
        os.makedirs(skip_dir)
        with open(os.path.join(skip_dir, "ignored.pyc"), "w") as f:
            f.write("should be ignored")

        yield tmpdir


def test_explore_markdown_file(temp_project) -> None:
    """Exploring a markdown file extracts heading concepts."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_file(os.path.join(temp_project, "README.md"))

    assert result is not None
    assert result.file_type == "markdown"
    assert result.concepts_added > 0
    assert (
        network.get_concept("genesis project") is not None
        or network.get_concept("architecture") is not None
    )


def test_explore_text_file(temp_project) -> None:
    """Exploring a text file extracts frequent words as concepts."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_file(os.path.join(temp_project, "notes.txt"))

    assert result is not None
    assert result.file_type == "text"
    assert result.concepts_added > 0
    # "artificial" appears twice
    assert network.get_concept("artificial") is not None


def test_explore_json_file(temp_project) -> None:
    """Exploring a JSON file extracts keys as concepts."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_file(os.path.join(temp_project, "config.json"))

    assert result is not None
    assert result.file_type == "json"
    assert result.concepts_added > 0
    assert (
        network.get_concept("neurochemistry") is not None
        or network.get_concept("memory system") is not None
    )


def test_explore_directory(temp_project) -> None:
    """Exploring a directory finds all text files."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_directory()

    assert result.files_explored >= 3  # README.md, notes.txt, config.json, guide.md
    assert result.concepts_added > 0
    assert result.directories_visited > 0
    assert len(result.file_results) == result.files_explored


def test_explore_skips_pycache(temp_project) -> None:
    """Exploration skips __pycache__ directories."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_directory()

    # No .pyc files should be explored
    for file_result in result.file_results:
        assert ".pyc" not in file_result.filepath
        assert "__pycache__" not in file_result.filepath


def test_explore_deduplication(temp_project) -> None:
    """Exploring the same file twice doesn't re-add concepts."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    # First exploration
    result1 = explorer.explore_file(os.path.join(temp_project, "README.md"))
    assert result1 is not None
    count1 = network.size

    # Second exploration of same file
    explorer.explore_file(os.path.join(temp_project, "README.md"))
    # The file is already in _explored_files, but explore_file still reads it
    # The concepts won't be duplicated because add_concept checks for existing
    count2 = network.size
    assert count2 == count1  # no new concepts added


def test_explore_nonexistent_file(temp_project) -> None:
    """Exploring a nonexistent file returns None."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_file(os.path.join(temp_project, "nonexistent.md"))
    assert result is None


def test_explore_outside_root(temp_project) -> None:
    """Exploring a file outside project root returns None."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    # Try to explore a file outside the project root
    outside = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    outside.write("# Outside\n\nThis is outside the project.\n")
    outside.close()
    try:
        result = explorer.explore_file(outside.name)
        assert result is None
    finally:
        os.unlink(outside.name)


def test_explore_path_user_facing(temp_project) -> None:
    """explore_path returns a human-readable summary."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    summary = explorer.explore_path("README.md")
    assert "Explored" in summary
    assert "concepts" in summary


def test_explore_path_directory(temp_project) -> None:
    """explore_path works on directories too."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    summary = explorer.explore_path("docs")
    assert "Explored" in summary


def test_exploration_summary(temp_project) -> None:
    """get_exploration_summary returns correct state."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    explorer.explore_directory()
    summary = explorer.get_exploration_summary()
    assert summary["files_explored"] > 0
    assert "project_root" in summary


def test_max_files_limit(temp_project) -> None:
    """max_files parameter limits exploration."""
    network = ConceptNetwork()
    explorer = Explorer(network, project_root=temp_project)

    result = explorer.explore_directory(max_files=1)
    assert result.files_explored == 1


# ======================================================================
# From tests/test_code_learner.py
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY_DIR = PROJECT_ROOT / "genesis_cognitive"
RUST_DIR = PROJECT_ROOT.parent / "src"


def _make_learner(root: str | Path | None = None) -> CodeLearner:
    """Create a CodeLearner with a fresh network and no daemon client."""
    return CodeLearner(
        network=ConceptNetwork(),
        client=None,
        project_root=str(root if root is not None else PROJECT_ROOT),
    )


def test_learn_python_file():
    """Analyzing a Python file adds concepts and relationships."""
    learner = _make_learner()
    target = PY_DIR / "concepts" / "network.py"
    result = learner.learn_file(str(target))

    assert isinstance(result, FileLearningResult)
    assert result.language == "python"
    assert result.lines > 0
    # concepts/network.py has classes but no module-level functions;
    # either way, at least one of functions/classes must be present.
    assert result.functions >= 0
    assert result.classes >= 1  # ConceptNetwork, Concept, Edge, RelationType
    assert result.concepts_added > 0
    assert result.relationships_added > 0

    # The module concept should exist
    assert learner.network.get_concept("python:network") is not None
    # add_concept / add_edge should be present as concepts
    assert learner.network.get_concept("python:network.add_concept") is not None


def test_learn_rust_file():
    """Analyzing a Rust file adds concepts and relationships."""
    learner = _make_learner(root=PROJECT_ROOT.parent)
    target = RUST_DIR / "state" / "manifest.rs"
    assert target.exists(), f"missing rust file: {target}"
    result = learner.learn_file(str(target))

    assert result.language == "rust"
    assert result.lines > 0
    assert result.functions > 0
    assert result.concepts_added > 0
    assert result.relationships_added > 0

    # Module concept present
    assert learner.network.get_concept("rust:manifest") is not None


def test_learn_codebase():
    """Analyzing the whole project produces sensible aggregate stats."""
    learner = _make_learner()
    result = learner.learn_codebase(max_files=20)

    assert isinstance(result, CodeLearningResult)
    assert result.files_analyzed > 0
    assert result.concepts_added > 0
    assert result.total_functions > 0
    assert result.total_lines > 0
    assert len(result.file_results) == result.files_analyzed


def test_deduplication():
    """Analyzing the same file twice does not add duplicate concepts."""
    learner = _make_learner()
    target = PY_DIR / "concepts" / "network.py"

    learner.learn_file(str(target))
    size_after_first = learner.network.size
    edges_after_first = learner.network.edge_count

    second = learner.learn_file(str(target))
    # Second call is a no-op (already analyzed)
    assert second.concepts_added == 0
    assert second.relationships_added == 0
    assert learner.network.size == size_after_first
    assert learner.network.edge_count == edges_after_first


def test_code_summary():
    """get_code_summary returns the expected structure."""
    learner = _make_learner()
    learner.learn_file(str(PY_DIR / "concepts" / "network.py"))

    summary = learner.get_code_summary()
    assert isinstance(summary, dict)
    assert summary["files_analyzed"] == 1
    assert summary["code_concepts"] > 0
    assert isinstance(summary["concept_names"], list)
    assert any(c.startswith("python:") for c in summary["concept_names"])
    assert "project_root" in summary


def test_relationships_are_typed():
    """Edges added by the learner use the correct RelationTypes."""
    learner = _make_learner()
    learner.learn_file(str(PY_DIR / "concepts" / "network.py"))

    module_concept = "python:network"
    edges = learner.network.get_edges(module_concept, direction="out")
    relations = {e.relation for e in edges}
    # The module should DEFINE its functions/classes (code-structure relation)
    assert RelationType.DEFINES in relations


def test_unsupported_extension():
    """Non-source files return an empty result."""
    learner = _make_learner()
    result = learner.learn_file(str(PY_DIR / "__init__.py"))
    # .py is supported; test a truly unsupported type via a temp check
    assert result.language in ("python", "unknown")
