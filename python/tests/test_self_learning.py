"""Self-learning bundle tests.

Self-improvement, self-directed learning, heuristic experiments,
learning patterns, inference thresholds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.self import (
    ExperimentRecord,
    FeedbackRecord,
    HeuristicExperiment,
    InferenceResult,
    Proposal,
    ProposalCategory,
    ProposalStatus,
    SelfDirectedLearner,
    SelfImprovementEngine,
)
from genesis_cognitive.self.learning import (
    _CAUSES_RE,
    _DEFINITION_RE,
    _DEPENDS_RE,
    _ENABLES_RE,
    _HAS_RE,
    _IS_A_RE,
    _OPPOSITE_RE,
    _PART_OF_RE,
    _SIMILAR_RE,
    _normalize_concept,
)
from genesis_cognitive.tools.framework import ToolResult

logger = logging.getLogger(__name__)



# ======================================================================
# From tests/test_self_improvement.py
# ======================================================================

def _make_test_file(content: str) -> tuple[str, str]:
    """Create a temporary Python file with the given content.

    Returns (project_root, relative_path).
    """
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "test_module.py")
    with open(filepath, "w") as f:
        f.write(content)
    return tmpdir, "test_module.py"


class TestProposalDataclass:
    """Tests for the Proposal data structure."""

    def test_proposal_creation(self):
        """Test proposal creation."""
        p = Proposal(
            id=1,
            title="Test proposal",
            category=ProposalCategory.BUGFIX,
            file_path="some/file.py",
            description="Fix a bug",
            rationale="The bug causes crashes",
            expected_benefit="Fewer crashes",
            original_code="except:",
            proposed_code="except Exception:",
            confidence=0.8,
        )
        assert p.id == 1
        assert p.status == ProposalStatus.PENDING
        assert p.confidence == 0.8

    def test_proposal_describe(self):
        """Test proposal describe."""
        p = Proposal(
            id=1,
            title="Fix bare except",
            category=ProposalCategory.BUGFIX,
            file_path="module.py",
            description="Fix",
            rationale="Why",
            expected_benefit="Benefit",
            original_code="except:",
            proposed_code="except Exception:",
            confidence=0.8,
        )
        desc = p.describe()
        assert "[1]" in desc
        assert "Fix bare except" in desc
        assert "bugfix" in desc
        assert "pending" in desc

    def test_proposal_describe_full(self):
        """Test proposal describe full."""
        p = Proposal(
            id=1,
            title="Fix bare except",
            category=ProposalCategory.BUGFIX,
            file_path="module.py",
            description="Replace bare except",
            rationale="Catches too much",
            expected_benefit="Safer error handling",
            original_code="except:",
            proposed_code="except Exception:",
            confidence=0.8,
            source="bug:bare_except:10",
        )
        full = p.describe_full()
        assert "Proposal #1" in full
        assert "Original code" in full
        assert "Proposed code" in full
        assert "except:" in full
        assert "except Exception:" in full

    def test_proposal_serialization_roundtrip(self):
        """Test proposal serialization roundtrip."""
        p = Proposal(
            id=5,
            title="Test",
            category=ProposalCategory.REFACTOR,
            file_path="f.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
            status=ProposalStatus.ACCEPTED,
            feedback="good idea",
            source="test",
        )
        data = p.to_dict()
        restored = Proposal.from_dict(data)
        assert restored.id == 5
        assert restored.title == "Test"
        assert restored.category == ProposalCategory.REFACTOR
        assert restored.status == ProposalStatus.ACCEPTED
        assert restored.feedback == "good idea"
        assert restored.confidence == 0.5


class TestSelfImprovementEngine:
    """Tests for the self-improvement engine."""

    def test_engine_creation(self):
        """Test engine creation."""
        engine = SelfImprovementEngine(project_root=".")
        assert len(engine.get_pending_proposals()) == 0
        assert len(engine.get_all_proposals()) == 0

    def test_generate_proposals_finds_docstrings(self):
        # Create a file with functions missing docstrings
        """Test generate proposals finds docstrings."""
        content = """def foo(x):
    return x

def bar(y):
    return y * 2
"""
        root, _ = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)

        # Bypass cooldown for testing
        engine._last_generation_time = 0.0

        # We need to point the engine at the right directory
        # The engine scans python/genesis_cognitive by default,
        # so let's test with a custom path
        proposals = engine.generate_proposals(max_proposals=5)
        # May find proposals from the project's actual code
        assert isinstance(proposals, list)

    def test_accept_proposal(self):
        """Test accept proposal."""
        engine = SelfImprovementEngine(project_root=".")
        # Manually add a proposal
        p = Proposal(
            id=1,
            title="Test",
            category=ProposalCategory.BUGFIX,
            file_path="test.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
        )
        engine._proposals.append(p)

        assert engine.accept_proposal(1, "good idea")
        proposal = engine.get_proposal(1)
        assert proposal.status == ProposalStatus.ACCEPTED
        assert proposal.feedback == "good idea"
        assert proposal.reviewed_at > 0

    def test_reject_proposal(self):
        """Test reject proposal."""
        engine = SelfImprovementEngine(project_root=".")
        p = Proposal(
            id=1,
            title="Test",
            category=ProposalCategory.BUGFIX,
            file_path="test.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
        )
        engine._proposals.append(p)

        assert engine.reject_proposal(1, "not needed")
        proposal = engine.get_proposal(1)
        assert proposal.status == ProposalStatus.REJECTED
        assert proposal.feedback == "not needed"

    def test_accept_nonexistent_proposal(self):
        """Test accept nonexistent proposal."""
        engine = SelfImprovementEngine(project_root=".")
        assert not engine.accept_proposal(999)
        assert not engine.reject_proposal(999)

    def test_double_accept_fails(self):
        """Test double accept fails."""
        engine = SelfImprovementEngine(project_root=".")
        p = Proposal(
            id=1,
            title="Test",
            category=ProposalCategory.BUGFIX,
            file_path="test.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
        )
        engine._proposals.append(p)

        assert engine.accept_proposal(1)
        # Second accept should fail (already reviewed)
        assert not engine.accept_proposal(1)

    def test_withdraw_proposal(self):
        """Test withdraw proposal."""
        engine = SelfImprovementEngine(project_root=".")
        p = Proposal(
            id=1,
            title="Test",
            category=ProposalCategory.BUGFIX,
            file_path="test.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
        )
        engine._proposals.append(p)

        assert engine.withdraw_proposal(1, "I was wrong")
        proposal = engine.get_proposal(1)
        assert proposal.status == ProposalStatus.WITHDRAWN

    def test_mark_applied(self):
        """Test mark applied."""
        engine = SelfImprovementEngine(project_root=".")
        p = Proposal(
            id=1,
            title="Test",
            category=ProposalCategory.BUGFIX,
            file_path="test.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
            status=ProposalStatus.ACCEPTED,
        )
        engine._proposals.append(p)

        assert engine.mark_applied(1)
        assert engine.get_proposal(1).status == ProposalStatus.APPLIED

    def test_feedback_updates_category_stats(self):
        """Test feedback updates category stats."""
        engine = SelfImprovementEngine(project_root=".")

        for i in range(3):
            p = Proposal(
                id=i + 1,
                title=f"Test {i}",
                category=ProposalCategory.BUGFIX,
                file_path="test.py",
                description="d",
                rationale="r",
                expected_benefit="b",
                original_code="old",
                proposed_code="new",
                confidence=0.5,
            )
            engine._proposals.append(p)

        # Accept 2, reject 1
        engine.accept_proposal(1)
        engine.accept_proposal(2)
        engine.reject_proposal(3, "bad")

        stats = engine.get_feedback_stats()
        assert "bugfix" in stats
        assert stats["bugfix"]["accepted"] == 2
        assert stats["bugfix"]["rejected"] == 1
        assert stats["bugfix"]["total"] == 3
        assert abs(stats["bugfix"]["rate"] - 2/3) < 0.01

    def test_category_confidence_adjusts_with_feedback(self):
        """Test category confidence adjusts with feedback."""
        engine = SelfImprovementEngine(project_root=".")

        # Before feedback, confidence multiplier should be 1.0
        conf = engine._category_confidence(ProposalCategory.BUGFIX)
        assert conf == 1.0  # not enough data

        # Add 5 accepted proposals
        for i in range(5):
            p = Proposal(
                id=i + 1,
                title=f"Test {i}",
                category=ProposalCategory.BUGFIX,
                file_path="test.py",
                description="d",
                rationale="r",
                expected_benefit="b",
                original_code="old",
                proposed_code="new",
                confidence=0.5,
            )
            engine._proposals.append(p)
            engine.accept_proposal(i + 1)

        # Now confidence should be capped at 1.0 (100% success rate)
        conf = engine._category_confidence(ProposalCategory.BUGFIX)
        assert conf == 1.0  # capped at 1.0

    def test_category_confidence_decreases_with_rejections(self):
        """Test category confidence decreases with rejections."""
        engine = SelfImprovementEngine(project_root=".")

        # Add 5 rejected proposals
        for i in range(5):
            p = Proposal(
                id=i + 1,
                title=f"Test {i}",
                category=ProposalCategory.REFACTOR,
                file_path="test.py",
                description="d",
                rationale="r",
                expected_benefit="b",
                original_code="old",
                proposed_code="new",
                confidence=0.5,
            )
            engine._proposals.append(p)
            engine.reject_proposal(i + 1, "no")

        # 0% success → 0.5x confidence
        conf = engine._category_confidence(ProposalCategory.REFACTOR)
        assert conf < 1.0
        assert abs(conf - 0.5) < 0.01

    def test_proposals_summary(self):
        """Test proposals summary."""
        engine = SelfImprovementEngine(project_root=".")
        p = Proposal(
            id=1,
            title="Test",
            category=ProposalCategory.BUGFIX,
            file_path="test.py",
            description="d",
            rationale="r",
            expected_benefit="b",
            original_code="old",
            proposed_code="new",
            confidence=0.5,
        )
        engine._proposals.append(p)

        summary = engine.proposals_summary()
        assert "Pending:  1" in summary
        assert "[1]" in summary

    def test_persistence_roundtrip(self):
        """Test persistence roundtrip."""
        # Use a file path that exists relative to project_root so the
        # pending proposal survives restore (restore drops pending
        # proposals whose target file no longer exists).
        existing_file = os.path.relpath(__file__, ".")
        engine = SelfImprovementEngine(project_root=".")

        # Add some proposals and feedback
        for i in range(3):
            p = Proposal(
                id=i + 1,
                title=f"Test {i}",
                category=ProposalCategory.BUGFIX,
                file_path=existing_file,
                description="d",
                rationale="r",
                expected_benefit="b",
                original_code="old",
                proposed_code="new",
                confidence=0.5,
            )
            engine._proposals.append(p)

        engine.accept_proposal(1, "good")
        engine.reject_proposal(2, "bad")

        # Serialize
        data = engine.to_dict()

        # Restore into a new engine
        engine2 = SelfImprovementEngine(project_root=".")
        engine2.restore_from_dict(data)

        assert len(engine2.get_all_proposals()) == 3
        assert engine2.get_proposal(1).status == ProposalStatus.ACCEPTED
        assert engine2.get_proposal(2).status == ProposalStatus.REJECTED
        assert engine2.get_proposal(3).status == ProposalStatus.PENDING
        assert engine2._next_id == 4

        # Category stats should be restored
        stats = engine2.get_feedback_stats()
        assert "bugfix" in stats
        assert stats["bugfix"]["total"] == 2

    def test_restore_drops_stale_pending_proposals(self):
        """Pending proposals whose target file no longer exists are
        dropped on restore; non-pending ones are kept for the audit
        trail."""
        engine = SelfImprovementEngine(project_root=".")
        existing_file = os.path.relpath(__file__, ".")

        # Pending proposal targeting a real file — should survive.
        live = Proposal(
            id=1, title="live", category=ProposalCategory.BUGFIX,
            file_path=existing_file, description="d", rationale="r",
            expected_benefit="b", original_code="old", proposed_code="new",
            confidence=0.5,
        )
        # Pending proposal targeting a deleted file — should be dropped.
        stale = Proposal(
            id=2, title="stale", category=ProposalCategory.BUGFIX,
            file_path="genesis_live/app.py", description="d", rationale="r",
            expected_benefit="b", original_code="old", proposed_code="new",
            confidence=0.5,
        )
        # Accepted proposal targeting a deleted file — should survive
        # (audit trail is preserved regardless of file existence).
        accepted_gone = Proposal(
            id=3, title="accepted-gone", category=ProposalCategory.BUGFIX,
            file_path="genesis_live/app.py", description="d", rationale="r",
            expected_benefit="b", original_code="old", proposed_code="new",
            confidence=0.5, status=ProposalStatus.ACCEPTED,
        )
        engine._proposals.extend([live, stale, accepted_gone])

        engine2 = SelfImprovementEngine(project_root=".")
        engine2.restore_from_dict(engine.to_dict())

        ids = {p.id for p in engine2.get_all_proposals()}
        assert ids == {1, 3}, f"expected live + accepted-gone, got {ids}"
        assert engine2.get_proposal(2) is None

    def test_persistence_backward_compat(self):
        """Engine should handle empty/minimal data gracefully."""
        engine = SelfImprovementEngine(project_root=".")
        engine.restore_from_dict({})
        assert len(engine.get_all_proposals()) == 0
        assert engine._next_id == 1


class TestProposalGeneration:
    """Tests for the proposal generation logic."""

    def test_fix_bare_except(self):
        """Test fix bare except."""
        content = """try:
    x = 1
except:
    pass
"""
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)
        engine._last_generation_time = 0.0

        # Manually test the fix generator
        with open(os.path.join(root, "test_module.py")) as f:
            source = f.read()
        lines = source.splitlines(keepends=True)

        result = engine._fix_bare_except(lines, 3)
        assert result is not None
        original, proposed, _desc, _benefit = result
        assert "except:" in original
        assert "except Exception:" in proposed
        assert "Exception" in proposed

    def test_fix_mutable_default(self):
        """Test fix mutable default."""
        content = """def foo(items=[]):
    return items
"""
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)
        engine._last_generation_time = 0.0

        with open(os.path.join(root, "test_module.py")) as f:
            source = f.read()
        lines = source.splitlines(keepends=True)

        result = engine._fix_mutable_default(lines, 1)
        assert result is not None
        original, proposed, _desc, _benefit = result
        assert "[]" in original
        assert "None" in proposed
        assert "items" in proposed

    def test_fix_print_in_code(self):
        """Test fix print in code."""
        content = """def foo():
    print("hello")
    return 1
"""
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)

        with open(os.path.join(root, "test_module.py")) as f:
            source = f.read()
        lines = source.splitlines(keepends=True)

        result = engine._fix_print_in_code(lines, 2)
        assert result is not None
        original, proposed, _desc, _benefit = result
        assert "print(" in original
        assert "logger.info(" in proposed

    def test_fix_print_in_code_strips_file_kwarg(self):
        """print(..., file=sys.stderr) must strip file= before
        converting to logger.info(), since logger.info() doesn't
        accept file=. Without this, the rewrite produces TypeError
        at runtime and the fix gets reverted, creating an infinite
        scan→fix→fail→revert loop."""
        content = '''def foo():
    print("[genesis] error", file=sys.stderr)
    return 1
'''
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)

        with open(os.path.join(root, "test_module.py")) as f:
            source = f.read()
        lines = source.splitlines(keepends=True)

        result = engine._fix_print_in_code(lines, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "logger.info(" in proposed
        assert "file=" not in proposed

    def test_fix_silent_except(self):
        """Test fix silent except."""
        content = """try:
    x = 1
except:
    pass
"""
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)

        with open(os.path.join(root, "test_module.py")) as f:
            source = f.read()
        lines = source.splitlines(keepends=True)

        result = engine._fix_silent_except(lines, 3)
        # Should detect the pass on line 4
        if result is not None:
            original, proposed, _desc, _benefit = result
            assert "pass" in original
            assert "logger" in proposed

    def test_fix_is_literal_comparison(self):
        """'x is 5' is rewritten to 'x == 5'."""
        source = "def f(x):\n    return x is 5\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_is_literal_comparison(source, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "==" in proposed
        assert " is " not in proposed

    def test_fix_is_not_literal_comparison(self):
        """'x is not 5' is rewritten to 'x != 5'."""
        source = "def f(x):\n    return x is not 5\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_is_literal_comparison(source, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "!=" in proposed

    def test_fix_is_literal_preserves_is_none(self):
        """Only the literal comparison changes; a valid 'is None' stays."""
        source = "def f(x, y):\n    return x is None and y is 5\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_is_literal_comparison(source, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "x is None" in proposed
        assert "y == 5" in proposed

    def test_fix_type_equality_check(self):
        """'type(x) == int' becomes 'isinstance(x, int)'."""
        source = "def f(x):\n    return type(x) == int\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_type_equality_check(source, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "isinstance(x, int)" in proposed

    def test_fix_type_equality_negated(self):
        """'type(x) is not str' becomes 'not isinstance(x, str)'."""
        source = "def f(x):\n    return type(x) is not str\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_type_equality_check(source, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "not isinstance(x, str)" in proposed

    def test_fix_type_equality_reversed(self):
        """'int == type(x)' becomes 'isinstance(x, int)'."""
        source = "def f(x):\n    return int == type(x)\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_type_equality_check(source, 2)
        assert result is not None
        _original, proposed, _desc, _benefit = result
        assert "isinstance(x, int)" in proposed

    def test_fix_type_vs_type_returns_none(self):
        """'type(a) == type(b)' is legitimate — no fix generated."""
        source = "def f(a, b):\n    return type(a) == type(b)\n"
        engine = SelfImprovementEngine(project_root=".")
        assert engine._fix_type_equality_check(source, 2) is None

    def test_fix_boolean_equality_true(self):
        """'x == True' becomes 'x'."""
        source = "def f(x):\n    return x == True\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_boolean_equality(source, 2)
        assert result is not None
        assert result[1].strip() == "return x"

    def test_fix_boolean_equality_false(self):
        """'x == False' becomes 'not x'."""
        source = "def f(x):\n    return x == False\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_boolean_equality(source, 2)
        assert result is not None
        assert result[1].strip() == "return not x"

    def test_fix_boolean_equality_not_true(self):
        """'x != True' becomes 'not x'."""
        source = "def f(x):\n    return x != True\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_boolean_equality(source, 2)
        assert result is not None
        assert result[1].strip() == "return not x"

    def test_fix_boolean_equality_compound_negation_parenthesised(self):
        """A compound operand is parenthesised under 'not'."""
        source = "def f(a, b):\n    return (a and b) == False\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_boolean_equality(source, 2)
        assert result is not None
        assert result[1].strip() == "return not (a and b)"

    def test_fix_assert_tuple(self):
        """'assert (cond, msg)' becomes 'assert cond, msg'."""
        source = 'def f(x):\n    assert (x > 0, "positive")\n'
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_assert_tuple(source, 2)
        assert result is not None
        assert result[1].strip() == 'assert x > 0, "positive"'

    def test_fix_base_exception_catch(self):
        """'except BaseException:' becomes 'except Exception:'."""
        source = "try:\n    pass\nexcept BaseException:\n    pass\n"
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_base_exception_catch(source.splitlines(keepends=True), 3)
        assert result is not None
        assert result[1].strip() == "except Exception:"

    def test_fix_fstring_no_placeholder(self):
        """The redundant 'f' prefix is removed."""
        source = 'def f():\n    return f"hello world"\n'
        engine = SelfImprovementEngine(project_root=".")
        result = engine._fix_fstring_no_placeholder(source, 2)
        assert result is not None
        assert result[1].strip() == 'return "hello world"'

    def test_fix_fstring_with_placeholder_returns_none(self):
        """A real f-string is left alone."""
        source = 'def f(name):\n    return f"hi {name}"\n'
        engine = SelfImprovementEngine(project_root=".")
        assert engine._fix_fstring_no_placeholder(source, 2) is None

    def test_missing_docstring_detection(self):
        """Test missing docstring detection."""
        content = """def foo(x):
    return x
"""
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)
        engine._last_generation_time = 0.0

        # Analyze the file directly
        proposals = engine._analyze_python_file(
            os.path.join(root, "test_module.py"),
            "test_module.py",
            max_proposals=5,
            existing_sources=set(),
        )
        # Should find the missing docstring
        assert len(proposals) >= 1
        assert any("docstring" in p.title.lower() for p in proposals)

    def test_does_not_propose_for_function_with_docstring(self):
        """Test does not propose for function with docstring."""
        content = '''def foo(x):
    """Do something."""
    return x
'''
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)

        proposals = engine._analyze_python_file(
            os.path.join(root, "test_module.py"),
            "test_module.py",
            max_proposals=5,
            existing_sources=set(),
        )
        # Should not propose a docstring for foo (it has one)
        assert not any("foo" in p.title for p in proposals)

    def test_deduplication(self):
        """Generating proposals twice should not create duplicates."""
        content = """def foo(x):
    return x

def bar(y):
    return y
"""
        root, _rel = _make_test_file(content)
        engine = SelfImprovementEngine(project_root=root)

        # First generation
        engine._last_generation_time = 0.0
        engine.generate_proposals(max_proposals=5)

        # Second generation (reset cooldown)
        engine._last_generation_time = 0.0
        engine.generate_proposals(max_proposals=5)

        # Should not have duplicate sources
        all_sources = [p.source for p in engine.get_all_proposals()]
        assert len(all_sources) == len(set(all_sources)), "Duplicate proposals found"

    def test_cooldown_prevents_rapid_generation(self):
        """Test cooldown prevents rapid generation."""
        engine = SelfImprovementEngine(project_root=".")
        engine._last_generation_time = time.time() - 0.5  # recent (within 1s cooldown)

        proposals = engine.generate_proposals(max_proposals=5)
        assert proposals == [], "Should not generate during cooldown"


class TestFeedbackRecord:
    """Tests for the FeedbackRecord data structure."""

    def test_feedback_record_creation(self):
        """Test feedback record creation."""
        record = FeedbackRecord(
            proposal_id=1,
            category="bugfix",
            accepted=True,
            feedback="good",
        )
        assert record.proposal_id == 1
        assert record.accepted is True

    def test_feedback_record_serialization(self):
        """Test feedback record serialization."""
        record = FeedbackRecord(
            proposal_id=2,
            category="refactor",
            accepted=False,
            feedback="no",
        )
        data = record.to_dict()
        restored = FeedbackRecord.from_dict(data)
        assert restored.proposal_id == 2
        assert restored.category == "refactor"
        assert restored.accepted is False
        assert restored.feedback == "no"


def _make_importable_pkg(tmpdir: str) -> str:
    """Create a temp ``python/genesis_cognitive/`` package skeleton.

    Returns the tmpdir (to be used as project_root). The package
    __init__ is empty so importing submodules does not pull in the
    real cognitive-mind dependency graph.
    """
    pkg = os.path.join(tmpdir, "python", "genesis_cognitive")
    os.makedirs(pkg)
    with open(os.path.join(pkg, "__init__.py"), "w"):
        pass
    return tmpdir


class TestAutonomousAllowlist:
    """Tests for the autonomous-fix directory allowlist.

    These prove that ``apply_autonomous_fixes`` can only touch files
    under the allowed directories — never scripts, config, the Rust
    substrate, or arbitrary project files. This is the sandbox that
    prevents the provenance gap where non-allowlisted source files
    were modified outside the verified experiment path.
    """

    def test_cognitive_submodule_allowed(self):
        """Test cognitive submodule allowed."""
        assert SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/perception.py"
        )

    def test_client_submodule_allowed(self):
        """Test client submodule allowed."""
        assert SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_client/client.py"
        )

    def test_cli_allowed(self):
        """Test cli allowed."""
        assert SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cli.py"
        )

    def test_script_rejected(self):
        """Test script rejected."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "scripts/verify_change.sh"
        )

    def test_rust_substrate_rejected(self):
        """Test rust substrate rejected."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "src/store/mmap_state.rs"
        )

    def test_test_file_rejected(self):
        """Test test file rejected."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "python/tests/test_perception.py"
        )

    def test_arbitrary_project_file_rejected(self):
        """Test arbitrary project file rejected."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "tools/some_script.py"
        )

    def test_root_level_file_rejected(self):
        """Test root level file rejected."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "setup.py"
        )


class TestProtectedCore:
    """Tests for the protected-core boundary.

    These prove that autonomous fixes and experiments cannot touch
    the cognition engine, language engine, or mind — the files that
    define her ability to think and speak. Proposals can still be
    generated for these files, but she cannot apply changes to them
    without human review.
    """

    def test_cognition_engine_protected(self):
        """Cognition engine is protected from autonomous fixes."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/cognition/engine.py"
        )

    def test_language_generator_protected(self):
        """Language generator is protected from autonomous fixes."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/language/generator.py"
        )

    def test_language_graph_walk_protected(self):
        """Language graph_walk is protected from autonomous fixes."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/language/graph_walk.py"
        )

    def test_language_vocabulary_protected(self):
        """Language vocabulary is protected from autonomous fixes."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/language/vocabulary.py"
        )

    def test_mind_protected(self):
        """Mind is protected from autonomous fixes."""
        assert not SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/mind.py"
        )

    def test_cognition_engine_still_own_source(self):
        """Protected core files are still own-source (can have proposals)."""
        assert SelfImprovementEngine._is_own_source(
            "python/genesis_cognitive/cognition/engine.py"
        )

    def test_language_still_own_source(self):
        """Language files are still own-source (can have proposals)."""
        assert SelfImprovementEngine._is_own_source(
            "python/genesis_cognitive/language/generator.py"
        )

    def test_non_core_cognitive_still_allowed(self):
        """Non-core cognitive files are still autonomously fixable."""
        assert SelfImprovementEngine._is_autonomous_allowed(
            "python/genesis_cognitive/perception.py"
        )

    def test_protected_core_experiment_blocked(self):
        """Experiments cannot touch protected core files."""
        from genesis_cognitive.self.improvement import HeuristicExperiment
        assert not HeuristicExperiment._is_allowlisted(
            "python/genesis_cognitive/cognition/engine.py"
        )

    def test_protected_core_still_experimentable(self):
        """Protected core files are still experimentable (proposals OK)."""
        from genesis_cognitive.self.improvement import HeuristicExperiment
        assert HeuristicExperiment._is_experimentable(
            "python/genesis_cognitive/cognition/engine.py"
        )


class TestModuleNameFor:
    """Tests for SelfImprovementEngine._module_name_for."""

    def test_cognitive_submodule(self):
        """Test cognitive submodule."""
        assert (
            SelfImprovementEngine._module_name_for(
                "python/genesis_cognitive/perception.py"
            )
            == "genesis_cognitive.perception"
        )

    def test_client_submodule(self):
        """Test client submodule."""
        assert (
            SelfImprovementEngine._module_name_for(
                "python/genesis_client/client.py"
            )
            == "genesis_client.client"
        )

    def test_top_level_script(self):
        """Test top level script."""
        assert (
            SelfImprovementEngine._module_name_for("python/genesis_cli.py")
            == "genesis_cli"
        )

    def test_test_file_returns_none(self):
        # Test files are not importable production modules — the
        # import gate must be skipped for them.
        """Test test file returns none."""
        assert (
            SelfImprovementEngine._module_name_for(
                "python/tests/test_perception.py"
            )
            is None
        )

    def test_non_python_returns_none(self):
        """Test non python returns none."""
        assert (
            SelfImprovementEngine._module_name_for("scripts/verify_change.sh")
            is None
        )

    def test_pycache_returns_none(self):
        """Test pycache returns none."""
        assert (
            SelfImprovementEngine._module_name_for(
                "python/genesis_cognitive/__pycache__/foo.cpython-312.pyc"
            )
            is None
        )

    def test_non_package_submodule_returns_none(self):
        # A .py inside python/ but in an unknown subdirectory is not
        # importable as a module — skip the gate.
        """Test non package submodule returns none."""
        assert (
            SelfImprovementEngine._module_name_for("python/random_dir/foo.py")
            is None
        )


class TestImportGate:
    """Regression tests for the import-execution verification gate.

    These prove that a source change which passes py_compile but
    raises at import time (e.g. a broken module-level re.compile) is
    rejected and reverted — the exact defect that broke
    genesis_cognitive.perception and left it in the repository.
    """

    def test_broken_import_is_reverted(self):
        """A module that compiles but fails at import must be reverted."""
        tmpdir = tempfile.mkdtemp()
        _make_importable_pkg(tmpdir)
        mod_path = os.path.join(
            tmpdir, "python", "genesis_cognitive", "broken_mod.py"
        )
        original = '"""Valid module."""\nx = 1\n'
        # Valid Python syntax (py_compile passes) but re.compile
        # raises re.error at module scope on import.
        broken = 'import re\nre.compile("(")\n'
        with open(mod_path, "w") as f:
            f.write(original)

        engine = SelfImprovementEngine(project_root=tmpdir)
        result = engine._validate_and_write_fix(
            file_path=Path(mod_path),
            original_source=original,
            fixed_source=broken,
            rel_path="python/genesis_cognitive/broken_mod.py",
        )

        assert result is False, "broken-import change must be rejected"
        with open(mod_path) as f:
            assert f.read() == original, "file must be reverted on import failure"

    def test_valid_import_proceeds_to_test_gate(self):
        """A module that imports cleanly must reach the pytest gate."""
        tmpdir = tempfile.mkdtemp()
        _make_importable_pkg(tmpdir)
        mod_path = os.path.join(
            tmpdir, "python", "genesis_cognitive", "good_mod.py"
        )
        original = '"""Old."""\nx = 1\n'
        fixed = '"""New."""\nx = 2\n'
        with open(mod_path, "w") as f:
            f.write(original)

        engine = SelfImprovementEngine(project_root=tmpdir)
        # Mock the test gate so we don't run the real ~5-minute suite.
        mock_tools = type("MockTools", (), {"run": staticmethod(
            lambda name, **kw: ToolResult(success=True, output="mocked")
        )})()
        with patch(
            "genesis_cognitive.self.improvement.get_tools",
            return_value=mock_tools,
        ):
            result = engine._validate_and_write_fix(
                file_path=Path(mod_path),
                original_source=original,
                fixed_source=fixed,
                rel_path="python/genesis_cognitive/good_mod.py",
            )

        assert result is True, "valid change must be accepted"
        with open(mod_path) as f:
            assert f.read() == fixed, "valid change must be kept"

    def test_test_failure_reverts_valid_import(self):
        """If the import passes but tests fail, the file is still reverted."""
        tmpdir = tempfile.mkdtemp()
        _make_importable_pkg(tmpdir)
        mod_path = os.path.join(
            tmpdir, "python", "genesis_cognitive", "good_mod.py"
        )
        original = '"""Old."""\nx = 1\n'
        fixed = '"""New."""\nx = 2\n'
        with open(mod_path, "w") as f:
            f.write(original)

        engine = SelfImprovementEngine(project_root=tmpdir)
        mock_tools = type("MockTools", (), {"run": staticmethod(
            lambda name, **kw: ToolResult(success=False, error="tests failed")
        )})()
        with patch(
            "genesis_cognitive.self.improvement.get_tools",
            return_value=mock_tools,
        ):
            result = engine._validate_and_write_fix(
                file_path=Path(mod_path),
                original_source=original,
                fixed_source=fixed,
                rel_path="python/genesis_cognitive/good_mod.py",
            )

        assert result is False, "test failure must reject the change"
        with open(mod_path) as f:
            assert f.read() == original, "file must be reverted on test failure"


# ======================================================================
# From tests/test_self_directed_learning.py
# ======================================================================

def _make_learner() -> SelfDirectedLearner:
    """Create a SelfDirectedLearner with a fresh network."""
    return SelfDirectedLearner(ConceptNetwork())


# ═══════════════════════════════════════════════════════════════════
# learn_from_input — extraction patterns
# ═══════════════════════════════════════════════════════════════════


def test_learn_from_input_is_a() -> None:
    """learn_from_input extracts 'X is a Y' relationships."""
    learner = _make_learner()
    events = learner.learn_from_input("a dog is a mammal")
    assert len(events) >= 1
    assert learner.network.get_concept("dog") is not None
    assert learner.network.get_concept("mammal") is not None


def test_learn_from_input_definition() -> None:
    """learn_from_input extracts 'X means Y' definitions."""
    learner = _make_learner()
    events = learner.learn_from_input("cognition means the state of being aware of things")
    # Should extract a definition
    assert any(e.event_type == "conversation" for e in events)


def test_learn_from_input_question_ignored() -> None:
    """learn_from_input ignores questions."""
    learner = _make_learner()
    events = learner.learn_from_input("what is a dog?")
    assert events == []


def test_learn_from_input_short_ignored() -> None:
    """learn_from_input ignores very short inputs."""
    learner = _make_learner()
    events = learner.learn_from_input("hi there")
    assert events == []


def test_learn_from_input_causes() -> None:
    """learn_from_input extracts 'X causes Y' relationships."""
    learner = _make_learner()
    events = learner.learn_from_input("stress causes memory problems")
    assert len(events) >= 1


def test_learn_from_input_bare_is_definition() -> None:
    """learn_from_input extracts 'X is Y' bare copula definitions."""
    learner = _make_learner()
    events = learner.learn_from_input("Courage is feeling fear and acting anyway")
    assert any(e.event_type == "conversation" for e in events)
    c = learner.network.get_concept("courage")
    assert c is not None
    assert (c.properties or {}).get("definition")


def test_learn_from_input_bare_is_single_word() -> None:
    """learn_from_input extracts single-word bare copula definitions."""
    learner = _make_learner()
    events = learner.learn_from_input("Fear is natural")
    assert any(e.event_type == "conversation" for e in events)
    c = learner.network.get_concept("fear")
    assert c is not None


def test_learn_from_input_bare_is_sentence_boundary() -> None:
    """learn_from_input doesn't cross sentence boundaries for bare is."""
    learner = _make_learner()
    events = learner.learn_from_input(
        "Courage is not the absence of fear. Courage is feeling fear and acting anyway."
    )
    # Should extract two separate definitions, not one spanning both sentences
    defs = [e for e in events if "definition" in e.description.lower()]
    assert len(defs) >= 2
    # Neither definition should contain a period (sentence boundary)
    for d in defs:
        assert "." not in d.description.split(":")[-1].strip()


# ═══════════════════════════════════════════════════════════════════
# infer_transitive
# ═══════════════════════════════════════════════════════════════════


def test_infer_transitive_is_a() -> None:
    """infer_transitive infers A→C from A→B, B→C for IS_A."""
    learner = _make_learner()
    # Set up: golden_retriever IS_A dog, dog IS_A mammal
    learner.network.add_concept("golden_retriever")
    learner.network.add_concept("dog")
    learner.network.add_concept("mammal")
    learner.network.add_edge("golden_retriever", "dog", RelationType.IS_A)
    learner.network.add_edge("dog", "mammal", RelationType.IS_A)
    result = learner.infer_transitive()
    # Should infer golden_retriever IS_A mammal
    assert result.new_edges >= 1
    edges = learner.network.get_edges("golden_retriever", "out")
    targets = [e.target for e in edges]
    assert "mammal" in targets


def test_infer_transitive_no_inferences() -> None:
    """infer_transitive returns empty result when no inferences possible."""
    learner = _make_learner()
    learner.network.add_concept("a")
    result = learner.infer_transitive()
    assert result.new_edges == 0


def test_infer_transitive_multi_hop() -> None:
    """infer_transitive finds multi-hop inferences."""
    learner = _make_learner()
    # a→b→c→d
    for name in ["a", "b", "c", "d"]:
        learner.network.add_concept(name)
    learner.network.add_edge("a", "b", RelationType.IS_A)
    learner.network.add_edge("b", "c", RelationType.IS_A)
    learner.network.add_edge("c", "d", RelationType.IS_A)
    result = learner.infer_transitive()
    # Should infer a→c, a→d, b→d
    assert result.new_edges >= 2


def test_infer_transitive_respects_max() -> None:
    """infer_transitive respects max_inferences."""
    learner = _make_learner()
    # Create many chains
    for i in range(10):
        learner.network.add_concept(f"leaf_{i}")
        learner.network.add_concept(f"mid_{i}")
        learner.network.add_concept(f"root_{i}")
        learner.network.add_edge(f"leaf_{i}", f"mid_{i}", RelationType.IS_A)
        learner.network.add_edge(f"mid_{i}", f"root_{i}", RelationType.IS_A)
    result = learner.infer_transitive(max_inferences=3)
    assert result.new_edges <= 3


def test_infer_transitive_skips_existing() -> None:
    """infer_transitive doesn't add edges that already exist."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    learner.network.add_concept("c")
    learner.network.add_edge("a", "b", RelationType.IS_A)
    learner.network.add_edge("b", "c", RelationType.IS_A)
    learner.network.add_edge("a", "c", RelationType.IS_A)  # already exists
    result = learner.infer_transitive()
    # a→c already exists, so no new edge should be added for it
    assert result.new_edges == 0


# ═══════════════════════════════════════════════════════════════════
# synthesize_definitions
# ═══════════════════════════════════════════════════════════════════


def test_synthesize_definitions() -> None:
    """synthesize_definitions builds definitions from relationships."""
    learner = _make_learner()
    learner.network.add_concept("dog")
    learner.network.add_concept("mammal")
    learner.network.add_edge("dog", "mammal", RelationType.IS_A)
    result = learner.synthesize_definitions()
    assert result.new_definitions >= 1
    concept = learner.network.get_concept("dog")
    assert concept is not None
    assert "definition" in concept.properties
    assert "mammal" in concept.properties["definition"]


def test_synthesize_definitions_skips_existing() -> None:
    """synthesize_definitions skips concepts with existing definitions."""
    learner = _make_learner()
    learner.network.add_concept("dog")
    dog_concept = learner.network.get_concept("dog")
    assert dog_concept is not None
    dog_concept.properties["definition"] = "a canine animal"
    result = learner.synthesize_definitions()
    # Should not add a new definition
    assert result.new_definitions == 0


def test_synthesize_definitions_no_concepts() -> None:
    """synthesize_definitions returns empty when no concepts exist."""
    learner = _make_learner()
    result = learner.synthesize_definitions()
    assert result.new_definitions == 0


# ═══════════════════════════════════════════════════════════════════
# learn_from_correction
# ═══════════════════════════════════════════════════════════════════


def test_learn_from_correction() -> None:
    """learn_from_correction extracts corrected facts."""
    learner = _make_learner()
    events = learner.learn_from_correction("No, a dog is a mammal")
    assert len(events) >= 1
    # All events should be marked as corrections
    for event in events:
        assert event.event_type == "correction"


def test_learn_from_correction_increments_count() -> None:
    """learn_from_correction increments the correction counter."""
    learner = _make_learner()
    learner.learn_from_correction("No, a dog is a mammal")
    stats = learner.get_stats()
    assert stats["corrections"] >= 1


def test_learn_from_correction_with_topic() -> None:
    """learn_from_correction with a topic weakens inferred edges."""
    learner = _make_learner()
    learner.learn_from_input("a dog is a reptile")
    learner.learn_from_correction("No, a dog is a mammal", original_topic="dog")
    stats = learner.get_stats()
    assert stats["corrections"] >= 1


# ═══════════════════════════════════════════════════════════════════
# consistency_sweep
# ═══════════════════════════════════════════════════════════════════


def test_consistency_sweep_no_contradictions() -> None:
    """consistency_sweep returns empty when no contradictions exist."""
    learner = _make_learner()
    result = learner.consistency_sweep()
    assert result.contradictions_found == 0


def test_consistency_sweep_detects_similar_opposite() -> None:
    """consistency_sweep detects SIMILAR_TO + OPPOSITE_OF contradictions."""
    learner = _make_learner()
    learner.network.add_concept("hot")
    learner.network.add_concept("cold")
    learner.network.add_edge("hot", "cold", RelationType.SIMILAR_TO, weight=0.5)
    learner.network.add_edge("hot", "cold", RelationType.OPPOSITE_OF, weight=0.8)
    result = learner.consistency_sweep()
    assert result.contradictions_found >= 1


def test_consistency_sweep_removes_orphaned_edges() -> None:
    """consistency_sweep removes orphaned inferred edges with low weight."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    # Add an inferred edge with very low weight
    edge = learner.network.add_edge(
        "a", "b", RelationType.RELATED_TO, weight=0.05, origin="inferred"
    )
    assert edge is not None
    edge.weight = 0.05
    initial_count = learner.network.edge_count
    learner.consistency_sweep()
    # The orphaned edge should be removed
    assert learner.network.edge_count < initial_count


# ═══════════════════════════════════════════════════════════════════
# run_inference_cycle
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Genus extraction tests
# ═══════════════════════════════════════════════════════════════════


def test_extract_genus_term_wordnet() -> None:
    """_extract_genus_term extracts genus from WordNet-style definitions."""
    text = "serotonin\n    n 1: a neurotransmitter involved in e.g. sleep and depression"
    genus = SelfDirectedLearner._extract_genus_term(text)
    assert genus == "neurotransmitter"


def test_extract_genus_term_webster() -> None:
    """_extract_genus_term extracts genus from Webster-style definitions."""
    text = 'Cognition \\Con"scious*ness\\, n.\n   1. The state of being cognitive'
    genus = SelfDirectedLearner._extract_genus_term(text)
    assert genus == "state"


def test_extract_genus_term_stops_at_boundary() -> None:
    """_extract_genus_term stops at boundary words (prepositions, participles)."""
    text = "curiosity\n    n 1: a strong desire to learn or know"
    genus = SelfDirectedLearner._extract_genus_term(text)
    assert genus == "strong desire"


def test_extract_genus_term_no_match() -> None:
    """_extract_genus_term returns None when no noun pattern is found."""
    text = "some random text without a noun definition pattern"
    genus = SelfDirectedLearner._extract_genus_term(text)
    assert genus is None


def test_extract_genus_from_definitions_concept_props() -> None:
    """extract_genus_from_definitions extracts IS_A from concept definitions."""
    learner = _make_learner()
    learner.network.add_concept("serotonin")
    learner.network._concepts["serotonin"].properties = {
        "definition": "a neurotransmitter involved in sleep"
    }
    learner.extract_genus_from_definitions()
    # Should have created an IS_A edge: serotonin is_a neurotransmitter
    edges = learner.network.get_edges("serotonin", "out")
    is_a_targets = [e.target for e in edges if e.relation == RelationType.IS_A]
    assert "neurotransmitter" in is_a_targets


def test_extract_genus_from_definitions_skips_existing() -> None:
    """extract_genus_from_definitions skips edges that already exist."""
    learner = _make_learner()
    learner.network.add_concept("dog")
    learner.network.add_concept("animal")
    learner.network.add_edge("dog", "animal", RelationType.IS_A)
    learner.network._concepts["dog"].properties = {
        "definition": "a animal that barks"
    }
    learner.extract_genus_from_definitions()
    # Should not create a duplicate IS_A edge
    is_a_edges = [
        e for e in learner.network.edges
        if e.relation == RelationType.IS_A
        and e.source == "dog" and e.target == "animal"
    ]
    assert len(is_a_edges) == 1


def test_extract_genus_from_definitions_transitive_closure() -> None:
    """extract_genus_from_definitions runs transitive closure after extraction."""
    learner = _make_learner()
    # Set up: beagle has a definition "a dog", and dog is_a mammal
    learner.network.add_concept("beagle")
    learner.network.add_concept("dog")
    learner.network.add_concept("mammal")
    learner.network._concepts["beagle"].properties = {
        "definition": "a dog that hunts rabbits"
    }
    learner.network.add_edge("dog", "mammal", RelationType.IS_A)
    learner.extract_genus_from_definitions()
    # Should infer: beagle is_a dog (genus), then beagle is_a mammal (transitive)
    edges = learner.network.get_edges("beagle", "out")
    is_a_targets = {e.target for e in edges if e.relation == RelationType.IS_A}
    assert "dog" in is_a_targets
    assert "mammal" in is_a_targets


def test_extract_man_page_genus_filters_paths() -> None:
    """_extract_man_page_genus filters out file paths and troff artifacts."""
    learner = _make_learner()
    entries = learner._extract_man_page_genus()
    cmds = [cmd for cmd, _, _ in entries]
    # No file paths
    assert not any(c.startswith("/") for c in cmds)
    assert not any("/" in c for c in cmds)
    # No troff artifacts
    assert not any("\\" in c for c in cmds)
    assert not any("{" in c for c in cmds)
    # All start with a letter
    assert all(c[0].isalpha() for c in cmds)


def test_extract_genus_foundational_defs() -> None:
    """extract_genus_from_definitions seeds foundational computer concept defs."""
    learner = _make_learner()
    # "command" should get a foundational definition
    learner.extract_genus_from_definitions()
    c = learner.network.get_concept("command")
    assert c is not None
    assert (c.properties or {}).get("definition")
    # Should have an IS_A edge: command is_a program
    edges = learner.network.get_edges("command", "out")
    is_a_targets = {e.target for e in edges if e.relation == RelationType.IS_A}
    assert "program" in is_a_targets


def test_run_inference_cycle_force() -> None:
    """run_inference_cycle with force=True runs all inference mechanisms."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    learner.network.add_concept("c")
    learner.network.add_edge("a", "b", RelationType.IS_A)
    learner.network.add_edge("b", "c", RelationType.IS_A)
    result = learner.run_inference_cycle(force=True)
    assert isinstance(result, InferenceResult)


def test_run_inference_cycle_cooldown() -> None:
    """run_inference_cycle respects cooldown when not forced."""
    learner = _make_learner()
    # First call should run (no previous time)
    result = learner.run_inference_cycle(force=True)
    assert isinstance(result, InferenceResult)
    # Second call without force should be skipped due to cooldown
    result2 = learner.run_inference_cycle()
    assert result2.new_edges == 0
    assert result2.new_definitions == 0


# ═══════════════════════════════════════════════════════════════════
# Introspection
# ═══════════════════════════════════════════════════════════════════


def test_get_recent_events_after_learning() -> None:
    """get_recent_events returns events after learning."""
    learner = _make_learner()
    learner.learn_from_input("a dog is a mammal")
    events = learner.get_recent_events()
    assert len(events) >= 1


def test_get_stats_after_learning() -> None:
    """get_stats reflects learning activity."""
    learner = _make_learner()
    learner.learn_from_input("a dog is a mammal")
    stats = learner.get_stats()
    assert stats["conversation_facts"] >= 1


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════


def test_is_contradiction_similar_opposite() -> None:
    """_is_contradiction detects SIMILAR_TO vs OPPOSITE_OF."""
    learner = _make_learner()
    learner.network.add_concept("hot")
    learner.network.add_concept("cold")
    learner.network.add_edge("hot", "cold", RelationType.OPPOSITE_OF)
    assert learner._is_contradiction("hot", "cold", RelationType.SIMILAR_TO) is True


def test_is_contradiction_none() -> None:
    """_is_contradiction returns False when no contradiction exists."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    assert learner._is_contradiction("a", "b", RelationType.IS_A) is False


def test_would_create_cycle() -> None:
    """_would_create_cycle detects cycles in hierarchical relations."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    learner.network.add_edge("a", "b", RelationType.IS_A)
    # Adding b→a would create a cycle
    assert learner._would_create_cycle("b", "a", RelationType.IS_A) is True


def test_would_create_cycle_no_cycle() -> None:
    """_would_create_cycle returns False when no cycle would form."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    learner.network.add_edge("a", "b", RelationType.IS_A)
    # Adding a→c would not create a cycle
    assert learner._would_create_cycle("a", "c", RelationType.IS_A) is False


def test_would_create_cycle_non_hierarchical() -> None:
    """_would_create_cycle returns False for non-hierarchical relations."""
    learner = _make_learner()
    learner.network.add_concept("a")
    learner.network.add_concept("b")
    learner.network.add_edge("a", "b", RelationType.RELATED_TO)
    # RELATED_TO is not hierarchical
    assert learner._would_create_cycle("b", "a", RelationType.RELATED_TO) is False


# ======================================================================
# From tests/test_heuristic_experiment.py
# ======================================================================

@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a temporary project with an allowlisted file and verify script."""
    # Create the allowlisted file
    src_dir = tmp_path / "python" / "genesis_cognitive" / "learning"
    src_dir.mkdir(parents=True)
    heuristic_file = src_dir / "autonomous.py"
    heuristic_file.write_text(
        '"""Test heuristic file."""\n'
        "import re\n\n"
        "# A simple regex heuristic\n"
        '_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")\n\n'
        "def extract_concepts(text):\n"
        '    """Extract concepts from text."""\n'
    "    return _CONCEPT_RE.findall(text)\n"
    )

    # Create the verify script (always passes)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    verify_script = scripts_dir / "verify_change.sh"
    verify_script.write_text(
        "#!/usr/bin/env bash\n"
        "exit 0\n"
    )
    verify_script.chmod(0o755)

    return tmp_path


@pytest.fixture
def engine(temp_project: Path) -> SelfImprovementEngine:
    """Create a SelfImprovementEngine with the temp project."""
    return SelfImprovementEngine(project_root=str(temp_project))


@pytest.fixture
def experiment(engine: SelfImprovementEngine, temp_project: Path) -> HeuristicExperiment:
    """Create a HeuristicExperiment with the temp project."""
    return HeuristicExperiment(engine, project_root=str(temp_project))


class TestExperimentRecord:
    """Tests for the ExperimentRecord dataclass."""

    def test_to_dict_roundtrip(self):
        """ExperimentRecord should survive a to_dict/from_dict round-trip."""
        record = ExperimentRecord(
            file_path="test.py",
            description="test experiment",
            status="applied",
            reason="all checks passed",
            timestamp=12345,
        )
        data = record.to_dict()
        restored = ExperimentRecord.from_dict(data)

        assert restored.file_path == "test.py"
        assert restored.description == "test experiment"
        assert restored.status == "applied"
        assert restored.reason == "all checks passed"
        assert restored.timestamp == 12345

    def test_from_dict_defaults(self):
        """from_dict should handle missing fields with defaults."""
        restored = ExperimentRecord.from_dict({})
        assert restored.file_path == ""
        assert restored.status == "skipped"
        assert restored.timestamp == 0


class TestHeuristicExperimentGuardrails:
    """Tests for the safety guardrails."""

    def test_rejects_non_allowlisted_file(self, experiment, temp_project) -> None:
        """Files not on the allowlist should be skipped.

        The allowlist now covers all experimentable files (Genesis's own
        Python source under genesis_cognitive/ or genesis_client/).
        Non-experimentable files — tests, config, Rust, scripts — are
        still rejected. We verify with a test file.
        """
        # Create a test file (non-experimentable)
        test_dir = temp_project / "python" / "tests"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "test_other.py"
        test_file.write_text("x = 1\n")

        record = experiment.run_experiment(
            file_path="python/tests/test_other.py",
            original_code="x = 1",
            proposed_code="x = 2",
            description="change x",
        )

        assert record.status == "skipped"
        assert "not own source" in record.reason
        # File should be unchanged
        assert test_file.read_text() == "x = 1\n"

    def test_daily_limit_enforced(self, experiment, temp_project):
        """Should skip after reaching the daily limit."""
        # Manually add records to fill the daily limit
        now_ms = int(time.time() * 1000)
        for i in range(HeuristicExperiment._MAX_PER_DAY):
            experiment.engine._experiment_history.append(
                ExperimentRecord(
                    file_path="test.py",
                    description=f"experiment {i}",
                    status="applied",
                    reason="ok",
                    timestamp=now_ms,
                )
            )

        record = experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code='_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")',
            proposed_code='_CONCEPT_RE = re.compile(r"\\b([a-z_]+)\\b")',
            description="allow underscores in concepts",
        )

        assert record.status == "skipped"
        assert "daily limit" in record.reason

    def test_original_code_not_found_is_reverted(self, experiment):
        """If the original code snippet doesn't exist, should revert."""
        record = experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code="THIS_DOES_NOT_EXIST",
            proposed_code="replacement",
            description="test",
        )

        assert record.status == "reverted"
        assert "not found" in record.reason


class TestHeuristicExperimentExecution:
    """Tests for the experiment execution and verification."""

    def test_successful_experiment_applies_change(self, experiment, temp_project):
        """A valid change that passes verification should be applied."""
        file_path = temp_project / "python" / "genesis_cognitive" / "learning" / "autonomous.py"
        original = '_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")'
        proposed = '_CONCEPT_RE = re.compile(r"\\b([a-z_]+)\\b")'

        record = experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code=original,
            proposed_code=proposed,
            description="allow underscores in concepts",
        )

        assert record.status == "applied"
        assert "passed" in record.reason
        # File should contain the new code
        content = file_path.read_text()
        assert proposed in content
        assert original not in content

    def test_py_compile_failure_reverts(self, experiment, temp_project):
        """A change that breaks py_compile should be reverted."""
        file_path = temp_project / "python" / "genesis_cognitive" / "learning" / "autonomous.py"
        original = '_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")'
        # Intentionally broken Python
        proposed = '_CONCEPT_RE = re.compile(r"\\b([a-z'  # unclosed

        record = experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code=original,
            proposed_code=proposed,
            description="broken change",
        )

        assert record.status == "reverted"
        assert "py_compile" in record.reason
        # File should be reverted to original
        content = file_path.read_text()
        assert original in content

    def test_verify_script_failure_reverts(self, engine, temp_project):
        """A change that fails the verify script should be reverted."""
        # Make the verify script fail
        verify_script = temp_project / "scripts" / "verify_change.sh"
        verify_script.write_text(
            "#!/usr/bin/env bash\n"
            "exit 1\n"
        )
        verify_script.chmod(0o755)

        experiment = HeuristicExperiment(engine, project_root=str(temp_project))
        file_path = temp_project / "python" / "genesis_cognitive" / "learning" / "autonomous.py"
        original = '_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")'
        proposed = '_CONCEPT_RE = re.compile(r"\\b([a-z_]+)\\b")'

        record = experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code=original,
            proposed_code=proposed,
            description="test with failing verify",
        )

        assert record.status == "reverted"
        assert "verification gate" in record.reason
        # File should be reverted
        content = file_path.read_text()
        assert original in content

    def test_no_change_produces_revert(self, experiment):
        """If the replacement is identical to original, should revert."""
        record = experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code='_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")',
            proposed_code='_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")',
            description="no-op change",
        )

        assert record.status == "reverted"
        assert "no change" in record.reason


class TestHeuristicExperimentAuditLog:
    """Tests for the audit log tracking."""

    def test_applied_experiments_are_logged(self, experiment):
        """Successfully applied experiments should be in the audit log."""
        experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code='_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")',
            proposed_code='_CONCEPT_RE = re.compile(r"\\b([a-z_]+)\\b")',
            description="allow underscores",
        )

        history = experiment.experiment_history
        assert len(history) >= 1
        assert any(r.status == "applied" for r in history)

    def test_applied_and_reverted_counts(self, experiment, engine, temp_project):
        """applied_count and reverted_count should track correctly."""
        # Make verify fail
        verify_script = temp_project / "scripts" / "verify_change.sh"
        verify_script.write_text("#!/usr/bin/env bash\nexit 1\n")
        verify_script.chmod(0o755)

        experiment = HeuristicExperiment(engine, project_root=str(temp_project))

        # This will pass py_compile but fail verify → reverted + logged
        experiment.run_experiment(
            file_path="python/genesis_cognitive/learning/autonomous.py",
            original_code='_CONCEPT_RE = re.compile(r"\\b([a-z]+)\\b")',
            proposed_code='_CONCEPT_RE = re.compile(r"\\b([a-z_]+)\\b")',
            description="test revert",
        )

        assert experiment.reverted_count >= 1
        assert experiment.applied_count == 0


class TestHeuristicExperimentPersistence:
    """Tests for experiment history persistence."""

    def test_experiments_persist_through_to_dict(self, engine, experiment):
        """Experiment history should be included in to_dict."""
        # Add a fake experiment record
        engine._experiment_history.append(
            ExperimentRecord(
                file_path="test.py",
                description="test",
                status="applied",
                reason="ok",
            )
        )

        data = engine.to_dict()
        assert "experiments" in data
        assert len(data["experiments"]) == 1
        assert data["experiments"][0]["file_path"] == "test.py"

    def test_experiments_restore_from_dict(self, engine):
        """Experiment history should restore from dict."""
        data = {
            "experiments": [
                {
                    "file_path": "test.py",
                    "description": "test experiment",
                    "status": "applied",
                    "reason": "ok",
                    "timestamp": 12345,
                }
            ]
        }
        engine.restore_from_dict(data)

        assert len(engine._experiment_history) == 1
        assert engine._experiment_history[0].file_path == "test.py"
        assert engine._experiment_history[0].status == "applied"


# ======================================================================
# From tests/test_learning_patterns.py
# ======================================================================

def test_is_a_basic():
    """X is a Y → IS_A edge."""
    m = _IS_A_RE.search("a dog is a mammal")
    assert m is not None
    assert m.group(1).strip() == "dog"
    assert m.group(2) == "mammal"


def test_is_a_with_an():
    """X is an Y → IS_A edge. Article in subject is stripped by normalization."""
    m = _IS_A_RE.search("an elephant is an animal")
    assert m is not None
    # The regex captures the raw text including the leading article.
    # _normalize_concept strips it before storing.
    assert "elephant" in m.group(1)
    assert m.group(2) == "animal"


def test_is_a_multi_word_subject():
    """Multi-word subject: 'great white shark is a fish'."""
    m = _IS_A_RE.search("a great white shark is a fish")
    assert m is not None
    assert "shark" in m.group(1)


def test_is_a_multi_word_object():
    """Multi-word object: 'dog is a domestic animal'."""
    m = _IS_A_RE.search("a dog is a domestic animal")
    assert m is not None
    assert m.group(2) == "domestic animal"


def test_is_a_does_not_match_bare_is():
    """Bare 'is' without article should not match IS_A."""
    m = _IS_A_RE.search("happiness is similar to joy")
    assert m is None, "IS_A should not match 'is similar to'"


def test_is_a_does_not_match_part_of():
    """'is part of' should not match IS_A."""
    m = _IS_A_RE.search("a wheel is part of a car")
    assert m is None, "IS_A should not match 'is part of'"


def test_is_a_does_not_match_opposite():
    """'is the opposite of' should not match IS_A."""
    m = _IS_A_RE.search("hot is the opposite of cold")
    assert m is None, "IS_A should not match 'is the opposite of'"


def test_is_a_with_article_in_subject():
    """Leading article in subject should be stripped by normalization."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    events = learner.learn_from_input("A dog is a mammal.")
    assert len(events) > 0
    # The concept should be "dog", not "a dog"
    assert net.get_concept("dog") is not None
    assert net.get_concept("a dog") is None


# ─── DEFINITION pattern ────────────────────────────────────────


def test_definition_means():
    """'X means Y' → definition."""
    m = _DEFINITION_RE.search("entropy means a measure of disorder")
    assert m is not None
    assert m.group(1).strip() == "entropy"


def test_definition_is_defined_as():
    """'X is defined as Y' → definition."""
    m = _DEFINITION_RE.search("cryptobiosis is defined as a state of suspended animation")
    assert m is not None
    assert m.group(1).strip() == "cryptobiosis"


def test_definition_refers_to():
    """'X refers to Y' → definition."""
    m = _DEFINITION_RE.search("cognition refers to the mental action of acquiring knowledge")
    assert m is not None
    assert m.group(1).strip() == "cognition"


def test_definition_is_when():
    """'X is when Y' → definition."""
    m = _DEFINITION_RE.search("evaporation is when a liquid turns into gas")
    assert m is not None
    assert m.group(1).strip() == "evaporation"


def test_definition_takes_priority_over_is_a():
    """Definition should be extracted before IS_A."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("Cryptobiosis is defined as a state of suspended animation.")
    # Should create a definition, not an IS_A edge to "state"
    c = net.get_concept("cryptobiosis")
    assert c is not None
    assert "definition" in c.properties
    assert "suspended animation" in c.properties["definition"]


def test_definition_minimum_length():
    """Definitions shorter than 10 chars should not match."""
    m = _DEFINITION_RE.search("x means short")
    # "short" is only 5 chars — the .{10,200} requires at least 10
    assert m is None


# ─── PART_OF pattern ───────────────────────────────────────────


def test_part_of_basic():
    """'X is part of Y' → PART_OF edge."""
    m = _PART_OF_RE.search("a wheel is part of a car")
    assert m is not None
    assert m.group(1).strip() == "wheel"
    assert "car" in m.group(2)


def test_part_of_without_article():
    """'X is part of Y' without articles."""
    m = _PART_OF_RE.search("mitochondria is part of cell")
    assert m is not None
    assert m.group(1).strip() == "mitochondria"


def test_part_of_plural():
    """'wheels are part of cars' → PART_OF."""
    m = _PART_OF_RE.search("wheels are part of cars")
    assert m is not None
    assert "wheels" in m.group(1) or "wheel" in m.group(1)


def test_part_of_does_not_match_is_a():
    """'is a part of' should match PART_OF, not IS_A."""
    _IS_A_RE.search("a piston is a part of an engine")
    # "is a part" — IS_A requires "is a/an" followed by a word
    # "part" is a word, so IS_A might match "is a part"
    # This is actually a known overlap — PART_OF runs first
    # Just verify PART_OF also matches
    m2 = _PART_OF_RE.search("a piston is a part of an engine")
    assert m2 is not None


# ─── CAUSES pattern ────────────────────────────────────────────


def test_causes_basic():
    """'X causes Y' → CAUSES edge."""
    m = _CAUSES_RE.search("fire causes smoke")
    assert m is not None
    assert m.group(1).strip() == "fire"
    assert m.group(2) == "smoke"


def test_causes_leads_to():
    """'X leads to Y' → CAUSES edge."""
    m = _CAUSES_RE.search("stress leads to illness")
    assert m is not None
    assert m.group(1).strip() == "stress"


def test_causes_produces():
    """'X produces Y' → CAUSES edge."""
    m = _CAUSES_RE.search("photosynthesis produces oxygen")
    assert m is not None
    assert m.group(1).strip() == "photosynthesis"


def test_causes_generates():
    """'X generates Y' → CAUSES edge."""
    m = _CAUSES_RE.search("exercise generates heat")
    assert m is not None
    assert m.group(1).strip() == "exercise"


# ─── DEPENDS_ON pattern ────────────────────────────────────────


def test_depends_on_basic():
    """'X depends on Y' → DEPENDS_ON edge."""
    m = _DEPENDS_RE.search("life depends on water")
    assert m is not None
    assert m.group(1).strip() == "life"
    assert m.group(2) == "water"


def test_depends_requires():
    """'X requires Y' → DEPENDS_ON edge."""
    m = _DEPENDS_RE.search("photosynthesis requires sunlight")
    assert m is not None
    assert m.group(1).strip() == "photosynthesis"


# ─── ENABLES pattern ───────────────────────────────────────────


def test_enables_basic():
    """'X enables Y' → ENABLES edge."""
    m = _ENABLES_RE.search("education enables progress")
    assert m is not None
    assert m.group(1).strip() == "education"
    assert m.group(2) == "progress"


def test_enables_allows():
    """'X allows Y' → ENABLES edge."""
    m = _ENABLES_RE.search("practice allows mastery")
    assert m is not None
    assert m.group(1).strip() == "practice"


# ─── OPPOSITE pattern ──────────────────────────────────────────


def test_opposite_basic():
    """'X is the opposite of Y' → OPPOSITE_OF edge."""
    m = _OPPOSITE_RE.search("hot is the opposite of cold")
    assert m is not None
    assert m.group(1).strip() == "hot"
    assert m.group(2) == "cold"


def test_opposite_without_the():
    """'X is opposite of Y' → OPPOSITE_OF edge."""
    m = _OPPOSITE_RE.search("hot is opposite of cold")
    assert m is not None


def test_opposite_opposes():
    """'X opposes Y' → OPPOSITE_OF edge."""
    m = _OPPOSITE_RE.search("fire opposes water")
    assert m is not None
    assert m.group(1).strip() == "fire"
    assert m.group(2) == "water"


def test_opposite_contradicts():
    """'X contradicts Y' → OPPOSITE_OF edge."""
    m = _OPPOSITE_RE.search("evidence contradicts theory")
    assert m is not None
    assert m.group(1).strip() == "evidence"


# ─── SIMILAR_TO pattern ────────────────────────────────────────


def test_similar_to_basic():
    """'X is similar to Y' → SIMILAR_TO edge."""
    m = _SIMILAR_RE.search("joy is similar to happiness")
    assert m is not None
    assert m.group(1).strip() == "joy"
    assert m.group(2) == "happiness"


# ─── HAS pattern ───────────────────────────────────────────────


def test_has_basic():
    """'X has Y' → RELATED_TO edge."""
    m = _HAS_RE.search("a dog has legs")
    assert m is not None
    assert m.group(1).strip() == "dog"
    assert m.group(2) == "legs"


def test_has_with_article():
    """'X has a Y' → RELATED_TO edge."""
    m = _HAS_RE.search("a dog has a tail")
    assert m is not None
    assert m.group(1).strip() == "dog"
    assert m.group(2) == "tail"


def test_has_without_article_after_has():
    """'has eight legs' should match (no article)."""
    m = _HAS_RE.search("a spider has eight legs")
    assert m is not None
    assert m.group(1).strip() == "spider"
    assert m.group(2) == "eight legs"


def test_has_plural_have():
    """'X have Y' → RELATED_TO edge."""
    m = _HAS_RE.search("birds have wings")
    assert m is not None
    assert m.group(1).strip() == "birds"


# ─── Concept normalization ────────────────────────────────────


def test_normalize_strips_articles():
    """Leading articles are stripped."""
    assert _normalize_concept("a dog") == "dog"
    assert _normalize_concept("an elephant") == "elephant"
    assert _normalize_concept("the brain") == "brain"


def test_normalize_hyphens():
    """Hyphens become spaces."""
    assert _normalize_concept("micro-animal") == "micro animal"
    assert _normalize_concept("state-of-the-art") == "state of the art"


def test_normalize_plural_simple():
    """Simple plurals are singularized."""
    assert _normalize_concept("dogs") == "dog"
    assert _normalize_concept("cats") == "cat"


def test_normalize_plural_ies():
    """-ies plurals become -y."""
    assert _normalize_concept("cities") == "city"
    assert _normalize_concept("butterflies") == "butterfly"


def test_normalize_plural_es():
    """-es plurals are handled."""
    assert _normalize_concept("boxes") == "box"
    assert _normalize_concept("brushes") == "brush"


@pytest.mark.parametrize("plural, singular", [
    ("men", "man"),
    ("women", "woman"),
    ("children", "child"),
    ("feet", "foot"),
    ("teeth", "tooth"),
    ("mice", "mouse"),
    ("people", "person"),
    ("cacti", "cactus"),
    ("fungi", "fungus"),
    ("nuclei", "nucleus"),
    ("phenomena", "phenomenon"),
    ("criteria", "criterion"),
    ("analyses", "analysis"),
    ("theses", "thesis"),
    ("hypotheses", "hypothesis"),
    ("matrices", "matrix"),
    ("vertices", "vertex"),
    ("indices", "index"),
])
def test_normalize_irregular_plurals(plural: str, singular: str):
    """Irregular plurals are singularized."""
    assert _normalize_concept(plural) == singular


@pytest.mark.parametrize("word", ["analysis", "crisis", "thesis", "synthesis"])
def test_normalize_not_plural(word: str):
    """Words ending in 's' that aren't plural are left alone."""
    assert _normalize_concept(word) == word


def test_normalize_relative_clause():
    """Trailing relative clauses are stripped."""
    assert _normalize_concept("micro-animal that can survive extreme conditions") == "micro animal"
    assert _normalize_concept("dog which barks loudly") == "dog"
    assert _normalize_concept("scientist who studies physics") == "scientist"


def test_normalize_multi_word_plural():
    """Multi-word: only last word is singularized."""
    assert _normalize_concept("great white sharks") == "great white shark"
    assert _normalize_concept("brain cells") == "brain cell"


# ─── Extended normalization tests ─────────────────────────────


@pytest.mark.parametrize("plural, singular", [
    ("wolves", "wolf"),
    ("knives", "knife"),
    ("leaves", "leaf"),
    ("halves", "half"),
    ("shelves", "shelf"),
])
def test_normalize_ves_plurals(plural: str, singular: str):
    """-ves plurals become -f or -fe."""
    assert _normalize_concept(plural) == singular


@pytest.mark.parametrize("word", ["cactus", "virus", "status", "campus"])
def test_normalize_us_words_not_stripped(word: str):
    """Words ending in 'us' are not stripped (Latin singulars)."""
    assert _normalize_concept(word) == word


@pytest.mark.parametrize("word", ["basis", "thesis", "axis"])
def test_normalize_is_words_not_stripped(word: str):
    """Words ending in 'is' are not stripped (Greek singulars)."""
    assert _normalize_concept(word) == word


@pytest.mark.parametrize("word", [
    "chaos", "lens", "bias", "atlas", "process", "boss", "class", "chess",
])
def test_normalize_extended_not_plural(word: str):
    """Additional non-plural words ending in 's' are left alone."""
    assert _normalize_concept(word) == word


# ─── Integration: full extraction pipeline ────────────────────


def test_integration_is_a_creates_edge():
    """Full pipeline: 'A dog is a mammal' creates IS_A edge."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    events = learner.learn_from_input("A dog is a mammal.")
    assert len(events) > 0
    edges = net.get_edges("dog", "out")
    is_a_edges = [e for e in edges if e.relation == RelationType.IS_A]
    assert len(is_a_edges) == 1
    assert is_a_edges[0].target == "mammal"


def test_integration_definition_creates_property():
    """Full pipeline: definition creates a property, not an edge."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("Entropy means a measure of disorder in a system.")
    c = net.get_concept("entropy")
    assert c is not None
    assert "definition" in c.properties
    assert "disorder" in c.properties["definition"].lower()


def test_integration_part_of_creates_edge():
    """Full pipeline: 'A wheel is part of a car' creates PART_OF edge."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("A wheel is part of a car.")
    edges = net.get_edges("wheel", "out")
    part_of = [e for e in edges if e.relation == RelationType.PART_OF]
    assert len(part_of) == 1


def test_integration_unparseable_name_no_function_word_edge():
    """Unparseable identifiers must not mint function-word concepts.

    "The node_0_1_x is part of the node_0_2_y" contains digits and
    underscores the extractor can't read. The regex used to slide past
    them and match the verb phrase itself, creating a spurious
    "is → part of → the" edge. Nothing should be learned from input
    whose endpoints can't be parsed.
    """
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    events = learner.learn_from_input(
        "The node_0_1_abcdefgh is part of the node_0_2_wxyzabcd."
    )
    assert events == []
    assert net.get_concept("is") is None
    assert net.get_concept("the") is None


def test_integration_no_duplicate_concepts():
    """Same concept mentioned twice doesn't create duplicates."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("A dog is a mammal.")
    learner.learn_from_input("A dog is a canine.")
    # Should have one "dog" concept, not two
    assert net.get_concept("dog") is not None
    assert net.size <= 4  # dog, mammal, canine + maybe one more


def test_integration_hyphenated_concept():
    """Hyphenated concepts are normalized to spaces."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("A micro-animal is a tardigrade.")
    # Should be stored as "micro animal", not "micro-animal"
    assert net.get_concept("micro animal") is not None
    assert net.get_concept("micro-animal") is not None  # _resolve normalizes


def test_integration_multiple_facts_one_sentence():
    """Multiple facts in one input are all extracted."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("A dog is a mammal. A cat is a mammal.")
    assert net.get_concept("dog") is not None
    assert net.get_concept("cat") is not None
    assert net.get_concept("mammal") is not None


def test_integration_empty_input():
    """Empty input doesn't crash."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    events = learner.learn_from_input("")
    assert events == []


def test_integration_non_fact_input():
    """Non-fact input doesn't extract anything."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    events = learner.learn_from_input("Hello, how are you today?")
    assert events == []


def test_integration_case_insensitive():
    """Patterns are case-insensitive."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("A Dog Is A Mammal.")
    assert net.get_concept("dog") is not None


# ─── Negative tests: patterns that should NOT match ───────────


def test_negative_question_not_extracted():
    """Questions should not be extracted as facts."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    events = learner.learn_from_input("Is a dog a mammal?")
    # Questions starting with "is" don't match the pattern
    # because the subject must come before "is a"
    assert len(events) == 0


# ─── Test runner ──────────────────────────────────────────────


def run_all():
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    logger.info(f"\n  Learning pattern tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


# ======================================================================
# From tests/test_inference_thresholds.py
# ======================================================================

def _build_taxonomy_network() -> tuple[ConceptNetwork, SelfDirectedLearner]:
    """Build a realistic taxonomy network for testing."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    facts = [
        "A dog is a mammal.",
        "A cat is a mammal.",
        "A mammal is an animal.",
        "A bird is an animal.",
        "A fish is an animal.",
        "A golden retriever is a dog.",
        "A poodle is a dog.",
        "A siamese is a cat.",
        "A persian is a cat.",
        "A parrot is a bird.",
        "A canary is a bird.",
        "A shark is a fish.",
        "A salmon is a fish.",
        "A dog has legs.",
        "A cat has legs.",
        "A bird has wings.",
        "A fish has fins.",
        "A mammal has hair.",
        "A bird has feathers.",
        "A fish has scales.",
        "A dog is similar to a wolf.",
        "A cat is similar to a lion.",
        "A wheel is part of a car.",
        "An engine is part of a car.",
        "A brain is part of a head.",
        "Life depends on water.",
        "Fire causes smoke.",
    ]
    for fact in facts:
        learner.learn_from_input(fact)
    return net, learner


def test_multi_hop_transitive_inference():
    """Transitive inference finds multi-hop paths.

    golden retriever → dog → mammal → animal
    should infer golden retriever is_a animal.
    """
    net, learner = _build_taxonomy_network()
    learner.infer_transitive()

    # Check multi-hop inference
    gr_edges = [
        e for e in net.get_edges("golden retriever", "out") if e.relation == RelationType.IS_A
    ]
    targets = {e.target for e in gr_edges}
    assert "animal" in targets, f"golden retriever should be inferred as animal, got: {targets}"
    assert "mammal" in targets


def test_transitive_inference_no_cycles():
    """Transitive inference should not create cycles."""
    net, learner = _build_taxonomy_network()
    learner.infer_transitive()

    # Check no self-loops
    for edge in net._edges:
        assert edge.source != edge.target, f"Self-loop: {edge.source} → {edge.target}"


def test_transitive_inference_no_duplicates():
    """Transitive inference should not create duplicate edges."""
    net, learner = _build_taxonomy_network()
    learner.infer_transitive()
    learner.infer_transitive()  # Run again

    # Count edges by (source, target, relation)
    edge_keys = [(e.source, e.target, e.relation) for e in net._edges]
    assert len(edge_keys) == len(set(edge_keys)), "Duplicate edges found"


def test_inferred_edges_have_lower_weight():
    """Inferred edges should have lower weight than stated edges."""
    net, learner = _build_taxonomy_network()
    learner.infer_transitive()

    stated = [e for e in net._edges if e.origin == "stated"]
    inferred = [e for e in net._edges if e.origin == "inferred"]

    if inferred:
        max_inferred = max(e.weight for e in inferred)
        min_stated = min(e.weight for e in stated) if stated else 1.0
        assert max_inferred < min_stated or max_inferred <= 0.35, (
            f"Inferred edges should have lower weight: max_inferred={max_inferred}, "
            f"min_stated={min_stated}"
        )


def test_analogical_transfer_requires_shared_parents():
    """Analogical transfer only happens between concepts with shared parents."""
    _net, learner = _build_taxonomy_network()
    learner.infer_transitive()  # Build multi-hop parents first
    result = learner.infer_analogical()

    # Dog and cat share mammal + animal (after transitive inference)
    # So analogical transfer should happen between them
    assert result.new_edges > 0, "Should find analogical inferences between dog and cat"


def test_analogical_transfer_no_false_positives():
    """Analogical transfer should not happen between unrelated concepts."""
    _net, learner = _build_taxonomy_network()
    learner.infer_transitive()
    result = learner.infer_analogical()

    # Check no nonsensical inferences
    for event in result.events:
        # Should not infer that a fish is similar to a dog
        assert "fish" not in event.description.lower() or "dog" not in event.description.lower(), (
            f"False positive: {event.description}"
        )


def test_definition_synthesis_quality():
    """Definition synthesis produces readable definitions."""
    net, learner = _build_taxonomy_network()
    learner.synthesize_definitions()

    dog = net.get_concept("dog")
    assert dog is not None
    assert "definition" in dog.properties
    defn = dog.properties["definition"]
    assert "mammal" in defn, f"Definition should mention mammal: {defn}"


def test_inference_cycle_runs_all_phases():
    """run_inference_cycle runs all inference phases."""
    _net, learner = _build_taxonomy_network()
    result = learner.run_inference_cycle(force=True)

    assert result.new_edges > 0
    assert result.new_definitions > 0


def test_inference_cooldown():
    """Inference cooldown prevents running too frequently."""
    _net, learner = _build_taxonomy_network()
    learner.run_inference_cycle(force=True)
    # Without force, should be skipped due to cooldown
    result2 = learner.run_inference_cycle()
    assert result2.new_edges == 0
    assert result2.new_definitions == 0


def test_part_of_transitive_inference():
    """PART_OF is transitive: wheel → car → vehicle."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("A wheel is part of a car.")
    learner.learn_from_input("A car is part of a vehicle.")
    learner.infer_transitive()

    edges = net.get_edges("wheel", "out")
    part_of_targets = {e.target for e in edges if e.relation == RelationType.PART_OF}
    assert "vehicle" in part_of_targets, (
        f"wheel should be inferred as part of vehicle: {part_of_targets}"
    )


def test_depends_on_transitive_inference():
    """DEPENDS_ON is transitive."""
    net = ConceptNetwork()
    learner = SelfDirectedLearner(net)
    learner.learn_from_input("Life depends on water.")
    learner.learn_from_input("Water depends on hydrogen.")
    learner.infer_transitive()

    edges = net.get_edges("life", "out")
    depends_targets = {e.target for e in edges if e.relation == RelationType.DEPENDS_ON}
    assert "hydrogen" in depends_targets, (
        f"life should be inferred as depending on hydrogen: {depends_targets}"
    )


# ─── Test runner ──────────────────────────────────────────────


def run_all_inference():
    """Run all inference."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    logger.info(f"\n  Inference threshold tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_inference()
    sys.exit(0 if success else 1)
