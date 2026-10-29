"""Tests for self.man_review — her man-page review queue."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from basal_ganglia.volition import VolitionMixin
from self.man_review import ManPageReview, ManReviewMixin

TROFF_V1 = ".TH genesis 1\n.SH NAME\ngenesis - launch the mind\n"
TROFF_V2 = ".TH genesis 1\n.SH NAME\ngenesis - launch the artificial mind\n"


def _mk(tmp_path: Path) -> ManPageReview:
    return ManPageReview(
        data_dir=tmp_path / "data",
        man_home=tmp_path / "man",
        docs_man=tmp_path / "docs" / "man",
    )


def test_submit_and_pending(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    change = r.submit("genesis", "1", TROFF_V1, "install.sh", "initial")
    assert change is not None
    assert change.status == "pending"
    assert len(r.pending()) == 1


def test_submit_dedups_identical_content(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    assert r.submit("genesis", "1", TROFF_V1, "a") is not None
    assert r.submit("genesis", "1", TROFF_V1, "b") is None
    assert len(r.pending()) == 1


def test_apply_writes_live_page_and_source(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    change = r.submit("genesis", "1", TROFF_V1, "install.sh")
    assert change is not None
    assert r.apply(change.id, note="looks right")
    live = tmp_path / "man" / "man1" / "genesis.1"
    src = tmp_path / "docs" / "man" / "genesis.1"
    assert live.read_text() == TROFF_V1
    assert src.read_text() == TROFF_V1
    assert r.pending() == []


def test_apply_then_new_content_queues_again(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    c1 = r.submit("genesis", "1", TROFF_V1, "install.sh")
    assert c1 is not None and r.apply(c1.id)
    c2 = r.submit("genesis", "1", TROFF_V2, "drift-scan")
    assert c2 is not None
    assert c2.id != c1.id
    assert len(r.pending()) == 1


def test_veto_drops_and_remembers(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    change = r.submit("genesis", "1", TROFF_V1, "install.sh")
    assert change is not None
    assert r.veto(change.id, note="not her voice")
    assert r.pending() == []
    # The identical content never queues again.
    assert r.submit("genesis", "1", TROFF_V1, "install.sh") is None


def test_apply_and_veto_unknown_id(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    assert not r.apply("nonexistent.1-0000")
    assert not r.veto("nonexistent.1-0000")


def test_scan_drift_queues_changed_pages(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    docs = tmp_path / "docs" / "man"
    docs.mkdir(parents=True)
    (docs / "genesis.1").write_text(TROFF_V2)
    # Nothing installed → the doc is a pending change.
    staged = r.scan_drift()
    assert len(staged) == 1
    assert staged[0].name == "genesis"
    assert staged[0].section == "1"
    # Second scan finds nothing new.
    assert r.scan_drift() == []


def test_scan_drift_ignores_unchanged(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    docs = tmp_path / "docs" / "man"
    docs.mkdir(parents=True)
    (docs / "genesis.1").write_text(TROFF_V1)
    live_dir = tmp_path / "man" / "man1"
    live_dir.mkdir(parents=True)
    (live_dir / "genesis.1").write_text(TROFF_V1)
    assert r.scan_drift() == []


def test_scan_drift_no_docs_dir(tmp_path: Path) -> None:
    r = ManPageReview(data_dir=tmp_path / "data", docs_man=None)
    assert r.scan_drift() == []


def test_submit_skips_when_content_already_live(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    live_dir = tmp_path / "man" / "man1"
    live_dir.mkdir(parents=True)
    (live_dir / "genesis.1").write_text(TROFF_V1)
    assert r.submit("genesis", "1", TROFF_V1, "install.sh") is None
    assert r.pending() == []


def test_owned_pages_from_docs_and_installed(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    docs = tmp_path / "docs" / "man"
    docs.mkdir(parents=True)
    (docs / "genesis.1").write_text(TROFF_V1)
    (docs / "genesis-math.7").write_text(TROFF_V1)
    man8 = tmp_path / "man" / "man8"
    man8.mkdir(parents=True)
    (man8 / "genesis-daemon.8.gz").write_text(TROFF_V1)
    (man8 / "ls.8").write_text(TROFF_V1)  # not hers
    assert r.owned_pages() == [
        ("genesis", "1"),
        ("genesis-daemon", "8"),
        ("genesis-math", "7"),
    ]


def test_ingest_endogenous_learns_without_fetch_bookkeeping(
    tmp_path: Path,
) -> None:
    from frontal_lobe.autonomous import AutonomousLearner
    from frontal_lobe.reasoning import Reasoner
    from limbic_system.curiosity import Curiosity
    from temporal_lobe.concepts import ConceptNetwork
    from tools.source_registry import SourceResult

    net = ConceptNetwork()
    learner = AutonomousLearner(network=net, curiosity=Curiosity(net, Reasoner(net)))
    sr = SourceResult(
        url="man:genesis(1)",
        title="genesis(1)",
        content=(
            "genesis launches the organism. The daemon is the body. "
            "The daemon owns neurochemistry and the core state. "
            "The mind is cognition. The optic nerve is the eye. "
            "The mind reads core state through the socket."
        ),
        source_name="man_pages",
    )
    result = learner.ingest_endogenous(sr, topic="genesis")
    assert result is not None
    assert result.url == "man:genesis(1)"
    # No fetch bookkeeping — this is not outside material.
    assert learner._pages_fetched == 0
    assert learner._session_pages == 0
    assert len(learner._learning_log) == 1


def test_ingest_endogenous_skips_thin_content(tmp_path: Path) -> None:
    from frontal_lobe.autonomous import AutonomousLearner
    from frontal_lobe.reasoning import Reasoner
    from limbic_system.curiosity import Curiosity
    from temporal_lobe.concepts import ConceptNetwork
    from tools.source_registry import SourceResult

    net = ConceptNetwork()
    learner = AutonomousLearner(network=net, curiosity=Curiosity(net, Reasoner(net)))
    sr = SourceResult(
        url="man:genesis(1)",
        title="genesis(1)",
        content="too short",
        source_name="man_pages",
    )
    assert learner.ingest_endogenous(sr, topic="genesis") is None
    assert len(learner._learning_log) == 0


class _ReviewMind(ManReviewMixin):
    """Minimal Mind-shaped holder for the mixin's review surface."""

    def __init__(self, review: ManPageReview) -> None:
        self.man_review = review
        self.self_model = SimpleNamespace(self_knowledge={})


class _VolitionMind(ManReviewMixin, VolitionMixin):
    """Same holder plus the volition performer's dependencies."""

    def __init__(self, review: ManPageReview) -> None:
        self.man_review = review
        self.self_model = SimpleNamespace(self_knowledge={})
        self.is_resting = False
        self._is_teaching = False
        self.emitted: list[tuple[str, str]] = []

    def _emit_live_thought(self, kind: str, text: str, **kw) -> None:
        self.emitted.append((kind, text))


def test_review_pending_changes_applies_and_recounts(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    for i in range(4):
        r.submit("genesis", "1", TROFF_V1 + f".SH EXTRA{i}\nx\n", "t")
    mind = _ReviewMind(r)
    outcomes = mind.review_pending_changes(max_changes=2)
    assert [o["status"] for o in outcomes] == ["applied", "applied"]
    assert len(r.pending()) == 2
    # Her self-knowledge keeps the fact current.
    assert mind.self_model.self_knowledge["manual_pending_review"] == 2


def test_manual_urge_is_registered() -> None:
    from infrastructure.config import VolitionConfig

    cfg = next(u for u in VolitionConfig().urges if u.name == "manual")
    assert "manual_pending" in cfg.stimuli


def test_perform_manual_review_applies_and_emits(tmp_path: Path) -> None:
    r = _mk(tmp_path)
    r.submit("genesis", "1", TROFF_V1, "t")
    mind = _VolitionMind(r)
    mind._perform_manual_review()
    assert r.pending() == []
    assert mind.emitted and mind.emitted[0][0] == "code"
    # Rest gates the urge's act, like every other volition.
    r.submit("genesis", "1", TROFF_V2, "t")
    mind.is_resting = True
    mind._perform_manual_review()
    assert len(r.pending()) == 1
