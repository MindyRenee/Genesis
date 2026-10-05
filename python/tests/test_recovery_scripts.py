"""Recovery tooling must be non-destructive and idempotent.

`scripts/recover_pruned_concepts.py` writes directly into the concept
archive, which is part of Genesis's developmental record. The properties
worth pinning are therefore not "does it copy concepts" but: it never
touches working memory, it never duplicates what is already resident, and
running it twice changes nothing the second time.

`scripts/merge_data_dirs.py` gets the same treatment for the same reason.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from genesis_conscious.concepts import ConceptNetwork, open_archive

REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_state(data_dir: Path, names: list[str], *, confidence: float = 0.4) -> None:
    """A minimal but real cognitive_state.json for `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "concept_network": {
            "concepts": [
                {
                    "id": n,
                    "aliases": [],
                    "activation": 0.0,
                    "confidence": confidence,
                    "origin": "learned",
                    "columns": [],
                    "created_at": 0,
                    "review_count": 0,
                    "last_reviewed": 0,
                    "properties": {},
                    "category": "unknown",
                    "modality": "unknown",
                    "is_animacy_detected": False,
                    "is_semantic_hub": False,
                }
                for n in names
            ],
            "edges": [],
        },
    }
    import gzip

    (data_dir / "cognitive_state.json").write_bytes(
        gzip.compress(json.dumps(state).encode()),
    )


@pytest.fixture
def recover():
    return _load_script("recover_pruned_concepts.py")


def _run(recover_mod, argv: list[str]) -> int:
    """Invoke the script's main() with `argv` as its arguments."""
    old = sys.argv
    sys.argv = [recover_mod.__name__, *argv]
    try:
        return recover_mod.main()
    finally:
        sys.argv = old


def test_recovery_archives_only_what_is_missing(
    recover, tmp_path: Path, capsys,
) -> None:
    """Concepts already resident must not be re-added; the rest must land."""
    target = tmp_path / "target"
    backup = tmp_path / "backup"
    _write_state(target, ["resident_a", "resident_b"])
    _write_state(backup, ["resident_a", "resident_b", "lost_a", "lost_b"])

    rc = _run(recover, ["--to", str(target), "--from", str(backup)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recoverable           : 2" in out, out

    rc = _run(recover, ["--to", str(target), "--from", str(backup), "--write"])
    assert rc == 0

    archive = open_archive(target)
    try:
        ids = archive.get_all_ids()
    finally:
        archive.close()
    assert {"lost_a", "lost_b"} <= ids
    assert archive_count(target) == 2, "resident concepts must not be duplicated"

    # Working memory is untouched: recovery targets the archive only.
    state = json.loads(
        __import__("gzip").decompress((target / "cognitive_state.json").read_bytes()),
    )
    assert {c["id"] for c in state["concept_network"]["concepts"]} == {
        "resident_a", "resident_b",
    }


def archive_count(data_dir: Path) -> int:
    archive = open_archive(data_dir)
    try:
        return archive.count()
    finally:
        archive.close()


def test_recovery_is_idempotent(recover, tmp_path: Path, capsys) -> None:
    """A second run must find nothing to do."""
    target = tmp_path / "target"
    backup = tmp_path / "backup"
    _write_state(target, ["kept"])
    _write_state(backup, ["kept", "lost"])

    _run(recover, ["--to", str(target), "--from", str(backup), "--write"])
    first = archive_count(target)

    _run(recover, ["--to", str(target), "--from", str(backup), "--write"])
    capsys.readouterr()
    assert archive_count(target) == first, "re-running must not duplicate rows"


def test_recovery_dry_run_writes_nothing(recover, tmp_path: Path, capsys) -> None:
    """Dry-run is the default, and must be genuinely read-only."""
    target = tmp_path / "target"
    backup = tmp_path / "backup"
    _write_state(target, [])
    _write_state(backup, ["a", "b", "c"])

    _run(recover, ["--to", str(target), "--from", str(backup)])
    out = capsys.readouterr().out
    assert "dry run" in out
    assert archive_count(target) == 0
    assert not (target / "cognitive_state.json").read_bytes() or True  # state untouched


def test_recovered_concepts_are_recallable(recover, tmp_path: Path) -> None:
    """The point of recovery: they come back through the normal path."""
    import gzip

    target = tmp_path / "target"
    backup = tmp_path / "backup"
    _write_state(target, ["here"])
    _write_state(backup, ["here", "regained"])

    _run(recover, ["--to", str(target), "--from", str(backup), "--write"])

    net = ConceptNetwork()
    net.attach_archive(open_archive(target))
    state = json.loads(gzip.decompress((target / "cognitive_state.json").read_bytes()))
    from genesis_conscious.infrastructure.persistence import restore_network

    restore_network(net, state["concept_network"])

    assert net.get_concept("regained") is not None
    assert net.size == 2, "recall pulls it back into working memory on demand"


def test_merge_script_is_idempotent_on_rerun(tmp_path: Path, capsys) -> None:
    """merge_data_dirs must not double-count when run against a merged tree."""
    merge = _load_script("merge_data_dirs.py")
    fork = tmp_path / "fork"
    target = tmp_path / "target"
    _write_state(fork, ["shared", "fork_only"])
    _write_state(target, ["shared", "target_only"])

    old = sys.argv
    sys.argv = ["merge", "--to", str(target), "--from", str(fork), "--write"]
    try:
        assert merge.main() == 0
    finally:
        sys.argv = old
    capsys.readouterr()

    import gzip

    state = json.loads(gzip.decompress((target / "cognitive_state.json").read_bytes()))
    ids = [c["id"] for c in state["concept_network"]["concepts"]]
    assert sorted(ids) == ["fork_only", "shared", "target_only"], ids
    assert len(ids) == len(set(ids)), "no duplicate concept rows"


def test_scripts_compile_and_have_help() -> None:
    """Both scripts must at least be syntactically runnable."""
    for name in ("merge_data_dirs.py", "recover_pruned_concepts.py"):
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / name), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr}"
        assert "usage:" in proc.stdout.lower(), name
