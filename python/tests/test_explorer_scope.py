"""Explorer filesystem scope — whole-machine read with a credential denylist.

Genesis's explorer can read anywhere the deployment allows
(``GENESIS_EXPLORE_ROOTS``, default the whole filesystem), but
credential/key material and pseudo-filesystems are never in scope:
whatever it reads is committed to the concept network.
"""

import os
from pathlib import Path

import pytest
from frontal_lobe.tools.explorer import Explorer, default_allowed_roots
from temporal_lobe.concepts import ConceptNetwork


def _explorer(root: Path, roots: list[str] | None = None) -> Explorer:
    return Explorer(
        ConceptNetwork(),
        project_root=str(root),
        allowed_roots=roots,
    )


def _write(path: Path, text: str = "notes about cognition and memory\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ── default_allowed_roots ────────────────────────────────────────────


def test_default_roots_whole_filesystem(monkeypatch) -> None:
    monkeypatch.delenv("GENESIS_EXPLORE_ROOTS", raising=False)
    assert default_allowed_roots() == ["/"]


def test_default_roots_env_narrows(monkeypatch, tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv(
        "GENESIS_EXPLORE_ROOTS", f"{a}{os.pathsep}{b}{os.pathsep}"
    )
    assert default_allowed_roots() == [str(a), str(b)]


# ── scope ────────────────────────────────────────────────────────────


def test_outside_project_root_readable(tmp_path) -> None:
    """Whole-filesystem scope reads files outside the project root."""
    project = _write(tmp_path / "proj" / "README.md", "# Proj\n")
    outside = _write(tmp_path / "elsewhere" / "arc_notes.txt")
    ex = _explorer(project.parent, roots=["/"])

    result = ex.explore_file(str(outside))

    assert result is not None
    assert result.file_type == "text"
    assert "arc_notes.txt" in result.filepath


def test_default_scope_is_project_root(tmp_path) -> None:
    """Without allowed_roots the explorer keeps the project sandbox."""
    project = tmp_path / "proj"
    outside = _write(tmp_path / "elsewhere" / "notes.txt")
    ex = _explorer(project)

    assert ex.explore_file(str(outside)) is None


def test_narrowed_roots_deny_outside(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    denied_file = _write(tmp_path / "denied" / "notes.txt")
    ex = _explorer(tmp_path / "proj", roots=[str(allowed)])

    assert ex.explore_file(str(denied_file)) is None
    assert "Refusing" in ex.explore_path(str(denied_file))


def test_relative_path_resolves_under_project_root(tmp_path) -> None:
    project = tmp_path / "proj"
    _write(project / "docs" / "guide.md", "# Guide\n\ncognition stuff\n")
    ex = _explorer(project, roots=["/"])

    assert "guide.md" in ex.explore_path("docs/guide.md")


# ── denylist ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", [".env", ".netrc", "id_rsa", "id_ed25519", "cert.pem",
             "secrets.json", "credentials", "backup.p12"]
)
def test_sensitive_files_denied(tmp_path, name) -> None:
    """Credential material is refused even under whole-filesystem scope."""
    f = _write(tmp_path / "home" / name, "SECRET=hunter2\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    assert ex.explore_file(str(f)) is None
    assert not ex._inside_root(f)


def test_sensitive_directory_denied(tmp_path) -> None:
    """A harmless-looking file inside a credential dir is still refused."""
    f = _write(tmp_path / "home" / ".ssh" / "notes.txt")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    assert ex.explore_file(str(f)) is None


def test_pseudo_filesystem_denied(tmp_path) -> None:
    ex = _explorer(tmp_path / "proj", roots=["/"])

    for pseudo in ("/proc/version", "/sys/kernel", "/dev/null"):
        assert not ex._inside_root(Path(pseudo))

    result = ex.explore_directory("/proc", max_files=5)
    assert result.files_explored == 0


def test_directory_walk_prunes_sensitive_subdir(tmp_path) -> None:
    """Walking a parent must not descend into a credential directory."""
    base = tmp_path / "home"
    _write(base / "normal" / "notes.txt")
    _write(base / "keyrings" / "stored.txt", "password=abc\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_directory(str(base))

    paths = [r.filepath for r in result.file_results]
    assert any("notes.txt" in p for p in paths)
    assert not any("stored.txt" in p for p in paths)


def test_symlink_escape_denied(tmp_path) -> None:
    """A symlink inside an allowed root cannot reach a denied tree."""
    allowed = tmp_path / "allowed"
    target = _write(tmp_path / "denied" / "real.txt")
    link = allowed / "link.txt"
    allowed.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    ex = _explorer(allowed, roots=[str(allowed)])

    assert ex.explore_file(str(link)) is None


# ── ARC task files ───────────────────────────────────────────────────


def test_arc_task_json_readable(tmp_path) -> None:
    """An ARC-style task JSON outside the project root is learnable."""
    task = _write(
        tmp_path / "ARC" / "data" / "training" / "0a938d79.json",
        '{"train": [{"input": [[0, 0, 2]], "output": [[2, 0, 0]]}],'
        ' "test": [{"input": [[1, 1, 0]]}]}',
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(task))

    assert result is not None
    assert result.file_type == "json"
    assert ex.network.get_concept("train") is not None
