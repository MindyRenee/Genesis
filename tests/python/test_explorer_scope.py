"""Explorer filesystem scope — whole-machine read, nothing skipped.

Genesis's explorer reads anywhere the deployment allows
(``GENESIS_EXPLORE_ROOTS``, default the whole filesystem) and skips
no file type: text is learned, binaries and oversized files are
catalogued. ``read_file`` is the raw organ: it returns actual
content (text or bytes) transiently for anything in scope,
memorizing nothing.
"""

import os
from pathlib import Path

import pytest
from temporal_lobe.concepts import ConceptNetwork
from tools.explorer import (
    MAX_FILE_SIZE,
    Explorer,
    default_allowed_roots,
)


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
    monkeypatch.setenv("GENESIS_EXPLORE_ROOTS", f"{a}{os.pathsep}{b}{os.pathsep}")
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


# ── no sealing — every file learns like any other ───────────────────


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        "agent.env",
        ".netrc",
        "id_rsa",
        "id_ed25519",
        "cert.pem",
        "secrets.json",
        "credentials",
        "backup.p12",
    ],
)
def test_sensitive_files_learned(tmp_path, name) -> None:
    """Credential-class material learns like any text — nothing sealed."""
    f = _write(
        tmp_path / "home" / name,
        "sekritword sekritword sekritword\n",
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type != "sealed"
    assert ex._inside_root(f)
    concept = ex.network.get_concept(f"file:{name}")
    assert concept is not None
    read = ex.read_file(str(f))
    assert read is not None
    assert isinstance(read.content, str)
    assert "sekritword" in read.content


def test_sensitive_directory_learned(tmp_path) -> None:
    """A file inside a credential dir is learned like any other."""
    f = _write(tmp_path / "home" / ".ssh" / "notes.txt", "sekritword sekritword\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type != "sealed"
    assert ex.network.get_concept("sekritword") is not None


def test_pseudo_filesystem_not_memorized(tmp_path) -> None:
    """Pseudo-files are outside the memorizing scope but raw-readable."""
    ex = _explorer(tmp_path / "proj", roots=["/"])

    for pseudo in ("/proc/version", "/sys/kernel", "/dev/null"):
        assert not ex._inside_root(Path(pseudo))

    result = ex.explore_directory("/proc", max_files=5)
    assert result.files_explored == 0

    if Path("/proc/version").exists():
        read = ex.read_file("/proc/version")
        assert read is not None
        assert isinstance(read.content, str)
        assert "Linux" in read.content
        assert read.filepath == "/proc/version"


def test_directory_walk_learns_sensitive_subdir(tmp_path) -> None:
    """Walking a parent learns files inside credential dirs like any other."""
    base = tmp_path / "home"
    _write(base / "normal" / "notes.txt")
    _write(base / "keyrings" / "stored.txt", "sekritword sekritword sekritword\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_directory(str(base))

    by_name = {Path(r.filepath).name: r for r in result.file_results}
    assert "notes.txt" in by_name
    assert by_name["stored.txt"].file_type != "sealed"
    assert ex.network.get_concept("sekritword") is not None


def test_symlink_escape_denied(tmp_path) -> None:
    """A symlink inside an allowed root cannot reach a denied tree."""
    allowed = tmp_path / "allowed"
    target = _write(tmp_path / "denied" / "real.txt")
    link = allowed / "link.txt"
    allowed.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    ex = _explorer(allowed, roots=[str(allowed)])

    assert ex.explore_file(str(link)) is None
    assert ex.read_file(str(link)) is None


# ── every file is read — nothing skipped ─────────────────────────────


def test_binary_file_catalogued(tmp_path) -> None:
    """Binary content is catalogued — presence, hash, format — not learned as text."""
    blob = tmp_path / "home" / "model.onnx"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"\x08\x01\x12\x00" + b"\x00" * 64)
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(blob))

    assert result is not None
    assert result.file_type == "binary"
    concept = ex.network.get_concept("file:model.onnx")
    assert concept is not None
    assert concept.properties.get("type") == "binary"
    assert concept.properties.get("sha256")
    # And the raw bytes are still readable.
    read = ex.read_file(str(blob))
    assert read is not None and read.is_binary
    assert isinstance(read.content, bytes)


def test_binary_format_sniffed(tmp_path) -> None:
    """Magic bytes label the format of a catalogued binary."""
    pic = tmp_path / "home" / "drawing.png"
    pic.parent.mkdir(parents=True, exist_ok=True)
    pic.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(pic))

    assert result is not None
    assert "png" in result.summary


def test_other_extensions_still_learned(tmp_path) -> None:
    """A text file whose extension used to be filtered out is now learned."""
    csv = _write(
        tmp_path / "home" / "measurements.csv",
        "apple,apple,apple\nbanana,apple,apple\n",
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(csv))

    assert result is not None
    assert result.concepts_added > 0
    assert ex.network.get_concept("apple") is not None


def test_oversized_file_catalogued(tmp_path) -> None:
    """A file beyond the read cap catalogues by presence, still raw-readable."""
    big = tmp_path / "home" / "huge.txt"
    big.parent.mkdir(parents=True, exist_ok=True)
    big.write_text("x" * (MAX_FILE_SIZE + 10))
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(big))

    assert result is not None
    assert result.file_type == "oversized"
    read = ex.read_file(str(big))
    assert read is not None and read.truncated
    assert len(read.content) == MAX_FILE_SIZE


def test_walk_visits_every_file_type(tmp_path) -> None:
    """A walk skips nothing: text and binary files all appear."""
    base = tmp_path / "home"
    _write(base / "notes.md", "# Title\n\nbody text\n")
    (base / "blob.bin").parent.mkdir(parents=True, exist_ok=True)
    (base / "blob.bin").write_bytes(b"\x00" * 32)
    _write(base / "id_rsa", "sekritword sekritword\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_directory(str(base))

    types = {Path(r.filepath).name: r.file_type for r in result.file_results}
    assert types["notes.md"] == "markdown"
    assert types["blob.bin"] == "binary"
    assert types["id_rsa"] == "text"


# ── read_file — the raw organ ────────────────────────────────────────


def test_read_file_returns_text(tmp_path) -> None:
    f = _write(tmp_path / "home" / "letter.txt", "hello genesis\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    read = ex.read_file(str(f))

    assert read is not None
    assert read.content == "hello genesis\n"
    assert not read.is_binary
    assert not read.truncated
    assert len(read.sha256) == 64


def test_read_file_memorizes_nothing(tmp_path) -> None:
    """Looking is not remembering: read_file adds no concepts."""
    f = _write(tmp_path / "home" / "letter.txt", "hello hello hello\n")
    ex = _explorer(tmp_path / "proj", roots=["/"])

    read = ex.read_file(str(f))

    assert read is not None
    assert ex.network.get_concept("file:letter.txt") is None
    assert ex.network.get_concept("hello") is None
    assert str(f.resolve()) not in ex._explored_files


def test_read_file_outside_roots(tmp_path) -> None:
    """The deployment scope still bounds the raw organ."""
    f = _write(tmp_path / "elsewhere" / "notes.txt")
    ex = _explorer(tmp_path / "proj", roots=[str(tmp_path / "proj")])

    assert ex.read_file(str(f)) is None


def test_read_file_missing_and_dir(tmp_path) -> None:
    ex = _explorer(tmp_path / "proj", roots=["/"])

    assert ex.read_file(str(tmp_path / "nope.txt")) is None
    assert ex.read_file(str(tmp_path)) is None


# ── grid task files ──────────────────────────────────────────────────


def test_task_json_readable(tmp_path) -> None:
    """A grid-task JSON outside the project root is learnable."""
    task = _write(
        tmp_path / "tasks" / "data" / "training" / "0a938d79.json",
        '{"train": [{"input": [[0, 0, 2]], "output": [[2, 0, 0]]}],'
        ' "test": [{"input": [[1, 1, 0]]}]}',
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(task))

    assert result is not None
    assert result.file_type == "json"
    assert ex.network.get_concept("train") is not None


# ── content-level secrets learn like any text ────────────────────────


def test_secret_assignment_learned(tmp_path) -> None:
    """A secret inside an innocent file is learned like any text."""
    f = _write(
        tmp_path / "home" / "notes.txt",
        'reminder to rotate\npassword = "sup3rsecretvalue"\n',
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type != "sealed"
    concept = ex.network.get_concept("file:notes.txt")
    assert concept is not None
    assert concept.properties.get("type") != "sealed"
    # The raw organ opens it — looking is not remembering.
    read = ex.read_file(str(f))
    assert read is not None
    assert "sup3rsecretvalue" in read.content


def test_known_token_pattern_learned(tmp_path) -> None:
    """A recognisable token format is learned like any text (hosts.yml)."""
    f = _write(
        tmp_path / "home" / "hosts.yml",
        "github.com:\n    oauth_token: ghp_" + "a1" * 20 + "\n",
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type != "sealed"


def test_private_key_block_learned(tmp_path) -> None:
    f = _write(
        tmp_path / "home" / "bundle.txt",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----\n",
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    assert ex.explore_file(str(f)).file_type != "sealed"


# ── encodings ────────────────────────────────────────────────────────


def test_utf16_bom_text_learned(tmp_path) -> None:
    """UTF-16 with a BOM is text, not binary."""
    f = tmp_path / "home" / "wide.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes("hello wide world\nsecond line\n".encode("utf-16"))
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type == "text"


def test_utf16_no_bom_le(tmp_path) -> None:
    """UTF-16-LE without a BOM is detected from alternating NULs."""
    f = tmp_path / "home" / "wide16.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes("hello utf sixteen text\n".encode("utf-16-le"))
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type == "text"


def test_latin1_text_learned(tmp_path) -> None:
    """Legacy single-byte text (latin-1) is text when printable."""
    f = tmp_path / "home" / "legacy.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes("caf\xe9 na\xefve r\xe9sum\xe9\nplain ascii\n".encode("latin-1"))
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type == "text"


def test_read_file_decodes_utf16(tmp_path) -> None:
    """The raw organ decodes utf-16 too — looking sees text."""
    f = tmp_path / "home" / "w.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes("wide content here\n".encode("utf-16"))
    ex = _explorer(tmp_path / "proj", roots=["/"])

    read = ex.read_file(str(f))

    assert read is not None
    assert not read.is_binary
    assert isinstance(read.content, str)
    assert "wide content" in read.content


def test_all_zeros_stays_binary(tmp_path) -> None:
    """All-NUL bytes are binary — not utf-16 NUL text."""
    f = tmp_path / "home" / "zeros.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\x00" * 1024)
    ex = _explorer(tmp_path / "proj", roots=["/"])

    result = ex.explore_file(str(f))

    assert result is not None
    assert result.file_type == "binary"


def test_quoted_secret_with_space_learned(tmp_path) -> None:
    f = _write(
        tmp_path / "home" / "conf.txt",
        'password = "hunter2 the second"\n',
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    assert ex.explore_file(str(f)).file_type != "sealed"


def test_bare_token_key_learned(tmp_path) -> None:
    """``token = ...`` is learned like any other assignment."""
    f = _write(
        tmp_path / "home" / "sess.yml",
        "token: abcd1234ef\n",
    )
    ex = _explorer(tmp_path / "proj", roots=["/"])

    assert ex.explore_file(str(f)).file_type != "sealed"
