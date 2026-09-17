"""Tests for the project creator, composer, and file operation tools."""

import tempfile
from pathlib import Path

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.tools.framework import (
    ToolRegistry,
    delete_file,
    list_dir,
    make_dir,
    move_file,
    run_shell,
    write_file,
)
from genesis_cognitive.tools.project_composer import (
    compose_main_module,
    compose_test_module,
    gather_knowledge,
)
from genesis_cognitive.tools.project_creator import (
    MAX_PROJECT_SIZE_BYTES,
    _sanitize_name,
    archive_project,
    clear_project_notes,
    create_project,
    leave_project_note,
    list_projects,
    manage_project_lifecycle,
    projects_with_notes,
    read_project_notes,
    restore_project,
)

# ─── Tool sandboxing ─────────────────────────────────────────────────


def test_write_file_creates_file() -> None:
    """write_file creates a file with the given content."""
    with tempfile.TemporaryDirectory() as root:
        result = write_file("test.txt", "hello world", project_root=root)
        assert result.success
        assert (Path(root) / "test.txt").read_text() == "hello world"


def test_write_file_creates_parent_dirs() -> None:
    """write_file creates parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as root:
        result = write_file("sub/dir/test.txt", "content", project_root=root)
        assert result.success
        assert (Path(root) / "sub" / "dir" / "test.txt").read_text() == "content"


def test_write_file_rejects_escape() -> None:
    """write_file rejects paths that escape the project root."""
    with tempfile.TemporaryDirectory() as root:
        result = write_file("../../escape.txt", "content", project_root=root)
        assert not result.success
        assert "outside project root" in result.error


def test_make_dir_creates_directory() -> None:
    """make_dir creates a directory with parents."""
    with tempfile.TemporaryDirectory() as root:
        result = make_dir("my_project/src", project_root=root)
        assert result.success
        assert (Path(root) / "my_project" / "src").is_dir()


def test_list_dir_lists_contents() -> None:
    """list_dir returns directory entries with d/f prefixes."""
    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "file.txt").write_text("x")
        (Path(root) / "subdir").mkdir()
        result = list_dir(".", project_root=root)
        assert result.success
        assert "f file.txt" in result.output
        assert "d subdir" in result.output


def test_delete_file_removes_file() -> None:
    """delete_file removes a file."""
    with tempfile.TemporaryDirectory() as root:
        p = Path(root) / "test.txt"
        p.write_text("x")
        result = delete_file("test.txt", project_root=root)
        assert result.success
        assert not p.exists()


def test_delete_file_refuses_directories() -> None:
    """delete_file refuses to delete directories."""
    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "subdir").mkdir()
        result = delete_file("subdir", project_root=root)
        assert not result.success
        assert "directory" in result.error.lower()


def test_move_file_moves() -> None:
    """move_file moves a file to a new location."""
    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "src.txt").write_text("content")
        result = move_file("src.txt", "dst.txt", project_root=root)
        assert result.success
        assert not (Path(root) / "src.txt").exists()
        assert (Path(root) / "dst.txt").read_text() == "content"


def test_tool_registry_registers_new_tools() -> None:
    """The tool registry includes the new file operation tools."""
    registry = ToolRegistry()
    names = {t["name"] for t in registry.list_tools()}
    assert "write_file" in names
    assert "make_dir" in names
    assert "list_dir" in names
    assert "delete_file" in names
    assert "move_file" in names


# ─── Name sanitization ────────────────────────────────────────────────


def test_sanitize_name_lowercases() -> None:
    """_sanitize_name lowercases the input."""
    assert _sanitize_name("MyProject") == "myproject"


def test_sanitize_name_replaces_spaces() -> None:
    """_sanitize_name replaces spaces with underscores."""
    assert _sanitize_name("my project") == "my_project"


def test_sanitize_name_replaces_hyphens() -> None:
    """_sanitize_name replaces hyphens with underscores."""
    assert _sanitize_name("my-project") == "my_project"


def test_sanitize_name_strips_special_chars() -> None:
    """_sanitize_name removes special characters."""
    assert _sanitize_name("hello!@#world") == "helloworld"


def test_sanitize_name_prefixes_digit_start() -> None:
    """_sanitize_name prefixes names starting with a digit."""
    assert _sanitize_name("123project") == "p_123project"


def test_sanitize_name_defaults_to_project() -> None:
    """_sanitize_name defaults to 'project' for empty input."""
    assert _sanitize_name("") == "project"
    assert _sanitize_name("!@#") == "project"


# ─── Composer: knowledge gathering & code composition ───────────────


def _seeded_network() -> ConceptNetwork:
    """Build a small concept network with real typed edges for tests."""
    from genesis_cognitive.concepts import RelationType
    net = ConceptNetwork()
    net.add_concept(
        "memory", confidence=0.9,
        properties={"definition": "storing and retrieving information"},
    )
    net.add_concept(
        "learning", confidence=0.9,
        properties={"definition": "acquiring skill or knowledge"},
    )
    net.add_concept("stress", confidence=0.6, properties={"definition": "strain or pressure"})
    net.add_edge("memory", "learning", RelationType.ENABLES, 1.0)
    net.add_edge("stress", "memory", RelationType.HARMS, 0.6)
    return net


def test_gather_knowledge_returns_domain() -> None:
    """gather_knowledge collects the topic, its neighbors, and edges."""
    net = _seeded_network()
    k = gather_knowledge("memory", net)
    assert k is not None
    assert k.topic == "memory"
    assert "memory" in [e[0] for e in k.entries]
    assert "learning" in [e[0] for e in k.entries]
    # The ENABLES edge should be in the relations
    rels = [(s, r) for s, r, t in k.relations]
    assert ("memory", "enables") in rels
    assert ("stress", "harms") in rels


def test_gather_knowledge_unknown_topic_returns_none() -> None:
    """gather_knowledge returns None for a topic not in the network."""
    net = _seeded_network()
    assert gather_knowledge("nonexistent_concept_xyz", net) is None


def test_compose_main_module_compiles() -> None:
    """The composed main.py must be syntactically valid Python."""
    import py_compile
    import tempfile
    net = _seeded_network()
    k = gather_knowledge("memory", net)
    assert k is not None
    src = compose_main_module("memory", "a knowledge base about memory", k)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        py_compile.compile(path, doraise=True)
    finally:
        Path(path).unlink(missing_ok=True)


def test_compose_main_module_has_query_api() -> None:
    """The composed module contains the query API functions."""
    net = _seeded_network()
    k = gather_knowledge("memory", net)
    assert k is not None
    src = compose_main_module("memory", "a knowledge base about memory", k)
    assert "def find(" in src
    assert "def definition_of(" in src
    assert "def all_names(" in src
    assert "def neighbors(" in src
    assert "def related_to(" in src
    # Adaptive: the ENABLES relation gets its own function
    assert "def enables(" in src
    # The HARMS relation gets its own function too
    assert "def harms(" in src


def test_compose_test_module_compiles() -> None:
    """The composed test module must be syntactically valid Python."""
    import py_compile
    import tempfile
    net = _seeded_network()
    k = gather_knowledge("memory", net)
    assert k is not None
    src = compose_test_module("memory", k)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        py_compile.compile(path, doraise=True)
    finally:
        Path(path).unlink(missing_ok=True)


def test_compose_main_module_adaptive_structure() -> None:
    """Different domains produce different relation functions."""
    from genesis_cognitive.concepts import RelationType
    net = ConceptNetwork()
    net.add_concept("fire", confidence=0.8, properties={"definition": "combustion"})
    net.add_concept("smoke", confidence=0.7, properties={"definition": "visible particles"})
    net.add_edge("fire", "smoke", RelationType.CAUSES, 0.9)
    k = gather_knowledge("fire", net)
    assert k is not None
    src = compose_main_module("fire", "about fire", k)
    assert "def causes(" in src
    # No enables edge in this domain, so no enables() function
    assert "def enables(" not in src


# ─── Project creation ─────────────────────────────────────────────────


def test_create_project_scaffolds_structure() -> None:
    """create_project creates a complete Python project layout."""
    with tempfile.TemporaryDirectory() as data_dir:
        result = create_project(
            description="a simple calculator",
            data_dir=data_dir,
        )
        assert result.error == ""
        assert result.name == "a_simple_calculator"
        # pyproject, readme, init, main, __main__, test init, conftest, test
        assert result.files_created == 8
        project_path = Path(data_dir) / "projects" / result.name
        assert project_path.exists()
        assert (project_path / "pyproject.toml").exists()
        assert (project_path / "README.md").exists()
        assert (project_path / "src" / result.name / "__init__.py").exists()
        assert (project_path / "src" / result.name / "main.py").exists()
        assert (project_path / "src" / result.name / "__main__.py").exists()
        assert (project_path / "tests" / f"test_{result.name}.py").exists()


def test_create_project_compiles() -> None:
    """The generated main.py should compile successfully."""
    with tempfile.TemporaryDirectory() as data_dir:
        result = create_project(
            description="hello world",
            data_dir=data_dir,
        )
        assert result.compiled, f"Compile failed: {result.error}"


def test_create_project_tests_pass() -> None:
    """The generated tests should pass."""
    with tempfile.TemporaryDirectory() as data_dir:
        result = create_project(
            description="hello world",
            data_dir=data_dir,
        )
        assert result.tests_passed, (
            f"Tests failed (compiled={result.compiled}, error={result.error})"
        )


def test_list_projects_empty() -> None:
    """list_projects returns empty when no projects exist."""
    with tempfile.TemporaryDirectory() as data_dir:
        assert list_projects(data_dir) == []


def test_list_projects_after_creation() -> None:
    """list_projects returns projects after creation."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="test project one", data_dir=data_dir)
        create_project(description="test project two", data_dir=data_dir)
        projects = list_projects(data_dir)
        assert len(projects) == 2
        names = {p["name"] for p in projects}
        assert "test_project_one" in names
        assert "test_project_two" in names


def test_create_project_name_collision() -> None:
    """Creating the same project twice returns 'already exists' — no duplicates.

    The old behavior appended _2, _3, _4 … producing modes, modes_2,
    modes_3, modes_4, modes_5, modes_6, modes_7, modes_8 — eight copies
    of the same project. That's a loop, not creative output. She should
    pick a different topic instead of making duplicates.
    """
    with tempfile.TemporaryDirectory() as data_dir:
        r1 = create_project(description="hello world", data_dir=data_dir)
        assert r1.name == "hello_world"
        assert r1.error is None or r1.error == ""

        r2 = create_project(description="hello world", data_dir=data_dir)
        assert r2.name == "hello_world"
        assert r2.error == "already exists"
        assert r1.path == r2.path
        # Only one project should exist — no duplicate
        projects = list_projects(data_dir)
        assert len(projects) == 1


# ─── Storage management ─────────────────────────────────────────────


def test_list_projects_includes_size_and_status() -> None:
    """list_projects returns size_bytes and status for each project."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="alpha project", data_dir=data_dir)
        projects = list_projects(data_dir)
        assert len(projects) == 1
        p = projects[0]
        assert p["status"] == "active"
        assert p["size_bytes"] > 0
        assert "files" in p


def test_archive_project_compresses_and_removes() -> None:
    """archive_project creates a .tar.zst and removes the expanded dir."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="to archive", data_dir=data_dir)
        # Should be active before archiving
        assert list_projects(data_dir)[0]["status"] == "active"
        assert archive_project("to_archive", data_dir)
        # Now it should be archived
        projects = list_projects(data_dir)
        assert len(projects) == 1
        assert projects[0]["status"] == "archived"
        assert projects[0]["name"] == "to_archive"
        # The expanded directory should be gone
        assert not (Path(data_dir) / "projects" / "to_archive").is_dir()


def test_archive_nonexistent_project_returns_false() -> None:
    """archive_project returns False for a project that doesn't exist."""
    with tempfile.TemporaryDirectory() as data_dir:
        assert not archive_project("nope", data_dir)


def test_restore_project_brings_it_back() -> None:
    """restore_project extracts the archive back into a directory."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="round trip", data_dir=data_dir)
        assert archive_project("round_trip", data_dir)
        # Now restore it
        assert restore_project("round_trip", data_dir)
        projects = list_projects(data_dir)
        # Should have both an active and an archived entry (archive kept)
        active = [p for p in projects if p["status"] == "active"]
        archived = [p for p in projects if p["status"] == "archived"]
        assert len(active) == 1
        assert active[0]["name"] == "round_trip"
        assert len(archived) == 1
        # The restored project should still have its files
        proj_path = Path(data_dir) / "projects" / "round_trip"
        assert (proj_path / "pyproject.toml").exists()
        assert (proj_path / "src" / "round_trip" / "main.py").exists()


def test_restore_does_not_clobber_existing() -> None:
    """restore_project won't overwrite an existing expanded project."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="exists", data_dir=data_dir)
        assert archive_project("exists", data_dir)
        # Restore once
        assert restore_project("exists", data_dir)
        # Restore again should fail (already exists)
        assert not restore_project("exists", data_dir)


def test_manage_project_lifecycle_archives_oversized() -> None:
    """manage_project_lifecycle archives projects exceeding the size cap."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="big project", data_dir=data_dir)
        # Use a tiny cap to force archiving
        archived = manage_project_lifecycle(data_dir, max_size_bytes=1)
        assert "big_project" in archived
        projects = list_projects(data_dir)
        assert any(p["status"] == "archived" for p in projects)


def test_manage_project_lifecycle_archives_oldest_over_limit() -> None:
    """manage_project_lifecycle archives oldest projects past max_active."""
    import time
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="first", data_dir=data_dir)
        time.sleep(0.05)
        create_project(description="second", data_dir=data_dir)
        time.sleep(0.05)
        create_project(description="third", data_dir=data_dir)
        # max_active=1 → archive the two oldest
        archived = manage_project_lifecycle(data_dir, max_active=1)
        active = [p for p in list_projects(data_dir) if p["status"] == "active"]
        assert len(active) == 1
        assert len(archived) == 2


def test_max_project_size_is_reasonable() -> None:
    """The default size cap is a sensible bound for a Python project."""
    assert MAX_PROJECT_SIZE_BYTES == 50 * 1024 * 1024


# ─── Shell tool ───────────────────────────────────────────────────────


def test_run_shell_echo() -> None:
    """run_shell executes a simple command and returns output."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("echo hello", project_root=root)
        assert result.success
        assert "hello" in result.output


def test_run_shell_writes_file() -> None:
    """run_shell can write files in the project root."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("echo content > test.txt", project_root=root)
        assert result.success
        assert (Path(root) / "test.txt").read_text().strip() == "content"


def test_run_shell_cwd_is_project_root() -> None:
    """run_shell runs with cwd set to project_root."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("pwd", project_root=root)
        assert result.success
        assert root in result.output or Path(root).name in result.output


def test_run_shell_blocks_sudo() -> None:
    """run_shell blocks sudo commands."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("sudo echo hi", project_root=root)
        assert not result.success
        assert "blocked" in result.error.lower()


def test_run_shell_blocks_rm_rf_root() -> None:
    """run_shell blocks rm -rf /."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("rm -rf /", project_root=root)
        assert not result.success
        assert "blocked" in result.error.lower()


def test_run_shell_blocks_network_tools() -> None:
    """run_shell blocks curl and wget."""
    with tempfile.TemporaryDirectory() as root:
        for cmd in ("curl http://example.com", "wget http://example.com"):
            result = run_shell(cmd, project_root=root)
            assert not result.success, f"Should block: {cmd}"


def test_run_shell_returns_exit_code() -> None:
    """run_shell returns the exit code in data."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("exit 42", project_root=root)
        assert not result.success
        assert result.data["returncode"] == 42


def test_run_shell_empty_command() -> None:
    """run_shell rejects empty commands."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("", project_root=root)
        assert not result.success
        assert "empty" in result.error.lower()


def test_run_shell_timeout() -> None:
    """run_shell times out long-running commands."""
    with tempfile.TemporaryDirectory() as root:
        result = run_shell("sleep 10", project_root=root, timeout=1)
        assert not result.success
        assert "timed out" in result.error.lower()


def test_tool_registry_includes_run_shell() -> None:
    """The tool registry includes run_shell."""
    registry = ToolRegistry()
    names = {t["name"] for t in registry.list_tools()}
    assert "run_shell" in names


# ─── Project notes (mentor feedback) ──────────────────────────────


def test_leave_project_note_creates_file() -> None:
    """leave_project_note writes NOTES.md in the project directory."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="test topic", data_dir=data_dir)
        ok = leave_project_note(data_dir, "test_topic", "Good work on this.")
        assert ok
        notes = read_project_notes(data_dir, "test_topic")
        assert notes is not None
        assert "Good work on this." in notes


def test_leave_project_note_appends() -> None:
    """Multiple notes accumulate (append, not overwrite)."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="test topic", data_dir=data_dir)
        leave_project_note(data_dir, "test_topic", "First note.")
        leave_project_note(data_dir, "test_topic", "Second note.")
        notes = read_project_notes(data_dir, "test_topic")
        assert notes is not None
        assert "First note." in notes
        assert "Second note." in notes


def test_leave_project_note_nonexistent_project() -> None:
    """leave_project_note returns False for a nonexistent project."""
    with tempfile.TemporaryDirectory() as data_dir:
        ok = leave_project_note(data_dir, "nope", "note")
        assert not ok


def test_read_project_notes_none_when_absent() -> None:
    """read_project_notes returns None when there are no notes."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="test topic", data_dir=data_dir)
        notes = read_project_notes(data_dir, "test_topic")
        assert notes is None


def test_projects_with_notes_lists_only_noted() -> None:
    """projects_with_notes returns only projects that have NOTES.md."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="alpha", data_dir=data_dir)
        create_project(description="beta", data_dir=data_dir)
        leave_project_note(data_dir, "alpha", "note on alpha")
        noted = projects_with_notes(data_dir)
        assert "alpha" in noted
        assert "beta" not in noted


def test_clear_project_notes_removes_file() -> None:
    """clear_project_notes removes the NOTES.md file."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="test topic", data_dir=data_dir)
        leave_project_note(data_dir, "test_topic", "a note")
        assert read_project_notes(data_dir, "test_topic") is not None
        ok = clear_project_notes(data_dir, "test_topic")
        assert ok
        assert read_project_notes(data_dir, "test_topic") is None


def test_clear_project_notes_no_file() -> None:
    """clear_project_notes returns False when there are no notes."""
    with tempfile.TemporaryDirectory() as data_dir:
        create_project(description="test topic", data_dir=data_dir)
        ok = clear_project_notes(data_dir, "test_topic")
        assert not ok
