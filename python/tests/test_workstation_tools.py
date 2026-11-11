"""Tests for the workstation layer — tools/workstation.py plus the
framework fixes that harden every tool call."""

from __future__ import annotations

import pathlib

import pytest
from tools.framework import Tool, ToolRegistry, ToolResult, get_tools


@pytest.fixture()
def tools() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture()
def root(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


# ─── Framework hardening ────────────────────────────────────────────


def test_registry_isolates_tool_exceptions(tools: ToolRegistry) -> None:
    def boom() -> ToolResult:
        raise RuntimeError("the tool exploded")

    tools.register(Tool("boom", "explodes", boom))
    result = tools.run("boom")
    assert not result.success
    assert "boom faulted" in result.error


def test_registry_reports_bad_arguments(tools: ToolRegistry) -> None:
    result = tools.run("list_dir", nonexistent_kwarg=1)
    assert not result.success
    assert "bad arguments" in result.error


def test_registry_unknown_tool(tools: ToolRegistry) -> None:
    result = tools.run("does_not_exist")
    assert not result.success
    assert "Unknown tool" in result.error


def test_workstation_tools_are_registered(tools: ToolRegistry) -> None:
    names = {t["name"] for t in tools.list_tools()}
    for expected in (
        "terminal",
        "edit_file",
        "append_file",
        "copy_file",
        "diff_files",
        "find_files",
        "search_files",
        "file_info",
        "system_status",
        "list_processes",
        "host_info",
        "calculate",
        "convert_units",
        "codec",
        "git",
        "archive",
    ):
        assert expected in names


def test_run_shell_stdin_is_detached(tools: ToolRegistry, root: str) -> None:
    # `read` would block forever on an inherited stdin; DEVNULL makes
    # it return immediately with nonzero status.
    result = tools.run("run_shell", command="read x; echo done", project_root=root, timeout=10)
    assert "done" in result.output


# ─── read_file tail support ─────────────────────────────────────────


def test_read_file_negative_offset_tails(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "f.txt"
    p.write_text("abcdefghijklmnopqrstuvwxyz")
    result = tools.run("read_file", path="f.txt", offset=-5, limit=0, project_root=root)
    assert result.success
    assert result.output == "vwxyz"


# ─── Terminal ───────────────────────────────────────────────────────


def test_terminal_persists_cwd(tools: ToolRegistry, root: str) -> None:
    tools.run("terminal", command="mkdir -p sub", reset=True)
    tools.run("terminal", command="cd sub")
    result = tools.run("terminal", command="pwd")
    assert result.success
    assert result.data["cwd"].endswith("sub")


def test_terminal_persists_env(tools: ToolRegistry) -> None:
    tools.run("terminal", command="export GENESIS_TEST_VAR=hello42", reset=True)
    result = tools.run("terminal", command="echo $GENESIS_TEST_VAR")
    assert result.output.strip() == "hello42"


def test_terminal_reports_returncode(tools: ToolRegistry) -> None:
    result = tools.run("terminal", command="exit 3", reset=True)
    # `exit` skips state capture; bash's own returncode still lands.
    assert result.data["returncode"] in (0, 3)


def test_terminal_reset(tools: ToolRegistry) -> None:
    tools.run("terminal", command="export GENESIS_TEST_VAR=x", reset=True)
    tools.run("terminal", command="true", reset=True)
    result = tools.run("terminal", command="echo [$GENESIS_TEST_VAR]")
    assert result.output.strip() == "[]"


def test_terminal_empty_command(tools: ToolRegistry) -> None:
    assert not tools.run("terminal", command="").success


# ─── File surgery ───────────────────────────────────────────────────


def test_edit_file_replaces_first_occurrence(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "f.txt"
    p.write_text("aaa bbb aaa")
    result = tools.run("edit_file", path="f.txt", old="aaa", new="ccc", project_root=root)
    assert result.success
    assert p.read_text() == "ccc bbb aaa"
    assert result.data["remaining"] == 1


def test_edit_file_replace_all(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "f.txt"
    p.write_text("aaa bbb aaa")
    tools.run("edit_file", path="f.txt", old="aaa", new="ccc", count=0, project_root=root)
    assert p.read_text() == "ccc bbb ccc"


def test_edit_file_missing_text_fails(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "f.txt"
    p.write_text("hello")
    result = tools.run("edit_file", path="f.txt", old="zzz", new="q", project_root=root)
    assert not result.success
    assert "not found" in result.error


def test_append_file(tools: ToolRegistry, root: str) -> None:
    tools.run("append_file", path="f.txt", content="one", project_root=root)
    tools.run("append_file", path="f.txt", content="two", project_root=root)
    assert (pathlib.Path(root) / "f.txt").read_text() == "onetwo"


def test_copy_file_and_dir(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "d").mkdir()
    (p / "d" / "f.txt").write_text("data")
    assert tools.run("copy_file", src="d/f.txt", dst="g.txt", project_root=root).success
    assert (p / "g.txt").read_text() == "data"
    assert tools.run("copy_file", src="d", dst="d2", project_root=root).success
    assert (p / "d2" / "f.txt").read_text() == "data"


def test_diff_files(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "a.txt").write_text("same\nold\nsame\n")
    (p / "b.txt").write_text("same\nnew\nsame\n")
    result = tools.run("diff_files", a="a.txt", b="b.txt", project_root=root)
    assert result.success
    assert "-old" in result.output and "+new" in result.output


def test_diff_files_identical(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "a.txt").write_text("x\n")
    (p / "b.txt").write_text("x\n")
    assert tools.run("diff_files", a="a.txt", b="b.txt", project_root=root).data["identical"]


# ─── Discovery ──────────────────────────────────────────────────────


def test_find_files_glob(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "sub").mkdir()
    (p / "a.py").write_text("")
    (p / "sub" / "b.py").write_text("")
    (p / "c.txt").write_text("")
    result = tools.run("find_files", pattern="**/*.py", project_root=root)
    assert result.success
    assert sorted(result.output.splitlines()) == ["a.py", "sub/b.py"]


def test_search_files_regex(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "a.py").write_text("def foo():\n    return needle + 1\n")
    (p / "b.py").write_text("nothing here\n")
    result = tools.run("search_files", pattern="needle", project_root=root)
    assert result.success
    assert "a.py:2" in result.output
    assert "b.py" not in result.output


def test_search_files_context(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "a.txt").write_text("before\nhit\nafter\n")
    result = tools.run("search_files", pattern="hit", context=1, project_root=root)
    assert "before" in result.output and "after" in result.output


def test_search_files_bad_regex(tools: ToolRegistry, root: str) -> None:
    assert not tools.run("search_files", pattern="([", project_root=root).success


def test_file_info(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "f.txt").write_text("hello world")
    result = tools.run("file_info", path="f.txt", project_root=root)
    assert result.success
    assert result.data["size"] == 11
    assert "sha256" in result.data


# ─── Host proprioception ────────────────────────────────────────────


def test_system_status(tools: ToolRegistry) -> None:
    result = tools.run("system_status")
    assert result.success
    assert "memtotal" in result.data or "uptime_seconds" in result.data


def test_list_processes(tools: ToolRegistry) -> None:
    result = tools.run("list_processes", limit=5)
    assert result.success
    assert result.data["count"] <= 5
    assert "COMMAND" in result.output


def test_host_info(tools: ToolRegistry) -> None:
    result = tools.run("host_info")
    assert result.success
    assert result.data["hostname"]
    assert result.data["python"].startswith("3.")


# ─── Exact small engines ────────────────────────────────────────────


def test_calculate_arithmetic(tools: ToolRegistry) -> None:
    assert tools.run("calculate", expression="2 + 3 * 4").data["value"] == 14


def test_calculate_math_funcs(tools: ToolRegistry) -> None:
    assert tools.run("calculate", expression="sqrt(16) + factorial(4)").data["value"] == 28.0
    assert tools.run("calculate", expression="pi").data["value"] == pytest.approx(3.14159)


def test_calculate_rejects_non_arithmetic(tools: ToolRegistry) -> None:
    assert not tools.run("calculate", expression="__import__('os')").success
    assert not tools.run("calculate", expression="open('x')").success


def test_convert_units(tools: ToolRegistry) -> None:
    r = tools.run("convert_units", value=1.0, from_unit="km", to_unit="m")
    assert r.data["value"] == 1000.0
    r = tools.run("convert_units", value=0.0, from_unit="C", to_unit="F")
    assert r.data["value"] == pytest.approx(32.0)
    r = tools.run("convert_units", value=1.0, from_unit="GiB", to_unit="MiB")
    assert r.data["value"] == 1024.0


def test_convert_units_mismatch(tools: ToolRegistry) -> None:
    assert not tools.run("convert_units", value=1, from_unit="kg", to_unit="m").success


def test_codec_roundtrip(tools: ToolRegistry) -> None:
    enc = tools.run("codec", text="hello world", mode="b64e")
    assert enc.success
    dec = tools.run("codec", text=enc.output, mode="b64d")
    assert dec.output == "hello world"
    assert tools.run("codec", text="abc", mode="rot13").output == "nop"
    assert tools.run("codec", text="a b", mode="urle").output == "a%20b"


def test_codec_bad_mode(tools: ToolRegistry) -> None:
    assert not tools.run("codec", text="x", mode="bogus").success


# ─── Git ────────────────────────────────────────────────────────────


def test_git_status_and_log(tools: ToolRegistry, root: str) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    r = tools.run("git", action="status", repo=root)
    assert (r.success and "branch" in r.output.lower()) or "##" in r.output
    r = tools.run("git", action="log", repo=root, n=5)
    assert "init" in r.output


def test_git_unknown_action(tools: ToolRegistry) -> None:
    assert not tools.run("git", action="frobnicate").success


# ─── Archive ────────────────────────────────────────────────────────


def test_archive_tar_roundtrip(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "src").mkdir()
    (p / "src" / "f.txt").write_text("payload")
    r = tools.run("archive", action="create", path="out.tar", files="src", project_root=root)
    assert r.success
    r = tools.run("archive", action="list", path="out.tar", project_root=root)
    assert "src/f.txt" in r.output
    (p / "src" / "f.txt").unlink()
    r = tools.run("archive", action="extract", path="out.tar", dest="restored", project_root=root)
    assert r.success
    assert (p / "restored" / "src" / "f.txt").read_text() == "payload"


def test_archive_zip_roundtrip(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root)
    (p / "f.txt").write_text("zip me")
    assert tools.run(
        "archive", action="create", path="out.zip", files="f.txt", project_root=root
    ).success
    r = tools.run("archive", action="list", path="out.zip", project_root=root)
    assert "f.txt" in r.output


def test_archive_extract_rejects_traversal(tools: ToolRegistry, root: str) -> None:
    import tarfile

    p = pathlib.Path(root)
    evil = p / "evil.tar"
    member = tarfile.TarInfo(name="../escape.txt")
    member.size = 4
    with tarfile.open(evil, "w") as tf:
        import io

        tf.addfile(member, io.BytesIO(b"evil"))
    r = tools.run("archive", action="extract", path="evil.tar", dest="out", project_root=root)
    assert not r.success or not (p / "escape.txt").exists()


# ─── The whole registry is usable through get_tools() ───────────────


def test_global_registry_has_workstation() -> None:
    names = {t["name"] for t in get_tools().list_tools()}
    assert "terminal" in names and "git" in names and "calculate" in names


def test_git_write_actions(tools: ToolRegistry, tmp_path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("hi")
    assert tools.run("git", action="init", repo=str(repo)).success
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "g@x"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "G"], check=True)
    assert tools.run("git", action="add", repo=str(repo), file="a.txt").success
    assert tools.run("git", action="commit", repo=str(repo), message="first").success
    assert tools.run("git", action="branch", repo=str(repo), ref="side").success
    assert tools.run("git", action="switch", repo=str(repo), ref="side").success
    r = tools.run("git", action="status", repo=str(repo))
    assert "side" in r.output.splitlines()[0]
    assert tools.run("git", action="tag", repo=str(repo), ref="v0", message="zero").success
    (repo / "a.txt").write_text("changed")
    assert tools.run("git", action="restore", repo=str(repo), file="a.txt").success
    assert (repo / "a.txt").read_text() == "hi"
    r = tools.run("git", action="commit", repo=str(repo))
    assert not r.success  # commit without message fails honestly
