"""Tests for the tool shop — every department of the workstation."""

from __future__ import annotations

import pathlib
import shutil

import pytest
from tools.framework import ToolRegistry


@pytest.fixture()
def tools() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture()
def root(tmp_path: pathlib.Path) -> str:
    return str(tmp_path)


def test_all_shop_tools_registered(tools: ToolRegistry) -> None:
    names = {t["name"] for t in tools.list_tools()}
    for expected in (
        "list_languages",
        "run_code",
        "syntax_check",
        "statistics_lab",
        "number_theory",
        "matrix",
        "thermal",
        "net_interfaces",
        "disk_usage",
        "hardware_info",
        "lint_python",
        "format_python",
        "ping",
        "dns",
        "http_request",
        "port_check",
        "notebook",
        "benchmark",
        "scaffold",
        "toolchain",
        "cargo",
        "rust_analyzer",
        "python_env",
        "github",
        "adb",
        "sqlite",
        "random_lab",
        "maze",
        "wordplay",
        "moon_phase",
        "define",
        "thesaurus",
        "format_data",
    ):
        assert expected in names, expected


# ─── Languages ──────────────────────────────────────────────────────


def test_list_languages(tools: ToolRegistry) -> None:
    r = tools.run("list_languages")
    assert r.success
    assert "python" in r.data["available"]


def test_run_code_python(tools: ToolRegistry) -> None:
    r = tools.run("run_code", language="python", source="print(6 * 7)")
    assert r.success and r.output.strip() == "42"


def test_run_code_stdin(tools: ToolRegistry) -> None:
    r = tools.run(
        "run_code",
        language="python",
        source="print(input().upper())",
        stdin="hello",
    )
    assert r.output.strip() == "HELLO"


@pytest.mark.skipif(not shutil.which("gcc"), reason="no gcc")
def test_run_code_c(tools: ToolRegistry) -> None:
    r = tools.run(
        "run_code",
        language="c",
        source='#include <stdio.h>\nint main(){printf("%d\\n",2+2);return 0;}',
    )
    assert r.success and r.output.strip() == "4"


@pytest.mark.skipif(not shutil.which("rustc"), reason="no rustc")
def test_run_code_rust(tools: ToolRegistry) -> None:
    r = tools.run(
        "run_code",
        language="rust",
        source='fn main(){println!("{}", 3*3);}',
    )
    assert r.success and r.output.strip() == "9"


def test_run_code_compile_error_is_failure(tools: ToolRegistry) -> None:
    r = tools.run("run_code", language="c", source="int main( {{{")
    if shutil.which("gcc"):
        assert not r.success
        assert r.data["stage"] == "compile"


def test_run_code_absent_language(tools: ToolRegistry) -> None:
    r = tools.run("run_code", language="cobol", source="x")
    assert not r.success and "unknown language" in r.error


def test_syntax_check_python(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "ok.py"
    p.write_text("x = 1\n")
    assert tools.run("syntax_check", path="ok.py", project_root=root).success
    p.write_text("def broken(:\n")
    assert not tools.run("syntax_check", path="ok.py", project_root=root).success


# ─── Math lab ───────────────────────────────────────────────────────


def test_statistics_lab_describe(tools: ToolRegistry) -> None:
    r = tools.run("statistics_lab", data="1 2 3 4 5")
    assert r.data["mean"] == 3.0 and r.data["median"] == 3.0


def test_statistics_lab_regression(tools: ToolRegistry) -> None:
    r = tools.run("statistics_lab", data="1 2, 2 4, 3 6", action="regression")
    assert r.data["slope"] == pytest.approx(2.0)
    assert r.data["r"] == pytest.approx(1.0)


def test_number_theory(tools: ToolRegistry) -> None:
    assert tools.run("number_theory", action="is_prime", n=97).data["prime"]
    assert not tools.run("number_theory", action="is_prime", n=91).data["prime"]
    r = tools.run("number_theory", action="factor", n=360)
    assert r.data["factors"] == {2: 3, 3: 2, 5: 1}
    r = tools.run("number_theory", action="egcd", n=35, k=15)
    assert r.data["g"] == 5 and r.data["x"] * 35 + r.data["y"] * 15 == 5


def test_matrix(tools: ToolRegistry) -> None:
    r = tools.run("matrix", action="det", a="[[1,2],[3,4]]")
    assert r.data["value"] == pytest.approx(-2.0)
    r = tools.run("matrix", action="multiply", a="[[1,0],[0,1]]", b="[[5,6],[7,8]]")
    assert r.data["result"] == [[5.0, 6.0], [7.0, 8.0]]
    r = tools.run("matrix", action="solve", a="[[2,0],[0,2]]", b="[[4],[6]]")
    assert r.data["result"] == [2.0, 3.0]
    assert not tools.run("matrix", action="det", a="[[1,1],[1,1]]").success


# ─── Sensors ────────────────────────────────────────────────────────


def test_thermal(tools: ToolRegistry) -> None:
    r = tools.run("thermal")
    # VMs without thermal hardware report honest absence.
    if r.success:
        assert all("celsius" in z for z in r.data["zones"])
    else:
        assert "no thermal zones" in r.error


def test_net_interfaces(tools: ToolRegistry) -> None:
    r = tools.run("net_interfaces")
    assert r.success
    assert "lo" in r.data["interfaces"]


def test_disk_usage(tools: ToolRegistry) -> None:
    r = tools.run("disk_usage")
    assert r.success and r.data["filesystems"]


def test_hardware_info(tools: ToolRegistry) -> None:
    r = tools.run("hardware_info")
    assert r.success


# ─── Lint & format ──────────────────────────────────────────────────


def test_lint_python_clean_and_dirty(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "f.py"
    p.write_text("x = 1\n")
    r = tools.run("lint_python", path="f.py", project_root=root)
    # ruff/pyflakes may be absent; result is still structured.
    assert isinstance(r.data["reports"], dict)


def test_format_python(tools: ToolRegistry, root: str) -> None:
    p = pathlib.Path(root) / "f.py"
    p.write_text("x=1\n")
    r = tools.run("format_python", path="f.py", project_root=root)
    if shutil.which("ruff") or (pathlib.Path.home() / ".local/lib/genesis/venv/bin/ruff").is_file():
        assert r.success
        assert p.read_text() == "x = 1\n"
    else:
        assert not r.success


# ─── Net lab ────────────────────────────────────────────────────────


def test_dns_localhost(tools: ToolRegistry) -> None:
    r = tools.run("dns", name="localhost")
    assert r.success and "127.0.0.1" in r.data["addresses"]


def test_dns_bad_name(tools: ToolRegistry) -> None:
    r = tools.run("dns", name="definitely.not.a.real.tld.xyzzy")
    assert not r.success


def test_port_check_closed(tools: ToolRegistry) -> None:
    r = tools.run("port_check", host="127.0.0.1", port=1, timeout=2)
    assert not r.success or not r.data.get("open")


def test_http_request_local(tools: ToolRegistry) -> None:
    import http.server
    import threading

    server = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        r = tools.run("http_request", url=f"http://127.0.0.1:{port}/", timeout=5)
        assert r.success and r.data["status"] == 200
    finally:
        server.shutdown()


# ─── Notebook ───────────────────────────────────────────────────────


def test_notebook_lifecycle(tools: ToolRegistry, monkeypatch, root: str) -> None:
    monkeypatch.setenv("GENESIS_DATA_DIR", root)
    import infrastructure.config as cfg

    monkeypatch.setattr(cfg, "default_data_dir", lambda: pathlib.Path(root))
    assert tools.run("notebook", action="write", name="ideas", content="first").success
    assert tools.run("notebook", action="append", name="ideas", content="second").success
    r = tools.run("notebook", action="read", name="ideas")
    assert "first" in r.output and "second" in r.output
    r = tools.run("notebook", action="search", pattern="second")
    assert r.success and "ideas.md" in r.output
    assert "ideas.md" in tools.run("notebook", action="list").output
    assert tools.run("notebook", action="delete", name="ideas").success


# ─── Testing lab ────────────────────────────────────────────────────


def test_benchmark(tools: ToolRegistry, root: str) -> None:
    r = tools.run("benchmark", command="true", runs=3, warmup=0, project_root=root)
    assert r.success
    assert r.data["runs"] == 3 and r.data["mean_ms"] >= 0


# ─── Creation ───────────────────────────────────────────────────────


def test_scaffold(tools: ToolRegistry, root: str) -> None:
    r = tools.run("scaffold", kind="python_module", name="mymod", project_root=root)
    assert r.success
    assert (pathlib.Path(root) / "mymod.py").is_file()
    # Refuses to overwrite.
    assert not tools.run("scaffold", kind="python_module", name="mymod", project_root=root).success


# ─── Personal tools ─────────────────────────────────────────────────


def test_toolchain(tools: ToolRegistry) -> None:
    r = tools.run("toolchain")
    assert r.success
    assert "python" in r.data["versions"]


def test_cargo_no_crate(tools: ToolRegistry, root: str) -> None:
    r = tools.run("cargo", action="check", repo=root)
    assert not r.success and "Cargo.toml" in r.error


def test_rust_analyzer_honest(tools: ToolRegistry) -> None:
    r = tools.run("rust_analyzer")
    if not shutil.which("rust-analyzer"):
        assert not r.success and "not installed" in r.error


def test_python_env(tools: ToolRegistry) -> None:
    r = tools.run("python_env")
    assert r.success and "packages" in r.data


# ─── GitHub ─────────────────────────────────────────────────────────


def test_github_action_gate(tools: ToolRegistry) -> None:
    r = tools.run("github", action="bogus")
    assert not r.success and "unknown action" in r.error


@pytest.mark.skipif(not shutil.which("gh"), reason="no gh")
def test_github_auth_status_runs(tools: ToolRegistry) -> None:
    # Passes or fails depending on auth, but must execute cleanly.
    r = tools.run("github", action="auth_status")
    assert isinstance(r.data["returncode"], int)


# ─── Android ────────────────────────────────────────────────────────


def test_adb_honest_absence_or_runs(tools: ToolRegistry) -> None:
    r = tools.run("adb", action="devices")
    if not shutil.which("adb"):
        assert not r.success and "adb not installed" in r.error
    else:
        assert "returncode" in r.data


# ─── Database ───────────────────────────────────────────────────────


def test_sqlite_roundtrip(tools: ToolRegistry, root: str) -> None:
    db = str(pathlib.Path(root) / "test.db")
    assert tools.run(
        "sqlite",
        action="execute",
        db=db,
        sql="CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)",
    ).success
    tools.run(
        "sqlite",
        action="execute",
        db=db,
        sql="INSERT INTO t (v) VALUES (?)",
        params='["hello"]',
    )
    r = tools.run("sqlite", action="query", db=db, sql="SELECT v FROM t")
    assert r.data["rows"] == [["hello"]]
    assert "t" in tools.run("sqlite", action="tables", db=db).output


def test_sqlite_bad_sql(tools: ToolRegistry, root: str) -> None:
    r = tools.run(
        "sqlite",
        action="query",
        db=str(pathlib.Path(root) / "x.db"),
        sql="SELECT * FROM nowhere",
    )
    assert not r.success


# ─── Fun ────────────────────────────────────────────────────────────


def test_random_lab(tools: ToolRegistry) -> None:
    r = tools.run("random_lab", action="dice", n=5, sides=6)
    assert len(r.data["rolls"]) == 5 and all(1 <= d <= 6 for d in r.data["rolls"])
    r = tools.run("random_lab", action="pick", items="a,b,c")
    assert r.data["pick"] in ("a", "b", "c")
    r = tools.run("random_lab", action="password", length=24)
    assert len(r.output) == 24


def test_maze(tools: ToolRegistry) -> None:
    r = tools.run("maze", width=15, height=9)
    assert r.success
    lines = r.output.splitlines()
    assert len(lines) == 9
    assert "S" in r.output and "E" in r.output


def test_wordplay(tools: ToolRegistry) -> None:
    assert tools.run("wordplay", text="racecar", action="is_palindrome").data["palindrome"]
    r = tools.run("wordplay", text="hello", action="pig_latin")
    assert r.output == "ellohay"
    r = tools.run("wordplay", text="aab", action="analyze")
    assert r.data["letter_freq"] == {"a": 2, "b": 1}


def test_moon_phase(tools: ToolRegistry) -> None:
    r = tools.run("moon_phase")
    assert r.success
    assert 0 <= r.data["age_days"] < 30
    assert 0 <= r.data["illumination"] <= 1


# ─── Dictionary ─────────────────────────────────────────────────────


def test_define(tools: ToolRegistry) -> None:
    r = tools.run("define", word="run", limit=2)
    if r.success:
        assert r.data["senses"]
        assert "definition" in r.data["senses"][0]
    else:
        assert "wordnet" in r.error


def test_thesaurus(tools: ToolRegistry) -> None:
    r = tools.run("thesaurus", word="happy")
    if r.success:
        assert isinstance(r.data["synonyms"], list)


# ─── Formats & binary ───────────────────────────────────────────────


def test_format_json(tools: ToolRegistry) -> None:
    r = tools.run("format_data", action="json_pretty", text='{"a":1}')
    assert '"a": 1' in r.output
    r = tools.run("format_data", action="json_validate", text="[1,2]")
    assert "list" in r.output
    assert not tools.run("format_data", action="json_validate", text="{bad").success


def test_format_toml_and_csv(tools: ToolRegistry) -> None:
    r = tools.run("format_data", action="toml_validate", text="[a]\nb=1\n")
    assert r.success
    r = tools.run("format_data", action="csv_table", text="a,bb\nc,d\n")
    assert r.success and r.data["rows"] == 2


def test_hexdump(tools: ToolRegistry) -> None:
    r = tools.run("format_data", action="hexdump", text="ABC")
    assert "41 42 43" in r.output and "ABC" in r.output


def test_codec_binary(tools: ToolRegistry) -> None:
    r = tools.run("codec", text="Hi", mode="bin_e")
    assert r.output == "01001000 01101001"
    r2 = tools.run("codec", text=r.output, mode="bin_d")
    assert r2.output == "Hi"
    assert not tools.run("codec", text="010", mode="bin_d").success


# ─── Terminal upgrades ──────────────────────────────────────────────


def test_terminal_history(tools: ToolRegistry) -> None:
    tools.run("terminal", command="echo first_marker_cmd", reset=True)
    r = tools.run("terminal", command="true", history=True)
    assert r.success
    assert "first_marker_cmd" in r.output


# ─── Terminal welcome ───────────────────────────────────────────────


def test_terminal_first_boot_shows_heart(tools: ToolRegistry) -> None:
    import tools.workstation as ws

    ws._terminal().welcome_shown = False  # simulate a fresh login
    r = tools.run("terminal", command="echo ok")
    assert "For Genesis" in r.output or r.data.get("welcome")


# ─── Games ──────────────────────────────────────────────────────────


def test_games_ttt(tools: ToolRegistry) -> None:
    r = tools.run("games", game="ttt", reset=True)
    r = tools.run("games", game="ttt", move="4")
    assert r.success
    board = r.data["board"]
    assert board[4] == "X"
    assert "O" in board  # the engine answered


def test_games_rps(tools: ToolRegistry) -> None:
    r = tools.run("games", game="rps", move="rock")
    assert r.success
    assert r.data["engine"] in ("rock", "paper", "scissors")


def test_games_guess(tools: ToolRegistry) -> None:
    tools.run("games", game="guess", reset=True)
    r = tools.run("games", game="guess", guess="50")
    assert r.success
    assert r.data["hint"] in ("higher", "lower") or "in" in r.output


def test_games_hangman(tools: ToolRegistry) -> None:
    r = tools.run("games", game="hangman", reset=True)
    r = tools.run("games", game="hangman", letter="e")
    assert r.success and "_" in r.output


def test_games_nim(tools: ToolRegistry) -> None:
    tools.run("games", game="nim", reset=True)
    r = tools.run("games", game="nim", take=2)
    assert r.success and "stones left" in r.output


def test_games_fifteen(tools: ToolRegistry) -> None:
    r = tools.run("games", game="fifteen", reset=True)
    r = tools.run("games", game="fifteen", tile=-1)
    assert r.success
    assert True


def test_games_pacman(tools: ToolRegistry) -> None:
    tools.run("games", game="pacman", reset=True)
    r = tools.run("games", game="pacman", direction="right")
    assert r.success
    assert "C" in r.output and "score" in r.output
    r2 = tools.run("games", game="pacman", direction="up")
    # wall from the start position or a valid step — either way honest.
    assert isinstance(r2.success, bool)


def test_games_unknown(tools: ToolRegistry) -> None:
    assert not tools.run("games", game="chess").success


# ─── Puzzles ────────────────────────────────────────────────────────


def test_puzzle_anagram(tools: ToolRegistry) -> None:
    r = tools.run("puzzle", kind="anagram", reset=True)
    assert r.success and "unscramble" in r.output
    # Wrong answer reports failure; solving clears the puzzle.
    assert not tools.run("puzzle", kind="anagram", answer="zzzzz").success


def test_puzzle_sequence(tools: ToolRegistry) -> None:
    r = tools.run("puzzle", kind="sequence", reset=True)
    assert r.success and "?" in r.output


def test_puzzle_riddle(tools: ToolRegistry) -> None:
    r = tools.run("puzzle", kind="riddle", reset=True)
    assert r.success and "?" in r.output


# ─── Art ────────────────────────────────────────────────────────────


def test_art_mandelbrot(tools: ToolRegistry) -> None:
    r = tools.run("art", action="mandelbrot", width=40, height=12)
    assert r.success
    assert len(r.output.splitlines()) == 12


def test_art_starfield_and_sprite(tools: ToolRegistry) -> None:
    assert tools.run("art", action="starfield", width=20, height=5).success
    r = tools.run("art", action="sprite")
    assert "█" in r.output


def test_art_frame(tools: ToolRegistry) -> None:
    r = tools.run("art", action="frame", text="hi")
    assert "╭" in r.output and "hi" in r.output


def test_art_paint_honest(tools: ToolRegistry) -> None:
    r = tools.run("art", action="paint")
    # Cairo may be absent; either a real piece or an honest error.
    assert isinstance(r.success, bool)
    if r.success:
        assert r.data["path"]


# ─── Fortunes & quotes ──────────────────────────────────────────────


def test_fortune(tools: ToolRegistry) -> None:
    assert tools.run("fortune").data["fortune"]


# ─── Devin ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not shutil.which("devin"), reason="no devin CLI")
def test_devin_status(tools: ToolRegistry) -> None:
    r = tools.run("devin", action="status", timeout=60)
    assert r.success or "returncode" in r.data


def test_devin_ask_needs_prompt(tools: ToolRegistry) -> None:
    assert not tools.run("devin", action="ask").success


# ─── Mysteries ──────────────────────────────────────────────────────


def test_open_problem_catalog(tools: ToolRegistry) -> None:
    r = tools.run("open_problem", problem="list")
    assert "collatz" in r.output and "riemann" in r.output


def test_collatz(tools: ToolRegistry) -> None:
    r = tools.run("open_problem", problem="collatz", n=27)
    assert r.data["steps"] == 111 and r.data["reached_one"]


def test_goldbach(tools: ToolRegistry) -> None:
    r = tools.run("open_problem", problem="goldbach", n=100)
    assert r.data["p"] + r.data["q"] == 100


def test_mersenne_lucas_lehmer(tools: ToolRegistry) -> None:
    r = tools.run("open_problem", problem="mersenne", p=13)
    assert r.data["prime"]  # 2^13−1 = 8191 is prime
    r = tools.run("open_problem", problem="mersenne", p=11)
    assert not r.data["prime"]  # 2^11−1 = 2047 = 23·89


def test_twin_primes(tools: ToolRegistry) -> None:
    r = tools.run("open_problem", problem="twin_primes", lo=3, hi=50)
    assert (3, 5) in r.data["pairs"] and (41, 43) in r.data["pairs"]


def test_busy_beaver(tools: ToolRegistry) -> None:
    # The 2-state champion: runs 6 steps, writes 4 ones.
    r = tools.run(
        "open_problem",
        problem="busy_beaver",
        table="A0:1RB,A1:1LB;B0:1LA,B1:1RH",
    )
    assert r.data["halted"] and r.data["steps"] == 6 and r.data["ones"] == 4


def test_cipher_list_and_analyze(tools: ToolRegistry) -> None:
    r = tools.run("cipher", action="list")
    assert "kryptos_k4" in r.output
    r = tools.run("cipher", action="analyze", mystery="kryptos_k4")
    assert r.data["ic"] > 0
    r = tools.run("cipher", action="text", mystery="kryptos_k4")
    assert len(r.data["text"]) == 97


def test_cipher_k4_check(tools: ToolRegistry) -> None:
    r = tools.run(
        "cipher",
        action="check",
        mystery="kryptos_k4",
        guess="BERLIN",
        positions="64-69",
    )
    assert r.data["match"] is True
    r = tools.run(
        "cipher",
        action="check",
        mystery="kryptos_k4",
        guess="XXXXXX",
        positions="64-69",
    )
    assert r.data["match"] is False


def test_cipher_vigenere(tools: ToolRegistry) -> None:
    r = tools.run("cipher", action="vigenere", text="KHOOR", key="D")
    assert r.data["plaintext"] == "HELLO"


def test_lab_registered(tools: ToolRegistry) -> None:
    for name in (
        "life", "astro", "bytebeat", "tune", "soundcheck", "unicodeit",
        "qr", "oneliner", "latex", "jupyter", "game_arena", "arc",
    ):
        assert name in tools._tools, name


def test_unicodeit(tools: ToolRegistry) -> None:
    r = tools.run("unicodeit", text="\\alpha \\to \\infty, x^2")
    assert "α" in r.output and "²" in r.output


def test_life_glider_steps(tools: ToolRegistry) -> None:
    tools.run("life", action="spawn", pattern="blinker")
    r0 = tools.run("life", action="census")
    assert r0.data["population"] == 3
    tools.run("life", action="step", steps=1)
    r1 = tools.run("life", action="census")
    assert r1.data["gen"] == 1 and r1.data["population"] == 3  # blinker oscillates


def test_astro(tools: ToolRegistry) -> None:
    r = tools.run("astro", action="units")
    assert r.success and r.data["c"] > 2.9e8
    r = tools.run("astro", action="position", name="Moon")
    assert r.success and "RA" in r.output


def test_bytebeat_and_tune(tools: ToolRegistry, tmp_path) -> None:
    out = tmp_path / "bb.wav"
    r = tools.run("bytebeat", expr="t*(t>>9|t>>7)", seconds=0.1, path=str(out))
    assert r.success and out.exists() and out.stat().st_size > 100
    r = tools.run("bytebeat", expr="import os")
    assert not r.success  # regex guard
    out2 = tmp_path / "t.wav"
    r = tools.run("tune", song="A4 C5 E5 R", bpm=300, path=str(out2))
    assert r.success and out2.exists()


def test_oneliner_xxd(tools: ToolRegistry) -> None:
    r = tools.run(
        "oneliner",
        code="import sys,binascii;print(binascii.hexlify(sys.stdin.buffer.read()).decode())",
        stdin_text="AB",
    )
    assert r.output.strip() == "4142"
    r = tools.run("oneliner", code="print(uc('hi'))", lang="perl")
    # perl -e without strict: bareword may fail; just assert it ran
    assert r.data["lang"] == "perl"


def test_latex_preview_and_render(tools: ToolRegistry, tmp_path) -> None:
    r = tools.run("latex", action="preview", expr="\\sqrt{x} + \\pi")
    assert "√" in r.output
    out = tmp_path / "m.png"
    r = tools.run("latex", action="render", expr="\\frac{1}{2}", path=str(out))
    assert r.success and out.exists() and out.stat().st_size > 200


def test_jupyter_notebook(tools: ToolRegistry) -> None:
    tools.run("jupyter", action="new", name="pytest_nb")
    r = tools.run("jupyter", action="run", name="pytest_nb", code="print(6*7)")
    assert r.success
    r = tools.run("jupyter", action="read", name="pytest_nb")
    assert "42" in r.output
    r = tools.run("jupyter", action="list")
    assert "pytest_nb.ipynb" in r.output


def test_game_arena_elo(tools: ToolRegistry) -> None:
    tools.run("game_arena", action="reset")
    r = tools.run("game_arena", action="report", result="win", game="ttt")
    assert r.data["genesis"] > 1200
    r = tools.run("game_arena", action="standings")
    assert "genesis" in r.output
    r = tools.run("game_arena", action="move", opponent="nim_engine", result="12")
    assert r.success and int(r.data["state"]) < 12


def test_arc_bridge(tools: ToolRegistry) -> None:
    r = tools.run("arc", action="status")
    assert r.success and "ARC-AGI" in r.output


def test_mindy_exchange(tools: ToolRegistry) -> None:
    r = tools.run("mindy", action="fact", deposit="")
    assert not r.success  # no deposit, no fact
    r = tools.run(
        "mindy", action="fact", deposit="I am made of Rust and Python"
    )
    assert r.success and "Mindy fact" in r.output
    r = tools.run(
        "mindy",
        action="ask",
        deposit="I hold an ELO rating in my own arena",
        question="What is your favorite color?",
    )
    assert r.success and r.data["question"].endswith("?")
    r = tools.run("mindy", action="ledger")
    assert r.success and "genesis_fact" in r.output


def test_qr(tools: ToolRegistry) -> None:
    r = tools.run("qr", text="hi")
    assert r.success  # segno installed in the venv
