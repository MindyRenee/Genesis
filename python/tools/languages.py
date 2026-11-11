"""Languages — compilers and interpreters on her bench.

`list_languages` probes which toolchains the host actually has;
`run_code` compiles (when needed) and executes a snippet through the
real toolchain in a temp dir; `syntax_check` does compile-only
validation without running anything. Absent languages report their
absence honestly — never a fake success.

Toolchain docs: https://doc.rust-lang.org/rustc/ ·
https://gcc.gnu.org/onlinedocs/ · https://dev.java/ ·
https://docs.python.org/3/ · https://www.typescriptlang.org/docs/
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank

_LANGS: dict[str, dict[str, Any]] = {
    "python": {"probe": "python3", "kind": "interpreted", "ext": ".py"},
    "rust": {"probe": "rustc", "kind": "compiled", "ext": ".rs"},
    "c": {"probe": "gcc", "kind": "compiled", "ext": ".c"},
    "cpp": {"probe": "g++", "kind": "compiled", "ext": ".cpp"},
    "java": {"probe": "javac", "kind": "compiled", "ext": ".java"},
    "javascript": {"probe": "node", "kind": "interpreted", "ext": ".js"},
    "typescript": {"probe": "tsc", "kind": "compiled", "ext": ".ts"},
    "go": {"probe": "go", "kind": "compiled", "ext": ".go"},
    "bash": {"probe": "bash", "kind": "interpreted", "ext": ".sh"},
}


def list_languages(**_kwargs: Any) -> ToolResult:
    """Probe the host's language toolchains — what she can write in.

    For each known language reports whether its compiler/interpreter
    is installed and where. Read-only. See each toolchain's official
    docs: rustc https://doc.rust-lang.org/rustc/ · gcc
    https://gcc.gnu.org/onlinedocs/ · node https://nodejs.org/docs/
    """
    rows = []
    available: dict[str, str] = {}
    for lang, spec in _LANGS.items():
        found = shutil.which(spec["probe"])
        if found:
            available[lang] = found
        rows.append(f"{lang:11} {spec['kind']:11} {found or 'absent'}")
    return ToolResult(
        success=True,
        output="\n".join(rows),
        data={"available": available},
    )


def _compile_cmd(lang: str, src: pathlib.Path, outdir: pathlib.Path) -> list[str] | None:
    if lang == "rust":
        return ["rustc", "--edition", "2021", "-O", str(src), "-o", str(outdir / "prog")]
    if lang == "c":
        return ["gcc", "-O2", "-o", str(outdir / "prog"), str(src), "-lm"]
    if lang == "cpp":
        return ["g++", "-O2", "-o", str(outdir / "prog"), str(src)]
    if lang == "java":
        return ["javac", "-d", str(outdir), str(src)]
    if lang == "typescript":
        return ["tsc", "--outDir", str(outdir), str(src)]
    if lang == "go":
        return ["go", "build", "-o", str(outdir / "prog"), str(src)]
    return None


def _run_cmd(lang: str, src: pathlib.Path, outdir: pathlib.Path) -> list[str] | None:
    if lang == "python":
        return ["python3", str(src)]
    if lang in ("rust", "c", "cpp", "go"):
        return [str(outdir / "prog")]
    if lang == "java":
        # Class name = source filename stem.
        return ["java", "-cp", str(outdir), src.stem]
    if lang == "javascript":
        return ["node", str(src)]
    if lang == "typescript":
        return ["node", str(outdir / (src.stem + ".js"))]
    if lang == "bash":
        return ["bash", str(src)]
    return None


def run_code(
    language: str,
    source: str,
    timeout: int = 30,
    stdin: str = "",
) -> ToolResult:
    """Compile (when needed) and run a code snippet for real.

    ``language``: python, rust, c, cpp, java, javascript, typescript,
    go, bash — whichever the host toolchain provides. ``source`` is
    the program text (java: filename determines the class, so a class
    named Main needs ``source`` saved as Main.java — use file= via
    run_shell for multi-class programs). Execution is captured and
    returned with the returncode; ``stdin`` feeds the program.

    Language docs: https://docs.python.org/3/ ·
    https://doc.rust-lang.org/ · https://en.cppreference.com/ ·
    https://dev.java/learn/
    """
    lang = language.strip().lower()
    spec = _LANGS.get(lang)
    if spec is None:
        return ToolResult(
            success=False,
            error=f"unknown language {language!r}; known: {', '.join(sorted(_LANGS))}",
        )
    if shutil.which(spec["probe"]) is None:
        return ToolResult(
            success=False,
            error=f"{lang} toolchain absent on this host ({spec['probe']} not found)",
        )
    if is_blank(source):
        return ToolResult(success=False, error="empty source")
    timeout = max(1, min(timeout, 600))

    with tempfile.TemporaryDirectory(prefix="genesis_run_") as td:
        work = pathlib.Path(td)
        name = "Main" if lang == "java" else "prog"
        src = work / f"{name}{spec['ext']}"
        src.write_text(source, encoding="utf-8")
        try:
            if spec["kind"] == "compiled":
                cmd = _compile_cmd(lang, src, work)
                assert cmd is not None
                comp = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=min(timeout, 120),
                    stdin=subprocess.DEVNULL,
                )
                if comp.returncode != 0:
                    return ToolResult(
                        success=False,
                        error=f"compile failed:\n{comp.stderr}",
                        data={"stage": "compile", "stderr": comp.stderr},
                    )
            run = _run_cmd(lang, src, work)
            assert run is not None
            result = subprocess.run(
                run,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin,
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                data={"returncode": result.returncode, "language": lang},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"timed out after {timeout}s")
        except (OSError, subprocess.SubprocessError) as e:
            return ToolResult(success=False, error=str(e))


LANGUAGES_TOOLS = (
    (
        "list_languages",
        "Probe which language toolchains the host has — python, rust, "
        "c, cpp, java, javascript, typescript, go, bash — with the "
        "binary path for each that is present.",
        list_languages,
    ),
    (
        "run_code",
        "Compile and run a snippet through the real toolchain: "
        "language= python|rust|c|cpp|java|javascript|typescript|go|bash, "
        "source= program text, stdin= input, timeout= seconds. Returns "
        "stdout/stderr/returncode; compile errors are reported as "
        "failures, not crashes.",
        run_code,
    ),
)
