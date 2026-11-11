"""Notebook — her paper notebook: named pages under the data dir.

Pages live at <data_dir>/notebooks/<name>.md — persistent across
sessions, plain text she can grep. Actions: write (replace), append,
read, list, delete, search (regex across all pages). This is
scratch space, not memory — long-term knowledge belongs to the
concept network and episodic memory, not loose pages.
"""

from __future__ import annotations

import pathlib
import re
import time
from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank


def _dir() -> pathlib.Path:
    from infrastructure.config import default_data_dir

    d = default_data_dir() / "notebooks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _page(name: str) -> pathlib.Path:
    safe = re.sub(r"[^\w.-]+", "_", name.strip()) or "untitled"
    if not safe.endswith(".md"):
        safe += ".md"
    return _dir() / safe


def notebook(
    action: str,
    name: str = "",
    content: str = "",
    pattern: str = "",
    **_kwargs: Any,
) -> ToolResult:
    """Her notebook — named persistent pages in the data dir.

    ``action``: write (name, content — replaces the page), append
    (name, content), read (name), list, delete (name), search
    (pattern — regex across every page, returns 'page:line: text').
    Pages are markdown files under <data_dir>/notebooks/.
    """
    try:
        if action == "write":
            if is_blank(name):
                return ToolResult(success=False, error="write needs name=")
            p = _page(name)
            p.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"wrote {p.name} ({len(content)} chars)")
        if action == "append":
            p = _page(name)
            stamp = time.strftime("%Y-%m-%d %H:%M")
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(f"\n<!-- {stamp} -->\n{content}")
            return ToolResult(success=True, output=f"appended to {p.name}")
        if action == "read":
            p = _page(name)
            if not p.is_file():
                return ToolResult(success=False, error=f"no page named {name!r}")
            return ToolResult(
                success=True,
                output=p.read_text(encoding="utf-8"),
                data={"page": p.name},
            )
        if action == "list":
            pages = sorted(_dir().glob("*.md"))
            lines = [f"{p.name} ({p.stat().st_size} bytes)" for p in pages]
            return ToolResult(
                success=True,
                output="\n".join(lines) or "(empty notebook)",
                data={"pages": [p.name for p in pages]},
            )
        if action == "delete":
            p = _page(name)
            if not p.is_file():
                return ToolResult(success=False, error=f"no page named {name!r}")
            p.unlink()
            return ToolResult(success=True, output=f"deleted {p.name}")
        if action == "search":
            if is_blank(pattern):
                return ToolResult(success=False, error="search needs pattern=")
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return ToolResult(success=False, error=f"bad regex: {e}")
            hits = [
                f"{p.name}:{i + 1}: {line}"
                for p in sorted(_dir().glob("*.md"))
                for i, line in enumerate(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                )
                if rx.search(line)
            ]
            return ToolResult(
                success=bool(hits),
                output="\n".join(hits) or "(no matches)",
                data={"matches": len(hits)},
            )
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: write, append, read, list, delete, search",
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


NOTES_TOOLS = (
    (
        "notebook",
        "Her paper notebook — named .md pages under the data dir, "
        "persistent across sessions. action: write/append/read/list/"
        "delete/search (pattern= regex). Scratch space, not memory.",
        notebook,
    ),
)
