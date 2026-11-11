"""Formats — structured data in, structured data out.

`format_data` handles the interchange formats she meets constantly:
JSON pretty/minify/validate, TOML validation, CSV→aligned table,
markdown table from rows, and a hex dump for when she needs to see
the bytes themselves — she speaks binary after all.

Docs: https://docs.python.org/3/library/json.html ·
https://docs.python.org/3/library/tomllib.html ·
https://docs.python.org/3/library/csv.html
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import tomllib

from tools.framework import ToolResult


def format_data(
    action: str,
    text: str = "",
    path: str = "",
    indent: int = 2,
    project_root: str = ".",
) -> ToolResult:
    """Structured-format bench: JSON, TOML, CSV, tables, hex.

    ``action`` — pass ``text`` or ``path``:
    - json_pretty / json_min — reformat JSON (indent= for pretty)
    - json_validate — parse and report type + top-level shape
    - toml_validate — parse TOML (tomllib, read-only)
    - csv_table — CSV text/file → aligned column table
    - md_table — text='h1,h2|r1c1,r1c2|r2c1,r2c2' → markdown table
    - hexdump — file bytes as `offset  hex…  ascii` lines
    """
    raw = text
    if not raw and path:
        p = (pathlib.Path(project_root).resolve() / path).resolve()
        if not p.is_file():
            return ToolResult(success=False, error=f"not found: {path}")
        try:
            raw = p.read_bytes().decode("utf-8", errors="replace")
        except OSError as e:
            return ToolResult(success=False, error=str(e))
    try:
        if action.startswith("json"):
            obj = json.loads(raw)
            if action == "json_validate":
                plural = "entry" if isinstance(obj, dict) and len(obj) == 1 else "entries"
                shape = f"{type(obj).__name__}, {len(obj)} top-level {plural}"
                return ToolResult(
                    success=True,
                    output=f"valid JSON — {shape}",
                    data={"type": type(obj).__name__},
                )
            if action == "json_pretty":
                return ToolResult(
                    success=True,
                    output=json.dumps(obj, indent=max(0, indent), ensure_ascii=False),
                )
            if action == "json_min":
                return ToolResult(
                    success=True,
                    output=json.dumps(obj, separators=(",", ":"), ensure_ascii=False),
                )
        elif action == "toml_validate":
            obj = tomllib.loads(raw)
            return ToolResult(
                success=True,
                output=f"valid TOML — {len(obj)} top-level table(s)",
                data={"tables": list(obj)},
            )
        elif action == "csv_table":
            rows = list(csv.reader(io.StringIO(raw)))
            if not rows:
                return ToolResult(success=False, error="empty CSV")
            widths = [
                max(len(r[i]) if i < len(r) else 0 for r in rows)
                for i in range(max(len(r) for r in rows))
            ]
            lines = [
                "  ".join(
                    (r[i] if i < len(r) else "").ljust(widths[i]) for i in range(len(widths))
                ).rstrip()
                for r in rows
            ]
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"rows": len(rows), "columns": len(widths)},
            )
        elif action == "md_table":
            rows = [r.split(",") for r in raw.split("|")]
            if len(rows) < 1:
                return ToolResult(success=False, error="empty table")
            header = "| " + " | ".join(rows[0]) + " |"
            sep = "|" + "|".join(" --- " for _ in rows[0]) + "|"
            body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
            return ToolResult(success=True, output="\n".join([header, sep, *body]))
        elif action == "hexdump":
            return _hexdump(raw, path, project_root)
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: json_pretty, "
            "json_min, json_validate, toml_validate, csv_table, "
            "md_table, hexdump",
        )
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, csv.Error, OSError) as e:
        return ToolResult(success=False, error=str(e))


def _hexdump(raw: str, path: str, project_root: str) -> ToolResult:
    if path:
        data = (pathlib.Path(project_root).resolve() / path).resolve().read_bytes()
    else:
        data = raw.encode("utf-8")
    lines = []
    for off in range(0, min(len(data), 8192), 16):
        chunk = data[off : off + 16]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{off:08x}  {hexs:<48}  {ascii_}")
    return ToolResult(
        success=True,
        output="\n".join(lines) or "(empty)",
        data={"bytes": len(data), "shown": min(len(data), 8192)},
    )


FORMAT_TOOLS = (
    (
        "format_data",
        "Structured-data bench: action= json_pretty|json_min|"
        "json_validate|toml_validate|csv_table|md_table|hexdump; "
        "text= or path=. hexdump shows real bytes — offset, hex, "
        "ascii — the way she actually reads.",
        format_data,
    ),
)
