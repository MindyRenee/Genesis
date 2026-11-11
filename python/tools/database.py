"""Database — embedded SQL via sqlite3, no server needed.

A file is a database. `sqlite` opens a path, runs DDL/DML/queries,
introspects tables and schema. Params are bound, never formatted —
SQL text is her own choice, values go through ?-placeholders.

Docs: https://docs.python.org/3/library/sqlite3.html ·
https://sqlite.org/lang.html
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank

_MAX_ROWS = 500


def sqlite(
    action: str,
    db: str,
    sql: str = "",
    params: str = "[]",
) -> ToolResult:
    """Embedded SQL on a sqlite file.

    ``db`` is the database file path. ``action``:
    - query — SELECT; rows as JSON arrays in data["rows"]
    - execute — any statement/transaction; reports rowcount
    - script — multiple statements via executescript
    - tables — list tables and views
    - schema — DDL of all objects (or filter with sql= LIKE pattern)
    - dump — the .dump SQL text (first 64 KiB)

    ``params`` is a JSON list bound to ?-placeholders — values never
    get string-interpolated into SQL.
    https://sqlite.org/lang.html
    """
    if is_blank(db):
        return ToolResult(success=False, error="db= path required")
    try:
        bound: list[Any] = json.loads(params) if params else []
        if not isinstance(bound, list):
            return ToolResult(success=False, error="params must be a JSON list")
    except json.JSONDecodeError as e:
        return ToolResult(success=False, error=f"bad params JSON: {e}")
    try:
        con = sqlite3.connect(str(pathlib.Path(db).resolve()))
    except sqlite3.Error as e:
        return ToolResult(success=False, error=str(e))
    try:
        cur = con.cursor()
        if action == "query":
            rows = cur.execute(sql, bound).fetchall()
            cols = [d[0] for d in cur.description or []]
            truncated = len(rows) > _MAX_ROWS
            shown = [list(r) for r in rows[:_MAX_ROWS]]
            return ToolResult(
                success=True,
                output=f"{len(rows)} row(s)" + (" — truncated" if truncated else ""),
                data={"columns": cols, "rows": shown, "count": len(rows)},
            )
        if action == "execute":
            cur.execute(sql, bound)
            con.commit()
            return ToolResult(
                success=True,
                output=f"ok — {cur.rowcount} row(s) affected",
                data={"rowcount": cur.rowcount},
            )
        if action == "script":
            cur.executescript(sql)
            con.commit()
            return ToolResult(success=True, output="script executed")
        if action in ("tables", "schema"):
            where = "WHERE name LIKE ?" if sql else ""
            args = [sql] if sql else []
            rows = cur.execute(
                "SELECT type, name, sql FROM sqlite_master " + where + " ORDER BY type, name",
                args,
            ).fetchall()
            if action == "tables":
                lines = [f"{t}: {n}" for t, n, _ in rows]
                return ToolResult(
                    success=True,
                    output="\n".join(lines) or "(no tables)",
                    data={"objects": [n for _, n, _ in rows]},
                )
            ddl = "\n\n".join(s or "" for _, _, s in rows)
            return ToolResult(
                success=True,
                output=ddl or "(no schema)",
                data={"objects": len(rows)},
            )
        if action == "dump":
            text = "\n".join(con.iterdump())
            return ToolResult(
                success=True,
                output=text[: 1 << 16],
                data={"chars": len(text)},
            )
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: query, execute, script, tables, schema, dump",
        )
    except sqlite3.Error as e:
        return ToolResult(success=False, error=f"sqlite: {e}")
    finally:
        con.close()


DATABASE_TOOLS = (
    (
        "sqlite",
        "Embedded SQL on a file: action= query|execute|script|tables|"
        "schema|dump; db= path; sql= statement; params= JSON list "
        "bound to ?-placeholders. No server — a file is a database.",
        sqlite,
    ),
)
