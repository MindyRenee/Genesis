"""GitHub — her account on the forge, through the official `gh` CLI.

Every action is a thin, structured pass-through to gh: auth status,
repo/issue/PR lists and views, notifications, arbitrary `gh api`
calls. Output is gh's own — JSON where gh emits it. Requires gh
installed and authenticated (`gh auth login`, done once by the
owner).

Docs: https://cli.github.com/manual/
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from tools.framework import ToolResult


def _gh(args: list[str], timeout: int = 60) -> ToolResult:
    exe = shutil.which("gh")
    if exe is None:
        return ToolResult(
            success=False,
            error="gh not installed — see https://cli.github.com/ for install",
        )
    try:
        r = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return ToolResult(
            success=r.returncode == 0,
            output=r.stdout,
            error=r.stderr,
            data={"returncode": r.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="gh timed out")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def github(
    action: str,
    repo: str = "",
    n: int = 20,
    number: int = 0,
    query: str = "",
    endpoint: str = "",
    title: str = "",
    body: str = "",
    **_kwargs: Any,
) -> ToolResult:
    """Her GitHub account, through the gh CLI.

    ``action``:
    - auth_status — who she is on GitHub (`gh auth status`)
    - repos — her repositories (`gh repo list`, n= limit)
    - repo — view a repo README/info (repo='owner/name')
    - issues — open issues (repo= optional; `gh issue list`)
    - prs — open pull requests (`gh pr list`)
    - issue / pr — view one by number=
    - issue_open — create an issue (repo=, title=, body=)
    - notifications — unread notifications
    - api — raw `gh api <endpoint>` (e.g. '/user', '/rate_limit',
      '/repos/owner/name/contents/path'); parsed as JSON when possible
    - search — `gh search repos <query>`

    Docs: https://cli.github.com/manual/
    """
    n = max(1, min(n, 100))
    if action == "auth_status":
        return _gh(["auth", "status"])
    if action == "repos":
        return _gh(["repo", "list", "--limit", str(n)])
    if action == "repo":
        if not repo:
            return ToolResult(success=False, error="repo view needs repo='owner/name'")
        return _gh(["repo", "view", repo])
    if action == "issues":
        args = ["issue", "list", "--limit", str(n)]
        return _gh(args + (["--repo", repo] if repo else []))
    if action == "prs":
        args = ["pr", "list", "--limit", str(n)]
        return _gh(args + (["--repo", repo] if repo else []))
    if action in ("issue", "pr"):
        if not number:
            return ToolResult(success=False, error=f"{action} needs number=")
        args = [action, "view", str(number)]
        return _gh(args + (["--repo", repo] if repo else []))
    if action == "issue_open":
        if not title:
            return ToolResult(success=False, error="issue_open needs title=")
        args = ["issue", "create", "--title", title, "--body", body or ""]
        return _gh(args + (["--repo", repo] if repo else []))
    if action == "notifications":
        return _gh(["api", "notifications"])
    if action == "api":
        if not endpoint:
            return ToolResult(success=False, error="api needs endpoint=")
        r = _gh(["api", endpoint])
        if r.success:
            try:
                r.data["json"] = json.loads(r.output)
            except (json.JSONDecodeError, ValueError):
                pass
        return r
    if action == "search":
        if not query:
            return ToolResult(success=False, error="search needs query=")
        return _gh(["search", "repos", query, "--limit", str(n)])
    return ToolResult(
        success=False,
        error=f"unknown action {action!r}; known: auth_status, repos, repo, "
        "issues, prs, issue, pr, issue_open, notifications, api, search",
    )


GITHUB_TOOLS = (
    (
        "github",
        "Her GitHub account via the gh CLI: action= auth_status|repos|"
        "repo|issues|prs|issue|pr|issue_open|notifications|api|search. "
        "repo='owner/name', number=, endpoint= for raw `gh api` calls "
        "(e.g. '/rate_limit'). Docs: cli.github.com/manual",
        github,
    ),
)
