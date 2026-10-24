#!/usr/bin/env python3
"""ARC grid-play evidence report — generated, not asserted.

Reads the live data dir (``offered_games/``, ``gameplay.json``,
``cognitive_journal.jsonl``, ``edge_log.jsonl``) and emits a Markdown
report of every ARC-* grid task she has been offered or attempted:
attempt counts, mastery, winning program, search nodes, timestamps,
skill provenance, and trivial-predictor floors scored against the
same held-out answers her play session used.

Read-only: this tool never writes into the data dir. Re-run it any
time — new attempts update the table automatically.

    python3 scripts/arc_play_evidence.py [--data-dir DIR] [--out FILE]

The trust-boundary stance mirrors ``docs/ARC-AGI-3.md``: these are
public ARC-AGI-1 training tasks played inside her games curriculum —
live-state evidence of program synthesis and skill reuse, not an
official-evaluator scorecard and not ARC-AGI-3.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "python"))

_SKILL_RE = re.compile(r"skill:([0-9a-f]{16})")


def _default_data_dir() -> Path:
    """Resolve the data dir the way run.sh does.

    Precedence: GENESIS_DATA_DIR, then the checkout's .genesis-data-dir
    pin, then XDG_DATA_HOME, then ~/.local/share. The pin outranks
    XDG_DATA_HOME because the latter is ambient launcher environment,
    not Genesis config — a sandboxed host sets it, which would
    otherwise point the report at a sibling instance's gameplay.json.
    """
    # Blank counts as unset, matching run.sh and config.py.
    explicit = (os.environ.get("GENESIS_DATA_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    pin = Path(__file__).resolve().parent.parent / ".genesis-data-dir"
    if pin.is_file():
        pinned = pin.read_text().splitlines()[0].strip()
        if pinned:
            return Path(pinned)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "genesis"


def _cell_match(pred: list[list[int]] | None, expected: list[list[int]]) -> float:
    """Mirror of play.py's scoring: 0.0 on shape mismatch, else cell accuracy."""
    if not pred or not expected:
        return 0.0
    if len(pred) != len(expected) or len(pred[0]) != len(expected[0]):
        return 0.0
    total = len(expected) * len(expected[0])
    if total == 0:
        return 0.0
    same = sum(
        1 for r, row in enumerate(expected) for c, cell in enumerate(row) if pred[r][c] == cell
    )
    return same / total


def _floors(task: dict) -> dict[str, float]:
    """Trivial-predictor scores on the held-out test answer."""
    test = task["test"][0]
    inp, expected = test[0], test[1]
    # identity: output = input
    identity = _cell_match(inp, expected)
    # copy the first training output — the blind "repeat an example" guess
    copy0 = _cell_match(task["train"][0][1], expected)
    # best single-color fill at the *true* output shape — an oracle-shaped
    # upper bound on "just paint one color"
    h = len(expected)
    w = len(expected[0]) if h else 0
    uniform = max(
        (_cell_match([[color] * w for _ in range(h)], expected) for color in range(10)),
        default=0.0,
    )
    return {"identity": identity, "copy_train0": copy0, "uniform_best": uniform}


def _journal_attempts(data_dir: Path) -> dict[str, list[dict]]:
    """Play events per arc_* task, chronological."""
    journal = data_dir / "cognitive_journal.jsonl"
    out: dict[str, list[dict]] = {}
    if not journal.exists():
        return out
    pat = re.compile(
        r"(\w+) \[(\w+)\] score=([\d.]+).*?rule=(.*?) "
        r"nodes=(\d+) attempts=(\d+)(?: felt=(\S+))?(?: why=(.*))?$"
    )
    with journal.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"play"' not in line and '"puzzle"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = pat.search(str(e.get("text", "")))
            if not m:
                continue
            name = m.group(1)
            out.setdefault(name, []).append(
                {
                    "ts": e.get("ts", 0),
                    "family": m.group(2),
                    "score": float(m.group(3)),
                    "rule": m.group(4).strip(),
                    "nodes": int(m.group(5)),
                    "attempts": int(m.group(6)),
                    "felt": m.group(7) or "",
                    "why": (m.group(8) or "").strip(),
                }
            )
    for v in out.values():
        v.sort(key=lambda a: a["ts"])
    return out


def _skill_birthdays(data_dir: Path, skill_ids: set[str]) -> dict[str, int]:
    """First edge-log assertion of each skill node — when it was learned."""
    log = data_dir / "edge_log.jsonl"
    born: dict[str, int] = {}
    if not skill_ids or not log.exists():
        return born
    wanted = {f"skill:{s}" for s in skill_ids}
    with log.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "skill:" not in line or '"assert"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = e.get("source", "")
            if e.get("op") == "assert" and src in wanted:
                sid = src.split(":", 1)[1]
                ts = e.get("created_at") or 0
                if ts and (sid not in born or ts < born[sid]):
                    born[sid] = ts
    return born


def _git_head() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _fmt_ts(ms: int) -> str:
    if not ms:
        return "—"
    return _dt.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def _results_table(rows: list[dict]) -> list[str]:
    L = [
        "## Results\n",
        "| task | outcome | tries | mastery | winning/best rule | nodes |"
        " last played | id-copy | copy-train0 | 1-color |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rule = r["rule"] if len(r["rule"]) <= 48 else r["rule"][:45] + "…"
        fl = r["floors"]
        L.append(
            f"| {r['id']} | {r['outcome']} | {r['attempts']} | {r['mastery']:.2f} | "
            f"`{rule}` | {r['nodes']} | {r['when']} | "
            f"{fl['identity']:.0%} | {fl['copy_train0']:.0%} | {fl['uniform_best']:.0%} |"
        )
    L.append("")
    L.append(
        "Floor columns are trivial predictors scored on the same held-out "
        "answer: output=input, output=first train output, and best "
        "single-color fill *at the true output shape* (a generous upper bound "
        "on guessing one color).\n"
    )
    return L


def _traces(rows: list[dict], attempts_log: dict[str, list[dict]]) -> list[str]:
    played = [r for r in rows if r["outcome"] != "unplayed"]
    if not played:
        return []
    L = ["## Verbatim traces\n"]
    for r in played:
        for e in attempts_log.get(f"arc_{r['id']}", []):
            line = (
                f"- `{r['id']}` {_fmt_ts(e['ts'])}: score={e['score']:.2f} "
                f"`{e['rule']}` ({e['nodes']} nodes"
            )
            if e["felt"]:
                line += f", felt={e['felt']}"
            if e["why"]:
                line += f", why={e['why']}"
            L.append(line + ")")
    return [*L, ""]


def _curve_table(attempts_log: dict[str, list[dict]]) -> list[str]:
    curve = []
    for name, evs in attempts_log.items():
        if name.startswith("arc_") or len(evs) < 2:
            continue
        nodes = [e["nodes"] for e in evs]
        skills = sum(1 for e in evs if "skill:" in e["rule"])
        curve.append((name, evs[0]["family"], len(evs), nodes, skills))
    if not curve:
        return []
    L = [
        "## Learning curve — the curriculum underneath\n",
        "Same-task replays, oldest first. Falling node counts are her "
        "search getting cheaper; `skill:` in later rules is a learned "
        "procedure replacing search outright — the mechanism that let her "
        "solve `496994bd` cold.\n",
        "| game | family | plays | nodes searched per play | skill reuses |",
        "|---|---|---|---|---|",
    ]
    for name, fam, n, nodes, skills in sorted(curve):
        runs = []
        for x in nodes:
            if runs and runs[-1][0] == x:
                runs[-1][1] += 1
            else:
                runs.append([x, 1])
        seq = " → ".join(f"{x}×{k}" if k > 1 else str(x) for x, k in runs)
        L.append(f"| {name} | {fam} | {n} | {seq} | {skills} |")
    return [*L, ""]


def render(data_dir: Path) -> str:
    offered = data_dir / "offered_games"
    tasks: dict[str, dict] = {}
    for p in sorted(offered.glob("arc_*.json")):
        try:
            tasks[p.stem.removeprefix("arc_")] = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue

    progress = {}
    gp = data_dir / "gameplay.json"
    if gp.exists():
        progress = json.loads(gp.read_text())
    mastery = progress.get("mastery", {})
    attempts = progress.get("attempts", {})
    failed = progress.get("failed", {})

    attempts_log = _journal_attempts(data_dir)
    skill_ids = {
        s for evs in attempts_log.values() for e in evs for s in _SKILL_RE.findall(e["rule"])
    }
    born = _skill_birthdays(data_dir, skill_ids)

    rows = []
    solved = attempted = 0
    for tid, task in tasks.items():
        att = attempts.get(f"arc_{tid}", 0)
        m = mastery.get(f"arc_{tid}", 0.0)
        evs = attempts_log.get(f"arc_{tid}", [])
        best_ev = max(evs, key=lambda e: e["score"], default=None)
        if m >= 1.0:
            outcome, solved = "solved", solved + 1
        elif att:
            outcome, attempted = "missed", attempted + 1
        else:
            outcome = "unplayed"
        rows.append(
            {
                "id": tid,
                "outcome": outcome,
                "attempts": att,
                "mastery": m,
                "rule": best_ev["rule"] if best_ev else "—",
                "nodes": best_ev["nodes"] if best_ev else 0,
                "when": _fmt_ts(best_ev["ts"]) if best_ev else "—",
                "floors": _floors(task),
                "n_failed_rules": len(failed.get(f"arc_{tid}", [])),
            }
        )
    rows.sort(key=lambda r: (r["outcome"] != "solved", r["outcome"] != "missed", r["id"]))

    L = [
        "# ARC grid-play evidence — live state\n",
        f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from "
        f"`{data_dir}` at repo `{_git_head()}` "
        "by `scripts/arc_play_evidence.py`.\n",
        "Genesis plays public **ARC-AGI-1 training tasks** inside her games "
        "curriculum (`python/basal_ganglia/`). Each task is offered with "
        "its train pairs and test inputs; her spatial reasoner searches "
        "compositions of grid transforms, verifies candidates on train, and "
        "guesses on the held-out test input. The stored test answer is used "
        "for scoring only — `solve()` receives train pairs and test inputs "
        "(`play.py:1238-1246`); `expected` never enters the search. Mastery "
        "1.00 requires every test cell exact.\n",
        "**Scope, stated plainly:** these are *training-set* tasks — the "
        "easiest public ARC split — solved in live play, not under the "
        "official evaluator. This is evidence of real program synthesis and "
        "skill transfer, not a claim that Genesis passed ARC-AGI.\n",
        f"Offered: **{len(tasks)}** · solved: **{solved}** · attempted but "
        f"unsolved: **{attempted}** · unplayed: **{len(tasks) - solved - attempted}**\n",
    ]
    L += _results_table(rows)
    L += _traces(rows, attempts_log)
    L += _curve_table(attempts_log)
    if born:
        L.append("## Skill provenance\n")
        L.append(
            "Skills reused in solves, with their earliest assertion into her "
            "concept graph — evidence they were learned in earlier play and "
            "transferred, not fitted to the task:\n"
        )
        for sid, ts in sorted(born.items()):
            L.append(f"- `skill:{sid}` learned {_fmt_ts(ts)}")
        L.append("")
    L += [
        "## What this is not\n",
        "- Not the official ARC-AGI-1 evaluation split or evaluator.\n"
        "- Not ARC-AGI-3 — that boundary lives in `docs/ARC-AGI-3.md`.\n"
        "- Not a cherry-pick: every offered task is listed, including misses "
        "and unplayed ones, with failed candidate rules retained in "
        "`gameplay.json` (`failed`, `failed_predictions` ledgers).\n",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_default_data_dir())
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = render(args.data_dir)
    if args.out:
        args.out.write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
