"""ARC showcase — Genesis's own channel while she plays.

One private Discord channel watches her practice runs: a live board
message that keeps redrawing as she moves, her thoughts and learning
events verbatim, and a short digest of what each life taught her —
taken from the actual WorldModel delta, not written for her.

Nothing here touches policy, planning, or timing. The feed is a
bounded-queue observer in a daemon thread; if Discord is down the run
is unaffected.

Privacy and disclosure:

- Set ``GENESIS_DISCORD_ARC_WEBHOOK`` to a webhook on a *private*
  channel. Webhooks can only post, so the channel stays read-only
  for her. ``GENESIS_DISCORD_WEBHOOK`` is deliberately not a fallback
  — that channel is her public voice, not a spectator feed.
- Captions are deliberately coarse: level, turn, qualitative fuel.
  No coordinates, entity ids, routes, or score internals — the point
  is to show learning happening, not to publish the solution.
- Raw traces, recordings, and scorecards stay local in the run dir.

Dev notes — the engineer's voice — are posted through ``devnote`` or
the CLI (``python -m parietal_lobe.discord_showcase note "text"``), short and
plain: "updated perception", "fixed color error", "improved spatial".
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from typing import Any

from discord_feed import DiscordFeed

logger = logging.getLogger(__name__)

# The official ARC palette (arc_agi.rendering.COLOR_MAP), kept here so
# the renderer does not import the eval harness.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),  # 0 white
    (204, 204, 204),  # 1 off-white
    (153, 153, 153),  # 2 neutral light
    (102, 102, 102),  # 3 neutral
    (51, 51, 51),  # 4 off black
    (0, 0, 0),  # 5 black
    (229, 58, 163),  # 6 magenta
    (255, 123, 204),  # 7 magenta light
    (249, 60, 49),  # 8 red
    (30, 147, 255),  # 9 blue
    (136, 216, 241),  # 10 blue light
    (255, 220, 0),  # 11 yellow
    (255, 133, 27),  # 12 orange
    (146, 18, 49),  # 13 maroon
    (79, 204, 48),  # 14 green
    (163, 86, 214),  # 15 purple
)

_WEBHOOK_ENV = "GENESIS_DISCORD_ARC_WEBHOOK"

# Live-board redraw cadence in turns — a stride is 5 cells per turn, so
# ~4 turns gives visible motion without outrunning Discord's limits.
_LIVE_EVERY_TURNS = 4
# Seconds between relayed thoughts — her stream is faster than a
# channel should hear.
_THOUGHT_MIN_GAP = 2.5
# Cell scale for the board render (64x64 grid → 256px).
_SCALE = 4


def render_frame_png(grid: list[list[int]], scale: int = _SCALE) -> bytes:
    """Render one ARC frame grid to PNG bytes using the official palette."""
    from PIL import Image

    h = len(grid)
    w = len(grid[0]) if h else 0
    img = Image.frombytes(
        "RGB",
        (w, h),
        b"".join(
            bytes(PALETTE[v] if 0 <= v < len(PALETTE) else PALETTE[5])
            for row in grid
            for v in row
        ),
    )
    img = img.resize((w * scale, h * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fuel_word(fuel: int | None) -> str:
    """Qualitative fuel reading — the crowd sees a gauge, not a number."""
    if fuel is None:
        return ""
    if fuel <= 0:
        return "out of fuel"
    if fuel < 25:
        return "fuel critical"
    if fuel < 55:
        return "fuel low"
    return "fuel ok"


_OUTCOME_PHRASES = {
    "collect": "learned color-{c} can be collected",
    "win": "learned color-{c} finishes a level",
    "transform": "learned color-{c} rewrites her state",
    "hazard": "learned color-{c} is deadly",
    "block": "learned color-{c} is solid",
    "nothing": "learned color-{c} is inert",
}


def _digest_learning(before: dict, after: dict) -> list[str]:
    """Diff two ``export_learning`` snapshots into plain facts.

    These are system annotations describing what the model recorded —
    not Genesis speaking, so they are allowed to be terse and factual.
    """
    out: list[str] = []
    aff_b = before.get("affordances", {})
    aff_a = after.get("affordances", {})
    for color, verdicts in aff_a.items():
        new = set(verdicts) - set(aff_b.get(color, {}))
        for outcome in sorted(new):
            phrase = _OUTCOME_PHRASES.get(outcome)
            if phrase:
                out.append(phrase.format(c=color))
    new_t = len(after.get("transforms", {})) - len(before.get("transforms", {}))
    if new_t > 0:
        out.append(f"learned {new_t} state transform(s)")
    sc_new = set(after.get("state_changers", [])) - set(
        before.get("state_changers", [])
    )
    for c in sorted(sc_new):
        out.append(f"found a state-changer (color-{c})")
    bk_b = before.get("blocked_kinds", {})
    bk_a = after.get("blocked_kinds", {})
    new_bk = sum(max(0, len(v) - len(bk_b.get(k, []))) for k, v in bk_a.items())
    if new_bk > 0:
        out.append(f"mapped {new_bk} locked gate(s)")
    ok_b = before.get("open_kinds", {})
    ok_a = after.get("open_kinds", {})
    new_ok = sum(max(0, len(v) - len(ok_b.get(k, []))) for k, v in ok_a.items())
    if new_ok:
        out.append(f"confirmed {new_ok} gate(s) open")
    return out


class ArcShowcase:
    """Observer that narrates one practice run to Discord.

    Every method is safe to call with ``feed=None`` — the showcase
    becomes a no-op and the run is byte-identical without a webhook.
    """

    def __init__(
        self,
        feed: DiscordFeed | None,
        game: str,
        *,
        live_every: int = _LIVE_EVERY_TURNS,
    ) -> None:
        self._feed = feed
        self._game = game
        self._live_every = max(1, live_every)
        self._turn = 0
        self._level = -1
        self._last_thought = 0.0
        self._life_snapshot: dict = {}
        self._lives = 0
        self._level_won = 0

    @classmethod
    def from_env(cls, game: str, **kw) -> ArcShowcase:
        """Build from ``GENESIS_DISCORD_ARC_WEBHOOK``; no webhook →
        inert showcase. Her public channel is never a fallback."""
        url = os.environ.get(_WEBHOOK_ENV, "").strip()
        feed = DiscordFeed(url) if url else None
        return cls(feed, game, **kw)

    def devnote(self, text: str) -> None:
        """Engineer caption — plainly not Genesis. Keep it short."""
        if self._feed:
            self._feed.post(f"⚙ {text.strip()}")

    def announce(self, steps: int) -> None:
        if self._feed:
            self._feed.post(
                f"**Genesis is practicing `{self._game}`** — "
                f"{steps} turns. Non-LLM cognitive architecture; "
                "everything she does is learned from the pixels. "
                "Local practice, not an official evaluation."
            )

    def on_thought(self, kind: str, content: str) -> None:
        """Relay a live thought verbatim — her words, her cadence."""
        if not self._feed or not content.strip():
            return
        now = time.monotonic()
        if now - self._last_thought < _THOUGHT_MIN_GAP:
            return
        self._last_thought = now
        self._feed.post(content.strip(), "Genesis")

    def on_event(self, kind: str, text: str) -> None:
        """Relay a world-model event — these are her epistemics."""
        if not self._feed:
            return
        if kind in {"win", "transformed", "self_found"}:
            self._feed.post(text.strip(), "Genesis")
        elif kind in {"depleted", "suspect", "blocked"}:
            self._feed.post(f"_{text.strip()}_", "Genesis")

    def on_step(
        self,
        grid: list[list[int]],
        *,
        turn: int,
        level: int,
        fuel: int | None = None,
        model: Any = None,
    ) -> None:
        """Keep the live-board message current; snapshot per-life state."""
        self._turn = turn
        level_changed = level != self._level
        if level_changed:
            self._level = level
            if model is not None:
                try:
                    self._life_snapshot = model.export_learning()
                except Exception:  # noqa: BLE001 — observer must not raise
                    self._life_snapshot = {}
        if not self._feed:
            return
        if not level_changed and turn % self._live_every != 0:
            return
        fuel_txt = _fuel_word(fuel)
        caption = f"`{self._game} · level {level + 1} · turn {turn}"
        if fuel_txt:
            caption += f" · {fuel_txt}"
        caption += "`"
        try:
            png = render_frame_png(grid)
        except Exception as e:  # noqa: BLE001 — observer must not raise
            logger.debug(f"[showcase] frame render failed: {e}")
            return
        self._feed.update_live(png, caption, "Genesis")

    def on_life_end(self, level: int, outcome: str, model: Any = None) -> None:
        """Post the per-life digest: what she learned since last life."""
        if not self._feed:
            return
        self._lives += 1
        learned: list[str] = []
        if model is not None:
            try:
                learned = _digest_learning(
                    self._life_snapshot, model.export_learning()
                )
            except Exception:  # noqa: BLE001 — observer must not raise
                learned = []
        header = f"life {self._lives} ended — {outcome} (level {level + 1})"
        body = "\n".join(f"· {line}" for line in learned[:6])
        self._feed.post(
            f"**{header}**" + (f"\n{body}" if body else "\n· no new findings")
        )
        if model is not None:
            try:
                self._life_snapshot = model.export_learning()
            except Exception:  # noqa: BLE001 — observer must not raise
                self._life_snapshot = {}

    def on_level_complete(self, level: int, model: Any = None) -> None:
        self._level_won += 1
        if self._feed:
            self._feed.post(
                f"**Genesis cleared level {level + 1} of `{self._game}`.**",
            )
        if model is not None:
            self._life_snapshot = model.export_learning()

    def on_finish(self, meta: dict) -> None:
        """Close the run — honest totals, diagnostic framing."""
        if not self._feed:
            return
        levels = meta.get("levels_completed", 0)
        steps = meta.get("steps", 0)
        self._feed.post(
            f"**Run over** — `{self._game}`: {levels} level(s) cleared in "
            f"{steps} turns across {self._lives} lives. "
            "Local diagnostic practice; no official evaluation was run."
        )

    def close(self) -> None:
        if self._feed:
            self._feed.close()


def main(argv: list[str]) -> int:
    """``python -m parietal_lobe.discord_showcase note "text"`` — one devnote."""
    if len(argv) >= 2 and argv[0] == "note":
        url = os.environ.get(_WEBHOOK_ENV, "").strip()
        if not url:
            sys.stderr.write("no webhook set (GENESIS_DISCORD_ARC_WEBHOOK)\n")
            return 1
        feed = DiscordFeed(url)
        try:
            feed.post(f"⚙ {' '.join(argv[1:]).strip()}")
        finally:
            feed.close()
        return 0
    sys.stderr.write(__doc__ or "")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
