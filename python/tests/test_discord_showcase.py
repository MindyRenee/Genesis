"""Tests for worlds.discord_showcase — the ARC channel layer.

The DiscordFeed object is faked entirely; no webhook is contacted.
"""

from __future__ import annotations

import io
import time

from worlds.discord_showcase import (
    ArcShowcase,
    _digest_learning,
    _fuel_word,
    render_frame_png,
)


class _FakeFeed:
    """Records the calls the showcase makes."""

    def __init__(self):
        self.texts: list[tuple[str, str | None]] = []
        self.images: list[tuple] = []
        self.live: list[tuple] = []
        self.closed = False

    def post(self, text, author=None):
        self.texts.append((text, author))

    def post_image(self, png, caption="", author=None, filename="frame.png"):
        self.images.append((png, caption, author, filename))

    def update_live(self, png, caption="", author=None):
        self.live.append((png, caption, author))

    def close(self):
        self.closed = True


def _grid(v: int = 3, w: int = 64, h: int = 64) -> list[list[int]]:
    return [[v] * w for _ in range(h)]


class _FakeModel:
    def __init__(self, data):
        self._data = data
        self.exports = 0

    def export_learning(self):
        self.exports += 1
        return self._data


def test_render_frame_png_is_valid_image():
    grid = _grid()
    grid[0][0] = 12  # orange avatar pixel
    png = render_frame_png(grid, scale=4)
    assert png[:4] == b"\x89PNG"
    from PIL import Image

    img = Image.open(io.BytesIO(png))
    assert img.size == (256, 256)


def test_render_skips_out_of_palette_values():
    grid = _grid()
    grid[5][5] = -1  # internal sprite marker, not a palette index
    png = render_frame_png(grid, scale=1)
    assert png[:4] == b"\x89PNG"


def test_fuel_word_qualitative():
    assert _fuel_word(None) == ""
    assert _fuel_word(0) == "out of fuel"
    assert _fuel_word(10) == "fuel critical"
    assert _fuel_word(40) == "fuel low"
    assert _fuel_word(90) == "fuel ok"


def test_showcase_noop_without_feed():
    show = ArcShowcase(None, "ls20")
    show.announce(10)
    show.devnote("updated perception")
    show.on_thought("thought", "something")
    show.on_step(_grid(), turn=1, level=0, fuel=50)
    show.on_life_end(0, "reserve emptied")
    show.on_level_complete(0)
    show.on_finish({"levels_completed": 1, "steps": 10})
    show.close()


def test_live_board_throttled_to_cadence():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20", live_every=4)
    for turn in range(1, 9):
        show.on_step(_grid(), turn=turn, level=0, fuel=80)
    # First frame (level change) plus turns 4 and 8.
    assert len(feed.live) == 3
    assert "level 1" in feed.live[-1][1]
    assert "turn 8" in feed.live[-1][1]


def test_live_board_caption_is_coarse():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20")
    show.on_step(_grid(), turn=1, level=0, fuel=10)
    caption = feed.live[0][1]
    assert "fuel critical" in caption
    # No coordinates or entity ids leak into captions.
    for token in ("(", ")", "cell", "pos"):
        assert token not in caption


def test_thoughts_relayed_as_genesis_throttled():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20")
    show.on_thought("thought", "the edge bar shrank when I moved")
    show.on_thought("thought", "immediately suppressed")
    assert len(feed.texts) == 1
    assert feed.texts[0][1] == "Genesis"
    show._last_thought = 0  # simulate the gap elapsing
    time.sleep(0.01)
    show.on_thought("learning", "next thought through")
    assert len(feed.texts) == 2


def test_events_relayed_as_her_epistemics():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20")
    show.on_event("win", "level complete")
    show.on_event("depleted", "the reserve was empty")
    assert feed.texts[0] == ("level complete", "Genesis")
    assert feed.texts[1][0].startswith("_")
    assert feed.texts[1][1] == "Genesis"


def test_life_digest_reports_real_deltas():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20")
    before = {"affordances": {}, "transforms": {}, "state_changers": []}
    after = {
        "affordances": {"11": {"collect": 3}},
        "transforms": {"0": [["sig"], [["sig2", 2]]]},
        "state_changers": [0, 11],
    }
    model = _FakeModel(after)
    show._life_snapshot = before
    show.on_life_end(0, "reserve emptied", model=model)
    text = feed.texts[0][0]
    assert "life 1 ended" in text
    assert "color-11 can be collected" in text
    assert "state transform" in text
    assert "state-changer" in text


def test_life_digest_no_new_findings():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20")
    same = {"affordances": {"11": {"collect": 3}}}
    show._life_snapshot = same
    show.on_life_end(0, "reset", model=_FakeModel(same))
    assert "no new findings" in feed.texts[0][0]


def test_digest_ignores_open_and_blocked_regressions():
    before = {"open_kinds": {"9": [["a"], ["b"]]}, "blocked_kinds": {"9": [["a"]]}}
    after = {"open_kinds": {"9": [["a"], ["b"], ["c"]]}, "blocked_kinds": {"9": [["a"], ["b"]]}}
    lines = _digest_learning(before, after)
    assert any("gate" in line for line in lines)


def test_level_and_finish_announced_honestly():
    feed = _FakeFeed()
    show = ArcShowcase(feed, "ls20")
    show.on_level_complete(0)
    show.on_finish({"levels_completed": 1, "steps": 42})
    texts = [t for t, _ in feed.texts]
    assert any("cleared level 1" in t for t in texts)
    # The run summary must never read as an official result.
    assert any("no official evaluation" in t for t in texts)
