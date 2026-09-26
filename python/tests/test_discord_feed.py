"""Tests for discord_feed — the read-only Discord mirror.

Network is faked throughout: ``urllib.request.urlopen`` is patched so
no real webhook is ever contacted.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import discord_feed
from discord_feed import DiscordFeed, FeedLogHandler, StderrTee

URL = "https://discord.invalid/api/webhooks/1/token"


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_feed(posts: list[dict], monkeypatch, **kwargs) -> DiscordFeed:
    """A feed whose POSTs are captured into ``posts`` as JSON dicts."""

    def fake_urlopen(req, timeout=0):
        posts.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse()

    monkeypatch.setattr(discord_feed.urllib.request, "urlopen", fake_urlopen)
    kwargs.setdefault("post_interval", 0.0)
    return DiscordFeed(URL, **kwargs)


def _wait_for(predicate, timeout=3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_from_env_disabled_without_var(monkeypatch):
    monkeypatch.delenv("GENESIS_DISCORD_WEBHOOK", raising=False)
    assert DiscordFeed.from_env() is None


def test_posts_genesis_and_you_lines(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("  genesis> hello there")
        feed.post_line("  you> hi genesis")
        assert _wait_for(lambda: len(posts) >= 2)
    finally:
        feed.close()
    authors = {p.get("username"): p["content"] for p in posts}
    assert authors["Genesis"] == "hello there"
    assert authors["you"] == "hi genesis"


def test_coalesces_same_author_burst(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("genesis~ [thought] one")
        feed.post_line("genesis~ [thought] two")
        feed.post_line("genesis~ [thought] three")
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        feed.close()
    # The burst should have been coalesced into a single message.
    assert len(posts) == 1
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["content"] == (
        "[thought] one\n[thought] two\n[thought] three"
    )


def test_unprefixed_lines_use_webhook_default(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("[genesis] Daemon started.")
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        feed.close()
    assert "username" not in posts[0]
    assert posts[0]["content"] == "[genesis] Daemon started."


def test_chunks_long_content(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("x" * 5000)
        assert _wait_for(lambda: len(posts) >= 3)
    finally:
        feed.close()
    assert all(len(p["content"]) <= discord_feed._MAX_CONTENT for p in posts)
    assert "".join(p["content"] for p in posts) == "x" * 5000


def test_mentions_disabled(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("@everyone hello")
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        feed.close()
    assert posts[0]["allowed_mentions"] == {"parse": []}


def test_permanent_http_error_disables_feed(monkeypatch):
    posts: list[dict] = []

    def dead_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(URL, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(discord_feed.urllib.request, "urlopen", dead_urlopen)
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed.post("hello")
        assert _wait_for(lambda: feed._disabled)
        feed.post("never sent")
    finally:
        feed.close()
    assert posts == []


def test_log_handler_mirrors_records(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    handler = FeedLogHandler(feed)
    test_logger = logging.getLogger("test_discord_feed")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)
    try:
        test_logger.info("  genesis> composed words")
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        test_logger.removeHandler(handler)
        feed.close()
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["content"] == "composed words"


def test_stderr_tee_mirrors_and_forwards(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    real = io.StringIO()
    tee = StderrTee(real, feed)
    try:
        tee.write("[genesis] Daemon started.\npartial")
        tee.write(" line\n")
        tee.flush()
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        feed.close()
    assert real.getvalue() == "[genesis] Daemon started.\npartial line\n"
    assert posts[0]["content"] == "[genesis] Daemon started.\npartial line"


def test_blank_lines_skipped(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("")
        feed.post_line("   \n  ")
        feed.post("   ")
        time.sleep(0.2)
    finally:
        feed.close()
    assert posts == []
