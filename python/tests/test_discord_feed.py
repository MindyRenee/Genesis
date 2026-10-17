"""Tests for discord_feed — the webhook transport behind her channels.

Network is faked throughout: ``urllib.request.urlopen`` is patched so
no real webhook is ever contacted.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import discord_feed
from discord_feed import DiscordFeed

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


def test_post_with_and_without_author(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("hello there", "Genesis")
        feed.post("ops metric")
        assert _wait_for(lambda: len(posts) >= 2)
    finally:
        feed.close()
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["content"] == "hello there"
    assert "username" not in posts[1]


def test_coalesces_same_author_burst(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("[thought] one", "Genesis")
        feed.post("[thought] two", "Genesis")
        feed.post("[thought] three", "Genesis")
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        feed.close()
    # The burst should have been coalesced into a single message.
    assert len(posts) == 1
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["content"] == (
        "[thought] one\n[thought] two\n[thought] three"
    )


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


def test_blank_lines_skipped(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("")
        feed.post("   ")
        time.sleep(0.2)
    finally:
        feed.close()
    assert posts == []


def _capture_requests(calls: list, monkeypatch, reply: bytes = b"{}"):
    """Capture raw requests (method, url, content_type, body)."""

    class Resp(_FakeResponse):
        def read(self):
            return reply

    def fake_urlopen(req, timeout=0):
        calls.append(
            {
                "method": req.get_method(),
                "url": req.full_url,
                "content_type": req.headers.get("Content-type", ""),
                "body": req.data,
            }
        )
        return Resp()

    monkeypatch.setattr(discord_feed.urllib.request, "urlopen", fake_urlopen)


def _multipart_payload(body: bytes) -> dict:
    """Pull the payload_json part out of a multipart request body."""
    text = body.decode("utf-8", "replace")
    marker = 'name="payload_json"'
    start = text.index(marker) + len(marker)
    json_start = text.index("{", start)
    # The JSON part ends at the next boundary line.
    end = text.index("\r\n----", json_start)
    return json.loads(text[json_start:end])


def test_post_image_sends_multipart(monkeypatch):
    calls: list = []
    _capture_requests(calls, monkeypatch)
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed.post_image(b"\x89PNG fake", caption="board", author="Genesis")
        assert _wait_for(lambda: len(calls) >= 1)
    finally:
        feed.close()
    call = calls[0]
    assert call["method"] == "POST"
    assert call["content_type"].startswith("multipart/form-data")
    payload = _multipart_payload(call["body"])
    assert payload["content"] == "board"
    assert payload["username"] == "Genesis"
    assert payload["attachments"][0]["filename"] == "frame.png"
    assert b"\x89PNG fake" in call["body"]


def test_live_update_creates_then_patches(monkeypatch):
    calls: list = []
    # wait=true replies carry the created message id.
    _capture_requests(calls, monkeypatch, reply=b'{"id": "42"}')
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed.update_live(b"png-one", caption="t1")
        assert _wait_for(lambda: feed._live_id == "42")
        feed.update_live(b"png-two", caption="t2")
        assert _wait_for(lambda: len(calls) >= 2)
    finally:
        feed.close()
    first, second = calls[0], calls[1]
    assert first["method"] == "POST"
    assert "wait=true" in first["url"]
    assert second["method"] == "PATCH"
    assert second["url"].endswith("/messages/42")


def test_live_updates_collapse_pending(monkeypatch):
    calls: list = []
    _capture_requests(calls, monkeypatch, reply=b'{"id": "7"}')
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        # Pause the poster: fill the queue while it sleeps.
        feed._cond.acquire()
        try:
            feed.update_live(b"png-stale", caption="old")
            feed.update_live(b"png-fresh", caption="new")
            kinds = [i[0] for i in feed._queue]
            assert kinds.count("live") == 1
            assert feed._queue[0][3] == b"png-fresh"
        finally:
            feed._cond.release()
    finally:
        feed.close()


def test_live_patch_failure_recreates(monkeypatch):
    calls: list = []

    class Resp(_FakeResponse):
        def read(self):
            return b'{"id": "9"}'

    def flaky_urlopen(req, timeout=0):
        calls.append(req)
        if req.get_method() == "PATCH":
            raise urllib.error.HTTPError(
                URL, 404, "gone", {}, io.BytesIO(b"")
            )
        return Resp()

    monkeypatch.setattr(
        discord_feed.urllib.request, "urlopen", flaky_urlopen
    )
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed.update_live(b"png", caption="t1")
        assert _wait_for(lambda: feed._live_id == "9")
        feed.update_live(b"png2", caption="t2")
        # PATCH 404s permanently → the feed drops the id and re-POSTs.
        assert _wait_for(
            lambda: sum(1 for r in calls if r.get_method() == "POST") >= 2
        )
    finally:
        feed.close()
