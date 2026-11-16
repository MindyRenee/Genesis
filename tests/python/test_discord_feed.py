"""Tests for discord.feed — the webhook transport behind her channels.

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

import discord.feed
from discord.feed import DiscordFeed, DiscordListener, FeedLogHandler, StderrTee

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

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", fake_urlopen)
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
        # A kind with no default speaker (pulse) leaves the webhook's
        # own name on the message — no username override is sent.
        feed.post("ops metric", kind=discord.feed.KIND_PULSE)
        assert _wait_for(lambda: len(posts) >= 2)
    finally:
        feed.close()
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["embeds"][0]["description"] == "hello there"
    assert "username" not in posts[1]
    assert posts[1]["embeds"][0]["description"] == "ops metric"


def test_post_line_classifies_prefixes(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("  genesis> hello there")
        feed.post_line("  you> hi genesis")
        feed.post_line("[genesis] Daemon started.")
        assert _wait_for(lambda: len(posts) >= 3)
    finally:
        feed.close()
    # Her words ride a card, the human's post as plain text, and
    # terminal noise attributes to "session" — never to her. A
    # uniform [tag] lifts into the card header.
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["embeds"][0]["description"] == "hello there"
    assert posts[1]["username"] == "you"
    assert posts[1]["content"] == "hi genesis"
    assert posts[2]["username"] == "session"
    assert posts[2]["embeds"][0]["author"] == {"name": "genesis"}
    assert posts[2]["embeds"][0]["description"] == "Daemon started."


def test_genesis_tilde_split_by_tag(monkeypatch):
    """genesis~ is not one stream: outward utterances ride her speech
    card, activity rides teal, thoughts go purple, and untagged
    chrome is session noise."""
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("  genesis~ [question] what is that sound?")
        feed.post_line("  genesis~ [play] shuffling the board")
        feed.post_line("  genesis~ [thought] the room is quiet")
        feed.post_line("  genesis~ (thinking...)")
        assert _wait_for(lambda: len(posts) >= 4)
    finally:
        feed.close()
    style = discord.feed._KIND_STYLE
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["embeds"][0]["color"] == style[discord.feed.KIND_SPEECH][0]
    assert posts[0]["embeds"][0]["author"] == {"name": "question"}
    assert posts[1]["embeds"][0]["color"] == style[discord.feed.KIND_ACTIVITY][0]
    assert posts[1]["embeds"][0]["author"] == {"name": "play"}
    assert posts[2]["username"] == "Genesis 💭"
    assert posts[2]["embeds"][0]["author"] == {"name": "thought"}
    assert posts[3]["username"] == "session"
    assert posts[3]["embeds"][0]["description"] == "(thinking...)"


def test_broadcast_echo_not_mirrored(monkeypatch):
    """The genesis~ [broadcast] line is already on the channel — the
    explicit post carries it untagged, so the mirror must skip the
    echo rather than land the same utterance twice."""
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("  genesis~ [broadcast] her own words")
        feed.post_line("  genesis~ [thought] a musing")
        assert _wait_for(lambda: len(posts) >= 1)
        time.sleep(0.3)
    finally:
        feed.close()
    assert len(posts) == 1
    # Inner life posts under her marked name, on a purple card whose
    # [thought] tag lifts out of the text into the card's header.
    assert posts[0]["username"] == "Genesis 💭"
    assert posts[0]["embeds"][0]["description"] == "a musing"
    assert posts[0]["embeds"][0]["author"] == {"name": "thought"}


def test_inner_lines_route_to_inner_feed(monkeypatch):
    """With an inner page configured, her thoughts leave the shared
    channel entirely — inner world is not the self∩outer overlap."""
    main_posts: list[dict] = []
    inner_posts: list[dict] = []

    def fake_urlopen(req, timeout=0):
        posts = inner_posts if "inner" in req.full_url else main_posts
        posts.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse()

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", fake_urlopen)
    main = DiscordFeed(URL, post_interval=0.0)
    main.inner_feed = DiscordFeed(URL + "-inner", post_interval=0.0)
    try:
        main.post_line("  genesis~ [thought] a musing")
        main.post_line("  genesis~ [dream] something vivid")
        main.post_line("  genesis> said aloud")
        main.post_line("  genesis~ [question] are you there?")
        assert _wait_for(lambda: len(inner_posts) >= 1 and len(main_posts) >= 1)
        time.sleep(0.3)
    finally:
        main.close()  # closing the parent drains the inner page too
    # Thoughts and dreams go inner; replies and outward questions
    # stay on the shared page — speech is the overlap, and the two
    # speech lines coalesce into one blurple card.
    assert len(main_posts) == 1
    assert len(inner_posts) == 1
    assert inner_posts[0]["username"] == "Genesis 💭"
    assert inner_posts[0]["embeds"][0]["description"] == (
        "[thought] a musing\n[dream] something vivid"
    )
    assert main_posts[0]["embeds"][0]["description"] == ("said aloud\n[question] are you there?")


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
    # The burst should have been coalesced into a single message, its
    # uniform [thought] tag lifted into the card header once.
    assert len(posts) == 1
    assert posts[0]["username"] == "Genesis 💭"
    assert posts[0]["embeds"][0]["description"] == "one\ntwo\nthree"
    assert posts[0]["embeds"][0]["author"] == {"name": "thought"}


def test_chunks_long_content(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("x" * 5000)
        assert _wait_for(lambda: len(posts) >= 3)
    finally:
        feed.close()
    texts = [p["embeds"][0]["description"] for p in posts]
    assert all(len(t) <= discord.feed._MAX_CONTENT for t in texts)
    assert "".join(texts) == "x" * 5000


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

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", dead_urlopen)
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
    test_logger = logging.getLogger("test_discord.feed")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)
    try:
        test_logger.info("  genesis> composed words")
        assert _wait_for(lambda: len(posts) >= 1)
    finally:
        test_logger.removeHandler(handler)
        feed.close()
    assert posts[0]["username"] == "Genesis"
    assert posts[0]["embeds"][0]["description"] == "composed words"


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
    assert posts[0]["embeds"][0]["description"] == ("[genesis] Daemon started.\npartial line")
    assert posts[0]["username"] == "session"


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

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", fake_urlopen)


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
    embed = payload["embeds"][0]
    assert embed["description"] == "board"
    assert embed["image"]["url"] == "attachment://frame.png"
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
            raise urllib.error.HTTPError(URL, 404, "gone", {}, io.BytesIO(b""))
        return Resp()

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", flaky_urlopen)
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed.update_live(b"png", caption="t1")
        assert _wait_for(lambda: feed._live_id == "9")
        feed.update_live(b"png2", caption="t2")
        # PATCH 404s permanently → the feed drops the id and re-POSTs.
        assert _wait_for(lambda: sum(1 for r in calls if r.get_method() == "POST") >= 2)
    finally:
        feed.close()


def test_live_text_creates_then_patches(monkeypatch):
    """The board message is one JSON message edited in place — no
    multipart, no attachment, wait=true so its id comes back."""
    calls: list = []
    _capture_requests(calls, monkeypatch, reply=b'{"id": "42"}')
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed.update_live_text("board one")
        assert _wait_for(lambda: len(calls) >= 1)
        feed.update_live_text("board two")
        assert _wait_for(lambda: len(calls) >= 2)
    finally:
        feed.close()
    first, second = calls[0], calls[1]
    assert first["method"] == "POST"
    assert "wait=true" in first["url"]
    assert first["content_type"] == "application/json"
    first_body = json.loads(first["body"])
    assert first_body["embeds"][0]["description"] == "board one"
    assert second["method"] == "PATCH"
    assert second["url"].endswith("/messages/42")
    assert json.loads(second["body"])["embeds"][0]["description"] == "board two"


def test_live_text_collapses_with_pending_live(monkeypatch):
    """Both live kinds target the same message — a queued board never
    stacks up behind a stale frame, or vice versa."""
    calls: list = []
    _capture_requests(calls, monkeypatch, reply=b'{"id": "7"}')
    feed = DiscordFeed(URL, post_interval=0.0)
    try:
        feed._cond.acquire()
        try:
            feed.update_live(b"png-stale", caption="old")
            feed.update_live_text("fresh board")
            kinds = [i[0] for i in feed._queue]
            assert kinds == ["live_text"]
        finally:
            feed._cond.release()
    finally:
        feed.close()


# ═══════════════════════════════════════════════════════════════════
# DiscordListener — her ear on the channel (bot-token reads)
# ═══════════════════════════════════════════════════════════════════


class _ScriptedGet:
    """Scripted stand-in for DiscordListener._get."""

    def __init__(self, latest=None, polls=None):
        self.calls: list[str] = []
        self.latest = latest if latest is not None else [{"id": "100"}]
        self.polls = list(polls or [])

    def __call__(self, path):
        self.calls.append(path)
        if "/users/@me" in path:
            return {"id": "999"}, None, False
        if "limit=1" in path:
            return list(self.latest), None, False
        if self.polls:
            item = self.polls.pop(0)
            if isinstance(item, tuple):
                return item
            return item, None, False
        return [], None, False


def _scripted_listener(monkeypatch, delivered: list, script: _ScriptedGet) -> DiscordListener:
    listener = DiscordListener(
        "tok",
        "wh-url",
        lambda author, content, message: delivered.append((author, content)),
        poll_interval=0.01,
    )
    monkeypatch.setattr(listener, "_resolve_channel", lambda url, stop=None: "chan123")
    monkeypatch.setattr(listener, "_get", script)
    return listener


def _msg(mid, content, username="amy", **extra):
    return {
        "id": mid,
        "content": content,
        "author": {"username": username},
        **extra,
    }


def test_listener_delivers_oldest_first(monkeypatch):
    """Newest-first payloads reach her in the order they were said."""
    delivered: list = []
    script = _ScriptedGet(
        polls=[
            [
                _msg("103", "third", "amy"),
                _msg("102", "second", "bob"),
                _msg("101", "first", "amy"),
            ]
        ]
    )
    listener = _scripted_listener(monkeypatch, delivered, script)
    listener.start()
    try:
        # The cursor tracks the newest delivered snowflake.
        assert _wait_for(lambda: "after=103" in str(script.calls))
    finally:
        listener.stop()
    assert delivered == [
        ("amy", "first"),
        ("bob", "second"),
        ("amy", "third"),
    ]


def test_listener_skips_own_posts_bots_and_empty(monkeypatch):
    """Webhook posts (hers + the mirror's), other bots, and empty
    content never reach her — hearing them would loop or be noise."""
    delivered: list = []
    script = _ScriptedGet(
        polls=[
            [
                _msg("104", "real words", "amy"),
                _msg("103", "", "amy"),
                _msg("102", "beep", "robot", author={"username": "r", "bot": True}),
                _msg("101", "her own post", "Genesis", webhook_id="w1"),
            ]
        ]
    )
    listener = _scripted_listener(monkeypatch, delivered, script)
    listener.start()
    try:
        assert _wait_for(lambda: "after=104" in str(script.calls))
    finally:
        listener.stop()
    assert delivered == [("amy", "real words")]


def test_listener_deafens_on_auth_failure(monkeypatch):
    """A rejected bot token deafens her for the session — one
    warning, no crash, no churn."""
    delivered: list = []
    script = _ScriptedGet(polls=[(None, None, True)])
    listener = _scripted_listener(monkeypatch, delivered, script)
    listener.start()
    try:
        assert _wait_for(lambda: listener.deaf)
    finally:
        listener.stop()
    assert delivered == []


def test_listener_resolve_channel_from_webhook(monkeypatch):
    """The channel id comes from the webhook object itself."""
    calls: list = []

    class Resp(_FakeResponse):
        def read(self):
            return b'{"channel_id": "777"}'

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return Resp()

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", fake_urlopen)
    assert DiscordListener._resolve_channel(URL) == "777"
    assert calls == [URL]


def test_from_feed_requires_token_and_feed(monkeypatch):
    monkeypatch.delenv("GENESIS_DISCORD_BOT_TOKEN", raising=False)
    feed = DiscordFeed.__new__(DiscordFeed)
    feed._url = URL
    assert DiscordListener.from_feed(feed, lambda *a: None) is None
    monkeypatch.setenv("GENESIS_DISCORD_BOT_TOKEN", "tok")
    assert DiscordListener.from_feed(feed, lambda *a: None) is not None
    assert DiscordListener.from_feed(None, lambda *a: None) is None


# ---------------------------------------------------------------
# Shutdown — teardown must not outlive the urlopen bound
# ---------------------------------------------------------------


def _hanging_urlopen(req, timeout=0):
    """A request that never answers — dead resolver/down webhook."""
    time.sleep(30)
    return _FakeResponse()


def test_close_binds_inflight_post_to_drain_deadline(monkeypatch):
    """close() drains for _DRAIN_DEADLINE at most — an in-flight POST
    expires *with* the deadline instead of parking the join for the
    full urlopen bound on top of it."""
    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", _hanging_urlopen)
    feed = DiscordFeed(URL, post_interval=0.0)
    feed.post("pending line")
    time.sleep(0.3)  # let the poster enter the hanging request
    start = time.monotonic()
    feed.close()
    elapsed = time.monotonic() - start
    # The thread must be *dead* when close() returns — the pre-fix
    # join expired at DRAIN+2 while the wedged request ran on to 15 s.
    assert not feed._thread.is_alive()
    assert elapsed < discord.feed._DRAIN_DEADLINE + 2.0


def test_close_interrupts_backoff_sleep(monkeypatch):
    """A rate-limit backoff must not outlive close() — the poster
    wakes on the close notify instead of sleeping out retry_after."""

    def rate_limited(req, timeout=0):
        raise urllib.error.HTTPError(URL, 429, "rl", {}, io.BytesIO(b'{"retry_after": 600}'))

    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", rate_limited)
    feed = DiscordFeed(URL, post_interval=0.0)
    feed.post("x")
    # Wait for the first POST to fail so the poster is parked in
    # the backoff sleep before close() lands.
    assert _wait_for(lambda: feed._fail_streak >= 1)
    feed.close()
    # Without the interruptible wait the poster slept retry_after=600 s
    # past close() — the thread must already be dead here.
    assert not feed._thread.is_alive()


def test_listener_stop_interrupts_inflight_get(monkeypatch):
    """stop() cuts an in-flight poll — the ear must not hold
    teardown for the rest of a wedged urlopen's bound."""
    monkeypatch.setattr(discord.feed.urllib.request, "urlopen", _hanging_urlopen)
    listener = DiscordListener("tok", "wh-url", lambda *a: None, poll_interval=0.01)
    monkeypatch.setattr(listener, "_resolve_channel", lambda url, stop=None: "chan")
    listener.start()
    time.sleep(0.2)  # let the poll thread enter a hanging _get
    start = time.monotonic()
    listener.stop()
    assert time.monotonic() - start < 3.0


# ---------------------------------------------------------------
# Terminal → Discord readability: ANSI strips, \r collapses, panels
# fence for monospace
# ---------------------------------------------------------------


def test_post_line_strips_ansi_dim_and_colour(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("  \033[2mfeeling: calm · confidence: 0.72\033[0m")
        assert _wait_for(lambda: posts)
    finally:
        feed.close()
    content = posts[-1]["embeds"][0]["description"]
    assert "\x1b" not in content
    assert content == "feeling: calm · confidence: 0.72"


def test_post_line_collapses_carriage_return_rewrites(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("working... 40%\r\033[Kworking... done")
        assert _wait_for(lambda: posts)
    finally:
        feed.close()
    # only the final frame of the in-place rewrite survives
    assert posts[-1]["embeds"][0]["description"] == "working... done"


def test_classify_sees_prefix_past_ansi(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post_line("\033[2mgenesis> hello there\033[0m")
        assert _wait_for(lambda: posts)
    finally:
        feed.close()
    # dim-wrapped prefix still classifies as her voice, escape-free
    assert posts[-1].get("username") == "Genesis"
    assert posts[-1]["embeds"][0]["description"] == "hello there"


def test_box_drawing_panel_posts_fenced(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("  ┌─ status ──\n  │  sal=0.42\n  └─────")
        assert _wait_for(lambda: posts)
    finally:
        feed.close()
    content = posts[-1]["embeds"][0]["description"]
    assert content.startswith("```\n") and content.endswith("\n```")
    assert "│  sal=0.42" in content


def test_clean_is_noop_on_plain_text(monkeypatch):
    posts: list[dict] = []
    feed = _make_feed(posts, monkeypatch)
    try:
        feed.post("a plain sentence · with a middot")
        assert _wait_for(lambda: posts)
    finally:
        feed.close()
    assert posts[-1]["embeds"][0]["description"] == "a plain sentence · with a middot"
