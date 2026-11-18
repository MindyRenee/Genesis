"""Discord ear — the afferent nerve of her public channel.

The channel itself — *her page*, a place in the outer world — is
``discord.channel.Channel``; the outbound transport is
``discord.feed``. This module is the inbound half: a webhook is
write-only, so hearing the public needs a bot identity
(``Authorization: Bot …``) reading history, and that read loop is an
organ of the body, not part of the page. The ear lives in self —
it listens outward, but it is hers.

Each poll fetches messages after the last seen snowflake;
webhook-authored posts (her own shares and the whole mirror) are
skipped — hearing herself would be a feedback loop — and so are
other bots.

Failure isolation mirrors the feed's: transient errors log and keep
listening; a permanent auth failure (401/403) deafens the ear for
the session. Nothing here raises into Genesis.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread
from infrastructure.net import bounded_urlopen, parse_retry_after
from inhibition import NeuromodulatoryGate

logger = logging.getLogger(__name__)

_BOT_TOKEN_ENV = "GENESIS_DISCORD_BOT_TOKEN"
_API = "https://discord.com/api/v10"

# Seconds between channel reads. The ear paces itself like the
# poster — an ear that polls aggressively just buys 429s.
_LISTEN_INTERVAL = 15.0
# Max messages fetched per poll.
_LISTEN_LIMIT = 50


class DiscordListener:
    """Her ear on the channel — polls for what the public says.

    Needs ``GENESIS_DISCORD_BOT_TOKEN`` plus the channel's webhook
    URL (the channel id resolves from the webhook object on the ear's
    own thread, so construction never blocks on the network). Without
    either, no ear exists — her channel stays write-only, exactly as
    before webhooks could be heard at all.
    """

    def __init__(
        self,
        token: str,
        webhook_url: str,
        on_message,
        *,
        poll_interval: float = _LISTEN_INTERVAL,
    ) -> None:
        self._token = token
        self._webhook_url = webhook_url
        self._channel_id: str | None = None
        # The bot's own user id — learned from /users/@me on the
        # listener thread so construction stays off the network. An
        # @-mention of this id is the one inbound message whose
        # content survives intent redaction, so _hear_channel needs
        # it to count a mention as addressing her.
        self.bot_id: str | None = None
        self._on_message = on_message
        self._poll_interval = poll_interval
        self._stop = NeuromodulatoryGate()
        # Optional ``() -> bool`` the world installs: False while she
        # rests. A closed relay doesn't drop the batch — the cursor
        # simply holds, so the public's words wait at the door and
        # arrive in order when she wakes.
        self.is_awake = None
        self.deaf = False
        # Set when a human-authored message arrives with every field
        # redacted — the signature of a bot without the privileged
        # Message Content intent. Warns once per session.
        self._blind_warned = False
        self._thread = named_thread("discord-listen", self._run, comm="g-disc-listen")

    @classmethod
    def from_feed(cls, feed, on_message) -> DiscordListener | None:
        """Build an ear on the channel ``feed`` posts to.

        ``feed`` is duck-typed — the ear lives in the body and must
        not reach into the outer world for a type; all it needs is
        the webhook URL the feed was built with. Returns None without
        a bot token or a URL — no ear exists then, same as before.
        """
        token = os.environ.get(_BOT_TOKEN_ENV, "").strip()
        url = getattr(feed, "_url", "") if feed is not None else ""
        if not token or not url:
            return None
        return cls(token, url, on_message)

    @staticmethod
    def _resolve_channel(webhook_url: str, stop: NeuromodulatoryGate | None = None) -> str | None:
        """GET the webhook object → the channel it posts to."""
        req = urllib.request.Request(
            webhook_url,
            headers={"User-Agent": "GenesisChannelListener/1.0"},
            method="GET",
        )
        try:
            with bounded_urlopen(req, timeout=10, stop=stop) as r:
                data = json.loads(r.read())
            return data.get("channel_id") or None
        except Exception as e:  # noqa: BLE001
            note_swallowed("body.hearing.discord_ear.resolve_channel", e)
            return None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # join() on a thread that was never started raises
        # RuntimeError — a mind that never finished starting still
        # runs this teardown path.
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _get(self, path: str) -> tuple[object | None, float | None, bool]:
        """Bot-authed GET → (payload, retry_after, permanent)."""
        req = urllib.request.Request(
            f"{_API}{path}",
            headers={
                "Authorization": f"Bot {self._token}",
                "User-Agent": "GenesisChannelListener/1.0",
            },
            method="GET",
        )
        try:
            with bounded_urlopen(req, timeout=15, stop=self._stop) as r:
                return json.loads(r.read() or b"null"), None, False
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return None, parse_retry_after(e), False
            permanent = e.code in (401, 403)
            logger.debug(f"[discord] listen GET {path} failed: HTTP {e.code}")
            return None, None, permanent
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.debug(f"[discord] listen GET {path} error: {e}")
            return None, None, False

    def _run(self) -> None:
        # Resolve the channel first — the webhook object carries its
        # channel_id. Retried on the poll interval so a Discord blip
        # at boot doesn't deafen her for the session.
        while self._channel_id is None and not self.deaf and not self._stop.is_set():
            self._channel_id = self._resolve_channel(self._webhook_url, self._stop)
            if self._channel_id is None:
                self._stop.wait(self._poll_interval)
        if self.deaf or self._stop.is_set():
            return
        # Learn her own user id — an @-mention of it is the only
        # message kind whose content a bot without the Message
        # Content intent can still read.
        me, _retry, _perm = self._get("/users/@me")
        if isinstance(me, dict):
            self.bot_id = me.get("id")
        # She joins the channel now, not at the start of its history —
        # the cursor begins at the newest message so boot doesn't
        # replay the backlog into her ear.
        cursor = self._latest()
        while not self._stop.is_set():
            if self.deaf:
                return
            if not cursor:
                if self._stop.wait(self._poll_interval):
                    return
                cursor = self._latest()
                continue
            data, retry_after, permanent = self._get(
                f"/channels/{self._channel_id}/messages?after={cursor}&limit={_LISTEN_LIMIT}"
            )
            if permanent:
                self.deaf = True
                logger.warning(
                    "[discord] listener deafened — bot token rejected or no channel access"
                )
                return
            if retry_after is not None:
                self._stop.wait(retry_after)
                continue
            if not isinstance(data, list):
                self._stop.wait(self._poll_interval)
                continue
            # A closed relay holds the batch unconsumed: the cursor
            # doesn't advance, so the same messages re-arrive on the
            # next poll after she wakes — deferral, not a gate.
            if self.is_awake is not None and not self.is_awake():
                if self._stop.wait(self._poll_interval):
                    return
                continue
            # Newest-first payload → deliver oldest-first so her
            # world hears speech in the order it happened.
            for message in reversed(data):
                if not isinstance(message, dict):
                    continue
                mid = message.get("id")
                if mid:
                    cursor = mid
                self._deliver(message)
            self._stop.wait(self._poll_interval)

    def _latest(self) -> str | None:
        """The newest message id on the channel, or None."""
        data, _retry, permanent = self._get(f"/channels/{self._channel_id}/messages?limit=1")
        if permanent:
            self.deaf = True
            logger.warning("[discord] listener deafened — bot token rejected or no channel access")
            return None
        if isinstance(data, list) and data:
            return data[0].get("id")
        return None

    def _deliver(self, message: dict) -> None:
        """Hand one public message to the callback, if it's real speech."""
        # webhook_id marks posts made through a webhook — hers and
        # the mirror's. Hearing them back would loop her output into
        # her input.
        if message.get("webhook_id"):
            return
        author = message.get("author") or {}
        if author.get("bot"):
            return
        content = (message.get("content") or "").strip()
        if not content:
            # A human message cannot truly be empty — Discord requires
            # content, an embed, or an attachment. Arriving with all
            # three blank means the gateway redacted them: the bot is
            # missing the Message Content intent and every ordinary
            # message reads as silence. Warn once so the deafness is
            # diagnosable; only @-mentions reach her until then.
            if (
                not self._blind_warned
                and not message.get("embeds")
                and not message.get("attachments")
            ):
                self._blind_warned = True
                logger.warning(
                    "[discord] a public message arrived with every field "
                    "redacted — the bot is missing the Message Content "
                    "intent (dev portal → Bot → Privileged Gateway "
                    "Intents). Her ear hears only @-mentions until it "
                    "is enabled."
                )
            return
        try:
            self._on_message(author.get("username") or "someone", content, message)
        except Exception as e:  # noqa: BLE001
            note_swallowed("body.hearing.discord_ear.deliver", e)
