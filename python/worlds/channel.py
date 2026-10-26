"""Channel — her page in the outer world.

A Discord channel is a place in the world, not a part of the mind:
the public gathers there, she posts there, and when a bot token
exists she can hear it. The transports live in ``discord_feed``
(the platform adapter — webhook out, bot-token polling in); this
object is the world-facing thing — the page's lifecycle, its ear,
and the door inbound speech comes through.

The channel owns no cognition: it delivers what it hears to
``on_speech`` (the OuterWorld registers its own handler — the
hearing is the world's to record; whether to answer is the mind's
to decide). Without a webhook there is no page and no channel
object — her world is simply without that place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from discord_feed import DiscordFeed, DiscordListener

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Channel"]


class Channel:
    """Her page — a place in the outer world where the public gathers.

    Owns the outbound transport (``feed`` — posts, images, the
    self-editing board) and the inbound ear (``listener`` — None
    when no bot token is configured; a webhook alone is write-only).
    ``on_speech`` is assigned by the world that holds the channel;
    until then inbound speech has somewhere to arrive but nobody to
    record it.

    Args:
        feed: The posting transport the page runs on.
        listener: An already-built ear, or None to build one from the
            feed (the usual path — ``DiscordListener.from_feed`` needs
            the bot-token env, so the ear may legitimately be absent).
    """

    def __init__(
        self,
        feed: DiscordFeed,
        listener: DiscordListener | None = None,
    ) -> None:
        self.feed = feed
        # The ear on the page. ``self._deliver`` is bound before the
        # listener exists, so the callback chain is stable even though
        # the world's handler attaches later.
        self.on_speech: Callable[[str, str, dict], None] | None = None
        self.listener = (
            listener if listener is not None else DiscordListener.from_feed(feed, self._deliver)
        )

    @classmethod
    def from_env(cls) -> Channel | None:
        """Open her page when a channel webhook is configured.

        Returns None without ``GENESIS_DISCORD_WEBHOOK`` — there is
        no page then, and the world holds no channel.
        """
        feed = DiscordFeed.from_env()
        if feed is None:
            return None
        return cls(feed)

    def _deliver(self, author: str, content: str, message: dict) -> None:
        """The ear heard a public message — hand it to the world.

        The listener calls this from its poll thread. ``on_speech``
        may be unset if the world hasn't claimed the page yet (a
        channel the mind never adopted); unheard delivery is simply
        dropped, matching the feed's failure isolation.
        """
        if self.on_speech is not None:
            self.on_speech(author, content, message)

    def start(self) -> None:
        """Open the ear — begin polling the channel for public speech."""
        if self.listener is not None:
            self.listener.start()

    def stop(self) -> None:
        """Close the ear — idempotent; the listener guards its own join."""
        if self.listener is not None:
            self.listener.stop()

    def close(self) -> None:
        """Shut the page — ear first, then the poster's drain window."""
        self.stop()
        self.feed.close()

    @property
    def bot_id(self) -> Any:
        """The bot's own user id, once the ear has learned it."""
        return self.listener.bot_id if self.listener is not None else None

    @property
    def ear(self) -> str:
        """Ear status for status surfaces: ``off`` | ``deaf`` | ``on``."""
        if self.listener is None:
            return "off"
        return "deaf" if self.listener.deaf else "on"
