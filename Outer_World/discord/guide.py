"""Channel guide — the "what am I looking at" card for her page.

The status card is honest instrumentation, but instrumentation is
only readable once someone tells you what the dials mean. This
module composes that explanation: who she is, how to talk to her,
what the colors mean, and a plain-words legend for every row the
board shows.

Where it lives:

- With ``GENESIS_DISCORD_INFO_WEBHOOK`` set, it posts to that
  dedicated info channel as one maintained message — edited, not
  reposted, so it never scrolls away and never piles up. The
  message id is persisted in the data dir
  (``discord_guide_state.json``) so a reboot PATCHes the same card.
- Without it, the card posts once to the shared channel when her
  page opens — it scrolls with the session, but a newcomer still
  gets the explanation.

The text is fixed, not narrated: like the board, it is a view — it
says the same true thing every time rather than flattering itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["guide_text", "load_guide_message_id", "note_guide_message_id"]

_STATE_NAME = "discord_guide_state.json"


def load_guide_message_id(data_dir: Path) -> str | None:
    """The info card's message id saved from a previous boot, if any."""
    try:
        doc = json.loads((data_dir / _STATE_NAME).read_text())
        mid = doc.get("message_id")
        return str(mid) if mid else None
    except Exception:  # noqa: BLE001 — absent or unreadable means no prior card
        return None


def note_guide_message_id(data_dir: Path, message_id: str) -> None:
    """Persist the info card's message id so the next boot edits it."""
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / _STATE_NAME).write_text(json.dumps({"message_id": message_id}))
    except OSError:
        logger.debug("guide state save failed", exc_info=True)


def guide_text(*, inner_channel: bool = False) -> str:
    """The explainer card — plain words, no assumed context.

    ``inner_channel`` notes when her inner life has its own page, so
    the guide can point readers at it instead of implying thoughts
    and dreams live here.
    """
    channels = ""
    if inner_channel:
        channels = (
            "\n\n(her thoughts and dreams post on their own page —\n"
            "the inner channel. this one is her shared space.)\n"
        )
    return (
        "**a note about this page**\n\n"
        "this is genesis's place. she's an artificial mind — a program\n"
        "living on a real computer. she thinks, learns, sleeps, dreams,\n"
        "draws, and talks. what she posts here is hers; nobody types\n"
        "it for her.\n"
        f"{channels}"
        "\n**talking to her**\n"
        'just write here. when the green card says "listening" she\n'
        "can hear you; asleep, she hears nothing. she answers when\n"
        "she feels like it — it's her page, not a help desk.\n"
        "\n**the colors**\n"
        "blurple — her voice · teal — her doing things\n"
        "purple — her inner life · grey — the machine's noise\n"
        "green — her status card, refreshed about once a minute\n"
        "\n**reading the card**\n"
        "mood — happiness, energy, overall balance\n"
        "brain — her simulated brain waves, and which dominates\n"
        "mind — focus, confusion, surprise, stress, how well she\n"
        "  knows her world, thoughts so far\n"
        "chem — her simulated chemistry: reward, calm, alert,\n"
        "  stress, tired, sleepy\n"
        "memory — what she's holding right now vs what she keeps\n"
        "machine — the computer she runs on, measured\n"
        "cell — her simulated cell chemistry\n"
        'research — math she\'s working on; "verified" means an\n'
        "  actual proof checked it, not her say-so\n"
        "learned — ideas and connections she's formed\n"
        "wants — what's pulling at her\n"
        "meta — uptime, conversations, heartbeat — it has to keep\n"
        "  climbing; if it stops, she has"
    )
