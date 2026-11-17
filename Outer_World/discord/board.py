"""Channel board — the live instrument card on her public page.

The board is a view of her vitals posted to the channel the world
holds: daemon CoreState numbers (phase, chemistry, memory tallies,
the inference block's tick counter) and runtime facts (uptime,
conversations, whether her ear is on). Her words do not live here —
they already post as their own readable cards. Nothing is composed —
the board's honesty is that it can't flatter itself: the tick count
either climbs or it doesn't.

The card is written for the people who wander in, not for the
operator who knows the internals: rows are labeled in plain words
(mood, memory, wants — never ``F``, ``allo``, ``stm``), and the
second line always says whether she's listening, since that's the
first thing a visitor wants to know. The full legend lives on the
guide card (``discord.guide``) — the board names things, it does
not explain them.

This is a *view*, not an organ: it renders state the body and mind
already produce and hands the text to the feed. It owns no rhythm —
``update`` is driven by the world's tick and rate-limited to
``_CHANNEL_BOARD_S`` so a webhook edit stays far under channel
limits. Fixed, not adaptive: the board is an instrument panel —
it should tick like a clock so a reader sees numbers move on
schedule.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

# Seconds between board refreshes.
_CHANNEL_BOARD_S = 45.0

# Wave-band labels for the board's power strip, ordered slow→fast.
_BOARD_WAVES = (
    ("delta", "δ"),
    ("theta", "θ"),
    ("alpha", "α"),
    ("beta1", "β1"),
    ("beta2", "β2"),
    ("beta3", "β3"),
    ("gamma", "γ"),
)
# The chemistry channels an observer reads as regulation: reward,
# mood, vigilance, stress, sleep pressure, sleep onset. The left
# word is what the row calls it on the card — plain names a reader
# already owns.
_BOARD_CHEMS = (
    ("reward", "dopamine"),
    ("calm", "serotonin"),
    ("alert", "norepinephrine"),
    ("stress", "cortisol"),
    ("tired", "adenosine"),
    ("sleepy", "melatonin"),
)
# Dominant-band frequency ranges (Hz) — the wave labels carry their
# physical scale, not just a name.
_BOARD_WAVE_HZ = {
    "delta": "0.5-4",
    "theta": "4-8",
    "sigma": "12-16",
    "alpha": "8-12",
    "beta1": "12-15",
    "beta2": "15-20",
    "beta3": "20-30",
    "gamma": "30-80",
    "epsilon": "100-200",
    "infraslow": "0.1-0.5",
}

# The board's line budget. Discord renders a code block in a
# horizontally-scrolling box — on a phone that's ~38-40 monospace
# columns, so every board line is packed under 36 chars. A wide row
# becomes indented continuation lines under the same label column;
# readings are never clipped, only reflowed.
_BOARD_WIDTH = 36
# Width of the label column every board row shares ("mood     ").
_BOARD_LABEL = 9


def _board_pack(label: str, bits: list[str]) -> list[str]:
    """Pack one labeled row of readings into ≤_BOARD_WIDTH lines.

    `bits` are self-contained readings ("δ 0.09", "short 41") joined
    by single spaces; the first line carries the label, continuations
    indent under the value column. A single bit wider than the value
    column still gets its own line — the row goes long rather than
    break a reading in half.
    """
    prefix = f"{label:<{_BOARD_LABEL}}"
    indent = " " * _BOARD_LABEL
    lines = []
    cur = prefix
    for bit in bits:
        at_col_start = cur == prefix or cur == indent
        cand = cur + bit if at_col_start else cur + " " + bit
        if not at_col_start and len(cand) > _BOARD_WIDTH:
            lines.append(cur)
            cur = indent + bit
        else:
            cur = cand
    lines.append(cur)
    return lines


def _board_power(v: float) -> str:
    """Wave power for the board — honest at the small end.

    A measured 0.004 is not 0.00: when two decimals would round a
    real reading to the lie "0.00" the board gives it a third. A
    band that is genuinely ~zero still prints "0.00" — the digits
    tell absent-tiny from absent.
    """
    if 0.0 < v < 0.005:
        return f"{v:.3f}"
    return f"{v:.2f}"


def _board_rate(v: float) -> str:
    """A big per-second rate in the compact form people actually
    read — ``1,204,882`` is noise on a phone; ``1.2M`` is a number."""
    if v >= 1_000_000:
        return f"{v / 1e6:.1f}M"
    if v >= 1_000:
        return f"{v / 1e3:.1f}K"
    return f"{v:.0f}"


class ChannelBoard:
    """Her page's instrument card — renders vitals, posts via the feed.

    Args:
        host: The mind the board reads (duck-typed — client, feel,
            brain_waves, network, volition, world, and the session
            counters). The board never writes to it; it is read-only
            instrumentation.
    """

    def __init__(self, host: Any) -> None:
        self._h = host
        self._next_board_at = 0.0
        self._guide_id_saved = False

    def update(self, now: float) -> None:
        """Push the card to her channel on a fixed cadence."""
        channel = getattr(getattr(self._h, "world", None), "channel", None)
        feed = channel.feed if channel is not None else None
        if feed is None or now < self._next_board_at:
            return
        self._next_board_at = now + _CHANNEL_BOARD_S
        try:
            feed.update_live_text(self.board_text())
        except Exception as e:  # noqa: BLE001
            note_swallowed("discord.board.update", e)
        self._save_guide_id(feed)

    def _save_guide_id(self, feed: Any) -> None:
        """Persist the info card's message id once the poster reports it.

        The guide posts at page-open time but the feed assigns its
        message id asynchronously on the poster thread; the board's
        cadence is the convenient place to notice the id arriving and
        write it down, so the next boot edits the same card instead
        of stacking a fresh one on the info page.
        """
        if self._guide_id_saved:
            return
        info = getattr(feed, "info_feed", None)
        mid = getattr(info, "live_message_id", None) if info is not None else None
        if not mid:
            return
        try:
            from discord.guide import note_guide_message_id

            data_dir = getattr(self._h, "data_dir", None)
            if data_dir:
                note_guide_message_id(Path(data_dir), str(mid))
            self._guide_id_saved = True
        except Exception as e:  # noqa: BLE001
            note_swallowed("discord.board.guide_id", e)

    def board_text(self) -> str:
        """Render the card — her vitals, not her voice."""
        h = self._h
        sleeping = bool(getattr(h, "_is_sleeping", False))
        lines = [f"genesis · {'asleep' if sleeping else 'awake'}"]
        try:
            core = h.client.get_state()
        except Exception:  # noqa: BLE001 — a dead daemon is telemetry too
            core = None
        if core is None:
            lines[0] += " · daemon unreachable"
        else:
            lines[0] += f" · {core.phase_name}"
        lines.extend(self._visitor_lines(sleeping))
        if core is not None:
            self._core_lines(core, lines)
        self._footer_lines(core, lines)
        return "```\n" + "\n".join(lines) + "\n```"

    def _visitor_lines(self, sleeping: bool) -> list[str]:
        """The lines a stranger needs first: can she hear me, and
        where does one learn what all this means."""
        channel = getattr(getattr(self._h, "world", None), "channel", None)
        ear = channel.ear if channel is not None else "off"
        if sleeping:
            talk = "sleeping — she can't hear you"
        elif ear == "on":
            talk = "listening — say hi here"
        elif ear == "deaf":
            talk = "she can't hear right now"
        else:
            talk = "she can't hear this channel"
        lines = [talk]
        if getattr(getattr(channel, "feed", None), "info_feed", None) is not None:
            lines.append("the info channel explains this")
        return lines

    def _core_lines(self, core, lines: list[str]) -> None:
        """Board rows sourced from the daemon's CoreState."""
        self._neuro_lines(core, lines)
        self._field_lines(lines)
        self._cell_line(lines)

    def _neuro_lines(self, core, lines: list[str]) -> None:
        """Rows for the neurochemical and inference state."""
        h = self._h
        try:
            lines[0] += f" · {h.feel(core).label}"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"feel() failed on board line: {e}")
        try:
            waves = h.brain_waves(core)
            bits = [
                f"{glyph} {_board_power(waves.powers[w])}"
                for name, glyph in _BOARD_WAVES
                for w in waves.powers
                if w.value == name
            ]
            if bits:
                lines.extend(_board_pack("brain", bits))
                dominant = getattr(waves, "dominant", None)
                if dominant is not None:
                    hz = _BOARD_WAVE_HZ.get(dominant.value, "?")
                    lines.append(f"mostly   {dominant.value} {hz} Hz")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"brain_waves failed on board line: {e}")
        lines.extend(
            _board_pack(
                "mood",
                [
                    f"{core.valence:+.2f}",
                    f"energy {core.arousal:.2f}",
                    f"balance {core.global_tone:.2f}",
                ],
            )
        )
        inf = core.inference
        if inf is not None:
            lines.extend(
                _board_pack(
                    "mind",
                    [
                        f"focus {inf.precision:.2f}",
                        f"confusion {inf.free_energy:.3f}",
                        f"surprise {inf.surprise_ema:.3f}",
                        f"stress {inf.allostasis_load:.2f}",
                        f"knows world {inf.model_maturity:.2f}",
                    ],
                )
            )
            lines.append(f"thoughts {inf.inference_tick_count:,}")
        chem = core.chemicals or {}
        bits = [f"{short} {chem[name]:.2f}" for short, name in _BOARD_CHEMS if name in chem]
        if bits:
            lines.extend(_board_pack("chem", bits))
        # stm must come from the live ring — MemoryStats reports
        # RingBuffer::count(); the CoreState memory-pointers field
        # at offset 2584 is a dead mirror that nothing writes.
        try:
            stm_live = h.client.get_memory_stats().stm_count
        except Exception:  # noqa: BLE001
            stm_live = core.stm_count
        lines.extend(
            _board_pack("memory", [f"short {stm_live}", f"long {core.ltm_episode_count:,}"])
        )

    def _field_lines(self, lines: list[str]) -> None:
        """Rows for the substrate's measured physical life.

        The field probe decomposes real instruction throughput and
        scheduler jitter at the wave bands' center frequencies, writes
        a carrier rhythm into real work, and reads real watts, rail
        voltage, and current where the machine's sensors report them.
        ``None`` means the probe isn't running: absent, never zero.
        """
        h = self._h
        try:
            probe = h.client.get_field_probe()
        except Exception:  # noqa: BLE001
            probe = None
        if probe is None:
            lines.append("machine  probe off")
            return
        field_bits = [
            f"{probe.carrier_freq:.1f} Hz",
            f"{_board_rate(probe.throughput_mean)} ops/s",
            f"jitter {probe.jitter_cv:.2f}",
            f"irq {probe.irq_rate:,.0f}/s",
        ]
        if probe.carrier_coherence is not None:
            field_bits.append(f"steady {probe.carrier_coherence:.2f}")
        lines.extend(_board_pack("machine", field_bits))
        elec_bits = []
        if probe.power_watts is not None:
            elec_bits.append(f"{probe.power_watts:.1f} W")
        if probe.rail_volts is not None:
            elec_bits.append(f"{probe.rail_volts:.2f} V")
        if probe.supply_amps is not None:
            elec_bits.append(f"{probe.supply_amps:.2f} A")
        if probe.power_coherence is not None:
            elec_bits.append(f"steady {probe.power_coherence:.2f}")
        if elec_bits:
            lines.extend(_board_pack("power", elec_bits))
        # The radio sense — the room's RF environment: WLAN
        # field strength + traffic flux, Bluetooth HCI event
        # flux. None-gated like everything else — absent radios
        # print nothing.
        rf_bits = []
        if probe.rf_rssi_dbm is not None:
            rf_bits.append(f"wifi {probe.rf_rssi_dbm:.0f} dBm")
        if probe.rf_noise_dbm is not None:
            rf_bits.append(f"noise {probe.rf_noise_dbm:.0f}")
        if probe.rf_bytes_s is not None:
            rf_bits.append(f"data {_board_rate(probe.rf_bytes_s)} B/s")
        if probe.bt_events_s is not None:
            rf_bits.append(f"bt {probe.bt_events_s:.0f} ev/s")
        if probe.bt_rssi_dbm is not None:
            rf_bits.append(f"bt {probe.bt_rssi_dbm:.0f} dBm")
        if probe.rf_retry_rate is not None:
            strain = (
                probe.rf_retry_rate
                + (probe.rf_fail_rate or 0.0)
                + (probe.rf_beacon_loss_rate or 0.0)
            )
            rf_bits.append(f"strain {strain:.1f}/s")
        if probe.rf_tx_mbps is not None:
            rf_bits.append(f"tx {probe.rf_tx_mbps:.0f} Mbps")
        if rf_bits:
            lines.extend(_board_pack("radio", rf_bits))

    def _cell_line(self, lines: list[str]) -> None:
        """The cell's modeled electrochemical state — the ion model's
        membrane potential is a Nernst-model quantity (simulated),
        not a measured rail; the real rail, when a sensor reports
        it, is on the power line above."""
        h = self._h
        try:
            ions = h.client.get_ion_summary()
            lines.extend(
                _board_pack(
                    "cell",
                    [
                        f"volts {ions.membrane_potential_mv:+.1f} mV",
                        f"flow {ions.net_membrane_current:.2f}",
                        f"fire {ions.excitability:.2f}",
                        f"gradient {ions.gradient_integrity:.2f}",
                        f"energy {ions.atp_availability:.2f}",
                    ],
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ion summary failed on board line: {e}")

    def _footer_lines(self, core, lines: list[str]) -> None:
        """Board rows sourced from her own runtime state."""
        h = self._h
        # The research bench — real math with an oracle: claims are
        # only "verified"/"refuted" by check_equality, never by her.
        # Absent until the first thread exists (an empty bench is not
        # a result).
        try:
            bench = h._research_workbench()
            threads = bench.threads()
        except Exception:  # noqa: BLE001
            bench, threads = None, []
        if threads:
            open_n = sum(1 for t in threads if t.status == "open")
            done_n = sum(1 for t in threads if t.status == "resolved")
            ver = sum(1 for t in threads for c in t.claims if c.provenance == "verified")
            ref = sum(1 for t in threads for c in t.claims if c.provenance == "refuted")
            proof_bits = [
                f"{open_n} open",
                f"{done_n} resolved",
                f"{ver} verified",
                f"{ref} refuted",
            ]
            # The ledger's digest anchors the tallies to the file —
            # anyone holding workbench.json can recompute it and
            # confirm the board isn't showing rewritten history.
            digest = getattr(bench, "digest", lambda: "")()
            if digest:
                proof_bits.append(f"#{digest}")
            lines.extend(_board_pack("research", proof_bits))
        # Growth she can be measured by — concept tally, edge tally,
        # and the plasticity gate that throttles learning.
        try:
            net = h.network
            grow_bits = [f"{net.size} ideas", f"{net.edge_count} links"]
            archive = getattr(net, "archive_size", 0)
            if archive:
                grow_bits.append(f"{archive} archived")
            if core is not None:
                grow_bits.append(f"learning {core.plasticity_gate:.2f}")
            lines.extend(_board_pack("learned", grow_bits))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"growth row failed on board line: {e}")
        # The cause — which drives are pulling at her right now.
        # Urge values are computed state, not narration.
        try:
            urges = sorted(
                h.volition.state().items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            top = [f"{name} {value:.2f}" for name, value in urges[:3] if value > 0.01]
            if top:
                lines.extend(_board_pack("wants", top))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"drive row failed on board line: {e}")
        up_s = max(
            0,
            int(time.time() - getattr(h, "_session_started_at", time.time())),
        )
        hours, minutes = divmod(up_s // 60, 60)
        meta_bits = [
            f"up {hours}h{minutes:02d}m",
            f"{getattr(h, '_interaction_count', 0)} chats",
        ]
        if core is not None:
            # Freshness and continuity proofs: the raw 5 Hz body
            # counter must climb between refreshes, the snapshot age
            # must stay ~0 (a replayed card can't hold it), and the
            # instance id proves the same process keeps answering.
            age_ms = max(0.0, time.time() * 1000 - core.last_updated)
            meta_bits += [
                f"hb {core.heartbeat:,}",
                f"fresh {age_ms / 1000:.1f}s",
                f"id {core.instance_id}",
            ]
        lines.extend(_board_pack("meta", meta_bits))
