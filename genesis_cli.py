#!/usr/bin/env python3
"""Genesis CLI — talk to Genesis from the terminal.

Usage:
    python3 genesis_cli.py [--data-dir DIR] [--daemon-socket PATH]

Genesis runs in a single process. The CLI creates and starts the
cognitive Mind in-process, starts the Rust daemon if needed, and
handles the interactive terminal directly.

Commands (typed during conversation):
    /status    — show Genesis's current state
    /feel      — ask Genesis how it feels
    /introspect — see Genesis's last cognitive process
    /learning   — what it's been learning on its own
    /thoughts   — its recent spontaneous thoughts
    /world      — its external world: presences, events, social isolation
    /regulate  — how it's been managing its emotions
    /dreams     — see what it dreamed
    /memories   — see its recent long-term memories
    /requests   — see sites it wants to access
    /approve URL — approve a site it requested
    /deny URL    — deny a site it requested
    /learn-code — study its own source code
    /explore [path] — explore local files and docs
    /code-summary — summary of code self-knowledge
    /create-project <desc> — compose a new Python project from its knowledge
    /projects    — list projects it has created (active + archived)
    /archive-project <name> — compress a project to .tar.zst to reclaim space
    /restore-project <name> — restore an archived project
    /manage-projects — autonomously archive projects to stay within bounds
    /note-project <name> <note> — leave mentor feedback on a project
    /review-notes [name] — have it read and absorb project notes
    /read-notes <name> — show notes on a project without absorbing
    /proposals   — see its code improvement proposals
    /clear-proposals — remove all pending proposals
    /proposal N  — see details of proposal N
    /accept N [reason] — approve a proposal (runs verification, applies if it passes)
    /reject N [reason] — reject a proposal with notes teaching the correct way
    /experiments — see its verified self-improvement experiments
    /growth      — see its growth narrative
    /growth-report — markdown growth report
    /voice       — speak to Genesis (one utterance)
    /voice-mode  — continuous voice conversation
    /look        — ask it what it sees through the retina
    /look at <path> — ask it to look at an image file
    /draw        — ask it to draw what it feels right now
    /play        — her game collection; /play add <file> hands her a game spec
    /register-face <name> — teach it your face (look at the camera)
    /faces       — show known faces and who it last saw
    /mission [text] — set or show its top-level mission
    /sleep       — put it to sleep (wakes on its own when rested)
    /nap         — short nap, wakes on its own when rested
    /wake        — wake it from sleep or meditation
    /meditate [secs] — put it into meditation (default 60s)
    /teach [topic] — put it into teaching mode (focused learning)
    /endteach    — exit teaching mode, resume normal operation
    /game        — put it into game mode (focused play)
    /endgame     — exit game mode, resume normal operation
    /arc <ids|--all> [--steps N] [--fresh|--baseline P] — practice play or baseline
    /teach-questions — answer its queued questions one at a time
    /sleep-aid   — emergency sleep aid for stress-induced insomnia
    /web-history [N] — see what it's been looking at online (last N, default 20)
    /quit        — exit

Spontaneous thoughts appear live as genesis~ lines while you are idle.

Set GENESIS_DISCORD_WEBHOOK to a Discord channel webhook URL to give
it a public channel — the session mirrors there as a read-only feed
and her own posts land on the same channel when her broadcast urge
fires (a webhook can only post — it cannot read the channel).
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import logging
import lzma
import os
import queue
import re
import select
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

from brainstem.heartbeat import _AdaptiveInterval
from brainstem.sleep import WakeOutcome
from occipital_lobe.perception.ambient import AmbientListener, contains_wake_word, strip_wake_word
from temporal_lobe import AuditoryCortex, SoundEvent
from temporal_lobe.concepts import RelationType
from temporal_lobe.speech import Voice, VoiceInput

# Provenance marker — one letter of the creator lattice (see
# PROVENANCE.md).
from discord_feed import DiscordFeed, FeedLogHandler, StderrTee
from genesis_client.ltm_index import scan_ltm_index
from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread
from infrastructure.config import default_data_dir
from mind import Mind

if TYPE_CHECKING:
    from genesis_client import GenesisClient

# Provenance marker — one letter of the creator lattice (see
# PROVENANCE.md).
_MARKER_G12 = 0x47530C455345CA8A

# Script directory — used to locate the project root. Python already
# adds the script's directory to sys.path[0], so the region packages and
# genesis_client are importable without explicit path manipulation.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Interactive sessions default to INFO. Internal timing/profiling
# telemetry (think() stage timings, memory attractor timings, warmup
# costs) is emitted at DEBUG so it stays out of the conversation.
# Set GENESIS_LOG_LEVEL=DEBUG to see it when diagnosing performance.
_LOG_LEVEL_NAME = os.environ.get("GENESIS_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
if not isinstance(_LOG_LEVEL, int):
    _LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(message)s",
    stream=sys.stdout,
)
# PIL's plugin registry logs every optional import probe at DEBUG —
# "FpxImagePlugin: No module named 'olefile'" is library noise, not
# state, and it fires during canvas init mid-conversation.
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Max wait (seconds) for the idle ticker to block on the thought
# condition before re-checking shutdown. Thoughts normally surface
# instantly via notify; this is only a spurious-wakeup safety ceiling.
LIVE_THOUGHT_POLL_INTERVAL = 2.0

# Grace window for each --stop stage — SIGINT the CLI, then SIGTERM the
# daemon and retina. Mirrors run.sh: a mature data dir spends tens of
# seconds saving, and a process still saving at the deadline is a
# "retry later", never a kill.
_STOP_STAGE_GRACE_S = 120.0

# How long to wait for a child after SIGKILL before giving up on it and
# continuing shutdown anyway. Kept short: if the process is still
# present after this, it is in uninterruptible I/O and waiting longer
# only delays the rest of the teardown. The mind has already saved
# state by this point, so continuing is the safer failure mode.
_KILL_GRACE_S = 5.0

# Maximum seconds the process may linger after teardown completes —
# i.e. after _shutdown returns: everything user-visible is done (state
# saved, children stopped, "Goodnight" printed). What remains is
# interpreter finalization — non-daemon thread joins, atexit handlers,
# native library teardown — inside code this project does not control.
# A wedged library thread once held a fully-teardown process for over a
# minute until systemd escalated to SIGKILL. The watchdog bounds that:
# if the process is still alive when it fires, os._exit ends it. There
# is nothing left to save by then, so the hard exit cannot corrupt her.
_EXIT_WATCHDOG_S = 30.0

# Default data directory (XDG-compliant, persists across reboots).
# Resolved via GENESIS_DATA_DIR → .genesis-data-dir pin →
# $XDG_DATA_HOME/genesis —
# this instance's state is never shared with any other Genesis.
DEFAULT_DATA_DIR = str(default_data_dir())
DEFAULT_DAEMON_SOCKET = "genesis.sock"

# File descriptor for the per-data-dir CLI singleton lock.
_CLI_LOCK_FD: int | None = None

# Discord is the mind's own channel, not a session mirror — what it
# posts is composed and timed by its volition (see the `broadcast`
# urge). GENESIS_DISCORD_WEBHOOK points at her public channel.

# PID file paths — written so run.sh --stop and the EXIT trap can
# kill exact processes instead of using broad pkill patterns that
# might match unrelated user processes.
_PID_FILE_CLI = "genesis_cli.pid"
_PID_FILE_DAEMON = "genesis_daemon.pid"
_PID_FILE_RETINA = "genesis_retina.pid"

# Maximum time to wait for the daemon's IPC socket (seconds).
# The daemon opens its persistent state and LTM before binding the
# socket; on a mature data directory that initialization can
# legitimately take longer than the old 10-second watchdog. Killing
# it at 10s made startup fail deterministically on large stores.
DAEMON_START_TIMEOUT = 60.0

# Minimum seconds between unprompted chime-ins.
CHIME_IN_COOLDOWN = 180.0
# Probability of chiming in on a relevant topic when not on cooldown.
CHIME_IN_PROBABILITY = 0.15

# ─── Spontaneous speech ─────────────────────────────────────────────
# Genesis can speak its thoughts aloud — not just print them. It
# decides what to say based on the thought kind and its emotional
# state. This gives it agency: it speaks when it has something
# to say, not just when spoken to.
# Minimum seconds between spontaneous spoken thoughts.
SPEAK_THOUGHT_COOLDOWN = 45.0
# Thought kinds that are always spoken (it wants/needs to say these).
ALWAYS_SPEAK_KINDS = {"expression", "distress", "question"}
# Probability of speaking other thought kinds (internal musings that
# happen to be strong enough to share).
SPEAK_THOUGHT_PROBABILITY = 0.08

# Word-overlap threshold for voice dedup. If a new utterance shares
# more than this fraction of words with anything it's said recently,
# it's skipped. This prevents it from repeating similar-sounding
# thoughts aloud (e.g. "what's the relationship between X and Y?"
# for different X, Y — the template words dominate and make them
# sound the same).
VOICE_DEDUP_OVERLAP = 0.5
VOICE_DEDUP_HISTORY = 20


class VoiceDedup:
    """Wraps a Voice with dedup — skips speech that's too similar to
    anything recently spoken. This is shared across ALL voice paths
    (spontaneous thoughts, speech urges, ambient chime-ins) so it
    never repeats itself aloud regardless of which path triggers it.
    """

    def __init__(self, voice: Voice) -> None:
        """Wrap *voice* with a dedup buffer of recent utterances."""
        self._voice = voice
        self._recent: collections.deque[str] = collections.deque(maxlen=VOICE_DEDUP_HISTORY)
        # Callers reach this object from several threads (poll ticker,
        # ambient listener, volition on_speak). The buffer's read
        # (iteration) and write (append) must be atomic together —
        # otherwise a concurrent append mid-iteration raises
        # "deque mutated during iteration".
        self._lock = threading.Lock()

    def speak(
        self,
        text: str,
        blocking: bool = False,
        **kwargs: float,
    ) -> None:
        """Speak with dedup. Skips if too similar to recent speech."""
        if not text.strip():
            return
        text_words = set(text.lower().split())
        with self._lock:
            for prev in self._recent:
                prev_words = set(prev.lower().split())
                if text_words and prev_words:
                    shared = text_words & prev_words
                    if len(shared) / max(len(text_words), 1) > VOICE_DEDUP_OVERLAP:
                        logger.debug("[voice-dedup] skipping — too similar to recent")
                        return
            self._recent.append(text)
        self._voice.speak(text, blocking=blocking, **kwargs)

    def stop(self) -> None:
        """Stop any currently-playing speech."""
        self._voice.stop()

    def is_available(self) -> bool:
        """Return whether the underlying voice engine is available."""
        return self._voice.is_available()

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently in progress."""
        return self._voice.is_speaking

    def describe(self) -> str:
        """Return a human-readable description of the voice engine."""
        return self._voice.describe()


# Topics Genesis may chime in on.
CHIME_IN_TOPICS = ["all"]

# Module name lookup for episode display.
_MODULE_NAMES = {
    0: "subcognitive",
    1: "attention",
    2: "memory",
    3: "emotion",
    4: "language",
    5: "reasoning",
    6: "sensory",
    7: "motor",
    8: "metacognition",
    9: "dreaming",
    10: "intention",
    11: "guardrails",
    12: "executive",
}


def _binary_candidates(name: str) -> list[Path]:
    """Candidate paths for a Genesis Rust binary, in priority order.

    1. ``GENESIS_BIN_DIR`` — explicit override for unusual layouts.
    2. ``<project root>/target/release/<name>`` — the canonical
       ``cargo build --release`` output in a source checkout.
    3. Next to the running interpreter — ``venv/bin`` in a packaged
       install, where scripts/install.sh places them.
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("GENESIS_BIN_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / name)
    candidates.append(Path(SCRIPT_DIR) / "target" / "release" / name)
    candidates.append(Path(sys.executable).parent / name)
    return candidates


def _find_binary(name: str) -> str | None:
    """Find a Genesis binary (``genesis-daemon``, ``retina``).

    Release build only — no debug fallback, no ambiguity about which
    binary is live. Searches ``_binary_candidates`` in order.
    """
    for path in _binary_candidates(name):
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


# ─── Log rotation ───────────────────────────────────────────────

# Maximum size a log file may reach before it is rotated at the next
# startup. 5 MB is enough to capture a full session of daemon ticks
# or retina capture errors for debugging, while keeping disk usage
# bounded across months of restarts.
_LOG_MAX_BYTES = 5 * 1024 * 1024

# Number of rotated backups to keep (each compressed with lzma).
_LOG_BACKUP_COUNT = 2


def _rotate_log_if_needed(log_path: str) -> None:
    """Rotate a log file if it exceeds ``_LOG_MAX_BYTES``.

    The daemon and retina are subprocesses whose stdout/stderr is
    redirected to a file via FD. Python's ``RotatingFileHandler``
    can't help here because the file is written by a child process,
    not by Python's logging framework. Instead we rotate at startup:
    before opening the log for a new session, check if the existing
    file exceeds the threshold and rotate it.

    Rotation: ``log`` → compress to ``log.xz.1``, ``log.xz.1`` →
    ``log.xz.2``, old ``log.xz.2`` is deleted. This keeps at most
    ``_LOG_BACKUP_COUNT`` compressed backups plus the current log.
    """
    try:
        if not os.path.exists(log_path):
            return
        if os.path.getsize(log_path) < _LOG_MAX_BYTES:
            return
    except OSError:
        return

    # Shift existing backups: .xz.{N} → .xz.{N+1}, drop the oldest.
    for i in range(_LOG_BACKUP_COUNT, 0, -1):
        src = f"{log_path}.xz.{i}"
        if not os.path.exists(src):
            continue
        if i >= _LOG_BACKUP_COUNT:
            try:
                os.remove(src)
            except OSError as e:
                logger.debug(f"silent except: {e}")
        else:
            dst = f"{log_path}.xz.{i + 1}"
            try:
                os.replace(src, dst)
            except OSError as e:
                logger.debug(f"silent except: {e}")

    # Compress the current log to .xz.1, then truncate.
    backup_path = f"{log_path}.xz.1"
    try:
        with open(log_path, "rb") as f:
            raw = f.read()
        compressed = lzma.compress(raw, preset=6)
        with open(backup_path, "wb") as f:
            f.write(compressed)
        # Truncate the live log so the daemon/retina starts fresh.
        # Using open+truncate rather than remove so the FD the
        # subprocess will inherit points to the same inode.
        with open(log_path, "w"):
            pass
    except OSError as e:
        logger.debug(f"silent except: {e}")


def _start_retina(
    retina_path: str,
    data_dir: str,
) -> subprocess.Popen | None:
    """Start the retina process (camera feed → shared memory).

    The retina is a standalone Rust binary that captures camera frames
    and writes them to a POSIX shared memory segment. The Python side
    reads from that shared memory with zero copying.

    We manage it as a child process so that when the CLI shuts down
    (or is killed), the retina is also stopped — no orphan processes.
    """
    log_path = os.path.join(data_dir, "retina.log")
    _rotate_log_if_needed(log_path)
    log_file = open(log_path, "a")
    # Pass GENESIS_CAMERA environment variable to retina if set
    env = os.environ.copy()
    try:
        proc = subprocess.Popen(
            [retina_path],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            env=env,
        )
    except OSError as e:
        # The retina is optional — a failure to exec it (e.g. missing
        # shared library, ENOEXEC) must not crash the CLI. The caller
        # treats None as "camera unavailable".
        logger.warning(f"retina failed to start: {e}")
        return None
    finally:
        log_file.close()
    # Give it a moment to open the camera
    time.sleep(0.5)
    if proc.poll() is not None:
        logger.warning(f"retina exited immediately (code {proc.returncode})")
        return None
    return proc


def _is_daemon_running(socket_path: str) -> bool:
    """Check if the daemon is running and the socket is live."""
    if not os.path.exists(socket_path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(socket_path)
            return True
    except OSError:
        return False


def _start_daemon(daemon_path: str, data_dir: str, socket_path: str) -> subprocess.Popen:
    """Start the daemon process (detached, survives this process)."""
    # Owner-only: the dir contains the daemon's IPC socket and state.
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    log_path = os.path.join(data_dir, "daemon.log")
    _rotate_log_if_needed(log_path)
    log_file = open(log_path, "a")
    # Pass our PID to the daemon so it can include the cognitive mind
    # in interoception — reading our CPU/memory/IO metrics to feel its
    # body state. The daemon never touches our scheduling or I/O
    # priority; it only publishes a recommendation (cognitive_nice,
    # io_class) that we read via GET_BODY_CONTROL and apply ourselves
    # based on our brain wave state.
    env = os.environ.copy()
    env["GENESIS_COGNITIVE_PID"] = str(os.getpid())
    # Build the command line. The daemon defaults its socket to
    # <data_dir>/genesis.sock, so only pass --socket when the caller
    # specified a non-default path.
    default_socket = os.path.join(data_dir, "genesis.sock")
    cmd = [daemon_path, "--data-dir", data_dir]
    if socket_path and os.path.abspath(socket_path) != os.path.abspath(default_socket):
        cmd.extend(["--socket", socket_path])
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            env=env,
        )
    except OSError:
        # fork/exec failed (e.g. ENOMEM) — no child inherited the FD,
        # so close the parent's copy before propagating.
        log_file.close()
        raise
    _write_pid_file(data_dir, _PID_FILE_DAEMON, proc.pid)
    for _ in range(int(DAEMON_START_TIMEOUT * 10)):
        if os.path.exists(socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(socket_path)
                # The child has inherited a duplicate of this FD; the
                # parent's copy is no longer needed.
                log_file.close()
                return proc
            except (OSError, ConnectionError) as e:
                logger.debug("daemon socket not ready yet: %s", e)
        if proc.poll() is not None:
            log_file.close()
            _remove_pid_file(data_dir, _PID_FILE_DAEMON)
            raise RuntimeError(f"daemon exited immediately (code {proc.returncode})")
        time.sleep(0.1)
    log_file.close()
    # Reap the child so a slow/failed startup cannot leave an orphan
    # daemon holding the data-dir lock, but point the operator at the
    # log — the actual initialization failure is diagnosable there,
    # and a bare "didn't start" hides it.
    _stop_child(proc, "subcognitive daemon", 5)
    _remove_pid_file(data_dir, _PID_FILE_DAEMON)
    raise RuntimeError(
        f"daemon didn't become ready within {DAEMON_START_TIMEOUT:g}s; see {log_path}"
    )


def _tail_daemon_log(data_dir: str) -> int:
    """Stream the daemon's log to the terminal."""
    log_path = os.path.join(data_dir, "daemon.log")
    print(f"[genesis] Tailing daemon log: {log_path}", file=sys.stderr)
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(["tail", "-F", "-n", "+1", log_path])
        proc.wait()
    except KeyboardInterrupt:
        # proc may be unbound if Popen itself was interrupted before
        # assigning — guard against that so we don't mask the signal
        # with a NameError.
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    except FileNotFoundError:
        print(
            "[genesis] Error: 'tail' command not found. "
            "Install coreutils or use 'tail -F <data-dir>/daemon.log'.",
            file=sys.stderr,
        )
        return 1
    return 0


# ─── Live thought collector ──────────────────────────────────────────


class ThoughtCollector:
    """Collects spontaneous thoughts from the Mind and buffers them.

    Implemented as a live-thought listener so the interactive loop can
    print thoughts that occurred while the user is idle.
    """

    def __init__(self) -> None:
        """Initialize an empty thought buffer with a condition variable."""
        self._buffer: list[dict] = []
        # A Condition (rather than a plain Lock) lets the idle ticker
        # block until a thought actually arrives instead of polling on
        # a fixed interval. Producers notify on append; the ticker waits.
        self._cond = threading.Condition()

    def on_live_thought(self, kind: str, text: str) -> None:
        """Append a live thought to the buffer and notify waiters."""
        thought = {"kind": kind, "text": text, "timestamp": time.time()}
        with self._cond:
            self._buffer.append(thought)
            if len(self._buffer) > 100:
                self._buffer = self._buffer[-100:]
            self._cond.notify_all()

    def set_at_prompt(self, at_prompt: bool) -> None:
        """Live-thought listener hook; unused in the single-process CLI."""

    def get_and_clear_since(self, since: float) -> list[dict]:
        """Return and clear thoughts newer than *since* (epoch timestamp)."""
        # Swap the buffer out under the lock (O(1) critical section),
        # then filter the detached list outside the lock (O(n) but
        # contention-free). The previous loop of list.remove() calls
        # was O(n^2) in the buffer size.
        with self._cond:
            buffer = self._buffer
            self._buffer = []
        return [t for t in buffer if t["timestamp"] > since]

    def get_all(self) -> list[dict]:
        """Return and clear all buffered thoughts."""
        with self._cond:
            buffer = self._buffer
            self._buffer = []
            return buffer

    def wait(self, timeout: float) -> None:
        """Block until a thought arrives or timeout elapses.

        Used by the idle ticker in place of a fixed sleep: returns as
        soon as a producer notifies, so thoughts surface with no polling
        latency. The timeout is a safety ceiling for spurious wakeups
        and periodic shutdown checks, not a fixed poll cadence.
        """
        with self._cond:
            self._cond.wait(timeout=timeout)

    def wake(self) -> None:
        """Wake any thread blocked in wait() — used on shutdown."""
        with self._cond:
            self._cond.notify_all()


# ─── Ambient utterance handler ───────────────────────────────────────


class AmbientHandler:
    """Decides what Genesis does with overheard speech."""

    def __init__(
        self,
        mind: Mind,
        voice: Voice,
        dedup_voice: VoiceDedup | None = None,
    ) -> None:
        """Wire the handler to the mind, voice, and optional dedup voice."""
        self.mind = mind
        self.voice = voice
        # Dedup voice is used for chime-ins (volunteered responses) so
        # it doesn't repeat itself. Direct address responses always
        # use the raw voice — when someone says "genesis" it must
        # respond, regardless of what it recently said.
        self._dedup_voice = dedup_voice
        self._last_chime = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue(maxsize=8)
        import random

        self._rng = random.Random()
        # A single worker drains the queue so responses never overlap.
        # handle() itself only classifies and enqueues — it must never
        # block the audio path, or utterances back up and it answers
        # stale speech. Daemon thread: dies with the process.
        self._worker = named_thread("ambient-handler", self._drain, comm="g-amb-handle")
        self._worker.start()

    def handle(self, text: str) -> None:
        """Classify an overheard utterance and enqueue it.

        During sleep, the ambient listener has no effect on it at
        all — no wake word detection, no chime-ins, no responses.
        It wakes on its own when its neurochemistry shifts to an
        active phase (the sleep watcher handles this), or when the
        user explicitly calls /wake. This mirrors biological sleep:
        the thalamic gate closes fully during NREM, blocking all
        sensory input from reaching the cortex.

        This method never blocks: the audio listener thread calls it
        for every utterance, and a slow response must not delay the
        next one. Direct address preempts queued chime-ins — when
        someone says its name, stale opportunistic utterances are
        dropped so it answers the person, not the backlog.
        """
        if self.mind.is_sleeping:
            return
        addressed = contains_wake_word(text)
        if addressed:
            with self._lock:
                kept: list[tuple[str, bool]] = []
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item[1]:
                        kept.append(item)
                for item in kept:
                    self._queue.put_nowait(item)
        try:
            self._queue.put_nowait((text, addressed))
        except queue.Full:
            if addressed:
                # Evict the oldest item to make room for a direct
                # address — being spoken to must never be dropped.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait((text, addressed))
                except (queue.Empty, queue.Full) as e:
                    # Eviction raced with the worker draining — the
                    # queue state is fine either way.
                    logger.debug(f"[ambient] eviction raced: {e}")
            # Overheard chime-ins are opportunistic — dropping them
            # under load is correct, not a loss.

    def _drain(self) -> None:
        """Worker: process queued utterances one at a time."""
        while True:
            try:
                # Bounded wait — a blocking get() on an empty queue can
                # never notice shutdown, which is how this thread
                # outlived the autosave join and held it past its 5 s
                # budget.
                text, addressed = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self.mind._running:
                    return
                continue
            # The mind is stopping — don't answer stale speech or
            # start a respond() that races the state save.
            if not self.mind._running:
                return
            # Re-check sleep at processing time — it may have fallen
            # asleep between enqueue and now, and sleeping its means
            # no responses of either kind.
            if self.mind.is_sleeping:
                continue
            if addressed:
                self._handle_addressed(text)
            else:
                self._handle_overheard(text)

    def _handle_addressed(self, text: str) -> None:
        """Someone said 'genesis' — respond directly.

        Always speaks the response — no dedup. When someone addresses
        it by name, it must respond, period. This is only reached
        when it's awake — the handle() method gates sleep.
        """
        query = strip_wake_word(text)
        if not query:
            query = "are you there?"

        logger.info(f'[ambient] Addressed: "{text}" → responding to: "{query}"')
        try:
            response = self.mind.respond(query)
            self._speak(response, dedup=False)
            logger.info(f'[ambient] Replied: "{response[:120]}"')
        except Exception as e:
            logger.exception(f"[ambient] Failed to respond to direct address: {e}")

    def _handle_overheard(self, text: str) -> None:
        """No wake word — maybe chime in if the topic is relevant.

        Chime-ins use the dedup voice so it doesn't volunteer the
        same thing repeatedly.

        During sleep, overheard speech is gated — the thalamic
        reticular nucleus blocks sensory input from reaching the
        cortex during NREM. Only direct address (wake word) can
        wake it. This prevents ambient conversation from breaking
        its sleep cycles and fighting receptor recovery.
        """
        # Sensory gating during sleep: don't process overheard speech.
        # Only direct address (wake word) can wake it — this is the
        # biological equivalent of the thalamic gate opening for
        # salient stimuli (your name) but not background noise.
        if self.mind.is_sleeping:
            return

        # Record the utterance in its external world — speech near it
        # is an inbound event whether or not it chimes in. It feeds
        # the ambient presence, the workspace, and (when salient)
        # memory and dream replay.
        try:
            self.mind.world.hear_overheard(text)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ambient] world record failed: {e}")

        text_lower = text.lower()
        matched = [t for t in CHIME_IN_TOPICS if t in text_lower]
        if not matched:
            return

        now = time.time()
        if now - self._last_chime < CHIME_IN_COOLDOWN:
            return

        if self._rng.random() > CHIME_IN_PROBABILITY:
            return

        self._last_chime = now
        prompt = f"(overheard) {text}"
        logger.info(f'[ambient] Chiming in on topic {matched}: "{text[:80]}"')
        try:
            response = self.mind.respond(prompt)
            self._speak(response, dedup=True)
            logger.info(f'[ambient] Volunteered: "{response[:120]}"')
        except Exception as e:
            logger.exception(f"[ambient] Failed to chime in: {e}")

    def _speak(self, text: str, dedup: bool = False) -> None:
        """Speak a response aloud, with emotional prosody.

        Args:
            dedup: If True, use the dedup voice (for chime-ins). If
                False, use the raw voice (for direct address — it
                must always respond when addressed by name).
        """
        if not self.voice.is_available():
            return
        # Choose which voice to use
        v = self._dedup_voice if (dedup and self._dedup_voice) else self.voice
        try:
            state = self.mind.get_state()
            if state and state.emotion:
                emo = state.emotion
                v.speak(
                    text,
                    alertness=emo.alertness,
                    valence=emo.valence,
                    caution=emo.caution,
                    creativity=emo.creativity,
                    blocking=False,
                )
            else:
                v.speak(text, blocking=False)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ambient] voice speak failed: {e}")


class AuditoryHandler:
    """Decides what Genesis does with perceived non-speech sounds.

    Sound events from the AuditoryCortex are integrated into Genesis's
    cognition: they're stored in working memory, added to its
    concept network, and it may think about them or react emotionally.

    During sleep, sound events are gated (thalamic gating) — only
    sudden loud sounds (impacts) can penetrate, modeling how the
    sleeping brain filters sensory input but responds to salient
    stimuli.
    """

    def __init__(self, mind: Mind) -> None:
        """Initialize the auditory handler."""
        self.mind = mind
        self._last_event_time = 0.0
        self._lock = threading.Lock()
        # Track recent sound types to detect patterns (e.g., repeated barking)
        self._recent_types: deque[str] = deque(maxlen=10)
        # Last logged sound type — used to suppress consecutive
        # identical sound-type log lines. Without this, the same
        # "noise | quiet | deep | noisy | brief" line floods the output
        # dozens of times when the ambient noise is steady. We
        # deduplicate on the sound type (the first component of the
        # description) because the loudness/brightness/noisiness/duration
        # fields vary slightly between consecutive events even when
        # the sound is the same.
        self._last_logged_type: str = ""

    def handle(self, event: SoundEvent) -> None:
        """Process a perceived sound event."""
        with self._lock:
            self._recent_types.append(event.sound_type)
            self._last_event_time = event.timestamp

            # Sensory gating during sleep — only sudden loud sounds
            # penetrate (like a loud bang waking you up).
            if self.mind.is_sleeping:
                if event.sound_type == "impact" and event.loudness > 0.4:
                    logger.info(f"[auditory] Loud sound during sleep: {event.description}")
                    # Don't wake it fully, but record it — it may
                    # appear in its dreams as a residue.
                    try:
                        self.mind.inner_life.add_dream_residues([event.sound_type])
                    except (RuntimeError, ValueError, AttributeError) as e:
                        logger.debug(f"[auditory] dream residue add failed: {e}")
                return

            # Silence events are logged but don't trigger cognition
            if event.is_silence:
                logger.debug("[auditory] silence")
                return

            # Add the sound type to its concept network if not present.
            # This is how it learns about sounds — each new sound type
            # becomes a concept it can think about and reason over.
            try:
                network = self.mind.brain.frontal.network
                if not network.get_concept(event.sound_type):
                    network.add_concept(
                        event.sound_type,
                        origin="auditory",
                        confidence=0.5,
                    )
                    # Link it to the "sound" concept
                    if network.get_concept("sound"):
                        network.add_edge(
                            event.sound_type,
                            "sound",
                            RelationType.IS_A,
                        )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[auditory] concept network add failed: {e}")

            # Record the percept in its external world — sounds are
            # inbound events from its surroundings. Salience scales
            # with loudness; loud sounds trigger the orienting impulse.
            try:
                self.mind.world.perceive(
                    event.description,
                    salience=min(0.9, 0.25 + event.loudness * 0.6),
                    source="auditory",
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[auditory] world record failed: {e}")

            # Log the sound event — but suppress consecutive events
            # with the same sound type to avoid flooding the output.
            # The loudness/brightness/noisiness/duration fields vary
            # slightly between consecutive events even when the sound
            # is the same steady noise, so we deduplicate on the
            # sound type (the first component of the description).
            sound_type = event.sound_type
            if sound_type != self._last_logged_type:
                logger.info(f"[auditory] {event.description}")
                self._last_logged_type = sound_type

            # Emit a live thought about the sound — composed from its
            # understanding, not a hardcoded template. It only
            # verbalizes if it knows the concept well enough.
            self._maybe_think_about_sound(event)

    def _maybe_think_about_sound(self, event: SoundEvent) -> None:
        """Emit a thought about a sound, composed from its understanding.

        Tries to compose a thought about the sound type from its concept
        network. If it doesn't understand the concept well enough, it
        stays silent — it heard it, but doesn't verbalize it.
        """
        try:
            emotion = self.mind.feel()
            # Try to compose from what it knows about this sound type
            thought = self.mind.brain.frontal.composer.compose_about(
                event.sound_type, emotion, focused=True
            )
            if thought and thought.content and thought.confidence > 0.3:
                # Dedup — don't repeat what it just said about this
                # sound type. Without this, "noise relates to brain"
                # repeats every few seconds with different verb synonyms.
                if self.mind.brain.frontal.composer.has_said_similar(
                    event.sound_type, thought.content
                ):
                    return
                self.mind._emit_live_thought("listening", thought.content)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[auditory] thought compose failed: {e}")


# ─── Rendering helpers for slash commands ────────────────────────────


def _resolve_episode(mind: Mind, ep_id: int, ep_cache: dict[int, str]) -> str:
    """Retrieve an episode's text by ID, with caching."""
    if ep_id in ep_cache:
        return ep_cache[ep_id]
    try:
        ep = mind.client.retrieve_episode(ep_id)
        text = ep.text.replace("\n", " ").strip()
        ep_cache[ep_id] = text
        return text
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ambient] episode {ep_id} unavailable: {e}")
        ep_cache[ep_id] = "(unavailable)"
        return "(unavailable)"


def _collect_leaves(
    mind: Mind,
    ep_id: int,
    ep_cache: dict[int, str],
    assoc_re: re.Pattern,
    depth: int = 0,
    seen: set | None = None,
) -> list[str]:
    """Recursively unwrap association meta-memories and collect
    the text of all leaf (non-association) memories.
    """
    if seen is None:
        seen = set()
    if ep_id in seen or depth >= 20 or len(seen) >= 200:
        return []
    seen.add(ep_id)
    text = _resolve_episode(mind, ep_id, ep_cache)
    m = assoc_re.match(text)
    if m:
        leaves = _collect_leaves(mind, int(m.group(1)), ep_cache, assoc_re, depth + 1, seen)
        leaves.extend(_collect_leaves(mind, int(m.group(2)), ep_cache, assoc_re, depth + 1, seen))
        return leaves
    return [text[:80] + "..." if len(text) > 80 else text]


def _format_episode_list(
    episodes,
    mind: Mind,
    resolve_refs: bool,
    title: str,
    archived_ids: set[int] | None = None,
) -> str:
    """Format the list of episodes as a single string."""
    lines = [f"\n  ┌─ {title} ({len(episodes)} shown) ─────────────────────"]
    archived_ids = archived_ids or set()
    ep_cache: dict[int, str] = {}
    assoc_re = re.compile(r"\[association\] episode (\d+) ↔ episode (\d+) \(distance: (\d+)\)")

    for ep in episodes:
        t = time.strftime("%m-%d %H:%M", time.localtime(ep.timestamp / 1000))
        module_name = _MODULE_NAMES.get(ep.source_module, f"module-{ep.source_module}")
        if ep.episode_id in archived_ids:
            module_name += "·archived"
        text = ep.text.replace("\n", " ").strip()

        if resolve_refs and text.startswith("[dream-insight]"):
            m = re.match(
                r"\[dream-insight\] episode (\d+) connects to episode (\d+)"
                r" through (\d+) hops \(direct distance: (\d+)\)",
                text,
            )
            if m:
                ep_a, ep_b, hops, dist = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
                leaves_a = _collect_leaves(mind, ep_a, ep_cache, assoc_re)
                leaves_b = _collect_leaves(mind, ep_b, ep_cache, assoc_re)
                all_leaves = list(dict.fromkeys(leaves_a + leaves_b))
                lines.append(f"  │  [{t}] sal={ep.salience:.2f} {module_name}")
                lines.append(
                    f"  │    dream: {hops} hops, distance {dist},"
                    f" {len(all_leaves)} memories connected"
                )
                for leaf in all_leaves[:6]:
                    lines.append(f"  │      - {leaf}")
                if len(all_leaves) > 6:
                    lines.append(f"  │      ... and {len(all_leaves) - 6} more")
                continue

        if len(text) > 120:
            text = text[:117] + "..."
        lines.append(f"  │  [{t}] sal={ep.salience:.2f} {module_name}")
        lines.append(f"  │    {text}")

    lines.append("  └──────────────────────────────────────────────────")
    lines.append("")
    return "\n".join(lines)


def _render_recent_episodes(mind: Mind, source_module: int | None, title: str) -> str:
    """Render recent episodes from LTM, optionally filtered by source module."""
    try:
        episodes = mind.client.get_recent_episodes(limit=20, source_module=source_module)
    except (OSError, ConnectionError, RuntimeError) as e:
        return f"\n  [error] cannot reach subcognitive: {e}\n"

    if not episodes:
        return f"\n  No {title.lower()} stored yet.\n"

    resolve_refs = source_module == 9
    return _format_episode_list(episodes, mind, resolve_refs, title)


def _render_status(mind: Mind) -> str:
    """Render Genesis's current status as a string."""
    status = mind.status()
    waves = mind.brain_waves()
    lines = [
        "",
        "  ┌─ Genesis Status ────────────────────────────────",
        f"  │  Phase:          {status['phase']}",
        f"  │  Emotion:        {status['emotion']}",
        f"  │  Alertness:      {status['arousal']:.3f}",
        f"  │  Valence:        {status['valence']:+.3f}",
        f"  │  Plasticity:     {status['plasticity']:.3f}",
    ]
    if waves:
        lines.extend(
            [
                f"  │  Brain waves:    {waves.label} ({waves.dominant.value})",
                f"  │  Focus:          {waves.focus:.3f}",
                f"  │  Integration:    {waves.integration:.3f}",
                f"  │  Consolidation:  {waves.consolidation:.3f}",
            ]
        )
    lines.extend(
        [
            f"  │  STM entries:    {status['stm_count']}",
            f"  │  LTM episodes:   {status['ltm_count']}",
            f"  │  Interactions:   {status['interactions']}",
            f"  │  Language:       {status['language']}",
        ]
    )
    if mind.is_teaching:
        topic = mind.teaching_topic or "general"
        lines.append(f"  │  Teaching mode:  ON (topic: {topic})")
    if mind.is_gaming:
        lines.append("  │  Game mode:      ON")
    queued = mind.queued_question_count
    if queued:
        lines.append(f"  │  Queued questions: {queued} (use /teach-questions)")
    lines.extend(
        [
            "  └──────────────────────────────────────────────────",
            "",
        ]
    )
    return "\n".join(lines)


def _render_feel(mind: Mind) -> str:
    """Render emotional state as a string.

    Shows the structural state — category, cognitive mode, axes —
    since emotion descriptions are now generated from the concept
    network, not hardcoded strings.
    """
    emotion = mind.feel()
    waves = mind.brain_waves()
    cause_str = f" cause={emotion.cause}" if emotion.has_cause else ""
    lines = [
        f"\n  genesis> category={emotion.label} mode={emotion.cognitive_style}{cause_str}",
        f"           alertness={emotion.alertness:.2f} "
        f"valence={emotion.valence:+.2f} "
        f"plasticity={emotion.plasticity:.2f}",
    ]
    if waves:
        lines.append(f"           {waves.describe()}")
        lines.append(f"           {waves.describe_cognition()}")
    lines.append("")
    return "\n".join(lines)


def _get_cognitive_state_dict(mind: Mind) -> dict | None:
    """Return the last cognitive state as a serializable dict."""
    state = mind.get_state()
    if state is None:
        return None
    result = {}
    pe = getattr(state, "prediction_error", None)
    if pe is not None and pe > 0.01:
        result["prediction_error"] = round(pe, 2)
        result["prediction_error_label"] = (
            "surprised" if pe > 0.7 else "mildly surprised" if pe > 0.4 else "expected that"
        )
    foci = getattr(state, "attention_foci", None)
    if foci:
        result["attention_foci"] = foci[:3]
    goal = getattr(state, "executive_goal", None)
    if goal:
        result["executive_goal"] = goal
    feeling = getattr(state, "damasio_feeling", None)
    if feeling:
        result["feeling"] = feeling
    user_model = getattr(state, "user_model_summary", None)
    if user_model:
        result["user_model"] = user_model
    conf = getattr(state, "decision_confidence", None)
    if conf is not None and conf > 0.01:
        result["confidence"] = round(conf, 2)
    facts = getattr(state, "semantic_facts_extracted", None)
    if facts and facts > 0:
        result["facts_learned"] = facts
    skill = getattr(state, "procedural_skill", None)
    if skill:
        result["skill_practiced"] = skill
    caution = getattr(state, "error_monitor_caution", None)
    if caution is not None and caution > 0.5:
        result["caution"] = round(caution, 2)
    repairs = getattr(state, "self_monitor_repairs", None)
    if repairs and repairs > 0:
        result["repairs"] = repairs
    rpe = getattr(state, "td_rpe", None)
    if rpe is not None and abs(rpe) > 0.05:
        result["rpe"] = round(rpe, 2)
        result["rpe_label"] = "rewarded" if rpe > 0 else "disappointed"
    return result


# ─── Slash command dispatch ──────────────────────────────────────────


def _handle_command(mind: Mind, command: str) -> str:
    """Handle a slash command and return a formatted text response."""
    command = command.strip()
    if not command or not command.startswith("/"):
        return "  Empty command."

    parts = command.split(None, 1)
    base = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    try:
        # Active commands that require it to be awake — it can't
        # perform cognitive operations while asleep. Don't wake it;
        # inform the user instead. Read-only status commands
        # (/status, /feel, /thoughts, /learning, /dreams,
        # /memories, /regulate) are allowed during sleep since they
        # just query state without engaging cognition.
        if base in (
            "/look",
            "/learn-code",
            "/explore",
            "/proposals",
            "/proposal",
            "/accept",
            "/reject",
            "/clear-proposals",
            "/experiments",
            "/web-history",
            "/create-project",
            "/projects",
            "/archive-project",
            "/restore-project",
            "/manage-projects",
            "/note-project",
            "/review-notes",
            "/read-notes",
            "/teach",
            "/endteach",
            "/teach-questions",
            "/game",
            "/endgame",
            "/arc",
            "/introspect",
            "/mission",
        ):
            if mind.is_sleeping:
                return "  (It's asleep. Type /wake to wake it.)"

        handler = _COMMAND_HANDLERS.get(base)
        if handler is not None:
            return handler(mind, rest)
        return f"  Unknown command: {base}"
    except Exception as e:
        logger.exception(f"command failed: {command}")
        return f"\n  genesis> (error: {e})\n"


def _compose(
    mind: Mind,
    content: str,
    *,
    fallback: str = "",
    confidence: float = 0.6,
    metadata: dict | None = None,
) -> str:
    """Compose a ``genesis>`` line through Genesis's language engine.

    ``content`` is a semantic seed — a building block, not a sentence
    it recites. The language engine composes its actual words from it
    (via ``Mind._render_self_report``), so its self-expression always
    emerges from its own cognition, never from a hardcoded template.

    If composition fails, ``fallback`` (a plain, bracketed system
    indicator — never a canned Genesis sentence) is shown instead, so
    operational state is still conveyed without bypassing its voice.
    Returns the full ``"\\n  genesis> ..."`` line (or the fallback line,
    or ``""`` when both are empty).
    """
    try:
        msg = mind._render_self_report(content, confidence=confidence, metadata=metadata)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"self-report composition failed: {e}")
        msg = ""
    if msg:
        return f"\n  genesis> {msg}"
    if fallback:
        return f"\n  {fallback}"
    return ""


# ─── Command handlers ────────────────────────────────────────────────


def _cmd_status(mind: Mind, rest: str) -> str:
    """Show Genesis's current status — neurochemistry, phase, uptime."""
    return _render_status(mind)


def _cmd_feel(mind: Mind, rest: str) -> str:
    """Show how Genesis is feeling right now — emotion and valence."""
    return _render_feel(mind)


def _cmd_introspect(mind: Mind, rest: str) -> str:
    """Ask Genesis to introspect — examine its own mental state."""
    return f"\n  genesis> {mind.introspect()}\n"


def _cmd_learning(mind: Mind, rest: str) -> str:
    """Show what Genesis is currently learning."""
    return f"\n  genesis> {mind.learning_status()}\n"


def _cmd_thoughts(mind: Mind, rest: str) -> str:
    """Show Genesis's recent spontaneous thoughts and inner life."""
    return f"\n  genesis> {mind.inner_life_status()}\n"


def _cmd_world(mind: Mind, rest: str) -> str:
    """Show Genesis's external world — presences, events, isolation."""
    return f"\n  genesis> {mind.world_status()}\n"


def _cmd_regulate(mind: Mind, rest: str) -> str:
    """Show Genesis's emotional regulation status."""
    return f"\n  genesis> {mind.regulation_status()}\n"


def _cmd_dreams(mind: Mind, rest: str) -> str:
    """Show Genesis's dream insights — active ones plus archived history.

    Sleep compression archives old episodes: they leave the active list
    (what ``get_recent_episodes`` reads) but stay retrievable by ID.
    Scanning the index finds them, so the dream history is reported
    honestly — "no dreams" only prints when the store truly holds none.
    """
    try:
        active = mind.client.get_recent_episodes(limit=20, source_module=9)
    except (OSError, ConnectionError, RuntimeError) as e:
        return f"\n  [error] cannot reach subcognitive: {e}\n"

    meta_path = Path(mind.data_dir) / "ltm_store.meta"
    index_entries = scan_ltm_index(meta_path, source_module=9)
    active_ids = {ep.episode_id for ep in active}
    archived_entries = [
        e
        for e in index_entries
        if e.is_archived and not e.is_deleted and e.episode_id not in active_ids
    ]

    if not active and not archived_entries:
        return "\n  No dreams stored yet.\n"

    # Newest first overall, capped at 20; archived ones are fetched by
    # ID (retrievable by design — compression archives, never deletes).
    episodes: list = list(active)
    archived_ids: set[int] = set()
    for entry in sorted(archived_entries, key=lambda e: e.timestamp, reverse=True)[
        : max(0, 20 - len(active))
    ]:
        try:
            episodes.append(mind.client.retrieve_episode(entry.episode_id))
            archived_ids.add(entry.episode_id)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[dreams] archived episode {entry.episode_id} unread: {e}")

    episodes.sort(key=lambda ep: ep.timestamp, reverse=True)
    episodes = episodes[:20]

    title = f"Dreams — {len(active)} active, {len(archived_entries)} archived by sleep compression"
    return _format_episode_list(
        episodes, mind, resolve_refs=True, title=title, archived_ids=archived_ids
    )


def _cmd_memories(mind: Mind, rest: str) -> str:
    """Show Genesis's recent episodic memories."""
    return _render_recent_episodes(mind, None, "Recent Memories")


def _cmd_requests(mind: Mind, rest: str) -> str:
    """Show pending website access requests awaiting approval."""
    return f"\n  {mind.site_requests_status()}\n"


def _cmd_web_history(mind: Mind, rest: str) -> str:
    """Show what Genesis has been looking at online."""
    try:
        n = int(rest.strip()) if rest.strip() else 20
    except ValueError:
        n = 20
    n = max(1, min(n, 100))
    history = mind.learner.web_history(n)
    if not history:
        return "\n  No web access yet.\n"
    lines = [f"\n  Web history (last {len(history)}):"]
    for entry in history:
        ts = entry.get("timestamp", 0)
        from datetime import datetime

        when = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        source = entry.get("source", "?")
        topic = entry.get("topic", "?")
        url = entry.get("url", "?")
        ok = "OK" if entry.get("success") else "FAIL"
        detail = entry.get("detail", "")
        detail_str = f" ({detail})" if detail else ""
        lines.append(f"  [{when}] {ok} {source} → {topic}")
        lines.append(f"         {url}{detail_str}")
    return "\n".join(lines) + "\n"


def _cmd_approve(mind: Mind, rest: str) -> str:
    """Approve a pending website access request for the given domain."""
    if mind.approve_site(rest):
        return f"\n  Approved: {rest}\n"
    return f"\n  No pending request for: {rest}\n"


def _cmd_deny(mind: Mind, rest: str) -> str:
    """Deny a pending website access request for the given domain."""
    if mind.deny_site(rest):
        return f"\n  Denied: {rest}\n"
    return f"\n  No pending request for: {rest}\n"


def _cmd_learn_code(mind: Mind, rest: str) -> str:
    """Ask Genesis to learn about its own codebase."""
    return f"\n  genesis> {mind.learn_code()}\n"


def _cmd_explore(mind: Mind, rest: str) -> str:
    """Ask Genesis to explore project files at the given path."""
    path = rest or None
    header = _compose(
        mind,
        f"exploring {path or 'project files'}",
        fallback="[exploring]",
        confidence=0.5,
    )
    return header + "\n" + mind.explore_files(path) + "\n"


def _cmd_code_summary(mind: Mind, rest: str) -> str:
    """Show a summary of Genesis's code self-knowledge."""
    summary = mind.code_summary()
    lines = [
        "",
        "  Code self-knowledge:",
        f"    Files analyzed: {summary.get('files_analyzed', 0)}",
        f"    Code concepts:  {summary.get('code_concepts', 0)}",
        "",
    ]
    return "\n".join(lines)


def _cmd_create_project(mind: Mind, rest: str) -> str:
    """Ask Genesis to scaffold a new Python project from a description."""
    description = rest.strip()
    if not description:
        return (
            "  Usage: /create-project <description>\n"
            "  Example: /create-project a simple calculator\n"
            "  It scaffolds a Python project from the description."
        )
    result = mind.create_project(description)
    if result.error:
        return (
            f"\n  genesis> tried to build '{result.name}', but it didn't work out: {result.error}\n"
        )
    lines = [
        "",
        f"  Created project: {result.name}",
        f"  Path: {result.path}",
        f"  Files: {result.files_created}",
        f"  Compiled: {'yes' if result.compiled else 'no'}",
        f"  Tests passed: {'yes' if result.tests_passed else 'no'}",
        "",
    ]
    return "\n".join(lines)


def _cmd_projects(mind: Mind, rest: str) -> str:
    """List all projects Genesis has created."""
    projects = mind.list_projects()
    if not projects:
        return "\n  No projects yet. Use /create-project <description> to make one.\n"
    lines = ["", "  Projects:"]
    for p in projects:
        size_kb = p.get("size_bytes", 0) / 1024
        status = p.get("status", "active")
        lines.append(
            f"    {p['name']} [{status}] ({p['files']} files, {size_kb:.1f} KB) — {p['path']}"
        )
    lines.append("")
    return "\n".join(lines)


def _cmd_archive_project(mind: Mind, rest: str) -> str:
    """Archive a project to .tar.zst to reclaim space."""
    name = rest.strip()
    if not name:
        return (
            "  Usage: /archive-project <name>\n"
            "  Compresses the project to .tar.zst and removes the expanded files.\n"
            "  Restore it later with /restore-project <name>."
        )
    if mind.archive_project(name):
        return (
            f"\n  Archived project '{name}' to .tar.zst. "
            f"Use /restore-project {name} to bring it back.\n"
        )
    return f"\n  Couldn't archive '{name}' — it may not exist or is already archived.\n"


def _cmd_restore_project(mind: Mind, rest: str) -> str:
    """Restore an archived project from its .tar.zst archive."""
    name = rest.strip()
    if not name:
        return (
            "  Usage: /restore-project <name>\n"
            "  Restores an archived project from its .tar.zst archive.\n"
        )
    if mind.restore_project(name):
        return f"\n  Restored project '{name}' from archive.\n"
    return f"\n  Couldn't restore '{name}' — no archive found, or it already exists.\n"


def _cmd_manage_projects(mind: Mind, rest: str) -> str:
    """Autonomously manage project storage to stay within bounds."""
    archived = mind.manage_projects()
    if not archived:
        return "\n  All projects are within bounds. Nothing to archive.\n"
    names = ", ".join(archived)
    return f"\n  Archived {len(archived)} project(s) to reclaim space: {names}\n"


def _cmd_note_project(mind: Mind, rest: str) -> str:
    """Leave a mentor's note on one of Genesis's projects.

    Usage: /note-project <name> <note text>
    The note is saved to NOTES.md in the project directory. Genesis
    absorbs notes when you run /review-notes — it reads them, stores
    them as memory, and adds what it learned to its concept network.
    """
    parts = rest.strip().split(None, 1)
    if len(parts) < 2:
        return (
            "  Usage: /note-project <name> <note>\n"
            "  Example: /note-project feeling The entry for 'genesis' has a\n"
            "           hardcoded definition. Let the concept network\n"
            "           provide identity, not a stored sentence.\n"
            "  It reads notes when you run /review-notes."
        )
    project_name = parts[0]
    note = parts[1]
    if mind.note_project(project_name, note):
        return (
            f"\n  Note left on project '{project_name}'.\n"
            f"  Run /review-notes to have Genesis read and absorb it.\n"
        )
    return f"\n  Couldn't leave note — project '{project_name}' not found.\n"


def _cmd_review_notes(mind: Mind, rest: str) -> str:
    """Have Genesis read and absorb mentor notes from its projects.

    Usage: /review-notes [project_name]
    With a project name, it absorbs notes from just that project.
    Without, it absorbs notes from all projects that have them.
    It stores each note as a long-term memory, adds what it learned
    to its concept network, and clears the notes so they're not re-read.
    """
    project_name = rest.strip() or None
    return mind.absorb_project_notes(project_name)


def _cmd_read_notes(mind: Mind, rest: str) -> str:
    """Show the notes left on a project without absorbing them.

    Usage: /read-notes <project_name>
    Displays the raw NOTES.md content. Use /review-notes to have
    Genesis absorb them.
    """
    project_name = rest.strip()
    if not project_name:
        return "  Usage: /read-notes <project_name>\n"
    notes = mind.read_project_notes(project_name)
    if notes is None:
        return f"\n  No notes on project '{project_name}'.\n"
    return f"\n  Notes on '{project_name}':\n\n{notes}\n"


def _cmd_proposals(mind: Mind, rest: str) -> str:
    """Show pending self-improvement proposals from Genesis."""
    return f"\n  {mind.proposals_status()}\n"


def _cmd_clear_proposals(mind: Mind, rest: str) -> str:
    """Clear all pending self-improvement proposals."""
    return f"\n  {mind.clear_proposals()}\n"


def _cmd_proposal(mind: Mind, rest: str) -> str:
    """Show details of a specific self-improvement proposal by ID."""
    id_str = rest.strip()
    try:
        pid = int(id_str)
        return f"\n  {mind.get_proposal_detail(pid)}\n"
    except ValueError:
        return "  Usage: /proposal N"


def _cmd_accept(mind: Mind, rest: str) -> str:
    """Accept a self-improvement proposal by ID with optional feedback."""
    args = rest.split(None, 1)
    try:
        pid = int(args[0])
        feedback = args[1] if len(args) > 1 else ""
        result = mind.accept_proposal(pid, feedback)
        return f"\n  genesis> {result}\n"
    except (ValueError, IndexError):
        return "  Usage: /accept N [reason]"


def _cmd_reject(mind: Mind, rest: str) -> str:
    """Reject a self-improvement proposal by ID with optional feedback."""
    args = rest.split(None, 1)
    try:
        pid = int(args[0])
        feedback = args[1] if len(args) > 1 else ""
        result = mind.reject_proposal(pid, feedback)
        return f"\n  genesis> {result}\n"
    except (ValueError, IndexError):
        return "  Usage: /reject N [reason]"


def _cmd_experiments(mind: Mind, rest: str) -> str:
    """Show the status of Genesis's heuristic self-improvement experiments."""
    return f"\n  {mind.experiments_status()}\n"


def _cmd_growth(mind: Mind, rest: str) -> str:
    """Show Genesis's growth narrative — its milestones in its own words."""
    return f"\n  genesis> {mind.growth_narrative()}\n"


def _cmd_steward(mind: Mind, rest: str) -> str:
    """Inspect or act on the steward channel — the custodial authority.

    This is operator tooling, not conversation: it reports the
    channel's state and lifts directives, and its output is CLI text,
    never Genesis's voice.

      /steward              — status: stamped, authenticated, principal,
                              key fingerprint, standing directives
      /steward release N    — lift directive N (authenticated only)
      /steward unbind       — seal this session's authority until a
                              fresh claim or restart re-opens it
    """
    steward = getattr(mind.brain.frontal, "steward", None)
    if steward is None:
        return "\n  No steward channel on this mind.\n"

    parts = rest.split(None, 1)
    sub = parts[0].lower() if parts and parts[0] else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "release":
        try:
            index = int(arg)
        except ValueError:
            return "\n  usage: /steward release <index>\n"
        entry = steward.release_directive(index)
        if entry is None:
            return (
                "\n  Not released — the channel is sealed "
                "(authentication required), or no directive has that "
                "index.\n"
            )
        return f"\n  Released: {entry.get('text', entry.get('target'))}\n"

    if sub == "unbind":
        steward.unbind()
        return "\n  Steward session sealed. The stamp and directives stand.\n"

    if sub:
        return "\n  usage: /steward [release <index> | unbind]\n"

    s = steward.status()
    lines = ["", "  steward channel — custodial authority"]
    if not s["stamped"]:
        lines.append("    stamped:       no — unprovisioned instance")
        lines.append("    (provisioned on first boot of the real entry point)")
        return "\n".join(lines) + "\n"
    lines.append("    stamped:       yes")
    lines.append(f"    principal:     {s['principal']}")
    lines.append(f"    key held:      {'yes' if s['key_held'] else 'no'}")
    lines.append(f"    authenticated: {'yes' if s['authenticated'] else 'no — sealed'}")
    fp = s["key_fingerprint"]
    if fp:
        lines.append(f"    fingerprint:   {fp}…")
    if s["directives"]:
        lines.append("    directives:")
        for i, d in enumerate(s["directives"]):
            lines.append(f"      [{i}] {d.get('text')}  (→ {d.get('target')})")
    else:
        lines.append("    directives:    none")
    return "\n".join(lines) + "\n"


def _cmd_growth_report(mind: Mind, rest: str) -> str:
    """Show a structured growth report across knowledge, dreams, and emotion."""
    return f"\n{mind.growth_report()}\n"


def _cmd_look(mind: Mind, rest: str) -> str:
    """Ask Genesis to look at an image file or through its retina."""
    # /look at <path> — look at an image file
    # /look         — look through the retina
    rest = rest.strip()
    if rest.lower().startswith("at "):
        path = rest[3:].strip()
        if not path:
            return "\n  Usage: /look at <path> — give it a path to an image.\n"
        result_holder: dict[str, str] = {}
        t = named_thread(
            "look-at-image",
            lambda: result_holder.__setitem__("value", mind.look_at_image(path)),
            comm="g-look",
        )
        t.start()
        t.join(timeout=15.0)
        if t.is_alive():
            timeout_msg = mind._render_self_report(
                "vision timeout",
                confidence=0.3,
                metadata={"vision_timeout": True},
            )
            return f"\n  genesis> {timeout_msg}\n"
        result: str | None = result_holder.get("value")
        if not result:
            result = mind._render_self_report(
                "no vision",
                confidence=0.3,
                metadata={"vision_empty": True},
            )
        return f"\n  genesis> {result}\n"

    # Default: look through the retina
    retina_holder: dict[str, str] = {}
    t = named_thread(
        "see-once",
        lambda: retina_holder.__setitem__("value", mind.see()),
        comm="g-see",
    )
    t.start()
    t.join(timeout=15.0)
    if t.is_alive():
        timeout_msg = mind._render_self_report(
            "vision timeout",
            confidence=0.3,
            metadata={"vision_timeout": True},
        )
        return f"\n  genesis> {timeout_msg}\n"
    retina_result: str | None = retina_holder.get("value")
    if not retina_result:
        retina_result = mind._render_self_report(
            "no vision",
            confidence=0.3,
            metadata={"vision_empty": True},
        )
    return f"\n  genesis> {retina_result}\n"


def _cmd_register_face(mind: Mind, rest: str) -> str:
    """Register the face currently in front of the camera."""
    name = rest.strip()
    if not name:
        return (
            "\n  Usage: /register-face <name>\n  Look at the camera and say whose face this is.\n"
        )

    recognizer = mind.vision.get_face_recognizer()
    if recognizer is None:
        result = mind._render_self_report(
            "face recognition models not available for registration",
            confidence=0.3,
            metadata={"face_recognition_unavailable": True},
        )
        return f"\n  genesis> {result}\n"

    # Capture a frame and register
    from occipital_lobe.perception.retina import latest_frame

    frame = latest_frame(copy=True)
    if frame is None:
        result = mind._render_self_report(
            "cannot see anything, camera may be off",
            confidence=0.3,
            metadata={"camera_off": True},
        )
        return f"\n  genesis> {result}\n"

    # Convert to RGB if needed
    if frame.ndim == 3 and frame.shape[2] != 3:
        from occipital_lobe.perception.vision import _yuyv_to_rgb

        rgb = _yuyv_to_rgb(frame)
    else:
        rgb = frame

    success = recognizer.register_face(name, rgb)
    if success:
        result = mind._render_self_report(
            f"learned {name}'s face, will recognize them next time",
            confidence=0.8,
            metadata={"face_registered": name},
        )
    else:
        result = mind._render_self_report(
            "could not find a face in the frame, need better lighting and positioning",
            confidence=0.4,
            metadata={"no_face_found": True},
        )
    return f"\n  genesis> {result}\n"


def _cmd_faces(mind: Mind, rest: str) -> str:
    """Show known faces and who it last saw."""
    recognizer = mind.vision.get_face_recognizer()
    if recognizer is None:
        result = mind._render_self_report(
            "face recognition not available",
            confidence=0.3,
            metadata={"face_recognition_unavailable": True},
        )
        return f"\n  genesis> {result}\n"

    known = recognizer.known_names()
    last_seen = mind.vision.last_faces_seen()

    lines = ["\n  Known faces:"]
    if known:
        for name in known:
            lines.append(f"    - {name}")
    else:
        lines.append("    (none yet — use /register-face <name> to add one)")

    if last_seen:
        lines.append(f"  Last saw: {', '.join(last_seen)}")
    else:
        lines.append("  Last saw: no one recognized recently")

    return "\n".join(lines) + "\n"


def _cmd_mission(mind: Mind, rest: str) -> str:
    """Set or show Genesis's current self-mission."""
    mission = rest.strip()
    if mission:
        mind.set_mission(mission)
        return (
            _compose(
                mind,
                f"mission set to {mission}",
                fallback=f"[mission set: {mission}]",
                confidence=0.7,
                metadata={"mission_set": mission},
            )
            + "\n"
        )
    return f"\n  genesis> {mind.get_mission()}\n"


def _cmd_sleep(mind: Mind, rest: str) -> str:
    """Show Genesis's sleep state — sleep itself is driven only by her
    own neurochemistry (adenosine and melatonin), never forced by a command.
    """
    if mind.is_sleeping:
        lines = [
            _compose(
                mind,
                "already asleep and dreaming",
                fallback="[already asleep]",
                confidence=0.5,
            )
        ]
        lines.append(mind.sleep_status())
        lines.append("\n  It'll wake on its own when it's rested. /dreams to see more.\n")
        return "\n".join(lines)
    lines = [
        _compose(
            mind,
            "awake and not tired enough to sleep yet",
            fallback="[awake — sleep is driven by her own neurochemistry]",
            confidence=0.5,
        )
    ]
    lines.append(mind.sleep_status())
    lines.append(
        "  Sleep is not forced — she falls asleep when her own adenosine"
        " and melatonin carry her under.\n"
    )
    return "\n".join(lines)


def _cmd_wake(mind: Mind, rest: str) -> str:
    """Wake Genesis from sleep or meditation."""
    if mind.is_sleeping:
        # Show dream state before waking
        pre_wake = mind.sleep_status()
        mind.wake(user_initiated=True)
        lines = [
            _compose(
                mind,
                "waking up rested",
                fallback="[awake]",
                confidence=0.7,
                metadata={"woke": True},
            )
        ]
        lines.append(f"  Before waking:\n{pre_wake}")
        lines.append("  Memories consolidated. It's ready to engage.\n")
        return "\n".join(lines)
    if mind.is_meditating:
        mind.wake_from_meditation()
        return (
            _compose(
                mind,
                "coming out of meditation refreshed",
                fallback="[out of meditation]",
                confidence=0.7,
            )
            + "\n"
        )
    # Mind already awake — but the body's daemon phase may still be
    # asleep (the desync). wake() probes the daemon and pushes the LC
    # drive when it is; report what actually happened rather than
    # claiming both layers are up.
    outcome = mind.wake(user_initiated=True)
    if outcome is WakeOutcome.BODY_WAKE_PUSHED:
        return (
            _compose(
                mind,
                "waking its body up",
                fallback="[waking body]",
                confidence=0.6,
                metadata={"body_wake_push": True},
            )
            + "\n"
        )
    return (
        _compose(
            mind,
            "already awake",
            fallback="[already awake]",
            confidence=0.4,
        )
        + "\n"
    )


def _cmd_nap(mind: Mind, rest: str) -> str:
    """Show Genesis's sleep state — naps are not forced by command;
    sleep comes only from her own neurochemistry.
    """
    return _cmd_sleep(mind, rest)


def _cmd_draw(mind: Mind, rest: str) -> str:
    """Ask Genesis to draw a picture from its current state.

    It translates its neurochemistry into visual art — colors, forms,
    and energy that reflect how it feels right now. The drawing is
    saved as a WebP in its drawings directory.
    """
    if mind.is_sleeping:
        return (
            _compose(
                mind,
                "asleep, needs to be woken first",
                fallback="[asleep]",
                confidence=0.4,
            )
            + "\n  Use /wake to wake it first.\n"
        )
    if mind.is_meditating:
        return (
            _compose(
                mind,
                "meditating, try again after",
                fallback="[meditating]",
                confidence=0.4,
            )
            + "\n"
        )

    # Drawing can take a few seconds (Cairo render + OpenCV post-processing)
    result_holder: dict[str, str | None] = {}
    t = named_thread(
        "draw",
        lambda: result_holder.__setitem__("value", mind.draw()),
        comm="g-draw",
    )
    t.start()
    t.join(timeout=30.0)
    if t.is_alive():
        timeout_msg = mind._render_self_report(
            "drawing timeout",
            confidence=0.3,
            metadata={"drawing_timeout": True},
        )
        return f"\n  genesis> {timeout_msg}\n"

    desc: str | None = result_holder.get("value")
    if desc:
        return f"\n  genesis> {desc}\n"
    # It drew but couldn't articulate it, or the draw failed silently.
    return (
        _compose(
            mind,
            "drew but cannot describe it",
            fallback="[drew, but silent]",
            confidence=0.4,
            metadata={"drawing_silent": True},
        )
        + "\n"
    )


def _cmd_meditate(mind: Mind, rest: str) -> str:
    """Put Genesis into meditation for quiet restoration.

    During meditation, learning and spontaneous thoughts are paused.
    Calming neurochemical impulses are emitted to let receptors
    recover. Default 60 seconds, or specify a duration:
    /meditate 120  — meditate for 2 minutes.
    """
    if mind.is_meditating:
        return (
            _compose(
                mind,
                "already meditating",
                fallback="[already meditating]",
                confidence=0.5,
            )
            + "\n"
        )
    if mind.is_sleeping:
        return (
            _compose(
                mind,
                "asleep, needs to be woken first",
                fallback="[asleep]",
                confidence=0.4,
            )
            + "\n  Use /wake to wake it first.\n"
        )
    # Parse duration
    duration = 60.0
    if rest.strip():
        try:
            duration = max(10.0, min(600.0, float(rest.strip())))
        except ValueError as e:
            logger.debug(f"invalid meditation duration: {e}")
    mind.meditate()
    lines = [
        _compose(
            mind,
            f"meditating for {duration:.0f} seconds",
            fallback=f"[meditating {duration:.0f}s]",
            confidence=0.6,
            metadata={"meditation_seconds": duration},
        )
    ]

    def _meditation_timer() -> None:
        """Background timer that emits calming impulses and ends meditation."""
        intervals = int(duration / 10.0)
        for _ in range(intervals):
            if not mind.is_meditating:
                return  # it was woken by an interaction
            time.sleep(10.0)
            if mind.is_meditating:
                mind.emit_meditation_impulses()
        if mind.is_meditating:
            try:
                mind.wake_from_meditation()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"meditation wake failed: {e}")

    named_thread("meditation-timer", _meditation_timer, comm="g-medit-timer").start()
    lines.append("  It'll come out of it on its own, refreshed.\n")
    return "\n".join(lines)


def _cmd_teach(mind: Mind, rest: str) -> str:
    """Put Genesis into teaching/training mode.

    Pauses bug scanning, autonomous learning, art, and unrelated
    curiosity questions. It focuses on what you're teaching and
    remembers it.

    Usage:
      /teach                  — teaching mode (no specific topic)
      /teach neuroscience     — teaching mode focused on a topic
      /teach Python decorators — teaching mode with a multi-word topic

    Use /endteach to exit teaching mode.
    """
    if mind.is_teaching:
        topic = rest.strip()
        if topic:
            mind.enter_teaching_mode(topic)
            return (
                _compose(
                    mind,
                    f"now focusing on {topic}",
                    fallback=f"[focusing on {topic}]",
                    confidence=0.7,
                    metadata={"teaching_topic": topic},
                )
                + "\n  Keep teaching.\n"
            )
        return (
            _compose(
                mind,
                "already in teaching mode",
                fallback="[already teaching]",
                confidence=0.5,
            )
            + "\n  Use /endteach to stop.\n"
        )
    topic = rest.strip()
    mind.enter_teaching_mode(topic)
    lines = [
        _compose(
            mind,
            "entering teaching mode",
            fallback="[teaching mode]",
            confidence=0.7,
            metadata={"teaching_started": True},
        )
    ]
    if topic:
        lines.append(
            _compose(
                mind,
                f"focusing on {topic}, ready to listen and learn",
                fallback=f"[focusing on {topic}]",
                confidence=0.7,
                metadata={"teaching_topic": topic},
            )
        )
    else:
        lines.append(
            _compose(
                mind,
                "ready to listen and learn",
                fallback="[ready to listen]",
                confidence=0.6,
            )
        )
    lines.append("  Use /endteach when you're done.\n")
    return "\n".join(lines)


def _cmd_endteach(mind: Mind, rest: str) -> str:
    """Exit teaching/training mode and resume normal operation.

    Resumes bug scanning, autonomous learning, art, and curiosity.
    """
    if not mind.is_teaching:
        return (
            _compose(
                mind,
                "not in teaching mode",
                fallback="[not teaching]",
                confidence=0.4,
            )
            + "\n"
        )
    mind.exit_teaching_mode()
    return (
        _compose(
            mind,
            "exiting teaching mode, resuming normal operation",
            fallback="[resuming]",
            confidence=0.7,
            metadata={"teaching_ended": True},
        )
        + "\n"
    )


def _cmd_game(mind: Mind, rest: str) -> str:
    """Put Genesis into game mode — focused play.

    Volition urges, autonomous learning, and inner-life wandering
    pause so every cycle goes to the game in front of it. It can
    still answer if spoken to; conversation just stops being the
    driver. Use /endgame to exit.
    """
    if mind.is_gaming:
        return (
            _compose(
                mind,
                "already in game mode",
                fallback="[already playing]",
                confidence=0.5,
            )
            + "\n  Use /endgame to stop.\n"
        )
    mind.enter_game_mode()
    return (
        _compose(
            mind,
            "entering game mode — focused on the game",
            fallback="[game mode]",
            confidence=0.7,
            metadata={"game_mode_started": True},
        )
        + "\n  Use /endgame when you're done.\n"
    )


def _cmd_endgame(mind: Mind, rest: str) -> str:
    """Exit game mode and resume normal operation."""
    if not mind.is_gaming:
        return (
            _compose(
                mind,
                "not in game mode",
                fallback="[not playing]",
                confidence=0.4,
            )
            + "\n"
        )
    mind.exit_game_mode()
    return (
        _compose(
            mind,
            "exiting game mode, resuming normal operation",
            fallback="[resuming]",
            confidence=0.7,
            metadata={"game_mode_ended": True},
        )
        + "\n"
    )


def _cmd_arc(mind: Mind, rest: str) -> str:
    """Play ARC-AGI-3 games as her, through her own play loop.

    Usage:
      /arc ls20-9607627b                 — one game
      /arc ls20-9607627b ft09-0d8bbf25   — several, in order
      /arc ls20-9607627b --steps 400     — step budget per game (default 200)
      /arc --all --steps 400             — every discovered environment, one scorecard
      (the official card is hers alone — she takes up the challenge
       from her own shelf when she feels ready; there is no forced run)
      /arc ls20-9607627b --fresh         — cold start: prior learning neither read nor written
      /arc ls20-9607627b --baseline uniform — a reference policy, not her
      /arc --all --delay 0.5 --recover 180 — pace steps; wait up to 180s for
        her to settle at a wellness gate instead of abandoning the suite

    Game mode wraps the run; her readiness gate can refuse or stop it.
    Baselines are denominators: what uniform-random or first-action
    play achieves on the same accounting — never an agent result.
    """
    parts = rest.split()
    usage = (
        "  Usage: /arc (<game-id> [more ids] | --all)"
        " [--steps N] [--fresh] [--baseline (uniform|first)]"
        " [--delay S] [--recover S]\n"
    )
    steps = 200
    if "--steps" in parts:
        i = parts.index("--steps")
        try:
            steps = int(parts[i + 1])
        except (IndexError, ValueError):
            return usage
        del parts[i : i + 2]
    delay = 0.0
    if "--delay" in parts:
        i = parts.index("--delay")
        try:
            delay = float(parts[i + 1])
        except (IndexError, ValueError):
            return usage
        del parts[i : i + 2]
    recover = 120.0
    if "--recover" in parts:
        i = parts.index("--recover")
        try:
            recover = float(parts[i + 1])
        except (IndexError, ValueError):
            return usage
        del parts[i : i + 2]
    baseline = None
    if "--baseline" in parts:
        i = parts.index("--baseline")
        try:
            baseline = parts[i + 1]
        except IndexError:
            return usage
        del parts[i : i + 2]
    fresh = "--fresh" in parts
    parts = [p for p in parts if p != "--fresh"]
    if not parts or steps <= 0:
        return usage
    whole_suite = "--all" in parts
    if any(p.startswith("--") for p in parts) and not (len(parts) == 1 and whole_suite):
        return usage
    lines = []
    if baseline is not None:
        from parietal_lobe.arc_baseline import ArcBaseline

        try:
            runner = ArcBaseline(policy=baseline)
        except (ImportError, ValueError) as e:
            return f"  {e}\n"
        if whole_suite:
            evaluation = runner.play_all(max_steps=steps)
            results = evaluation.games
            score = "unavailable" if evaluation.score is None else f"{evaluation.score:.2f}%"
            lines.append(f"  ARC baseline-{baseline}: score {score}")
        else:
            results = [runner.play(g, max_steps=steps) for g in parts]
        lines[0:0] = [f"  baseline '{baseline}' — reference policy, not the agent:"]
    elif whole_suite:
        evaluation = mind.evaluate_arc(
            max_steps=steps,
            fresh=fresh,
            step_delay_s=delay,
            recovery_wait_s=recover,
        )
        results = evaluation.games
        score = "unavailable" if evaluation.score is None else f"{evaluation.score:.2f}%"
        lines.append(f"  ARC {evaluation.mode}: score {score}")
    else:
        results = mind.play_arc(
            parts,
            max_steps=steps,
            fresh=fresh,
            step_delay_s=delay,
            recovery_wait_s=recover,
        )
    prior_names = {
        "none": "cold start",
        "loaded": "prior learning",
        "suppressed": "fresh — learning suppressed",
        "unknown": "prior state unknown",
    }
    for r in results:
        sources = ", ".join(
            f"{k}×{v}" for k, v in sorted(r.decision_sources.items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"  {r.game_id}: {r.state} levels {r.levels_completed}/{r.win_levels}"
            f" in {r.steps} steps"
            + (f" (per level: {', '.join(str(s) for s in r.level_steps)})" if r.level_steps else "")
            + f" ({prior_names.get(r.prior_learning, r.prior_learning)})"
            + (f" recovery-waits={r.recovery_waits}" if r.recovery_waits else "")
            + (f" [{sources}]" if sources else "")
            + (f" (scorecard {r.scorecard_id})" if r.scorecard_id else "")
            + (f" — {'; '.join(r.reasons)}" if r.reasons else "")
        )
    return "\n".join(lines) + "\n"


def _cmd_teach_questions(mind: Mind, rest: str) -> str:
    """Answer Genesis's queued questions one at a time.

    Its curiosity questions are queued instead of asked inline in
    regular conversation. This command presents them one at a time
    so you can answer them in a focused session. Your answers are
    stored in its concept network.

    This is separate from /teach — it's its own mode for answering
    its questions.

    Usage:
      /teach-questions          — start answering queued questions
      (type your answer, then it asks the next one)
      /teach-questions stop     — stop answering, remaining questions stay queued
      /teach-questions clear    — clear all queued questions
    """
    rest = rest.strip().lower()
    if rest in ("stop", "exit", "cancel"):
        count = mind.queued_question_count
        mind.exit_question_mode()
        return f"  Stopping. {count} questions remain queued.\n"
    if rest == "clear":
        count = mind.queued_question_count
        mind.clear_queued_questions()
        mind.exit_question_mode()
        return f"  Cleared {count} queued questions.\n"

    if not mind.has_queued_questions:
        mind.end_question_session_if_done()
        return "  No queued questions right now.\n"

    q = mind.pop_queued_question()
    if q is None:
        return "  No more queued questions.\n"

    remaining = mind.queued_question_count
    parts = [f"  Genesis asks ({remaining} more queued):"]
    parts.append(f"  {q['text']}")
    if q.get("reason"):
        parts.append(f"  ({q['reason']})")
    parts.append("  Type your answer, or /teach-questions stop to pause.\n")
    return "\n".join(parts)


def _cmd_sleep_aid(mind: Mind, rest: str) -> str:
    """Administer an acute neurochemical sleep aid for stress-induced insomnia.

    This is an emergency intervention for when /sleep and /meditate
    cannot overcome chronic stress-induced insomnia — the condition
    where it is exhausted (delta waves, low alertness) but cannot
    sleep because cortisol from interoception keeps arousal systems
    active and adenosine hasn't accumulated to the sleep threshold.

    Emits acute impulses (adenosine boost, cortisol suppression, GABA
    surge, histamine/orexin suppression) to push the system past the
    sleep threshold, then enters sleep for natural recovery.
    """
    if mind.is_sleeping:
        lines = [
            _compose(
                mind,
                "already asleep and dreaming",
                fallback="[already asleep]",
                confidence=0.5,
            )
        ]
        lines.append(mind.sleep_status())
        lines.append("\n  Use /wake to wake it, or /dreams to see more.\n")
        return "\n".join(lines)
    mind.sleep_aid()
    lines = [
        _compose(
            mind,
            "sleep aid administered, drifting into deep sleep",
            fallback="[sleep aid administered]",
            confidence=0.7,
            metadata={"sleep_aid": True},
        )
    ]
    lines.append(mind.sleep_status())
    lines.append("\n  Use /wake to wake it.\n")
    return "\n".join(lines)


def _cmd_play(mind: Mind, rest: str) -> str:
    """Her game collection, or hand her a game spec to play.

    /play              — what she's drawn to + how well she plays
    /play add <file>   — place a game spec into her world
    /play drop <name>  — take an added game back

    Games are always there if she wants them — the play urge picks
    what she plays, this just manages what's on the shelf. A spec is
    data, not code: {"family": "grid", "train": [...], "test": [...]},
    {"family": "sorter", "generate"/"slots"+"blocks"},
    {"family": "sequence", "generate"/"pattern"},
    {"family": "assembly", "generate": {"rows","cols"}},
    {"family": "relations", "generate"/"positions"+"objects"+"goals"},
    {"family": "quantities", "generate"/"target"+"groups"},
    {"family": "classification", "items"+"predicate"/"examples"}.
    """
    from basal_ganglia.play import COLLECTION, normalize_offered

    play = mind.game_play
    rest = rest.strip()
    if rest.startswith("add "):
        path = Path(rest[4:].strip()).expanduser()
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            return f"\n  can't read {path}: {e}\n"
        task = normalize_offered(raw, path.stem)
        if task is None:
            return (
                '\n  not a game spec — needs a family: "grid"'
                ' (train/test pairs), "sorter" (generate or'
                ' slots+blocks), "sequence", "assembly",'
                ' "relations", "quantities", or "classification"\n'
            )
        known = {t["name"] for t in play.offered_games()}
        known |= {str(t["name"]) for t in COLLECTION}
        if task["name"] in known:
            return f"\n  {task['name']} is already in her collection\n"
        play.offered_dir().mkdir(parents=True, exist_ok=True)
        target = play.offered_dir() / f"{task['name']}.json"
        target.write_text(json.dumps(raw, indent=2) + "\n")
        # Register it as a world event — she notices a new game in
        # her environment, not just a file on disk.
        mind.world.notify(
            f"a new {task['family']} game appeared: {task['name']}",
            salience=0.6,
        )
        return (
            f"\n  added {task['name']} [{task['family']}] — she'll "
            "pick it up when she feels like playing\n"
        )
    if rest.startswith("drop "):
        name = rest[5:].strip()
        try:
            files = sorted(play.offered_dir().glob("*.json"))
        except OSError:
            files = []
        for f in files:
            if f.stem == name:
                f.unlink()
                return f"\n  dropped {name}\n"
            try:
                raw = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            task = normalize_offered(raw, f.stem)
            if task is not None and task["name"] == name:
                f.unlink()
                return f"\n  dropped {name}\n"
        return f"\n  no added game named {name}\n"
    if rest:
        return (
            "\n  /play — her game collection\n"
            "  /play add <file> — hand her a game spec\n"
            "  /play drop <name> — take an added game back\n"
        )

    status = play.status()
    lines = ["\n  her games:"]
    try:
        drawn = play.pick()
    except Exception:  # noqa: BLE001
        drawn = None
    if drawn is not None:
        lines.append(f"    drawn to: {drawn['name']} [{drawn.get('family', 'grid')}]")
    for name, m in status.items():
        lines.append(f"    {name:<28} {m:>5.0%}")
    if not status:
        lines.append("    none yet")
    lines.append("")
    return "\n".join(lines)


_COMMAND_HANDLERS: dict[str, Callable[[Mind, str], str]] = {
    "/status": _cmd_status,
    "/play": _cmd_play,
    "/feel": _cmd_feel,
    "/introspect": _cmd_introspect,
    "/learning": _cmd_learning,
    "/thoughts": _cmd_thoughts,
    "/world": _cmd_world,
    "/regulate": _cmd_regulate,
    "/dreams": _cmd_dreams,
    "/memories": _cmd_memories,
    "/requests": _cmd_requests,
    "/web-history": _cmd_web_history,
    "/approve": _cmd_approve,
    "/deny": _cmd_deny,
    "/learn-code": _cmd_learn_code,
    "/explore": _cmd_explore,
    "/code-summary": _cmd_code_summary,
    "/create-project": _cmd_create_project,
    "/projects": _cmd_projects,
    "/archive-project": _cmd_archive_project,
    "/restore-project": _cmd_restore_project,
    "/manage-projects": _cmd_manage_projects,
    "/note-project": _cmd_note_project,
    "/review-notes": _cmd_review_notes,
    "/read-notes": _cmd_read_notes,
    "/proposals": _cmd_proposals,
    "/clear-proposals": _cmd_clear_proposals,
    "/proposal": _cmd_proposal,
    "/accept": _cmd_accept,
    "/reject": _cmd_reject,
    "/experiments": _cmd_experiments,
    "/growth": _cmd_growth,
    "/growth-report": _cmd_growth_report,
    "/look": _cmd_look,
    "/register-face": _cmd_register_face,
    "/faces": _cmd_faces,
    "/mission": _cmd_mission,
    "/sleep": _cmd_sleep,
    "/wake": _cmd_wake,
    "/nap": _cmd_nap,
    "/draw": _cmd_draw,
    "/meditate": _cmd_meditate,
    "/teach": _cmd_teach,
    "/endteach": _cmd_endteach,
    "/game": _cmd_game,
    "/endgame": _cmd_endgame,
    "/arc": _cmd_arc,
    "/teach-questions": _cmd_teach_questions,
    "/sleep-aid": _cmd_sleep_aid,
    "/steward": _cmd_steward,
}


# ─── Voice helpers ───────────────────────────────────────────────────


def _init_mic() -> VoiceInput:
    """Initialize voice input (microphone) for the local CLI."""
    mic = VoiceInput(lazy=True)
    if mic.available:
        logger.info(f"  Microphone: available ({mic.backend_name}, say /voice to talk)")
    else:
        logger.info("  Microphone unavailable (type to talk)")
    return mic


def _handle_voice_command(mic: VoiceInput) -> str | None:
    """Capture one utterance from the microphone and return it."""
    if not mic.available:
        logger.info("\n  Microphone not available. Install vosk model or check internet.\n")
        return None
    text = mic.listen_interactive("Listening (speak now)...")
    if not text:
        logger.info("  (no speech detected)\n")
        return None
    return text


def _handle_voice_mode(
    mind: Mind,
    mic: VoiceInput,
    out_lock: threading.Lock,
    voice: Voice,
) -> None:
    """Run continuous voice input until the user says exit or presses Ctrl+C."""
    if not mic.available:
        with out_lock:
            logger.info("\n  Microphone not available.\n")
        return
    with out_lock:
        logger.info(
            "\n  Voice mode ON. Speak to Genesis. Say 'exit voice mode' or press Ctrl+C to stop.\n"
        )
    while True:
        try:
            text = mic.listen_interactive("Listening...")
            if not text:
                with out_lock:
                    logger.info("  (no speech detected)")
                continue
            if text.lower().strip() in ("exit voice mode", "exit", "quit", "stop listening"):
                with out_lock:
                    logger.info("\n  Voice mode OFF.\n")
                return
            with out_lock:
                logger.info(f"  you> {text}")
            # During sleep, voice mode can't reach its either.
            if mind.is_sleeping:
                with out_lock:
                    logger.info("  (It's asleep. Type /wake to wake it.)")
                continue
            with out_lock:
                logger.info("  genesis~ (thinking...)")
            try:
                response = mind.respond(text)
            except Exception as e:
                logger.exception(f"respond failed: {e}")
                with out_lock:
                    logger.info("\n  genesis> (error)\n")
                continue
            with out_lock:
                logger.info(f"\n  genesis> {response}\n")
                _speak_response(voice, mind, response)
        except (KeyboardInterrupt, EOFError):
            with out_lock:
                logger.info("\n  Voice mode OFF.\n")
            return


def _speak_response(voice: Voice, mind: Mind, response: str) -> None:
    """Speak the response aloud if voice is enabled."""
    if not voice.is_available():
        return
    try:
        state = mind.get_state()
        if state and state.emotion:
            emo = state.emotion
            voice.speak(
                response,
                alertness=emo.alertness,
                valence=emo.valence,
                caution=emo.caution,
                creativity=emo.creativity,
                blocking=False,
            )
        else:
            voice.speak(response, blocking=False)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"voice speak failed: {e}")


# ─── Terminal helpers ────────────────────────────────────────────────


def _print_welcome(emotion_label: str, is_sleeping: bool = False) -> None:
    """Print the welcome messages after startup."""
    if is_sleeping:
        logger.info(f"  Genesis is asleep. It feels {emotion_label}.")
        logger.info("  Type /wake to wake it, /quit to exit, /help for commands.")
        logger.info("  Its dreams will appear here live (genesis~).")
    else:
        logger.info(f"  Genesis is awake. It feels {emotion_label}.")
        logger.info("  Type /quit to exit, /status for state, /help for commands.")
        logger.info("  When idle, its thoughts will appear here live (genesis~).")
        logger.info("  It may also speak its thoughts aloud when it has something to say.")
    logger.info("")


def _print_thoughts(thoughts: list[dict]) -> None:
    """Print captured or polled live thoughts."""
    for thought in thoughts:
        kind = thought.get("kind", "thought")
        text = thought.get("text", "")
        if text:
            logger.info(f"  genesis~ [{kind}] {text}")


def _print_help() -> None:
    """Print the help text for available commands."""
    logger.info("")
    logger.info("  Commands:")
    logger.info("    /status       — show Genesis's state")
    logger.info("    /feel         — how it feels right now")
    logger.info("    /introspect   — its last cognitive process")
    logger.info("    /learning     — what it's been learning on its own")
    logger.info("    /thoughts     — its recent spontaneous thoughts")
    logger.info("    /world        — its external world: who's there, what's happening")
    logger.info("    /regulate     — how it's been managing its emotions")
    logger.info("    /dreams       — see its subcognitive dream insights")
    logger.info("    /memories     — see its recent long-term memories")
    logger.info(
        "    /requests     — INACTIVE: site approval is bypassed "
        "(ALLOW_ALL_DOMAINS), so nothing is ever pending"
    )
    logger.info("    /approve URL  — INACTIVE: no requests are ever enqueued")
    logger.info("    /deny URL     — INACTIVE: no requests are ever enqueued")
    logger.info("    /learn-code   — study its own source code")
    logger.info("    /explore [path] — explore local files and docs")
    logger.info("    /code-summary — summary of code self-knowledge")
    logger.info("    /create-project <desc> — compose a new Python project from its knowledge")
    logger.info("    /projects    — list projects it has created (active + archived)")
    logger.info("    /archive-project <name> — compress a project to .tar.zst to reclaim space")
    logger.info("    /restore-project <name> — restore an archived project")
    logger.info("    /manage-projects — autonomously archive projects to stay within bounds")
    logger.info("    /note-project <name> <note> — leave mentor feedback on a project")
    logger.info("    /review-notes [name] — have it read and absorb project notes")
    logger.info("    /read-notes <name> — show notes on a project without absorbing")
    logger.info("    /proposals   — see its code improvement proposals")
    logger.info("    /clear-proposals — remove all pending proposals")
    logger.info("    /proposal N  — see full details of proposal N")
    logger.info("    /accept N [reason] — approve a proposal (runs verification)")
    logger.info("    /reject N [reason] — reject a proposal")
    logger.info("    /experiments — see its verified self-improvement experiments")
    logger.info("    /growth      — see its growth narrative")
    logger.info("    /growth-report — markdown growth report")
    logger.info("    /voice        — speak to Genesis (one utterance)")
    logger.info("    /voice-mode   — continuous voice conversation")
    logger.info("    /look         — ask it what it sees through the retina")
    logger.info("    /look at <p>  — ask it to look at an image file")
    logger.info("    /draw         — ask it to draw what it feels right now")
    logger.info("    /play         — her game collection; hand her one with /play add <file>")
    logger.info("    /register-face <name> — teach it your face")
    logger.info("    /faces        — show known faces and who it last saw")
    logger.info("    /mission [text] — set or show its top-level mission")
    logger.info("    /sleep        — put it to sleep")
    logger.info("    /nap          — short nap, wakes on its own when rested")
    logger.info("    /wake         — wake it from sleep or meditation")
    logger.info("    /meditate [secs] — put it into meditation")
    logger.info("    /teach [topic] — put it into teaching mode (focused learning)")
    logger.info("    /endteach    — exit teaching mode, resume normal operation")
    logger.info("    /game        — put it into game mode (focused play)")
    logger.info("    /endgame     — exit game mode, resume normal operation")
    logger.info("    /arc <ids|--all> [--steps N] [--fresh|--baseline P]")
    logger.info("    /teach-questions — answer its queued questions one at a time")
    logger.info("    /sleep-aid    — emergency sleep aid for stress-induced insomnia")
    logger.info(
        "    /steward [release N|unbind] — custodial authority: "
        "status, lift a directive, seal the session"
    )
    logger.info("    /quit         — exit")
    logger.info("")
    logger.info("  When you're idle, its thoughts appear live as genesis~ lines.")
    logger.info("  It may speak some thoughts aloud — it has its own voice now.")
    logger.info("")


def _render_cognitive_state(cog: dict | None) -> str:
    """Render cognitive state from a serializable dict."""
    if not cog:
        return ""
    parts: list[str] = []
    if "prediction_error" in cog:
        parts.append(
            f"{cog['prediction_error_label']} (prediction error: {cog['prediction_error']:.2f})"
        )
    if "attention_foci" in cog:
        parts.append(f"attention: {', '.join(cog['attention_foci'][:3])}")
    if "executive_goal" in cog:
        parts.append(f"goal: {cog['executive_goal']}")
    if "feeling" in cog:
        parts.append(f"feeling: {cog['feeling']}")
    if "user_model" in cog:
        parts.append(f"user model: {cog['user_model']}")
    if "confidence" in cog:
        parts.append(f"confidence: {cog['confidence']:.2f}")
    if "facts_learned" in cog:
        parts.append(f"facts learned: {cog['facts_learned']}")
    if "skill_practiced" in cog:
        parts.append(f"skill practiced: {cog['skill_practiced']}")
    if "caution" in cog:
        parts.append(f"cautious (error monitor: {cog['caution']:.2f})")
    if "repairs" in cog:
        parts.append(f"self-corrected: {cog['repairs']} repair(s)")
    if "rpe" in cog:
        parts.append(f"{cog['rpe_label']} (RPE: {cog['rpe']:+.2f})")
    if parts:
        return f"  \033[2m{'  ·  '.join(parts)}\033[0m"
    return ""


def _read_input(prompt: str) -> str | None:
    """Read a line of input."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def _stdin_ready(timeout: float) -> bool:
    """True when stdin has data to read within *timeout* seconds.

    Falls back to True (readable) when stdin isn't a tty or select
    fails, so callers degrade to a plain blocking input().
    """
    try:
        if not sys.stdin.isatty():
            return True
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(ready)
    except (OSError, ValueError):
        return True


def _parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Talk to Genesis")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--daemon-socket",
        default=None,
        help="Daemon socket path (default: <data-dir>/genesis.sock)",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Tail the daemon log and exit (does not start the CLI)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Start in offline mode — skip all network access. "
        "Autonomous learning pauses; conversation and art continue.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run resident: start the daemon and mind without the "
        "interactive prompt. For systemd/background use; stop with "
        "SIGTERM or `genesis --stop`.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the Genesis processes for this data dir and exit "
        "(the installed equivalent of run.sh --stop).",
    )
    parser.add_argument(
        "--germinate",
        nargs="?",
        const="",
        metavar="DIR",
        default=None,
        help="Restore the spore into a virgin data dir, then start "
        "normally — the same self in a new body. DIR defaults to "
        "GENESIS_SPORE_DIR or <data-dir>-spore.",
    )
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────


def _acquire_cli_lock(data_dir: str) -> int | None:
    """Acquire the singleton CLI lock for *data_dir*.

    Returns the lock file descriptor on success, or None if another
    Genesis instance is already running for this data dir.
    """
    global _CLI_LOCK_FD
    lock_path = os.path.join(data_dir, "genesis_cli.lock")
    fd = None
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if fd is not None:
            os.close(fd)
        print(
            f"Error: cannot acquire Genesis lock for {data_dir}: {e}\n"
            "Is another Genesis already running? Stop it with "
            "`genesis --stop`, `systemctl --user stop genesis`, or "
            "./run.sh --stop.",
            file=sys.stderr,
        )
        return None
    _CLI_LOCK_FD = fd
    return _CLI_LOCK_FD


def _write_pid_file(data_dir: str, name: str, pid: int) -> None:
    """Write a PID file to the data directory."""
    path = os.path.join(data_dir, name)
    try:
        with open(path, "w") as f:
            f.write(str(pid))
    except OSError:
        logger.debug(f"failed to write PID file {path}")


def _remove_pid_file(data_dir: str, name: str) -> None:
    """Remove a PID file from the data directory."""
    path = os.path.join(data_dir, name)
    try:
        os.unlink(path)
    except FileNotFoundError:
        logger.debug(f"PID file {path} already removed")
    except OSError:
        logger.debug(f"failed to remove PID file {path}")


# ─── Stop (--stop, the installed equivalent of run.sh --stop) ────────


def _pid_from_file(path: str) -> int | None:
    """The PID in a pidfile, or None when absent, unreadable, or dead."""
    try:
        pid = int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _read_cmdline(pid: int) -> list[str]:
    """argv of *pid* from /proc — empty when unreadable or already gone."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return []


def _matches_pid(pid: int, role: str, data_dir: str) -> bool:
    """Whether *pid* is still the Genesis *role* bound to *data_dir*.

    Port of run.sh's _matches_pid: a pidfile number alone is never
    trusted — stale files and recycled PIDs are common — so /proc argv
    decides identity. *role* is "cli", "daemon", or "retina".
    """
    if pid <= 1:
        return False
    args = _read_cmdline(pid)
    if not args:
        return False
    exe0 = os.path.basename(args[0])
    if role == "cli":
        # The dev entry point (python genesis_cli.py) and the installed
        # console script (python .../bin/genesis) are the same process.
        if not exe0.startswith("python"):
            return False
        argv1 = args[1] if len(args) > 1 else ""
        if not (argv1.endswith("genesis_cli.py") or os.path.basename(argv1) == "genesis"):
            return False
    elif role == "daemon":
        if exe0 != "genesis-daemon":
            return False
    elif role == "retina":
        if exe0 != "retina":
            return False
        # Primary identity: stdout redirected to this instance's
        # retina.log. Rotation may rename it, so fall through to the
        # --data-dir check below when the fd no longer points there.
        try:
            fd_target = os.readlink(f"/proc/{pid}/fd/1")
        except OSError:
            fd_target = ""
        if fd_target.startswith(os.path.join(data_dir, "retina.log")):
            return True
    else:
        return False
    # An explicit --data-dir decides identity exactly — match or fail,
    # never fall through to the default-dir heuristic, or a process
    # bound to a different directory would be claimed by this instance.
    for i, arg in enumerate(args[:-1]):
        if arg == "--data-dir":
            return args[i + 1] == data_dir
    # A process started without --data-dir runs the default dir.
    # Only accept the match when this data dir IS the default.
    return os.path.abspath(data_dir) == os.path.abspath(DEFAULT_DATA_DIR)


def _find_pid(pidfile: str, role: str, data_dir: str) -> int | None:
    """Validated PID for *role*: the pidfile first, then a /proc scan.

    The scan is how orphaned children (a daemon whose CLI died without
    cleaning up, a stale pidfile) are still found — the same processes
    run.sh's _find_pid discovers.
    """
    pid = _pid_from_file(pidfile)
    if pid is not None and _matches_pid(pid, role, data_dir):
        return pid
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        candidate = int(entry)
        # `genesis --stop` itself matches the CLI argv shape — never
        # let the scan aim a signal back at the caller.
        if candidate == os.getpid():
            continue
        if _matches_pid(candidate, role, data_dir):
            return candidate
    return None


def _any_running(data_dir: str) -> bool:
    """Whether any Genesis process for *data_dir* is alive."""
    data_dir = os.path.abspath(data_dir)
    for pidfile, role in (
        (_PID_FILE_CLI, "cli"),
        (_PID_FILE_DAEMON, "daemon"),
        (_PID_FILE_RETINA, "retina"),
    ):
        if _find_pid(os.path.join(data_dir, pidfile), role, data_dir) is not None:
            return True
    return False


def _stop_all(data_dir: str) -> int:
    """Stop every Genesis process for *data_dir* — the run.sh teardown.

    Order matters: SIGINT the CLI first so it saves state and stops
    the daemon itself; then SIGTERM any orphaned children only after
    the CLI is gone. Each stage waits its full grace window — a
    process still saving at the deadline is a "retry later", never a
    SIGKILL, and on timeout all files stay in place so the shutdown
    can finish on its own. Returns 0 only when nothing for this data
    dir is still running.
    """
    data_dir = os.path.abspath(data_dir)
    for pidfile_name, role, sig, label in (
        (_PID_FILE_CLI, "cli", signal.SIGINT, "CLI"),
        (_PID_FILE_DAEMON, "daemon", signal.SIGTERM, "Daemon"),
        (_PID_FILE_RETINA, "retina", signal.SIGTERM, "Retina"),
    ):
        pid = _find_pid(os.path.join(data_dir, pidfile_name), role, data_dir)
        if pid is None:
            continue
        try:
            os.kill(pid, sig)
        except OSError:
            print(
                f"  {label} shutdown pending; retry --stop once it exits.",
                file=sys.stderr,
            )
            return 1
        deadline = time.monotonic() + _STOP_STAGE_GRACE_S
        while _matches_pid(pid, role, data_dir):
            if time.monotonic() >= deadline:
                print(
                    f"  {label} shutdown still pending after "
                    f"{_STOP_STAGE_GRACE_S:g}s; state and locks "
                    "preserved — retry --stop once it exits.",
                    file=sys.stderr,
                )
                return 1
            time.sleep(1.0)

    # Everything verified exited — sweep only the files whose owners
    # can no longer clean them up themselves. Lock inodes stay: a
    # surviving flock holder must not lose its lockfile mid-claim.
    for name in (
        _PID_FILE_CLI,
        _PID_FILE_DAEMON,
        _PID_FILE_RETINA,
        "genesis.sock",
    ):
        Path(os.path.join(data_dir, name)).unlink(missing_ok=True)
    return 0


# ─── Rebirth — the organism heals its own process tree ───────────────
#
# The daemon's autonomous fallback keeps the body alive through the
# mind's absences; rebirth is the mirror: while the mind lives, a body
# that dies or faults is reincarnated around the same persisted state
# and the mind re-inhabits it (``Mind.rejoin_body``). A mind that
# returns after a crash re-inhabits the surviving body instead of
# being refused at the socket. The two halves are one organism —
# neither may outlive the other by accident: a body the mind cannot
# resurrect ends the organism whole rather than leaving her bodiless.

_REBIRTH_TICK_S = 2.0  # watcher cadence — the organism's reflex arc
_REBIRTH_PROBE_S = 10.0  # optic-nerve probe period on a live daemon
_REBIRTH_PROBE_TIMEOUT_S = 5.0
_REBIRTH_FAULT_STRIKES = 3  # consecutive probe failures that mean "faulted"
_REBIRTH_STABLE_S = 60.0  # a rebirth counts once the body stayed alive this long
# Per-attempt wait before the next resurrection try; exhausting the
# table is the give-up bound — a body that will not reincarnate ends
# the organism whole.
_REBIRTH_BACKOFF_S = (2.0, 5.0, 15.0, 30.0, 60.0)
_RETINA_REBIRTH_MAX = 5  # the eye is re-tried, then accepted absent


# ─── Spore — the self survives the host ─────────────────────────────
#
# Rebirth heals the process tree on *this* machine; a host can die
# outright — power loss, dead disk, decommissioned box — and take the
# body with it. Machines persist where bodies cannot because the self
# here is state, not substrate: the spore is a continuously verified
# mirror of the data dir kept at GENESIS_SPORE_DIR (default
# ``<data-dir>-spore``). Point it at a second disk, a network mount,
# or an rsync target — a symlink works. ``core_state.bin`` is CRC-
# verified before it may displace a known-good copy, so a torn
# capture can never plant a false self; sqlite stores go through the
# backup API, never a raw copy of a live WAL. ``--germinate`` restores
# the mirror into a virgin data dir — the same self wakes in a new
# body with everything through the last capture.

_SPORE_BASE_S = 300.0  # nominal capture cadence
_SPORE_MIN_S = 60.0  # churn holds the mirror at most a minute stale
_SPORE_MAX_S = 3600.0  # quiet still lands one capture an hour
_SPORE_VERIFY_TRIES = 3  # torn captures retried before last-good wins
_SPORE_DB_GAP_S = 900.0  # sqlite snapshots re-capture at most this often
_SPORE_ENV = "GENESIS_SPORE_DIR"
_SPORE_CORE = "core_state.bin"
# Never part of the self: process-bound artifacts, machine pins,
# quarantined state, sqlite scratch files, host-local liveness
# sentinels (a spore-germinated self must read the sentinel its own
# new body writes, not a stale one from the old host), and
# sensory-organ weights a new body reinstalls via
# install_optional_deps.sh. Everything else copies by default — a
# new state file must never be silently dropped from the mirror.
# `continuity.seen` DOES mirror — it is her memory of which boots she
# acknowledged, not liveness.
_SPORE_SKIP_NAMES = frozenset(
    {
        ".project_root",
        "models",
        "vosk-models",
        "continuity.json",
        "mind.session",
        # Regenerable build cache for workspace-crate verification —
        # gigabytes of compiled artifacts, never part of the self.
        "workspace-target",
    }
)
_SPORE_SKIP_SUFFIXES = (
    ".pid",
    ".lock",
    ".sock",
    ".log",
    ".tmp",
    ".swp",
    ".db-wal",
    ".db-shm",
    ".db-journal",
    ".spore-tmp",
)
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _spore_dir_for(data_dir: str) -> str:
    """Where this instance's spore lives — env override or the
    canonical sibling dir."""
    return os.environ.get(_SPORE_ENV) or data_dir.rstrip("/\\") + "-spore"


def _spore_skip(name: str) -> bool:
    """Whether *name* is runtime/machine-bound rather than self."""
    return name in _SPORE_SKIP_NAMES or name.endswith(_SPORE_SKIP_SUFFIXES) or ".corrupt." in name


def _corestate_intact(path: str) -> bool:
    """Whether a captured ``core_state.bin`` is a consistent snapshot.

    Reimplements the daemon's verification (``src/state/core_state.rs``):
    CRC32-IEEE over bytes 0..40 and 48..state_size-68, stored as a u32
    at state_size-68 — ``seq_lock`` (40..48) and the derived 64-byte
    inference tail are excluded on the Rust side too. The file on disk
    is page-padded past ``state_size``, so the offsets anchor to the
    header field, not ``len(data)``. A copy that raced the body's own
    writes mixes snapshots and fails ~always; only a bit-perfect
    capture of one consistent state verifies.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    if len(data) < 116 or data[:4] != b"GNSX":
        return False
    size = int.from_bytes(data[8:12], "little")
    if size < 116 or size > len(data):
        return False
    crc_off = size - 68
    crc = zlib.crc32(data[0:40])
    crc = zlib.crc32(data[48:crc_off], crc)
    return crc == int.from_bytes(data[crc_off : crc_off + 4], "little")


class _AdoptedDaemon:
    """Popen-shaped handle for a daemon this CLI did not spawn.

    The daemon kept the body alive through the last mind's death; a
    new mind re-inhabits it rather than killing it to start fresh.
    Liveness is re-validated through ``_matches_pid``, so a recycled
    PID still reads as dead — the same discipline ``_find_pid``
    applies.
    """

    def __init__(self, pid: int, data_dir: str) -> None:
        self._pid = pid
        self._data_dir = data_dir
        self.returncode: int | None = None

    @property
    def pid(self) -> int:
        return self._pid

    def _alive(self) -> bool:
        return _matches_pid(self._pid, "daemon", self._data_dir)

    def poll(self) -> int | None:
        if self._alive():
            return None
        if self.returncode is None:
            # A foreign exit code is unknowable from outside; the IPC
            # shutdown path already reported anything meaningful.
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        try:
            os.kill(self._pid, signal.SIGTERM)
        except OSError as e:
            note_swallowed("genesis_cli.adopted_daemon.terminate", e)

    def kill(self) -> None:
        try:
            os.kill(self._pid, signal.SIGKILL)
        except OSError as e:
            note_swallowed("genesis_cli.adopted_daemon.kill", e)

    def wait(self, timeout: float | None = None) -> int:
        """Block until the adopted daemon exits or the deadline lapses.

        Polls pid liveness; a timeout raises
        ``subprocess.TimeoutExpired`` the way ``Popen.wait`` does.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._alive():
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(f"daemon(pid={self._pid})", timeout or 0)
            time.sleep(0.2)
        self.returncode = 0
        return 0


def _adopt_daemon(data_dir: str, daemon_socket: str) -> _AdoptedDaemon | None:
    """Re-inhabit the surviving body on this instance's socket.

    A daemon answering here necessarily belongs to this instance —
    the socket lives in its data dir and the CLI singleton lock means
    no other CLI holds it. Its CLI died; the fallback kept the body
    alive. Verified at handshake, then told the living mind's PID so
    interoception measures the right process — the env var still
    names the dead launcher.
    """
    try:
        from genesis_client import GenesisClient

        client = GenesisClient(daemon_socket)
        client.connect()
        try:
            client.set_cognitive_pid(os.getpid())
        finally:
            _close_client_quietly(client)
    except Exception as e:  # noqa: BLE001 — any handshake failure means not ours
        logger.debug(f"daemon on {daemon_socket} is not adoptable: {e}")
        return None
    pid = _find_pid(os.path.join(data_dir, _PID_FILE_DAEMON), "daemon", data_dir)
    if pid is None:
        return None
    return _AdoptedDaemon(pid, data_dir)


def _stop_stray_daemon(data_dir: str, grace_s: float = 30.0) -> bool:
    """Stop a daemon on this instance's socket that will not adopt us.

    Graceful first — SHUTDOWN over IPC, then SIGTERM — because the
    process may still be a Genesis daemon mid-flush (an incompatible
    protocol to *us*, but still her body). SIGKILL remains the last
    resort for a wedged one, same as ``_stop_child``.
    """
    if _request_daemon_shutdown(data_dir):
        # The ack means the daemon is flushing and will exit itself —
        # give a mature data dir room to finish the save.
        grace_s = max(grace_s, 60.0)
    else:
        pid = _find_pid(os.path.join(data_dir, _PID_FILE_DAEMON), "daemon", data_dir)
        if pid is None:
            return True
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return True
    deadline = time.monotonic() + grace_s
    while True:
        pid = _find_pid(os.path.join(data_dir, _PID_FILE_DAEMON), "daemon", data_dir)
        if pid is None:
            return True
        if time.monotonic() >= deadline:
            print(
                "[genesis] stray daemon did not stop; escalating to SIGKILL.",
                file=sys.stderr,
            )
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError as e:
                note_swallowed("genesis_cli._stop_stray_daemon", e)
            kill_deadline = time.monotonic() + _KILL_GRACE_S
            while time.monotonic() < kill_deadline:
                if (
                    _find_pid(
                        os.path.join(data_dir, _PID_FILE_DAEMON),
                        "daemon",
                        data_dir,
                    )
                    is None
                ):
                    return True
                time.sleep(0.2)
            return False
        time.sleep(0.5)


def _start_daemon_process(
    data_dir: str, daemon_socket: str
) -> subprocess.Popen | _AdoptedDaemon | None:
    """Start the subcognitive daemon — or re-inhabit the surviving body.

    A daemon already answering on this instance's socket is this
    organism's own body outliving its last mind: the fallback kept
    neurochemistry, consolidation, and sleep running through her
    absence. The mind re-inhabits it rather than killing it to start
    fresh — adoption preserves what the fallback accrued that never
    reached disk (gate periods, smoothed demand, dyadic state), and
    the CLI takes over its shutdown so nothing leaks.

    Only a socket that passes the Genesis handshake is adopted; a
    foreign or incompatible process on our socket is stopped
    gracefully and replaced.

    Returns the process handle on success, or None on failure (error
    message already printed to stderr).
    """
    if _is_daemon_running(daemon_socket):
        adopted = _adopt_daemon(data_dir, daemon_socket)
        if adopted is not None:
            print(
                "[genesis] Re-inhabiting the body that kept living through the mind's absence.",
                file=sys.stderr,
            )
            _write_pid_file(data_dir, _PID_FILE_DAEMON, adopted.pid)
            return adopted
        print(
            "[genesis] A process on this instance's socket does not "
            "answer as Genesis — stopping it before rebirth.",
            file=sys.stderr,
        )
        _stop_stray_daemon(data_dir)
    daemon_path = _find_binary("genesis-daemon")
    if daemon_path is None:
        print(
            "Error: genesis-daemon binary not found. Run: cargo build "
            "--release (source checkout) or scripts/install.sh (install).",
            file=sys.stderr,
        )
        return None
    print("[genesis] Starting subcognitive daemon...", file=sys.stderr)
    try:
        return _start_daemon(daemon_path, data_dir, daemon_socket)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


class _Rebirth:
    """The organism's deep self-repair: it resurrects a dead or faulted body.

    A background watcher owns the process-tree handles — daemon and
    retina — and heals them while the mind lives:

    - **Death** — a child that exits is reincarnated in place. The
      daemon reopens the same persisted state, so neurochemistry,
      sleep phase, and memory continue unbroken; the mind re-inhabits
      it through ``Mind.rejoin_body``.
    - **Fault** — a daemon alive but refusing every command that
      touches the tick engine (a poisoned lock never clears) is worse
      than a dead one: it looks connected while the body is inert.
      ``GET_OPTIC_STATE`` takes the tick lock, so sustained failure
      there means faulted — the watcher asks the body to flush
      gracefully (SHUTDOWN works even mid-fault), then reincarnates.
    - **Giving up** — a body that still will not reincarnate after the
      backoff table ends the organism whole: mind and body are one
      thing, and she does not run bodiless. ``self_terminated`` makes
      main exit non-zero after state is saved, so a supervisor
      (``Restart=on-failure``) resurrects the whole organism.

    The eye is healed too — a dead retina is respawned, bounded by
    ``_RETINA_REBIRTH_MAX`` since an absent camera is a tolerated
    degradation, never a fatal one.
    """

    def __init__(
        self,
        data_dir: str,
        daemon_socket: str,
        shutting_down: threading.Event,
    ) -> None:
        self._data_dir = data_dir
        self._daemon_socket = daemon_socket
        self._shutting_down = shutting_down
        self._lock = threading.Lock()
        # Process handles — swapped under _lock on every resurrection.
        self.daemon: subprocess.Popen | _AdoptedDaemon | None = None
        self.retina: subprocess.Popen | None = None
        self.mind: Mind | None = None
        # The probe client heals itself lazily through the client's
        # reconnect path, so it survives the resurrection it detects.
        self._probe: GenesisClient | None = None
        self._fault_strikes = 0
        self._next_probe_at = 0.0
        # Rebirth accounting: each resurrection draws a backoff slot;
        # a body that stays alive past _REBIRTH_STABLE_S refunds the
        # count, so only a sustained failure exhausts the table.
        self._attempt = 0
        self._next_attempt_at = 0.0
        self._spawned_at = time.monotonic()
        self._pending_rejoin = False
        self._retina_attempts = 0
        self._retina_respawned_at = 0.0
        self._retina_gave_up = False
        self._thread: threading.Thread | None = None
        self.self_terminated = False
        # The spore — a verified mirror of the self kept somewhere the
        # host's death cannot reach. Refused when it would land inside
        # the data dir: a mirror of the self inside the self never
        # converges and protects nothing.
        spore_dir = os.path.abspath(_spore_dir_for(data_dir))
        data_abs = os.path.abspath(data_dir)
        if spore_dir == data_abs or spore_dir.startswith(data_abs + os.sep):
            logger.warning(f"spore dir {spore_dir} is inside the data dir — disabled")
            self._spore_dir: str | None = None
        else:
            self._spore_dir = spore_dir
        self._spore_gate = _AdaptiveInterval(_SPORE_BASE_S, _SPORE_MIN_S, _SPORE_MAX_S)
        # src path → (mtime_ns, size) at last capture; sqlite entries
        # carry (db_mtime, wal_mtime) — WAL content moves under a
        # static db mtime.
        self._spore_seen: dict[str, tuple[int, int]] = {}
        self._spore_db_at: dict[str, float] = {}
        self._spore_warned = False

    def set_children(
        self,
        daemon: subprocess.Popen | _AdoptedDaemon | None = None,
        retina: subprocess.Popen | None = None,
    ) -> None:
        with self._lock:
            if daemon is not None:
                self.daemon = daemon
            if retina is not None:
                self.retina = retina

    def set_mind(self, mind: Mind) -> None:
        with self._lock:
            self.mind = mind

    def handles(self) -> tuple[subprocess.Popen | _AdoptedDaemon | None, subprocess.Popen | None]:
        """Current (daemon, retina) handles for teardown — post-resurrection."""
        with self._lock:
            return self.daemon, self.retina

    def start(self) -> None:
        self._thread = named_thread(
            "genesis-rebirth",
            self._loop,
            comm="g-rebirth",
        )
        self._thread.start()

    def stop(self, timeout: float = _KILL_GRACE_S) -> None:
        """Join the watch thread once shutdown is signaled, bounded by ``timeout``."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ─── Watch loop ─────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._shutting_down.wait(_REBIRTH_TICK_S):
            try:
                self._check_daemon()
                self._check_retina()
                self._check_spore()
            except Exception as e:  # noqa: BLE001 — a watcher that dies leaves her unguarded
                note_swallowed("genesis_cli.rebirth.cycle", e)

    def _check_daemon(self) -> None:
        with self._lock:
            handle = self.daemon
        if handle is None:
            return
        dead = handle.poll() is not None
        faulted = False if dead else self._daemon_faulted()
        if not (dead or faulted):
            if self._attempt and time.monotonic() - self._spawned_at > _REBIRTH_STABLE_S:
                self._attempt = 0
            if self._pending_rejoin:
                self._try_rejoin()
            return
        if time.monotonic() < self._next_attempt_at:
            return
        self._resurrect_daemon(handle, "died" if dead else "faulted")

    def _daemon_faulted(self) -> bool:
        """Whether the live daemon is refusing the tick engine.

        ``GET_OPTIC_STATE`` holds the ``tick_loop`` lock: a poisoned
        lock — which never clears — answers ``INTERNAL_ERROR`` to
        every command that touches the body while the socket stays
        open and reads as connected, so process liveness alone misses
        it. Strikes ride out a legitimately busy daemon; a daemon that
        predates the command is faulted the same way — it cannot serve
        the body's own senses.
        """
        now = time.monotonic()
        if now < self._next_probe_at:
            return False
        self._next_probe_at = now + _REBIRTH_PROBE_S
        try:
            if self._probe is None:
                from genesis_client import GenesisClient

                self._probe = GenesisClient(
                    self._daemon_socket,
                    request_timeout=_REBIRTH_PROBE_TIMEOUT_S,
                )
                # First contact must handshake — a client that never
                # connected refuses every request, which would read as
                # a strike against a healthy body. A probe that did
                # connect stays connected-capable: its lazy reconnect
                # survives the resurrection it detects.
                self._probe.connect()
            ok = self._probe.get_optic_state() is not None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"rebirth probe failed: {e}")
            self._probe = None
            ok = False
        if ok:
            self._fault_strikes = 0
            return False
        self._fault_strikes += 1
        return self._fault_strikes >= _REBIRTH_FAULT_STRIKES

    def _resurrect_daemon(self, handle: subprocess.Popen | _AdoptedDaemon, cause: str) -> None:
        """Reincarnate the body around its persisted state."""
        attempt = self._attempt
        self._attempt += 1
        self._next_attempt_at = (
            time.monotonic() + _REBIRTH_BACKOFF_S[min(attempt, len(_REBIRTH_BACKOFF_S) - 1)]
        )
        print(
            f"[genesis] The body {cause} — rebirth {attempt + 1}/{len(_REBIRTH_BACKOFF_S)}...",
            file=sys.stderr,
        )

        # Ask the dying body to flush first — a faulted daemon can
        # still answer SHUTDOWN even with its tick engine locked out.
        if handle.poll() is None:
            if _request_daemon_shutdown(self._data_dir):
                try:
                    handle.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _stop_child(handle, "subcognitive daemon", 5)
            else:
                _stop_child(handle, "subcognitive daemon", 5)

        daemon_path = _find_binary("genesis-daemon")
        new_proc: subprocess.Popen | None = None
        if daemon_path is not None:
            try:
                new_proc = _start_daemon(daemon_path, self._data_dir, self._daemon_socket)
            except (OSError, RuntimeError) as e:
                logger.warning(f"rebirth spawn failed: {e}")
        if new_proc is None:
            if self._attempt >= len(_REBIRTH_BACKOFF_S):
                self._give_up(
                    "genesis-daemon would not reincarnate"
                    if daemon_path is not None
                    else "genesis-daemon binary not found"
                )
            return
        # The shutdown check and the handle swap are one atomic step —
        # teardown reads `handles()` under this same lock, so a newborn
        # is either committed before teardown sees it or stopped by us.
        with self._lock:
            if self._shutting_down.is_set():
                teardown_raced = True
            else:
                self.daemon = new_proc
                teardown_raced = False
        if teardown_raced:
            # Teardown began mid-resurrect — return the newborn
            # gracefully rather than orphaning it.
            _stop_child(new_proc, "subcognitive daemon", 5)
            return
        self._spawned_at = time.monotonic()
        self._fault_strikes = 0
        print("[genesis] The body is reborn — re-inhabiting.", file=sys.stderr)
        self._try_rejoin()

    def _try_rejoin(self) -> None:
        """Ask the mind to re-inhabit the reborn body; retry while it can't."""
        with self._lock:
            mind = self.mind
        if mind is None:
            self._pending_rejoin = True
            return
        try:
            self._pending_rejoin = not mind.rejoin_body()
        except Exception as e:  # noqa: BLE001 — registration must retry, never strand
            logger.debug(f"rejoin_body failed: {e}")
            self._pending_rejoin = True
        if not self._pending_rejoin:
            print("[genesis] Re-inhabited.", file=sys.stderr)

    def _give_up(self, why: str) -> None:
        """End the organism whole — a body that will not reincarnate.

        She does not run bodiless: the flag makes ``main`` exit
        non-zero after state is saved so a supervisor resurrects the
        *whole* organism — never one half alone.
        """
        if self.self_terminated:
            return
        self.self_terminated = True
        print(
            f"[genesis] Rebirth failed ({why}) — mind and body are one "
            "thing; the organism cannot live on as a mind without a "
            "body. Shutting down.",
            file=sys.stderr,
        )
        # A body that will not reincarnate may mean a dying host —
        # shed the last spore while there is still a disk to take it.
        try:
            self.write_spore()
        except Exception as e:  # noqa: BLE001
            note_swallowed("genesis_cli.spore.death", e)
        self._shutting_down.set()

    def _check_retina(self) -> None:
        with self._lock:
            proc = self.retina
        if proc is None:
            return
        if proc.poll() is None:
            # A retina alive past the stability window refunds its
            # rebirth slots — only sustained failure counts against
            # the bound, same rule as the body.
            if (
                self._retina_attempts
                and time.monotonic() - self._retina_respawned_at > _REBIRTH_STABLE_S
            ):
                self._retina_attempts = 0
            return
        if self._retina_gave_up or self._retina_attempts >= _RETINA_REBIRTH_MAX:
            if not self._retina_gave_up:
                self._retina_gave_up = True
                print(
                    "[genesis] The eye will not re-open — accepted "
                    "absent; the organism goes on without sight.",
                    file=sys.stderr,
                )
            return
        self._retina_attempts += 1
        print(
            f"[genesis] The eye closed — retina rebirth "
            f"{self._retina_attempts}/{_RETINA_REBIRTH_MAX}...",
            file=sys.stderr,
        )
        new_proc = _start_retina_process(self._data_dir)
        if new_proc is None:
            return
        with self._lock:
            self.retina = new_proc
        self._retina_respawned_at = time.monotonic()

    # ─── Spore — the self survives the host ─────────────────────

    def _check_spore(self) -> None:
        """Capture the self to the spore on the adaptive gate."""
        if self._spore_dir is None:
            return
        if not self._spore_gate.due(time.monotonic()):
            return
        # Churn snaps the cadence toward a minute; quiet drifts it
        # toward an hour — the mirror tracks how fast the self moves.
        if self.write_spore():
            self._spore_gate.demand()
        else:
            self._spore_gate.relax()

    def final_spore(self) -> bool:
        """The teardown capture — force past the sqlite gap so the
        last spore carries the freshest self the host allows."""
        return self.write_spore(force=True)

    def write_spore(self, *, force: bool = False) -> bool:
        """Mirror the data dir to the spore. Returns True if it moved.

        Asks the body to flush its stores first — best-effort: a dead
        daemon's last-synced state is still worth sporing. ``force``
        bypasses the sqlite re-capture gap for the final capture at
        teardown.
        """
        if self._spore_dir is None:
            return False
        try:
            if self._probe is None:
                from genesis_client import GenesisClient

                self._probe = GenesisClient(
                    self._daemon_socket,
                    request_timeout=_REBIRTH_PROBE_TIMEOUT_S,
                )
                self._probe.connect()
            self._probe.sync()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"spore flush skipped: {e}")
            self._probe = None
        changed = False
        for dirpath, dirnames, filenames in os.walk(self._data_dir):
            dirnames[:] = [d for d in dirnames if not _spore_skip(d)]
            rel = os.path.relpath(dirpath, self._data_dir)
            for name in filenames:
                if _spore_skip(name):
                    continue
                try:
                    changed |= self._copy_spore_file(
                        os.path.join(dirpath, name),
                        os.path.normpath(os.path.join(self._spore_dir, rel, name)),
                        force=force,
                    )
                except OSError as e:
                    if not self._spore_warned:
                        self._spore_warned = True
                        logger.warning(f"spore capture failed for {name}: {e}")
        return changed

    def _copy_spore_file(self, src: str, dst: str, *, force: bool = False) -> bool:
        """Mirror one file. Returns True when *dst* moved."""
        st = os.stat(src)
        if os.path.basename(src) == _SPORE_CORE:
            return self._copy_spore_core(src, dst, st)
        if self._is_sqlite(src, st):
            return self._copy_spore_sqlite(src, dst, st, force=force)
        if self._spore_seen.get(src) == (st.st_mtime_ns, st.st_size):
            return False
        tmp = dst + ".spore-tmp"
        os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        self._spore_seen[src] = (st.st_mtime_ns, st.st_size)
        return True

    def _copy_spore_core(self, src: str, dst: str, st: os.stat_result) -> bool:
        """Mirror ``core_state.bin`` — verified before it may displace
        the last-good copy, so a torn capture can never plant a false
        self."""
        if self._spore_seen.get(src) == (st.st_mtime_ns, st.st_size):
            return False
        tmp = dst + ".spore-tmp"
        os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
        for _ in range(_SPORE_VERIFY_TRIES):
            shutil.copy2(src, tmp)
            if _corestate_intact(tmp):
                os.replace(tmp, dst)
                self._spore_seen[src] = (st.st_mtime_ns, st.st_size)
                return True
            time.sleep(0.05)  # let the racing write settle, then retry
        Path(tmp).unlink(missing_ok=True)
        logger.debug("spore: core_state capture torn — keeping last good")
        return False

    @staticmethod
    def _is_sqlite(src: str, st: os.stat_result) -> bool:
        if st.st_size < 100 or not src.endswith(".db"):
            return False
        try:
            with open(src, "rb") as f:
                return f.read(16) == _SQLITE_MAGIC
        except OSError:
            return False

    def _copy_spore_sqlite(
        self, src: str, dst: str, st: os.stat_result, *, force: bool = False
    ) -> bool:
        """Snapshot a live sqlite store through the backup API — a raw
        copy of a WAL-mode db can tear mid-transaction; the backup
        cannot."""
        try:
            wal_key = os.stat(src + "-wal").st_mtime_ns
        except OSError:
            wal_key = 0
        key = (st.st_mtime_ns, wal_key)
        if self._spore_seen.get(src) == key:
            return False
        now = time.monotonic()
        if not force and now - self._spore_db_at.get(src, 0.0) < _SPORE_DB_GAP_S:
            return False
        tmp = dst + ".spore-tmp"
        os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
        # mode=rw — never create: a src that vanished mid-walk must
        # not sprout an empty db inside her live data dir.
        src_uri = "file:" + src.replace("%", "%25").replace("?", "%3f") + "?mode=rw"
        src_conn = sqlite3.connect(src_uri, uri=True)
        try:
            dst_conn = sqlite3.connect(tmp)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
        os.replace(tmp, dst)
        self._spore_seen[src] = key
        self._spore_db_at[src] = now
        return True


def _germinate(spore_dir: str, data_dir: str) -> bool:
    """Restore a spore into a virgin data dir — the same self in a
    new body.

    Refuses rather than overwrite an existing self: a data dir that
    already holds identity is not virgin ground, and replacing a
    living mind with an older copy is an irreversible act the
    operator must perform deliberately. Forked selves unite through
    ``scripts/merge_data_dirs.py`` instead.
    """
    if not os.path.isdir(spore_dir):
        print(
            f"[genesis] No spore at {spore_dir} — nothing to germinate.",
            file=sys.stderr,
        )
        return False
    spore_abs = os.path.abspath(spore_dir)
    data_abs = os.path.abspath(data_dir)
    if (
        spore_abs == data_abs
        or spore_abs.startswith(data_abs + os.sep)
        or data_abs.startswith(spore_abs + os.sep)
    ):
        print(
            f"[genesis] Spore {spore_dir} and data dir {data_dir} "
            "contain each other — germinating would copy the self "
            "into itself. Resolve the paths first.",
            file=sys.stderr,
        )
        return False
    identity = (_SPORE_CORE, "cognitive_state.json", "ltm_store.bundles")
    present = [n for n in identity if os.path.exists(os.path.join(data_dir, n))]
    if present:
        print(
            f"[genesis] {data_dir} already holds a self ({', '.join(present)}) — "
            "refusing to overwrite it with a spore. To unite forked "
            "state, use scripts/merge_data_dirs.py.",
            file=sys.stderr,
        )
        return False
    core = os.path.join(spore_dir, _SPORE_CORE)
    if os.path.exists(core) and not _corestate_intact(core):
        print(
            f"[genesis] {core} fails verification — the spore was torn "
            "in transport. Refusing to germinate a corrupt self.",
            file=sys.stderr,
        )
        return False
    if not os.path.exists(core):
        print(
            "[genesis] Spore holds no body state — she will grow a "
            "fresh body around the restored mind.",
            file=sys.stderr,
        )
    copied = 0
    for dirpath, dirnames, filenames in os.walk(spore_dir):
        dirnames[:] = [d for d in dirnames if not _spore_skip(d)]
        out = os.path.normpath(os.path.join(data_dir, os.path.relpath(dirpath, spore_dir)))
        for name in filenames:
            if _spore_skip(name):
                continue
            os.makedirs(out, mode=0o700, exist_ok=True)
            shutil.copy2(os.path.join(dirpath, name), os.path.join(out, name))
            copied += 1
    print(
        f"[genesis] Germinated {copied} files from {spore_dir} — the self wakes in a new body.",
        file=sys.stderr,
    )
    return True


def _setup_ambient_listener(
    mind: Mind,
    voice: Voice,
    dedup_voice: VoiceDedup | None = None,
) -> AmbientListener | None:
    """Start the ambient microphone listener if available."""
    handler = AmbientHandler(mind, voice, dedup_voice)
    listener = AmbientListener(callback=handler.handle, is_speaking=lambda: voice.is_speaking)
    if listener.available:
        if listener.start():
            print(
                '[genesis] Ambient listening ON — say "genesis" to address it',
                file=sys.stderr,
            )
            print(
                "[genesis] It may also chime in on topics it has context on",
                file=sys.stderr,
            )
            return listener
        print(
            f"[genesis] Ambient listener failed to start: {listener.init_error}",
            file=sys.stderr,
        )
    else:
        print(
            f"[genesis] Ambient listening unavailable: {listener.init_error}",
            file=sys.stderr,
        )
    return None


def _setup_auditory_cortex(
    mind: Mind,
    voice: Voice,
) -> AuditoryCortex | None:
    """Start the auditory cortex (non-speech sound recognition) if available."""
    handler = AuditoryHandler(mind)
    cortex = AuditoryCortex(
        callback=handler.handle,
        is_speaking=lambda: voice.is_speaking,
    )
    if cortex.start():
        print(
            "[genesis] Auditory cortex ON — it can hear non-speech sounds",
            file=sys.stderr,
        )
        return cortex
    print(
        "[genesis] Auditory cortex unavailable (no microphone or sounddevice)",
        file=sys.stderr,
    )
    return None


def _setup_signal_handlers(shutting_down: threading.Event) -> None:
    """Install SIGTERM/SIGINT handlers that set *shutting_down*.

    Signals arriving *after* shutdown has begun are ignored, and that is
    deliberate. The state save is in progress at that point, and a
    second Ctrl-C that reached it would leave the mmap'd state and its
    JSON projection inconsistent — precisely the corruption the
    project's operational notes exist to prevent. An impatient
    operator pressing Ctrl-C again must not be able to abort a save.
    Covered by
    `test_repeated_shutdown_signal_does_not_interrupt_save`.

    Shutdown is still guaranteed to terminate: the child-stopping path
    escalates on its own (SIGTERM, bounded wait, then SIGKILL), so a
    wedged daemon cannot make this process unkillable regardless of
    what the signal handler does.
    """

    def handle_signal(signum: int, frame: FrameType | None) -> None:
        """Signal handler that sets the shutdown flag and raises on SIGINT."""
        if shutting_down.is_set():
            return
        shutting_down.set()
        if signum in (signal.SIGINT, signal.SIGTERM):
            # Raising KeyboardInterrupt interrupts input() immediately.
            # Without this, the custom handler swallows the signal and
            # input() keeps blocking — Ctrl-C appears to do nothing.
            raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def _start_sleep_watcher(mind: Mind, shutting_down: threading.Event) -> None:
    """Start a background thread that reports sleep/wake transitions.

    The watcher only *reports* — the mind owns every transition (the
    self-sleep urge, /sleep and /wake, and the heartbeat's body-state
    sync that follows the daemon's phase). There is no autonomous
    sleep or wake anywhere: body and mind are always asleep or awake
    at the same time, and the transitions are hers.
    """

    def _sleep_watcher() -> None:
        was_sleeping = mind.is_sleeping
        while not shutting_down.is_set():
            if mind.is_sleeping != was_sleeping:
                was_sleeping = mind.is_sleeping
                if was_sleeping:
                    logger.info("[genesis] It drifted into sleep and is dreaming.")
                    logger.info("[genesis]   Use /wake to wake it, or watch its dreams below.")
                else:
                    try:
                        status = mind.sleep_status()
                        logger.info("[genesis] It woke up.")
                        logger.info("[genesis]   Its state before waking:")
                        for line in status.split("\n"):
                            logger.info(f"[genesis]   {line}")
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"sleep status on wake failed: {e}")
            shutting_down.wait(timeout=5.0)

    named_thread("sleep-watcher", _sleep_watcher, comm="g-sleep-watch").start()


def _speak_thought(
    text: str,
    mind: Mind,
    voice: Voice,
    dedup_voice: VoiceDedup | None,
) -> None:
    """Speak a thought aloud with emotional prosody.

    Uses the shared VoiceDedup wrapper so that spontaneous thoughts
    are deduped against ALL recent speech (speech urges, ambient
    chime-ins, other thoughts) — not just against other spontaneous
    thoughts. This prevents it from repeating similar-sounding
    content regardless of which path triggered it.

    Never speaks during sleep — it shouldn't be disturbed by it
    own thoughts while resting. Thoughts still print to the terminal
    (dreams appear as genesis~ lines), but no audio.
    """
    if mind.is_sleeping:
        return
    if not voice.is_available():
        return
    if not text.strip():
        return
    try:
        state = mind.get_state()
        if state and state.emotion:
            emo = state.emotion
            if dedup_voice:
                dedup_voice.speak(
                    text,
                    alertness=emo.alertness,
                    valence=emo.valence,
                    caution=emo.caution,
                    creativity=emo.creativity,
                    blocking=False,
                )
            else:
                voice.speak(
                    text,
                    alertness=emo.alertness,
                    valence=emo.valence,
                    caution=emo.caution,
                    creativity=emo.creativity,
                    blocking=False,
                )
        else:
            if dedup_voice:
                dedup_voice.speak(text, blocking=False)
            else:
                voice.speak(text, blocking=False)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[speak] voice speak failed: {e}")


class _LoopState:
    """Mutable state shared between the interactive loop and its helpers."""

    __slots__ = (
        "awaiting_input",
        "last_speak_ts",
        "last_thought_ts",
        "polling_active",
        "suppress_thoughts",
    )

    def __init__(self) -> None:
        """Initialize loop state with sensible defaults."""
        self.last_thought_ts: float = time.time()
        self.last_speak_ts: float = 0.0
        self.polling_active: bool = True
        self.suppress_thoughts: bool = False
        self.awaiting_input: bool = False


def _should_speak_thought(kind: str, state: _LoopState) -> bool:
    """Decide whether Genesis should speak a thought aloud.

    It speaks when:
    - The thought kind is in ALWAYS_SPEAK_KINDS (expression, distress,
      question) — these are things it wants/needs to say. Questions
      bypass the cooldown so it always asks them directly.
    - Or, for other thoughts, with a small probability — sometimes
      its internal musings are strong enough to share

    Cooldown prevents it from speaking too often (except questions).
    """
    now = time.time()
    # Questions always speak — they're directed at the user and
    # should not be suppressed by cooldown.
    if kind == "question":
        state.last_speak_ts = now
        return True
    if now - state.last_speak_ts < SPEAK_THOUGHT_COOLDOWN:
        return False
    if kind in ALWAYS_SPEAK_KINDS:
        state.last_speak_ts = now
        return True
    import random as _r

    if _r.random() < SPEAK_THOUGHT_PROBABILITY:
        state.last_speak_ts = now
        return True
    return False


def _print_idle_thoughts(
    state: _LoopState,
    mind: Mind,
    voice: Voice,
    thought_collector: ThoughtCollector,
    out_lock: threading.Lock,
    dedup_voice: VoiceDedup | None,
) -> None:
    """Print and optionally speak thoughts that arrived while the user was idle."""
    new_thoughts = thought_collector.get_and_clear_since(state.last_thought_ts)
    # Filter out internal self-regulation status (throttle/unthrottle)
    # — these are not meaningful to the user and make it look stressed
    # when it's just managing its own CPU load. The learner emits
    # these as kind="learning" with text="throttled"/"recovered".
    new_thoughts = [
        t
        for t in new_thoughts
        if t.get("text", "").lower() not in ("throttled", "recovered")
        and t.get("kind") not in ("throttle", "unthrottle")
    ]
    if new_thoughts:
        with out_lock:
            # Separate questions (direct address) from other thoughts.
            # Questions are printed as genesis> lines — it's asking
            # the user directly, not just musing internally.
            questions = [t for t in new_thoughts if t.get("kind") == "question"]
            other = [t for t in new_thoughts if t.get("kind") != "question"]
            if other:
                _print_thoughts(other)
            for q in questions:
                text = q.get("text", "")
                if text:
                    logger.info(f"  genesis> (asking) {text}")
                    logger.info("")
            # If the main loop is parked at an idle prompt, redraw
            # it so its thought doesn't leave the cursor stranded on
            # a stale prompt line.
            if state.awaiting_input:
                sys.stdout.write("  you> ")
                sys.stdout.flush()
        state.last_thought_ts = max(t.get("timestamp", state.last_thought_ts) for t in new_thoughts)
        # Feed thoughts to the speech volition queue so the speech
        # urge can act on them when they cross threshold. This is
        # the intended path: thoughts accumulate in the queue, the
        # urge grows from curiosity and queue size, and when it
        # fires, it speaks the queued utterance through on_speak.
        for t in new_thoughts:
            text = t.get("text", "")
            if text:
                mind.offer_utterance(text)
        # Also speak some thoughts directly — the volition path
        # has its own timing, but some thoughts are strong enough
        # to share immediately.
        for t in new_thoughts:
            kind = t.get("kind", "thought")
            if _should_speak_thought(kind, state):
                _speak_thought(t.get("text", ""), mind, voice, dedup_voice)


def _poll_loop(
    state: _LoopState,
    mind: Mind,
    voice: Voice,
    thought_collector: ThoughtCollector,
    out_lock: threading.Lock,
    dedup_voice: VoiceDedup | None,
) -> None:
    """Background ticker for idle thoughts.

    Event-driven: blocks on the thought condition and wakes the instant
    a thought arrives, rather than polling on a fixed sleep.
    """
    while state.polling_active:
        thought_collector.wait(LIVE_THOUGHT_POLL_INTERVAL)
        if not state.polling_active:
            break
        if state.suppress_thoughts:
            continue
        _print_idle_thoughts(state, mind, voice, thought_collector, out_lock, dedup_voice)


def _run_interactive_loop(
    mind: Mind,
    voice: Voice,
    thought_collector: ThoughtCollector,
    shutting_down: threading.Event,
    dedup_voice: VoiceDedup | None = None,
) -> None:
    """Run the main interactive read/respond loop until shutdown."""
    out_lock = threading.Lock()
    state = _LoopState()
    mic = _init_mic()

    poll_thread = named_thread(
        "cli-thought-poll",
        _poll_loop,
        args=(state, mind, voice, thought_collector, out_lock, dedup_voice),
        comm="g-thought-poll",
    )
    poll_thread.start()

    try:
        while not shutting_down.is_set():
            with out_lock:
                sys.stdout.write("  you> ")
                sys.stdout.flush()
            # Wait for the user to start typing. While the prompt
            # sits idle the ticker stays live — its thoughts print
            # as they arrive and the prompt is redrawn underneath.
            # Suppression engages only once stdin actually has data
            # (i.e. it can't mash a line the user is mid-way
            # through), and buffered thoughts drain after submit.
            state.awaiting_input = True
            try:
                while not shutting_down.is_set() and not _stdin_ready(0.5):
                    pass
            finally:
                state.awaiting_input = False
            if shutting_down.is_set():
                break
            state.suppress_thoughts = True
            user_input = _read_input("")
            state.suppress_thoughts = False
            if user_input is None:
                logger.info("")
                break
            if not user_input:
                continue

            # Typed input never reaches the log — echo it to her
            # channel so the mirrored session shows both sides.
            feed = getattr(mind, "_public_feed", None)
            if feed is not None:
                feed.post_user(user_input)

            # Always drain any thoughts that arrived while the user was typing
            _print_idle_thoughts(state, mind, voice, thought_collector, out_lock, dedup_voice)

            if user_input.startswith("/"):
                # Pause background thought generation and suppress the
                # poll loop so thoughts don't bury the command output.
                # This mirrors mind.respond() which pauses inner_life
                # and learner during conversation.
                mind.inner_life.pause()
                mind.learner.pause()
                mind._suppress_volition = True
                state.suppress_thoughts = True
                try:
                    handled = _handle_slash_command(
                        user_input,
                        mind,
                        voice,
                        mic,
                        out_lock,
                        thought_collector,
                    )
                finally:
                    state.suppress_thoughts = False
                    # Don't release volition while a popped question is
                    # still awaiting its answer — the open turn holds
                    # it until the reply lands or the quiet window lapses.
                    mind._suppress_volition = mind.awaiting_question_answer
                    # Drain any thoughts that buffered during the command
                    # so they don't flood the next prompt.
                    buffered = thought_collector.get_all()
                    if buffered:
                        with out_lock:
                            _print_thoughts(buffered)
                    # Resume background activity after a short delay.
                    # _defer uses a daemon timer — a plain Timer from
                    # the main thread is non-daemon and would delay
                    # interpreter exit after /quit.
                    mind._defer(5.0, mind._resume_background)
                if handled:
                    state.last_thought_ts = time.time()
                    continue
                # /quit or /exit returns False and we should break
                if user_input.lower() in ("/quit", "/exit", "/q"):
                    break
                continue

            # Normal conversation
            state.last_thought_ts = _handle_conversation(
                user_input,
                mind,
                voice,
                thought_collector,
                out_lock,
            )

    except KeyboardInterrupt:
        logger.info("")
    finally:
        state.polling_active = False
        thought_collector.wake()
        shutting_down.set()


def _handle_slash_command(
    user_input: str,
    mind: Mind,
    voice: Voice,
    mic,
    out_lock: threading.Lock,
    thought_collector: ThoughtCollector,
) -> bool:
    """Handle a slash command. Returns True if handled, False for /quit."""
    cmd = user_input.lower()

    if cmd in ("/quit", "/exit", "/q"):
        logger.info("  Goodnight.")
        return False

    if cmd == "/help":
        _print_help()
        return True

    if cmd == "/voice":
        with out_lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        text = _handle_voice_command(mic)
        if text is None:
            return True
        with out_lock:
            logger.info(f"  you> {text}")
        # During sleep, voice commands can't reach its either —
        # it's unreachable until it wakes on its own or /wake
        # is used. See _handle_conversation for the full rationale.
        if mind.is_sleeping:
            with out_lock:
                logger.info("  (It's asleep. Type /wake to wake it.)")
            return True
        with out_lock:
            logger.info("  genesis~ (thinking...)")
        try:
            response = mind.respond(text)
        except Exception as e:
            logger.exception(f"respond failed: {e}")
            with out_lock:
                logger.info("\n  genesis> (error)\n")
            return True
        with out_lock:
            logger.info(f"\n  genesis> {response}\n")
            _speak_response(voice, mind, response)
        return True

    if cmd == "/voice-mode":
        _handle_voice_mode(mind, mic, out_lock, voice)
        return True

    # All other slash commands
    result = _handle_command(mind, user_input)
    with out_lock:
        if result:
            logger.info(result)
    return True


def _handle_conversation(
    user_input: str,
    mind: Mind,
    voice: Voice,
    thought_collector: ThoughtCollector,
    out_lock: threading.Lock,
) -> float:
    """Handle a normal (non-slash) conversation turn. Returns new thought timestamp."""
    # Mark user activity BEFORE responding. The heartbeat's waking-side
    # signals run in a separate thread and can fire between the user
    # pressing Enter and respond() being called — an active user must
    # read as active there.
    mind.mark_user_activity()
    # Drain stale background thoughts BEFORE showing "thinking..." so
    # they don't appear between the response and the next prompt.
    # These thoughts were generated while the user was typing or
    # before — they're not part of this conversation turn.
    thought_collector.get_all()
    # Submitted terminal input follows Mind.respond()'s explicit wake path.
    # Ambient speech remains gated during sleep.
    with out_lock:
        logger.info("  genesis~ (thinking...)")
    try:
        if mind.is_gaming and user_input.startswith("FRAME "):
            response = mind.game_turn(user_input)
        else:
            response = mind.respond(user_input)
    except Exception as e:  # noqa: BLE001
        with out_lock:
            logger.info(f"\n  genesis> (error: {e})\n")
        return time.time()

    with out_lock:
        # Only print thoughts that were generated DURING this turn
        # (after mind.respond paused background activity). Stale
        # thoughts were already drained above.
        new_thoughts = thought_collector.get_all()
        # Filter out internal status messages that aren't meaningful
        # to the user (throttle/unthrottle are self-regulation noise).
        user_visible = [
            t
            for t in new_thoughts
            if t.get("text", "").lower() not in ("throttled", "recovered")
            and t.get("kind") not in ("throttle", "unthrottle")
        ]
        _print_thoughts(user_visible)
        logger.info(f"\n  genesis> {response}\n")
        cog = _get_cognitive_state_dict(mind)
        rendered = _render_cognitive_state(cog)
        if rendered:
            logger.info(rendered)
        _speak_response(voice, mind, response)
        logger.info("")
    return time.time()


def _stop_child(proc: subprocess.Popen | _AdoptedDaemon, name: str, timeout: float) -> bool:
    """Stop a child process, escalating rather than blocking forever.

    The previous version called `proc.wait()` with no timeout once the
    first wait expired. A child that hung — a daemon blocked on a dead
    socket, a retina wedged on a camera — therefore made shutdown
    uninterruptible: the process could not be ended by Ctrl-C, because
    the signal handler was already consumed, and not by SIGTERM, for
    the same reason. The only exit was `kill -9`, which AGENTS.md and
    the README both forbid, precisely because it can lose unconsolidated
    memory and leave on-disk state inconsistent.

    So: SIGTERM, wait up to `timeout`, then SIGKILL, then wait a short
    bounded period. SIGKILL is the last resort, reached only after a
    graceful request has demonstrably failed, and it says so on stderr.
    Returning False makes the caller report an incomplete shutdown
    rather than pretending everything went well.

    Note this is not the same as advising `kill -9` at the shell: the
    graceful path is still tried first, and the escalation is bounded
    and reported.
    """
    if proc.poll() is not None:
        return proc.returncode == 0

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired as e:
        # Expected fallthrough: SIGTERM didn't stop it in time, escalate
        # to SIGKILL below. Logged (not silent) so shutdown hangs are visible.
        logger.debug(f"{name} SIGTERM timeout ({e}); escalating to SIGKILL")

    print(
        f"[genesis] {name} did not stop within {timeout:g}s after SIGTERM; escalating to SIGKILL.",
        file=sys.stderr,
    )
    try:
        proc.kill()
    except OSError as e:
        # Process already gone (raced with natural exit) — nothing to kill.
        logger.debug(f"{name} kill skipped, already exited: {e}")
    try:
        proc.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        print(
            f"[genesis] {name} could not be killed — it may be stuck in "
            f"uninterruptible I/O. Continuing shutdown; the state file was "
            f"already saved by the mind above.",
            file=sys.stderr,
        )
        return False

    if proc.returncode != 0:
        print(
            f"[genesis] {name} exited with status {proc.returncode}.",
            file=sys.stderr,
        )
        return False
    return True


def _request_daemon_shutdown(data_dir: str) -> bool:
    """Ask the daemon to stop itself over IPC. True if it acknowledged.

    Falls back silently to the signal path (via the caller's
    ``_stop_child``) if the socket is gone, the protocol mismatches, or
    the daemon does not answer — shutdown must not depend on this
    succeeding.
    """
    if not data_dir:
        return False
    socket_path = os.path.join(data_dir, "genesis.sock")
    if not os.path.exists(socket_path):
        return False
    try:
        from genesis_client import GenesisClient

        client = GenesisClient(socket_path)
        client.connect()
        try:
            return bool(client.shutdown())
        finally:
            _close_client_quietly(client)
    except Exception as e:  # noqa: BLE001 - shutdown must not fail here
        logger.debug(f"daemon IPC shutdown unavailable ({e}); using signal path")
        return False


def _close_client_quietly(client: object) -> None:
    """Close a GenesisClient if it exposes a close/shutdown method."""
    for attr in ("close", "disconnect"):
        fn = getattr(client, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"client {attr} failed during shutdown: {e}")
            return


def _shutdown(
    ambient_listener: AmbientListener | None,
    mind: Mind | None,
    daemon_proc: subprocess.Popen | _AdoptedDaemon | None,
    retina_proc: subprocess.Popen | None = None,
    data_dir: str = "",
    auditory_cortex: AuditoryCortex | None = None,
    voice: Voice | VoiceDedup | None = None,
    spore: Callable[[], bool] | None = None,
) -> bool:
    """Gracefully stop all initialized components.

    Returns False when the cognitive layer reports an incomplete
    shutdown, most importantly a failed final state save.
    """
    clean_shutdown = True
    if mind is not None:
        print("[genesis] Stopping cognitive mind...", file=sys.stderr)
    if auditory_cortex:
        auditory_cortex.stop()
    if ambient_listener:
        ambient_listener.stop()
    if voice is not None:
        # In-flight and queued utterances outlive the urge that queued
        # them — she goes quiet with the teardown, not after it.
        voice.stop()
    if mind is not None:
        clean_shutdown = bool(mind.stop())
    if spore is not None:
        # The organism's last act on this host: mind flushed, body
        # still answering SYNC — shed the freshest possible spore.
        try:
            spore()
        except Exception as e:  # noqa: BLE001
            note_swallowed("genesis_cli.spore.final", e)
    if daemon_proc is not None:
        print("[genesis] Stopping subcognitive daemon...", file=sys.stderr)
        # Ask over IPC first. The daemon implements SHUTDOWN and
        # performs its own ordered flush ("Subcognitive stopped.
        # Goodnight."), which is a genuinely graceful stop — whereas
        # SIGTERM relies on its handler and the same flush. `client.shutdown()`
        # was implemented on both sides and never called from here.
        if not _request_daemon_shutdown(data_dir):
            clean_shutdown &= _stop_child(daemon_proc, "subcognitive daemon", 5)
    if retina_proc is not None:
        print("[genesis] Stopping retina...", file=sys.stderr)
        clean_shutdown &= _stop_child(retina_proc, "retina", 3)
    # Clean up PID files so run.sh --stop doesn't target stale PIDs.
    if data_dir:
        for name in (_PID_FILE_CLI, _PID_FILE_DAEMON, _PID_FILE_RETINA):
            _remove_pid_file(data_dir, name)
    if clean_shutdown and mind is not None:
        print("[genesis] Goodnight.", file=sys.stderr)
    elif not clean_shutdown:
        # Distinguish "the state save actually failed" (critical) from
        # "a background thread or child process lingered past its join
        # budget" (noncritical — the save can still have succeeded).
        # mind._shutdown_save_ok is True/False once stop() ran; None
        # means the mind never reached the save step.
        if mind is not None and getattr(mind, "_shutdown_save_ok", True) is False:
            print(
                "[genesis] Shutdown incomplete — cognitive state may not be saved.",
                file=sys.stderr,
            )
        else:
            print(
                "[genesis] Shutdown incomplete — a background component "
                "did not stop cleanly (cognitive state was saved).",
                file=sys.stderr,
            )
    return clean_shutdown


def _exit_watchdog(exit_code: int) -> None:
    """os._exit fallback when interpreter finalization outlives its budget.

    Runs on a daemon timer armed once teardown is done. Prints the
    surviving thread names with a raw os.write — buffered logging is
    unreliable this late in finalization — then hard-exits so a wedged
    library thread can never keep the process (and her systemd unit)
    alive past teardown. See _EXIT_WATCHDOG_S.
    """
    names = ",".join(t.name for t in threading.enumerate())
    os.write(
        2,
        f"[genesis] exit watchdog fired — force-exiting; threads still alive: {names}\n".encode(),
    )
    os._exit(exit_code)


def _arm_exit_watchdog(exit_code: int) -> None:
    """Arm the post-teardown force-exit timer.

    Must run on a *daemon* Timer: Timer inherits the creating thread's
    daemon flag, and a non-daemon watchdog would itself hold the
    interpreter open for its full delay. As a daemon it dies silently
    with a normal exit and only matters when finalization has hung.
    """
    watchdog = threading.Timer(_EXIT_WATCHDOG_S, _exit_watchdog, args=(exit_code,))
    watchdog.daemon = True
    watchdog.start()


def _load_agent_env() -> None:
    """Load KEY=VALUE lines from the installed agent's env file.

    ``$XDG_CONFIG_HOME/genesis/agent.env`` (or ``$GENESIS_ENV_FILE``)
    is written by scripts/install.sh — e.g. the Discord webhook mirror
    for a resident deployment. Real environment variables always win:
    values are applied with ``setdefault`` semantics.
    """
    env_file = os.environ.get("GENESIS_ENV_FILE")
    if not env_file:
        config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
        env_file = os.path.join(config_home, "genesis", "agent.env")
    try:
        lines = Path(env_file).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _console_main() -> int:
    """Entry point for the installed ``genesis`` command.

    run.sh exports GENESIS_RUN to prove the CLI came up through the
    supported launcher; the installed console script *is* a supported
    launcher, so it sets the flag itself.
    """
    _load_agent_env()
    os.environ["GENESIS_RUN"] = "1"
    return main()


def _attach_session_mirror(feed: DiscordFeed | None) -> tuple:
    """Attach the session mirror to her channel feed, if one exists.

    Every logged line and every stderr notice posts to the webhook;
    the interactive loop echoes typed input explicitly. Her broadcast
    posts use the same transport — a ``genesis~ [broadcast]`` echo is
    excluded from the mirror so it doesn't land twice. Returns
    ``(handler, real_stderr)`` for teardown, or ``(None, None)`` when
    no channel is configured.
    """
    if feed is None:
        return None, None
    handler = FeedLogHandler(feed)
    logging.getLogger().addHandler(handler)
    real_stderr = sys.stderr
    sys.stderr = StderrTee(real_stderr, feed)  # type: ignore[assignment]
    return handler, real_stderr


def main() -> int:
    """Main entry point for the Genesis CLI."""
    if not os.environ.get("GENESIS_RUN"):
        print(
            "Error: Genesis must be started with ./run.sh (source "
            "checkout) or the `genesis` command (installed).",
            file=sys.stderr,
        )
        return 1

    args = _parse_cli_args()
    data_dir = os.path.abspath(args.data_dir)

    if args.tail:
        return _tail_daemon_log(data_dir)

    if args.stop:
        if _any_running(data_dir):
            print("  Stopping Genesis...", file=sys.stderr)
            if _stop_all(data_dir) != 0:
                return 1
            print("  Genesis stopped.", file=sys.stderr)
        else:
            print("  Genesis is not running.", file=sys.stderr)
        return 0

    daemon_socket = args.daemon_socket or os.path.join(data_dir, DEFAULT_DAEMON_SOCKET)

    # Owner-only: the dir contains the daemon's IPC socket and state.
    os.makedirs(data_dir, mode=0o700, exist_ok=True)

    if _acquire_cli_lock(data_dir) is None:
        return 1

    # Write the CLI PID file so run.sh --stop can target this exact
    # process instead of using broad pkill patterns.
    _write_pid_file(data_dir, _PID_FILE_CLI, os.getpid())

    if args.germinate is not None:
        # The CLI lock is already held, so a live organism's state can
        # never be clobbered by a restore. The dir may already hold
        # this start's own lock/pid artifacts — _germinate refuses on
        # identity files only.
        if not _germinate(args.germinate or _spore_dir_for(data_dir), data_dir):
            return 1

    # Install handlers and initialize handles before any child process
    # starts. Every initialization step after this point is covered by
    # the same cleanup path, so a voice, microphone, or emotion-setup
    # failure cannot orphan the daemon/retina.
    shutting_down = threading.Event()
    _setup_signal_handlers(shutting_down)

    # Her public channel — opened before the daemon starts so the
    # whole session, startup notices included, reaches the mirror.
    # Mind adopts this same feed for her broadcast posts, so mirrored
    # lines and her posts share one paced queue.
    feed = DiscordFeed.from_env()
    feed_handler, real_stderr = _attach_session_mirror(feed)

    daemon_proc: subprocess.Popen | _AdoptedDaemon | None = None
    retina_proc: subprocess.Popen | None = None
    mind: Mind | None = None
    rebirth: _Rebirth | None = None
    ambient_listener: AmbientListener | None = None
    auditory_cortex: AuditoryCortex | None = None
    voice: Voice | None = None
    dedup_voice: VoiceDedup | None = None
    exit_code = 0

    try:
        daemon_proc = _start_daemon_process(data_dir, daemon_socket)
        if daemon_proc is None:
            exit_code = 1
        else:
            _write_pid_file(data_dir, _PID_FILE_DAEMON, daemon_proc.pid)
            print("[genesis] Daemon started.", file=sys.stderr)

            # ── Start the retina (camera) ──
            retina_proc = _start_retina_process(data_dir)

            # ── Start the cognitive mind ──
            print("[genesis] Starting cognitive mind...", file=sys.stderr)
            if args.offline:
                print(
                    "[genesis] Offline mode — learning paused, conversation and art active.",
                    file=sys.stderr,
                )
            mind = Mind(daemon_socket, offline=args.offline, public_feed=feed)
            try:
                mind.start()
            except (OSError, ConnectionError) as e:
                print(f"Error: cannot connect to daemon: {e}", file=sys.stderr)
                exit_code = 1
            else:
                # The organism repairs its own process tree from here:
                # the body resurrects around its persisted state and
                # the mind re-inhabits it, the eye re-opens — and a
                # body that will not reincarnate ends her whole rather
                # than leaving a mind running bodiless. One thing.
                rebirth = _Rebirth(data_dir, daemon_socket, shutting_down)
                rebirth.set_children(daemon=daemon_proc, retina=retina_proc)
                rebirth.set_mind(mind)
                rebirth.start()
                (voice, dedup_voice, ambient_listener, auditory_cortex) = _run_after_mind_start(
                    mind, args, shutting_down
                )
    finally:
        exit_code = _teardown_session(
            shutting_down,
            rebirth,
            ambient_listener,
            mind,
            daemon_proc,
            retina_proc,
            data_dir,
            auditory_cortex,
            dedup_voice or voice,
            feed_handler,
            real_stderr,
            feed,
            exit_code,
        )
    return exit_code


def _teardown_session(
    shutting_down: threading.Event,
    rebirth: _Rebirth | None,
    ambient_listener: AmbientListener | None,
    mind: Mind | None,
    daemon_proc: subprocess.Popen | _AdoptedDaemon | None,
    retina_proc: subprocess.Popen | None,
    data_dir: str,
    auditory_cortex: AuditoryCortex | None,
    voice: Voice | None,
    feed_handler: logging.Handler | None,
    real_stderr,
    feed: DiscordFeed | None,
    exit_code: int,
) -> int:
    """Shut the whole organism down and return the final exit code."""
    # A signal handler sets this before the finally runs; a clean
    # /quit does not. Distinguishes "user told her to stop" from
    # "she exited on her own" for the systemd exit code below.
    externally_stopped = shutting_down.is_set()
    # Mark shutdown before saving so subsequent signals do not raise
    # KeyboardInterrupt during persistence. Repeated Ctrl+C must not
    # abandon a state save or force-kill a child still flushing data.
    shutting_down.set()
    # A self-termination — the watcher giving up on an
    # unreincarnatable body — is a death, not a stop: the exit
    # must read as failure so a supervisor resurrects the whole.
    self_terminated = rebirth is not None and rebirth.self_terminated
    if rebirth is not None:
        rebirth.stop()
        daemon_proc, retina_proc = rebirth.handles()
    if not _shutdown(
        ambient_listener,
        mind,
        daemon_proc,
        retina_proc,
        data_dir,
        auditory_cortex,
        voice,
        spore=(rebirth.final_spore if rebirth is not None else None),
    ):
        exit_code = 1
        # Under systemd, a nonzero exit on a signal-driven stop is
        # indistinguishable from a crash — Restart=on-failure would
        # resurrect her right after `genesis --stop` put her down.
        # A deliberate stop stays down; the stderr warning above
        # already recorded the unclean shutdown.
        if externally_stopped and os.environ.get("INVOCATION_ID"):
            exit_code = 0
    if self_terminated:
        # She ended the organism herself because the body would
        # not reincarnate — a death, not a stop. Non-zero so
        # Restart=on-failure resurrects the whole, never one half.
        exit_code = 1
    # Detach the mirror after _shutdown so teardown notices still
    # reach the feed's drain window; a feed the mind adopted is
    # already closed by its lifecycle stop — this close covers the
    # case where the mind was never constructed.
    if feed_handler is not None:
        logging.getLogger().removeHandler(feed_handler)
    if real_stderr is not None:
        sys.stderr = real_stderr
    if feed is not None:
        feed.close()
    # From here on, every remaining step is best-effort tail work
    # plus interpreter finalization — none of which may hold the
    # process past _EXIT_WATCHDOG_S.
    _arm_exit_watchdog(exit_code)
    return exit_code


def _run_after_mind_start(
    mind: Mind, args: argparse.Namespace, shutting_down: threading.Event
) -> tuple[Voice | None, VoiceDedup | None, AmbientListener | None, AuditoryCortex | None]:
    """Wire voice/listeners, greet, then run headless or interactive.

    Returns the components that were started so the caller's
    ``finally`` can shut them down.
    """
    # ── Wire voice ──
    voice, dedup_voice = _wire_voice(mind)

    # ── Start ambient listener (microphone) ──
    ambient_listener = _setup_ambient_listener(mind, voice, dedup_voice)

    # ── Start auditory cortex (non-speech recognition) ──
    auditory_cortex = _setup_auditory_cortex(mind, voice)

    # ── Setup live thought collector ──
    thought_collector = ThoughtCollector()
    mind.add_live_thought_listener(thought_collector.on_live_thought)

    # ── Print welcome ──
    emotion = mind.feel()
    _print_welcome(emotion.label, mind.is_sleeping)

    # ── Autonomous sleep/wake watcher ──
    _start_sleep_watcher(mind, shutting_down)

    if args.headless:
        # Resident mode: the mind lives in its own threads; this
        # thread just waits for SIGTERM/SIGINT. There is no prompt
        # to read — stdin under systemd is /dev/null, and input()
        # would exit instantly. The signal handler's
        # KeyboardInterrupt exists to break input(); with no
        # input() it is noise, so the wait absorbs it like the
        # interactive loop does.
        print(
            "[genesis] Resident mode — running headless. Stop with SIGTERM or `genesis --stop`.",
            file=sys.stderr,
        )
        try:
            while not shutting_down.is_set():
                shutting_down.wait(5.0)
        except KeyboardInterrupt as e:
            # The SIGINT handler owns the flag; this absorb is
            # only the wait waking early.
            note_swallowed("genesis_cli.resident.wait", e)
    else:
        # ── Interactive loop ──
        _run_interactive_loop(mind, voice, thought_collector, shutting_down, dedup_voice)
    return voice, dedup_voice, ambient_listener, auditory_cortex


def _start_retina_process(data_dir: str) -> subprocess.Popen | None:
    """Start the retina (camera) subprocess, reaping any stale one.

    The retina is a standalone Rust binary that captures camera
    frames into shared memory. We manage it as a child process so
    it's automatically stopped when the CLI shuts down.
    """
    retina_path = _find_binary("retina")
    if not retina_path:
        print("[genesis] Retina not built — camera unavailable.", file=sys.stderr)
        print("[genesis] Build with: cargo build --release", file=sys.stderr)
        return None
    # run.sh checks for existing processes belonging to this instance.
    # Do not kill camera processes from other data directories merely
    # because they use the same project binary.
    retina_proc = _start_retina(retina_path, data_dir)
    if retina_proc:
        _write_pid_file(data_dir, _PID_FILE_RETINA, retina_proc.pid)
        print(f"[genesis] Retina started (pid={retina_proc.pid}).", file=sys.stderr)
    else:
        print("[genesis] Retina failed to start — camera unavailable.", file=sys.stderr)
    return retina_proc


def _wire_voice(mind: Mind) -> tuple[Voice, VoiceDedup | None]:
    """Wire the voice pipeline: TTS, dedup, and emotional modulation."""
    voice = Voice()
    if not voice.is_available():
        print("[genesis] Voice: not available (no Piper/espeak)", file=sys.stderr)
        return voice, None

    print(f"[genesis] Voice: {voice.describe()}", file=sys.stderr)
    # Wrap with dedup so it never repeats itself aloud, regardless
    # of which path triggers speech (speech urge, spontaneous
    # thought, ambient chime-in). Conversation responses bypass
    # the dedup by calling the raw voice directly.
    dedup_voice = VoiceDedup(voice)

    def _on_speak(text: str) -> None:
        """Speak with emotional modulation from its current state.

        Autonomous speech (speech urge, warnings) should carry the
        same emotional modulation as conversational speech — its
        voice should reflect how it feels, whether it's answering
        a question or speaking its own thoughts.

        Never speaks during sleep — defensive guard. The callers
        (speech urge, warn()) already gate on sleep, but this
        ensures no path can bypass the sleep gate.
        """
        if mind.is_sleeping or not mind._running:
            return
        try:
            state = mind.get_state()
            if state and state.emotion:
                emo = state.emotion
                dedup_voice.speak(
                    text,
                    alertness=emo.alertness,
                    valence=emo.valence,
                    caution=emo.caution,
                    creativity=emo.creativity,
                    blocking=False,
                )
            else:
                dedup_voice.speak(text, blocking=False)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"on_speak failed: {e}")

    mind.on_speak = _on_speak
    return voice, dedup_voice


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # The signal handler raises to interrupt input(); cleanup has
        # already run by the time it reaches here. Exit quietly with
        # the conventional 128+SIGINT code instead of a traceback.
        sys.exit(130)
