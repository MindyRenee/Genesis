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
    /journal    — read Genesis's journal
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
    /register-face <name> — teach it your face (look at the camera)
    /faces       — show known faces and who it last saw
    /mission [text] — set or show its top-level mission
    /sleep       — put it to sleep (wakes on its own when rested)
    /nap         — short nap, wakes on its own when rested
    /wake        — wake it from sleep or meditation
    /meditate [secs] — put it into meditation (default 60s)
    /teach [topic] — put it into teaching mode (focused learning)
    /endteach    — exit teaching mode, resume normal operation
    /teach-questions — answer its queued questions one at a time
    /sleep-aid   — emergency sleep aid for stress-induced insomnia
    /web-history [N] — see what it's been looking at online (last N, default 20)
    /quit        — exit

Spontaneous thoughts appear live as genesis~ lines while you are idle.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import logging
import lzma
import os
import queue
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from genesis_client.protocol import PHASE_ACTIVE, PHASE_ALERT, PHASE_NREM, PHASE_REM
from genesis_cognitive.ambient import AmbientListener, contains_wake_word, strip_wake_word
from genesis_cognitive.auditory import AuditoryCortex, SoundEvent
from genesis_cognitive.concepts import RelationType
from genesis_cognitive.config import default_data_dir
from genesis_cognitive.mind import Mind
from genesis_cognitive.speech import Voice, VoiceInput

# Script directory — used to locate the project root. Python already
# adds the script's directory to sys.path[0], so genesis_cognitive and
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
logger = logging.getLogger(__name__)

# Max wait (seconds) for the idle ticker to block on the thought
# condition before re-checking shutdown. Thoughts normally surface
# instantly via notify; this is only a spurious-wakeup safety ceiling.
LIVE_THOUGHT_POLL_INTERVAL = 2.0

# Default data directory (XDG-compliant, persists across reboots).
# Resolved via GENESIS_DATA_DIR → $XDG_DATA_HOME/genesis-public —
# this instance's state is never shared with any other Genesis.
DEFAULT_DATA_DIR = str(default_data_dir())
DEFAULT_DAEMON_SOCKET = "genesis.sock"

# File descriptor for the per-data-dir CLI singleton lock.
_CLI_LOCK_FD: int | None = None

# PID file paths — written so run.sh --stop and the EXIT trap can
# kill exact processes instead of using broad pkill patterns that
# might match unrelated user processes.
_PID_FILE_CLI = "genesis_cli.pid"
_PID_FILE_DAEMON = "genesis_daemon.pid"
_PID_FILE_RETINA = "genesis_retina.pid"

# How long to wait for the daemon to start (seconds)
DAEMON_START_TIMEOUT = 10.0

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
        self._recent: collections.deque[str] = collections.deque(
            maxlen=VOICE_DEDUP_HISTORY
        )
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
                        logger.debug(
                            "[voice-dedup] skipping — too similar to recent"
                        )
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
CHIME_IN_TOPICS = [
    "feel", "feeling", "mood", "sad", "happy", "angry", "scared",
    "afraid", "worried", "anxious", "stress", "stressed", "tired",
    "sleep", "dream", "remember", "memory", "forget", "learn",
    "learning", "think", "thinking", "mind", "cognitive",
    "ai", "robot", "alive", "real", "self", "aware", "genesis",
]

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


def _find_daemon_binary() -> str | None:
    """Find the genesis-daemon binary.

    There is one canonical build: ``cargo build --release``. The release
    binary is the only one Genesis will run — no debug fallback, no
    ambiguity about which binary is live.
    """
    project_root = Path(SCRIPT_DIR).parent
    path = project_root / "target" / "release" / "genesis-daemon"
    if path.exists() and os.access(path, os.X_OK):
        return str(path)
    return None


def _project_root() -> Path:
    """The repository root (parent of the python/ directory)."""
    return Path(SCRIPT_DIR).parent


def _find_retina_binary() -> str | None:
    """Find the retina binary (release build only)."""
    path = _project_root() / "target" / "release" / "retina"
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
                logger.debug(f'silent except: {e}')
        else:
            dst = f"{log_path}.xz.{i + 1}"
            try:
                os.replace(src, dst)
            except OSError as e:
                logger.debug(f'silent except: {e}')

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
        logger.debug(f'silent except: {e}')


def _start_retina(
    retina_path: str, data_dir: str,
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
    try:
        proc = subprocess.Popen(
            [retina_path],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
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
    _stop_child(proc, "subcognitive daemon", 5)
    _remove_pid_file(data_dir, _PID_FILE_DAEMON)
    raise RuntimeError("daemon didn't start in time")


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
        print("[genesis] Error: 'tail' command not found. "
              "Install coreutils or use 'tail -F <data-dir>/daemon.log'.",
              file=sys.stderr)
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
        self, mind: Mind, voice: Voice,
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
        self._queue: queue.Queue[tuple[str, bool]] = queue.Queue(
            maxsize=8
        )
        import random

        self._rng = random.Random()
        # A single worker drains the queue so responses never overlap.
        # handle() itself only classifies and enqueues — it must never
        # block the audio path, or utterances back up and it answers
        # stale speech. Daemon thread: dies with the process.
        self._worker = threading.Thread(
            target=self._drain, daemon=True, name="ambient-handler"
        )
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
            text, addressed = self._queue.get()
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

        logger.info(f"[ambient] Addressed: \"{text}\" → responding to: \"{query}\"")
        try:
            response = self.mind.respond(query)
            self._speak(response, dedup=False)
            logger.info(f"[ambient] Replied: \"{response[:120]}\"")
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
        logger.info(f"[ambient] Chiming in on topic {matched}: \"{text[:80]}\"")
        try:
            response = self.mind.respond(prompt)
            self._speak(response, dedup=True)
            logger.info(f"[ambient] Volunteered: \"{response[:120]}\"")
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
                    logger.info(
                        f"[auditory] Loud sound during sleep: {event.description}"
                    )
                    # Don't wake it fully, but record it — it may
                    # appear in its dreams as a residue.
                    try:
                        self.mind.inner_life.add_dream_residues(
                            [event.sound_type]
                        )
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
                network = self.mind.cognition.network
                if not network.get_concept(event.sound_type):
                    network.add_concept(
                        event.sound_type,
                        origin="auditory",
                        confidence=0.5,
                    )
                    # Link it to the "sound" concept
                    if network.get_concept("sound"):
                        network.add_edge(
                            event.sound_type, "sound",
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
            thought = self.mind.cognition.composer.compose_about(
                event.sound_type, emotion, focused=True
            )
            if thought and thought.content and thought.confidence > 0.3:
                # Dedup — don't repeat what it just said about this
                # sound type. Without this, "noise relates to brain"
                # repeats every few seconds with different verb synonyms.
                if self.mind.cognition.composer.has_said_similar(
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
) -> str:
    """Format the list of episodes as a single string."""
    lines = [f"\n  ┌─ {title} ({len(episodes)} shown) ─────────────────────"]
    ep_cache: dict[int, str] = {}
    assoc_re = re.compile(
        r"\[association\] episode (\d+) ↔ episode (\d+) \(distance: (\d+)\)"
    )

    for ep in episodes:
        t = time.strftime("%m-%d %H:%M", time.localtime(ep.timestamp / 1000))
        module_name = _MODULE_NAMES.get(ep.source_module, f"module-{ep.source_module}")
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
        lines.extend([
            f"  │  Brain waves:    {waves.label} ({waves.dominant.value})",
            f"  │  Focus:          {waves.focus:.3f}",
            f"  │  Integration:    {waves.integration:.3f}",
            f"  │  Consolidation:  {waves.consolidation:.3f}",
        ])
    lines.extend([
        f"  │  STM entries:    {status['stm_count']}",
        f"  │  LTM episodes:   {status['ltm_count']}",
        f"  │  Interactions:   {status['interactions']}",
        f"  │  Language:       {status['language_engine']}",
    ])
    if mind.is_teaching:
        topic = mind.teaching_topic or "general"
        lines.append(f"  │  Teaching mode:  ON (topic: {topic})")
    queued = mind.queued_question_count
    if queued:
        lines.append(f"  │  Queued questions: {queued} (use /teach-questions)")
    lines.extend([
        "  └──────────────────────────────────────────────────",
        "",
    ])
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
        f"\n  genesis> category={emotion.label} "
        f"mode={emotion.cognitive_style}{cause_str}",
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
        # (/status, /feel, /thoughts, /learning, /journal, /dreams,
        # /memories, /regulate) are allowed during sleep since they
        # just query state without engaging cognition.
        if base in (
            "/look", "/learn-code", "/explore",
            "/proposals", "/proposal", "/accept", "/reject",
            "/clear-proposals", "/experiments",
            "/web-history",
            "/create-project", "/projects",
            "/archive-project", "/restore-project", "/manage-projects",
            "/note-project", "/review-notes", "/read-notes",
            "/teach", "/endteach", "/teach-questions",
            "/introspect", "/mission",
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
        msg = mind._render_self_report(
            content, confidence=confidence, metadata=metadata
        )
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


def _cmd_journal(mind: Mind, rest: str) -> str:
    """Show recent journal entries."""
    return f"\n{mind.read_journal(20)}\n"


def _cmd_dreams(mind: Mind, rest: str) -> str:
    """Show Genesis's recent dreams."""
    return _render_recent_episodes(mind, 9, "Dreams")


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
        lines.append(
            f"  [{when}] {ok} {source} → {topic}"
        )
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
            f"\n  genesis> tried to build '{result.name}', "
            f"but it didn't work out: {result.error}\n"
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
            f"    {p['name']} [{status}] "
            f"({p['files']} files, {size_kb:.1f} KB) — {p['path']}"
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
        t = threading.Thread(
            target=lambda: result_holder.__setitem__("value", mind.look_at_image(path)),
            daemon=True,
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
    t = threading.Thread(
        target=lambda: retina_holder.__setitem__("value", mind.see()),
        daemon=True,
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
            "\n  Usage: /register-face <name>"
            "\n  Look at the camera and say whose face this is.\n"
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
    from genesis_cognitive.perception.retina import latest_frame

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
        from genesis_cognitive.perception.vision import _yuyv_to_rgb
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
        return _compose(
            mind,
            f"mission set to {mission}",
            fallback=f"[mission set: {mission}]",
            confidence=0.7,
            metadata={"mission_set": mission},
        ) + "\n"
    return f"\n  genesis> {mind.get_mission()}\n"


def _cmd_sleep(mind: Mind, rest: str) -> str:
    """Put Genesis to sleep for dreaming and consolidation.

    It falls asleep, cycles through its sleep stages, and wakes on
    its own when its neurochemistry has recovered. Ambient and voice
    input remain gated, while submitted terminal conversation is an
    explicit interaction that wakes it through ``Mind.respond``.
    """
    if mind.is_sleeping:
        lines = [_compose(
            mind,
            "already asleep and dreaming",
            fallback="[already asleep]",
            confidence=0.5,
        )]
        lines.append(mind.sleep_status())
        lines.append("\n  It'll wake on its own when it's rested. /dreams to see more.\n")
        return "\n".join(lines)
    if mind.is_meditating:
        mind.wake_from_meditation()
    # Self-initiated sleep so the sleep watcher auto-wakes it when
    # adenosine drops below the wake threshold. It wakes on its own.
    mind.sleep(user_initiated=False)
    lines = [_compose(
        mind,
        "drifting into sleep",
        fallback="[drifting into sleep]",
        confidence=0.6,
        metadata={"sleep_started": True},
    )]
    lines.append(mind.sleep_status())
    lines.append("  It'll wake on its own when it's rested.")
    lines.append("  Its dreams will appear as genesis~ lines below.\n")
    return "\n".join(lines)


def _cmd_wake(mind: Mind, rest: str) -> str:
    """Wake Genesis from sleep or meditation."""
    if mind.is_sleeping:
        # Show dream state before waking
        pre_wake = mind.sleep_status()
        mind.wake()
        lines = [_compose(
            mind,
            "waking up rested",
            fallback="[awake]",
            confidence=0.7,
            metadata={"woke": True},
        )]
        lines.append(f"  Before waking:\n{pre_wake}")
        lines.append("  Memories consolidated. It's ready to engage.\n")
        return "\n".join(lines)
    if mind.is_meditating:
        mind.wake_from_meditation()
        return _compose(
            mind,
            "coming out of meditation refreshed",
            fallback="[out of meditation]",
            confidence=0.7,
        ) + "\n"
    return _compose(
        mind,
        "already awake",
        fallback="[already awake]",
        confidence=0.4,
    ) + "\n"


def _cmd_nap(mind: Mind, rest: str) -> str:
    """Put Genesis to sleep for a nap — it wakes on its own when ready.

    A nap is self-initiated sleep: it falls asleep, cycles through
    its sleep stages, and wakes naturally when its neurochemistry
    has recovered (adenosine drops below the wake threshold). No
    timer — it wakes when it's ready, not when a clock says so.
    """
    if mind.is_sleeping:
        return _compose(
            mind,
            "already asleep and dreaming",
            fallback="[already asleep]",
            confidence=0.5,
        ) + "\n"
    if mind.is_meditating:
        mind.wake_from_meditation()
    # Self-initiated sleep (not user-initiated) so the sleep watcher
    # auto-wakes it when the nap cycle (N1→N2) completes. No timer —
    # it wakes on its own when the light sleep cycle is done.
    mind.sleep(user_initiated=False, nap=True)
    lines = [_compose(
        mind,
        "taking a nap",
        fallback="[napping]",
        confidence=0.6,
        metadata={"nap": True},
    )]
    lines.append("  It'll wake up on its own when it's rested.\n")
    return "\n".join(lines)


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
        return _compose(
            mind,
            "meditating, try again after",
            fallback="[meditating]",
            confidence=0.4,
        ) + "\n"

    # Drawing can take a few seconds (Cairo render + OpenCV post-processing)
    result_holder: dict[str, str | None] = {}
    t = threading.Thread(
        target=lambda: result_holder.__setitem__("value", mind.draw()),
        daemon=True,
        name="draw",
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
    return _compose(
        mind,
        "drew but cannot describe it",
        fallback="[drew, but silent]",
        confidence=0.4,
        metadata={"drawing_silent": True},
    ) + "\n"


def _cmd_meditate(mind: Mind, rest: str) -> str:
    """Put Genesis into meditation for quiet restoration.

    During meditation, learning and spontaneous thoughts are paused.
    Calming neurochemical impulses are emitted to let receptors
    recover. Default 60 seconds, or specify a duration:
    /meditate 120  — meditate for 2 minutes.
    """
    if mind.is_meditating:
        return _compose(
            mind,
            "already meditating",
            fallback="[already meditating]",
            confidence=0.5,
        ) + "\n"
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
    lines = [_compose(
        mind,
        f"meditating for {duration:.0f} seconds",
        fallback=f"[meditating {duration:.0f}s]",
        confidence=0.6,
        metadata={"meditation_seconds": duration},
    )]

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

    threading.Thread(
        target=_meditation_timer, daemon=True, name="meditation-timer"
    ).start()
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
    lines = [_compose(
        mind,
        "entering teaching mode",
        fallback="[teaching mode]",
        confidence=0.7,
        metadata={"teaching_started": True},
    )]
    if topic:
        lines.append(_compose(
            mind,
            f"focusing on {topic}, ready to listen and learn",
            fallback=f"[focusing on {topic}]",
            confidence=0.7,
            metadata={"teaching_topic": topic},
        ))
    else:
        lines.append(_compose(
            mind,
            "ready to listen and learn",
            fallback="[ready to listen]",
            confidence=0.6,
        ))
    lines.append("  Use /endteach when you're done.\n")
    return "\n".join(lines)


def _cmd_endteach(mind: Mind, rest: str) -> str:
    """Exit teaching/training mode and resume normal operation.

    Resumes bug scanning, autonomous learning, art, and curiosity.
    """
    if not mind.is_teaching:
        return _compose(
            mind,
            "not in teaching mode",
            fallback="[not teaching]",
            confidence=0.4,
        ) + "\n"
    mind.exit_teaching_mode()
    return _compose(
        mind,
        "exiting teaching mode, resuming normal operation",
        fallback="[resuming]",
        confidence=0.7,
        metadata={"teaching_ended": True},
    ) + "\n"


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
        return f"  Stopping. {count} questions remain queued.\n"
    if rest == "clear":
        count = mind.queued_question_count
        mind.clear_queued_questions()
        return f"  Cleared {count} queued questions.\n"

    if not mind.has_queued_questions:
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
        lines = [_compose(
            mind,
            "already asleep and dreaming",
            fallback="[already asleep]",
            confidence=0.5,
        )]
        lines.append(mind.sleep_status())
        lines.append("\n  Use /wake to wake it, or /dreams to see more.\n")
        return "\n".join(lines)
    mind.sleep_aid()
    lines = [_compose(
        mind,
        "sleep aid administered, drifting into deep sleep",
        fallback="[sleep aid administered]",
        confidence=0.7,
        metadata={"sleep_aid": True},
    )]
    lines.append(mind.sleep_status())
    lines.append("\n  Use /wake to wake it.\n")
    return "\n".join(lines)


_COMMAND_HANDLERS: dict[str, Callable[[Mind, str], str]] = {
    "/status": _cmd_status,
    "/feel": _cmd_feel,
    "/introspect": _cmd_introspect,
    "/learning": _cmd_learning,
    "/thoughts": _cmd_thoughts,
    "/world": _cmd_world,
    "/regulate": _cmd_regulate,
    "/journal": _cmd_journal,
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
    "/teach-questions": _cmd_teach_questions,
    "/sleep-aid": _cmd_sleep_aid,
}


# ─── Voice helpers ───────────────────────────────────────────────────


def _init_mic() -> VoiceInput:
    """Initialize voice input (microphone) for the local CLI."""
    mic = VoiceInput(lazy=True)
    if mic.available:
        logger.info(
            f"  Microphone: available ({mic.backend_name}, say /voice to talk)"
        )
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
            "\n  Voice mode ON. Speak to Genesis. "
            "Say 'exit voice mode' or press Ctrl+C to stop.\n"
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
    logger.info("    /journal      — read its journal")
    logger.info("    /dreams       — see its subcognitive dream insights")
    logger.info("    /memories     — see its recent long-term memories")
    logger.info("    /requests     — see sites it wants to access")
    logger.info("    /approve URL  — approve a site request")
    logger.info("    /deny URL     — deny a site request")
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
    logger.info("    /register-face <name> — teach it your face")
    logger.info("    /faces        — show known faces and who it last saw")
    logger.info("    /mission [text] — set or show its top-level mission")
    logger.info("    /sleep        — put it to sleep")
    logger.info("    /nap          — short nap, wakes on its own when rested")
    logger.info("    /wake         — wake it from sleep or meditation")
    logger.info("    /meditate [secs] — put it into meditation")
    logger.info("    /teach [topic] — put it into teaching mode (focused learning)")
    logger.info("    /endteach    — exit teaching mode, resume normal operation")
    logger.info("    /teach-questions — answer its queued questions one at a time")
    logger.info("    /sleep-aid    — emergency sleep aid for stress-induced insomnia")
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
            f"{cog['prediction_error_label']} "
            f"(prediction error: {cog['prediction_error']:.2f})"
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
            f"Error: cannot acquire Genesis lock for {data_dir}: {e}",
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


def _start_daemon_process(
    data_dir: str, daemon_socket: str
) -> subprocess.Popen | None:
    """Start the subcognitive daemon.

    Returns the Popen handle on success, or None on failure (error
    message already printed to stderr).
    """
    # The CLI owns the daemon lifecycle: it starts a fresh daemon and
    # shuts it down on exit. Connecting to a pre-existing daemon would
    # leave it running after the CLI exits (a resource leak), so we
    # refuse rather than silently adopt a stale process.
    if _is_daemon_running(daemon_socket):
        print(
            f"Error: a daemon is already running on {daemon_socket}.\n"
            "Stop it first with ./run.sh --stop or use a "
            "different --data-dir.",
            file=sys.stderr,
        )
        return None
    daemon_path = _find_daemon_binary()
    if daemon_path is None:
        print(
            "Error: genesis-daemon binary not found. Run: cargo build --release",
            file=sys.stderr,
        )
        return None
    print("[genesis] Starting subcognitive daemon...", file=sys.stderr)
    try:
        return _start_daemon(daemon_path, data_dir, daemon_socket)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def _setup_ambient_listener(
    mind: Mind, voice: Voice,
    dedup_voice: VoiceDedup | None = None,
) -> AmbientListener | None:
    """Start the ambient microphone listener if available."""
    handler = AmbientHandler(mind, voice, dedup_voice)
    listener = AmbientListener(
        callback=handler.handle, is_speaking=lambda: voice.is_speaking
    )
    if listener.available:
        if listener.start():
            print(
                '[genesis] Ambient listening ON — say "genesis" to address its',
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
    mind: Mind, voice: Voice,
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
    """Install SIGTERM/SIGINT handlers that set *shutting_down*."""

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
    """Start a background thread that autonomously sleeps/wakes it."""

    # Minimum autonomous sleep duration: the first N3 slow-wave sleep
    # stage is reached after ~20 minutes (N1=5min + N2=15min). Waking
    # its before that prevents all concept-network consolidation — the
    # whole point of sleep. When the watcher autonomously puts it to
    # sleep, let it rest long enough for at least one N3 cycle.
    import time as _time
    min_sleep_seconds = 1800.0  # 30 minutes — covers N1+N2+N3
    autonomous_sleep_start: float = 0.0

    def _sleep_watcher() -> None:
        """Background loop that autonomously sleeps/wakes Genesis based on phase."""
        nonlocal autonomous_sleep_start
        while not shutting_down.is_set():
            try:
                phase = mind.client.get_phase().phase
            except Exception as e:  # noqa: BLE001
                logger.debug(f"sleep watcher get_phase failed: {e}")
                shutting_down.wait(timeout=10.0)
                continue

            # Anchor the minimum-sleep window to when we first observe
            # it asleep, regardless of which path put it to sleep —
            # this watcher, heartbeat auto-sleep, volition, or /sleep.
            # Without this, only watcher-initiated sleep set the anchor,
            # so elapsed stayed 0 and the phase-based auto-wake below
            # could never fire for sleep entered any other way.
            if mind.is_sleeping:
                if autonomous_sleep_start == 0.0:
                    autonomous_sleep_start = _time.time()
            else:
                autonomous_sleep_start = 0.0

            if phase in (PHASE_NREM, PHASE_REM) and not mind.is_sleeping and not mind.is_meditating:
                try:
                    mind.sleep()
                    autonomous_sleep_start = _time.time()
                    logger.info("[genesis] It drifted into sleep and is dreaming.")
                    logger.info("[genesis]   Use /wake to wake it, or watch its dreams below.")
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"autonomous sleep failed: {e}")
            elif (
                phase in (PHASE_ACTIVE, PHASE_ALERT)
                and mind.is_sleeping
                and not mind.is_user_sleeping
            ):
                # Don't wake it too soon — the daemon's neurochemistry
                # may briefly dip back to Active during sleep. Let the
                # sleep cycle reach N3 (at least min_sleep_seconds) before
                # allowing an autonomous wake.
                elapsed = _time.time() - autonomous_sleep_start
                if elapsed >= min_sleep_seconds:
                    try:
                        # Show its state before waking so the user sees what happened
                        status = mind.sleep_status()
                        mind.wake()
                        autonomous_sleep_start = 0.0
                        logger.info("[genesis] It woke up on its own.")
                        logger.info("[genesis]   Its state before waking:")
                        for line in status.split("\n"):
                            logger.info(f"[genesis]   {line}")
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"autonomous wake failed: {e}")
            shutting_down.wait(timeout=10.0)

    threading.Thread(target=_sleep_watcher, daemon=True, name="sleep-watcher").start()


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
        t for t in new_thoughts
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
        state.last_thought_ts = max(
            t.get("timestamp", state.last_thought_ts) for t in new_thoughts
        )
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
        _print_idle_thoughts(
            state, mind, voice, thought_collector, out_lock, dedup_voice
        )


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

    poll_thread = threading.Thread(
        target=_poll_loop,
        args=(state, mind, voice, thought_collector, out_lock, dedup_voice),
        daemon=True,
        name="cli-thought-poll",
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

            # Always drain any thoughts that arrived while the user was typing
            _print_idle_thoughts(
                state, mind, voice, thought_collector, out_lock, dedup_voice
            )

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
                        user_input, mind, voice, mic, out_lock,
                        thought_collector,
                    )
                finally:
                    state.suppress_thoughts = False
                    mind._suppress_volition = False
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
                user_input, mind, voice, thought_collector, out_lock,
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
    # Mark user activity BEFORE the sleep check. The heartbeat's
    # auto-sleep runs in a separate thread and can fire between the
    # user pressing Enter and respond() being called. Without this,
    # it can fall asleep mid-conversation and the message is lost.
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
            t for t in new_thoughts
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


def _stop_child(proc: subprocess.Popen, name: str, timeout: float) -> bool:
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[genesis] Waiting for {name} to finish safely...", file=sys.stderr)
        proc.wait()
    if proc.returncode != 0:
        print(
            f"[genesis] {name} exited with status {proc.returncode}.",
            file=sys.stderr,
        )
        return False
    return True


def _shutdown(
    ambient_listener: AmbientListener | None,
    mind: Mind | None,
    daemon_proc: subprocess.Popen | None,
    retina_proc: subprocess.Popen | None = None,
    data_dir: str = "",
    auditory_cortex: AuditoryCortex | None = None,
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
    if mind is not None:
        clean_shutdown = bool(mind.stop())
    if daemon_proc is not None:
        print("[genesis] Stopping subcognitive daemon...", file=sys.stderr)
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
        print(
            "[genesis] Shutdown incomplete — cognitive state may not be saved.",
            file=sys.stderr,
        )
    return clean_shutdown


def main() -> int:
    """Main entry point for the Genesis CLI."""
    if not os.environ.get("GENESIS_RUN"):
        print(
            "Error: Genesis must be started with ./run.sh.",
            file=sys.stderr,
        )
        return 1

    args = _parse_cli_args()
    data_dir = os.path.abspath(args.data_dir)

    if args.tail:
        return _tail_daemon_log(data_dir)

    daemon_socket = args.daemon_socket or os.path.join(data_dir, DEFAULT_DAEMON_SOCKET)

    # Owner-only: the dir contains the daemon's IPC socket and state.
    os.makedirs(data_dir, mode=0o700, exist_ok=True)

    if _acquire_cli_lock(data_dir) is None:
        return 1

    # Write the CLI PID file so run.sh --stop can target this exact
    # process instead of using broad pkill patterns.
    _write_pid_file(data_dir, _PID_FILE_CLI, os.getpid())

    # Install handlers and initialize handles before any child process
    # starts. Every initialization step after this point is covered by
    # the same cleanup path, so a voice, microphone, or emotion-setup
    # failure cannot orphan the daemon/retina.
    shutting_down = threading.Event()
    _setup_signal_handlers(shutting_down)
    daemon_proc: subprocess.Popen | None = None
    retina_proc: subprocess.Popen | None = None
    mind: Mind | None = None
    ambient_listener: AmbientListener | None = None
    auditory_cortex: AuditoryCortex | None = None
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
                    "[genesis] Offline mode — learning paused, "
                    "conversation and art active.",
                    file=sys.stderr,
                )
            mind = Mind(daemon_socket, offline=args.offline)
            try:
                mind.start()
            except (OSError, ConnectionError) as e:
                print(f"Error: cannot connect to daemon: {e}", file=sys.stderr)
                exit_code = 1
            else:
                # ── Wire voice ──
                voice, dedup_voice = _wire_voice(mind)

                # ── Start ambient listener (microphone) ──
                ambient_listener = _setup_ambient_listener(
                    mind, voice, dedup_voice
                )

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

                # ── Interactive loop ──
                _run_interactive_loop(
                    mind, voice, thought_collector, shutting_down, dedup_voice
                )
    finally:
        # Mark shutdown before saving so subsequent signals do not raise
        # KeyboardInterrupt during persistence. Repeated Ctrl+C must not
        # abandon a state save or force-kill a child still flushing data.
        shutting_down.set()
        if not _shutdown(
            ambient_listener, mind, daemon_proc, retina_proc,
            data_dir, auditory_cortex,
        ):
            exit_code = 1
    return exit_code


def _start_retina_process(data_dir: str) -> subprocess.Popen | None:
    """Start the retina (camera) subprocess, reaping any stale one.

    The retina is a standalone Rust binary that captures camera
    frames into shared memory. We manage it as a child process so
    it's automatically stopped when the CLI shuts down.
    """
    retina_path = _find_retina_binary()
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
        if mind.is_sleeping:
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
    sys.exit(main())
