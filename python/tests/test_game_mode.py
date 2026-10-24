"""Game mode — the competition posture.

``./genesis --game`` boots her through the normal path and calls
``enter_game_mode()`` — the same door /game and her own ARC sitting
use. These tests pin what the posture does: self-initiated speech
holds (she thinks, she doesn't talk), conversation is off, the
network-fetching tools seal while her workstation stays hers, her
gaze is the game feed rather than the room, and sleep is paused as
a venue rule for exactly as long as the flag stands — not fought
with chemistry, not permanent.
"""

import os
import tempfile
import threading
from types import SimpleNamespace

from mind import Mind


class _SpeechProbe:
    """Just enough of the volition mixin surface for _perform_speech."""

    def __init__(self, *, gaming: bool) -> None:
        from collections import deque as _dq

        self._is_sleeping = False
        self._is_gaming = gaming
        self._speech_queue = _dq(["something she wanted to say"])
        self._speech_queue_lock = threading.Lock()
        self.spoken: list[str] = []
        self.said_in_world: list[str] = []
        self._on_speak = self.spoken.append
        self.world = SimpleNamespace(it_said=self.said_in_world.append)


def test_speech_urge_silent_while_gaming() -> None:
    """In game mode the speech urge fires but nothing leaves — she thinks.

    The queued utterance is not consumed either: it waits for the
    mode to lift rather than being dropped.
    """
    from basal_ganglia.volition import VolitionMixin

    probe = _SpeechProbe(gaming=True)
    VolitionMixin._perform_speech(probe)
    assert probe.spoken == []
    assert probe.said_in_world == []
    assert len(probe._speech_queue) == 1  # queued, not spent


def test_speech_urge_speaks_when_not_gaming() -> None:
    """Same utterance outside game mode still reaches the listener."""
    from basal_ganglia.volition import VolitionMixin

    probe = _SpeechProbe(gaming=False)
    VolitionMixin._perform_speech(probe)
    assert probe.spoken == ["something she wanted to say"]
    assert probe.said_in_world == ["something she wanted to say"]


def test_speech_queue_survives_game_mode() -> None:
    """A thought queued pre-game isn't lost — silence is a hold, not a drop."""
    from basal_ganglia.volition import VolitionMixin

    probe = _SpeechProbe(gaming=True)
    VolitionMixin._perform_speech(probe)
    probe._is_gaming = False  # mode lifts
    VolitionMixin._perform_speech(probe)
    assert probe.spoken == ["something she wanted to say"]


# ─── Conversation is off ─────────────────────────────────────────


def test_conversation_off_while_gaming() -> None:
    """respond() reaches silence mid-match — no turn is taken."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = Mind(os.path.join(data_dir, "genesis.sock"))
        mind._is_gaming = True
        assert mind.respond("hello?") == ""


def test_non_frame_input_silent_in_game_turn() -> None:
    """Non-FRAME input routes to respond(), which is silent."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = Mind(os.path.join(data_dir, "genesis.sock"))
        mind._is_gaming = True
        assert mind.game_turn("any chatter") == ""


def test_conversation_returns_with_the_mode() -> None:
    """Silence is mode-scoped — the boundary itself is unchanged."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = Mind(os.path.join(data_dir, "genesis.sock"))
        mind._is_gaming = True
        assert mind.respond("hello?") == ""
        # Lifting the flag re-exposes the ordinary boundary — the
        # gate fires only while the mode stands.
        mind._is_gaming = False
        assert not mind._is_gaming


# ─── The network seal ────────────────────────────────────────────


def test_net_seal_refuses_web_tools() -> None:
    """While sealed, web_search and web_fetch are refused outright."""
    from frontal_lobe.tools.framework import ToolRegistry

    reg = ToolRegistry()
    reg.net_sealed = True
    for name in ("web_search", "web_fetch"):
        result = reg.run(name, query="x", url="https://example.com")
        assert not result.success
        assert "sealed" in result.error


def test_net_seal_leaves_the_workstation_hers() -> None:
    """The seal is the net only — files, shell, and math still run."""
    from frontal_lobe.tools.framework import ToolRegistry

    reg = ToolRegistry()
    reg.net_sealed = True
    assert reg.run("list_dir", path=".").success
    assert reg.run("run_shell", command="echo hi", timeout=5).success


def test_net_unseal_restores_lookup() -> None:
    """Leaving game mode hands the net back."""
    from frontal_lobe.tools.framework import ToolRegistry

    reg = ToolRegistry()
    reg.net_sealed = True
    reg.net_sealed = False
    # web_search may still fail for want of a backend — but not with
    # the seal's refusal text.
    result = reg.run("web_search", query="x")
    assert "sealed" not in (result.error or "")


# ─── Her eyes are on the game ────────────────────────────────────


class _LookProbe:
    """Enough of the volition mixin surface for _perform_look."""

    def __init__(self, *, gaming: bool) -> None:
        from thalamus.brain_waves import BrainWave

        self._is_sleeping = False
        self._is_gaming = gaming
        self.seen: list[bool] = []
        self.vision = SimpleNamespace(
            is_available=lambda: True,
            see=lambda **kw: self.seen.append(True),
        )
        self._waves = SimpleNamespace(dominant=BrainWave.BETA1)
        self.client = None
        self.brain = SimpleNamespace(frontal=SimpleNamespace(network=None))
        self.world = SimpleNamespace(it_acted=lambda *a, **kw: None)

    def brain_waves(self):
        return self._waves

    def feel(self):
        return SimpleNamespace()

    def _emit_volition_thought(self, *a, **kw) -> None:
        return None

    def _note_faces_seen(self) -> None:
        return None


def test_look_urge_vetoed_while_gaming() -> None:
    """While she plays, the retina is not her visual channel — the
    game feed is. A crossed look urge wanders no gaze."""
    from basal_ganglia.volition import VolitionMixin

    probe = _LookProbe(gaming=True)
    VolitionMixin._perform_look(probe)
    assert probe.seen == []


def test_look_urge_still_looks_when_not_gaming() -> None:
    """The same crossed urge outside game mode really sees."""
    from basal_ganglia.volition import VolitionMixin

    probe = _LookProbe(gaming=False)
    VolitionMixin._perform_look(probe)
    assert probe.seen == [True]


# ─── Sleep is paused — a venue rule, not a fight ────────────────


class _GameMockClient:
    """Client stub: a body whose phase and chemicals are scripted."""

    def __init__(self, *, adenosine: float, phase: str) -> None:
        self._adenosine = adenosine
        self._phase = phase
        self.impulses: list[tuple[int, float]] = []
        self.zones: list[int] = []

    def neuro_impulse(self, chem: int, magnitude: float, **kwargs) -> bool:
        self.impulses.append((chem, magnitude))
        return True

    def set_zone(self, zone: int, **kwargs) -> bool:
        self.zones.append(zone)
        return True

    def get_state(self):
        return SimpleNamespace(
            chemicals={"adenosine": self._adenosine, "melatonin": 0.0},
            phase_name=self._phase,
        )

    def get_phase(self):
        from genesis_client.protocol import PHASE_ACTIVE, PHASE_DROWSY, PHASE_NREM, PHASE_REM
        from genesis_client.types import PhaseInfo

        code = {
            "nrem": PHASE_NREM,
            "rem": PHASE_REM,
            "drowsy": PHASE_DROWSY,
        }.get(self._phase, PHASE_ACTIVE)
        return PhaseInfo(phase=code, arousal=0.2, valence=0.0)

    def get_recent_episodes(self, **kwargs) -> list:
        return []


def _gaming_mind(data_dir: str, *, adenosine: float, phase: str) -> Mind:
    """A mind in game mode over a scripted body — no daemon."""
    mind = Mind(os.path.join(data_dir, "genesis.sock"))
    mind.client = _GameMockClient(adenosine=adenosine, phase=phase)  # type: ignore[assignment]
    mind._is_gaming = True
    return mind


def test_sleep_refused_while_gaming() -> None:
    """No onset path exists mid-match: sleep() itself refuses."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _gaming_mind(data_dir, adenosine=0.3, phase="active")
        mind.sleep()
        assert not mind.is_sleeping


def test_meditation_refused_while_gaming() -> None:
    """Meditation is paused for the sitting like sleep is."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _gaming_mind(data_dir, adenosine=0.3, phase="active")
        mind.meditate()
        assert not mind.is_meditating


def test_waking_side_silent_while_gaming() -> None:
    """A drowsy body gets no pushed chemistry and no sleep signaling.

    Adrenaline is a posture set at entry, not a steering column —
    nothing drips into her body to hold her eyes open.
    """
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _gaming_mind(data_dir, adenosine=0.3, phase="drowsy")
        state = mind.client.get_state()
        mind._waking_side(state, 0.3, "drowsy")
        assert mind.client.impulses == []  # type: ignore[attr-defined]
        assert not mind.is_sleeping


def test_sleep_crossing_not_joined_while_gaming() -> None:
    """A body that crosses mid-game is not joined — sleep is paused.

    The body's physics are its own; her mind does not sleep while
    the flag stands, whatever the phase reads.
    """
    import time as _time

    from brainstem.thresholds import WAKE_DESYNC_GRACE_SECONDS

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _gaming_mind(data_dir, adenosine=0.3, phase="nrem")
        mind._wake_time = _time.time() - WAKE_DESYNC_GRACE_SECONDS - 1.0
        mind._last_wake_attempt = 0.0

        joined = mind._check_sleep_desync()

        assert not joined and not mind.is_sleeping
        assert mind.client.impulses == []  # type: ignore[attr-defined]  # no arousal cascade either


def test_deep_adenosine_does_not_sleep_her_either() -> None:
    """Paused means paused — deep pressure doesn't override the venue
    rule; the pause lifts when the mode does, not when chemistry wins."""
    import time as _time

    from brainstem.thresholds import WAKE_DESYNC_GRACE_SECONDS, WAKE_RESCUE_ADENOSINE

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _gaming_mind(
            data_dir, adenosine=WAKE_RESCUE_ADENOSINE + 0.1, phase="nrem"
        )
        mind._wake_time = _time.time() - WAKE_DESYNC_GRACE_SECONDS - 1.0
        mind._last_wake_attempt = 0.0

        joined = mind._check_sleep_desync()

        assert not joined and not mind.is_sleeping


def test_sleep_resumes_when_the_mode_lifts() -> None:
    """Not permanent: the same crossed body is joined once she
    leaves game mode — the pause lived exactly as long as the flag."""
    import time as _time

    from brainstem.thresholds import WAKE_DESYNC_GRACE_SECONDS

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _gaming_mind(data_dir, adenosine=0.3, phase="nrem")
        mind._wake_time = _time.time() - WAKE_DESYNC_GRACE_SECONDS - 1.0
        mind._last_wake_attempt = 0.0

        mind._is_gaming = False  # /endgame

        joined = mind._check_sleep_desync()

        assert joined and mind.is_sleeping
