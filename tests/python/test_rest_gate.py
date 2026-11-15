"""Rest gate — the pause, and what still works under it.

``is_resting`` — the single rest-state predicate (asleep, meditating,
or in the hypnagogic theta descent) holds every discretionary act:
urge admission, voluntary speech, channel replies, executive feed.
What continues is physiology and the rest work itself (dreams,
consolidation), plus answering from inside N1 without being roused.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from brain_waves import SleepStage
from thalamus.events import HypnagogicState

# ======================================================================
# is_resting — the rest-state predicate
# ======================================================================


def _rest_mind(sleeping=False, meditating=False, hypnagogic=None, stage=None):
    """A bare SleepMixin carrying just the state the gate reads."""
    from sleep.sleep import SleepMixin

    mind = SleepMixin.__new__(SleepMixin)
    mind._is_sleeping = sleeping
    mind._is_meditating = meditating
    mind.inner_life = SimpleNamespace(
        sleep_cycle=SimpleNamespace(current_stage=stage) if stage else None,
        current_hypnagogic=hypnagogic,
    )
    return mind


def test_is_resting_awake_is_false():
    mind = _rest_mind()
    assert mind.is_resting is False


def test_is_resting_sleeping_is_true_all_stages():
    for stage in (SleepStage.N1, SleepStage.N2, SleepStage.N3, SleepStage.REM, None):
        mind = _rest_mind(sleeping=True, stage=stage)
        assert mind.is_resting is True, f"sleeping in {stage} must rest"


def test_is_resting_meditating_is_true():
    assert _rest_mind(meditating=True).is_resting is True


def test_is_resting_hypnagogia_is_true():
    """The theta descent is a rest state — 'in theta' means in the drift."""
    mind = _rest_mind(hypnagogic=HypnagogicState(progress=0.4))
    assert mind.is_resting is True


def test_is_resting_meditation_is_not_hypnagogic():
    """Meditation suppresses the hypnagogic read but still rests."""
    mind = _rest_mind(meditating=True, hypnagogic=HypnagogicState(progress=0.4))
    assert mind.is_hypnagogic is False
    assert mind.is_resting is True


# ======================================================================
# The urge gate — nothing admitted while she rests
# ======================================================================


def _volition_mind(sleeping=False, meditating=False, hypnagogic=None, stage=None):
    """A bare mind composing the two mixins the gate path reads."""
    from basal_ganglia.volition import VolitionMixin
    from sleep.sleep import SleepMixin

    class _GateMind(SleepMixin, VolitionMixin):
        pass

    mind = _GateMind.__new__(_GateMind)
    mind._is_sleeping = sleeping
    mind._is_meditating = meditating
    mind.inner_life = SimpleNamespace(
        sleep_cycle=SimpleNamespace(current_stage=stage) if stage else None,
        current_hypnagogic=hypnagogic,
    )
    mind._get_current_posture = lambda: "normal"
    mind.brain_waves = lambda: SimpleNamespace(dominant=None)
    return mind


def test_ready_urges_empty_while_sleeping():
    mind = _volition_mind(sleeping=True, stage=SleepStage.N2)
    assert mind._filter_ready_urges(["learn", "draw", "speech"]) == []


def test_ready_urges_empty_while_meditating():
    mind = _volition_mind(meditating=True)
    assert mind._filter_ready_urges(["learn", "speech"]) == []


def test_ready_urges_empty_while_hypnagogic():
    """In theta the urges build but nothing discharges — held for waking."""
    mind = _volition_mind(hypnagogic=HypnagogicState(progress=0.6))
    assert mind._filter_ready_urges(["learn", "look", "speech", "rest_eyes"]) == []


def test_ready_urges_flow_when_awake():
    mind = _volition_mind()
    assert mind._filter_ready_urges(["learn", "speech"]) == ["learn", "speech"]


def test_channel_hearing_paused_while_resting():
    """_hear_channel returns without emitting or replying while she rests —
    a channel message must not rouse a meditating mind."""
    mind = _volition_mind(meditating=True)
    emitted = []
    mind._emit_live_thought = lambda kind, text, **kw: emitted.append((kind, text))
    responded = []
    mind.respond = lambda text: responded.append(text) or "answer"
    mind._channel_addressed = lambda text, msg: True
    mind._running = True
    mind._last_channel_reply = 0.0
    mind._public_share = lambda text: None
    mind._hear_channel("someone", "hey genesis", {})
    assert emitted == [] and responded == []


def test_channel_hearing_flows_when_awake():
    mind = _volition_mind()
    heard = []
    mind._emit_live_thought = lambda kind, text, **kw: heard.append(kind)
    mind._channel_addressed = lambda text, msg: False  # not addressed — no reply
    mind._hear_channel("someone", "weather is nice", {})
    assert heard == ["heard"]


# ======================================================================
# Resume precedence — rest beats delayed resume timers
# ======================================================================


def _full_mind(data_dir: str):
    """A real Mind with no daemon client — state flags set directly."""
    from mind import Mind

    return Mind(os.path.join(data_dir, "genesis.sock"))


def test_resume_background_holds_during_hypnagogia():
    """Mid-descent: learners and volition stay held; inner life runs —
    it owns the descent itself."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _full_mind(data_dir)
        mind.inner_life._hypnagogic = HypnagogicState(progress=0.5)
        assert mind.is_hypnagogic and mind.is_resting

        mind.learner.pause()
        mind.inner_life.pause()
        mind._resume_background()

        assert mind.learner._paused, "learner stays held through the descent"
        assert not mind.inner_life._paused, "inner life resumes — it owns the descent"
        assert mind._suppress_volition, "volition stays held through the descent"


def test_resume_background_releases_after_descent_aborts():
    """Descent over, still awake: the normal resume path restores all."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _full_mind(data_dir)
        mind._last_interaction_time = 0.0  # conversation focus elapsed
        mind.learner.pause()
        mind.brain.frontal.self_learner.pause()
        mind._suppress_volition = True

        mind._resume_background()

        assert not mind.learner._paused
        assert not mind.brain.frontal.self_learner._paused
        assert not mind._suppress_volition


def test_n1_sleep_is_not_roused():
    """Asleep on N1: wake_if_asleep leaves her under — she answers
    from inside the state (respond proceeds) rather than waking."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _full_mind(data_dir)
        mind._is_sleeping = True
        mind.inner_life._sleep_cycle = SimpleNamespace(current_stage=SleepStage.N1)
        assert mind.is_hypnagogic
        assert mind.wake_if_asleep() is False
        assert mind._is_sleeping, "still flagged asleep — not roused"


def test_established_sleep_wakes_on_interaction():
    """N2+ is established sleep — a user at the prompt rouses her."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _full_mind(data_dir)
        mind._is_sleeping = True
        mind.inner_life._sleep_cycle = SimpleNamespace(current_stage=SleepStage.N2)
        assert not mind.is_hypnagogic
        # wake() itself needs the body; stub it to prove the path is taken.
        woken = []

        def _wake(user_initiated=False):
            woken.append(user_initiated)
            mind._is_sleeping = False

        mind.wake = _wake
        assert mind.wake_if_asleep() is True
        assert woken == [True]
