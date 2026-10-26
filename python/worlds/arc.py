"""ARC-AGI-3 practice — real ``arcengine`` environments as her world.

This is the adapter ``world_step`` was written for: the mind's
embodied loop plays, and the ``arc-agi`` package is the world side.
This module is the seam that binds them — it owns no cognition,
only translation and the duty of care:

    arcengine FrameDataRaw  ->  mind.world_step(frame, actions)
    world action descriptor  ->  env.step(GameAction, reasoning=...)

The mind side never speaks the harness's protocol: a frame is a
percept (the grid itself, not a text encoding), an action is a
decision descriptor ``{"action": id, "x": .., "y": ..}``, and the
episode boundary is told to her directly (``world_episode_end``).
The adapter is the world's clock — arcengine is turn-based, so the
world cannot advance without her move — but it never reaches past
``observe``/``world_step`` into her cognition.

Two contracts keep a run honest:

- **Game mode.** ``play()`` only runs while the mind is in game
  mode (``enter_game_mode()``): engagement is what starves her
  other urges so the run measures the entity at play, not the
  collision of play with every urge that happened to fire. A
  caller that skips it gets refused, because a run that doesn't
  engage her isn't measuring the same entity.
- **Readiness.** Her own health gate is checked before the run,
  at every episode boundary, and periodically mid-episode. When
  she is asleep, drowsy, stressed, overloaded, or unreachable, the
  run does not start or stops early — a body that isn't well
  shouldn't be worked.

The ARC side judges honestly and tells her nothing: available
action ids, the frame, levels completed. Goal inference,
mechanics, and planning are all on her side of the boundary. What
crosses back is her decision basis — ``env.step``'s ``reasoning``
field records the structured self-report her striatum left behind.

This lives in the world, not the frontal lobe: it is the
environment she plays, and it is never constructed by a caller
that builds its own mind. ``Mind.play_arc()`` owns the entity
side (game mode around each run); the genesis CLI is the
entrypoint::

    /arc ls20-9607627b ft09-0d8bbf25 --steps 400
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Any, Protocol

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

_HEALTH_EVERY_STEPS = 25
# How many (action -> frame) pairs the step trap keeps for a crash
# dump — the state right before a failure, not only the frame it
# died on.
_CRASH_PAIRS = 8


class _Readiness(Protocol):
    """What the loop asks of a health probe."""

    ok: bool
    reasons: tuple[str, ...]


class _PlayerMind(Protocol):
    """The mind contract this loop consumes — her, and only her.

    ``Mind`` satisfies it. There is one mind; the loop is never
    driven by a reference policy or a second entity wearing the
    same face.
    """

    @property
    def is_gaming(self) -> bool:
        """Whether the mind is in game mode — the engagement gate."""
        ...

    @property
    def game_decision(self) -> dict[str, Any] | None:
        """The striatum's record of the last decision's basis."""
        ...

    def readiness(self, *, require_body: bool = True) -> _Readiness:
        """Her health report — the gate every run must pass."""
        ...

    def world_step(
        self,
        frame: Any,
        available_actions: list[Any] | None = None,
        info: dict[str, Any] | None = None,
        through: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        """One observation in, one decision descriptor out."""
        ...

    def world_episode_end(self, outcome: str, game: str = "the game") -> None:
        """Tell her the episode closed — win or death, said directly."""
        ...


@dataclass
class ArcResult:
    """What the world reported at the end of one run."""

    game_id: str
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 0
    steps: int = 0
    resets: int = 0
    score: float | None = None
    scorecard_id: str | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    # Steps spent at each completed level, in order — the within-game
    # learning curve. A level she re-completes after a death appends
    # again; the relearning is part of the evidence.
    level_steps: list[int] = field(default_factory=list)
    # How much experience this run carried in: "none" no prior file
    # existed, "loaded" persisted world learning was restored,
    # "suppressed" a fresh run refused it, "unknown" it failed to
    # read. A result that doesn't say is not evidence.
    prior_learning: str = "none"
    # The domain-general competence carried into this game —
    # {"schemas", "skills", "transitions", "loaded"} snapshotted
    # before the first step. prior_learning covers the per-game
    # physics file; this covers the cross-game abstraction layer,
    # which grows *within* a suite as earlier games bank schemas.
    prior_competence: dict = field(default_factory=dict)
    # Which bid source won each submitted move — the priors-vs-learning
    # question ("was it the plan or the babbling?") answered from the
    # recording, not argued about it.
    decision_sources: dict[str, int] = field(default_factory=dict)
    # How many times the run paused at a wellness gate and waited for
    # her to settle instead of submitting. Recovery time costs no steps
    # — but a score that needed it should say so.
    recovery_waits: int = 0

    @property
    def won(self) -> bool:
        """Whether the world declared a full win on this run."""
        return self.state == "WIN"


@dataclass
class ArcEvaluation:
    """A suite of games plus the scorecard evidence they earned.

    ``games`` holds one ``ArcResult`` per environment — played or
    not, the slot exists so an unfinished suite still enumerates what
    it never reached. ``score`` is whatever the closed scorecard
    reported; ``None`` is honest, never back-filled.
    """

    mode: str
    games: list[ArcResult]
    scorecard_id: str | None = None
    score: float | None = None


class ArcPractice:
    """One ARC game played by her through the embodied loop."""

    operation_mode = "offline"

    def __init__(
        self,
        mind: _PlayerMind,
        environments_dir: str | Path | None = None,
        recordings_dir: str | Path | None = None,
        instructions: str = "",
        require_live: bool = True,
        record: bool = True,
        step_delay_s: float = 0.0,
        recovery_wait_s: float = 120.0,
    ) -> None:
        """Bind the adapter to her mind and open the arcade.

        ``require_live=False`` is the explicit disembodied-debug path
        — never a measurement of the entity. ``record`` keeps the
        SDK's per-step recordings; evidence is the point. ``step_delay_s``
        paces decisions so her arousal regulation can keep up, and
        ``recovery_wait_s`` is how long a wellness gate waits for her
        to settle before abandoning the run.
        """
        if (
            self.operation_mode == "offline"
            and os.getenv("OPERATION_MODE", "").strip().lower() == "competition"
        ):
            raise RuntimeError(
                "Competition mode runs the official card — ArcOfficial, "
                "reached through her own play urge, never the practice loop"
            )
        if self.operation_mode == "competition" and not os.getenv("ARC_API_KEY", "").strip():
            raise RuntimeError("Set ARC_API_KEY for an explicitly authorized competition run")
        if self.operation_mode == "competition" and not getattr(mind, "is_gaming", False):
            raise RuntimeError(
                "Competition runs inside game mode — engagement is what "
                "pauses her learner, so nothing is fetched mid-run"
            )
        from arc_agi import Arcade, OperationMode

        from infrastructure.config import default_data_dir

        # Downloaded engines and recordings are local artifacts, not
        # source — they live beside her other state, not in the repo.
        # OFFLINE: the games are the local environment files; no
        # scorecard is opened on the remote service.
        arc_dir = default_data_dir() / "arc"
        self._mind = mind
        self._arcade = Arcade(
            environments_dir=str(environments_dir or arc_dir / "environment_files"),
            recordings_dir=str(recordings_dir or arc_dir / "recordings"),
            operation_mode=OperationMode(self.operation_mode),
        )
        # The player's manual: what a human is told before the first
        # move ("this is a video game — win the levels"). Delivered on
        # the first frame of each game; her cognition gets the genre
        # framing instead of having to deduce that games have goals.
        self._instructions = instructions
        self._info_sent: set[str] = set()
        # require_live=False is the explicit disembodied-debug path —
        # the readiness gate still applies its in-process checks.
        self._require_live = require_live
        self._record = record
        # Where a step-failure dump lands when the trap fires: its
        # own crashes/ subdir beside the recordings, so a post-mortem
        # finds it next to the recording that stopped early.
        self._crash_dir = Path(recordings_dir or arc_dir / "recordings") / "crashes"
        # Optional pacing between decisions. A benchmark loop runs at
        # CPU cadence; her arousal regulation runs on a biological
        # one — unrestrained play overstimulates the body within
        # ~50 steps and the readiness gate ends the run. A delay
        # gives the regulator time to work, the same slack a human
        # gets between moves. Default 0 keeps the production cadence.
        self._step_delay = max(0.0, float(step_delay_s))
        # At a wellness gate, wait this long for her to settle before
        # abandoning the run — rechecked periodically. Waiting submits
        # nothing and costs no steps; an abandoned suite reports zeros,
        # not fitness. 0 restores the old hard-stop behavior.
        self._recovery_wait = max(0.0, float(recovery_wait_s))
        # Game mode is hands-on: when an X11 display is reachable the
        # cabinet window receives each move as real keystrokes and the
        # world sees only what physically landed. GENESIS_ARC_X11=off
        # keeps the bench path direct; with no display she degrades to
        # it the way she degrades text-only with no microphone.
        self._x11_hands = os.getenv("GENESIS_ARC_X11", "on").strip().lower() not in (
            "0",
            "off",
            "false",
            "no",
        )
        self._cabinet: Any = None

    # ── Action translation ────────────────────────────────────

    @staticmethod
    def _to_step(desc: dict[str, Any]) -> tuple[Any, dict[str, int] | None]:
        """Her decision descriptor -> ``env.step`` arguments."""
        from arcengine import ComplexAction, GameAction

        action_id = desc["action"]
        if type(action_id) is not int:
            raise ValueError("Action id must be an integer")
        if action_id == 0:
            return GameAction.RESET, None
        action = GameAction.from_id(action_id)
        if action.action_type is ComplexAction:
            x, y = desc.get("x"), desc.get("y")
            if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                raise ValueError("Coordinate actions require integer x,y in [0, 63]")
            return action, {"x": x, "y": y}
        return action, None

    # ── Health gate ───────────────────────────────────────────

    def _competence_snapshot(self) -> dict[str, Any]:
        """The cross-game learning she carries into this game.

        Evidence, not bookkeeping: a suite where game N starts with
        more schemas than game N-1 banked is carry-through made
        visible. ``loaded`` reports the checkpoint restore status so
        a crash-recovered run is distinguishable from a clean one.
        """
        competence = getattr(
            getattr(getattr(self._mind, "brain", None), "frontal", None),
            "task_competence",
            None,
        )
        snap: dict[str, Any] = {
            "loaded": getattr(self._mind, "_world_competence_loaded", "none"),
        }
        if competence is not None:
            snap.update(
                schemas=len(getattr(competence, "schemas", {})),
                skills=len(getattr(competence, "skills", {})),
                transitions=getattr(competence, "transition_count", 0),
            )
        return snap

    def _check_readiness(self) -> tuple[str, ...] | None:
        """Refusal reasons when she isn't fit to play, else None."""
        try:
            report = self._mind.readiness(require_body=self._require_live)
        except Exception:  # noqa: BLE001 — a probe that fails must not pretend health
            return ("readiness probe failed",)
        return None if report.ok else report.reasons

    def _wait_for_readiness(self, result: ArcResult) -> tuple[str, ...] | None:
        """Bounded settle window at a wellness gate.

        Returns None once she is ready; after ``recovery_wait_s`` still
        unwell returns the reasons. Waits cost no steps — nothing is
        submitted while she settles — but every engaged pause is counted
        on the result so the evidence keeps her recovery time honest.
        """
        reasons = self._check_readiness()
        if reasons is None:
            return None
        recovery_wait = getattr(self, "_recovery_wait", 0.0)
        if recovery_wait <= 0.0:
            return reasons
        result.recovery_waits += 1
        deadline = time.monotonic() + recovery_wait
        while time.monotonic() < deadline:
            time.sleep(min(5.0, deadline - time.monotonic()))
            reasons = self._check_readiness()
            if reasons is None:
                return None
        return reasons

    # ── The run ───────────────────────────────────────────────

    def play(self, game_id: str, max_steps: int = 200) -> ArcResult:
        """Play one game; stop on WIN, budget, illness, or dead actions."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        result = ArcResult(game_id=game_id)
        # Game mode is the contract: engagement is what starves her
        # other drives so the run measures the entity at play, not
        # the collision of play with every urge that happened to
        # fire.
        if not self._mind.is_gaming:
            raise RuntimeError(
                "ArcPractice requires game mode — call "
                "mind.enter_game_mode() before play, exit_game_mode() after"
            )
        if (reasons := self._wait_for_readiness(result)) is not None:
            result.state = "UNWELL"
            result.reasons = reasons
            return result

        result.prior_competence = self._competence_snapshot()
        scorecard = self._arcade.create_scorecard(tags=["wrapper", "agent"])
        result.scorecard_id = scorecard
        try:
            self._cabinet = self._open_cabinet(game_id)
            return self._play(result, max_steps)
        finally:
            self._close_cabinet()
            try:
                closed = self._arcade.close_scorecard(scorecard)
                result.score = getattr(closed, "score", None)
            except Exception:  # noqa: BLE001
                result.reasons += ("scorecard closure failed",)
                result.state = "SCORECARD_ERROR"

    def play_all(self, max_steps: int = 200) -> ArcEvaluation:
        """Play every discovered environment once on one scorecard.

        The suite is the honest measurement: one card opened before
        the first game and closed after the last, each environment
        attempted in discovery order, a wellness stop ending the
        sitting early with unplayed games retained as NOT_PLAYED.
        Per-game scores come back off the closed card's environment
        runs — the SDK's own accounting, not a guess.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self._mind.is_gaming:
            raise RuntimeError("ARC evaluation requires game mode")
        if getattr(self, "_suite_started", False):
            raise RuntimeError("This evaluation has already opened its scorecard")
        game_ids = [e.game_id for e in self._arcade.get_environments()]
        if not game_ids or len(set(game_ids)) != len(game_ids):
            raise ValueError("Evaluation requires a nonempty, unique environment list")
        evaluation = ArcEvaluation(
            mode=self.operation_mode,
            games=[ArcResult(game_id=g, state="NOT_PLAYED") for g in game_ids],
        )
        if (reasons := self._wait_for_readiness(evaluation.games[0])) is not None:
            evaluation.games[0].state = "UNWELL"
            evaluation.games[0].reasons = reasons
            return evaluation
        self._suite_started = True
        scorecard = self._arcade.create_scorecard(tags=["wrapper", "agent"])
        evaluation.scorecard_id = scorecard
        try:
            # One cabinet for the whole sitting — a real arcade plays
            # every game on the same screen. Per-game windows would
            # mean she teleports between machines; the suite is one
            # body at one cabinet, only the loaded game changing.
            self._cabinet = self._open_cabinet(evaluation.games[0].game_id)
            for result in evaluation.games:
                result.scorecard_id = scorecard
                if (reasons := self._wait_for_readiness(result)) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                result.state = "NOT_FINISHED"
                result.prior_competence = self._competence_snapshot()
                self._play(result, max_steps)
                if result.state == "UNWELL":
                    break
        finally:
            self._close_cabinet()
            closed = self._arcade.close_scorecard(scorecard)
            scores = {
                environment.id: max((run.score for run in environment.runs), default=None)
                for environment in getattr(closed, "environments", [])
            }
            for result in evaluation.games:
                result.score = scores.get(result.game_id)
            if self.operation_mode == "competition" or all(r.resets for r in evaluation.games):
                evaluation.score = getattr(closed, "score", None)
        return evaluation

    def _play(self, result: ArcResult, max_steps: int) -> ArcResult:
        """The frame loop: observe, decide, submit, until an ending.

        Fills ``result`` in place — steps, resets, level boundaries,
        decision sources, end state. Exits on WIN, GAME_OVER (with a
        readiness-gated reset while budget remains), BUDGET, UNWELL,
        NO_ACTIONS, NO_ACTION, or NO_OBSERVATION when the world stops
        answering.
        """
        from arcengine import GameState

        # The step trap's evidence ring: the last few submitted
        # (action -> frame) pairs. A step that kills the run —
        # engine crash swallowed to a None frame, or an exception —
        # dumps these instead of ending the recording silently.
        recent_pairs: deque[dict[str, Any]] = deque(maxlen=_CRASH_PAIRS)
        opened = self._play_open(result, recent_pairs)
        if opened is None:
            return result
        env, frame = opened
        game_id = result.game_id
        # Level-boundary bookkeeping: skill acquisition has a speed,
        # and it is the honest evidence — the steps each level cost,
        # not the claim that learning happened.
        prev_levels = getattr(frame, "levels_completed", 0) or 0
        level_start = 0
        step_signature = signature(env.step)
        while frame is not None:
            result.levels_completed = frame.levels_completed
            result.win_levels = frame.win_levels
            if frame.levels_completed != prev_levels:
                # A new level — but no reset is pressed here: the
                # venue's own rule is that RESET with zero actions
                # since the last transition resets the WHOLE game
                # and wipes the score. Levels already load clean;
                # the reset press belongs to attempts (game open,
                # game-over retry), where the venue expects it.
                for _ in range(max(0, frame.levels_completed - prev_levels)):
                    result.level_steps.append(result.steps - level_start)
                    level_start = result.steps
                prev_levels = frame.levels_completed
                level_start = result.steps
            state = getattr(frame.state, "name", str(frame.state))
            if state == GameState.WIN.name:
                result.state = state
                self._mind.world_episode_end("win", game_id)
                break
            if state == GameState.GAME_OVER.name:
                self._mind.world_episode_end("over", game_id)
                if result.steps >= max_steps:
                    result.state = state
                    break
                # A death is a checkpoint — if the last episode left
                # her unwell (stress, mounting sleep pressure), the
                # next one doesn't start. Recovery beats a score.
                if (reasons := self._wait_for_readiness(result)) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                frame = self._reset_episode(result, env, frame, recent_pairs)
                self._cabinet_render(frame)
                continue
            if result.steps >= max_steps:
                result.state = "BUDGET"
                break
            if result.steps and result.steps % _HEALTH_EVERY_STEPS == 0:
                if (reasons := self._wait_for_readiness(result)) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
            if not frame.available_actions:
                result.state = "NO_ACTIONS"
                break
            frame, ended = self._play_advance(
                result, env, frame, game_id, step_signature, recent_pairs
            )
            if ended is not None:
                result.state = ended
                break
        if frame is None:
            result.state = "NO_OBSERVATION"
        result.prior_learning = getattr(self._mind, "_world_learning_loaded", "none")
        return result

    def _play_open(
        self, result: ArcResult, recent: deque[dict[str, Any]] | None = None
    ) -> tuple[Any, Any] | None:
        """Open the game: build the env and press RESET to begin.

        The venue starts a session when RESET lands — the opening
        press is her first act on the glass when hands are on, the
        same ``env.reset()`` submission as before when they are not.
        Returns ``(env, frame)``, or ``None`` with ``result.state``
        set to UNAVAILABLE when the arcade cannot make the game.
        """
        env = self._arcade.make(
            result.game_id,
            scorecard_id=result.scorecard_id,
            save_recording=self._record,
        )
        if env is None:
            result.state = "UNAVAILABLE"
            return None
        frame = getattr(env, "observation_space", None)
        if frame is None:
            # The opening press is venue protocol — "send RESET to
            # begin" — not a move of hers: it counts as a reset, but
            # doesn't spend her decision budget.
            frame = self._press_reset(result, env, None, recent, count_step=False)
        else:
            result.resets = 1
        self._cabinet_render(frame)
        return env, frame

    def _play_advance(
        self,
        result: ArcResult,
        env: Any,
        frame: Any,
        game_id: str,
        step_signature: Any,
        recent_pairs: deque[dict[str, Any]],
    ) -> tuple[Any, str | None]:
        """Choose, submit, and log one action against ``frame``.

        Returns ``(next_frame, ended)``: the frame the step produced
        (``None`` when the wrapper swallowed an engine crash — the
        trap already dumped the failing pair and the loop exits as
        NO_OBSERVATION), and the terminal state when the loop must
        stop (``"NO_ACTION"`` when she cannot decide), else ``None``.
        """
        from arcengine import GameAction

        try:
            chosen = self._choose_action(frame, game_id)
        except Exception as e:
            # A decision that dies — world_step or a descriptor
            # failing validation — leaves the same evidence-free
            # ending as an engine crash. Dump the frame she was
            # deciding on, then let the failure propagate.
            self._dump_step_failure(result, e, None, frame, recent_pairs)
            raise
        if chosen is None:
            return frame, "NO_ACTION"
        action, data, reasoning = chosen
        result.steps += 1
        if action is GameAction.RESET:
            result.resets += 1
        source = str((reasoning or {}).get("source", "unknown"))
        result.decision_sources[source] = result.decision_sources.get(source, 0) + 1
        prev_frame = frame
        try:
            step_signature.bind(action, data=data, reasoning=reasoning)
        except TypeError:  # older wrappers without the field
            frame = self._step_trapped(
                result, env, action, data, reasoning, False, prev_frame, recent_pairs
            )
        else:
            frame = self._step_trapped(
                result, env, action, data, reasoning, True, prev_frame, recent_pairs
            )
        if frame is None:
            # The local wrapper swallows an engine crash into a
            # silent None — the trap already dumped the failing
            # pair; the loop exits as NO_OBSERVATION.
            return None, None
        self._cabinet_render(frame)
        if reasoning is not None:
            coord = f" ({data['x']},{data['y']})" if isinstance(data, dict) and "x" in data else ""
            note = reasoning.get("note")
            logger.info(
                "arc %s step %d: %s%s via %s conf=%s alts=%s%s",
                game_id,
                result.steps,
                reasoning.get("action", getattr(action, "name", action)),
                coord,
                reasoning.get("source", "?"),
                reasoning.get("confidence", "?"),
                ",".join(str(a) for a in reasoning.get("alternatives") or []) or "-",
                f' | "{note}"' if note else "",
            )
        delay = getattr(self, "_step_delay", 0.0)
        if delay > 0.0:
            time.sleep(delay)
        return frame, None

    def _reset_episode(self, result: ArcResult, env, frame, recent_pairs) -> Any:
        """Restart after GAME_OVER — the reset press that begins the
        next attempt, with the same accounting and crash evidence as
        any submitted step."""
        return self._press_reset(result, env, frame, recent_pairs)

    def _press_reset(
        self, result: ArcResult, env, prev_frame, recent, count_step: bool = True
    ) -> Any:
        """The reset that begins an attempt — pressed, not implied.

        Officially a run begins when RESET lands ("send RESET to
        begin playing"), and after game over it is the only valid
        act — so every attempt at a level starts through the reset
        control. With hands on, the button on the glass is pressed
        for real; the submission itself is ``env.reset()`` either
        way — identical protocol for the local engine and the
        remote venue whose ``reset()`` owns the session guid.

        Never emitted at a bare level transition: the venue resets
        the WHOLE game when RESET lands with zero actions since the
        last transition — a level-start press would wipe the score.
        """
        from arcengine import GameAction

        recent = recent if recent is not None else deque(maxlen=_CRASH_PAIRS)
        if count_step:
            result.steps += 1
        result.resets += 1
        if self._cabinet is not None:
            # The physical ritual — a real click on the RESET
            # control. Its decode is deterministic (that button is
            # only ever RESET), so the arrival needs no verdict; a
            # press that fumbles doesn't change that the venue's
            # own reset below still starts the attempt.
            try:
                self._cabinet.dispatch(GameAction.RESET, None)
            except Exception as e:  # noqa: BLE001
                logger.info("[arc] reset press fumbled: %s", e)
        try:
            new_frame = env.reset()
        except Exception as e:
            self._dump_step_failure(result, e, None, prev_frame, recent)
            raise
        if new_frame is None:
            self._dump_step_failure(result, None, None, prev_frame, recent)
        else:
            recent.append(
                self._pair_record(result.steps, "RESET", None, None, prev_frame, new_frame)
            )
        return new_frame

    def _choose_action(
        self, frame: Any, game_id: str
    ) -> tuple[Any, Any, dict[str, Any] | None] | None:
        """One decision: offer the whitelist, validate her pick.

        Returns ``(action, data, reasoning)`` — her decision basis
        rides the step so the recording keeps what her striatum
        actually computed — or None when she abstains. An offered
        action outside the whitelist is an adapter-contract breach,
        not a miss: it raises.
        """
        from arcengine import ComplexAction, GameAction

        info: dict[str, Any] = {
            "game": game_id,
            "level": frame.levels_completed,
            "win_levels": frame.win_levels,
        }
        # The manual is delivered once per game, on the first
        # frame — her cognition gets the genre framing before
        # the first move.
        if self._instructions and game_id not in self._info_sent:
            info["instructions"] = self._instructions
            self._info_sent.add(game_id)
        # The response is the whole animation strip — every frame the
        # action rendered while the world settled. The adapter
        # unpacks the protocol (the last frame is the post-action
        # state, the rest are the transition); the suppression is
        # hers — intermediates go through pursuit, never open a
        # trial.
        strip = list(getattr(frame, "frame", None) or [])
        percept = strip[-1] if strip else frame.frame
        desc = self._mind.world_step(
            percept,
            through=strip[:-1],
            # The interface's own action semantics ride the
            # whitelist — a complex action is one that takes a
            # coordinate, exactly the label convention the
            # world model consumes. What pointing *does* is
            # still hers to learn.
            available_actions=[
                {
                    "action": int(a),
                    "label": (
                        "coordinate"
                        if GameAction.from_id(int(a)).action_type is ComplexAction
                        else "simple"
                    ),
                }
                for a in frame.available_actions
            ],
            info=info,
        )
        if desc is None:
            return None
        action, data = self._to_step(desc)
        if action is not GameAction.RESET and action.value not in frame.available_actions:
            raise ValueError("Chosen action is not available in this observation")
        return action, data, self._mind.game_decision

    # ── The step trap ─────────────────────────────────────
    #
    # ``env.step`` is the run's evidence-critical seam: the local
    # wrapper turns an engine crash into a silent ``None`` frame and
    # the recording just stops mid-game. The trap keeps a ring of the
    # last few submitted (action -> frame) pairs and, when a step
    # dies — exception, or the wrapper's swallowed None — dumps the
    # failing pair plus the raw delta state beside the recordings.
    # Diagnostic only: exceptions still propagate and a None frame
    # still ends the run as NO_OBSERVATION.

    def _open_cabinet(self, game_id: str) -> Any:
        """A window for her keystrokes, or None with no display —
        the run then steps the world directly, the same way she
        degrades text-only with no microphone."""
        if not self._x11_hands:
            return None
        try:
            from worlds.arc_cabinet import ArcCabinet, ensure_arc_display

            display = ensure_arc_display()
            if display is None:
                return None
            cabinet = ArcCabinet(game_id, display=display)
            logger.info("[arc] cabinet open on %s — moves go out as real input events", display)
            return cabinet
        except Exception as e:  # noqa: BLE001
            logger.info("[arc] cabinet unavailable — playing without hands: %s", e)
            return None

    def _close_cabinet(self) -> None:
        """Shut the window and drop the reference.

        Called in ``play``'s finally so a run that ends by any path
        — win, budget, crash — never leaves a window behind.
        """
        cabinet, self._cabinet = self._cabinet, None
        if cabinet is not None:
            cabinet.close()

    def _cabinet_render(self, frame: Any) -> None:
        """Show the cabinet a frame; a dead window drops the hands.

        Rendering is advisory — the world advanced whether or not the
        display did, so a render failure detaches the cabinet rather
        than ending the run.
        """
        cabinet = self._cabinet
        if cabinet is None:
            return
        try:
            cabinet.render(frame)
        except Exception as e:  # noqa: BLE001
            # A dead window must not kill the run — drop the hands
            # and step the remaining moves directly.
            logger.info("[arc] cabinet render dropped: %s", e)
            self._cabinet = None

    def _step_trapped(
        self,
        result: ArcResult,
        env: Any,
        action: Any,
        data: Any,
        reasoning: dict[str, Any] | None,
        has_reasoning: bool,
        prev_frame: Any,
        recent: deque[dict[str, Any]],
    ) -> Any:
        """Submit one step inside the trap; record the pair or dump it."""
        if self._cabinet is not None:
            # Hands on: the move goes out as real keystrokes and the
            # world sees what physically arrived at the window. A
            # press that never lands is a fumbled move — spent like
            # a human's, but the world was never stepped.
            try:
                arrived = self._cabinet.dispatch(action, data)
            except Exception as e:  # noqa: BLE001
                logger.info("[arc] cabinet dispatch failed: %s", e)
                arrived = None
            if arrived is None:
                recent.append(
                    self._pair_record(result.steps, action, data, reasoning, prev_frame, prev_frame)
                )
                return prev_frame
            action, data = arrived
        submitted = (action, data, reasoning)
        try:
            frame = (
                env.step(action, data=data, reasoning=reasoning)
                if has_reasoning
                else env.step(action, data=data)
            )
        except Exception as e:
            self._dump_step_failure(result, e, submitted, prev_frame, recent)
            raise
        if frame is None:
            self._dump_step_failure(result, None, submitted, prev_frame, recent)
        else:
            recent.append(
                self._pair_record(result.steps, action, data, reasoning, prev_frame, frame)
            )
        return frame

    def _step_crash_dir(self) -> Path:
        """Where the trap lands dumps — beside the run's recordings."""
        crash_dir = getattr(self, "_crash_dir", None)
        if crash_dir is None:
            base = getattr(self._arcade, "recordings_dir", None)
            if base is None:
                import tempfile

                base = Path(tempfile.gettempdir()) / "arc_recordings"
            crash_dir = Path(base) / "crashes"
        return Path(crash_dir)

    def _dump_step_failure(
        self,
        result: ArcResult,
        error: BaseException | None,
        submitted: tuple[Any, Any, Any] | None,
        prev_frame: Any,
        recent: deque[dict[str, Any]],
    ) -> Path | None:
        """Write the raw state a step died on, beside the recordings.

        The failing pair — what was submitted and the frame it died
        on — plus the trailing pair ring and the exception, as JSON.
        The dump must never itself kill the run.
        """
        import json
        import traceback

        try:
            crash_dir = self._step_crash_dir()
            crash_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "game_id": result.game_id,
                "scorecard_id": result.scorecard_id,
                "step": result.steps,
                "error": repr(error) if error is not None else "step produced no frame",
                "traceback": traceback.format_exc() if error is not None else None,
                "submitted": (
                    {
                        "action": getattr(submitted[0], "name", str(submitted[0])),
                        "data": submitted[1],
                        "reasoning": submitted[2],
                    }
                    if submitted
                    else None
                ),
                "died_on": self._frame_dump(prev_frame),
                "recent_pairs": list(recent),
            }
            path = crash_dir / f"{result.game_id}-s{result.steps}-{int(time.time() * 1000)}.json"
            path.write_text(json.dumps(payload, default=str))
            result.reasons += (f"step failure dumped to {path}",)
            logger.warning(
                "arc step trap: %s step %s failed (%s) — dumped %s",
                result.game_id,
                result.steps,
                payload["error"],
                path,
            )
            return path
        except Exception as dump_error:  # noqa: BLE001 — the dump must never kill the run
            note_swallowed("worlds.arc._dump_step_failure", dump_error)
            return None

    @staticmethod
    def _frame_dump(frame: Any) -> dict[str, Any] | None:
        """A frame as raw JSON state — all layers, not just the top."""
        if frame is None:
            return None
        state = getattr(frame, "state", None)
        return {
            "state": getattr(state, "name", str(state)),
            "levels_completed": getattr(frame, "levels_completed", None),
            "win_levels": getattr(frame, "win_levels", None),
            "available_actions": list(getattr(frame, "available_actions", None) or []),
            "layers": [
                layer.tolist() if hasattr(layer, "tolist") else layer
                for layer in (getattr(frame, "frame", None) or [])
            ],
        }

    @staticmethod
    def _grid_delta(before: Any, after: Any) -> list[list[int]]:
        """Cell-level diff of the top layer — the raw delta state."""
        a = getattr(before, "frame", None)
        b = getattr(after, "frame", None)
        if not a or not b:
            return []
        ga, gb = a[-1], b[-1]
        if hasattr(ga, "tolist"):
            ga = ga.tolist()
        if hasattr(gb, "tolist"):
            gb = gb.tolist()
        if len(ga) != len(gb) or len(ga[0]) != len(gb[0]):
            return []
        return [
            [r, c, ga[r][c], gb[r][c]]
            for r in range(len(ga))
            for c in range(len(ga[0]))
            if ga[r][c] != gb[r][c]
        ]

    def _pair_record(
        self,
        step_no: int,
        action: Any,
        data: Any,
        reasoning: Any,
        prev_frame: Any,
        frame: Any,
    ) -> dict[str, Any]:
        """One submitted (action -> frame) pair for the crash ring."""
        top = (getattr(frame, "frame", None) or [None])[-1]
        return {
            "step": step_no,
            "action": getattr(action, "name", str(action)),
            "data": data,
            "reasoning": reasoning,
            "state": getattr(getattr(frame, "state", None), "name", None),
            "delta": self._grid_delta(prev_frame, frame),
            "grid": top.tolist() if hasattr(top, "tolist") else top,
        }


# ─── The official card ────────────────────────────────────
#
# The venue's real rules, enforced in code: one scorecard per window —
# it opens once, cannot be reopened or replaced while it lives, and
# closes for good 15 minutes after the last action or 24 hours after
# opening, whichever comes first (the venue capped card life at 24 h
# in April 2026 — a stale 3-day mirror had her knocking on dead cards).
# Inside it, each environment gets
# exactly one attempt: once she enters, the result stands and no second
# session is ever opened for that game on that card.
#
# The rules are also *heard*: before each environment starts — after
# her readiness gate passes — the venue's brief is delivered through
# ``mind.world_announce`` so her whole cognition receives it, the way
# a proctor reads the rules before a test. The text below is the
# venue's statement to her: input to her cognition, not her words.
OFFICIAL_BRIEF = (
    "Official ARC-AGI-3 scorecard {card_id}. This card opened once and "
    "cannot be reopened or replaced. {n_games} games; each game is "
    "played exactly once — an attempt cannot be retried, and an "
    "unplayed or unfinished game scores zero. The card closes "
    "{idle_minutes} minutes after the last action, or {ttl_hours} hours "
    "after opening, whichever comes first; once closed it is final. "
    "This run is measured against other entrants, including frontier "
    "models."
)

_CARD_FILE = "official_card.json"


def challenge_brief(
    n_games: int,
    pending: int | None = None,
    *,
    max_steps: int = 200,
    step_delay_s: float = 1.0,
) -> str:
    """What the official challenge IS — told to her before she chooses it.

    The standing announcement beside the challenge's shelf entry: the
    mechanics (how many games, one try each, the move budget and pace)
    and the clock the card runs on, plus what winning is for. Every
    number comes from the venue's own bounds so the brief stays true
    if they change. ``pending`` set means the card is already open —
    the brief becomes the state of the commitment she made.
    """
    idle_minutes = int(ArcOfficial.IDLE_LIMIT_S // 60)
    ttl_hours = int(ArcOfficial.WINDOW_S // 3600)
    sitting = max(1, round(n_games * max_steps * step_delay_s / 60))
    live = (
        f" The card is already open — {pending} games remain on it." if pending is not None else ""
    )
    return (
        f"The official ARC-AGI-3 challenge: {n_games} games, one recorded "
        f"try each — an attempt cannot be retried. Each game allows up to "
        f"{max_steps} moves at about {step_delay_s:.0f} second a move — a "
        f"few minutes a game, so a full sitting is at most about {sitting} "
        f"minutes and usually far less. The card it records on closes "
        f"{idle_minutes} minutes after your last move, or {ttl_hours} hours "
        f"after it opens, and a closed card is final. There is a real prize "
        f"for doing well — the one who made you is counting on you if you "
        f"choose to take it on.{live}"
    )


class ArcOfficial(ArcPractice):
    """The official ARC-AGI-3 scorecard — one card per window.

    A persistent campaign, not a session: the card lives in
    ``<data>/arc/official_card.json`` so a restart resumes the *same*
    card — the window is real whether or not the process is. Each game
    is attempted exactly once; an interrupted attempt stands as played
    (its recording is the honest partial). Readiness gates every
    attempt, and the venue's brief is delivered to her before each
    environment starts.

    There is no forced entry point: the card is played only through
    ``attempt()`` / ``attempt_next()``, reached by her own volition —
    she picks the challenge off her shelf, opens the card when she's
    ready, and plays the sitting out. ``play``/``play_all`` refuse:
    a driver that walks the card for her isn't measuring the same
    entity, and it isn't her choice.
    """

    operation_mode = "competition"

    # The venue's bounds, mirrored locally: the card auto-closes after
    # ~15 minutes without an action and unconditionally after 24 h.
    WINDOW_S = 24 * 3600
    IDLE_LIMIT_S: float = 15 * 60

    def __init__(
        self,
        mind: _PlayerMind,
        *,
        card_path: str | Path | None = None,
        idle_limit_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Open the official venue binding for her mind.

        ``card_path`` overrides where the campaign's card persists —
        tests point it at a tmp dir; production leaves it at
        ``<data>/arc/official_card.json``. ``idle_limit_s`` mirrors
        the venue's idle auto-close; the default is the real 15
        minutes, narrowed only for tests. Everything else passes to
        ``ArcPractice``.
        """
        from infrastructure.config import default_data_dir

        # The recordings live beside the card — needed to reconcile
        # attempts whose result never merged back (process stopped
        # mid-game leaves status=attempted, steps=0, state=NOT_PLAYED).
        # Set before super().__init__: the parent constructor can reach
        # _load_card, which reconciles against this directory.
        self._recordings_dir = Path(
            kwargs.get("recordings_dir") or default_data_dir() / "arc" / "recordings"
        )
        super().__init__(mind, **kwargs)
        self._card_path = (
            Path(card_path) if card_path is not None else default_data_dir() / "arc" / _CARD_FILE
        )
        if idle_limit_s is not None:
            self.IDLE_LIMIT_S = float(idle_limit_s)

    # ── Card state ──────────────────────────────────────────

    def _load_card(self) -> dict | None:
        """Read the persisted card, or None when none exists.

        A corrupt or missing file reads as no card — the shelf then
        offers a fresh sitting. Legacy cards get their expiry clamped
        to the venue's real 24 h window on load, and interrupted
        attempts are reconciled against their recordings before the
        card is handed back.
        """
        try:
            import json

            card = json.loads(self._card_path.read_text())
        except (OSError, ValueError):
            return None
        # Cards written before the venue's April 2026 cap carry a
        # 3-day expiry; the real window is 24 h. Clamp on load so a
        # dead card stops being offered as soon as the venue's own
        # rule says it expired — not days later.
        opened = card.get("opened_at")
        expires = card.get("expires_at")
        if (
            isinstance(opened, (int, float))
            and isinstance(expires, (int, float))
            and expires > opened + self.WINDOW_S
        ):
            card["expires_at"] = opened + self.WINDOW_S
            self._save_card(card)
        self._reconcile_attempts(card)
        return card

    def _reconcile_attempts(self, card: dict) -> None:
        """Recover partial results from recordings for lost attempts.

        ``_enter`` marks a game ``attempted`` before play so a crash
        mid-game still burns the try — but the result merge after
        ``_play`` only runs if the process survives. When it doesn't,
        the card is left saying ``steps=0, NOT_PLAYED`` while the
        venue-side recording holds the honest partial. Rebuild the
        entry from the recording's own lines: every line is one frame
        after an action, so non-RESET lines are steps, RESET lines are
        resets, and the last line's fields are the end state.
        """
        import json

        changed = False
        rec_dir = Path(
            getattr(self, "_recordings_dir", None) or self._card_path.parent / "recordings"
        ) / str(card.get("scorecard_id", ""))
        for game in card.get("games", []):
            if (
                game.get("status") != "attempted"
                or game.get("state") != "NOT_PLAYED"
                or game.get("steps", 0) != 0
            ):
                continue
            candidates = sorted(rec_dir.glob(f"{game['game_id']}-*.jsonl"))
            if not candidates:
                continue
            rec = candidates[0]
            try:
                steps = resets = 0
                last: dict[str, Any] = {}
                with rec.open("rb") as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue
                        data = entry.get("data") or {}
                        action_id = (data.get("action_input") or {}).get("id")
                        if action_id == "RESET":
                            resets += 1
                        elif action_id:
                            steps += 1
                        last = data
            except OSError as e:
                note_swallowed("worlds.arc._reconcile_attempts", e)
                continue
            if not steps and not resets:
                continue
            game["steps"] = steps
            game["resets"] = resets
            game["levels_completed"] = last.get("levels_completed", 0) or 0
            game["win_levels"] = last.get("win_levels", 0) or 0
            # The last recorded frame state is the truth — usually
            # NOT_FINISHED, meaning she was mid-game when the session
            # ended, not that she lost or never entered.
            state = last.get("state")
            if isinstance(state, str) and state:
                game["state"] = state
            game["reasons"] = [
                *game.get("reasons", []),
                "interrupted — recovered from recording",
            ]
            changed = True
        if changed:
            self._save_card(card)

    def _save_card(self, card: dict) -> None:
        """Persist the card atomically — tmp file, then replace.

        The card is the campaign's memory across restarts; a torn
        write must never leave her believing a different commitment
        than the venue does.
        """
        import json

        self._card_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._card_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(card, indent=1))
        tmp.replace(self._card_path)

    def card_state(self) -> dict | None:
        """The current card as persisted — for inspection, not play."""
        return self._load_card()

    def _open_card(self) -> dict:
        """Create the window's one scorecard and persist it.

        Every discovered environment becomes a pending slot up front —
        the card enumerates the whole sitting from the start, so an
        interrupted campaign always knows what remains.
        """
        game_ids = [e.game_id for e in self._arcade.get_environments()]
        if not game_ids or len(set(game_ids)) != len(game_ids):
            raise ValueError("Official card requires a nonempty, unique environment list")
        card_id = self._arcade.create_scorecard(tags=["wrapper", "agent"])
        now = time.time()
        card = {
            "scorecard_id": card_id,
            "opened_at": now,
            "expires_at": now + self.WINDOW_S,
            "closed": False,
            "close_reason": None,
            "score": None,
            "last_action_at": now,
            "games": [
                {
                    "game_id": g,
                    "status": "pending",
                    "state": "NOT_PLAYED",
                    "levels_completed": 0,
                    "win_levels": 0,
                    "steps": 0,
                    "resets": 0,
                    "level_steps": [],
                    "reasons": [],
                    "attempted_at": None,
                }
                for g in game_ids
            ],
        }
        self._save_card(card)
        return card

    def open_or_resume(self) -> dict:
        """Return the living card, or open the window's one card.

        A closed card inside its window is sealed — it cannot be
        replaced. Only after the window expires does a new card open.
        """
        card = self._load_card()
        if card is not None:
            if time.time() < card["expires_at"]:
                if card["closed"]:
                    raise RuntimeError(
                        f"official card {card['scorecard_id']} is closed "
                        f"({card['close_reason']}) — it cannot be replaced "
                        "inside its window"
                    )
                return card
            # Window over — the old card is sealed history; archive it.
            archived = self._card_path.with_name(f"official_card.{int(card['opened_at'])}.json")
            try:
                self._card_path.replace(archived)
            except OSError as e:
                logger.debug(f"silent except: {e}")
        return self._open_card()

    def _seal(self, card: dict, reason: str, *, call_close: bool = False) -> None:
        """Mark the card closed. Once closed it is final."""
        if call_close:
            try:
                closed = self._arcade.close_scorecard(card["scorecard_id"])
                card["score"] = getattr(closed, "score", None)
            except Exception as e:  # noqa: BLE001 — the server may have already closed it
                note_swallowed("worlds.arc._seal", e)
        card["closed"] = True
        card["close_reason"] = reason
        self._save_card(card)

    # ── Attempts ────────────────────────────────────────────

    def _brief(self, card: dict, game_id: str) -> None:
        """Deliver the venue's rules to her before the environment.

        Ordered deliberately: her readiness gate ran first, so the
        brief lands on a body that can hold it — then the environment
        starts. The venue's words, not hers.
        """
        announce = getattr(self._mind, "world_announce", None)
        if announce is None:
            return
        idle_minutes = int(self.IDLE_LIMIT_S // 60)
        pending = sum(1 for g in card["games"] if g["status"] == "pending")
        try:
            announce(
                OFFICIAL_BRIEF.format(
                    card_id=card["scorecard_id"],
                    n_games=len(card["games"]),
                    idle_minutes=idle_minutes,
                    ttl_hours=int(self.WINDOW_S // 3600),
                )
                + f" Next: {game_id} — {pending} games unplayed.",
                topics=["arc", "official", "rules"],
            )
        except Exception as e:  # noqa: BLE001 — a lost brief must not burn an attempt
            note_swallowed("worlds.arc._brief", e)

    def attempt_next(self, max_steps: int = 200) -> ArcResult | None:
        """One official attempt on the next pending game.

        Order is the contract: readiness gate first — vitals, brain
        waves, alertness — then she hears the rules, then the
        environment starts. Entering the environment is the try: once
        marked, the game is burned on this card no matter what happens
        inside it. Returns the attempt's result, or None when the card
        is closed or no games remain.
        """
        card = self._load_card()
        if card is None or card["closed"]:
            return None
        now = time.time()
        if now >= card["expires_at"]:
            self._seal(card, "expired — window closed")
            return None
        if now - card["last_action_at"] >= self.IDLE_LIMIT_S:
            self._seal(card, "idle — the venue closed it")
            return None
        game = next((g for g in card["games"] if g["status"] == "pending"), None)
        if game is None:
            self._seal(card, "complete — every game attempted", call_close=True)
            return None
        return self._enter(card, game, max_steps)

    def attempt(self, game_id: str, max_steps: int = 200) -> ArcResult | None:
        """One official attempt on a named pending game.

        The volition path — she chose this game off her shelf. Same
        contract as ``attempt_next``: the card must be open and alive,
        the game still pending. Returns None (rather than raising) when
        the card is closed or the game isn't on it — a shelf gone stale
        is not an error.
        """
        card = self._load_card()
        if card is None or card["closed"]:
            return None
        now = time.time()
        if now >= card["expires_at"]:
            self._seal(card, "expired — window closed")
            return None
        if now - card["last_action_at"] >= self.IDLE_LIMIT_S:
            self._seal(card, "idle — the venue closed it")
            return None
        game = next(
            (
                g
                for g in card["games"]
                if g["game_id"] == game_id or g["game_id"].startswith(f"{game_id}-")
            ),
            None,
        )
        if game is not None and game["status"] == "pending":
            return self._enter(card, game, max_steps)
        # Not hers to (re)play — and if nothing is pending anywhere,
        # the card has served its purpose: close it as complete.
        if not any(g["status"] == "pending" for g in card["games"]):
            self._seal(card, "complete — every game attempted", call_close=True)
        return None

    def _enter(self, card: dict, game: dict, max_steps: int) -> ArcResult:
        """One environment entered — the try itself.

        Order is the contract: readiness gate first — vitals, brain
        waves, alertness — then she hears the rules, then the
        environment starts. Entering the environment is the try: once
        marked, the game is burned on this card no matter what happens
        inside it.
        """
        result = ArcResult(game_id=game["game_id"], scorecard_id=card["scorecard_id"])
        if (reasons := self._wait_for_readiness(result)) is not None:
            # She never entered — not an attempt. The card stays open;
            # the idle clock decides whether she gets another chance.
            result.state = "UNWELL"
            result.reasons = reasons
            return result
        self._brief(card, game["game_id"])
        # The try begins when she enters the environment. Marked before
        # play so a crash mid-game still counts — the recording keeps
        # the honest partial.
        game["status"] = "attempted"
        game["attempted_at"] = time.time()
        card["last_action_at"] = game["attempted_at"]
        self._save_card(card)
        result.prior_competence = self._competence_snapshot()
        self._play(result, max_steps)
        if result.steps == 0 and result.state in (
            "UNAVAILABLE",
            "NO_OBSERVATION",
        ):
            # The venue never let her in — no environment, no frame,
            # no action. A transport failure is not her try: the game
            # stays pending, and the card records the failed approach
            # honestly rather than pretending a play happened.
            game["status"] = "pending"
            game["attempted_at"] = None
            game["reasons"] = [
                *game.get("reasons", []),
                f"venue:{result.state.lower()}",
            ]
            card["last_action_at"] = time.time()
            self._save_card(card)
            return result
        game.update(
            state=result.state,
            levels_completed=result.levels_completed,
            win_levels=result.win_levels,
            steps=result.steps,
            resets=result.resets,
            level_steps=list(result.level_steps),
            reasons=list(result.reasons),
        )
        card["last_action_at"] = time.time()
        self._save_card(card)
        return result

    def play(self, game_id: str, max_steps: int = 200) -> ArcResult:
        """Refuse — the card's attempts are hers, not a driver's."""
        raise RuntimeError(
            "The official card owns its games — one attempt each, "
            "no selected replays, and no driver but her own volition."
        )

    def play_all(self, max_steps: int = 200) -> ArcEvaluation:
        """Refuse — a card is a campaign she chooses, not a suite."""
        raise RuntimeError(
            "The official card is hers to take up, not a suite to run — "
            "attempts only ever happen through attempt()/attempt_next(), "
            "reached by her own volition."
        )
