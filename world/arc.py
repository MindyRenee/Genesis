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

import os
import time
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Any, Protocol

_HEALTH_EVERY_STEPS = 25


class _Readiness(Protocol):
    """What the loop asks of a health probe."""

    ok: bool
    reasons: tuple[str, ...]


class _PlayerMind(Protocol):
    """The mind contract this loop consumes.

    ``Mind`` satisfies it; so does the named reference policy in
    ``world/arc_baseline.py`` — the denominator runs through the
    same loop by wearing the same face, which is what makes its
    number comparable to hers.
    """

    @property
    def is_gaming(self) -> bool: ...
    @property
    def game_decision(self) -> dict[str, Any] | None: ...
    def readiness(self, *, require_body: bool = True) -> _Readiness: ...
    def world_step(
        self,
        frame: Any,
        available_actions: list[Any] | None = None,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...
    def world_episode_end(self, outcome: str, game: str = "the game") -> None: ...


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

    @property
    def won(self) -> bool:
        return self.state == "WIN"


@dataclass
class ArcEvaluation:
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
    ) -> None:
        if (
            self.operation_mode == "offline"
            and os.getenv("OPERATION_MODE", "").strip().lower() == "competition"
        ):
            raise RuntimeError("Competition mode requires ArcCompetition.play_all()")
        if self.operation_mode == "competition" and not os.getenv("ARC_API_KEY", "").strip():
            raise RuntimeError("Set ARC_API_KEY for an explicitly authorized competition run")
        if self.operation_mode == "competition" and not getattr(mind, "_offline", False):
            raise RuntimeError("Competition requires Genesis to start with offline cognition")
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
        # Optional pacing between decisions. A benchmark loop runs at
        # CPU cadence; her arousal regulation runs on a biological
        # one — unrestrained play overstimulates the body within
        # ~50 steps and the readiness gate ends the run. A delay
        # gives the regulator time to work, the same slack a human
        # gets between moves. Default 0 keeps the production cadence.
        self._step_delay = max(0.0, float(step_delay_s))

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
        if (reasons := self._check_readiness()) is not None:
            result.state = "UNWELL"
            result.reasons = reasons
            return result

        result.prior_competence = self._competence_snapshot()
        scorecard = self._arcade.create_scorecard()
        result.scorecard_id = scorecard
        try:
            return self._play(result, max_steps)
        finally:
            try:
                closed = self._arcade.close_scorecard(scorecard)
                result.score = getattr(closed, "score", None)
            except Exception:  # noqa: BLE001
                result.reasons += ("scorecard closure failed",)
                result.state = "SCORECARD_ERROR"

    def play_all(self, max_steps: int = 200) -> ArcEvaluation:
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
        if (reasons := self._check_readiness()) is not None:
            evaluation.games[0].state = "UNWELL"
            evaluation.games[0].reasons = reasons
            return evaluation
        self._suite_started = True
        scorecard = self._arcade.create_scorecard()
        evaluation.scorecard_id = scorecard
        try:
            for result in evaluation.games:
                result.scorecard_id = scorecard
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                result.state = "NOT_FINISHED"
                result.prior_competence = self._competence_snapshot()
                self._play(result, max_steps)
                if result.state == "UNWELL":
                    break
        finally:
            closed = self._arcade.close_scorecard(scorecard)
            if self.operation_mode == "competition" or all(r.resets for r in evaluation.games):
                evaluation.score = getattr(closed, "score", None)
        return evaluation

    def _play(self, result: ArcResult, max_steps: int) -> ArcResult:
        from arcengine import GameAction, GameState

        game_id = result.game_id
        env = self._arcade.make(
            game_id,
            scorecard_id=result.scorecard_id,
            save_recording=self._record,
        )
        if env is None:
            result.state = "UNAVAILABLE"
            return result
        frame = getattr(env, "observation_space", None)
        if frame is None:
            frame = env.reset()
        result.resets = 1
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
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                result.steps += 1
                result.resets += 1
                frame = env.reset()
                continue
            if result.steps >= max_steps:
                result.state = "BUDGET"
                break
            if result.steps and result.steps % _HEALTH_EVERY_STEPS == 0:
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
            if not frame.available_actions:
                result.state = "NO_ACTIONS"
                break
            chosen = self._choose_action(frame, game_id)
            if chosen is None:
                result.state = "NO_ACTION"
                break
            action, data, reasoning = chosen
            result.steps += 1
            if action is GameAction.RESET:
                result.resets += 1
            source = str((reasoning or {}).get("source", "unknown"))
            result.decision_sources[source] = (
                result.decision_sources.get(source, 0) + 1
            )
            try:
                step_signature.bind(action, data=data, reasoning=reasoning)
            except TypeError:  # older wrappers without the field
                frame = env.step(action, data=data)
            else:
                frame = env.step(action, data=data, reasoning=reasoning)
            delay = getattr(self, "_step_delay", 0.0)
            if delay > 0.0:
                time.sleep(delay)
        if frame is None:
            result.state = "NO_OBSERVATION"
        result.prior_learning = getattr(
            self._mind, "_world_learning_loaded", "none"
        )
        return result

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
        desc = self._mind.world_step(
            frame.frame[-1],
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
                        if GameAction.from_id(int(a)).action_type
                        is ComplexAction
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


class ArcCompetition(ArcPractice):
    operation_mode = "competition"

    def play(self, game_id: str, max_steps: int = 200) -> ArcResult:
        raise RuntimeError("Competition requires play_all(), not selected games")
