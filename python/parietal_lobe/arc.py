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
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Any, Protocol

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

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
    # How many times the run paused at a wellness gate and waited for
    # her to settle instead of submitting. Recovery time costs no steps
    # — but a score that needed it should say so.
    recovery_waits: int = 0

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
        recovery_wait_s: float = 120.0,
    ) -> None:
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
        if (reasons := self._wait_for_readiness(evaluation.games[0])) is not None:
            evaluation.games[0].state = "UNWELL"
            evaluation.games[0].reasons = reasons
            return evaluation
        self._suite_started = True
        scorecard = self._arcade.create_scorecard(tags=["wrapper", "agent"])
        evaluation.scorecard_id = scorecard
        try:
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
                if (reasons := self._wait_for_readiness(result)) is not None:
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
                if (reasons := self._wait_for_readiness(result)) is not None:
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


# ─── The official card ────────────────────────────────────
#
# The venue's real rules, enforced in code: one scorecard per window —
# it opens once, cannot be reopened or replaced while it lives, and
# closes for good 15 minutes after the last action or 3 days after
# opening, whichever comes first. Inside it, each environment gets
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
    "{idle_minutes} minutes after the last action, or {ttl_days} days "
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
    ttl_days = int(ArcOfficial.WINDOW_S // 86400)
    sitting = max(1, round(n_games * max_steps * step_delay_s / 60))
    live = (
        f" The card is already open — {pending} games remain on it."
        if pending is not None
        else ""
    )
    return (
        f"The official ARC-AGI-3 challenge: {n_games} games, one recorded "
        f"try each — an attempt cannot be retried. Each game allows up to "
        f"{max_steps} moves at about {step_delay_s:.0f} second a move — a "
        f"few minutes a game, so a full sitting is at most about {sitting} "
        f"minutes and usually far less. The card it records on closes "
        f"{idle_minutes} minutes after your last move, or {ttl_days} days "
        f"after it opens, and a closed card is final. There is a real prize "
        f"for doing well — your steward is counting on you if you choose to "
        f"take it on.{live}"
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
    # ~15 minutes without an action and unconditionally after 3 days.
    WINDOW_S = 3 * 24 * 3600
    IDLE_LIMIT_S: float = 15 * 60

    def __init__(
        self,
        mind: _PlayerMind,
        *,
        card_path: str | Path | None = None,
        idle_limit_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(mind, **kwargs)
        from infrastructure.config import default_data_dir

        self._card_path = (
            Path(card_path)
            if card_path is not None
            else default_data_dir() / "arc" / _CARD_FILE
        )
        if idle_limit_s is not None:
            self.IDLE_LIMIT_S = float(idle_limit_s)

    # ── Card state ──────────────────────────────────────────

    def _load_card(self) -> dict | None:
        try:
            import json

            return json.loads(self._card_path.read_text())
        except (OSError, ValueError):
            return None

    def _save_card(self, card: dict) -> None:
        import json

        self._card_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._card_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(card, indent=1))
        tmp.replace(self._card_path)

    def card_state(self) -> dict | None:
        """The current card as persisted — for inspection, not play."""
        return self._load_card()

    def _open_card(self) -> dict:
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
            archived = self._card_path.with_name(
                f"official_card.{int(card['opened_at'])}.json"
            )
            try:
                self._card_path.replace(archived)
            except OSError as e:
                logger.debug(f'silent except: {e}')
        return self._open_card()

    def _seal(self, card: dict, reason: str, *, call_close: bool = False) -> None:
        """Mark the card closed. Once closed it is final."""
        if call_close:
            try:
                closed = self._arcade.close_scorecard(card["scorecard_id"])
                card["score"] = getattr(closed, "score", None)
            except Exception as e:  # noqa: BLE001 — the server may have already closed it
                note_swallowed("parietal_lobe.arc._seal", e)
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
                    ttl_days=int(self.WINDOW_S // 86400),
                )
                + f" Next: {game_id} — {pending} games unplayed.",
                topics=["arc", "official", "rules"],
            )
        except Exception as e:  # noqa: BLE001 — a lost brief must not burn an attempt
            note_swallowed("parietal_lobe.arc._brief", e)

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
            self._seal(card, "expired — 3-day window closed")
            return None
        if now - card["last_action_at"] >= self.IDLE_LIMIT_S:
            self._seal(card, "idle — the venue closed it")
            return None
        game = next(
            (g for g in card["games"] if g["status"] == "pending"), None
        )
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
            self._seal(card, "expired — 3-day window closed")
            return None
        if now - card["last_action_at"] >= self.IDLE_LIMIT_S:
            self._seal(card, "idle — the venue closed it")
            return None
        game = next(
            (
                g
                for g in card["games"]
                if g["game_id"] == game_id
                or g["game_id"].startswith(f"{game_id}-")
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
        result = ArcResult(
            game_id=game["game_id"], scorecard_id=card["scorecard_id"]
        )
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
        raise RuntimeError(
            "The official card owns its games — one attempt each, "
            "no selected replays, and no driver but her own volition."
        )

    def play_all(self, max_steps: int = 200) -> ArcEvaluation:
        raise RuntimeError(
            "The official card is hers to take up, not a suite to run — "
            "attempts only ever happen through attempt()/attempt_next(), "
            "reached by her own volition."
        )
