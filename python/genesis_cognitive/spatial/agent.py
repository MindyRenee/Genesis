"""Interactive agent — acting in unseen grid environments.

The static solver answers "what rule maps input to output?" The
interactive problem asks something harder: *act* in an unseen
environment, observe the consequences, and find the goal — with no
instructions. That is a perception→action loop, which is exactly the
shape of Genesis's architecture.

This module provides the policy layer. The environment side (frames,
actions, win states) is deliberately kept generic so the policy can
be tested against hand-built mock environments.

Policy v2 — agency + navigation:

    1. Agency detection: after each action, diff consecutive frames.
       The object whose cells consistently move when it acts is
       *its avatar* — self-recognition in a visual field, learned
       from contingency rather than told.
    2. Action-effect model: each action gets a learned displacement
       vector (dr, dc) estimated from the avatar's centroid shift.
       This is an affordance model: "what does this button DO to me?"
    3. Goal navigation: once agency and effects are known, perceive
       the scene, pick a goal object (nearest non-avatar object —
       unexplored affordances first), and choose the action whose
       displacement best closes the distance.
    4. Novelty fallback: before the model is learned, and with
       probability epsilon, explore — try untried actions.
    5. Task competence: when a shared ``TaskCompetence`` is attached,
       the environment is recognized as a task schema, every step's
       transition feeds the shared affordance model, and a won
       episode's control map consolidates as a reusable skill that
       primes navigation on structurally similar environments.

On WIN/GAME_OVER the episode resets, but the learned model (avatar
color, action displacements, object memory) persists — it gets to
keep what it figured out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..reasoning import (
    GoalCondition,
    ProcedureStep,
    TaskCompetence,
    TaskContext,
)
from .grid import Grid
from .scene import perceive


@dataclass
class ActionStats:
    """What the agent has learned about an action's effect."""

    attempts: int = 0
    total_change: float = 0.0  # summed fraction of cells changed
    wins: int = 0
    dr: float = 0.0  # learned mean row displacement of the avatar
    dc: float = 0.0  # learned mean column displacement

    @property
    def mean_change(self) -> float:
        return self.total_change / self.attempts if self.attempts else 0.0


# A track binds to the same object across frames if its centroid stays
# within this many cells. Movers in these environments travel ~1 cell
# per step; 3.5 tolerates faster movers without letting teleporting
# spawns fake a trajectory.
_TRACK_MAX_SHIFT = 3.5


@dataclass
class _ObjectTrack:
    """A persistent object identity bound across consecutive frames.

    Colors are bindings; a track is the object itself. Evidence about
    autonomous motion accrues on the track, so two same-colored
    objects can't conflate, and a newly spawned object can't inherit
    another's motion history.
    """

    track_id: int
    color: int
    centroid: tuple[float, float]
    cells: frozenset[tuple[int, int]]
    velocity: tuple[float, float] = (0.0, 0.0)
    moved_frames: int = 0  # frames where it displaced on its own


@dataclass
class EpisodeResult:
    """Outcome of one episode (one attempt at a level)."""

    steps: int
    state: str  # "WIN", "GAME_OVER", or "BUDGET_EXHAUSTED"
    levels_completed: int
    actions_tried: dict[str, ActionStats] = field(default_factory=dict)


def frame_to_grid(frame: Any) -> Grid:
    """Convert an environment frame to a Grid.

    Frames arrive as a *list of layers* (each a 2-D array); the final
    layer is the composed scene. A bare 2-D array is accepted too.
    """
    if isinstance(frame, (list, tuple)) and frame and hasattr(frame[0], "shape"):
        frame = frame[-1]
    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    return Grid.from_lists([list(r) for r in rows])


def frame_diff(a: Grid, b: Grid) -> float:
    """Fraction of cells that differ between two frames."""
    if a.shape != b.shape:
        return 1.0
    total = a.width * a.height
    changed = sum(1 for r, c, v in a.iter_cells() if v != b.at(r, c))
    return changed / total


def _color_centroid(grid: Grid, color: int) -> tuple[float, float] | None:
    cells = grid.cells_with(color)
    if not cells:
        return None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return sum(rs) / len(rs), sum(cs) / len(cs)


class SpatialAgent:
    """An agency-detecting, navigating agent over the spatial layer.

    Args:
        seed: RNG seed for reproducible runs.
        epsilon: probability of exploratory (non-navigational) action.
        task_competence: shared task/skill substrate. When provided,
            environments are recognized as task schemas, step
            transitions feed the shared affordance model, and won
            control maps become skills that transfer to structurally
            similar environments. A private instance is created when
            none is given.
    """

    def __init__(
        self,
        seed: int = 42,
        epsilon: float = 0.3,
        task_competence: TaskCompetence | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.epsilon = epsilon
        self.task_competence = task_competence or TaskCompetence()
        # The recognized schema for the current environment/level,
        # plus skill priors adopted from it. Priors only fill gaps —
        # real observations always override transferred knowledge.
        self._task_context: TaskContext | None = None
        self._skill_priors: dict[str, tuple[float, float]] = {}
        self._role_priors: set[str] = set()
        self._episode_skill_ids: set[str] = set()
        self.stats: dict[Any, ActionStats] = {}
        self._last_frame: Grid | None = None
        self._last_action: Any = None
        # Learned self-model — persists across episodes.
        self.avatar_color: int | None = None
        self._avatar_votes: dict[int, int] = {}
        # Learned hazard model: colors that move on their own (not
        # caused by its action). Moving things that aren't its are
        # almost always things to avoid, not goals — collectibles and
        # exits are static in these environments.
        self._hazard_colors: set[int] = set()
        # Persistent object identities for this episode — mover
        # evidence accrues on tracks, not on colors.
        self._tracks: dict[int, _ObjectTrack] = {}
        self._next_track_id = 0
        # Schema-level prior: the mover speed skills of this family
        # carried. One sighting of matching motion then suffices.
        self._prior_hazard_speed: float | None = None
        # Per-hazard velocity: last observed (dr, dc) centroid shift.
        # Movers in these environments are usually ballistic (fixed
        # direction per step), so one-step extrapolation predicts
        # where they'll be when its move lands.
        self._hazard_velocity: dict[int, tuple[float, float]] = {}
        # Reward inference: colors whose objects vanished while the
        # avatar was touching them are pickups — consumable, valuable.
        # Learned empirically from consequences, not told.
        self._valuable_colors: set[int] = set()
        # Stations: objects that change the *avatar's* appearance when
        # touched (key/color/rotation stations in gated games).
        self._station_colors: set[int] = set()
        # Gated cells: goal positions it has reached but couldn't
        # collect — the object persisted. They stay locked until its
        # key state changes.
        self._gated_cells: set[tuple[int, int]] = set()
        self._touch_fail: dict[tuple[int, int], int] = {}
        self._avatar_sig: tuple[tuple[int, ...], ...] | None = None
        # Lethal colors: contact caused a respawn. Strongest evidence
        # there is — consumption signals can never un-mark these.
        self._lethal_colors: set[int] = set()
        # Life-loss across a hidden frame: the avatar sprite goes
        # invisible on the death frame, then reappears at spawn.
        self._missing_avatar_at: tuple[float, float] | None = None
        # Gate affordance memory: object pattern -> avatar signature
        # that opened it (or signatures that failed). Learned from
        # touch outcomes; persists across episodes.
        self._gate_openers: dict[
            tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]
        ] = {}
        self._gate_blocked: dict[
            tuple[tuple[int, ...], ...],
            set[tuple[tuple[int, ...], ...]],
        ] = {}
        # Station effect memory: station color -> which signature
        # dimension it perturbs ("recolor" | "reshape" | "reorient").
        self._station_effects: dict[int, str] = {}
        # Objects already interacted with (cells the avatar has reached).
        self._visited_goals: set[tuple[int, int]] = set()
        # Interaction novelty (the coach's hint): colors it has never
        # touched are worth touching — consequences can't be learned
        # without contact. This is a prior about *how to explore*,
        # not an answer.
        self._touched_colors: set[int] = set()
        # Encouragement / grit: after losing a life it gets a short
        # window of raised exploration — it shakes it off and tries
        # something different instead of repeating the fatal path.
        self._grit_steps: int = 0
        # Click memory: cell -> observed frame change. A cell that
        # already produced ~nothing is a dead spot — clicking it again
        # can never help, so complex actions target fresh cells.
        self._click_outcomes: dict[tuple[int, int], float] = {}
        self._pending_click: tuple[int, int] | None = None

    # ── Learning ──────────────────────────────────────────────

    def _stat(self, action: Any) -> ActionStats:
        return self.stats.setdefault(action, ActionStats())

    # ── Task competence ───────────────────────────────────────

    @staticmethod
    def _count_bucket(n: int) -> str:
        if n <= 0:
            return "none"
        if n <= 3:
            return "few"
        if n <= 8:
            return "some"
        return "many"

    @staticmethod
    def _grid_bucket(grid: Grid) -> str:
        cells = grid.width * grid.height
        if cells <= 64:
            return "tiny"
        if cells <= 400:
            return "small"
        if cells <= 4096:
            return "medium"
        return "large"

    def _env_features(self, grid: Grid) -> dict[str, Any]:
        """Environment-observable structure for schema matching.

        Only properties of the world itself belong in the signature —
        agent-model facts (avatar color, discovered hazards) are
        bindings that vary across episodes of the same task.
        """
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        return {
            "env.interactive": True,
            "grid.size": self._grid_bucket(grid),
            "colors.count": self._count_bucket(len(grid.colors())),
            "objects.count": self._count_bucket(len(scene.objects)),
        }

    def _recognize_env(self, action_space: list[Any]) -> TaskContext | None:
        """Bind the current frame to a task schema once per episode."""
        if self._task_context is not None or self._last_frame is None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="spatial.interactive",
            state=self._env_features(self._last_frame),
            actions=(str(a) for a in action_space),
            goal_conditions=[GoalCondition("episode.state", "eq", "WIN")],
            entities=("avatar", "object", "hazard", "pickup"),
            # Schema-level vocabulary: "navigate a self among objects
            # toward a terminal state" describes the family, not the
            # domain — a maze, a tile-world, or an arcade env share it.
            roles=("self", "object", "navigate"),
        )
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed navigation priors from skills won on similar envs.

        The transferable procedure for an interactive environment is
        the control map — which actions displace the avatar and by how
        much. Priors fill gaps only: once an action has real attempts
        its observed effect overrides the transferred value.
        """
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                if step.family == "control":
                    dr = step.parameters.get("dr")
                    dc = step.parameters.get("dc")
                    if isinstance(dr, int | float) and isinstance(
                        dc, int | float
                    ):
                        adopted = True
                        self._skill_priors.setdefault(
                            step.action, (float(dr), float(dc))
                        )
                elif step.family == "role":
                    # Object roles transfer as priors, not facts:
                    # the color is an instance binding, but "envs of
                    # this kind contain pickups/hazards" is schema-
                    # level knowledge. A hazard's speed is schema-
                    # level too — it says how fast threats move here.
                    role = step.parameters.get("role")
                    if isinstance(role, str):
                        adopted = True
                        self._role_priors.add(role)
                        if role == "hazard":
                            speed = step.parameters.get("speed")
                            if isinstance(speed, int | float):
                                self._prior_hazard_speed = max(
                                    self._prior_hazard_speed or 0.0,
                                    float(speed),
                                )
            if adopted:
                self._episode_skill_ids.add(match.skill.skill_id)

    def _model_displacement(
        self, action: Any, grid: Grid
    ) -> tuple[float, float] | None:
        """Predicted avatar displacement from the shared schema model.

        The schema's operator models accumulate every observed
        transition across episodes and structurally similar
        environments — including episodes that never produced a
        skill. This is the model-based lookahead: "what would this
        action do to my position?" answered by competence, not only
        by this env's local statistics.
        """
        if self._task_context is None:
            return None
        dr = dc = 0.0
        seen = False
        for p in self.task_competence.predict(
            self._task_context, str(action), self._transition_state(grid)
        ):
            if p.confidence < 0.2 or p.delta is None:
                continue
            if p.key == "avatar.r":
                dr, seen = p.delta, True
            elif p.key == "avatar.c":
                dc, seen = p.delta, True
        return (dr, dc) if seen else None

    def _transition_state(self, grid: Grid) -> dict[str, Any]:
        """Flat state vector for the shared transition learner."""
        bg = self._background(grid)
        scene = perceive(grid, background=bg, compute_relations=False)
        pos = self._avatar_pos(grid)
        return {
            "avatar.r": pos[0] if pos is not None else None,
            "avatar.c": pos[1] if pos is not None else None,
            "objects.count": len(scene.objects),
            "cells.nonempty": sum(
                1 for _, _, v in grid.iter_cells() if v != bg
            ),
        }

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: control map plus object roles.

        Control steps carry the action→displacement model. Role steps
        carry discovered object semantics — which kinds of things are
        consumable, dangerous, or transformative. The color is kept as
        the instance binding; what transfers is the role's existence.
        """
        steps = []
        for action in sorted(self.stats, key=str):
            st = self.stats[action]
            if st.attempts == 0:
                continue
            dr, dc = st.dr / st.attempts, st.dc / st.attempts
            if dr == 0.0 and dc == 0.0:
                continue
            steps.append(
                ProcedureStep(
                    action=str(action),
                    family="control",
                    parameters={"dr": round(dr, 3), "dc": round(dc, 3)},
                    description=(
                        f"{action} displaces avatar ({dr:+.2f},{dc:+.2f})"
                    ),
                )
            )
        for role, colors in (
            ("pickup", self._valuable_colors),
            ("hazard", self._hazard_colors),
            ("station", self._station_colors),
        ):
            for color in sorted(colors):
                params: dict[str, Any] = {"role": role, "color": color}
                # Hazard steps also carry the family's mover speed —
                # dynamics are schema-level even though color isn't.
                if role == "hazard":
                    v = self._hazard_velocity.get(color)
                    if v is not None:
                        params["speed"] = round(abs(v[0]) + abs(v[1]), 2)
                steps.append(
                    ProcedureStep(
                        action="touch",
                        family="role",
                        parameters=params,
                        description=f"{role} object (color {color})",
                    )
                )
        return steps

    def _track_objects(
        self, prev: Grid, cur: Grid
    ) -> dict[int, tuple[float, float]]:
        """Bind this frame's objects to persistent tracks.

        Returns ``{track_id: (dr, dc)}`` for tracks that genuinely
        moved. An object whose cell count collapsed keeps its identity
        (it's still *that* object) but its centroid shift is
        consumption, not motion — it earns no mover vote.
        """
        if not self._tracks:
            # First transition of the episode: seed tracks from the
            # previous frame so its motion is visible immediately.
            for obj in perceive(
                prev,
                background=self._background(prev),
                compute_relations=False,
            ).objects:
                self._tracks[self._next_track_id] = _ObjectTrack(
                    self._next_track_id, obj.color, obj.centroid, obj.cells
                )
                self._next_track_id += 1

        cur_objs = perceive(
            cur, background=self._background(cur), compute_relations=False
        ).objects
        pairs: list[tuple[float, int, int]] = []
        for obj in cur_objs:
            for tid, track in self._tracks.items():
                if track.color != obj.color:
                    continue
                d = (
                    abs(track.centroid[0] - obj.centroid[0])
                    + abs(track.centroid[1] - obj.centroid[1])
                )
                if d <= _TRACK_MAX_SHIFT:
                    pairs.append((d, tid, obj.index))
        pairs.sort()

        used_tracks: set[int] = set()
        used_objs: set[int] = set()
        moved: dict[int, tuple[float, float]] = {}
        obj_by_index = {o.index: o for o in cur_objs}
        for _d, tid, oidx in pairs:
            if tid in used_tracks or oidx in used_objs:
                continue
            used_tracks.add(tid)
            used_objs.add(oidx)
            obj = obj_by_index[oidx]
            track = self._tracks[tid]
            size_change = abs(len(obj.cells) - len(track.cells))
            if size_change < max(2, len(track.cells) * 0.25):
                dr = obj.centroid[0] - track.centroid[0]
                dc = obj.centroid[1] - track.centroid[1]
                track.velocity = (dr, dc)
                if dr or dc:
                    moved[tid] = (dr, dc)
            else:
                track.velocity = (0.0, 0.0)
            track.cells = obj.cells
            track.centroid = obj.centroid

        for obj in cur_objs:
            if obj.index in used_objs:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = _ObjectTrack(
                tid, obj.color, obj.centroid, obj.cells
            )
            used_tracks.add(tid)
        # An unmatched track is gone — destroyed or consumed. Identity
        # does not outlive the object's absence from the scene.
        self._tracks = {
            tid: t for tid, t in self._tracks.items() if tid in used_tracks
        }
        return moved

    def _learn_from_step(
        self, prev: Grid, cur: Grid, action: Any
    ) -> float:
        """Update the affordance/self model from one transition."""
        st = self._stat(action)
        st.attempts += 1
        change = frame_diff(prev, cur)
        st.total_change += change
        if change == 0.0:
            return 0.0

        moved = self._track_objects(prev, cur)

        # Agency detection: the track that displaced most is the
        # avatar candidate — the thing its action controls, detected
        # from contingency, not told.
        best_tid, best_shift = None, 0.0
        best_dr = best_dc = 0.0
        for tid, (dr, dc) in moved.items():
            shift = abs(dr) + abs(dc)
            if shift > best_shift:
                best_tid, best_shift = tid, shift
                best_dr, best_dc = dr, dc
        # Hazard learning: tracks that moved but aren't the avatar
        # candidate — things that move on their own are avoided as
        # goals. Votes accumulate per object so one noisy frame can't
        # mark a color forever, and a freshly spawned lookalike can't
        # inherit another object's motion record.
        for tid, (dr, dc) in moved.items():
            if tid == best_tid:
                continue
            track = self._tracks[tid]
            self._hazard_velocity[track.color] = track.velocity
            track.moved_frames += 1
            # A transferred "hazard" role prior lowers the evidence
            # bar: envs of this kind contain self-moving threats. If
            # the prior also carried the family's mover speed and this
            # object's motion matches it, one sighting suffices.
            threshold = 2 if "hazard" in self._role_priors else 3
            speed = abs(dr) + abs(dc)
            if (
                self._prior_hazard_speed is not None
                and abs(speed - self._prior_hazard_speed) <= 0.5
            ):
                threshold = 1
            if track.moved_frames >= threshold:
                self._hazard_colors.add(track.color)
        if best_tid is not None and best_shift > 0:
            # Attribute the winning displacement to this action.
            st.dr += best_dr
            st.dc += best_dc
            best_color = self._tracks[best_tid].color
            self._avatar_votes[best_color] = (
                self._avatar_votes.get(best_color, 0) + 1
            )
            self.avatar_color = max(
                self._avatar_votes,
                key=lambda c: self._avatar_votes[c],
            )
        return change

    def observe(self, frame: Any) -> float:
        """Record a frame; learn from the transition; return change."""
        grid = frame_to_grid(frame)
        change = 0.0
        if self._last_frame is not None and self._last_action is not None:
            prev = self._last_frame
            change = self._learn_from_step(
                prev, grid, self._last_action
            )
            if self._task_context is not None:
                # The shared substrate learns the same affordance data
                # in a domain-general, persistable form. Its check is
                # also a surprise signal: when the world defies an
                # action's learned effects, renew curiosity instead of
                # repeating a now-suspect policy.
                check = self.task_competence.record_transition(
                    self._task_context,
                    str(self._last_action),
                    self._transition_state(prev),
                    self._transition_state(grid),
                )
                if check.predicted and check.score < 0.6:
                    self._grit_steps = max(self._grit_steps, 20)
        if self._pending_click is not None:
            self._click_outcomes[self._pending_click] = change
            self._pending_click = None
        if self._last_frame is not None and self._last_action is not None:
            self._detect_consumption(self._last_frame, grid)
            self._detect_gates_and_stations(self._last_frame, grid)
        self._last_frame = grid
        return change

    def _detect_consumption(self, prev: Grid, cur: Grid) -> None:
        """Learn which colors are pickups: objects that vanish on touch.

        Compares the previous and current frames: an object whose cells
        were within reach of the avatar and are now gone (or mostly
        erased) was consumed by its action — a collectible, a step
        refill, a key. That color becomes a preferred goal. If it was
        mislabeled a hazard (its disappearance moved the centroid),
        that gets corrected too.
        """
        pos = self._avatar_pos(cur)
        if pos is None:
            pos = self._avatar_pos(prev)
        if pos is None:
            return
        bg = self._background(cur)
        scene = perceive(
            prev, background=self._background(prev),
            compute_relations=False,
        )
        for obj in scene.objects:
            if obj.color in (0, bg, self.avatar_color):
                continue
            if obj.color in self._lethal_colors:
                continue
            near = min(
                abs(pos[0] - r) + abs(pos[1] - c) for r, c in obj.cells
            )
            if near > 2:
                continue
            if near <= 1:
                self._touched_colors.add(obj.color)
            remaining = sum(
                1 for r, c in obj.cells if cur.at(r, c) == obj.color
            )
            if remaining <= len(obj.cells) // 2:
                self._valuable_colors.add(obj.color)
                self._hazard_colors.discard(obj.color)
                for track in self._tracks.values():
                    if track.color == obj.color:
                        track.moved_frames = 0
                self._hazard_velocity.pop(obj.color, None)
                # A gate just opened — bind its appearance to the
                # avatar signature that opened it.
                sig = self._avatar_signature(cur)
                if sig is not None:
                    self._gate_openers[
                        self._obj_signature(obj, prev)
                    ] = sig

    def _avatar_signature(
        self, grid: Grid
    ) -> tuple[tuple[int, ...], ...] | None:
        """The avatar's key state: its bounding-box pixel pattern.

        Captures color, shape, and rotation — a station that cycles
        any of the three changes this signature.
        """
        if self.avatar_color is None:
            return None
        cells = grid.cells_with(self.avatar_color)
        if not cells:
            return None
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        return tuple(
            tuple(grid.at(r, c) for c in range(c0, c1 + 1))
            for r in range(r0, r1 + 1)
        )

    def _detect_gates_and_stations(self, prev: Grid, cur: Grid) -> None:
        """Learn gate and station semantics from touch consequences.

        - Gate: the avatar reached an object's cells but it didn't
          vanish — it's locked until its key state changes. Two
          failed touches marks the cell gated.
        - Station: an object the avatar was near when its own color
          signature changed — it transformed its (key stations). When
          gated goals exist, stations become priority targets.
        """
        cur_pos = self._avatar_pos(cur)
        prev_pos = self._avatar_pos(prev)
        if cur_pos is None:
            # Avatar hidden — remember where it was last seen; if it
            # reappears elsewhere, a life was lost on this spot.
            if prev_pos is not None:
                self._missing_avatar_at = prev_pos
            return
        if self._missing_avatar_at is not None:
            died_at = self._missing_avatar_at
            self._missing_avatar_at = None
            if (
                abs(cur_pos[0] - died_at[0]) + abs(cur_pos[1] - died_at[1])
                > 4
            ):
                self._mark_lethal(died_at, prev)
        pos = cur_pos or prev_pos
        if pos is None:
            return
        # Life-loss detection: the avatar teleported more than one
        # step (respawn), meaning whatever it was touching killed it.
        if prev_pos is not None and pos is not None:
            jump = abs(pos[0] - prev_pos[0]) + abs(pos[1] - prev_pos[1])
            if jump > 4:
                self._mark_lethal(prev_pos, prev)
        sig = self._avatar_signature(cur)
        scene = perceive(
            prev, background=self._background(prev),
            compute_relations=False,
        )
        for obj in scene.objects:
            bg = self._background(cur)
            if obj.color in (0, bg, self.avatar_color):
                continue
            if obj.color in self._lethal_colors:
                continue
            near = min(
                abs(pos[0] - r) + abs(pos[1] - c) for r, c in obj.cells
            )
            if near > 1:
                continue
            self._touched_colors.add(obj.color)
            remaining = sum(
                1 for r, c in obj.cells if cur.at(r, c) == obj.color
            )
            if remaining > len(obj.cells) // 2:
                # Touched but persisted — a gate if it stays that way.
                for cell in obj.cells:
                    self._touch_fail[cell] = (
                        self._touch_fail.get(cell, 0) + 1
                    )
                    if self._touch_fail[cell] >= 2:
                        self._gated_cells.add(cell)
                # Record which avatar signature this gate rejected.
                if self._avatar_sig is not None:
                    self._gate_blocked.setdefault(
                        self._obj_signature(obj, prev), set()
                    ).add(self._avatar_sig)
            # Station: avatar's own signature changed while adjacent.
            if (
                self._avatar_sig is not None
                and sig is not None
                and sig != self._avatar_sig
            ):
                self._station_colors.add(obj.color)
                self._station_effects[obj.color] = self._classify_change(
                    self._avatar_sig, sig
                )
        if (
            self._avatar_sig is not None
            and sig is not None
            and sig != self._avatar_sig
            and self._gated_cells
        ):
            # Key state changed — previously locked gates may now
            # open. Clear the lockout so they get retried.
            self._gated_cells.clear()
            self._touch_fail.clear()
        self._avatar_sig = sig

    @staticmethod
    def _obj_signature(
        obj: Any, grid: Grid
    ) -> tuple[tuple[int, ...], ...]:
        """An object's bounding-box pixel pattern — its appearance."""
        r0 = min(r for r, _ in obj.cells)
        r1 = max(r for r, _ in obj.cells)
        c0 = min(c for _, c in obj.cells)
        c1 = max(c for _, c in obj.cells)
        return tuple(
            tuple(grid.at(r, c) for c in range(c0, c1 + 1))
            for r in range(r0, r1 + 1)
        )

    @staticmethod
    def _classify_change(
        before: tuple[tuple[int, ...], ...],
        after: tuple[tuple[int, ...], ...],
    ) -> str:
        """Which signature dimension did a station perturb?"""
        import collections

        flat_b = collections.Counter(v for row in before for v in row)
        flat_a = collections.Counter(v for row in after for v in row)
        if flat_a != flat_b:
            return "recolor"
        if len(before) != len(after) or len(before[0]) != len(after[0]):
            return "reshape"
        return "reorient"

    def _mark_lethal(self, at: tuple[float, float], grid: Grid) -> None:
        """Blame whatever was adjacent to a death spot."""
        for obj in perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        ).objects:
            if obj.color == self.avatar_color:
                continue
            if min(
                abs(at[0] - r) + abs(at[1] - c) for r, c in obj.cells
            ) <= 1:
                self._hazard_colors.add(obj.color)
                self._lethal_colors.add(obj.color)
                self._valuable_colors.discard(obj.color)
                self._station_colors.discard(obj.color)
        # Encouragement: losing a life isn't the end — a short burst
        # of curiosity follows so it tries a different path.
        self._grit_steps = 40

    # ── Goals ─────────────────────────────────────────────────

    def _avatar_pos(self, grid: Grid) -> tuple[float, float] | None:
        if self.avatar_color is None:
            return None
        return _color_centroid(grid, self.avatar_color)

    @staticmethod
    def _background(grid: Grid) -> int:
        """The frame's background color — the modal cell value.

        Environments don't promise a black background; the floor can
        itself be colorful. The most frequent color is the floor in
        practice — treating it as foreground drowns object detection.
        """
        from collections import Counter

        counts = Counter(v for _, _, v in grid.iter_cells())
        return counts.most_common(1)[0][0]

    def _pick_goal(self, grid: Grid) -> tuple[float, float] | None:
        """Nearest non-avatar object the agent hasn't reached yet."""
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        pos = self._avatar_pos(grid)
        cur_sig = self._avatar_signature(grid)

        def openable_now(obj: Any) -> bool:
            """This gate's recorded opener matches my current key."""
            if cur_sig is None:
                return False
            return (
                self._gate_openers.get(self._obj_signature(obj, grid))
                == cur_sig
            )

        goals = [
            obj
            for obj in scene.objects
            if obj.color != self.avatar_color
            and obj.color not in self._hazard_colors
            and (
                not any(cell in self._gated_cells for cell in obj.cells)
                or openable_now(obj)
            )
        ]
        if not goals:
            return None
        if pos is None:
            return goals[0].centroid
        # Highest priority: gates empirically known to open for the
        # avatar's current key state.
        openable = [g for g in goals if openable_now(g)]
        # Known pickups next — objects of colors empirically observed
        # to be consumable (they vanished on touch). Survival-critical
        # in games with step budgets.
        valuable = [g for g in goals if g.color in self._valuable_colors]
        # When gated goals exist, stations that change its key state
        # are the way forward — visit them before re-trying gates.
        stations = (
            [g for g in goals if g.color in self._station_colors]
            if self._gated_cells
            else []
        )
        # The hint: object colors never touched can't teach it
        # anything until contact — prefer them when nothing is
        # proven-openable or known-valuable.
        untouched = [
            g for g in goals if g.color not in self._touched_colors
        ]
        pool = openable or valuable or stations or untouched or goals
        # Prefer unvisited goals; fall back to nearest of any.
        unvisited = [
            g
            for g in pool
            if (round(g.centroid[0]), round(g.centroid[1]))
            not in self._visited_goals
        ]
        pool = unvisited or pool
        return min(
            pool,
            key=lambda g: abs(g.centroid[0] - pos[0])
            + abs(g.centroid[1] - pos[1]),
        ).centroid

    def _navigate(
        self, grid: Grid, action_space: list[Any]
    ) -> Any | None:
        """Choose the action whose learned effect closes distance to goal.

        Costs add a hazard penalty: landing within ~2 cells of a known
        moving-hazard color is expensive. It doesn't just avoid making
        hazards its destination — it avoids walking through them.
        """
        pos = self._avatar_pos(grid)
        goal = self._pick_goal(grid)
        if pos is None or goal is None:
            return None
        danger: list[tuple[float, float]] = []
        if self._hazard_colors:
            scene = perceive(
                grid, background=self._background(grid),
                compute_relations=False,
            )
            for obj in scene.objects:
                if obj.color in self._hazard_colors:
                    danger.extend(obj.cells)
                    v = self._hazard_velocity.get(obj.color)
                    if v is not None:
                        # Predict where the hazard will be next step —
                        # danger is its future footprint, not just now.
                        danger.extend(
                            (r + v[0], c + v[1]) for r, c in obj.cells
                        )
                    # Hazards can also spread (flood/tide mechanics):
                    # cells adjacent to a hazard cell are unsafe too.
                    danger.extend(
                        (r + dr, c + dc)
                        for r, c in obj.cells
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
        best, best_cost = None, float("inf")
        for action in action_space:
            st = self.stats.get(action)
            if st is not None and st.attempts > 0:
                step_dr = st.dr / st.attempts
                step_dc = st.dc / st.attempts
            else:
                # No local evidence — ask the schema's transition
                # model first (it carries every observed step across
                # similar envs), then verified-skill priors.
                disp = self._model_displacement(action, grid)
                if disp is not None:
                    step_dr, step_dc = disp
                else:
                    prior = self._skill_priors.get(str(action))
                    if prior is None:
                        continue  # untried actions belong to exploration
                    step_dr, step_dc = prior
            if step_dr == 0 and step_dc == 0:
                continue  # this action doesn't move the avatar
            new_r, new_c = pos[0] + step_dr, pos[1] + step_dc
            remaining = abs(goal[0] - new_r) + abs(goal[1] - new_c)
            if danger:
                nearest = min(
                    abs(new_r - dr) + abs(new_c - dc) for dr, dc in danger
                )
                if nearest <= 2:
                    remaining += (3 - nearest) * 4.0
            if remaining < best_cost:
                best, best_cost = action, remaining
        return best

    # ── Action selection ──────────────────────────────────────

    def choose_action(self, action_space: list[Any]) -> Any:
        """Pick an action: navigate when the model suffices, else explore.

        Exploration decays once the self-model is learned — these games
        often have step budgets, and wandering with epsilon=0.3 spends
        a third of its life exploring what it already knows.
        """
        if not action_space:
            raise ValueError("empty action space")

        self._recognize_env(action_space)

        eps = self.epsilon if self.avatar_color is None else self.epsilon * 0.25
        if self._grit_steps > 0:
            # Encouragement: for a while after losing a life, stay
            # curious — a fresh try beats repeating the fatal path.
            self._grit_steps -= 1
            eps = max(eps, self.epsilon)
        if self._rng.random() >= eps and self._last_frame is not None:
            nav = self._navigate(self._last_frame, action_space)
            if nav is not None:
                self._last_action = nav
                return nav

        # Exploration: prefer untried actions, then high-effect ones.
        untried = [a for a in action_space if self._stat(a).attempts == 0]
        if untried:
            action = self._rng.choice(untried)
        else:
            weights = [
                0.1 + self._stat(a).mean_change for a in action_space
            ]
            action = self._rng.choices(action_space, weights=weights)[0]
        self._last_action = action
        return action

    def _click_targets(self, grid: Grid) -> list[tuple[int, int]]:
        """Candidate click cells: object centroids first, then cells.

        Objects are salient in these environments — clicking a thing
        is far more likely to matter than clicking background. Each
        object's centroid comes before its remaining cells so a single
        click per object is tried before exhaustive coverage.
        """
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        targets: list[tuple[int, int]] = []
        rest: list[tuple[int, int]] = []
        for obj in scene.objects:
            if obj.color == self.avatar_color:
                continue
            targets.append(
                (round(obj.centroid[0]), round(obj.centroid[1]))
            )
            rest.extend(sorted(obj.cells))
        return targets + rest

    def action_data(self, action: Any, frame_shape: tuple[int, int]) -> dict:
        """Coordinates for complex actions: fresh, salient cells.

        Click memory drives targeting: prefer object cells never
        clicked, then any grid cell never clicked (a sweep with stride
        2 for coverage), then the cell with the best observed change.
        A cell that produced no change is never clicked twice.
        """
        grid = self._last_frame
        h, w = frame_shape
        if grid is not None:
            candidates = self._click_targets(grid)
            if not candidates:
                candidates = [
                    (r, c)
                    for r in range(0, h, 2)
                    for c in range(0, w, 2)
                ]
            fresh = [
                t for t in candidates if t not in self._click_outcomes
            ]
            if fresh:
                r, c = fresh[0]
            elif self._click_outcomes:
                # Everything tried — return to whatever did the most.
                r, c = max(
                    self._click_outcomes, key=lambda k: self._click_outcomes[k]
                )
            else:
                r, c = candidates[0]
            self._pending_click = (r, c)
            return {"x": c, "y": r}
        r = self._rng.randint(0, h - 1)
        c = self._rng.randint(0, w - 1)
        self._pending_click = (r, c)
        return {"x": c, "y": r}

    # ── Episode boundary ──────────────────────────────────────

    def mark_goal_reached(self) -> None:
        """Record that the avatar reached its current goal cell."""
        if self._last_frame is not None:
            goal = self._pick_goal(self._last_frame)
            pos = self._avatar_pos(self._last_frame)
            if goal is not None and pos is not None:
                if (
                    abs(goal[0] - pos[0]) + abs(goal[1] - pos[1]) <= 2
                ):
                    self._visited_goals.add(
                        (round(goal[0]), round(goal[1]))
                    )

    def on_episode_end(self, state: str) -> None:
        """Called on WIN / GAME_OVER / budget exhaustion.

        The learned self-model (avatar color, action effects, visited
        goals) persists — only the perceptual trace resets. The
        episode's outcome is also reported to the shared competence
        substrate: only a verified WIN consolidates the learned
        control map as a reusable skill.
        """
        if state == "WIN" and self._last_action is not None:
            self._stat(self._last_action).wins += 1
        if self._task_context is not None:
            if state != "WIN":
                for skill_id in self._episode_skill_ids:
                    self.task_competence.mark_skill_failure(skill_id)
            self.task_competence.record_episode(
                self._task_context,
                steps=self._skill_steps(),
                success=state == "WIN",
                verification_score=1.0 if state == "WIN" else 0.0,
                goal_conditions=[
                    GoalCondition("episode.state", "eq", "WIN")
                ],
                # The environment's terminal condition is the world
                # reporting the outcome.
                verification="external",
            )
            # Next level/instance re-recognizes — possibly the same
            # schema, possibly a new one. Object identities do not
            # carry across episodes: a new instance binds fresh.
            self._task_context = None
            self._episode_skill_ids.clear()
            self._tracks.clear()
            self._prior_hazard_speed = None
        self._last_frame = None
        self._last_action = None
        self._pending_click = None
