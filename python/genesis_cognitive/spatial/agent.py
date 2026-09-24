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
       The object whose cells consistently move when she acts is
       *her avatar* — self-recognition in a visual field, learned
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

On WIN/GAME_OVER the episode resets, but the learned model (avatar
color, action displacements, object memory) persists — she gets to
keep what she figured out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

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
    """

    def __init__(self, seed: int = 42, epsilon: float = 0.3) -> None:
        self._rng = random.Random(seed)
        self.epsilon = epsilon
        self.stats: dict[Any, ActionStats] = {}
        self._last_frame: Grid | None = None
        self._last_action: Any = None
        # Learned self-model — persists across episodes.
        self.avatar_color: int | None = None
        self._avatar_votes: dict[int, int] = {}
        # Learned hazard model: colors that move on their own (not
        # caused by her action). Moving things that aren't her are
        # almost always things to avoid, not goals — collectibles and
        # exits are static in these environments.
        self._hazard_colors: set[int] = set()
        self._mover_votes: dict[int, int] = {}
        # Per-hazard velocity: last observed (dr, dc) centroid shift.
        # Movers in these environments are usually ballistic (fixed
        # direction per step), so one-step extrapolation predicts
        # where they'll be when her move lands.
        self._hazard_velocity: dict[int, tuple[float, float]] = {}
        # Reward inference: colors whose objects vanished while the
        # avatar was touching them are pickups — consumable, valuable.
        # Learned empirically from consequences, not told.
        self._valuable_colors: set[int] = set()
        # Stations: objects that change the *avatar's* appearance when
        # touched (key/color/rotation stations in gated games).
        self._station_colors: set[int] = set()
        # Gated cells: goal positions she has reached but couldn't
        # collect — the object persisted. They stay locked until her
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
        # Interaction novelty (the coach's hint): colors she has never
        # touched are worth touching — consequences can't be learned
        # without contact. This is a prior about *how to explore*,
        # not an answer.
        self._touched_colors: set[int] = set()
        # Encouragement / grit: after losing a life she gets a short
        # window of raised exploration — she shakes it off and tries
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

        # Agency detection: find the color whose centroid moved most
        # consistently with this action. The avatar is whatever the
        # agent controls — detected from contingency, not told.
        best_color, best_shift = None, 0.0
        best_dr = best_dc = 0.0
        movers: list[int] = []
        for color in prev.colors() | cur.colors():
            if color == 0:
                continue
            c0 = _color_centroid(prev, color)
            c1 = _color_centroid(cur, color)
            if c0 is None or c1 is None:
                continue
            # A collected object's cells vanish — centroid shifts with
            # no motion. Don't let consumption masquerade as movement:
            # a real mover keeps (roughly) its cell count.
            n0 = len(prev.cells_with(color))
            n1 = len(cur.cells_with(color))
            if abs(n1 - n0) >= max(2, n0 * 0.25):
                continue
            dr, dc = c1[0] - c0[0], c1[1] - c0[1]
            shift = abs(dr) + abs(dc)
            if shift > 0:
                movers.append(color)
            if shift > best_shift:
                best_color, best_shift = color, shift
                best_dr, best_dc = dr, dc
        # Hazard learning: colors that moved but aren't the avatar
        # candidate this step — things that move on their own are
        # avoided as goals. Votes accumulate so one noisy frame can't
        # mark a goal color forever.
        for color in movers:
            if color == best_color:
                continue
            c0 = _color_centroid(prev, color)
            c1 = _color_centroid(cur, color)
            if c0 is not None and c1 is not None:
                self._hazard_velocity[color] = (
                    c1[0] - c0[0], c1[1] - c0[1]
                )
            self._mover_votes[color] = self._mover_votes.get(color, 0) + 1
            if self._mover_votes[color] >= 3:
                self._hazard_colors.add(color)
        if best_color is not None and best_shift > 0:
            # Attribute the winning displacement to this action.
            st.dr += best_dr
            st.dc += best_dc
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
            change = self._learn_from_step(
                self._last_frame, grid, self._last_action
            )
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
        erased) was consumed by her action — a collectible, a step
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
                self._mover_votes.pop(obj.color, None)
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
          vanish — it's locked until her key state changes. Two
          failed touches marks the cell gated.
        - Station: an object the avatar was near when her own color
          signature changed — it transformed her (key stations). When
          gated goals exist, stations become priority targets.
        """
        cur_pos = self._avatar_pos(cur)
        prev_pos = self._avatar_pos(prev)
        if cur_pos is None:
            # Avatar hidden — remember where she was last seen; if she
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
        # step (respawn), meaning whatever she was touching killed her.
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
        # of curiosity follows so she tries a different path.
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
        # When gated goals exist, stations that change her key state
        # are the way forward — visit them before re-trying gates.
        stations = (
            [g for g in goals if g.color in self._station_colors]
            if self._gated_cells
            else []
        )
        # The hint: object colors never touched can't teach her
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
        moving-hazard color is expensive. She doesn't just avoid making
        hazards her destination — she avoids walking through them.
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
            if st is None or st.attempts == 0:
                continue  # untried actions belong to exploration
            step_dr = st.dr / st.attempts
            step_dc = st.dc / st.attempts
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
        a third of her life exploring what she already knows.
        """
        if not action_space:
            raise ValueError("empty action space")

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
        goals) persists — only the perceptual trace resets.
        """
        if state == "WIN" and self._last_action is not None:
            self._stat(self._last_action).wins += 1
        self._last_frame = None
        self._last_action = None
        self._pending_click = None
