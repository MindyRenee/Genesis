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

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from frontal_lobe.reasoning import (
    GoalCondition,
    ProcedureStep,
    TaskCompetence,
    TaskContext,
)
from parietal_lobe.boards import ConstraintSession
from parietal_lobe.grid import Grid
from parietal_lobe.scene import perceive

_DBG = bool(os.environ.get("GENESIS_AGENT_DEBUG"))


def _dbg(msg: str) -> None:
    # stderr, not print — debug output must not pollute stdout
    sys.stderr.write(f"    [dbg] {msg}\n")


@dataclass
class ActionStats:
    """What the agent has learned about an action's effect."""

    attempts: int = 0
    total_change: float = 0.0  # summed fraction of cells changed
    wins: int = 0
    dr: float = 0.0  # learned mean row displacement of the avatar
    dc: float = 0.0  # learned mean column displacement
    moves: int = 0  # attempts where the avatar actually displaced
    # Per-transition rounded deltas: a teleport or a misattributed
    # track poisons a mean permanently, but never a mode — the modal
    # delta is the action's honest stride even under contamination.
    delta_counts: dict[tuple[int, int], int] = field(
        default_factory=dict
    )

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
    Layers may be numpy arrays or plain nested lists — detect either.
    """
    if isinstance(frame, Grid):
        return frame
    if isinstance(frame, (list, tuple)) and frame:
        first = frame[0]
        is_layered = hasattr(first, "shape") or (
            isinstance(first, (list, tuple))
            and first
            and isinstance(first[0], (list, tuple))
        )
        if is_layered:
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


class _GoalView:
    """Per-frame goal predicates for one ``_pick_goal`` call.

    Eligibility, probation, strikes, spend, and openability all close
    over a single frame's grid, body, and signature — holding them on
    a view object keeps ``_pick_goal`` an orchestration while the
    verdicts keep their original wording. Build it *after*
    ``_read_gate_requirements``: ``openable_now`` consults
    ``agent._req_diffs``.
    """

    def __init__(
        self,
        agent: SpatialAgent,
        grid: Grid,
        body: set[tuple[int, int]] | frozenset[tuple[int, int]] | None,
        cur_sig: Any,
    ) -> None:
        self.agent = agent
        self.grid = grid
        self.body = body
        self.cur_sig = cur_sig
        # A locked goal makes key-stations worth re-visiting — but a
        # walkable station is "open" the moment she stands on it, so
        # the open-color filter must not silence it. Only stations
        # earned by observed key-state changes get this exemption.
        self.stations_matter = bool(
            agent._gated_cells or agent._gate_blocked
        )
        # Interface territory: marked meters plus enrolled display
        # panels — pixels that are her own state, not the world.
        self.inside = agent._ui_cells | {
            cell
            for key in agent._readout_keys
            for cell in agent._readout_cells.get(key, frozenset())
        }

    def openable_now(self, obj: Any) -> bool:
        """This gate's recorded opener — or posted requirement —
        matches my current key."""
        agent = self.agent
        if agent._req_diffs.get(agent._obj_key(obj)) == set():
            return True
        if self.cur_sig is None:
            return False
        return (
            agent._gate_openers.get(agent._obj_signature(obj, self.grid))
            == self.cur_sig
        )

    def self_echo(self, obj: Any) -> bool:
        """An object that draws a fragment of her own body
        pixel-for-pixel elsewhere is a status glyph — her icon
        re-rendered on a panel — not something to walk to. The test
        is silhouette equality under translation for the matching
        body color: same cell count, same normalized shape."""
        body, grid = self.body, self.grid
        if not body:
            return False
        mine = {
            cell for cell in body if grid.at(*cell) == obj.color
        }
        if not mine or len(mine) != len(obj.cells):
            return False
        mr0 = min(r for r, _ in mine)
        mc0 = min(c for _, c in mine)
        or0 = min(r for r, _ in obj.cells)
        oc0 = min(c for _, c in obj.cells)
        return {(r - mr0, c - mc0) for r, c in mine} == {
            (r - or0, c - oc0) for r, c in obj.cells
        }

    def eligible(self, obj: Any) -> bool:
        """Everything but the provisional reachability verdict."""
        agent, body, inside = self.agent, self.body, self.inside
        if inside and all(cell in inside for cell in obj.cells):
            # An enrolled display's panel or a marked meter —
            # its pixels are the interface, her own state, not
            # a thing in the world.
            return False
        if self.self_echo(obj):
            return False
        return (
            obj.color != agent.avatar_color
            and not (body and set(obj.cells) & body)
            and obj.color not in agent._hazard_colors
            and (
                obj.color not in agent._open_colors
                or len(obj.cells) <= 64
                or (
                    self.stations_matter
                    and obj.color in agent._station_colors
                )
            )
            and not all(
                cell in agent._blocked_cells for cell in obj.cells
            )
            and (
                not any(
                    cell in agent._gated_cells for cell in obj.cells
                )
                or self.openable_now(obj)
            )
        )

    def on_probation(self, obj: Any) -> bool:
        """Goal BFS couldn't reach — retried as the map grows.

        Marks consolidate onto the neighbourhood that earned them —
        a flickering icon re-segments into fragments whose cells
        don't cover the mark, so the verdict has to match the same
        window that anchored it, not the pixels of this frame's
        fragment.
        """
        agent = self.agent
        return any(
            len(agent._visited_cells) - seen < max(200, seen // 8)
            and any(
                abs(mark[0] - r) <= 5 and abs(mark[1] - c) <= 5
                for r, c in obj.cells
            )
            for mark, seen in agent._unreachable_goals.items()
        )

    def struck_out(self, obj: Any) -> bool:
        """Unreachable three times running — a verdict, not a
        probation. The strike survives the pool-exhaustion wipe,
        so repeated proof against the same place is final."""
        return any(
            count >= 3
            and any(
                abs(mark[0] - r) <= 5 and abs(mark[1] - c) <= 5
                for r, c in obj.cells
            )
            for mark, count in self.agent._unreach_strikes.items()
        )

    def spent(self, g: Any) -> bool:
        """A goal is spent if any of its cells lies inside a footprint
        she already occupied — or reached for — at this signature.
        Spent goals leave the tier competition entirely; the
        whole-set fallback only fires when nothing is left to try."""
        return any(
            sig == self.cur_sig
            and any(
                r0 <= r < r0 + h and c0 <= c < c0 + w
                for r, c in g.cells
            )
            for r0, c0, h, w, sig in self.agent._tried_at_state
        )

    def frontier(self, scene: Any) -> list[Any] | None:
        """The eligible, unstruck, off-probation objects — with the
        probation-wipe fallback when every candidate is marked.
        ``None`` clears ``_goal_cells``: nothing is left to try."""
        agent = self.agent
        goals = [
            obj for obj in scene.objects
            if self.eligible(obj) and not self.on_probation(obj)
            and not self.struck_out(obj)
        ]
        if goals:
            return goals
        pool = [
            obj
            for obj in scene.objects
            if self.eligible(obj) and not self.struck_out(obj)
        ]
        if pool:
            # Every candidate is probationary — the verdicts were
            # recorded against a smaller map, and the frontier is
            # gone, so probation can never expire on its own.
            # Wipe the marks and let BFS retry: a sealed goal
            # simply re-earns its mark, a reachable one resumes.
            agent._unreachable_goals.clear()
            return pool
        agent._goal_cells = frozenset()
        return None

    def dump_drops(self, scene: Any) -> None:
        """Debug: why each small object fell out of the pool."""
        agent, body, inside = self.agent, self.body, self.inside
        for o in scene.objects:
            if len(o.cells) > 30:
                continue
            why = (
                "inside" if inside and all(c in inside for c in o.cells)
                else "echo" if self.self_echo(o)
                else "avcol" if o.color == agent.avatar_color
                else "body" if body and set(o.cells) & body
                else "haz" if o.color in agent._hazard_colors
                else "open" if (
                    o.color in agent._open_colors
                    and len(o.cells) > 64
                    and not (
                        self.stations_matter
                        and o.color in agent._station_colors
                    )
                )
                else "blk" if all(
                    c in agent._blocked_cells for c in o.cells
                )
                else "gated" if (
                    any(c in agent._gated_cells for c in o.cells)
                    and not self.openable_now(o)
                )
                else "prob" if self.on_probation(o)
                else "struck" if self.struck_out(o)
                else "OK"
            )
            if why != "OK":
                _dbg(
                    f"drop c{o.color}@{round(o.centroid[0])},"
                    f"{round(o.centroid[1])} n={len(o.cells)} {why}"
                )

    def dump_commit(
        self,
        scene: Any,
        committed: tuple[Any, frozenset[tuple[int, int]]],
        reach: dict[tuple[int, int], int] | None,
    ) -> None:
        """Debug: whether the committed goal survived this frame."""
        agent = self.agent
        col, cells = committed
        alive_obj = [
            obj for obj in scene.objects
            if obj.color == col and set(obj.cells) & cells
        ]
        if not alive_obj:
            _dbg(f"commit-gone c{col}")
        for obj in alive_obj:
            _dbg(
                f"commit-alive c{col} elig={self.eligible(obj)} "
                f"prob={self.on_probation(obj)} spent={self.spent(obj)} "
                f"reachN={len(reach or {})} "
                f"strikes={dict(agent._unreach_strikes)}"
            )


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
        self._init_agency_state()
        self._init_semantic_state()
        self._init_planning_state()
        self._init_attribution_state()
        self._init_interface_state()

    def _init_agency_state(self) -> None:
        """Body identity and self-motion evidence — the learned
        self-model that persists across episodes."""
        # Learned self-model — persists across episodes.
        self.avatar_color: int | None = None
        # Agency evidence, per action per color: [Σdr, Σdc, n]. The
        # body is the color whose displacement *depends on which
        # action* was taken — action-contingency, not action-
        # correlation. A HUD counter that changes every step and a
        # patrolling hazard both move on every action identically;
        # only the body answers differently to different commands.
        self._action_color_disp: dict[Any, dict[int, list[float]]] = {}
        # Last known avatar centroid + silhouette — the anchors for
        # object-level body tracking across same-colored distractors:
        # shape-under-translation is what a wall can't counterfeit.
        self._avatar_last: tuple[float, float] | None = None
        self._avatar_shape: frozenset[tuple[int, int]] = frozenset()
        self._avatar_sig: tuple[tuple[int, ...], ...] | None = None
        # Co-moving body colors: multi-color avatars present each
        # band as a separate object; a color whose displacement
        # agrees with the primary body's under every shared action
        # is part of her, not a hazard, goal, or consumable.
        self._co_colors: set[int] = set()
        # Solo-motion evidence: a color that displaced while the body
        # didn't (a bump, a frozen frame) moves on its own — enemies
        # patrol through her collisions; her bands wait with her.
        self._solo_moves: dict[int, int] = {}
        self._prev_anchor: tuple[int, int, int, int] | None = None
        # Life-loss across a hidden frame: the avatar sprite goes
        # invisible on the death frame, then reappears at spawn.
        self._missing_avatar_at: tuple[float, float] | None = None
        # Her signature has been *seen* to vary — only then can a
        # body/display token be trusted to carry state. Before that,
        # a constant body token would manufacture constant nonsense
        # diffs against every requirement.
        self._sig_varies = False

    def _init_semantic_state(self) -> None:
        """Learned world semantics — hazards, pickups, stations,
        gates, and collision-mapped terrain."""
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
        # Lethal colors: contact caused a respawn. Strongest evidence
        # there is — consumption signals can never un-mark these.
        self._lethal_colors: set[int] = set()
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
        # Station effect memory: station color -> the signature
        # dimensions it perturbs ("recolor" | "reshape" | "reorient").
        self._station_effects: dict[int, set[str]] = {}
        # Goals spent under a key state: (anchor row, col, h, w,
        # sig) — the footprint the avatar occupied or was refused
        # from. A state change makes every tried goal a fresh
        # question — that is the gate-retry loop, not memory loss.
        self._tried_at_state: set[
            tuple[int, int, int, int, tuple[tuple[int, ...], ...]]
        ] = set()
        # Terrain memory: cells the avatar tried to enter and couldn't
        # — collision-learned walls. The planner routes around them;
        # unknown cells stay optimistically passable until touched.
        self._blocked_cells: set[tuple[int, int]] = set()
        # One refused strip is a suspicion, not a wall: the box blames
        # every unproven cell inside it, and a bump at a corridor's
        # edge condemns the floor beside the real wall. Cells earn
        # blocked status the second time a refused strip covers them —
        # real walls are what refused destinations keep overlapping.
        self._block_suspect: dict[tuple[int, int], int] = {}
        # Anchors the avatar has occupied this episode — the
        # frontier-exploration signal: unvisited passable space is
        # what remains to be learned.
        self._visited_cells: set[tuple[int, int]] = set()
        # Proven-open colors: values found where her body has been.
        # Floor can't refuse a move — a wall wearing the frame's
        # modal color still can, so "blocked" is judged against
        # evidence of openness, not frequency.
        self._open_colors: set[int] = set()
        # Bump memory: (anchor_r, anchor_c, action) — a move that
        # displaced nothing at that spot must not be retried there.
        # Every action costs budget; a known bump is pure waste.
        # The split is by what refused: _bumped_moves holds refusals
        # into background-colored cells (terrain — permanent);
        # _bumped_gates holds refusals into colored cells (a gate
        # rejecting the current state — cleared on signature change,
        # when the same destination may accept).
        self._bumped_moves: set[tuple[int, int, Any]] = set()
        self._bumped_gates: set[tuple[int, int, Any]] = set()
        # Interaction novelty (the coach's hint): colors it has never
        # touched are worth touching — consequences can't be learned
        # without contact. This is a prior about *how to explore*,
        # not an answer.
        self._touched_colors: set[int] = set()

    def _init_planning_state(self) -> None:
        """Committed subgoals, planner targets, and exploration
        bookkeeping — the state a plan survives on."""
        # Committed subgoal: (color, remembered cells) of the object
        # she is traveling toward. Identity by shared footprint, not
        # centroid — occlusion reshapes a blob's centroid while its
        # cells persist, so a cell match keeps the target through
        # re-segmentation. Re-deciding nearest every frame ping-pongs
        # between distant goals and burns move budget; a plan means
        # finishing what she started.
        self._committed_goal: tuple[int, frozenset] | None = None
        # The goal object's live footprint — the planner's contact
        # target. Set alongside every _pick_goal verdict.
        self._goal_cells: frozenset = frozenset()
        # Frames the committed object has been absent from the scene.
        # Goal pixels flicker — occlusion and icon redraws split or
        # erase a blob for a frame — so a missing sighting holds the
        # trek rather than instantly forfeiting it to scenery.
        self._commit_missing = 0
        # The anchor the planner is currently routing toward — goal
        # approach or frontier. Re-picking nearest every frame
        # oscillates between midpoints; a plan means finishing it.
        self._plan_target: tuple[int, int] | None = None
        # Objects whose disappearance is fully explained by her own
        # silhouette — occluded, not proven consumed. Resolved when
        # the cells come back into view: still gone means taken,
        # back again means floor.
        self._occluded_objects: dict[int, set[tuple[int, int]]] = {}
        # Encouragement / grit: after losing a life it gets a short
        # window of raised exploration — it shakes it off and tries
        # something different instead of repeating the fatal path.
        self._grit_steps: int = 0
        # Click memory: cell -> observed frame change. A cell that
        # already produced ~nothing is a dead spot — clicking it again
        # can never help, so complex actions target fresh cells.
        self._click_outcomes: dict[tuple[int, int], float] = {}
        self._pending_click: tuple[int, int] | None = None

    def _init_attribution_state(self) -> None:
        """Outcome credit assignment — which actions, clicks, and
        colors precede wins and deaths."""
        # Death credit assignment, per-episode presence: action ->
        # [episodes used, episodes died]. The fatal choice in these
        # games is often delayed — a bad move dooms the level but the
        # death lands steps later, so last-action blame hits whatever
        # was in flight. Presence is the honest signal: an action that
        # only ever appears in episodes that end in death is lethal;
        # one present in wins and losses alike is innocent.
        self._action_outcomes: dict[Any, list[int]] = {}
        self._cur_episode_actions: set[Any] = set()
        self._episodes_total = 0
        self._episodes_died = 0
        # Same attribution at click granularity: a clicked cell whose
        # episodes reliably die is a lethal control — the "wrong
        # button" pattern. Cells that demonstrably respond (frame
        # change > 0) are reusable controls, not one-shot targets.
        self._cell_outcomes: dict[tuple[int, int], list[int]] = {}
        self._cur_episode_clicks: set[tuple[int, int]] = set()
        # Omen tracking: color -> [segments appeared, segments died].
        # A color that only ever shows up in segments that end in
        # death is a doom marker; whatever introduced it is suspect.
        self._color_outcomes: dict[int, list[int]] = {}
        self._cur_episode_colors: set[int] = set()
        self._seg_introduced: dict[int, Any] = {}
        # Introducer blame: action name or click cell -> times it
        # brought a known doom-marker into the world.
        self._doom_blame: dict[Any, int] = {}
        # Deaths with no killer in evidence — the step budget, not a
        # hazard. Counted so dithering has a felt cost without
        # poisoning the terrain memory. ``_last_death_at`` paces the
        # detector: a tracking flicker masquerades as a teleport-
        # respawn every other frame, and a life cannot actually be
        # spent twice in that span.
        self._spent_deaths = 0
        self._last_death_at = -99

    def _init_interface_state(self) -> None:
        """Non-body panels — constraint boards, detached readouts,
        interface regions, and reachability verdicts."""
        # The constraint-board specialist: when a screen is tiled
        # boards + marker blocks and the action space is clicks, its
        # induced rule turns wandering clicks into a plan. Silent
        # whenever the parse or the rule doesn't hold — the general
        # policy below owns those turns.
        self._boards = ConstraintSession()
        self._board_click: tuple[int, int] | None = None
        # Off-body state displays: an agent's key state isn't always
        # worn on its body — it can sit in a detached readout panel.
        # Static objects (same cells every frame) whose pattern
        # changes only on contact steps are enrolled as readouts and
        # appended to the avatar signature; meters that change freely
        # every step stay out.
        self._readout_stats: dict[tuple[int, int], list[int]] = {}
        self._readout_keys: list[tuple[int, int]] = []
        self._readout_cells: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
        # Interface regions: static panels that redraw on their own
        # schedule — step counters, life meters, detached signature
        # displays. The pixels change without her ever reaching the
        # panel, which no goal does: a goal holds still until it is
        # collected. Marked regions leave goal candidacy the way her
        # body does.
        self._ui_cells: set[tuple[int, int]] = set()
        # Goal targets the planner proved unreachable *with the map
        # known so far* — off-world UI objects and sealed rooms both
        # look like goals until BFS exhausts the passable space. Each
        # mark records the explored-anchor count at the time; as the
        # map grows the verdict is retried — an early "unreachable"
        # is often just "not discovered yet".
        self._unreachable_goals: dict[tuple[int, int], int] = {}
        # Strikes against the same spot surviving probation retries:
        # a place that keeps re-earning "unreachable" as the map
        # grows is *proven* unreachable — HUD furniture, sealed
        # scenery — and stops being worth the trek. Counted
        # separately so the pool-exhaustion wipe can retry the map
        # without erasing the accumulated evidence.
        self._unreach_strikes: dict[tuple[int, int], int] = {}
        # Requirement icons: the key state a gate *displays*, read
        # rather than trialed. Keyed by the gate object's anchor.
        # A display icon and her own state icon share one visual
        # language — the signature the gate wants is the signature
        # her readout shows, so the difference decomposes into which
        # state dimension to go change.
        self._gate_requirements: dict[
            tuple[int, int], tuple[int, frozenset[tuple[int, int]]]
        ] = {}

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
            step = self._action_step(st)
            if step is None:
                continue
            dr, dc = float(step[0]), float(step[1])
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
            cur_cells = set(obj.cells)
            prev_cells = set(track.cells)
            if cur_cells < prev_cells or cur_cells > prev_cells:
                # A proper subset/superset is a resize — a gauge
                # draining or a tide spreading — not a walk. Meters
                # shift centroid without going anywhere; counting
                # them as movers calls the status bar a hazard.
                track.velocity = (0.0, 0.0)
            else:
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

        # Displacement at color granularity, no track binding needed:
        # a rigid sprite jumping its own size outruns the bind radius
        # while its color's centroid still translates exactly by the
        # stride. This is the agency evidence — which color's delta
        # co-varies with the action — and, once the body is known,
        # the stride itself.
        color_disp = self._color_displacements(prev, cur)
        prim_move = (
            color_disp.get(self.avatar_color, (0.0, 0.0))
            if self.avatar_color is not None
            else None
        )
        for color, (dr, dc) in color_disp.items():
            if dr or dc:
                rec = self._action_color_disp.setdefault(
                    action, {}
                ).setdefault(color, [0.0, 0.0, 0])
                rec[0] += dr
                rec[1] += dc
                rec[2] += 1
                if (
                    prim_move is not None
                    and color != self.avatar_color
                    and abs(prim_move[0]) + abs(prim_move[1]) < 0.5
                    and abs(dr) + abs(dc) >= 0.75
                ):
                    self._solo_moves[color] = (
                        self._solo_moves.get(color, 0) + 1
                    )
        self._update_avatar(self._background(cur), prev, cur)
        # Hazard learning: tracks that moved but aren't the avatar
        # candidate — things that move on their own are avoided as
        # goals. Votes accumulate per object so one noisy frame can't
        # mark a color forever, and a freshly spawned lookalike can't
        # inherit another object's motion record.
        for tid, (dr, dc) in moved.items():
            track = self._tracks[tid]
            if track.color in self._avatar_colors:
                continue
            if track.color in self._open_colors:
                continue  # proven floor — she has stood in it
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
                if len(track.cells) <= cur.height * cur.width * 0.2:
                    self._hazard_colors.add(track.color)
                # else: a "mover" spanning a fifth of the board is
                # floor or UI whose apparent motion is occlusion —
                # treating it as a threat marks the map impassable.
        if self.avatar_color is not None:
            # The stride is the avatar *object's* centroid delta —
            # nearest-match tracked so static things sharing the
            # body's color can't dilute it, and attributed to the
            # action taken whatever else moved this frame.
            prev_obj = self._avatar_obj(prev)
            cur_obj = self._avatar_obj(cur)
            if prev_obj is not None and cur_obj is not None:
                dr = cur_obj.centroid[0] - prev_obj.centroid[0]
                dc = cur_obj.centroid[1] - prev_obj.centroid[1]
                if dr or dc:
                    st.dr += dr
                    st.dc += dc
                    st.moves += 1
                    key = (round(dr), round(dc))
                    st.delta_counts[key] = st.delta_counts.get(key, 0) + 1
        return change

    @staticmethod
    def _color_displacements(
        prev: Grid, cur: Grid
    ) -> dict[int, tuple[float, float]]:
        """Per-color centroid delta between two frames.

        Colors present in both frames only — a vanished object's
        disappearance reads as a centroid jump and would fake motion.
        """
        out: dict[int, tuple[float, float]] = {}
        for color in prev.colors() & cur.colors():
            prev_cells = prev.cells_with(color)
            cur_cells = cur.cells_with(color)
            if not prev_cells or not cur_cells:
                continue
            pr = sum(r for r, _ in prev_cells) / len(prev_cells)
            pc = sum(c for _, c in prev_cells) / len(prev_cells)
            nr = sum(r for r, _ in cur_cells) / len(cur_cells)
            nc = sum(c for _, c in cur_cells) / len(cur_cells)
            out[color] = (nr - pr, nc - pc)
        return out

    def _update_avatar(
        self, background: int, prev: Grid | None = None, cur: Grid | None = None
    ) -> None:
        """Decide which color is the body by action-contingency.

        For each color, compare its mean displacement under each
        action tried. The body's means *differ by action* (up under
        1, down under 2). Self-movers, HUD counters, and consumed
        objects all show action-agnostic deltas — their means agree.
        The floor itself is excluded: it shifts in mirror of the
        body and would fake contingency.
        """
        colors: set[int] = set()
        for per_color in self._action_color_disp.values():
            colors.update(per_color)
        colors.discard(background)
        self._find_body_color(colors, prev, cur)
        self._find_co_colors(colors)

    def _find_body_color(
        self, colors: set[int], prev: Grid | None, cur: Grid | None
    ) -> None:
        """The body color is the one whose per-action displacement
        means disagree most — self-movers and HUD counters move the
        same way under every action, so their means agree."""
        best_color, best_spread = self.avatar_color, 0.0
        for color in colors:
            means = [
                (rec[0] / rec[2], rec[1] / rec[2])
                for per_color in self._action_color_disp.values()
                for c, rec in per_color.items()
                if c == color and rec[2] >= 2
            ]
            if len(means) < 2:
                continue  # one action can't express contingency
            spread = max(
                abs(a[0] - b[0]) + abs(a[1] - b[1])
                for i, a in enumerate(means)
                for b in means[i + 1 :]
            )
            peak = max(abs(m[0]) + abs(m[1]) for m in means)
            if peak < 0.75:
                continue  # barely moves under anything — not a body
            if spread > best_spread:
                best_color, best_spread = color, spread
        if not (
            best_spread >= 1.0
            and best_color is not None
            and self.avatar_color != best_color
        ):
            return
        first = self.avatar_color is None
        self.avatar_color = best_color
        if not first:
            return
        # The body was found — every displacement it already
        # produced under each action is *that action's* learned
        # effect. Reattributing the ledger keeps early strides
        # instead of discarding everything learned pre-agency.
        for a, per_color in self._action_color_disp.items():
            rec = per_color.get(best_color)
            if rec is None:
                continue
            st = self._stat(a)
            st.dr, st.dc, st.moves = rec[0], rec[1], int(rec[2])
            # The color ledger is a summed centroid — the per-
            # transition detail is lost, so seed the modal
            # stride from the aggregate: a uniform estimate is
            # a weaker prior than real deltas but better than
            # discarding everything learned pre-agency.
            key = (
                round(rec[0] / rec[2]),
                round(rec[1] / rec[2]),
            )
            st.delta_counts[key] = int(rec[2])
        if prev is not None and cur is not None:
            # Bootstrap identity from arrival evidence: among
            # same-colored things, the body is the one whose
            # cells are newest — a wall's cells were already
            # there; the mover's just arrived.
            scene = perceive(
                cur,
                background=self._background(cur),
                compute_relations=False,
            )
            prev_cells = prev.cells_with(best_color)
            cands = [
                o for o in scene.objects if o.color == best_color
            ]
            if cands:
                obj = max(
                    cands,
                    key=lambda o: len(set(o.cells) - prev_cells),
                )
                self._avatar_shape = obj.cells
                self._avatar_last = obj.centroid

    def _find_co_colors(self, colors: set[int]) -> None:
        """Co-moving colors: a multi-color body's other bands move
        identically under the same actions. Any color whose
        per-action mean displacement agrees with the body's on
        every action they share is part of her — an enemy's
        motion doesn't shadow hers, and a HUD counter changes
        the same way under every action, so neither qualifies."""
        self._co_colors.clear()
        if self.avatar_color is None:
            return
        prim = {
            a: (rec[0] / rec[2], rec[1] / rec[2])
            for a, per_color in self._action_color_disp.items()
            if (rec := per_color.get(self.avatar_color))
            and rec[2] >= 1
        }
        for color in colors:
            if color == self.avatar_color:
                continue
            if self._solo_moves.get(color, 0) >= 3:
                # It moves when she doesn't — a self-mover
                # that happened to agree twice, not a limb.
                continue
            agree = 0
            for a, per_color in self._action_color_disp.items():
                rec = per_color.get(color)
                if rec is None or rec[2] < 1 or a not in prim:
                    continue
                mr, mc = rec[0] / rec[2], rec[1] / rec[2]
                pr, pc = prim[a]
                # A co-moving band's mean keeps the body's
                # *direction*, diluted in magnitude by static
                # same-colored objects. A self-mover's mean
                # holds its own heading under every action —
                # it can't align with both north and south.
                # Agreement is directional (cosine), not
                # magnitude.
                mag_m = (mr * mr + mc * mc) ** 0.5
                mag_p = (pr * pr + pc * pc) ** 0.5
                if (
                    mag_m >= 0.3
                    and mag_p > 0
                    and (mr * pr + mc * pc) / (mag_m * mag_p)
                    >= 0.7
                ):
                    agree += 1
            if agree >= 2:
                self._co_colors.add(color)
        # Her other bands can't be hazards — they only ever
        # "move" because she moved.
        self._hazard_colors -= self._avatar_colors

    @property
    def _avatar_colors(self) -> set[int]:
        """Every color that is part of her body."""
        out = set(self._co_colors)
        if self.avatar_color is not None:
            out.add(self.avatar_color)
        return out

    def _discontinuity(self, prev: Grid, cur: Grid) -> bool:
        """A frame pair no single action produced — not learnable.

        A level change redraws the board; a respawn or teleport moves
        the body farther than any learned stride. Transitions like
        these attributed to a motor command teach the opposite of
        what happened: a vanished maze reads as consumed, a new
        layout's cells read as reached.
        """
        if frame_diff(prev, cur) > 0.35:
            return True
        before = self._avatar_anchor(prev)
        after = self._avatar_anchor(cur)
        if before is None or after is None:
            return False
        st = self.stats.get(self._last_action)
        step = self._action_step(st) if st is not None else None
        if step is None:
            return False
        jump = abs(after[0] - before[0]) + abs(after[1] - before[1])
        return jump > 2.5 * (abs(step[0]) + abs(step[1]))

    def observe(self, frame: Any) -> float:
        """Record a frame; learn from the transition; return change."""
        grid = frame_to_grid(frame)
        change = 0.0
        discontinuous = (
            self._last_frame is not None
            and self._last_action is not None
            and self._discontinuity(self._last_frame, grid)
        )
        if discontinuous:
            # A respawn or a level change breaks every route in
            # progress — the remembered plan target anchors to a
            # world that no longer holds.
            self._plan_target = None
        if (
            self._last_frame is not None
            and self._last_action is not None
            and not discontinuous
        ):
            change = self._learn_transition(self._last_frame, grid)
        if self._pending_click is not None:
            if discontinuous:
                # The world reset between click and frame — no
                # outcome to attribute.
                self._pending_click = None
            else:
                self._click_outcomes[self._pending_click] = change
                self._cur_episode_clicks.add(self._pending_click)
                self._pending_click = None
        if (
            self._last_frame is not None
            and self._last_action is not None
            and not discontinuous
        ):
            self._track_readouts(self._last_frame, grid)
            self._detect_consumption(self._last_frame, grid)
            self._detect_gates_and_stations(self._last_frame, grid)
            self._learn_blocked(self._last_frame, grid, self._last_action)
        anchor = self._avatar_anchor(grid)
        if anchor is not None:
            self._visited_cells.add((anchor[0], anchor[1]))
            self._note_vacated(grid, anchor, discontinuous)
            self._prev_anchor = anchor
            # Occupancy outranks motion suspicion: a color she has
            # stood inside cannot be a hazard — floor-colored blobs
            # flicker at object edges and read as self-movers, but
            # she can't be killed by what she walks on.
            self._hazard_colors -= self._open_colors
            obj = self._avatar_obj(grid)
            if obj is not None:
                self._avatar_last = obj.centroid
                self._avatar_shape = obj.cells
        self._last_frame = grid
        self._boards.observe(grid)
        return change

    def _learn_transition(self, prev: Grid, grid: Grid) -> float:
        """Learn from one frame step: physics, task competence, omens."""
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
            # The terminal transition's changed variables are the
            # abduction signal for goal inference — keep them for
            # record_episode.
            self._last_transition_changes = check.observed_changes
            if check.predicted and check.score < 0.6:
                self._grit_steps = max(self._grit_steps, 20)
        # Omen tracking: colors that newly appear after an action
        # (or click) are that action's visible consequences. Doom
        # often leaves a marker long before the death lands — a
        # "bad" sprite, a wound, a wrong piece. When the segment
        # ends, colors introduced along the way get outcome records
        # and consistent doom-markers blame their introducer.
        source: Any = (
            self._pending_click
            if self._pending_click is not None
            else self._last_action
        )
        prev_colors = {v for _r, _c, v in prev.iter_cells()}
        for _r, _c, v in grid.iter_cells():
            if v not in prev_colors:
                self._seg_introduced.setdefault(v, source)
                self._cur_episode_colors.add(v)
        return change

    def _note_vacated(
        self, grid: Grid, anchor: tuple[int, int, int, int], discontinuous: bool
    ) -> None:
        """Whatever now fills the space she just vacated is by
        definition open — cells her body occupied cannot be
        solid. This is how floor-colored walls stay marked
        while real floor stays trusted. Only cells outside her
        current silhouette count: she still occupies the rest —
        what shows there is her own display, not freed floor."""
        cur_box = {
            (r, c)
            for r in range(anchor[0], anchor[0] + anchor[2])
            for c in range(anchor[1], anchor[1] + anchor[3])
        }
        box = None if discontinuous else self._prev_anchor
        if box is None:
            return
        for r in range(box[0], box[0] + box[2]):
            for c in range(box[1], box[1] + box[3]):
                if not (
                    0 <= r < grid.height and 0 <= c < grid.width
                ) or (r, c) in cur_box:
                    continue
                v = grid.at(r, c)
                if v not in self._avatar_colors:
                    self._open_colors.add(v)

    def _learn_blocked(self, prev: Grid, cur: Grid, action: Any) -> None:
        """Collision → terrain: a move that displaced nothing hit a wall.

        When the action's learned step says the avatar should have
        moved and its anchor didn't change, whatever lay in the
        footprint's destination is impassable terrain. Marked cells
        feed the planner's hard obstacles — one bump teaches, the
        next path routes around.
        """
        if self.avatar_color is None:
            return
        st = self.stats.get(action)
        if st is None or st.moves == 0:
            return
        step = self._action_step(st)
        if step is None:
            return  # not a movement action
        step_r, step_c = step
        before = self._avatar_anchor(prev)
        after = self._avatar_anchor(cur)
        if before is None or after is None:
            return
        if (before[0], before[1]) != (after[0], after[1]):
            return  # it moved — nothing was blocked
        r0, c0, h, w = before
        dr, dc = round(step_r), round(step_c)
        own = self._self_cells(prev)
        # The entered strip refused her — mark what can't be floor.
        # A wall in the frame's modal color is still a wall: only
        # colors proven open by prior occupation get the benefit of
        # the doubt. Movers (hazards) are excluded — their cells are
        # dangerous while they move, not permanently solid.
        bg = self._background(prev)
        refused_wall = False
        refused_gate = False
        candidates: list[tuple[int, int]] = []
        for r in range(r0 + dr, r0 + dr + h):
            for c in range(c0 + dc, c0 + dc + w):
                if (r, c) in own:
                    continue
                if not (0 <= r < prev.height and 0 <= c < prev.width):
                    refused_wall = True  # the frame's edge is terrain
                    continue
                v = prev.at(r, c)
                if (
                    v == bg
                    and v not in self._avatar_colors
                    and v not in self._open_colors
                    and v not in self._hazard_colors
                    and v not in self._lethal_colors
                ):
                    candidates.append((r, c))
                elif v != bg:
                    refused_gate = True
                # A refused colored cell is not terrain — it is a
                # gate candidate. The contact ledger already counted
                # it; after enough refusals _gated_cells hard-blocks
                # it, and a state change can reopen what a wall
                # never could.
        if len(candidates) == 1:
            # The strip has exactly one candidate — attribution is
            # certain: that cell is the wall, outright. Suspicion
            # counting exists only because a wider strip blames
            # every unproven cell inside it.
            self._blocked_cells.add(candidates[0])
            refused_wall = True
        else:
            for cell in candidates:
                self._block_suspect[cell] = (
                    self._block_suspect.get(cell, 0) + 1
                )
                if self._block_suspect[cell] >= 2:
                    self._blocked_cells.add(cell)
                    refused_wall = True
        if refused_wall:
            self._bumped_moves.add((r0, c0, action))
        elif refused_gate:
            self._bumped_gates.add((r0, c0, action))
        else:
            self._bumped_moves.add((r0, c0, action))

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
        body = self._self_cells(prev)
        cur_body = self._self_cells(cur)
        # Resolve what was hidden last time: occluded cells visible
        # again either came back (floor) or stayed gone (consumed).
        for color, cells in list(self._occluded_objects.items()):
            visible = cells - cur_body
            if not visible:
                continue
            if all(cur.at(r, c) != color for r, c in visible):
                self._valuable_colors.add(color)
                self._hazard_colors.discard(color)
            del self._occluded_objects[color]
        for obj in scene.objects:
            if obj.color in (0, bg, self.avatar_color):
                continue
            if body and set(obj.cells) & body:
                continue  # a vacated band isn't a consumed pickup
            if obj.color in self._lethal_colors:
                continue
            near = min(
                (
                    abs(br - r) + abs(bc - c)
                    for r, c in obj.cells
                    for br, bc in (body | cur_body)
                ),
                default=99,
            )
            if near > 2:
                continue
            if near <= 1:
                self._touched_colors.add(obj.color)
            remaining_cells = {
                c for c in obj.cells if cur.at(c[0], c[1]) == obj.color
            }
            vanished = set(obj.cells) - remaining_cells
            # Cells her body now covers did not disappear — they are
            # occluded. Consumption is proven only by cells gone
            # beyond her silhouette; a floor pocket she steps onto
            # "vanishes" by covering, not by being taken.
            uncovered_vanish = vanished - cur_body
            if uncovered_vanish and len(remaining_cells) <= len(obj.cells) // 2:
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
            elif vanished and not uncovered_vanish:
                # Fully covered: consumed or occluded — decide when
                # the cells come back into view.
                self._occluded_objects.setdefault(
                    obj.color, set()
                ).update(obj.cells)

    def _avatar_signature(
        self, grid: Grid
    ) -> tuple[tuple[int, ...], ...] | None:
        """The avatar's key state: its bounding-box pixel pattern.

        Captures color, shape, and rotation — a station that cycles
        any of the three changes this signature. Key state doesn't
        always live on the body: enrolled readout objects (status
        displays that mutate only on contact) append their patterns
        as extra rows, so a detached icon rewrites the signature the
        same way a worn key would.
        """
        if self.avatar_color is None:
            return None
        cells = self._self_cells(grid)
        if not cells:
            return None
        sig = self._cells_signature(cells, grid)
        for key in self._readout_keys:
            rcells = self._readout_cells.get(key)
            if rcells:
                sig = sig + self._cells_signature(rcells, grid)
        return sig

    def _contact_map(
        self, prev: Grid, cur: Grid
    ) -> tuple[
        set[tuple[int, int]],
        set[tuple[int, int]],
        set[tuple[int, int]],
        bool,
    ]:
        """Touch-attribution context for one transition.

        ``strip`` is the footprint she tried to occupy this step — on
        a bump it holds whatever refused her, on a move it is where
        she now stands. ``moved`` separates those two cases.
        """
        body = self._self_cells(prev)
        cur_body = self._self_cells(cur)
        strip: set[tuple[int, int]] = set()
        anchor = self._avatar_anchor(prev)
        cur_anchor = self._avatar_anchor(cur)
        st = self.stats.get(self._last_action)
        step = self._action_step(st) if st is not None and st.moves else None
        if anchor is not None and step is not None:
            dr, dc = step
            if dr or dc:
                r0, c0, h, w = anchor
                strip = {
                    (r, c)
                    for r in range(r0 + dr, r0 + dr + h)
                    for c in range(c0 + dc, c0 + dc + w)
                }
        moved = (
            anchor is not None
            and cur_anchor is not None
            and (anchor[0], anchor[1]) != (cur_anchor[0], cur_anchor[1])
        )
        return body, cur_body, strip, moved

    @staticmethod
    def _contact(
        cells: set[tuple[int, int]],
        body: set[tuple[int, int]],
        cur_body: set[tuple[int, int]],
        strip: set[tuple[int, int]],
        moved: bool,
    ) -> bool:
        """Whether she contacted these cells this step.

        Three honest cases: she stands on them (her cells overlap
        now); she reached for them (her destination strip holds
        them); or she bumped while adjacent. Sliding along a wall is
        none of those — proximity is not touch.
        """
        if cur_body and cells & cur_body:
            return True
        if cells & strip:
            return True
        return bool(
            not moved
            and body
            and min(
                abs(br - r) + abs(bc - c)
                for r, c in cells
                for br, bc in body
            )
            <= 1
        )

    def _track_readouts(self, prev: Grid, cur: Grid) -> None:
        """Enroll static objects that display agent state as readouts.

        A status panel is a static object — same cells every frame —
        whose *pattern* changes only on contact steps: it mutates when
        the avatar touches something that rewrites state, and holds
        otherwise. Meters and counters fail the test because they
        change on free steps too; world objects fail it because they
        move or vanish. Enrolled readouts append to the avatar
        signature, so gates key on shown state, not just body pixels.
        """
        pos = self._avatar_pos(cur)
        if pos is None or self.avatar_color is None:
            return
        bg = self._background(cur)
        prev_scene = perceive(
            prev, background=self._background(prev),
            compute_relations=False,
        )
        body, cur_body, strip, moved = self._contact_map(prev, cur)
        if not body:
            return
        # Contact step: her body reached a foreign object. Open colors
        # are excluded — standing on proven floor is not an event —
        # except stations she has identified, and small open objects:
        # walkable ground is vast while interactables are small, so a
        # bounded decal keeps counting as a touch after its color
        # proves open. Her own bands never count.
        contact = any(
            obj.color not in (0, bg, self.avatar_color)
            and (
                obj.color not in self._open_colors
                or obj.color in self._station_colors
                or len(obj.cells) <= 64
            )
            and not (body and set(obj.cells) & body)
            and self._contact(
                set(obj.cells), body, cur_body, strip, moved
            )
            for obj in prev_scene.objects
        )
        cur_scene = perceive(
            cur, background=bg, compute_relations=False,
        )
        prev_map = {self._obj_key(o): o for o in prev_scene.objects}
        cur_objects = list(cur_scene.objects)
        basis_changed = False
        for key, pobj in prev_map.items():
            if self._check_readout(
                key, pobj, cur_objects, prev, cur, contact
            ):
                basis_changed = True
        if basis_changed:
            # Enrollment coincides with the change it is for — the
            # frame that proved the object a display is also its
            # first real transition. Baseline the signature on the
            # previous frame's readout contents, so the gate/station
            # detector downstream still sees this delta instead of
            # the refresh swallowing it.
            cells = self._self_cells(cur)
            if cells:
                sig = self._cells_signature(cells, cur)
                for key in self._readout_keys:
                    rcells = self._readout_cells.get(key)
                    if rcells:
                        sig = sig + self._cells_signature(rcells, prev)
                self._avatar_sig = sig

    def _check_readout(
        self,
        key: tuple[int, int],
        pobj: Any,
        cur_objects: list[Any],
        prev: Grid,
        cur: Grid,
        contact: bool,
    ) -> bool:
        """Track one candidate display across the frame pair.

        Returns True when the enrollment basis changed (a readout
        joined or left the signature set)."""
        # The display's glyph can drift inside its panel — a
        # rotating icon keeps its region but not its cells. The
        # same display is whichever current object still covers
        # most of its old footprint; a walker loses overlap.
        pcells = set(pobj.cells)
        prows = [r for r, _ in pcells]
        pcols = [c for _, c in pcells]
        pr0, pr1 = min(prows), max(prows)
        pc0, pc1 = min(pcols), max(pcols)
        best, shared = None, 0
        for cand in cur_objects:
            # Region, not pixels: a redrawn glyph keeps none of
            # its old cells but all of its old rectangle.
            common = sum(
                1 for r, c in cand.cells
                if pr0 <= r <= pr1 and pc0 <= c <= pc1
            )
            if common > shared:
                best, shared = cand, common
        if best is None or shared < max(
            3, min(len(pcells), len(best.cells)) // 2
        ):
            return False  # moved or vanished — a world object
        rec = self._readout_stats.setdefault(key, [0, 0, 0])
        rec[0] += 1
        # Compare the panel's contents over the union box, not
        # object signatures — the same rectangle in both frames
        # is the honest "did the display change" test.
        rows = [r for r, _ in pcells] + [r for r, _ in best.cells]
        cols = [c for _, c in pcells] + [c for _, c in best.cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        # Her own body passing over the box is occlusion, not a
        # redraw — vacating the cells flips them back with no
        # contact, which otherwise reads as a free change. Cells
        # showing her colors in either frame can't testify about
        # the display, so they don't count.
        changed = any(
            prev.at(r, c) != cur.at(r, c)
            and prev.at(r, c) not in self._avatar_colors
            and cur.at(r, c) not in self._avatar_colors
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
        )
        if not changed:
            return False
        if contact:
            rec[1] += 1
        else:
            rec[2] += 1
            if rec[2] >= 2:
                # Redrew on its own schedule — a meter or a
                # detached display, the interface not the world.
                # One flicker isn't proof; two is a pattern.
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        self._ui_cells.add((rr, cc))
        if (
            key not in self._readout_keys
            and rec[0] >= 4
            and rec[1] >= 1
            and rec[2] == 0
        ):
            self._readout_keys.append(key)
            self._readout_keys.sort()
            # Enroll the rectangle, not the glyph: the signature
            # reads a stable region whose contents may move.
            self._readout_cells[key] = frozenset(
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
            )
            return True
        if key in self._readout_keys and rec[2] >= 3:
            # Repeated free changes — a meter after all. One or
            # two are contact-misses on station re-touches.
            self._readout_keys.remove(key)
            self._readout_cells.pop(key, None)
            return True
        return False

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
        body, cur_body, strip, moved = self._contact_map(prev, cur)
        self._record_state_marks(prev, cur, sig, strip, moved)
        for obj in scene.objects:
            self._classify_touch(
                obj, scene, prev, cur, body, cur_body, strip, moved, sig
            )
        self._note_signature_change(sig)

    def _record_state_marks(
        self,
        prev: Grid,
        cur: Grid,
        sig: Any,
        strip: set[tuple[int, int]],
        moved: bool,
    ) -> None:
        """Mark the tiles spent under this signature.

        Spending is positional: whatever lies inside the tile she
        stands on — or reached for and was refused — is consumed
        under this state. A station's offer is spent once used, a
        gate's try is recorded. Component-level keys fragment on
        occlusion (her sprite covers the cells it stands on), but
        an anchor's footprint is exact. Each mark goes under the
        pre-transition signature as well as the current one: a
        display can redraw a frame late, and the tile was spent
        either way.
        """
        if sig is None:
            return
        marks = {sig}
        if self._avatar_sig is not None and self._avatar_sig != sig:
            marks.add(self._avatar_sig)
        after_anchor = self._avatar_anchor(cur)
        if moved and after_anchor is not None:
            for m in marks:
                self._tried_at_state.add(
                    (after_anchor[0], after_anchor[1],
                     after_anchor[2], after_anchor[3], m)
                )
        if not moved and strip:
            r0 = min(r for r, _ in strip)
            c0 = min(c for _, c in strip)
            h = max(r for r, _ in strip) - r0 + 1
            w = max(c for _, c in strip) - c0 + 1
            for m in marks:
                self._tried_at_state.add((r0, c0, h, w, m))
        if self._avatar_sig is not None and self._avatar_sig != sig:
            # The change may be credited a frame late: whatever
            # she last stood on caused it, so that tile is spent
            # under the new state as well.
            pa = self._avatar_anchor(prev)
            if pa is not None:
                self._tried_at_state.add(
                    (pa[0], pa[1], pa[2], pa[3], sig)
                )

    def _classify_touch(
        self,
        obj: Any,
        scene: Any,
        prev: Grid,
        cur: Grid,
        body: set[tuple[int, int]] | frozenset[tuple[int, int]] | None,
        cur_body: set[tuple[int, int]] | frozenset[tuple[int, int]] | None,
        strip: set[tuple[int, int]],
        moved: bool,
        sig: Any,
    ) -> None:
        """Read one contacted object as gate, station, or neither."""
        bg = self._background(cur)
        if obj.color in (0, bg, self.avatar_color):
            return
        if body and set(obj.cells) & body:
            return  # her other bands aren't stations or gates
        if obj.color in self._lethal_colors:
            return
        cells = set(obj.cells)
        # Reached: she is on it, or it was her move's destination
        # (a landed station or an open gate both qualify).
        reached = bool(cur_body and cells & cur_body) or bool(cells & strip)
        # Refused: she tried to enter and stayed put — the gate
        # signal. The floor never refuses her, so walkable
        # terrain can never teach itself locked.
        refused = (not moved) and bool(cells & strip)
        if (
            not moved
            and not strip
            and body
            and min(
                abs(br - r) + abs(bc - c)
                for r, c in cells
                for br, bc in body
            )
            <= 1
        ):
            refused = True
        if not reached and not refused:
            return
        if (
            obj.color in self._open_colors
            and obj.color not in self._station_colors
            and len(obj.cells) > 64
        ):
            # She is standing in this color's cells — a floor
            # cannot gate the body that occupies it. A small
            # decal still can: stations are walked-on by
            # design, so their colors prove open on the first
            # touch while their effect fires on the next.
            return
        self._touched_colors.add(obj.color)
        remaining = sum(
            1 for r, c in obj.cells if cur.at(r, c) == obj.color
        )
        if (
            refused
            and remaining > len(obj.cells) // 2
            and obj.color not in self._open_colors
        ):
            # Refused and it persisted — a gate if it stays that
            # way. Only the cells she reached for accrue blame.
            # A color she has already walked on cannot be the
            # lock — without that exemption, floor cells inside a
            # refused strip get gated and the corridor itself
            # disappears from the planner.
            for cell in cells & strip:
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
            # And read what it asked for: gates that post their
            # requirement show a signature-shaped icon on or
            # beside the refused tile.
            req = self._requirement_token(obj, scene, prev)
            if req is not None:
                self._gate_requirements[self._obj_key(obj)] = req
        # Station: avatar's own signature changed while on it.
        if (
            reached
            and self._avatar_sig is not None
            and sig is not None
            and sig != self._avatar_sig
        ):
            self._station_colors.add(obj.color)
            self._station_effects[obj.color] = self._effect_dims(
                prev, cur
            )

    def _note_signature_change(self, sig: Any) -> None:
        """Bookkeep an observed signature transition and commit it."""
        if sig is not None and self._avatar_sig != sig:
            # Her shown state actually changed — displays and
            # signatures only carry state once they've been seen to.
            self._sig_varies = True
        if (
            self._avatar_sig is not None
            and sig is not None
            and sig != self._avatar_sig
            and (self._gated_cells or self._bumped_gates)
        ):
            # Key state changed — previously locked gates may now
            # open. Clear the lockout so they get retried. Learned
            # terrain stays: only background-colored cells land in
            # _blocked_cells now, and raw void is not a state gate —
            # refusing her twice is what the gate path is for.
            self._gated_cells.clear()
            self._bumped_gates.clear()
            self._touch_fail.clear()
            self._unreachable_goals.clear()
            self._unreach_strikes.clear()
            self._committed_goal = None
            self._plan_target = None
        self._avatar_sig = sig

    @staticmethod
    def _obj_key(obj: Any) -> tuple[int, int]:
        """An object's identity anchor: its top-left bounding corner."""
        return (
            min(r for r, _ in obj.cells),
            min(c for _, c in obj.cells),
        )

    @staticmethod
    def _cells_signature(
        cells: Any, grid: Grid
    ) -> tuple[tuple[int, ...], ...]:
        """A bounding-box pixel pattern over a cell set."""
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        return tuple(
            tuple(grid.at(r, c) for c in range(c0, c1 + 1))
            for r in range(r0, r1 + 1)
        )

    def _obj_signature(
        self, obj: Any, grid: Grid
    ) -> tuple[tuple[int, ...], ...]:
        """An object's bounding-box pixel pattern — its appearance."""
        return self._cells_signature(obj.cells, grid)

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

    def _body_token(
        self, grid: Grid
    ) -> tuple[int, frozenset[tuple[int, int]]] | None:
        """Her own display token: body color + silhouette."""
        cells = self._self_cells(grid)
        if not cells:
            return None
        from collections import Counter

        color = Counter(
            grid.at(r, c) for r, c in cells
        ).most_common(1)[0][0]
        return (color, self._norm_sil(cells))

    def _effect_dims(self, prev: Grid, cur: Grid) -> set[str]:
        """Which signature dimensions a station touch perturbed.

        Her state may be shown on the body, on a detached readout,
        or split across both — so the honest comparison is every
        display's token before and after, unioned. A station that
        recolors the icon in the HUD while the body only rotates
        reports both truths.
        """
        diffs: set[str] = set()
        t0, t1 = self._body_token(prev), self._body_token(cur)
        if t0 is not None and t1 is not None:
            diffs |= self._sig_diffs(t0, t1)
        for key in self._readout_keys:
            cells = self._readout_cells.get(key)
            if not cells:
                continue
            r0 = self._region_icon(cells, prev)
            r1 = self._region_icon(cells, cur)
            if r0 is not None and r1 is not None:
                diffs |= self._sig_diffs(r0, r1)
        if diffs:
            return diffs
        b = self._cells_signature(self._self_cells(prev), prev)
        a = self._cells_signature(self._self_cells(cur), cur)
        if b and a:
            return {self._classify_change(b, a)}
        return {"recolor"}

    # ── Requirement icons — reading what a gate asks for ────
    #
    # A requirement icon is a depiction, not a mechanism: a small
    # sprite on or beside a gate that shows the key state it wants,
    # drawn in the same visual language her own state icon uses.
    # The token form is (color, silhouette) — position-free — so a
    # requirement and her shown state compare dimension by
    # dimension: same silhouette at another rotation is a
    # reorientation, a different silhouette is a different shape.

    @staticmethod
    def _norm_sil(cells: Any) -> frozenset[tuple[int, int]]:
        """A silhouette normalized to its own top-left — position-free."""
        if not cells:
            return frozenset()
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)

    @staticmethod
    def _rot90(sil: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        """Quarter-turn a normalized silhouette within its own box."""
        r1 = max(r for r, _ in sil)
        return frozenset((c, r1 - r) for r, c in sil)

    def _sig_diffs(
        self,
        mine: tuple[int, frozenset[tuple[int, int]]],
        req: tuple[int, frozenset[tuple[int, int]]],
    ) -> set[str]:
        """Which signature dimensions two icon tokens differ on."""
        diffs: set[str] = set()
        if mine[0] != req[0]:
            diffs.add("recolor")
        if mine[1] != req[1]:
            rot = mine[1]
            matched = False
            for _ in range(3):
                rot = self._rot90(rot)
                if rot == req[1]:
                    matched = True
                    break
            diffs.add("reorient" if matched else "reshape")
        return diffs

    def _region_icon(
        self, rect: Any, grid: Grid
    ) -> tuple[int, frozenset[tuple[int, int]]] | None:
        """The icon inside a framed region — content minus its frame.

        A HUD panel is a border box with a glyph inside: the frame
        color owns the region's perimeter majority, the glyph is
        what's left. Returns (color, silhouette) or None when the
        region holds no distinct inner figure.
        """
        from collections import Counter

        colors = Counter(grid.at(r, c) for r, c in rect)
        if len(colors) < 2:
            return None
        frame_c, _n = colors.most_common(1)[0]
        inner = [(r, c) for r, c in rect if grid.at(r, c) != frame_c]
        if not inner:
            return None
        icon_c, _n = Counter(
            grid.at(r, c) for r, c in inner
        ).most_common(1)[0]
        cells = {(r, c) for r, c in inner if grid.at(r, c) == icon_c}
        return (icon_c, self._norm_sil(cells))

    def _self_display_tokens(
        self, grid: Grid
    ) -> list[tuple[int, frozenset[tuple[int, int]]]]:
        """Every candidate shown-state token, best source first.

        Enrolled readouts are the honest displays — they only
        redraw when her state does. Each contributes one candidate:
        which of them *is* the signature display isn't declared, so
        the requirement comparison picks the candidate that shares
        the icon language (fewest differing dimensions). Her body
        is a candidate too, but only once her state has been *seen*
        to vary: a body that can't change shape can't be the
        display, and a constant token would manufacture constant
        false requirements.
        """
        toks: list[tuple[int, frozenset[tuple[int, int]]]] = []
        for key in self._readout_keys:
            cells = self._readout_cells.get(key)
            if not cells:
                continue
            tok = self._region_icon(cells, grid)
            if tok is not None and tok not in toks:
                toks.append(tok)
        if self._sig_varies:
            tok = self._body_token(grid)
            if tok is not None and tok not in toks:
                toks.append(tok)
        return toks

    @staticmethod
    def _icon_like(sil: frozenset[tuple[int, int]]) -> bool:
        """Whether a silhouette looks like a depiction, not a tile.

        Icons have structure — notches, holes, an outline that isn't
        its own bounding box. A solid rectangle is terrain-shaped,
        not information-shaped.
        """
        if not sil:
            return False
        h = max(r for r, _ in sil) + 1
        w = max(c for _, c in sil) + 1
        return len(sil) < h * w

    @staticmethod
    def _nested_icon(obj: Any, scene: Any) -> Any | None:
        """The smaller icon strictly inside this object, if any —
        the display structure of a posted requirement.

        A display is tight around what it shows: the container's
        bounding box bounds the icon's a few times over at most.
        The floor an icon sits on is not a panel — without the
        bound, the whole walkable region "displays" every glyph
        it holds and outranks the actual signage.
        """
        if len(obj.cells) < 9:
            return None
        r0 = min(r for r, _ in obj.cells)
        r1 = max(r for r, _ in obj.cells)
        c0 = min(c for _, c in obj.cells)
        c1 = max(c for _, c in obj.cells)
        panel = (r1 - r0 + 1) * (c1 - c0 + 1)
        inside = {
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
        } - set(obj.cells)
        for o in scene.objects:
            if o is obj or not o.cells or len(o.cells) >= len(obj.cells):
                continue
            if not set(o.cells) <= inside:
                continue
            ih = max(r for r, _ in o.cells) - min(r for r, _ in o.cells) + 1
            iw = max(c for _, c in o.cells) - min(c for _, c in o.cells) + 1
            if panel <= 64 * ih * iw:
                return o
        return None

    @classmethod
    def _has_nested_icon(cls, obj: Any, scene: Any) -> bool:
        return cls._nested_icon(obj, scene) is not None

    def _requirement_token(
        self, gate: Any, scene: Any, grid: Grid
    ) -> tuple[int, frozenset[tuple[int, int]]] | None:
        """The signature a gate posts as required — read, not trialed.

        Candidates are the gate itself and any object inside its
        bounding box or pressed against it (the posted icon sits on
        the door, not across the room). Among icon-shaped candidates
        the one whose token differs from her shown state on the
        fewest dimensions is the requirement — a requirement icon
        exists to be compared against. Plain rectangular candidates
        are considered only when nothing icon-shaped is near, since
        a bar-shaped requirement is itself a bar.
        """
        if not gate.cells:
            return None
        r0 = min(r for r, _ in gate.cells) - 1
        r1 = max(r for r, _ in gate.cells) + 1
        c0 = min(c for _, c in gate.cells) - 1
        c1 = max(c for _, c in gate.cells) + 1
        body = self._self_cells(grid)
        cands = [gate]
        for o in scene.objects:
            if o is gate or not o.cells:
                continue
            if body and set(o.cells) & body:
                continue
            if all(
                r0 <= r <= r1 and c0 <= c <= c1 for r, c in o.cells
            ):
                cands.append(o)
        mines = self._self_display_tokens(grid)
        if not mines:
            return None
        iconic = [
            o
            for o in cands
            if self._icon_like(self._norm_sil(o.cells))
        ]
        pool = iconic or cands
        best, best_n = None, 99
        for o in pool:
            tok = (o.color, self._norm_sil(o.cells))
            n = min(len(self._sig_diffs(m, tok)) for m in mines)
            if n < best_n:
                best, best_n = tok, n
        return best

    def _mark_lethal(self, at: tuple[float, float], grid: Grid) -> None:
        """Blame whatever plausibly killed her at a death spot.

        Only two things can: an autonomous mover that reached her,
        or an object her move pressed into this step. A quiet death
        with neither is a spent resource — a step budget running
        out — and blaming the scenery would poison floor, gates,
        and stations as 'lethal' for a threat that isn't there.
        """
        body = self._self_cells(grid)
        strip = self._strip_cells(grid)
        blamed = False
        for obj in perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        ).objects:
            if obj.color == self.avatar_color:
                continue
            if body and set(obj.cells) & body:
                continue
            if min(
                abs(at[0] - r) + abs(at[1] - c) for r, c in obj.cells
            ) > 1:
                continue
            mover = obj.color in self._hazard_velocity or self._solo_moves.get(
                obj.color, 0
            ) >= 1 or any(
                t.color == obj.color
                and t.moved_frames > 0
                and set(t.cells) & set(obj.cells)
                for t in self._tracks.values()
            )
            contacted = bool(set(obj.cells) & strip)
            if not (mover or contacted):
                continue
            self._hazard_colors.add(obj.color)
            self._lethal_colors.add(obj.color)
            self._valuable_colors.discard(obj.color)
            self._station_colors.discard(obj.color)
            blamed = True
        clock = len(self._visited_cells)
        if not blamed:
            # Unwitnessed — either the budget ran out, or tracking
            # flickered onto a same-shaped icon for a frame. A life
            # can't be spent twice inside a few steps; detections
            # closer than that are the flicker, not deaths.
            if clock - self._last_death_at < 10:
                return
            self._spent_deaths += 1
        self._last_death_at = clock
        # Encouragement: losing a life isn't the end — a short burst
        # of curiosity follows so it tries a different path.
        self._grit_steps = 40

    def _strip_cells(self, grid: Grid) -> set[tuple[int, int]]:
        """The footprint her last action tried to enter."""
        anchor = self._avatar_anchor(grid)
        st = self.stats.get(self._last_action)
        step = self._action_step(st) if st is not None and st.moves else None
        if anchor is None or step is None or not (step[0] or step[1]):
            return set()
        r0, c0, h, w = anchor
        dr, dc = step
        return {
            (r, c)
            for r in range(r0 + dr, r0 + dr + h)
            for c in range(c0 + dc, c0 + dc + w)
        }

    # ── Goals ─────────────────────────────────────────────────

    def _avatar_pos(self, grid: Grid) -> tuple[float, float] | None:
        obj = self._avatar_obj(grid)
        return obj.centroid if obj is not None else None

    def _avatar_obj(self, grid: Grid) -> Any | None:
        """The avatar's *object* — bound by shape, then proximity.

        Color names the body but can't individuate it: walls can
        wear her color. What a wall can't copy is her *shape under
        translation* — a rigid body displaced is the same pixel
        pattern elsewhere. Among same-colored candidates prefer the
        object whose cells equal the last silhouette shifted; only
        then fall back to plain proximity.
        """
        if self.avatar_color is None:
            return None
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        cands = [o for o in scene.objects if o.color == self.avatar_color]
        if not cands:
            adopted = self._adopted_self(scene, grid)
            if adopted is not None:
                return adopted
            return None
        if self._avatar_last is not None and self._avatar_shape:
            last = self._avatar_last
            shape = self._avatar_shape
            sr = sum(r for r, _ in shape) / len(shape)
            sc = sum(c for _, c in shape) / len(shape)
            best, best_iou, best_d = None, 0.0, float("inf")
            for o in cands:
                # Integer translation aligning the remembered
                # silhouette onto this candidate — a rigid body's
                # own cells land exactly; a same-colored wall's don't.
                dr = round(o.centroid[0] - sr)
                dc = round(o.centroid[1] - sc)
                moved = {(r + dr, c + dc) for r, c in shape}
                inter = len(moved & set(o.cells))
                union = len(moved | set(o.cells))
                iou = inter / union if union else 0.0
                # Twins happen — a requirement icon can depict her
                # own silhouette in her own color. Shape can't
                # individuate identical sprites; continuity can: she
                # moves a stride at a time, so between equal matches
                # the one nearest where she was is her.
                d = (
                    abs(o.centroid[0] - last[0])
                    + abs(o.centroid[1] - last[1])
                )
                if iou > best_iou or (
                    iou == best_iou and d < best_d
                ):
                    best, best_iou, best_d = o, iou, d
            if best is not None and best_iou >= 0.5:
                # A match beyond her stride can't be continuity —
                # more likely her recolored body sitting nearby
                # while an icon inherited her old colors.
                if best_d <= 3.0:
                    return best
                adopted = self._adopted_self(scene, grid)
                return adopted if adopted is not None else best
            return min(
                cands,
                key=lambda o: abs(o.centroid[0] - last[0])
                + abs(o.centroid[1] - last[1]),
            )
        return max(cands, key=lambda o: len(o.cells))

    def _adopted_self(self, scene: Any, grid: Grid) -> Any | None:
        """Her body when appearance fails: the thing still at her seat.

        Her color and silhouette are not her — a station rewrites
        them in place. When the remembered colors find nothing (or
        only a discontinuous icon), the body is the object still
        sitting where she was: a transformed sprite keeps its seat
        and roughly its footprint, while a killer or wall standing
        on her spot doesn't match her size. Adopting it updates the
        tracked color and shape so following frames see the new
        skin as hers.
        """
        if self._avatar_last is None:
            return None
        n = len(self._avatar_shape)
        last = self._avatar_last
        last_cells = set(self._avatar_shape)
        readout_cells = {
            cell
            for key in self._readout_keys
            for cell in self._readout_cells.get(key, frozenset())
        }

        def occupies(obj: Any) -> bool:
            """Still seated where she was — not a neighbor claiming
            her spot. A transform redraws her on the same cells (or
            one step away when she just moved onto the station); a
            killer stands *beside* the empty footprint."""
            drift = (
                abs(obj.centroid[0] - last[0])
                + abs(obj.centroid[1] - last[1])
            )
            if drift <= 1.0:
                return True
            return (
                bool(last_cells)
                and len(set(obj.cells) & last_cells) * 3
                >= len(obj.cells)
            )

        adopted = min(
            (
                o
                for o in scene.objects
                if o.color not in (0, self._background(grid))
                and (n == 0 or len(o.cells) <= 2 * n)
                and len(o.cells) * 2 >= n
                and not set(o.cells) & readout_cells
                and occupies(o)
            ),
            key=lambda o: abs(o.centroid[0] - last[0])
            + abs(o.centroid[1] - last[1]),
            default=None,
        )
        if adopted is not None:
            self.avatar_color = adopted.color
            self._avatar_last = adopted.centroid
            self._avatar_shape = adopted.cells
            return adopted
        return None

    def _body_parts(self, grid: Grid) -> list[Any]:
        """Every object composing her body this frame.

        The primary band plus co-moving-color objects *adjacent* to
        it — a second band sits glued to the first. Membership is
        object-level, not color-level: a remote pickup wearing a
        body color is still the world's object, not her flesh.
        """
        prim = self._avatar_obj(grid)
        if prim is None:
            return []
        if not self._co_colors:
            return [prim]
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        prim_cells = set(prim.cells)
        parts = [prim]
        for o in scene.objects:
            if o.color not in self._co_colors:
                continue
            if any(
                (r + dr, c + dc) in prim_cells
                for r, c in o.cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                parts.append(o)
        return parts

    def _self_cells(self, grid: Grid) -> set[tuple[int, int]]:
        """All cells her body occupies — every band, not one color."""
        return {
            cell for part in self._body_parts(grid) for cell in part.cells
        }

    def _avatar_anchor(
        self, grid: Grid
    ) -> tuple[int, int, int, int] | None:
        """Avatar footprint: ``(r0, c0, h, w)`` bounding box, or None.

        Multi-cell avatars move by their own size — the planner needs
        the whole footprint, not a point, to judge passable space.
        """
        cells = self._self_cells(grid)
        if not cells:
            return None
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        return (r0, c0, r1 - r0 + 1, c1 - c0 + 1)

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

    def _pick_goal(
        self,
        grid: Grid,
        reach: dict[tuple[int, int], int] | None = None,
    ) -> tuple[float, float] | None:
        """Nearest *reachable* non-avatar object not yet reached.

        ``reach`` is the BFS distance map over passable anchors.
        When given, objects no reachable anchor can touch — sealed
        masses, unreachable readouts — sort last instead of burning
        the commitment slot and the probation marks cycle after
        cycle.
        """
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        pos = self._avatar_pos(grid)
        cur_sig = self._avatar_signature(grid)
        body = self._self_cells(grid)
        bg = self._background(grid)

        # Read the board: gates that post their requirement show a
        # signature-shaped icon — on the refused tile or beside it.
        # Comparing it against her own shown state splits the lock
        # into dimensions (color, shape, orientation), which names
        # the station class worth the walk — and a requirement her
        # shown state already satisfies predicts an open gate
        # before any success has been recorded.
        needed_dims = self._read_gate_requirements(scene, grid, body, bg)
        view = _GoalView(self, grid, body, cur_sig)

        if _DBG:
            view.dump_drops(scene)

        goals = view.frontier(scene)
        if goals is None:
            return None
        if pos is None:
            self._goal_cells = frozenset(goals[0].cells)
            return goals[0].centroid
        candidates = [g for g in goals if not view.spent(g)] or goals
        if _DBG and self._committed_goal is not None:
            view.dump_commit(scene, self._committed_goal, reach)

        tiers = self._goal_tiers(
            candidates, needed_dims, cur_sig, grid, scene, view.inside,
            view.openable_now,
        )

        candidates, tiers, pool, costs, sealed_ids, cand_objs = (
            self._reachable_goals(candidates, tiers, grid, reach)
        )
        # Commitment: the object she set out for stays the target
        # while it remains a plausible goal *at all* — re-deciding
        # "nearest" every frame, or dropping the trek because the
        # tier ladder shifted under her, ping-pongs between distant
        # goals and burns the move budget on transit, not progress.
        # Tiers arbitrate fresh choices; they don't commute a walk
        # already half-finished.
        held = self._hold_commitment(pool, cand_objs, sealed_ids, body)
        if held is not None:
            return held
        if not pool:
            if _DBG:
                _dbg(
                    f"pool-empty goals={len(goals)} "
                    f"cand={len(candidates)} sealed={len(sealed_ids)}"
                )
            self._committed_goal = None
            self._goal_cells = frozenset()
            return None
        if _DBG:
            def _c(objs: list[Any]) -> str:
                return ",".join(
                    f"c{o.color}@{round(o.centroid[0])},{round(o.centroid[1])}"
                    for o in objs[:4]
                )
            _dbg(
                "tiers "
                + " ".join(
                    f"{name}={_c(tier)}"
                    for name, tier in zip(
                        (
                            "open", "retried", "needed", "val",
                            "sta", "icon", "unt",
                        ),
                        tiers,
                        strict=True,
                    )
                )
                + f" cand={_c(candidates)}"
            )
        chosen = min(
            pool,
            key=lambda g: (
                costs.get(id(g)) is None,
                costs.get(id(g)) or 0.0,
                abs(g.centroid[0] - pos[0])
                + abs(g.centroid[1] - pos[1]),
            ),
        )
        self._committed_goal = (
            chosen.color,
            frozenset(chosen.cells),
        )
        self._goal_cells = frozenset(chosen.cells)
        self._commit_missing = 0
        return chosen.centroid

    def _read_gate_requirements(
        self,
        scene: Any,
        grid: Grid,
        body: set[tuple[int, int]] | frozenset[tuple[int, int]] | None,
        bg: int,
    ) -> set[str]:
        """Mine the board for posted gate requirements; return the
        dimensions my current signature is still missing."""
        self._req_diffs: dict[tuple[int, int], set[str]] = {}
        needed_dims: set[str] = set()
        mine_toks = self._self_display_tokens(grid)
        if not mine_toks:
            return needed_dims
        gate_spots = {cell[:2] for cell in self._gated_cells} | {
            (r, c) for r, c, _a in self._bumped_gates
        }
        for o in scene.objects:
            if not o.cells:
                continue
            if o.color in self._avatar_colors or o.color == bg:
                continue
            if body and set(o.cells) & body:
                continue
            key = self._obj_key(o)
            req = self._gate_requirements.get(key)
            if req is None and (
                key in gate_spots
                or any(cell in self._gated_cells for cell in o.cells)
                or any(
                    (r, c) in gate_spots for r, c in o.cells
                )
                or self._has_nested_icon(o, scene)
            ):
                req = self._requirement_token(o, scene, grid)
            if req is None:
                continue
            self._gate_requirements[key] = req
            d = min(
                (self._sig_diffs(m, req) for m in mine_toks),
                key=len,
            )
            self._req_diffs[key] = d
            needed_dims |= d
        return needed_dims

    def _goal_tiers(
        self,
        candidates: list[Any],
        needed_dims: set[str],
        cur_sig: Any,
        grid: Grid,
        scene: Any,
        inside: set[tuple[int, int]],
        openable_now: Any,
    ) -> list[list[Any]]:
        """Priority tiers over the surviving candidates."""
        # Highest priority: gates empirically known to open for the
        # avatar's current key state.
        openable = [g for g in candidates if openable_now(g)]
        # Next: a gate that refused a *different* key state. It is
        # the reason a state change was sought — once the signature
        # turns, retrying it outranks touching the station again.
        # A gate that already rejected this exact signature stays
        # out of the pool until the state changes once more.
        def refused_elsewhere(g: Any) -> bool:
            denied = self._gate_blocked.get(
                self._obj_signature(g, grid)
            )
            return bool(denied) and cur_sig not in (denied or set())

        retried = [g for g in candidates if refused_elsewhere(g)]
        # Requirement icons name the dimensions still missing — a
        # station whose observed effect covers one is the direct
        # route. It outranks generic pickups: in games where stations
        # are consumed on touch every station is also valuable, and
        # letting "valuable" run first would send her to the nearest
        # consumable instead of the one that unlocks the gate.
        needed = (
            [
                g
                for g in candidates
                if g.color in self._station_colors
                and self._station_effects.get(g.color, set()) & needed_dims
            ]
            if needed_dims
            else []
        )
        # Known pickups next — objects of colors empirically observed
        # to be consumable (they vanished on touch). Survival-critical
        # in games with step budgets.
        valuable = [
            g for g in candidates if g.color in self._valuable_colors
        ]
        # When gated goals exist, stations that change its key state
        # are the way forward — visit them before re-trying gates.
        # A recorded refusal (not yet a hard gate) counts: waiting
        # for the second bump before seeking the key is a wasted trip.
        stations = (
            [
                g
                for g in candidates
                if g.color in self._station_colors
                and not (
                    self._station_effects.get(g.color, set()) & needed_dims
                )
            ]
            if self._gated_cells or self._gate_blocked
            else []
        )
        # Signage: an object displaying a smaller glyph inside its
        # bounds is advertising a state — goals post their
        # requirement this way, stations their effect. It reads as
        # meaningful at sight, before any touch is spent learning
        # that bare scenery teaches nothing. A display *panel* is
        # the same structure pointed inward: when the nested glyph
        # is part of the interface, the container is furniture.
        iconed = [
            g
            for g in candidates
            if (ic := self._nested_icon(g, scene)) is not None
            and not set(ic.cells) <= inside
        ]
        # The hint: object colors never touched can't teach it
        # anything until contact — prefer them when nothing is
        # proven-openable or known-valuable.
        untouched = [
            g for g in candidates if g.color not in self._touched_colors
        ]
        return [
            openable, retried, needed, valuable, stations, iconed,
            untouched,
        ]

    def _reachable_goals(
        self,
        candidates: list[Any],
        tiers: list[list[Any]],
        grid: Grid,
        reach: dict[tuple[int, int], int] | None,
    ) -> tuple[
        list[Any],
        list[list[Any]],
        list[Any],
        dict[int, float | None],
        set[int],
        list[Any],
    ]:
        """Reachability and sealed-evidence pass.

        An object is committable only when some passable anchor can
        touch one of its cells. Without the map (early frames, no
        anchor yet) everything counts as reachable. Proven-sealed
        objects — every approach already refused *and the refusal
        taught nothing* — drop unconditionally: contact alone can't
        tell a sealed box from a goal; the refusal ledger can.
        Returns the filtered candidates, filtered tiers, the
        re-picked pool, the cost map, the sealed id set, and the
        pre-filter candidate list for commitment checks.
        """
        fp = self._avatar_anchor(grid)
        fh, fw = (fp[2], fp[3]) if fp is not None else (1, 1)
        reach_map = reach or {}

        def reach_cost(obj: Any) -> float | None:
            if reach is None or not reach_map:
                return 0.0
            cells = set(obj.cells)
            best: float | None = None
            for r0, c0 in reach_map:
                if any(
                    (rr, cc) in cells
                    for rr in range(r0 - 1, r0 + fh + 1)
                    for cc in range(c0 - 1, c0 + fw + 1)
                ) and self._live_entry(r0, c0, cells, fh, fw):
                    d = reach_map[(r0, c0)]
                    if best is None or d < best:
                        best = d
            return best

        costs: dict[int, float | None] = {
            id(g): reach_cost(g) for g in candidates
        }
        anything_reachable = any(
            c is not None for c in costs.values()
        )
        # A gate whose refusal posted a requirement stays alive —
        # the station tiers own it.
        sealed_ids = {
            id(g)
            for g in candidates
            if self._sealed(g, reach_map, fh, fw)
        }

        def alive(g: Any) -> bool:
            return id(g) not in sealed_ids and (
                not anything_reachable
                or costs.get(id(g)) is not None
            )

        # When anything at all is reachable, unreachable candidates
        # drop out of every tier — committing to one only restarts
        # the mark-and-repick cycle. Proven-sealed candidates drop
        # unconditionally: a dead end is never a goal.
        cand_objs = list(candidates)
        pool = next((t for t in tiers if t), candidates)
        if anything_reachable or sealed_ids:
            tiers = [[g for g in tier if alive(g)] for tier in tiers]
            candidates = [g for g in candidates if alive(g)]
            pool = next((t for t in tiers if t), candidates)
        return candidates, tiers, pool, costs, sealed_ids, cand_objs

    def _hold_commitment(
        self,
        pool: list[Any],
        cand_objs: list[Any],
        sealed_ids: set[int],
        body: set[tuple[int, int]] | frozenset[tuple[int, int]] | None,
    ) -> tuple[float, float] | None:
        """Reconcile the committed goal against this frame's pool.

        Returns the remembered or refreshed centroid while the trek
        still holds; ``None`` releases the pick to a fresh choice.
        """
        if self._committed_goal is None:
            return None
        col, cells = self._committed_goal
        if any(
            id(g) in sealed_ids
            and g.color == col
            and set(g.cells) & cells
            for g in cand_objs
        ):
            # Proven sealed — every approach already refused and
            # the refusal posted no requirement. A dead end gets
            # no flicker grace.
            self._committed_goal = None
            self._goal_cells = frozenset()
            self._commit_missing = 0
            return None
        # The trek holds only inside the active tier's pool:
        # a locked gate that posted a requirement yields to
        # the station that unlocks it — *that* is the walk,
        # not the gate it can't yet enter — and the retried
        # tier brings the gate back once the state turns.
        for g in pool:
            if g.color == col and set(g.cells) & cells:
                # Refresh the remembered silhouette — her
                # cover and re-segmentation reshape it frame
                # to frame. Fully occluded keeps the last
                # sighting.
                if g.cells:
                    self._committed_goal = (
                        col,
                        frozenset(g.cells),
                    )
                self._goal_cells = frozenset(g.cells)
                self._commit_missing = 0
                return g.centroid
        if any(
            g.color == col and set(g.cells) & cells
            for g in cand_objs
        ):
            # Perceived but out-tiered — a real demotion,
            # not a flicker. The remembered cells stay in
            # the requirement ledger for the retry.
            self._committed_goal = None
            self._goal_cells = frozenset()
            self._commit_missing = 0
        elif body and self._goal_cells and set(body) & set(
            self._goal_cells
        ):
            # Standing where it was and nothing answered —
            # the target is spent, not flickering.
            self._committed_goal = None
            self._goal_cells = frozenset()
            self._commit_missing = 0
        elif self._goal_cells and self._commit_missing < 8:
            # Not perceived this frame — occlusion and icon
            # redraws split a blob for a step. The remembered
            # footprint keeps pointing the plan at it.
            self._commit_missing += 1
            rs = [r for r, _ in self._goal_cells]
            cs = [c for _, c in self._goal_cells]
            return (
                sum(rs) / len(rs),
                sum(cs) / len(cs),
            )
        else:
            self._committed_goal = None
            self._commit_missing = 0
        return None

    # ── Deliberation: pathing over learned terrain ────────────

    @staticmethod
    def _action_step(st: ActionStats) -> tuple[int, int] | None:
        """The action's stride: the modal rounded displacement.

        Means lie under contamination — one respawn teleport or a
        confused track shifts the average enough to fail the clean-
        move gate, leaving an action with no vector forever. The mode
        keeps the true stride unless the noise itself outnumbers the
        moves.
        """
        if st.delta_counts:
            (dr, dc), _n = max(
                st.delta_counts.items(), key=lambda kv: kv[1]
            )
        elif st.moves:
            # No per-transition record (seeded or migrated stats) —
            # the mean is all there is.
            dr, dc = round(st.dr / st.moves), round(st.dc / st.moves)
        else:
            return None
        if (dr, dc) == (0, 0):
            return None
        return dr, dc

    def _step_vectors(
        self, action_space: list[Any]
    ) -> dict[Any, tuple[int, int]]:
        """Each action's learned move, as integer step vectors.

        The stride is the modal displacement — a single teleported
        frame can't shift a mode the way it shifts a mean.
        """
        out: dict[Any, tuple[int, int]] = {}
        for a in action_space:
            st = self.stats.get(a)
            if st is None or st.moves == 0:
                continue
            step = self._action_step(st)
            if step is not None:
                out[a] = step
        return out

    def _hard_cells(self, grid: Grid) -> set[tuple[int, int]]:
        """Cells impassable this frame: learned terrain + live hazards."""
        hard = set(self._blocked_cells)
        if self._gated_cells:
            # A gate marked before its color was proven open is
            # stale — she has since walked on cells of that color.
            open_now = self._open_colors
            for r, c in self._gated_cells:
                if 0 <= r < grid.height and 0 <= c < grid.width:
                    if grid.at(r, c) not in open_now:
                        hard.add((r, c))
                else:
                    hard.add((r, c))
        threats = (
            self._hazard_colors | self._lethal_colors
        ) - self._open_colors
        if threats:
            scene = perceive(
                grid, background=self._background(grid),
                compute_relations=False,
            )
            # A threat color covering a large share of the board is
            # floor or UI, never a dodgeable hazard — marking it
            # impassable locks the whole map for the planner.
            max_threat = grid.height * grid.width * 0.25
            for obj in scene.objects:
                if obj.color not in threats:
                    continue
                if len(obj.cells) > max_threat:
                    continue
                hard.update(obj.cells)
                v = self._hazard_velocity.get(obj.color)
                if v is not None:
                    # Hazards are ballistic — their next footprint is
                    # as dangerous as their current one.
                    hard.update(
                        (r + v[0], c + v[1]) for r, c in obj.cells
                    )
        return hard

    def _reachable_map(
        self, grid: Grid, action_space: list[Any]
    ) -> tuple[
        dict[tuple[int, int], int],
        dict[tuple[int, int], tuple[tuple[int, int] | None, Any]],
    ]:
        """(dist, came) over every passable anchor — one BFS.

        The whole reachable map, shared by pathing and by goal
        ranking: an object is committable only when some anchor in
        the map can touch it, so sealed scenery stops cycling
        through the goal slot instead of being marked unreachable
        over and over.
        """
        from collections import deque

        anchor = self._avatar_anchor(grid)
        if anchor is None:
            return {}, {}
        steps = self._step_vectors(action_space)
        if not steps:
            return {}, {}
        hard = self._hard_cells(grid)
        h, w = anchor[2], anchor[3]
        if _DBG:
            _dbg(
                f"reach hard={len(hard)} "
                f"blocked={len(self._blocked_cells)} "
                f"gated={len(self._gated_cells)} "
                f"refused={len(self._bumped_moves | self._bumped_gates)} "
                f"anchor={anchor[:2]} fwh={h}x{w} steps={steps}"
            )

        def passable(r: int, c: int) -> bool:
            if r < 0 or c < 0 or r + h > grid.height or c + w > grid.width:
                return False
            for rr in range(r, r + h):
                for cc in range(c, c + w):
                    if (rr, cc) in hard:
                        return False
            return True

        start = (anchor[0], anchor[1])
        refused = self._bumped_moves | self._bumped_gates
        came: dict[tuple[int, int], tuple[tuple[int, int] | None, Any]] = {
            start: (None, None)
        }
        dist: dict[tuple[int, int], int] = {start: 0}
        q = deque([start])
        while q:
            node = q.popleft()
            for a, (dr, dc) in steps.items():
                nxt = (node[0] + dr, node[1] + dc)
                if (
                    nxt in came
                    or (node[0], node[1], a) in refused
                    or not passable(*nxt)
                ):
                    continue
                came[nxt] = (node, a)
                dist[nxt] = dist[node] + 1
                q.append(nxt)
        if _DBG:
            _dbg(f"reach -> dist={len(dist)}")
        return dist, came

    def _sealed(
        self,
        obj: Any,
        reach_map: dict[tuple[int, int], int],
        fh: int,
        fw: int,
    ) -> bool:
        """Every approach already refused, and the refusal was silent.

        Contact is not reachability: a sealed box's outer wall is
        walkable, so the distance map alone keeps committing her to
        dead scenery — she orbits the perimeter bumping the same mass
        forever. An object is sealed when every reachable anchor that
        touches it has already had every overlapping-destination
        action refused, and none of those refusals posted a
        requirement. A gate that answered with a requirement is not
        sealed — it is a lock awaiting the matching state, and the
        station tiers own it. The verdict is computed per frame, so a
        signature change that reopens the refusal ledger unseals the
        object on its own.
        """
        if not reach_map or not obj.cells:
            return False
        if self._obj_key(obj) in self._gate_requirements:
            return False
        if obj.color in self._open_colors:
            # Walk-over ground can't hold a door shut: a refused
            # destination box blames every cell inside it, so a small
            # decal sharing a refused strip with a wall would take
            # the wall's verdict. Her body already proved these cells
            # never refuse — the blame belongs elsewhere in the box.
            return False
        cells = set(obj.cells)
        contact = False
        for r0, c0 in reach_map:
            if not any(
                (rr, cc) in cells
                for rr in range(r0 - 1, r0 + fh + 1)
                for cc in range(c0 - 1, c0 + fw + 1)
            ):
                continue
            contact = True
            if self._live_entry(r0, c0, cells, fh, fw):
                return False
        return contact

    def _live_entry(
        self,
        r0: int,
        c0: int,
        cells: set,
        fh: int,
        fw: int,
    ) -> bool:
        """An unrefused action whose destination box overlaps the cells.

        Reachability that can't act isn't reachability: an anchor in
        contact range whose every overlapping-destination step was
        already refused is a spent approach, not a path. One live
        entry is a real approach — a refusal teaches or opens it.
        """
        refused = self._bumped_moves | self._bumped_gates
        return any(
            (r0, c0, act) not in refused
            and any(
                (rr, cc) in cells
                for rr in range(r0 + dr, r0 + dr + fh)
                for cc in range(c0 + dc, c0 + dc + fw)
            )
            for act, (dr, dc) in self._step_vectors(
                list(self.stats)
            ).items()
        )

    def _plan(
        self,
        grid: Grid,
        action_space: list[Any],
        goal: tuple[float, float] | None,
        reach: tuple[
            dict[tuple[int, int], int],
            dict[tuple[int, int], tuple[tuple[int, int] | None, Any]],
        ] | None = None,
    ) -> Any | None:
        """First step of the shortest path to the goal — or frontier.

        Reaching means contact: footprint adjacent to any of the
        goal object's cells. If the goal is unreachable — or none
        is known — the target becomes the nearest unvisited passable
        anchor: frontier exploration, the maze-covering default
        instead of wall-chasing.
        """
        anchor = self._avatar_anchor(grid)
        if anchor is None:
            return None
        h, w = anchor[2], anchor[3]
        dist, came = (
            reach
            if reach is not None
            else self._reachable_map(grid, action_space)
        )
        if not came:
            return None
        start = (anchor[0], anchor[1])
        goal_cells = self._goal_cells
        goal_cell = (
            (round(goal[0]), round(goal[1])) if goal is not None else None
        )
        if goal_cell is not None and not goal_cells:
            goal_cells = frozenset({goal_cell})

        def reaches_goal(r: int, c: int) -> bool:
            # Reaching means a live approach: an anchor in contact
            # range — footprint dilated one cell — that still holds an
            # unrefused step whose destination overlaps the object.
            # A contact anchor with every entry refused is a spent
            # approach; routing to it restarts the grind.
            if goal_cell is None:
                return False
            for rr in range(r - 1, r + h + 1):
                for cc in range(c - 1, c + w + 1):
                    if (rr, cc) in goal_cells:
                        return self._live_entry(
                            r, c, set(goal_cells), h, w
                        )
            return False

        if reaches_goal(*start):
            # Already in contact range — the plan is done. Yielding
            # lets the contact gradient issue the entering step;
            # pathing to a different touching anchor would orbit.
            return None

        target, goal_hit = self._select_plan_target(
            came, dist, start, reaches_goal
        )
        if goal_cell is not None and goal_hit is None and len(dist) > 8:
            self._mark_unreachable(goal_cell)
        if target is None or target == start:
            return None
        while (parent := came[target][0]) != start:
            if parent is None:
                return None
            target = parent
        return came[target][1]

    def _select_plan_target(
        self,
        came: dict[tuple[int, int], tuple[tuple[int, int] | None, Any]],
        dist: dict[tuple[int, int], int],
        start: tuple[int, int],
        reaches_goal: Any,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """Choose this frame's route target; return (target, goal_hit).

        ``came`` inserts in BFS order — the first touching anchor
        found is the nearest plan."""
        frontier_hit: tuple[int, int] | None = None
        goal_hit: tuple[int, int] | None = None
        for node in came:
            if node == start:
                continue
            if reaches_goal(*node):
                goal_hit = node
                break
            if (
                frontier_hit is None
                and node not in self._visited_cells
            ):
                frontier_hit = node
        # Plan commitment: a route target survives the frame it was
        # chosen on while it stays valid — a live goal approach, or
        # a still-unvisited frontier. Re-deciding nearest every frame
        # ping-pongs between two midpoints: each anchor sees closer
        # unexplored space through the other and reverses the move
        # it just made. A large object offers contact anchors on
        # both sides of her — the same flip, one goal deep. But a
        # stale frontier never outranks a live approach for the
        # goal she is actually committed to now.
        prior = self._plan_target
        target: tuple[int, int] | None
        if goal_hit is not None:
            target = (
                prior
                if (
                    prior is not None
                    and prior in dist
                    and prior != start
                    and reaches_goal(*prior)
                )
                else goal_hit
            )
        elif (
            prior is not None
            and prior in dist
            and prior != start
            and prior not in self._visited_cells
        ):
            target = prior
        else:
            target = frontier_hit
        self._plan_target = target if target != start else None
        return target, goal_hit

    def _mark_unreachable(self, goal_cell: tuple[int, int]) -> None:
        """The whole passable space known so far was searched —
        probation, not a verdict: the mark records how much map
        was known, and new territory reopens the question.
        Strikes survive the retry: the same neighbourhood
        re-earning the verdict is accumulating proof."""
        strike = next(
            (
                cell
                for cell in self._unreach_strikes
                if abs(cell[0] - goal_cell[0]) <= 5
                and abs(cell[1] - goal_cell[1]) <= 5
            ),
            goal_cell,
        )
        self._unreach_strikes[strike] = (
            self._unreach_strikes.get(strike, 0) + 1
        )
        self._unreachable_goals[strike] = len(self._visited_cells)

    def _navigate(
        self, grid: Grid, action_space: list[Any]
    ) -> Any | None:
        """Choose the action whose learned effect closes distance to goal.

        Costs add a hazard penalty: landing within ~2 cells of a known
        moving-hazard color is expensive. It doesn't just avoid making
        hazards its destination — it avoids walking through them.
        """
        pos = self._avatar_pos(grid)
        if pos is None:
            return None
        reach = self._reachable_map(grid, action_space)
        goal = self._pick_goal(grid, reach[0])
        anchor = self._avatar_anchor(grid)
        # An action that already bumped at this anchor is a known
        # waste — remove it from consideration here.
        bumped = (
            {a for r, c, a in self._bumped_moves | self._bumped_gates
             if (r, c) == (anchor[0], anchor[1])}
            if anchor is not None
            else set()
        )
        # Deliberation first: once action steps are learned, plan
        # through known terrain rather than gradient-follow. ``goal``
        # may be None — planning then pursues the nearest frontier.
        planned = self._plan(grid, action_space, goal, reach)
        if planned is not None:
            return planned
        if goal is None:
            return None
        nav_space = [
            a
            for a in action_space
            if a not in bumped and not self._is_lethal(a)
        ]
        if not nav_space:
            return None  # every direction from here is a known wall
        # Hypothesis layer: the inferred goal conditions add a
        # gradient on top of geometric distance — an action predicted
        # to satisfy a goal condition costs less than one that merely
        # approaches the nearest object.
        ctx = self._task_context
        state = self._transition_state(grid) if ctx is not None else None
        danger = self._danger_cells(grid)
        return self._approach_step(
            grid, nav_space, pos, goal, anchor, ctx, state, danger
        )

    def _danger_cells(self, grid: Grid) -> list[tuple[float, float]]:
        """Known-hazard footprints — now, next predicted step, and
        one ring out (hazards can spread: flood/tide mechanics)."""
        danger: list[tuple[float, float]] = []
        if not self._hazard_colors:
            return danger
        scene = perceive(
            grid, background=self._background(grid),
            compute_relations=False,
        )
        max_threat = grid.height * grid.width * 0.25
        for obj in scene.objects:
            if obj.color in self._hazard_colors:
                if len(obj.cells) > max_threat:
                    continue
                danger.extend(obj.cells)
                v = self._hazard_velocity.get(obj.color)
                if v is not None:
                    # Predict where the hazard will be next step —
                    # danger is its future footprint, not just now.
                    danger.extend(
                        (r + v[0], c + v[1]) for r, c in obj.cells
                    )
                danger.extend(
                    (r + dr, c + dc)
                    for r, c in obj.cells
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
        return danger

    def _approach_step(
        self,
        grid: Grid,
        nav_space: list[Any],
        pos: tuple[float, float],
        goal: tuple[float, float],
        anchor: tuple[int, int, int, int] | None,
        ctx: Any,
        state: Any,
        danger: list[tuple[float, float]],
    ) -> Any | None:
        """Score each viable action by predicted distance to goal."""
        best, best_cost = None, float("inf")
        for action in nav_space:
            st = self.stats.get(action)
            step = self._action_step(st) if st is not None else None
            if step is not None:
                step_dr, step_dc = float(step[0]), float(step[1])
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
            if anchor is not None:
                # Cul-de-sac cost: an anchor whose exits are already
                # refused-by-terrain is a pocket. Gradient alone
                # walks back into it — the hill points at the goal
                # through a wall — so each walled-off exit prices the
                # trap in. Gate refusals don't count: a gate is a
                # question, not a dead end.
                dead = sum(
                    1
                    for r0, c0, _a in self._bumped_moves
                    if (r0, c0)
                    == (anchor[0] + int(step_dr), anchor[1] + int(step_dc))
                )
                remaining += dead * 4.0
            if ctx is not None:
                remaining -= (
                    self.task_competence.predicted_goal_gain(
                        ctx, str(action), state
                    )
                    * 8.0
                )
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

        # A constraint-board plan outranks the generic policy: the
        # specialist only speaks when its induced rule holds, so a
        # pick here is a directed click, not a wander.
        plan = self._boards.plan()
        self._board_click = plan if plan is not None and 6 in action_space else None
        if self._board_click is not None:
            self._last_action = 6
            self._cur_episode_actions.add(6)
            return 6

        self._recognize_env(action_space)

        eps = self.epsilon if self.avatar_color is None else self.epsilon * 0.25
        if self._grit_steps > 0:
            # Encouragement: for a while after losing a life, stay
            # curious — a fresh try beats repeating the fatal path.
            self._grit_steps -= 1
            eps = max(eps, self.epsilon)

        # Death avoidance: prefer actions whose episodes don't
        # reliably die. Falls back to the full space when everything
        # looks lethal — attributions are noisy, never a hard ban.
        safe = [a for a in action_space if not self._is_lethal(a)]
        pool = safe or action_space

        # Don't spend budget on moves known to bump at this anchor —
        # a wall doesn't become a door by asking again. Falls back to
        # the full pool when every option is a known wall.
        if self._last_frame is not None:
            anchor = self._avatar_anchor(self._last_frame)
            if anchor is not None and (self._bumped_moves or self._bumped_gates):
                refused = self._bumped_moves | self._bumped_gates
                unbumped = [
                    a
                    for a in pool
                    if (anchor[0], anchor[1], a) not in refused
                ]
                pool = unbumped or pool

        if self._rng.random() >= eps and self._last_frame is not None:
            # Plan over the whole learned step set — a bump is one
            # refused edge, not a reason to delete the action from
            # the graph. The fallback inside still respects this
            # anchor's refusals and the lethal list.
            nav = self._navigate(self._last_frame, action_space)
            if nav is not None:
                self._last_action = nav
                self._cur_episode_actions.add(nav)
                return nav

        # Exploration: prefer untried actions, then high-effect ones.
        untried = [a for a in pool if self._stat(a).attempts == 0]
        if untried:
            action = self._rng.choice(untried)
        else:
            # Hypothesis-aware exploration: undecided mechanic rules
            # whose context holds make their action worth running —
            # the test *is* the value.
            state = (
                self._transition_state(self._last_frame)
                if self._task_context is not None
                and self._last_frame is not None
                else None
            )
            weights = [
                0.1
                + self._stat(a).mean_change
                + (
                    self.task_competence.epistemic_bonus(
                        self._task_context, str(a), state
                    )
                    if state is not None
                    else 0.0
                )
                for a in pool
            ]
            action = self._rng.choices(pool, weights=weights)[0]
        self._last_action = action
        self._cur_episode_actions.add(action)
        return action

    def recommend(self, action_space: list[Any]) -> dict[str, float]:
        """The specialist's disposition over the space — a vote, not a command.

        Basal-ganglia arbitration blends these scores with its own
        policy, the world model's plan, and hypothesis terms: the
        navigational pick speaks loudest, lethal options lose their
        voice, untried controls keep a whisper of curiosity.
        """
        scores: dict[str, float] = {}
        nav = (
            self._navigate(self._last_frame, action_space)
            if self._last_frame is not None
            else None
        )
        for a in action_space:
            w = 0.1 + self._stat(a).mean_change
            if a == nav:
                w += 2.0
            if self._is_lethal(a):
                w = 0.0
            scores[str(a)] = max(w, 0.0)
        return scores

    def note_action(self, action: Any) -> None:
        """Record the move actually executed as the pending intention.

        When an arbiter (the basal ganglia) overrides the specialist's
        own pick, transition learning must still attribute the world's
        answer to what was *done*, not what was recommended.
        """
        self._last_action = action
        self._cur_episode_actions.add(action)

    def note_click(self, cell: tuple[int, int]) -> None:
        """Record the click actually executed for outcome attribution.

        When the arbiter's own field aims a coordinate action instead
        of the specialist's click memory, this keeps the executed —
        not suggested — cell as the one the world's answer attaches
        to. Same contract as ``note_action``, one level down.
        """
        self._pending_click = (int(cell[0]), int(cell[1]))

    def _is_lethal(self, action: Any) -> bool:
        """Whether *action* reliably precedes death, vs the baseline.

        Laplace-smoothed conditional rates: P(death | action used in
        episode) minus P(death | action not used). A positive lift with
        a minimum of evidence marks the action as lethal.
        """
        used, died = self._action_outcomes.get(action, (0, 0))
        if used < 2:
            return False
        p_used = (died + 1) / (used + 2)
        p_other = (self._episodes_died - died + 1) / (
            self._episodes_total - used + 2
        )
        return p_used - p_other > 0.4 or self._doom_blame.get(action, 0) >= 2

    def _is_lethal_cell(self, cell: tuple[int, int]) -> bool:
        """Same lift rule for clicked cells — the wrong-button signal."""
        used, died = self._cell_outcomes.get(cell, (0, 0))
        if used >= 2:
            p_used = (died + 1) / (used + 2)
            p_other = (self._episodes_died - died + 1) / (
                self._episodes_total - used + 2
            )
            if p_used - p_other > 0.4:
                return True
        return self._doom_blame.get(cell, 0) >= 2

    def _is_doom_color(self, color: int) -> bool:
        """Whether *color*'s appearances reliably precede death."""
        used, died = self._color_outcomes.get(color, (0, 0))
        if used < 2:
            return False
        p_used = (died + 1) / (used + 2)
        p_other = (self._episodes_died - died + 1) / (
            self._episodes_total - used + 2
        )
        return p_used - p_other > 0.4

    def _close_segment(self, died: bool) -> None:
        """Record outcomes for everything this segment touched.

        Called at episode end (``on_episode_end``) and at checkpoints
        (``mark_goal_reached``). Updates presence-based outcome records
        for actions, clicked cells, and newly-seen colors; on death,
        actions/cells that introduced known doom-markers take blame.
        """
        self._episodes_total += 1
        if died:
            self._episodes_died += 1
        for a in self._cur_episode_actions:
            rec = self._action_outcomes.setdefault(a, [0, 0])
            rec[0] += 1
            if died:
                rec[1] += 1
        self._cur_episode_actions.clear()
        for cell in self._cur_episode_clicks:
            crec = self._cell_outcomes.setdefault(cell, [0, 0])
            crec[0] += 1
            if died:
                crec[1] += 1
        self._cur_episode_clicks.clear()
        for color in self._cur_episode_colors:
            crec = self._color_outcomes.setdefault(color, [0, 0])
            crec[0] += 1
            if died:
                crec[1] += 1
        if died:
            # Blame the bringers of doom-markers — the action or click
            # that first made a doomed color appear.
            for color, source in self._seg_introduced.items():
                if self._is_doom_color(color):
                    self._doom_blame[source] = (
                        self._doom_blame.get(source, 0) + 1
                    )
        self._cur_episode_colors.clear()
        self._seg_introduced.clear()

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
        if self._board_click is not None:
            r, c = self._board_click
            self._pending_click = (r, c)
            self._boards.note_click((r, c))
            return {"x": c, "y": r}
        if grid is not None:
            candidates = [
                t for t in self._click_targets(grid)
                if not self._is_lethal_cell(t)
            ]
            if not candidates:
                candidates = [
                    (r, c)
                    for r in range(0, h, 2)
                    for c in range(0, w, 2)
                    if not self._is_lethal_cell((r, c))
                ]
            # A cell that produced real change is a working control —
            # re-press it before burning steps on untouched background.
            # A control that stops responding falls out of this set on
            # its next click (its recorded outcome updates to ~0).
            responsive = [
                t for t in self._click_outcomes
                if self._click_outcomes[t] > 0
                and not self._is_lethal_cell(t)
            ]
            fresh = [
                t for t in candidates if t not in self._click_outcomes
            ]
            if responsive:
                r, c = max(
                    responsive, key=lambda t: self._click_outcomes[t]
                )
            elif fresh:
                r, c = fresh[0]
            elif self._click_outcomes:
                # Everything tried — return to whatever did the most.
                r, c = max(
                    self._click_outcomes, key=lambda k: self._click_outcomes[k]
                )
            elif candidates:
                r, c = candidates[0]
            else:
                r, c = (0, 0)
            self._pending_click = (r, c)
            return {"x": c, "y": r}
        r = self._rng.randint(0, h - 1)
        c = self._rng.randint(0, w - 1)
        self._pending_click = (r, c)
        return {"x": c, "y": r}

    # ── Episode boundary ──────────────────────────────────────

    def mark_goal_reached(self) -> None:
        """Record that the avatar reached its current goal cell.

        Also a credit-assignment checkpoint: a level-up means every
        action since the last checkpoint belonged to a *surviving*
        segment. Closing the segment here keeps blame proportional —
        without it, an innocent action that appears in every segment
        looks as lethal as the action that only ever appears in dying
        ones.
        """
        self._close_segment(died=False)
        # A new level is a new layout: position-indexed memory —
        # walls, gates, visited anchors, bump records, object
        # bindings — belongs to the maze that was just cleared.
        # Physics (step vectors, hazard/valuable colors, avatar
        # signature) carries forward; that's the transfer.
        self._blocked_cells.clear()
        self._block_suspect.clear()
        self._gated_cells.clear()
        self._touch_fail.clear()
        self._bumped_moves.clear()
        self._bumped_gates.clear()
        self._visited_cells.clear()
        self._unreachable_goals.clear()
        self._unreach_strikes.clear()
        self._tried_at_state.clear()
        self._gate_requirements.clear()
        self._committed_goal = None
        self._goal_cells = frozenset()
        self._commit_missing = 0
        self._plan_target = None
        self._ui_cells.clear()
        self._tracks.clear()
        self._prev_anchor = None
        self._avatar_last = None
        self._occluded_objects.clear()
        self._last_death_at = -99

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
        # Episode-level credit assignment: every action, click, and
        # newly-seen color this segment gets an outcome record (died
        # or not). Presence — not last-action — because fatal choices
        # often land long before the death they cause.
        self._close_segment(died=state == "GAME_OVER")
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
                # Goal inference: the terminal state is abductive
                # evidence — the vars it ended on (the last change
                # first) are the schema's goal-hypothesis candidates.
                final_state=(
                    self._transition_state(self._last_frame)
                    if self._last_frame is not None
                    else None
                ),
                terminal_changes=getattr(
                    self, "_last_transition_changes", ()
                ),
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
        self._avatar_last = None
        self._avatar_shape = frozenset()
        self._prev_anchor = None
        self._committed_goal = None
        self._goal_cells = frozenset()
        self._commit_missing = 0
        self._plan_target = None
