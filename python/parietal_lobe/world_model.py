"""World model — persistent environment state built from perception.

Dorsal-stream state: the parietal lobe's running answer to "where are
things, which one is me, and what does touching them do?"

The dual of :class:`SituationModel`. Where the situation model
accumulates what was *said*, the world model accumulates what was
*seen* and *done*. A ``Scene`` arrives per observation; the model
maintains entity identity across frames, infers which entity is
herself (agency is learned from action contingency, never told),
learns what each action does to her body, and learns what touching
each kind of thing does. Environments are never named — the same
model serves a game, a robot camera, or a simulated room.

Architecture, mapped onto the organism:

- **Entity persistence** — a perceived object is matched to a tracked
  entity by color, shape, and motion continuity. The world keeps
  happening between observations whether or not she attends to it.
- **Agency** — the entity whose motion reliably follows her actions
  is her body. Contingency is counted per action step; the minimal
  self is inferred, never assigned.
- **Action effects** — each action id accumulates the displacement it
  actually produced, so "ACTION1" meaning "up" is learned, not
  configured. Different environments can assign different semantics
  to the same action alphabet.
- **Affordances** — contact outcomes accrue per entity kind (color):
  touching it collects, blocks, recolors, or kills. This is the
  environment's verb vocabulary, learned from experience.
- **Map + paths** — known-free/known-blocked cells and visit counts;
  ``path_step`` computes the shortest route to whatever target set
  the basal ganglia's policy selects.

``reset(keep_learning=True)`` clears positions at episode boundaries
while keeping learned effects and affordances — knowledge persists
across episodes; arrangements do not.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from parietal_lobe.boards import ConstraintSession
from parietal_lobe.scene import Scene

Cell = tuple[int, int]
# A body part's state: its color and normalized shape together.
# Rotation and shape changes both alter the shape tuple, so the
# pair captures every self-state dimension perception can see.
PartSig = tuple[int, tuple[Cell, ...]]

# Outcome kinds accumulated in ``affordances[color]``.
OUT_COLLECT = "collect"      # touched entity vanished — a pickup
OUT_BLOCK = "block"          # contact prevented movement — a wall
OUT_TRANSFORM = "transform"  # contact changed her body — a state changer
OUT_HAZARD = "hazard"        # contact preceded a world reset — deadly
OUT_WIN = "win"              # contact preceded a level win
OUT_NOTHING = "nothing"      # contact provably changed nothing — surface

# Observation bookkeeping.
_MAX_TRACK_DIST = 6.0         # centroid distance for same-entity match
_GONE_AFTER = 1               # misses before an entity is declared gone
_SELF_MIN_CONTINGENT = 2      # action-contingent moves before self is named
_SELF_MAX_SILENCE = 8         # unanswered actions before self is re-judged
_PART_MIN_CO_MOVES = 2        # consecutive co-moves before annexation
_STATIC_AFTER = 6             # observations unmoved → counts as terrain
_MAX_OBJECT_FRACTION = 0.66   # ≥⅔ of the field is fill, not figure
_MAX_TRANSLATE = 12           # largest per-step displacement searched
_RESET_MIN_ENTITIES = 3       # live entities whose total loss means new world
_CONTACT_FRESH_AFTER = 4      # ticks apart before a touch counts as new
_RESHAPE_MIN_FRACTION = 0.15  # changed-cells share that counts as reshaping
_COLLECT_CONFIRM = 3          # ticks a vanished contact must stay gone
_CLICK_RADIUS = 3             # cells around a distal act counted as its answer
_MOVE_FAILURE_RETRIES = 2     # observed no-ops before a state-action is refused
_MOVE_FAILURE_COOLDOWN = 20   # ticks before a changed world earns a retrial


def _rigid_translation(
    old: frozenset[Cell], new: frozenset[Cell]
) -> tuple[Cell, float]:
    """Best rigid translation explaining an entity's old → new cells.

    Velocity is measured as the displacement that maximizes cell
    overlap, not as a centroid delta: a partially occluded or
    re-segmented footprint drags the centroid toward the occlusion
    and reports sub-tile motion that never happened. The argmax-overlap
    estimate recovers the dominant translation regardless — a 5×5
    block sliding under an edge still votes overwhelmingly for its
    true shift.

    Returns ``(delta, confidence)`` where confidence is the fraction
    of the new footprint the translation explains; 0.0 means no
    plausible rigid move (teleport, wholesale re-segmentation) and
    the delta is meaningless evidence.
    """
    old_set = set(old)
    new_set = set(new)
    best: Cell = (0, 0)
    best_overlap = 0
    best_mag = _MAX_TRANSLATE * 2 + 1
    for dr in range(-_MAX_TRANSLATE, _MAX_TRANSLATE + 1):
        for dc in range(-_MAX_TRANSLATE, _MAX_TRANSLATE + 1):
            overlap = len(new_set & {(r + dr, c + dc) for r, c in old_set})
            mag = abs(dr) + abs(dc)
            if overlap > best_overlap or (
                overlap == best_overlap and mag < best_mag
            ):
                best, best_overlap, best_mag = (dr, dc), overlap, mag
    confidence = best_overlap / max(1, len(new_set))
    return best, confidence

# Canonical interface labels → hypothesized displacement. These are
# the shared convention an interface advertises (a labeled arrow
# button), not environment truth — the world may repurpose any action.
_LABEL_DISPLACEMENTS: dict[str, Cell] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class WorldEvent:
    """One thing that changed in the world on a single observation.

    ``kind`` is a small closed vocabulary the engine can react to;
    ``text`` is a plain description for memory records.
    """

    kind: str
    text: str
    salience: float = 0.3
    entity_id: int = -1
    other_id: int = -1


@dataclass(frozen=True)
class SimState:
    """A plannable world state.

    The minimum sufficient state for forward simulation: where her
    body is, what her body *is* (``signature`` — every part's color
    and shape; transformer contact rewrites it — that is what keys
    are), and which entities have been consumed. Frozen and hashable
    so planners can deduplicate visited states.
    """

    cells: frozenset[Cell]
    signature: tuple[PartSig, ...]
    gone: frozenset[int]


@dataclass
class WorldEntity:
    """An object that persists across observations."""

    id: int
    color: int
    cells: frozenset[Cell]
    centroid: tuple[float, float]
    shape: tuple[Cell, ...]
    first_seen: int
    last_seen: int
    velocity: tuple[float, float] = (0.0, 0.0)
    velocity_exact: bool = True  # last move measured rigidly, not guessed
    n_seen: int = 1
    churn_ticks: int = 0        # observations where its cells changed
    n_moved_free: int = 0       # moved while no action was pending
    action_moves: int = 0       # moved on the step right after her action
    # Moves after her action that matched the action's learned
    # effect (or happened before any effect was known). The body is
    # the thing that goes where the command sends it — a thing that
    # merely happens to move when she acts (a pad sliding on its own
    # track, an indicator flashing) does not answer the command.
    action_effect_moves: int = 0
    steps_after_action: int = 0
    is_self: bool = False
    part_of_self: bool = False  # annexed: rigidly rides with her body
    gone: bool = False
    gone_tick: int = -1         # observation when it vanished
    co_moves: int = 0           # consecutive frames co-translating with her
    # Conditional passability, observed per entity: which of her
    # signatures this thing has stopped, and which it has admitted.
    # A gate is a thing that blocks under some self-states and not
    # others — evidence about this entity, not its whole kind.
    blocked_states: set[tuple[PartSig, ...]] = field(default_factory=set)
    open_states: set[tuple[PartSig, ...]] = field(default_factory=set)
    # Per-sig verdict epoch for blocked_states: the epoch a flashed
    # (state-conditional) refusal was measured at, ``None`` for a
    # silent — unconditional — one. Entity evidence lives here so
    # one gate's conditional verdict cannot be overridden by a
    # silent wall of the same color.
    blocked_epochs: dict[tuple[PartSig, ...], int | None] = field(
        default_factory=dict
    )
    # When a match rewrote this entity's state this tick — recolor,
    # reshape, rotation — the (from → to) pair, for transform
    # attribution. ``None`` on an unremarkable observation.
    last_transform: tuple[PartSig, PartSig] | None = None

    def is_display(self, bounds: tuple[int, int]) -> bool:
        """A readout, not a thing: edge-anchored and churning on
        schedule — a counter, a gauge, a status row. It changes
        whether or not anything touches it, and it sits where
        displays sit. ``bounds`` is the field's (height, width)."""
        if self.n_seen < 6 or self.churn_ticks * 2 < self.n_seen:
            return False
        h, w = bounds
        return any(
            r < 2 or c < 2 or r >= h - 2 or c >= w - 2
            for r, c in self.cells
        )

    @property
    def spontaneous(self) -> bool:
        """Moves on its own — not her, not static terrain."""
        return self.n_moved_free > 0 and not self.is_self

    @property
    def static(self) -> bool:
        """Has never moved over enough observations — likely terrain."""
        return (
            self.n_seen >= _STATIC_AFTER
            and self.n_moved_free == 0
            and self.action_moves == 0
        )


@dataclass(frozen=True)
class _PendingCollect:
    """A contact-vanished entity awaiting the eaten/covered verdict."""

    color: int
    cells: frozenset[Cell]
    tick: int


class WorldModel:
    """Persistent model of a perceived environment.

    The harness feeds ``observe()`` a ``Scene`` per frame and calls
    ``note_action()`` when she acts; the model attributes the next
    observation's deltas to that action. What to *do* is not the
    model's business — action selection lives in the basal ganglia's
    ``WorldPolicy``.
    """

    def __init__(self) -> None:
        self.tick = 0
        self.entities: dict[int, WorldEntity] = {}
        self._next_id = 1
        self.self_id: int | None = None
        self._pending_action: str | None = None
        # Which entity was her when the pending action was issued.
        # If self is re-adopted before the world's answer arrives,
        # the measured delta spans two different bodies — identity
        # noise, not physics; the trial is discarded, not learned.
        self._pending_self_id: int | None = None
        # The cell a coordinate-typed action was aimed at — the
        # distal analogue of the pending action: without it a click
        # is unattributable, "ACTION6 happened" with nowhere it
        # landed.
        self._pending_target: Cell | None = None
        # Previous frame's cell→color snapshot — the before-image a
        # distal action's after-image is measured against.
        self._last_grid: dict[Cell, int] | None = None

        # Learned knowledge — survives episode resets.
        self.action_effects: dict[str, Counter[Cell]] = {}
        self.action_tries: Counter[str] = Counter()
        self.action_noops: Counter[str] = Counter()
        # Empirical dynamics the entity map failed to explain: a
        # state-action that repeatedly left the body unmoved is a
        # measured transition failure, not a route worth replanning.
        self._move_failures: dict[
            tuple[str, frozenset[Cell], tuple[PartSig, ...]],
            tuple[int, int],
        ] = {}
        self.affordances: dict[int, Counter[str]] = {}
        # Distal affordances — what acting *at* a kind does, learned
        # the same way contact affordances are learned: tries count
        # the experiments, changes sum the cells the world answered
        # with. ``click_changes[(action, color)] / tries`` is the
        # kind's responsiveness — a control that answers is worth
        # pressing again; a dead one stops paying for probes.
        self.click_tries: Counter[tuple[str, int | None]] = Counter()
        self.click_changes: Counter[tuple[str, int | None]] = Counter()
        # The last distal act: (cell, color-that-was-there). When the
        # world ends without a body to credit, this is the cause to
        # attribute — the thing she just acted on.
        self._last_clicked: tuple[Cell, int | None] | None = None
        # What contact does to her body: kind → the part state it
        # rewrote → what it rewrote it to. ``transforms[5][part_a]
        # [part_b] += 1`` means "the color-5 thing changed my part
        # from a to b". Simulation applies the modal rewrite.
        self.transforms: dict[int, dict[PartSig, Counter[PartSig]]] = {}

        # Interface conventions — priors, not truth. A harness may
        # label an action id with the canonical semantic the
        # interface advertises ("up", "coordinate", ...); the label
        # is a hypothesis that stands only until the action's real
        # effect is observed. ``action_effects`` stays pure evidence.
        self.action_labels: dict[str, str] = {}
        self.action_priors: dict[str, Cell] = {}
        self.coordinate_actions: set[str] = set()

        # Conditional passability lives on the entities themselves
        # (WorldEntity.blocked_states / open_states): "that gate
        # stopped me" is evidence about that gate, not everything
        # sharing its color. The kind level keeps the same
        # granularity: ``blocked_kinds[color]`` is the set of
        # self-states this kind has refused — a bump under one
        # signature extends to unbumped members of the kind under
        # that signature (walls stay walls), but never crosses
        # states: the gate that stopped one body may be exactly
        # the door the next body opens.
        self.blocked_kinds: dict[int, set[tuple[PartSig, ...]]] = {}
        # The admitting twin: kinds that let a self-state enter.
        # When a kind has both refused and admitted one state it is
        # a mixed kind — members differ — and unobserved members
        # fall back to per-entity evidence rather than the kind's
        # refusal.
        self.open_kinds: dict[int, set[tuple[PartSig, ...]]] = {}
        # Verdict epoch: a refusal that arrived the same tick part
        # of her changed is *conditional* — the thing did not stop
        # her body, it stopped the state her body was in, and said
        # so. ``None`` marks an unconditional (silent) refusal — a
        # plain wall. Conditional verdicts bind only in the epoch
        # they were measured: contact with something unexplained
        # may have rewritten state her sensors cannot see, and a
        # verdict from before that touch is stale evidence.
        self._blocked_epoch: dict[tuple[int, tuple[PartSig, ...]], int | None] = {}
        self._state_epoch = 0
        # Tick each kind was last in contact — a touch counts as
        # fresh only after a real absence. Perception flicker can
        # drop a contact for a tick and bring it back; that is the
        # same ongoing touch, not a new one, and it must not open
        # epochs on a loop.
        self._contact_kinds: dict[int, int] = {}
        # Kinds whose unexplained contact coincided with a state
        # epoch opening — candidate hidden-state changers. Their
        # rewrite is invisible to her signature, so the epoch is
        # the only trace it leaves.
        self._state_changers: set[int] = set()
        # The cerebellar comparator — efference copy against
        # reafference. ``None`` is the model standing alone (tests):
        # a mismatch is then binary surprise, exactly as before.
        # Wired in by Mind once the brain's regions exist.
        self.motor_comparator: Any = None
        # Cell -> entity ids, rebuilt per observation: forward
        # simulation asks "what is here" thousands of times a step.
        self._cell_entities: dict[Cell, list[int]] = {}

        # Spatial map — learned knowledge of this world's layout:
        # persists across retries like the physics, wiped only for
        # a new world (keep_learning=False).
        self.known_free: set[Cell] = set()
        self.known_blocked: set[Cell] = set()
        # Cells whose ownership is unexplained — no entity claimed
        # them — yet a move into them failed. The boundary of the
        # world is not an object, but it refuses entry all the same.
        # Free-space marking must not resurrect them.
        self.refused_cells: set[Cell] = set()
        # One refused strip is a suspicion, not a wall: a bump's
        # leading edge blames every unexplained cell inside it, so
        # floor beside a real wall reads the same as the wall. A
        # cell earns refused status only where refused strips keep
        # overlapping — certainty is a singleton strip (one cell
        # explains the refusal) or convergence (the same cell
        # under repeated refusals).
        self._refusal_suspects: Counter[Cell] = Counter()
        self.visits: Counter[Cell] = Counter()

        # Terrain kinds — the medium she moves through, not objects.
        # Nothing in perception marks ground as ground; the evidence
        # is that it re-occupies cells she vacates. A color learned
        # as terrain is never credited as a blocker, transformer,
        # or suspect — it cannot be what happened to her, it is
        # where she is. Learned physics: survives episode resets.
        self.terrain_kinds: set[int] = set()
        self._prev_footprint: frozenset[Cell] | None = None

        # Causal-suspicion bookkeeping (learned, survives resets):
        # contact_ticks = how often each kind was in contact;
        # death_suspects = how often contact coincided with an
        # unexplained ending. Ratio distinguishes a rare killer
        # from a wall she is always leaning on.
        self.contact_ticks: Counter[int] = Counter()
        self.death_suspects: Counter[int] = Counter()

        # A contact whose target vanished is only *maybe* a
        # collection: her own body covers what it steps on, and a
        # covered surface marking vanishes exactly like a consumed
        # pickup. Records sit until the world proves which —
        # reappearance says occlusion, staying gone says eaten.
        self._pending_collect: list[_PendingCollect] = []

        # Episodic recall's residue on this world: color → the
        # valence similar past experiences left behind. Written by
        # the embodiment layer's recall pass each window it fires;
        # read by the motor field as associative pull. Survives
        # episode resets (that is what memory is for) and clears
        # with the world.
        self.recalled_valence: dict[int, float] = {}

        # Consecutive action-observations the current self failed
        # to answer. Above the limit she is re-judged — a body that
        # stops answering her commands is evidence it was never
        # hers (a display element can inherit the flag through
        # re-segmentation, a dead avatar leaves none).
        self._self_silence = 0

        # Field dimensions, for readout-vs-thing discrimination.
        self._grid_bounds: tuple[int, int] = (0, 0)

        # Structured scenes — lattice parsing, marker relations,
        # the colour cycle clicks drive. Learns the same way the
        # rest of the model does: what the screen shows, induced
        # from demonstrated states, never assumed.
        self.boards = ConstraintSession()

    def _resolve_collections(self, events: list[WorldEvent]) -> None:
        """Settle vanished-contact records into honest outcomes.

        A thing that disappeared under her is eaten only if the world
        keeps it gone; if a same-colored thing reappears on its cells
        it was never consumed — she had covered it. Reappearance
        credits ``nothing``: contact observed to have no effect, which
        is itself learned knowledge (the kind is probed, answered,
        and no longer worth approaching). Neither verdict fires on
        the disappearance tick itself — the covering footprint is
        still there.
        """
        if not self._pending_collect:
            return
        waiting: list[_PendingCollect] = []
        for pend in self._pending_collect:
            revived = any(
                not e.gone
                and e.color == pend.color
                and e.cells & pend.cells
                for e in self.entities.values()
            )
            if revived:
                self.affordances.setdefault(pend.color, Counter())
                self.affordances[pend.color][OUT_NOTHING] += 1
                continue
            if self.tick - pend.tick < _COLLECT_CONFIRM:
                waiting.append(pend)
                continue
            self.affordances.setdefault(pend.color, Counter())
            self.affordances[pend.color][OUT_COLLECT] += 1
            me = self._effector()
            events.append(WorldEvent(
                "collected",
                f"I touched the color-{pend.color} thing and it vanished",
                salience=0.7,
                entity_id=me.id if me is not None else -1,
            ))
        self._pending_collect = waiting

    # ─── Observation ────────────────────────────────────────────

    def observe(
        self,
        scene: Scene,
        info: dict[str, Any] | None = None,
    ) -> list[WorldEvent]:
        """Integrate one perceived scene; return what changed.

        ``info`` is an optional caller-supplied dict of outcome flags
        the environment reports externally (``{"reset": True}``,
        ``{"win": True}``) — the model does not assume any particular
        environment protocol.
        """
        events: list[WorldEvent] = []

        # A reported terminal outcome ends the world that produced
        # it. The frame arriving now already shows the new world; the
        # causal contact was last tick's. Credit the cause from the
        # state still tracked, forget the place, keep the physics.
        if info:
            term = (
                OUT_HAZARD
                if info.get("reset") or info.get("death")
                else OUT_WIN if info.get("win") else None
            )
            if term is not None:
                events.extend(self.attribute_terminal(term))
                self.reset(
                    keep_learning=True,
                    keep_map=term != OUT_WIN,
                )

        self.tick += 1
        self._grid_bounds = (scene.grid.height, scene.grid.width)
        pending = self._pending_action
        self._pending_action = None
        pending_self = self._pending_self_id
        self._pending_self_id = None
        pending_target = self._pending_target
        self._pending_target = None

        # Figure/ground: a region occupying more than half the field
        # is the world itself, not a thing in it — floor and fog
        # should never be tracked as entities.
        field_area = scene.grid.height * scene.grid.width
        objects = [
            o for o in scene.objects
            if len(o.cells) <= field_area * _MAX_OBJECT_FRACTION
        ]
        matched, new_objects = self._match(objects)

        # An unannounced world reset: every tracked thing lost
        # continuity at once — the environment replaced the world
        # under her without saying so (a life ending silently, a
        # level swapped mid-episode). Partial turnover is ordinary
        # motion and occlusion; wholesale turnover is a new world.
        # Whatever she was touching when it ended is a death suspect
        # — contiguity is evidence, sharpened by co-occurrence.
        prior_live = sum(
            1 for e in self.entities.values() if not e.gone
        )
        if prior_live >= _RESET_MIN_ENTITIES and not matched:
            events.extend(self.suspect_terminal(pending))
            self.reset(keep_learning=True)
            events.append(WorldEvent(
                "reset", "the world reset", salience=0.9))
            matched, new_objects = self._match(objects)

        for entity in self.entities.values():
            entity.last_transform = None
        for entity, obj in matched:
            changed = entity.cells != obj.cells
            if changed:
                delta, confidence = _rigid_translation(
                    entity.cells, obj.cells
                )
            else:
                delta, confidence = (0, 0), 1.0
            entity.velocity = (float(delta[0]), float(delta[1]))
            entity.velocity_exact = confidence >= 0.5
            # Contingency is counted on real motion only: a changed
            # footprint without a coherent rigid translation is
            # segmentation churn — occlusion, camouflage drift, her
            # own cells sliding across same-colored ground — not the
            # entity going anywhere. A background that ripples as she
            # passes cannot be her body.
            rigid_move = delta != (0, 0) and entity.velocity_exact
            if pending is not None:
                entity.steps_after_action += 1
                if rigid_move:
                    entity.action_moves += 1
                    expected = self.effect_of(pending)
                    if expected is None or delta == expected:
                        entity.action_effect_moves += 1
            elif rigid_move:
                entity.n_moved_free += 1
            if entity.shape != obj.shape_signature:
                # Same color, new silhouette. Only a meaningful
                # change is a transform — a big region shedding a few
                # cells to occlusion is not "reshaped", it is
                # glimpsed differently.
                churn = len(entity.cells - obj.cells) + len(
                    obj.cells - entity.cells
                )
                old_shape = entity.shape
                entity.shape = obj.shape_signature
                if churn >= max(3, int(len(obj.cells)
                                       * _RESHAPE_MIN_FRACTION)):
                    entity.last_transform = (
                        (entity.color, old_shape),
                        (entity.color, obj.shape_signature),
                    )
                    events.append(WorldEvent(
                        "reshaped",
                        f"entity {entity.id} changed shape",
                        salience=0.6,
                        entity_id=entity.id,
                    ))
            if entity.cells != obj.cells:
                entity.churn_ticks += 1
            prev_cells = entity.cells
            entity.cells = obj.cells
            entity.centroid = obj.centroid
            entity.last_seen = self.tick
            entity.n_seen += 1
            if rigid_move:
                events.append(WorldEvent(
                    "moved",
                    f"entity {entity.id} moved to {entity.centroid}",
                    entity_id=entity.id,
                ))
            if entity.is_self or entity.part_of_self:
                # Ground crossed by a stride is covered ground even
                # though she never stands inside it — otherwise the
                # cells between footprint and landing stay
                # "unvisited" forever and the frontier keeps
                # sending her back to tiles she already passed.
                crossed = (
                    self._swept(prev_cells, delta)
                    if rigid_move
                    else set()
                )
                for cell in obj.cells | crossed:
                    self.known_free.add(cell)
                    self.visits[cell] += 1

        # Prediction check: what did the pending action do to her?
        # Runs after velocities are measured — comparing prediction
        # against last frame's motion answers the wrong question.
        # Only a confidently-measured motion can disappoint a
        # prediction — a non-rigid footprint means the observation
        # itself is ambiguous, not surprising.
        if (
            pending is not None
            and self.self_id is not None
            and self.self_id == pending_self
        ):
            self_ent = self.entities.get(self.self_id)
            if (
                self_ent is not None
                and not self_ent.gone
                and self_ent.velocity_exact
            ):
                expected = self.effect_of(pending)
                if expected is not None and expected != (0, 0):
                    observed = (
                        round(self_ent.velocity[0]),
                        round(self_ent.velocity[1]),
                    )
                    if observed != expected:
                        # The cerebellum grades the miss — a bump and
                        # a mislearned stride are not the same
                        # surprise — and accumulates it against the
                        # action's calibration. Without it the event
                        # carries the flat surprise it always did.
                        salience = 0.5
                        if self.motor_comparator is not None:
                            try:
                                magnitude = (
                                    self.motor_comparator.compare(
                                        pending, expected, observed
                                    )
                                )
                                salience = min(
                                    0.9, 0.4 + 0.5 * magnitude
                                )
                            except Exception:  # noqa: BLE001 — advisory
                                pass
                        events.append(WorldEvent(
                            "unexpected",
                            f"action {pending} moved me {observed}, "
                            f"expected {expected}",
                            salience=salience,
                            entity_id=self_ent.id,
                        ))

        # Annexed parts are tracked by attachment, not appearance.
        # A transform mid-move — recolor, reshape, rotation on the
        # tick she stepped onto the thing that did it — leaves no
        # same-color match: the part's new self sits where her
        # motion carried it. The object at the expected offset IS
        # the part, whatever it now looks like. A much bigger object
        # is not an adoption — camouflage (a part that melts into
        # same-colored ground) swallows the part, it does not
        # inflate it into the terrain.
        me_for_parts = self.self_entity
        if me_for_parts is not None and not me_for_parts.gone:
            shift = (
                round(me_for_parts.velocity[0]),
                round(me_for_parts.velocity[1]),
            )
            for part in self.entities.values():
                if (
                    not part.part_of_self
                    or part.gone
                    or part.last_seen >= self.tick
                ):
                    continue
                expected_cells = {
                    (r + shift[0], c + shift[1])
                    for r, c in part.cells
                }
                adopt = next(
                    (
                        o for o in new_objects
                        if len(o.cells) <= 2 * len(expected_cells)
                        and len(o.cells & expected_cells) * 2
                        >= len(expected_cells)
                    ),
                    None,
                )
                if adopt is None:
                    continue
                new_objects.remove(adopt)
                old_sig = (part.color, part.shape)
                kind = (
                    "recolored"
                    if adopt.color != part.color
                    else "reshaped"
                )
                part.last_transform = (
                    old_sig,
                    (adopt.color, adopt.shape_signature),
                )
                part.color = adopt.color
                part.shape = adopt.shape_signature
                part.cells = adopt.cells
                part.centroid = adopt.centroid
                part.velocity = me_for_parts.velocity
                part.velocity_exact = True
                part.last_seen = self.tick
                part.n_seen += 1
                events.append(WorldEvent(
                    kind,
                    f"part of me changed to color {adopt.color}",
                    salience=0.6,
                    entity_id=part.id,
                ))

        for obj in new_objects:
            reclaimed = self._recolor_match(obj)
            # _recolor_match stamps last_transform before returning
            # the entity, so the unpack always has something to read.
            if reclaimed is not None and reclaimed.last_transform is not None:
                before, _after = reclaimed.last_transform
                kind = (
                    "recolored" if before[0] != obj.color else "reshaped"
                )
                events.append(WorldEvent(
                    kind,
                    f"entity {reclaimed.id} changed to "
                    f"color {obj.color}",
                    salience=0.6,
                    entity_id=reclaimed.id,
                ))
                continue
            entity = self._spawn(obj)
            events.append(WorldEvent(
                "appeared",
                f"a color-{obj.color} thing appeared at {obj.centroid}",
                salience=0.4,
                entity_id=entity.id,
            ))

        # Entities not matched this frame may be gone.
        seen_ids = {e.id for e, _ in matched}
        for entity in self.entities.values():
            if (
                entity.last_seen < self.tick
                and not entity.gone
                and entity.id not in seen_ids
                and self.tick - entity.last_seen >= _GONE_AFTER
            ):
                entity.gone = True
                entity.gone_tick = self.tick
                events.append(WorldEvent(
                    "vanished",
                    f"entity {entity.id} vanished",
                    salience=0.4,
                    entity_id=entity.id,
                ))

        # A partial reset: only her body vanished while the rest of
        # the world persisted. The wholesale-turnover check above
        # cannot see this kind of death — the maze kept its
        # continuity — but the fatal contact/path still exists and is
        # the only evidence about what ended the attempt.
        current_self = (
            self.entities.get(self.self_id)
            if self.self_id is not None
            else None
        )
        if (
            current_self is not None
            and current_self.gone
            and current_self.gone_tick == self.tick
        ):
            events.extend(self.suspect_terminal(pending))

        # Whether the current self answered this action feeds the
        # liveness judgment inside _infer_self — evaluated after
        # velocities are measured, like everything else that reads
        # this frame's motion.
        me_now = self.self_entity
        if (
            pending is not None
            and me_now is not None
            and not me_now.gone
        ):
            if me_now.velocity_exact and me_now.velocity != (0.0, 0.0):
                self._self_silence = 0
            else:
                self._self_silence += 1

        # Agency before attribution: the steps that reveal which
        # entity is hers are also evidence of what the action does.
        self._resolve_collections(events)
        self._infer_self(events)
        if pending is not None:
            self._attribute_action(pending, events, pending_self)
        if pending is not None and pending_target is not None:
            self._attribute_click(pending, pending_target, scene, events)
        self._update_parts(events)
        self._learn_terrain(scene, pending)
        self._update_map(scene)
        self._update_passability()
        # Structured-scene parsing — boards, markers, and the click
        # cycle — reparsed per frame like the entities themselves.
        self.boards.observe(scene.grid)
        events.extend(self._contact_outcomes(pending, events))
        # The after-image for the next act's before-image: distal
        # attribution measures the world's answer against this
        # frame, so the snapshot is taken only after every update
        # that could mutate tracked state.
        self._last_grid = {
            (r, c): v for r, c, v in scene.grid.iter_cells()
        }
        # Forward-simulation runs thousands of contact checks a
        # step; a cell->entity index turns each into swept-size
        # lookups instead of a scan of everything alive.
        self._cell_entities = {}
        for entity in self.entities.values():
            for cell in entity.cells:
                self._cell_entities.setdefault(cell, []).append(entity.id)

        if info:
            if info.get("reset") or info.get("death"):
                events.append(WorldEvent(
                    "reset", "the world reset", salience=0.9))
            if info.get("win"):
                events.append(WorldEvent(
                    "win", "the level was won", salience=0.9))
        return events

    def _match(
        self, objects: list[Any]
    ) -> tuple[list[tuple[WorldEntity, Any]], list[Any]]:
        """Match perceived objects to tracked entities.

        Greedy nearest-centroid match among live entities of the same
        color. A second pass lets a *recolored* object keep its
        identity when it matches a just-vanished entity's footprint.
        """
        matched: list[tuple[WorldEntity, Any]] = []
        used: set[int] = set()
        new_objects = []
        for obj in objects:
            best: WorldEntity | None = None
            best_dist = _MAX_TRACK_DIST
            for entity in self.entities.values():
                if (
                    entity.gone
                    or entity.id in used
                    or entity.color != obj.color
                ):
                    continue
                # A footprint does not triple between observations
                # by ordinary motion — that is camouflage (the
                # entity melted into same-colored ground) or re-
                # segmentation, not continuity. Shrinking is fine:
                # occlusion legitimately hides most of a thing.
                if len(obj.cells) > 3 * len(entity.cells):
                    continue
                dist = (
                    (entity.centroid[0] - obj.centroid[0]) ** 2
                    + (entity.centroid[1] - obj.centroid[1]) ** 2
                ) ** 0.5
                if dist < best_dist:
                    best, best_dist = entity, dist
            if best is None:
                new_objects.append(obj)
            else:
                used.add(best.id)
                matched.append((best, obj))
        return matched, new_objects

    def _recolor_match(self, obj: Any) -> WorldEntity | None:
        """Reclaim an entity that changed appearance in place.

        A transform looks like a vanish + an appear; if the new object
        sits where a just-unmatched entity was, it is the same thing
        rewritten — recolor, reshape, and reorient all preserve the
        seat even when they change everything drawn on it. Cell
        overlap is the strongest evidence; a centroid that stayed put
        carries the same verdict when the new shape shares no cells
        with the old (a rotation can redraw every pixel around the
        same center). A footprint that tripled is camouflage or re-
        segmentation, not continuity.
        """
        for entity in self.entities.values():
            if (
                entity.gone
                or entity.last_seen >= self.tick
                or (entity.color == obj.color
                and entity.shape == obj.shape_signature)
            ):
                continue
            if len(obj.cells) > 3 * len(entity.cells):
                continue
            overlap = entity.cells & obj.cells
            if len(overlap) * 2 < len(entity.cells):
                drift = (
                    (entity.centroid[0] - obj.centroid[0]) ** 2
                    + (entity.centroid[1] - obj.centroid[1]) ** 2
                ) ** 0.5
                if drift > 2.5:
                    continue
            old_sig = (entity.color, entity.shape)
            entity.last_transform = (
                old_sig,
                (obj.color, obj.shape_signature),
            )
            entity.color = obj.color
            entity.shape = obj.shape_signature
            entity.cells = obj.cells
            entity.velocity = (
                obj.centroid[0] - entity.centroid[0],
                obj.centroid[1] - entity.centroid[1],
            )
            entity.velocity_exact = False
            entity.centroid = obj.centroid
            entity.last_seen = self.tick
            entity.n_seen += 1
            return entity
        return None

    def _spawn(self, obj: Any) -> WorldEntity:
        entity = WorldEntity(
            id=self._next_id,
            color=obj.color,
            cells=obj.cells,
            centroid=obj.centroid,
            shape=obj.shape_signature,
            first_seen=self.tick,
            last_seen=self.tick,
        )
        self.entities[entity.id] = entity
        self._next_id += 1
        return entity

    def _infer_self(self, events: list[WorldEvent]) -> None:
        """Name the entity whose motion follows her actions.

        Agency is contingency: her body is the thing that moved right
        after she acted, reliably more often than it moves on its own —
        and specifically the thing that goes where the command sends
        it. Candidacy needs ``action_effect_moves``: a thing that only
        happens to move when she acts is coupled to her, not her.

        The named self keeps its flag only while it keeps answering.
        Prolonged silence re-opens the question — a display element or
        a dead body's remnant can hold the flag while the real body
        goes on responding — but silence alone never fires her: if no
        candidate answers either, the incumbent stays (she may simply
        be pinned).
        """
        current = (
            self.entities.get(self.self_id)
            if self.self_id is not None
            else None
        )
        if current is not None and not current.gone:
            if self._self_silence < _SELF_MAX_SILENCE:
                return
        else:
            if current is not None:
                current.is_self = False
            current = None
            self.self_id = None

        best: WorldEntity | None = None
        best_score = 0.0
        for entity in self.entities.values():
            if (
                entity.gone
                or entity.part_of_self
                or entity.steps_after_action < _SELF_MIN_CONTINGENT
            ):
                continue
            rate = entity.action_moves / entity.steps_after_action
            free_rate = entity.n_moved_free / max(1, entity.n_seen)
            score = rate - free_rate * 0.5
            if (
                entity.action_effect_moves >= _SELF_MIN_CONTINGENT
                and score > best_score
            ):
                best, best_score = entity, score
        if best is None:
            # Nothing answers the command either — silence is not
            # divorce. Keep the incumbent; the flag stays only as
            # long as the entity itself does.
            if current is None:
                self.self_id = None
            return
        if best is current:
            return
        if current is not None:
            current.is_self = False
        best.is_self = True
        self.self_id = best.id
        self._self_silence = 0
        events.append(WorldEvent(
            "self_found",
            f"I am the color-{best.color} thing",
            salience=0.8,
            entity_id=best.id,
        ))

    def _update_parts(self, events: list[WorldEvent]) -> None:
        """Track which entities ride with her body as parts.

        A body need not be one component: a thing worn, held, or
        grown is contiguous with her and translates exactly when she
        does. Annexation needs consecutive rigid co-moves — a
        neighbor merely pacing alongside drifts off; a follower that
        matches her step for a moment still diverges. A part is
        dropped when it leaves her (gone, or left behind — no longer
        in contact after a move it did not share); a transform that
        keeps it attached does not cost annexation.
        """
        me = self.self_entity
        if me is None or me.gone:
            for entity in self.entities.values():
                if entity.part_of_self:
                    entity.part_of_self = False
                    entity.co_moves = 0
            return
        touching = set(me.cells)
        for r, c in me.cells:
            touching.update(
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        # Co-motion is only judged on a trustworthy measurement: a
        # churned self-motion is no basis to accuse a part of
        # diverging or to credit one for keeping up.
        moving = me.velocity != (0.0, 0.0) and me.velocity_exact
        for entity in self.entities.values():
            if entity.id == me.id:
                continue
            if entity.gone:
                if entity.part_of_self:
                    entity.part_of_self = False
                    entity.co_moves = 0
                    events.append(WorldEvent(
                        "part_lost",
                        f"the color-{entity.color} thing is no "
                        f"longer part of me",
                        salience=0.5,
                        entity_id=entity.id,
                    ))
                continue
            attached = bool(entity.cells & touching)
            if entity.part_of_self:
                if moving and not attached and entity.velocity != me.velocity:
                    entity.part_of_self = False
                    events.append(WorldEvent(
                        "part_lost",
                        f"the color-{entity.color} thing is no "
                        f"longer part of me",
                        salience=0.5,
                        entity_id=entity.id,
                    ))
                continue
            if not attached:
                entity.co_moves = 0
                continue
            if moving and entity.velocity == me.velocity:
                entity.co_moves += 1
                if entity.co_moves >= _PART_MIN_CO_MOVES:
                    entity.part_of_self = True
                    events.append(WorldEvent(
                        "part",
                        f"the color-{entity.color} thing is part "
                        f"of me",
                        salience=0.6,
                        entity_id=entity.id,
                    ))
            elif moving:
                entity.co_moves = 0

    def self_signature(self) -> tuple[PartSig, ...] | None:
        """What her body currently is — every part's (color, shape).

        The composite is the self-state: color pads rewrite one
        part's color, rotators its silhouette, so neither "color" nor
        "shape" alone names the state a gate examines. Canonical
        ordering keeps the key stable as parts come and go.
        """
        me = self.self_entity
        if me is None:
            return None
        sig = [(me.color, me.shape)]
        for entity in self.entities.values():
            if entity.part_of_self and not entity.gone:
                sig.append((entity.color, entity.shape))
        return tuple(sorted(sig))

    def self_footprint(self) -> frozenset[Cell]:
        """Every cell her body occupies — self plus annexed parts."""
        me = self.self_entity
        if me is None:
            return frozenset()
        cells = set(me.cells)
        for entity in self.entities.values():
            if entity.part_of_self and not entity.gone:
                cells.update(entity.cells)
        return frozenset(cells)

    def _cluster_ids(self, me: WorldEntity) -> set[int]:
        """Self plus parts — the ids that are her, not contacts."""
        return {
            e.id
            for e in self.entities.values()
            if e.id == me.id or (e.part_of_self and not e.gone)
        }

    def _attribute_action(
        self,
        action: str,
        events: list[WorldEvent],
        pending_self: int | None = None,
    ) -> None:
        """Record what the pending action did to her body."""
        self.action_tries[action] += 1
        if self.self_id is None:
            return
        if pending_self is not None and self.self_id != pending_self:
            # The body that acted is not the body answering — a self
            # re-adoption happened inside the trial. The delta this
            # frame shows belongs to a different entity's tracking,
            # so it says nothing about what the action does.
            return
        entity = self.entities.get(self.self_id)
        if entity is None:
            return
        if not entity.velocity_exact:
            # The footprint changed non-rigidly (occlusion, re-
            # segmentation): what the action did is unmeasurable this
            # trial. Counting a guessed delta would poison the modal
            # effect — an ambiguous trial is no evidence either way.
            return
        delta = (round(entity.velocity[0]), round(entity.velocity[1]))
        self.action_effects.setdefault(action, Counter())[delta] += 1
        if delta == (0, 0):
            self.action_noops[action] += 1

    def _attribute_click(
        self,
        action: str,
        target: Cell,
        scene: Scene,
        events: list[WorldEvent],
    ) -> None:
        """Record what a distal act at *target* did to the world.

        The distal analogue of contact attribution: she cannot touch
        the cell she acts on, so the evidence is what changed around
        it — a control answers locally. What she clicked *on* is the
        color that was there in the before-image; what it did is the
        count of cells the world moved within reach of the act. Both
        land on the kind (color), not the cell — the physics of
        "pressing a color-5 tile" transfer to every color-5 tile,
        which is the only generalization a first episode affords.
        """
        prev = self._last_grid
        if prev is None:
            return
        r0, c0 = int(target[0]), int(target[1])
        clicked_color = prev.get((r0, c0))
        changed = 0
        for dr in range(-_CLICK_RADIUS, _CLICK_RADIUS + 1):
            for dc in range(-_CLICK_RADIUS, _CLICK_RADIUS + 1):
                cell = (r0 + dr, c0 + dc)
                if cell in prev and scene.grid.at(*cell) != prev[cell]:
                    changed += 1
        key = (action, clicked_color)
        self.click_tries[key] += 1
        self.click_changes[key] += changed
        self._last_clicked = ((r0, c0), clicked_color)
        if changed:
            events.append(WorldEvent(
                "affected",
                f"acting at ({r0},{c0}) changed {changed} cells",
                salience=min(0.8, 0.3 + changed * 0.02),
            ))

    def is_display(self, entity: WorldEntity) -> bool:
        """Is this entity a readout rather than a thing in the
        world — a HUD element churning at the field's edge?"""
        return entity.is_display(self._grid_bounds)

    def click_affordance(self, action: str, color: int | None) -> float:
        """Mean cells a distal act at this kind has moved — its
        responsiveness. Zero is honest silence: probed or not, the
        world has never answered this press."""
        tries = self.click_tries.get((action, color), 0)
        if not tries:
            return 0.0
        return self.click_changes[(action, color)] / tries

    def board_plan(self) -> Cell | None:
        """The structured-scene machinery's next click, if it has one.

        When the screen parses as constraint boards and the induced
        rule says a press would move a violating element toward its
        asserted state, the answer is the screen cell to aim a
        coordinate action at. ``None`` when the scene holds no
        boards, no rule can be induced, or nothing would help — the
        same silence any unlearned mechanism returns.
        """
        return self.boards.plan()

    def click_responsiveness(self, color: int | None) -> float:
        """The best answer any distal act has gotten from this kind —
        action-agnostic, so the field can weigh a thing without
        caring which control she pressed it with."""
        best = 0.0
        for (a, c), tries in self.click_tries.items():
            if c == color and tries:
                best = max(best, self.click_changes[(a, c)] / tries)
        return best

    def attribute_terminal(self, outcome: str) -> list[WorldEvent]:
        """Credit whatever she was touching when the world ended.

        Terminal contacts are causal: a goal is the thing she reached,
        a hazard the thing that caught her. Called with the last
        pre-ending state still tracked — either from ``observe`` when
        the environment reports the outcome alongside its first new
        frame, or by a harness that learned of the end with no frame
        at all (then followed by ``reset``).
        """
        me = self._effector()
        if me is None:
            # No body to credit — the distal analogue of the contact
            # rule: whatever she last acted *at* was there when the
            # world ended. A click that ends an episode marks the
            # pressed kind, exactly as a fatal touch marks a hazard.
            if self._last_clicked is None:
                return []
            _cell, color = self._last_clicked
            if color is None:
                return []
            self.affordances.setdefault(color, Counter())[outcome] += 1
            return [WorldEvent(
                outcome,
                f"acting on the color-{color} thing ended the world"
                if outcome == OUT_HAZARD
                else f"acting on the color-{color} thing is the goal",
                salience=0.9,
            )]
        events = []
        credited: set[int] = set()
        for contact in self._contacts(me):
            credited.add(contact.color)
            self.affordances.setdefault(
                contact.color, Counter()
            )[outcome] += 1
            events.append(WorldEvent(
                outcome,
                f"the color-{contact.color} thing killed me"
                if outcome == OUT_HAZARD
                else f"the color-{contact.color} thing is the goal",
                salience=0.9,
                entity_id=me.id,
                other_id=contact.id,
            ))
        # The world can also end *on* a move — she steps onto the
        # goal and the level is gone before the contact is ever
        # observed. The cause then sits where her pending move was
        # carrying her: the cells she was entering when it ended.
        # Same rule convicts a hazard met on the way in. Floor and
        # her own body are not causes; contacts already credited are
        # not double-counted.
        pending = self._pending_action
        if pending is not None:
            expected = self.effect_of(pending)
            footprint = self.self_footprint()
            if expected not in (None, (0, 0)) and footprint:
                target = self._swept(footprint, expected) - footprint
                for entity in self.entities.values():
                    if (
                        entity.is_self
                        or entity.part_of_self
                        or entity.color in self.terrain_kinds
                        or entity.color in credited
                        or not entity.cells & target
                    ):
                        continue
                    credited.add(entity.color)
                    self.affordances.setdefault(
                        entity.color, Counter()
                    )[outcome] += 1
                    events.append(WorldEvent(
                        outcome,
                        f"the color-{entity.color} thing killed me"
                        if outcome == OUT_HAZARD
                        else f"the color-{entity.color} thing is the goal",
                        salience=0.9,
                        entity_id=me.id,
                        other_id=entity.id,
                    ))
        return events

    def suspect_terminal(
        self, pending: str | None = None
    ) -> list[WorldEvent]:
        """Hazard suspicion from an unannounced reset.

        The cause is unknown — the world never said what ended it —
        but contiguity is real evidence: whatever she was touching
        when the world ended was there when it did. The count lands
        in ``death_suspects``, where ``hazard_kind`` weighs it
        against everyday exposure — a wall she leans on constantly
        stays innocent while a thing seen mostly at endings convicts.
        """
        me = self._effector()
        if me is None:
            return []
        my_cells = self.self_footprint()
        events = []
        suspected: set[int] = set()
        for contact in self._contacts(me):
            if (
                contact.color in self.terrain_kinds
                or contact.cells > my_cells
            ):
                continue
            suspected.add(contact.color)
            self.death_suspects[contact.color] += 1
            events.append(WorldEvent(
                "suspect",
                f"the color-{contact.color} thing was touching me "
                f"when the world ended",
                salience=0.5,
                entity_id=me.id,
                other_id=contact.id,
            ))
        # A long stride can meet its killer inside the swept path
        # without the old frame ever showing adjacency — the world
        # ended before the contact frame existed. Credit the kinds
        # occupying where the pending action was carrying her, just
        # as a declared ending does.
        if pending is not None and my_cells:
            expected = self.effect_of(pending)
            if expected not in (None, (0, 0)):
                target = self._swept(my_cells, expected) - my_cells
                for entity in self.entities.values():
                    if (
                        entity.is_self
                        or entity.part_of_self
                        or entity.gone
                        or entity.color in self.terrain_kinds
                        or entity.color in suspected
                        or not entity.cells & target
                    ):
                        continue
                    suspected.add(entity.color)
                    self.death_suspects[entity.color] += 1
                    events.append(WorldEvent(
                        "suspect",
                        f"the color-{entity.color} thing was in my "
                        f"path when the world ended",
                        salience=0.5,
                        entity_id=me.id,
                        other_id=entity.id,
                    ))
        return events

    def hazard_kind(self, color: int) -> bool:
        """Is this kind deadly — confirmed kill or strong suspicion?

        Confirmed kills (explicit death/reset attribution) always
        count. Suspicion counts only when the kind is rarely touched
        *except* at endings: a wall she leans on every step is always
        "there when the world ended" yet plainly not a killer, while
        a thing seen mostly at death is dangerous in truth.
        """
        if self.affordance_of(color)[OUT_HAZARD] > 0:
            return True
        suspects = self.death_suspects[color]
        return suspects >= 3 and suspects * 2 > max(
            1, self.contact_ticks[color]
        )

    def _contact_outcomes(
        self,
        pending: str | None,
        prior_events: list[WorldEvent],
    ) -> list[WorldEvent]:
        """Attribute affordance outcomes to entities she touched.

        The attribution rule is uniform: whatever she was in contact
        with when an outcome befell her caused it. An entity that
        disappeared on contact is collectible; one sitting on the
        cell her action failed to enter is a blocker; one touching
        her on the tick her own state changed is a transformer.
        """
        events: list[WorldEvent] = []
        me = self._effector()
        if me is None:
            return events

        if me.gone:
            return events
        # Exposure bookkeeping — how often each kind is in contact
        # at all — is what makes a terminal co-occurrence mean
        # something (see hazard_kind).
        for contact in self._contacts(me):
            self.contact_ticks[contact.color] += 1
        cluster = self._cluster_ids(me)
        my_cells = set(me.cells)
        for entity in self.entities.values():
            if entity.part_of_self and not entity.gone:
                my_cells.update(entity.cells)
        adjacent = {
            (r + dr, c + dc)
            for r, c in my_cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        } - my_cells

        # Terrain exclusion: the medium she moves through fills
        # every destination she could enter, so crediting it as a
        # blocker would forbid all motion. It is where she is, not
        # what happened to her — it cannot block, be collected,
        # or be what transformed her. A pad she covers exactly is
        # an object, not the medium.
        substrates = {
            entity.id
            for entity in self.entities.values()
            if entity.color in self.terrain_kinds
            or entity.cells > my_cells
        }

        # When a known-effect action produced no motion, the blocker
        # is whatever sits in the tile she tried to cross — the
        # whole swept path, not only the destination strip: a wall
        # inside a stride refuses it and the blame belongs to the
        # wall's cells, not to empty floor at the far end.
        target_cells: set[Cell] = set()
        if pending is not None and me.velocity == (0.0, 0.0):
            expected = self.effect_of(pending)
            if expected not in (None, (0, 0)):
                target_cells = self._swept(my_cells, expected)
                sig = self.self_signature()
                if sig is not None:
                    key = (pending, frozenset(my_cells), sig)
                    tries, _last = self._move_failures.get(key, (0, 0))
                    self._move_failures[key] = (tries + 1, self.tick)

        blocked_hit = False
        my_sig = self.self_signature()
        for entity in self.entities.values():
            if entity.id in cluster or entity.id in substrates:
                continue
            if (
                entity.gone
                and entity.gone_tick == self.tick
                and entity.cells & (my_cells | adjacent)
            ):
                # Vanished on contact is only a *candidate* collection:
                # her own footprint covers what she steps on, and a
                # covered surface marking disappears exactly like a
                # consumed pickup. The verdict waits for the world —
                # _resolve_collections commits or drops it.
                self._pending_collect.append(_PendingCollect(
                    color=entity.color,
                    cells=entity.cells,
                    tick=self.tick,
                ))
            elif (
                not entity.gone
                and target_cells
                and entity.cells & target_cells
                and not entity.cells & my_cells
                and not (
                    # Proven-open things in the path aren't the wall
                    # — a collectible beside the blocker must not
                    # take the blocker's blame.
                    (my_sig is not None and my_sig in entity.open_states)
                    or self.affordance_of(entity.color)[OUT_COLLECT]
                    or self.affordance_of(entity.color)[OUT_TRANSFORM]
                    or self.affordance_of(entity.color)[OUT_WIN]
                    or self.affordance_of(entity.color)[OUT_NOTHING]
                )
            ):
                self.affordances.setdefault(entity.color, Counter())
                aff = self.affordances[entity.color]
                aff[OUT_BLOCK] += 1
                blocked_hit = True
                if my_sig is not None:
                    entity.blocked_states.add(my_sig)
                    entity.open_states.discard(my_sig)
                    self.blocked_kinds.setdefault(
                        entity.color, set()
                    ).add(my_sig)
                    # A refusal is a measurement taken under her
                    # current state — and part of that state may be
                    # hidden from her own sensors. The verdict binds
                    # only until something unexplained touches her:
                    # after that, what refused her may not refuse
                    # her now. Even a wall must be re-measured to
                    # keep its certainty.
                    entity.blocked_epochs[my_sig] = self._state_epoch
                    self._blocked_epoch[
                        (entity.color, my_sig)
                    ] = self._state_epoch
                self.known_blocked.update(entity.cells)
                events.append(WorldEvent(
                    "blocked",
                    f"the color-{entity.color} thing blocked me",
                    salience=0.5,
                    entity_id=me.id,
                    other_id=entity.id,
                ))

        # No entity explains the refusal — the space itself did.
        # The destination cells no live thing claims are the edge
        # of her world: mark them refused so the map, the field,
        # and the planner all stop routing through them. A strip
        # with exactly one unexplained cell is certain — that cell
        # is the wall. A wider strip is only a suspect: the same
        # cells must keep refusing to earn the mark, or floor
        # beside walls gets condemned into an unreachable island.
        if target_cells and not blocked_hit:
            occupied = {
                cell
                for e in self.entities.values()
                if not e.gone and e.id not in cluster
                for cell in e.cells
            }
            refused = target_cells - occupied - my_cells
            if len(refused) == 1:
                self.refused_cells.update(refused)
                self.known_blocked.update(refused)
                self.known_free.difference_update(refused)
            elif refused:
                for cell in refused:
                    self._refusal_suspects[cell] += 1
                    if self._refusal_suspects[cell] >= 2:
                        self.refused_cells.add(cell)
                        self.known_blocked.add(cell)
                        self.known_free.discard(cell)

        # Self-state change under contact: whatever touched her on
        # the tick part of her changed is what changed it. Every
        # contact is credited — including things consumed this tick,
        # which were in contact when it happened — and co-occurrence
        # over episodes sharpens which kind actually does it. The
        # observed from → to rewrite lands in ``transforms`` so the
        # change is simulatable, not just notable.
        transformed = [
            (e.last_transform[0], e.last_transform[1])
            for e in self.entities.values()
            if e.id in cluster and e.last_transform is not None
        ]
        if transformed:
            for contact in self._contacts(me, include_vanished=True):
                if contact.id in substrates:
                    continue
                self.affordances.setdefault(
                    contact.color, Counter()
                )[OUT_TRANSFORM] += 1
                for from_sig, to_sig in transformed:
                    self.transforms.setdefault(
                        contact.color, {}
                    ).setdefault(from_sig, Counter())[to_sig] += 1
                events.append(WorldEvent(
                    "transformed",
                    f"the color-{contact.color} thing changed me",
                    salience=0.7,
                    entity_id=me.id,
                    other_id=contact.id,
                ))
        return events

    def _learn_terrain(
        self, scene: Scene, pending: str | None
    ) -> None:
        """Learn which kinds are the medium she moves through.

        Perception never labels ground as ground — the floor is just
        another colored region. The evidence is what happens behind
        her: the cells she vacates are re-occupied by the medium.
        A region that reclaims her vacated footprint, and is much
        larger than she is, is the ground — not a thing she can
        encounter. Small tiles she crosses (pads, pickups) are
        objects: they are not large and they do not absorb space.

        The medium need not be an object at all: a background color
        that shows through where she stood is learned the same way.
        """
        footprint = self.self_footprint()
        if not footprint and pending is not None:
            effector = self._effector()
            if effector is not None:
                footprint = effector.cells
        prev = self._prev_footprint
        self._prev_footprint = footprint
        if not footprint or prev is None:
            return
        vacated = prev - footprint
        if not vacated:
            return
        reclaimed = Counter(
            scene.grid.at(r, c) for r, c in vacated
        )
        for color, cells in reclaimed.items():
            if color == scene.background and cells * 2 >= len(vacated):
                self.terrain_kinds.add(color)
        for entity in self.entities.values():
            if (
                entity.gone
                or entity.is_self
                or entity.part_of_self
                or len(entity.cells) < 4 * len(footprint)
            ):
                continue
            if len(entity.cells & vacated) * 2 >= len(vacated):
                self.terrain_kinds.add(entity.color)

    def _update_map(self, scene: Scene) -> None:
        """Maintain the occupancy map: free medium vs. solid things.

        ``known_free`` means "this cell is made of a medium observed
        to admit her" — not "nothing is drawn here." Unclaimed pixels
        are unexplored space; crossing them is how a refusal is
        measured. Treating every blank cell as walkable invents a map
        through walls and voids before either exists.
        """
        occupied = {
            cell
            for e in self.entities.values()
            if not e.gone and e.color not in self.terrain_kinds
            for cell in e.cells
        }
        # A refusal is measured truth only while the evidence stands:
        # a cell she now demonstrably occupies cannot have refused
        # her — the mark is stale, evict it. (This is how a map kept
        # across attempts stays honest when the world changes.)
        stood = self.self_footprint()
        if stood:
            self.known_free.update(stood)
            self.refused_cells.difference_update(stood)
            for cell in stood:
                self._refusal_suspects.pop(cell, None)
        for r, c, color in scene.grid.iter_cells():
            cell = (r, c)
            if (
                color not in self.terrain_kinds
                or cell in occupied
                or cell in self.refused_cells
            ):
                continue
            self.known_free.add(cell)
            self.known_blocked.discard(cell)
        sig = self.self_signature()
        for entity in self.entities.values():
            if (
                entity.gone
                or entity.is_self
                or entity.part_of_self
                or entity.color in self.terrain_kinds
            ):
                continue
            # Passability is learned, not assumed — and it is
            # conditional: a kind is map-solid only for the
            # self-state it has actually refused, never across
            # states. The gate that stopped one body condemns
            # nothing for the body a station just gave her; pure
            # hazards stay condemned for every state.
            refused_here = (
                sig is not None
                and sig in self.blocked_kinds.get(entity.color, ())
                and sig not in self.open_kinds.get(entity.color, ())
                and (
                    # An unmarked or unconditional verdict always
                    # binds; a conditional one binds only in the
                    # epoch it was measured.
                    self._blocked_epoch.get((entity.color, sig))
                    in (None, self._state_epoch)
                )
            )
            if refused_here or self.hazard_kind(entity.color):
                self.known_blocked.update(entity.cells)
                self.known_free.difference_update(entity.cells)
            elif entity.spontaneous:
                # Things that move on their own are not terrain, but
                # touching them may still be dangerous — affordance
                # decides, the map just records presence.
                self.known_free.difference_update(entity.cells)
            else:
                # Not refusing her current state: the cells' admission
                # is the entity's business (passable), not the map's —
                # evict whatever a former self-state had condemned.
                self.known_blocked.difference_update(entity.cells)

    def _update_passability(self) -> None:
        """Record which self-states can occupy an entity's cells.

        Standing on a live entity's footprint is the positive proof
        of conditional passability — the complement of a failed move.
        """
        me = self.self_entity
        if me is None or me.gone:
            return
        sig = self.self_signature()
        if sig is None:
            return
        my_cells = set(me.cells)
        for part in self.entities.values():
            if part.part_of_self and not part.gone:
                my_cells.update(part.cells)
        cluster = self._cluster_ids(me)
        touched: set[int] = set()
        for entity in self.entities.values():
            # Things that vanished under her this very tick were in
            # contact when they went — the tile she is standing on
            # reads exactly like one she consumed.
            if entity.id in cluster or (
                entity.gone and entity.gone_tick != self.tick
            ):
                continue
            if entity.cells & my_cells:
                entity.open_states.add(sig)
                # Occupation is the later word: a refusal this
                # thing once gave her is answered by her being here.
                entity.blocked_states.discard(sig)
                entity.blocked_epochs.pop(sig, None)
                self.open_kinds.setdefault(entity.color, set()).add(sig)
                touched.add(entity.color)
        # Touching something unexplained may have changed her in
        # ways her sensors do not report — hidden state exists in
        # some worlds. Each *new* unexplained contact opens a fresh
        # epoch: verdicts measured under the old one stay recorded
        # but stop binding until re-proven.
        # A contact is fresh only after an absence longer than the
        # flicker window — stepping off a station and back on is a
        # new touch; a thing dropping out of sight for a tick is not.
        newly = {
            color for color in touched
            if self.tick - self._contact_kinds.get(
                color, -_CONTACT_FRESH_AFTER - 1
            ) > _CONTACT_FRESH_AFTER
        }
        for color in touched:
            self._contact_kinds[color] = self.tick
        # "Unexplained" here means no evidence that a real outcome
        # occurred: a kind seen only to do *nothing visible* may
        # still have moved hidden state, and a station of that sort
        # acts on every fresh contact, not only the first.
        unexplained = {
            color for color in newly
            if not (set(self.affordance_of(color)) - {OUT_NOTHING})
        }
        if unexplained:
            self._state_epoch += 1
            self._state_changers.update(unexplained)

    def passable(
        self, entity: WorldEntity, self_sig: tuple[PartSig, ...]
    ) -> bool:
        """Would contact with this entity let her enter its cells?

        Hazards are never entered. Entities with a known positive
        affordance (collect / transform / goal) are entered by
        definition — the affordance IS what contact does. For the
        rest, passability is a learned (kind, self-state) relation:
        a pair never observed is optimistically open — the only way
        to learn that a gate opens for a self-state is to try it.
        """
        aff = self.affordance_of(entity.color)
        # Observed evidence about THIS entity dominates kind priors.
        if self_sig in entity.open_states:
            return True
        if self_sig in entity.blocked_states and (
            entity.blocked_epochs.get(self_sig)
            in (None, self._state_epoch)
        ):
            return False
        if self.hazard_kind(entity.color):
            return False
        if (
            aff[OUT_COLLECT] > 0
            or aff[OUT_TRANSFORM] > 0
            or aff[OUT_WIN] > 0
        ):
            return True
        if aff[OUT_NOTHING] > 0:
            # Walked on before without consequence — surface, not
            # wall. Kind-mixed gates keep their per-entity answer
            # via the open/blocked states checked above.
            return True
        if self_sig in self.blocked_kinds.get(entity.color, ()):
            # This kind has refused contact *in this self-state* —
            # the evidence transfers to unbumped members of the
            # kind under the refused signature (a wall need not
            # be re-probed piece by piece), but not across states:
            # an unobserved pair stays optimistically open, the
            # only way a conditional gate is ever found open. And
            # a verdict whose refusal announced a state-change binds
            # only until she next touches the unexplained — hidden
            # state may have moved since it was measured.
            if self_sig not in self.open_kinds.get(entity.color, ()) and (
                self._blocked_epoch.get((entity.color, self_sig))
                in (None, self._state_epoch)
            ):
                return False
            # The kind is mixed under this very state — members
            # differ, so this member is judged by what it alone
            # has shown: nothing yet.
        return True

    def _effector(self) -> WorldEntity | None:
        """The presumed self for contact attribution.

        Confirmed self if known; otherwise whatever moved with the
        pending action — contact outcomes can't wait for agency to
        be statistically confirmed, or the first touches are lost.
        """
        if self.self_id is not None:
            return self.entities.get(self.self_id)
        best: WorldEntity | None = None
        for entity in self.entities.values():
            if entity.gone or entity.velocity == (0.0, 0.0):
                continue
            if entity.action_moves >= 1 and (
                best is None or entity.action_moves > best.action_moves
            ):
                best = entity
        return best

    def _contacts(
        self, me: WorldEntity, include_vanished: bool = False
    ) -> list[WorldEntity]:
        """Every non-self entity in contact with her footprint.

        ``include_vanished`` also considers entities that vanished on
        this very tick — consumed things still count as contacts for
        the outcomes they coincided with.
        """
        footprint = set(me.cells)
        for entity in self.entities.values():
            if entity.part_of_self and not entity.gone:
                footprint.update(entity.cells)
        adjacent = {
            (r + dr, c + dc)
            for r, c in footprint
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        }
        contact = footprint | adjacent
        cluster = self._cluster_ids(me)
        return [
            entity
            for entity in self.entities.values()
            if entity.id not in cluster
            and (
                not entity.gone
                or (include_vanished and entity.gone_tick == self.tick)
            )
            and entity.cells & contact
        ]

    # ─── Learned semantics ──────────────────────────────────────

    @property
    def self_entity(self) -> WorldEntity | None:
        if self.self_id is None:
            return None
        return self.entities.get(self.self_id)

    def effect_of(self, action: str) -> Cell | None:
        """What an action does to her body when it isn't obstructed.

        A blocked attempt yields (0,0) — that's an outcome, not the
        action's semantics. The learned effect is the modal nonzero
        displacement; (0,0) only when every attempt failed. Pure
        observation — interface-convention priors never enter here.
        """
        counts = self.action_effects.get(action)
        if not counts:
            return None
        nonzero = {k: v for k, v in counts.items() if k != (0, 0)}
        if nonzero:
            return max(nonzero.items(), key=lambda kv: kv[1])[0]
        return (0, 0)

    def learn_action_label(self, action: str, label: str) -> None:
        """Record the canonical label an interface gives an action.

        The label is the equivalent of a printed button: a prior
        about what the action might do, held separately from learned
        effects. A directional label seeds a displacement guess;
        "coordinate" marks the action as taking a target cell.
        Experience overrides it wholesale.
        """
        if action in self.action_labels:
            return
        self.action_labels[action] = label
        delta = _LABEL_DISPLACEMENTS.get(label)
        if delta is not None:
            self.action_priors[action] = delta
        if label == "coordinate":
            self.coordinate_actions.add(action)

    def expected_effect(self, action: str) -> Cell | None:
        """Best current guess of the action's bodily effect.

        Observed effect if the action has been tried; otherwise the
        interface label's prior. Used for prediction and exploration
        hypotheses — never as learned fact.
        """
        effect = self.effect_of(action)
        if effect is not None:
            return effect
        return self.action_priors.get(action)

    def action_for_displacement(self, dr: int, dc: int) -> str | None:
        """Which known action moves her (dr, dc)? Learned, not fixed."""
        for action in self.action_effects:
            if self.effect_of(action) == (dr, dc):
                return action
        return None

    def affordance_of(self, color: int) -> Counter[str]:
        return self.affordances.get(color, Counter())

    def recalled_valence_of(self, color: int) -> float:
        """What episodic memory feels about this kind right now.

        Positive — remembered goals pull; negative — remembered
        hazards and frustrations repel. Written by the recall pass
        in the embodiment layer (pattern completion over her own
        event vocabulary), read by the motor field as one more
        voice in the field — never a command.
        """
        return self.recalled_valence.get(color, 0.0)

    def action_understood(self, action: str) -> bool:
        """Enough trials to know what the action does to her —
        and no evidence the entry has gone stale."""
        if self.action_tries[action] < _SELF_MIN_CONTINGENT:
            return False
        return not self.action_miscalibrated(action)

    def action_miscalibrated(self, action: str) -> bool:
        """The cerebellum's verdict: this action keeps surprising.

        Nominal experience is not comprehension — a body whose model
        of a move keeps being wrong has more babbling to do. Without
        a comparator wired in, stale-ness is undetectable, so the
        answer is no.
        """
        comparator = self.motor_comparator
        if comparator is None:
            return False
        try:
            return bool(comparator.miscalibrated(action))
        except Exception:  # noqa: BLE001 — advisory machinery
            return False

    # ─── Forward model ──────────────────────────────────────────

    def start_state(self) -> SimState | None:
        """The world as it stands, as a plannable state."""
        me = self.self_entity
        if me is None:
            return None
        sig = self.self_signature()
        if sig is None:
            return None
        return SimState(
            cells=self.self_footprint(),
            signature=sig,
            gone=frozenset(
                e.id for e in self.entities.values() if e.gone
            ),
        )

    def _swept(
        self, cells: frozenset[Cell] | set[Cell], effect: Cell
    ) -> set[Cell]:
        """Cells a stride crosses, destination included.

        A multi-cell stride is atomic in these worlds: obstruction
        anywhere along it refuses the whole move — the world does
        not slide her partway. So refusal evidence, and simulated
        passability, belong to the path the footprint travels, not
        only to the strip it lands on.
        """
        dr, dc = effect
        steps = max(abs(dr), abs(dc))
        out: set[Cell] = set()
        for k in range(1, steps + 1):
            for r, c in cells:
                out.add(
                    (
                        r + int(dr * k / steps),
                        c + int(dc * k / steps),
                    )
                )
        return out

    def simulate(self, state: SimState, action: str) -> SimState | None:
        """One forward step through learned physics.

        Applies the action's learned displacement to the simulated
        self, then resolves contact with whatever the new footprint
        would overlap — the same outcomes the real world applies:
        blocked moves fail, hazards end the line, collectibles are
        consumed, transformers rewrite a part's state. ``None``
        means the action's effect is unknown or the move fails in
        this state — either way there is nothing to plan on.

        A stride fails if *any* cell along its path is impassable —
        the sweep is checked, not only the destination: a wall
        inside the jump refuses the jump.
        """
        effect = self.effect_of(action)
        if effect is None or effect == (0, 0):
            return None
        failure = self._move_failures.get(
            (action, state.cells, state.signature)
        )
        if (
            failure is not None
            and failure[0] >= _MOVE_FAILURE_RETRIES
            and self.tick - failure[1] < _MOVE_FAILURE_COOLDOWN
        ):
            return None
        new_cells = frozenset(
            (r + effect[0], c + effect[1]) for r, c in state.cells
        )
        height, width = self._grid_bounds
        if height and width and any(
            not (0 <= r < height and 0 <= c < width) for r, c in new_cells
        ):
            return None
        swept = self._swept(state.cells, effect)
        gone = set(state.gone)
        new_sig = state.signature
        cluster = {
            e.id for e in self.entities.values()
            if e.is_self or e.part_of_self
        }
        hits: list[WorldEntity] = []
        seen: set[int] = set()
        index = self._cell_entities
        if index is not None:
            for cell in swept:
                for eid in index.get(cell, ()):
                    if (
                        eid in seen
                        or eid in state.gone
                        or eid in cluster
                    ):
                        continue
                    seen.add(eid)
                    entity = self.entities.get(eid)
                    if entity is not None and entity.cells & swept:
                        hits.append(entity)
        else:
            for entity in self.entities.values():
                if entity.id in state.gone or entity.id in cluster:
                    continue
                if entity.cells & swept:
                    hits.append(entity)
        # Swept cells no entity claims have no passability answer
        # from the entity ledger — the map's refusal marks are the
        # only evidence about them. Entity-covered cells defer to
        # the entity's own conditional passability.
        claimed = {cell for e in hits for cell in e.cells & swept}
        unclaimed = swept - claimed
        if (
            unclaimed & self.known_blocked
            or unclaimed - self.known_free
        ):
            return None
        for entity in hits:
            # Static is not the same as floor: a wall never moves
            # and is still a wall. Terrain and larger-than-her
            # entities count as obstacles unless contact evidence
            # says her state enters them.
            if (
                entity.color in self.terrain_kinds
                or entity.cells > new_cells
            ):
                if not self.passable(entity, new_sig):
                    return None
                continue
            if not self.passable(entity, new_sig):
                return None
            # Contact outcomes land at the destination — the tile
            # she finishes on. Something collected only in passing
            # is not the game's semantics: the stride ends where
            # it ends.
            if not entity.cells & new_cells:
                continue
            aff = self.affordance_of(entity.color)
            if aff[OUT_COLLECT] > 0:
                gone.add(entity.id)
            if aff[OUT_TRANSFORM] > 0:
                mapping = self.transforms.get(entity.color)
                if mapping:
                    new_sig = tuple(sorted(
                        mapping[part].most_common(1)[0][0]
                        if mapping.get(part)
                        else part
                        for part in new_sig
                    ))
        return SimState(cells=new_cells, signature=new_sig,
                        gone=frozenset(gone))

    # ─── Episodes ───────────────────────────────────────────────

    def reset(
        self,
        keep_learning: bool = True,
        keep_map: bool = True,
    ) -> None:
        """Start a new episode.

        With ``keep_learning`` the positions are forgotten but the
        physics are not. ``keep_map`` distinguishes a retry from a
        transition: death in the same maze remembers which cells were
        wall, floor, or refused; a won level is a new layout, where
        carrying old cell verdicts would build walls out of memory.
        ``keep_learning=False`` is a new world entirely: everything
        learned, map included, is wiped.
        """
        self.tick = 0
        self.entities.clear()
        self._cell_entities.clear()
        self.self_id = None
        self._pending_action = None
        self._pending_self_id = None
        self._pending_target = None
        self._last_clicked = None
        self._last_grid = None
        self._prev_footprint = None
        self._pending_collect.clear()
        self._self_silence = 0
        self._contact_kinds.clear()
        # A respawn is itself a hidden-state transition: whatever
        # the world does to her on death, conditional verdicts
        # measured before it were taken under a different state.
        self._state_epoch += 1
        if not keep_map:
            self.known_free.clear()
            self.known_blocked.clear()
            self.refused_cells.clear()
            self._refusal_suspects.clear()
            self._move_failures.clear()
            self.visits.clear()
        if not keep_learning:
            self.action_labels.clear()
            self.action_priors.clear()
            self.coordinate_actions.clear()
            self.action_effects.clear()
            self.action_tries.clear()
            self.action_noops.clear()
            self._move_failures.clear()
            self.affordances.clear()
            self.click_tries.clear()
            self.click_changes.clear()
            self.transforms.clear()
            self.contact_ticks.clear()
            self.death_suspects.clear()
            self.terrain_kinds.clear()
            self.blocked_kinds.clear()
            self.open_kinds.clear()
            self._blocked_epoch.clear()
            self._state_epoch = 0
            self._state_changers.clear()
            self.boards = ConstraintSession()
            self.known_free.clear()
            self.known_blocked.clear()
            self.refused_cells.clear()
            self._refusal_suspects.clear()
            self.visits.clear()
            self.recalled_valence.clear()
            if self.motor_comparator is not None:
                try:
                    self.motor_comparator.reset()
                except Exception:  # noqa: BLE001 — advisory machinery
                    pass

    def note_action(self, action: str, target: Cell | None = None) -> None:
        """Record the action just taken; the next observation is its effect.

        ``target`` is the cell a coordinate-typed action was aimed
        at — the distal analogue of contact: the world's answer is
        attributed to what was *there*, not only to the action id.
        """
        self._pending_action = action
        self._pending_self_id = self.self_id
        self._pending_target = (
            (int(target[0]), int(target[1])) if target is not None else None
        )
        if target is not None:
            # If the boards machinery planned this press, charge it —
            # its bookkeeping only counts a click it proposed.
            self.boards.note_click((int(target[0]), int(target[1])))

    def transition_state(self) -> dict[str, Any]:
        """The model flattened to named scalar variables.

        This is the state vector the shared ``TaskCompetence`` layer
        sees — hypothesis verification, goal inference, and skill
        matching all speak in flat variable names. Values are
        world-model facts only; agent-side bindings (which color is
        her, which cells were visited) belong to the episodic map,
        not the schema signature.
        """
        me = self.self_entity
        live = [
            e for e in self.entities.values()
            if not e.gone and not e.part_of_self
        ]
        aff_kind: dict[str, int] = {}
        for e in live:
            if e.is_self:
                continue
            for outcome, n in self.affordance_of(e.color).items():
                if n > 0:
                    aff_kind[f"kinds.{outcome}"] = (
                        aff_kind.get(f"kinds.{outcome}", 0) + 1
                    )
        return {
            "self.r": me.centroid[0] if me is not None else None,
            "self.c": me.centroid[1] if me is not None else None,
            "self.color": me.color if me is not None else None,
            "self.parts": sum(
                1
                for e in self.entities.values()
                if e.part_of_self and not e.gone
            ),
            "embodied": me is not None and not me.gone,
            "entities.live": sum(1 for e in live if not e.is_self),
            "entities.gone": sum(
                1 for e in self.entities.values() if e.gone
            ),
            "map.free": len(self.known_free),
            "map.blocked": len(self.known_blocked),
            **aff_kind,
        }

    def summary(self) -> str:
        """Plain-language state — the material for memory and speech."""
        parts = []
        me = self.self_entity
        if me is not None:
            parts.append(f"I am the color-{me.color} thing")
        live = [e for e in self.entities.values() if not e.gone]
        if live:
            parts.append(
                f"{len(live)} thing{'s' if len(live) != 1 else ''} around me"
            )
        for color, aff in sorted(self.affordances.items()):
            if not aff:
                continue
            out = aff.most_common(1)[0][0]
            parts.append(f"color-{color} things {out} me"
                         if out != OUT_COLLECT
                         else f"color-{color} things can be collected")
        return "; ".join(parts) if parts else "an unfamiliar place"
