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
_RESHAPE_MIN_FRACTION = 0.15  # changed-cells share that counts as reshaping
_COLLECT_CONFIRM = 3          # ticks a vanished contact must stay gone


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
    # When a match rewrote this entity's state this tick — recolor,
    # reshape, rotation — the (from → to) pair, for transform
    # attribution. ``None`` on an unremarkable observation.
    last_transform: tuple[PartSig, PartSig] | None = None

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

        # Learned knowledge — survives episode resets.
        self.action_effects: dict[str, Counter[Cell]] = {}
        self.action_tries: Counter[str] = Counter()
        self.action_noops: Counter[str] = Counter()
        self.affordances: dict[int, Counter[str]] = {}
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
        # sharing its color.

        # Episodic map — cleared on reset.
        self.known_free: set[Cell] = set()
        self.known_blocked: set[Cell] = set()
        # Cells whose ownership is unexplained — no entity claimed
        # them — yet a move into them failed. The boundary of the
        # world is not an object, but it refuses entry all the same.
        # Free-space marking must not resurrect them.
        self.refused_cells: set[Cell] = set()
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

        # Consecutive action-observations the current self failed
        # to answer. Above the limit she is re-judged — a body that
        # stops answering her commands is evidence it was never
        # hers (a display element can inherit the flag through
        # re-segmentation, a dead avatar leaves none).
        self._self_silence = 0

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
                self.reset(keep_learning=True)

        self.tick += 1
        pending = self._pending_action
        self._pending_action = None

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
            events.extend(self.suspect_terminal())
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
                for cell in obj.cells:
                    self.known_free.add(cell)
                    self.visits[cell] += 1

        # Prediction check: what did the pending action do to her?
        # Runs after velocities are measured — comparing prediction
        # against last frame's motion answers the wrong question.
        # Only a confidently-measured motion can disappoint a
        # prediction — a non-rigid footprint means the observation
        # itself is ambiguous, not surprising.
        if pending is not None and self.self_id is not None:
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
                        events.append(WorldEvent(
                            "unexpected",
                            f"action {pending} moved me {observed}, "
                            f"expected {expected}",
                            salience=0.5,
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
            self._attribute_action(pending, events)
        self._update_parts(events)
        self._learn_terrain()
        self._update_map(scene)
        self._update_passability()
        events.extend(self._contact_outcomes(pending, events))

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
        self, action: str, events: list[WorldEvent]
    ) -> None:
        """Record what the pending action did to her body."""
        self.action_tries[action] += 1
        if self.self_id is None:
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
            return []
        events = []
        for contact in self._contacts(me):
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
        return events

    def suspect_terminal(self) -> list[WorldEvent]:
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
        for contact in self._contacts(me):
            if (
                contact.color in self.terrain_kinds
                or contact.cells > my_cells
            ):
                continue
            self.death_suspects[contact.color] += 1
            events.append(WorldEvent(
                "suspect",
                f"the color-{contact.color} thing was touching me "
                f"when the world ended",
                salience=0.5,
                entity_id=me.id,
                other_id=contact.id,
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
        # is whatever sits on the cell she tried to enter — her cells
        # shifted by the expected displacement.
        target_cells: set[Cell] = set()
        if pending is not None and me.velocity == (0.0, 0.0):
            expected = self.effect_of(pending)
            if expected not in (None, (0, 0)):
                target_cells = {
                    (r + expected[0], c + expected[1]) for r, c in my_cells
                }

        blocked_hit = False
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
            ):
                self.affordances.setdefault(entity.color, Counter())
                aff = self.affordances[entity.color]
                aff[OUT_BLOCK] += 1
                blocked_hit = True
                sig = self.self_signature()
                if sig is not None:
                    entity.blocked_states.add(sig)
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
        # and the planner all stop routing through them.
        if target_cells and not blocked_hit:
            occupied = {
                cell
                for e in self.entities.values()
                if not e.gone and e.id not in cluster
                for cell in e.cells
            }
            refused = target_cells - occupied - my_cells
            if refused:
                self.refused_cells.update(refused)
                self.known_blocked.update(refused)
                self.known_free.difference_update(refused)

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

    def _learn_terrain(self) -> None:
        """Learn which kinds are the medium she moves through.

        Perception never labels ground as ground — the floor is just
        another colored region. The evidence is what happens behind
        her: the cells she vacates are re-occupied by the medium.
        A region that reclaims her vacated footprint, and is much
        larger than she is, is the ground — not a thing she can
        encounter. Small tiles she crosses (pads, pickups) are
        objects: they are not large and they do not absorb space.
        """
        footprint = self.self_footprint()
        prev = self._prev_footprint
        self._prev_footprint = footprint
        if not footprint or prev is None:
            return
        vacated = prev - footprint
        if not vacated:
            return
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
        """Maintain the occupancy map: free floor vs. solid things."""
        occupied = {
            cell
            for e in self.entities.values()
            if not e.gone and e.color not in self.terrain_kinds
            for cell in e.cells
        }
        for r, c, _color in scene.grid.iter_cells():
            cell = (r, c)
            if cell in occupied or cell in self.refused_cells:
                continue
            self.known_free.add(cell)
            self.known_blocked.discard(cell)
        for entity in self.entities.values():
            if (
                entity.gone
                or entity.is_self
                or entity.part_of_self
                or entity.color in self.terrain_kinds
            ):
                continue
            aff = self.affordance_of(entity.color)
            # Passability is learned, not assumed: a thing is
            # impassable only once touching it has actually blocked
            # or killed her. A motionless thing might be a wall —
            # or a gem sitting still. Static ≠ solid.
            if aff[OUT_BLOCK] > 0 or self.hazard_kind(entity.color):
                self.known_blocked.update(entity.cells)
                self.known_free.difference_update(entity.cells)
            elif entity.spontaneous:
                # Things that move on their own are not terrain, but
                # touching them may still be dangerous — affordance
                # decides, the map just records presence.
                self.known_free.difference_update(entity.cells)

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
        for entity in self.entities.values():
            if entity.id in cluster or entity.gone:
                continue
            if entity.cells & my_cells:
                entity.open_states.add(sig)

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
        if self_sig in entity.blocked_states:
            return False
        if self.hazard_kind(entity.color):
            return False
        if (
            aff[OUT_COLLECT] > 0
            or aff[OUT_TRANSFORM] > 0
            or aff[OUT_WIN] > 0
        ):
            return True
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

    def action_understood(self, action: str) -> bool:
        """Enough trials to know what the action does to her."""
        return self.action_tries[action] >= _SELF_MIN_CONTINGENT

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

    def simulate(self, state: SimState, action: str) -> SimState | None:
        """One forward step through learned physics.

        Applies the action's learned displacement to the simulated
        self, then resolves contact with whatever the new footprint
        would overlap — the same outcomes the real world applies:
        blocked moves fail, hazards end the line, collectibles are
        consumed, transformers rewrite a part's state. ``None``
        means the action's effect is unknown or the move fails in
        this state — either way there is nothing to plan on.
        """
        effect = self.effect_of(action)
        if effect is None or effect == (0, 0):
            return None
        new_cells = frozenset(
            (r + effect[0], c + effect[1]) for r, c in state.cells
        )
        gone = set(state.gone)
        new_sig = state.signature
        cluster = {
            e.id for e in self.entities.values()
            if e.is_self or e.part_of_self
        }
        hits: list[WorldEntity] = []
        for entity in self.entities.values():
            if entity.id in state.gone or entity.id in cluster:
                continue
            if entity.cells & new_cells:
                hits.append(entity)
        # Destination cells no entity claims have no passability
        # answer from the entity ledger — the map's refusal marks
        # are the only evidence about them. Entity-covered cells
        # defer to the entity's own conditional passability.
        claimed = {
            cell for e in hits for cell in e.cells & new_cells
        }
        if (new_cells - claimed) & self.known_blocked:
            return None
        for entity in hits:
            if (
                entity.color in self.terrain_kinds
                or entity.cells > new_cells
            ):
                continue
            if not self.passable(entity, new_sig):
                return None
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

    def reset(self, keep_learning: bool = True) -> None:
        """Start a new episode.

        With ``keep_learning`` the positions are forgotten but the
        physics are not — action effects, affordances, and tried
        counts carry across levels, which is what transfer means.
        """
        self.tick = 0
        self.entities.clear()
        self.self_id = None
        self._pending_action = None
        self._prev_footprint = None
        self.known_free.clear()
        self.known_blocked.clear()
        self.refused_cells.clear()
        self.visits.clear()
        self._pending_collect.clear()
        self._self_silence = 0
        if not keep_learning:
            self.action_effects.clear()
            self.action_tries.clear()
            self.action_noops.clear()
            self.affordances.clear()
            self.transforms.clear()
            self.contact_ticks.clear()
            self.death_suspects.clear()
            self.terrain_kinds.clear()

    def note_action(self, action: str) -> None:
        """Record the action just taken; the next observation is its effect."""
        self._pending_action = action

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
