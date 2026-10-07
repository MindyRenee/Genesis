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

from ..spatial.scene import Scene

Cell = tuple[int, int]

# Outcome kinds accumulated in ``affordances[color]``.
OUT_COLLECT = "collect"      # touched entity vanished — a pickup
OUT_BLOCK = "block"          # contact prevented movement — a wall
OUT_RECOLOR = "recolor"      # contact changed her color — a key/state changer
OUT_HAZARD = "hazard"        # contact preceded a world reset — deadly
OUT_WIN = "win"              # contact preceded a level win

# Observation bookkeeping.
_MAX_TRACK_DIST = 6.0         # centroid distance for same-entity match
_GONE_AFTER = 1               # misses before an entity is declared gone
_SELF_MIN_CONTINGENT = 2      # action-contingent moves before self is named
_STATIC_AFTER = 6             # observations unmoved → counts as terrain
_MAX_OBJECT_FRACTION = 0.66   # ≥⅔ of the field is fill, not figure

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
    body is, what color it is (transformer contact changes it —
    that is what keys are), and which entities have been consumed.
    Frozen and hashable so planners can deduplicate visited states.
    """

    cells: frozenset[Cell]
    color: int
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
    n_seen: int = 1
    n_moved_free: int = 0       # moved while no action was pending
    action_moves: int = 0       # moved on the step right after her action
    steps_after_action: int = 0
    is_self: bool = False
    gone: bool = False
    gone_tick: int = -1         # observation when it vanished
    # Conditional passability, observed per entity: which of her
    # colors this thing has stopped, and which it has admitted. A
    # gate is a thing that blocks under some self-states and not
    # others — evidence about this entity, not its whole kind.
    blocked_colors: set[int] = field(default_factory=set)
    open_colors: set[int] = field(default_factory=set)
    # If contact transformed her, what she became — the observed
    # result, which need not equal the entity's own color.
    recolor_result: int | None = None

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

        # Interface conventions — priors, not truth. A harness may
        # label an action id with the canonical semantic the
        # interface advertises ("up", "coordinate", ...); the label
        # is a hypothesis that stands only until the action's real
        # effect is observed. ``action_effects`` stays pure evidence.
        self.action_labels: dict[str, str] = {}
        self.action_priors: dict[str, Cell] = {}
        self.coordinate_actions: set[str] = set()

        # Conditional passability lives on the entities themselves
        # (WorldEntity.blocked_colors / open_colors): "that gate
        # stopped me" is evidence about that gate, not everything
        # sharing its color.

        # Episodic map — cleared on reset.
        self.known_free: set[Cell] = set()
        self.known_blocked: set[Cell] = set()
        self.visits: Counter[Cell] = Counter()

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
        self.tick += 1
        events: list[WorldEvent] = []
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

        # Prediction check: what did the pending action do to her?
        if pending is not None and self.self_id is not None:
            self_ent = self.entities.get(self.self_id)
            if self_ent is not None and not self_ent.gone:
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

        for entity, obj in matched:
            dr = obj.centroid[0] - entity.centroid[0]
            dc = obj.centroid[1] - entity.centroid[1]
            moved = entity.cells != obj.cells
            if pending is not None:
                entity.steps_after_action += 1
                if moved:
                    entity.action_moves += 1
            elif moved:
                entity.n_moved_free += 1
            entity.velocity = (dr, dc)
            entity.cells = obj.cells
            entity.centroid = obj.centroid
            entity.last_seen = self.tick
            entity.n_seen += 1
            if moved:
                events.append(WorldEvent(
                    "moved",
                    f"entity {entity.id} moved to {entity.centroid}",
                    entity_id=entity.id,
                ))
            if entity.is_self:
                for cell in obj.cells:
                    self.known_free.add(cell)
                    self.visits[cell] += 1

        for obj in new_objects:
            recolored = self._recolor_match(obj)
            if recolored is not None:
                events.append(WorldEvent(
                    "recolored",
                    f"entity {recolored.id} changed color to {obj.color}",
                    salience=0.6,
                    entity_id=recolored.id,
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

        # Agency before attribution: the steps that reveal which
        # entity is hers are also evidence of what the action does.
        self._infer_self(events)
        if pending is not None:
            self._attribute_action(pending, events)
        self._update_map(scene)
        self._update_passability()
        events.extend(self._contact_outcomes(pending, events, info))

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
        """Reclaim an entity that changed color in place.

        A recolor looks like a vanish + an appear; if the new object
        sits where a just-lost entity was and shares its shape, it is
        the same thing wearing a different color.
        """
        for entity in self.entities.values():
            if entity.gone or entity.color == obj.color:
                continue
            if entity.shape != obj.shape_signature:
                continue
            overlap = entity.cells & obj.cells
            if overlap and len(overlap) >= len(entity.cells) // 2:
                entity.color = obj.color
                entity.cells = obj.cells
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
        after she acted, reliably more often than it moves on its own.
        """
        if self.self_id is not None:
            current = self.entities.get(self.self_id)
            if current is not None and not current.gone:
                return
            self.self_id = None
        best: WorldEntity | None = None
        best_score = 0.0
        for entity in self.entities.values():
            if entity.gone or entity.steps_after_action < _SELF_MIN_CONTINGENT:
                continue
            rate = entity.action_moves / entity.steps_after_action
            free_rate = entity.n_moved_free / max(1, entity.n_seen)
            score = rate - free_rate * 0.5
            if entity.action_moves >= _SELF_MIN_CONTINGENT and score > best_score:
                best, best_score = entity, score
        if best is not None:
            best.is_self = True
            self.self_id = best.id
            events.append(WorldEvent(
                "self_found",
                f"I am the color-{best.color} thing",
                salience=0.8,
                entity_id=best.id,
            ))

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
        delta = (round(entity.velocity[0]), round(entity.velocity[1]))
        self.action_effects.setdefault(action, Counter())[delta] += 1
        if delta == (0, 0):
            self.action_noops[action] += 1

    def _contact_outcomes(
        self,
        pending: str | None,
        prior_events: list[WorldEvent],
        info: dict[str, Any] | None = None,
    ) -> list[WorldEvent]:
        """Attribute affordance outcomes to entities she touched.

        The attribution rule is uniform: whatever she was in contact
        with when an outcome befell her caused it. An entity that
        disappeared on contact is collectible; one sitting on the
        cell her action failed to enter is a blocker; one touching
        her on the tick her own state changed is a transformer; one
        she touched on the tick the world ended is the hazard — or,
        if it ended in victory, the goal.
        """
        events: list[WorldEvent] = []
        me = self._effector()
        if me is None:
            return events

        # Terminal outcomes are attributed even when she is gone —
        # dying is exactly when her last contacts matter most.
        if info:
            term = (
                OUT_HAZARD
                if info.get("reset") or info.get("death")
                else OUT_WIN if info.get("win") else None
            )
            if term is not None:
                for contact in self._contacts(me):
                    self.affordances.setdefault(
                        contact.color, Counter()
                    )[term] += 1
                    events.append(WorldEvent(
                        term,
                        f"the color-{contact.color} thing killed me"
                        if term == OUT_HAZARD
                        else f"the color-{contact.color} thing is the goal",
                        salience=0.9,
                        entity_id=me.id,
                        other_id=contact.id,
                    ))
        if me.gone:
            return events
        my_cells = set(me.cells)
        adjacent = {
            (r + dr, c + dc)
            for r, c in my_cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        } - my_cells

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

        for entity in self.entities.values():
            if entity.id == me.id or entity.color == me.color:
                continue
            if (
                entity.gone
                and entity.gone_tick == self.tick
                and entity.cells & (my_cells | adjacent)
            ):
                self.affordances.setdefault(entity.color, Counter())
                aff = self.affordances[entity.color]
                aff[OUT_COLLECT] += 1
                events.append(WorldEvent(
                    "collected",
                    f"I touched the color-{entity.color} thing and it vanished",
                    salience=0.7,
                    entity_id=me.id,
                    other_id=entity.id,
                ))
            elif (
                not entity.gone
                and target_cells
                and entity.cells & target_cells
            ):
                self.affordances.setdefault(entity.color, Counter())
                aff = self.affordances[entity.color]
                aff[OUT_BLOCK] += 1
                entity.blocked_colors.add(me.color)
                self.known_blocked.update(entity.cells)
                events.append(WorldEvent(
                    "blocked",
                    f"the color-{entity.color} thing blocked me",
                    salience=0.5,
                    entity_id=me.id,
                    other_id=entity.id,
                ))

        # Self-state change under contact: whatever touched her on
        # the tick her color changed is what changed it. Every
        # contact is credited — including things consumed this tick,
        # which were in contact when it happened — and co-occurrence
        # over episodes sharpens which kind actually does it.
        if any(
            e.kind == "recolored" and e.entity_id == me.id
            for e in prior_events
        ):
            for contact in self._contacts(me, include_vanished=True):
                self.affordances.setdefault(
                    contact.color, Counter()
                )[OUT_RECOLOR] += 1
                contact.recolor_result = me.color
                events.append(WorldEvent(
                    "transformed",
                    f"the color-{contact.color} thing changed me",
                    salience=0.7,
                    entity_id=me.id,
                    other_id=contact.id,
                ))
        return events

    def _update_map(self, scene: Scene) -> None:
        """Maintain the occupancy map: free floor vs. solid things."""
        occupied = {
            cell
            for e in self.entities.values()
            if not e.gone
            for cell in e.cells
        }
        for r, c, _color in scene.grid.iter_cells():
            cell = (r, c)
            if cell in occupied:
                continue
            self.known_free.add(cell)
            self.known_blocked.discard(cell)
        for entity in self.entities.values():
            if entity.gone or entity.is_self:
                continue
            aff = self.affordance_of(entity.color)
            # Passability is learned, not assumed: a thing is
            # impassable only once touching it has actually blocked
            # or killed her. A motionless thing might be a wall —
            # or a gem sitting still. Static ≠ solid.
            if aff[OUT_BLOCK] > 0 or aff[OUT_HAZARD] > 0:
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
        for entity in self.entities.values():
            if entity.id == me.id or entity.gone:
                continue
            if entity.cells & me.cells:
                entity.open_colors.add(me.color)

    def passable(self, entity: WorldEntity, self_color: int) -> bool:
        """Would contact with this entity let her enter its cells?

        Hazards are never entered. Entities with a known positive
        affordance (collect / transform / goal) are entered by
        definition — the affordance IS what contact does. For the
        rest, passability is a learned (kind, self-state) relation:
        a pair never observed is optimistically open — the only way
        to learn that a gate opens for a color is to try it.
        """
        aff = self.affordance_of(entity.color)
        # Observed evidence about THIS entity dominates kind priors.
        if self_color in entity.open_colors:
            return True
        if self_color in entity.blocked_colors:
            return False
        if aff[OUT_HAZARD] > 0:
            return False
        if (
            aff[OUT_COLLECT] > 0
            or aff[OUT_RECOLOR] > 0
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
        adjacent = {
            (r + dr, c + dc)
            for r, c in me.cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        }
        contact = set(me.cells) | adjacent
        return [
            entity
            for entity in self.entities.values()
            if entity.id != me.id
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
        return SimState(
            cells=me.cells,
            color=me.color,
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
        consumed, transformers change her color. ``None`` means the
        action's effect is unknown or the move fails in this state —
        either way there is nothing to plan on.
        """
        effect = self.effect_of(action)
        if effect is None or effect == (0, 0):
            return None
        new_cells = frozenset(
            (r + effect[0], c + effect[1]) for r, c in state.cells
        )
        gone = set(state.gone)
        new_color = state.color
        for entity in self.entities.values():
            if entity.id in state.gone or entity.is_self:
                continue
            if not (entity.cells & new_cells):
                continue
            if not self.passable(entity, state.color):
                return None
            aff = self.affordance_of(entity.color)
            if aff[OUT_COLLECT] > 0:
                gone.add(entity.id)
            if aff[OUT_RECOLOR] > 0:
                new_color = (
                    entity.recolor_result
                    if entity.recolor_result is not None
                    else entity.color
                )
        return SimState(cells=new_cells, color=new_color,
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
        self.known_free.clear()
        self.known_blocked.clear()
        self.visits.clear()
        if not keep_learning:
            self.action_effects.clear()
            self.action_tries.clear()
            self.action_noops.clear()
            self.affordances.clear()

    def note_action(self, action: str) -> None:
        """Record the action just taken; the next observation is its effect."""
        self._pending_action = action

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
