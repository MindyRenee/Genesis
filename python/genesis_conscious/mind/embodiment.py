"""Mind embodiment — the observe/act interface to an outer world.

The dual of the conversation loop. ``respond()`` ingests a sentence
and emits a sentence; ``observe()`` ingests a perceived scene and
``act()`` emits an action. The same organism sits inside both loops:
perception, the world model, volition, episodic memory, and felt
consequences. Idle faculties (sleep, proposals, code learning,
autonomous web learning) are not part of this loop — they fire from
the heartbeat and CLI, which an environment harness never drives.

What runs here is the embodied half of cognition:

- ``observe`` — a frame becomes a Scene (spatial substrate), the
  scene updates the world model (entity persistence, agency,
  action effects, affordances), and salient changes enter episodic
  memory with neurochemical consequence — collecting something feels
  rewarding, a reset feels like aversive surprise, the unexpected
  is attention-grabbing.
- ``act`` — volition consults the world model: pursue a known-value
  target through the learned map, or explore whatever action's
  outcome is least predicted. The returned descriptor is abstract
  (``{"action": id, "cell": (r, c) | None}``); the harness maps it
  onto the environment's protocol.
- ``reset_world`` — an episode boundary. Positions are forgotten;
  learned physics and affordances persist across episodes.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_DOPAMINE,
    CHEM_ENDORPHIN,
    CHEM_NAMES,
    CHEM_NOREPINEPHRINE,
    CHEM_SEROTONIN,
    EMOTIONAL_TAG_SIZE,
    MODULE_SENSORY,
)

from ..basal_ganglia.world_policy import WorldPolicy
from ..parietal_lobe.spatial_imagery import SpatialImagery
from ..parietal_lobe.world_model import (
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_WIN,
    WorldEvent,
    WorldModel,
)
from ..temporal_lobe.cognitive_map import CognitiveMap

if TYPE_CHECKING:
    from ..spatial.grid import Grid
    from ..spatial.scene import Scene

logger = logging.getLogger(__name__)

# Salience floor for storing a world event in episodic memory —
# appearances and routine motion are too frequent to record.
_STORE_SALIENCE = 0.5

# Consecutive unrecognized scene-level transformations before the
# change grounds as an epistemic gap — a single unexplained frame is
# ordinary complex change, a streak is a pattern she can't explain.
_UNEXPLAINED_CHANGE_STREAK = 3

# Felt consequences per event kind: (chem, amount) pairs. The
# pattern mirrors the puzzle affective loop — real outcomes produce
# real impulses, bounded so a bad step is information, never
# punishment (Schultz 1998 reward prediction error).
_EVENT_IMPULSES: dict[str, tuple[tuple[int, float], ...]] = {
    "collected": ((CHEM_DOPAMINE, 0.05), (CHEM_SEROTONIN, 0.02)),
    "win": (
        (CHEM_DOPAMINE, 0.08),
        (CHEM_SEROTONIN, 0.04),
        (CHEM_ENDORPHIN, 0.04),
    ),
    "reset": ((CHEM_NOREPINEPHRINE, 0.04), (CHEM_ACETYLCHOLINE, 0.03)),
    "unexpected": ((CHEM_NOREPINEPHRINE, 0.02), (CHEM_ACETYLCHOLINE, 0.02)),
    "transformed": ((CHEM_ACETYLCHOLINE, 0.02),),
    "self_found": ((CHEM_ACETYLCHOLINE, 0.04),),
    "blocked": ((CHEM_ACETYLCHOLINE, 0.01),),
}


class EmbodimentMixin:
    """Mixin for :class:`Mind` — the environment-facing loop."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any:
            """Tell type checkers that ``Mind`` supplies mixin attributes."""
            ...

    def _init_embodiment(self, seed: int | None = None) -> None:
        """Create the world model and the policy that acts on it."""
        self.world_model = WorldModel()
        self._world_policy = WorldPolicy(self.world_model, seed=seed)
        # Hippocampal-entorhinal allocentric map — the world-centered
        # layer that persists where the egocentric model resets.
        self.cognitive_map = CognitiveMap()
        # (action, predicted self-displacement) awaiting its outcome —
        # the intention half of the agency comparator.
        self._pending_world_action: tuple[str, Any] | None = None
        # Parietal imagery workspace — transform skills applied to
        # internal copies of the scene; recognition explains global
        # change that entity tracking cannot express.
        self._imagery = SpatialImagery(network=self.cognition.network)
        self._prev_world_grid: Grid | None = None
        # Consecutive observations whose global change no known
        # transform explains — a persistent streak grounds as an
        # epistemic-gap concept for curiosity to find.
        self._unexplained_change_streak = 0

    def observe(
        self,
        frame: Any = None,
        info: dict[str, Any] | None = None,
        background: int | None = None,
    ) -> list[WorldEvent]:
        """Take in one observation of the world; return what changed.

        ``frame`` may be a ``Scene`` (already perceived), a 2-D
        integer array (game pixels — quantized grids), a nested int
        list, or None to use the live camera frame. ``info`` carries
        environment-reported outcome flags (reset, win) when the
        environment supplies them. ``background`` is the empty-cell
        value when the caller knows it — a perceptual convention,
        like exposure — otherwise the frame's modal color is used.
        """
        scene = self._frame_to_scene(frame, background=background)
        if scene is None:
            return []
        events = self.world_model.observe(scene, info=info)
        events.extend(self._recognize_scene_change(scene))
        self._map_world(scene)
        self._resolve_world_action()
        for event in events:
            self._feel_world_event(event)
            if event.salience >= _STORE_SALIENCE:
                self._store_world_event(event)
                self._emit_live_thought("world", event.text, announce=False)
        return events

    def act(self, available_actions: list[Any]) -> dict[str, Any] | None:
        """Choose the next action from the environment's whitelist.

        ``available_actions`` items are action ids — or descriptors
        ``{"action": id, "label": canonical_label}`` when the
        interface advertises conventional semantics (a labeled
        button). Labels are registered as priors on the world model;
        observed effects always override them.

        The basal ganglia's world policy decides over the parietal
        world model's state; this records the choice — and the
        predicted bodily outcome — so the next ``observe`` can both
        attribute consequences to it and compare prediction against
        reality (the agency comparator: matched predictions are the
        felt sense that the body is hers).
        """
        ids: list[str] = []
        for item in available_actions:
            if isinstance(item, dict):
                aid = str(item.get("action"))
                label = item.get("label")
                if label is not None:
                    self.world_model.learn_action_label(aid, str(label))
                ids.append(aid)
            else:
                ids.append(str(item))
        choice = self._world_policy.choose(ids)
        if choice is not None:
            action = str(choice["action"])
            expected = self.world_model.expected_effect(action)
            self._pending_world_action = (action, expected)
            self.world_model.note_action(action)
            try:
                self.self_model.record_intention(
                    action,
                    intended_outcome=(
                        f"move {expected}" if expected else f"act {action}"
                    ),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"agency intention record failed: {e}")
        return choice

    def reset_world(self, keep_learning: bool = True) -> None:
        """Episode boundary — forget the layout, keep the physics."""
        self.world_model.reset(keep_learning=keep_learning)
        # The allocentric map ends its fix but keeps its structure —
        # hippocampal spatial memory is long-term memory.
        self.cognitive_map.reset_episode()
        self._pending_world_action = None
        self._prev_world_grid = None
        self._unexplained_change_streak = 0

    def world_summary(self) -> str:
        """What she currently understands about the outer world."""
        base = self.world_model.summary()
        return f"{base} — {self.cognitive_map.summary()} — {self._imagery.summary()}"

    # ─── Internals ──────────────────────────────────────────────

    def _map_world(self, scene: Scene) -> None:
        """Feed the allocentric map: parietal state → hippocampal map.

        The egocentric model reports where she is and what's visible;
        the map binds it to world-centered structure — places,
        transitions, landmarks — that survives episode boundaries.
        The action label is whichever pending action produced this
        observation, so edges record real traversals.
        """
        me = self.world_model.self_entity
        position = me.centroid if me is not None and not me.gone else None
        action = (
            self._pending_world_action[0]
            if self._pending_world_action is not None
            else None
        )
        entities = [
            (e.color, e.cells)
            for e in self.world_model.entities.values()
            if not e.gone and not e.is_self
        ]
        # When the allocentric fix is missing, self-reported velocity
        # feeds dead reckoning so the position estimate still tracks
        # her motion (corrected on the next fix).
        delta = (
            me.velocity
            if position is None and me is not None and not me.gone
            else None
        )
        try:
            map_events = self.cognitive_map.observe(
                position,
                entities,
                action=action,
                blocked_cells=set(self.world_model.known_blocked),
                bounds=(scene.grid.height, scene.grid.width),
                delta=delta,
            )
        except Exception as e:  # noqa: BLE001 — mapping must never break the loop
            logger.debug(f"cognitive map update failed: {e}")
            return
        for event in map_events:
            self._emit_live_thought("world", event.text, announce=False)
            if event.salience >= _STORE_SALIENCE:
                self._store_world_event(
                    WorldEvent(event.kind, event.text, event.salience)
                )
        try:
            self._bias_gaze_toward_goal(scene)
        except Exception as e:  # noqa: BLE001 — attention bias is advisory
            logger.debug(f"goal bias failed: {e}")

    def _bias_gaze_toward_goal(self, scene: Scene) -> None:
        """Top-down attention: learned goal entities draw her gaze.

        The world model's affordance ledger records which colors are
        goals (win) or valuable (collect). When such an entity is on
        screen, its centroid biases the saliency map — the frontal
        contribution to LIP-style priority. Cleared when nothing with
        a learned value is visible.
        """
        best = None
        best_score = 0.0
        for e in self.world_model.entities.values():
            if e.gone or e.is_self or not e.cells:
                continue
            aff = self.world_model.affordances.get(e.color)
            if not aff:
                continue
            score = (
                aff.get(OUT_WIN, 0) * 2.0
                + aff.get(OUT_COLLECT, 0)
                - aff.get(OUT_HAZARD, 0) * 2.0
            )
            if score > best_score:
                best, best_score = e, score
        if best is None:
            self.vision.clear_attention_bias()
            return
        self.vision.bias_attention_toward(
            best.centroid[0], best.centroid[1],
            scene.grid.height, scene.grid.width,
        )

    def _recognize_scene_change(self, scene: Scene) -> list[WorldEvent]:
        """Scene-level change recognition — what did the world do?

        Entity tracking explains object-level change; this asks the
        dual question: did the *whole scene* undergo one known
        transform (everything fell, the field rotated, all colors
        remapped)? Recognized transforms become ``transformed``
        events — high-signal descriptions of global regularity.
        Unrecognized complex change emits nothing: ordinary motion is
        already covered by entity events, and unexplained global
        change is a knowledge gap, not noise to record per frame.
        """
        prev = self._prev_world_grid
        self._prev_world_grid = scene.grid
        if prev is None or prev == scene.grid:
            return []
        try:
            hits = self._imagery.recognize_change(prev, scene.grid)
        except Exception as e:  # noqa: BLE001 — recognition is advisory
            logger.debug(f"scene-change recognition failed: {e}")
            return []
        if not hits:
            self._unexplained_change_streak += 1
            if self._unexplained_change_streak >= _UNEXPLAINED_CHANGE_STREAK:
                self._ground_unexplained_change()
            return []
        self._unexplained_change_streak = 0
        return [
            WorldEvent(
                "transformed",
                f"the scene {', '.join(hits)}",
                salience=0.6,
            )
        ]

    def _ground_unexplained_change(self) -> None:
        """Persistent unexplained global change becomes a concept.

        A single unrecognized frame is noise; a *streak* means the
        world is doing something she cannot explain — an epistemic
        gap. Grounding it as a low-confidence concept lets ordinary
        gap detection (isolation/uncertainty in the curiosity engine)
        surface it as a question when nothing stronger is active —
        and each repeat re-adds it, boosting activation so a persistent
        mystery stays salient instead of decaying.
        """
        try:
            self.cognition.network.add_concept(
                "unexplained world change",
                confidence=0.3,
                origin="perception",
                properties={
                    "kind": "unexplained_phenomenon",
                    "episodes": self._unexplained_change_streak,
                },
            )
        except Exception as e:  # noqa: BLE001 — grounding is advisory
            logger.debug(f"unexplained-change grounding failed: {e}")

    def _frame_to_scene(
        self, frame: Any, background: int | None = None
    ) -> Scene | None:
        """Normalize an observation into a Scene via the substrate."""
        from ..spatial.grid import Grid
        from ..spatial.scene import Scene, perceive

        if isinstance(frame, Scene):
            return frame
        if frame is None:
            return self.vision.perceive_scene(network=self.cognition.network)
        # A 2-D integer frame (game pixels) is already a grid —
        # no retina quantization needed.
        try:
            import numpy as np

            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    return perceive(
                        Grid.from_lists(frame.tolist()),
                        background=background,
                    )
                if frame.ndim == 3:
                    return self.vision.perceive_scene(
                        frame, network=self.cognition.network
                    )
        except ImportError as e:
            logger.debug(f"numpy frame normalization unavailable: {e}")
        if isinstance(frame, (list, tuple)) and frame:
            return perceive(
                Grid.from_lists([list(map(int, row)) for row in frame]),
                background=background,
            )
        return None

    def _resolve_world_action(self) -> None:
        """Close the agency loop: did the world obey the intention?

        The comparator model of agency (Wolpert, 1997): predicted
        bodily consequence vs observed. A match strengthens her felt
        ownership of the body the world model found; a mismatch —
        "that wasn't me / the world surprised me" — weakens it.
        """
        pending = self._pending_world_action
        self._pending_world_action = None
        if pending is None:
            return
        action, expected = pending
        me = self.world_model.self_entity
        if me is None or expected is None:
            # No prediction was made — there is nothing to compare.
            # Recording unlearned actions as failures would corrupt
            # the comparator with noise.
            return
        observed = (round(me.velocity[0]), round(me.velocity[1]))
        try:
            self.self_model.record_evaluated_outcome(
                action,
                intended_outcome=f"move {expected}",
                actual_outcome=f"move {observed}",
                matched=observed == expected,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"agency outcome record failed: {e}")

    def _feel_world_event(self, event: WorldEvent) -> None:
        """Felt consequence: world events drive neurochemistry."""
        impulses = _EVENT_IMPULSES.get(event.kind)
        if not impulses:
            return
        try:
            for chem, amount in impulses:
                self.client.neuro_impulse(chem, amount)
        except Exception as e:  # noqa: BLE001 — feeling must never break the loop
            logger.debug(f"world-event impulse failed: {e}")

    def _store_world_event(self, event: WorldEvent) -> None:
        """Episodic memory: salient world changes become recallable."""
        try:
            chem = self.client.get_state()
            tag = [
                float(chem.chemicals.get(CHEM_NAMES[i], 0.0))
                for i in range(EMOTIONAL_TAG_SIZE)
            ]
            self.client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=0,  # Input — something she perceived
                source_module=MODULE_SENSORY,
                salience=event.salience,
                emotional_tag=tag,
                text=event.text,
            )
        except Exception as e:  # noqa: BLE001 — memory loss must not crash the loop
            logger.debug(f"failed to store world event: {e}")
