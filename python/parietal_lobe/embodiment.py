"""Mind embodiment — the observe/act interface to an outer world.

The dual of the conversation loop. ``respond()`` ingests a sentence
and emits a sentence; ``observe()`` ingests a perceived scene and
``act()`` emits an action. The same organism sits inside both loops:
perception, the world model, volition, episodic memory, and felt
consequences. Idle faculties (sleep, proposals, autonomous web
learning) are not part of this loop — they fire from
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

import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from basal_ganglia.world_policy import _SELECT_THRESHOLD, WorldPolicy
from parietal_lobe.spatial_imagery import SpatialImagery
from parietal_lobe.world_model import (
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_TRANSFORM,
    OUT_WIN,
    WorldEvent,
    WorldModel,
)
from temporal_lobe.concepts import RelationType

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_ENDORPHIN,
    CHEM_NAMES,
    CHEM_NOREPINEPHRINE,
    CHEM_SEROTONIN,
    EMOTIONAL_TAG_SIZE,
    MODULE_SENSORY,
)
from genesis_client.swallow import note_swallowed
from infrastructure.config import default_data_dir

if TYPE_CHECKING:
    from parietal_lobe.grid import Grid
    from parietal_lobe.scene import Scene

logger = logging.getLogger(__name__)

# Salience floor for storing a world event in episodic memory —
# appearances and routine motion are too frequent to record.
_STORE_SALIENCE = 0.5

# Inner-voice budget per engaged session — salience, not chatter.
_WORLD_VOICE_CAP = 12

# The wire palette's color space for game grids seen as images:
# cell values are the 16 colors of the ARC palette (the official
# arc_agi.rendering.COLOR_MAP, alpha dropped). Kept here rather than
# imported so the mind layer stays free of the adapter's package.
_GAME_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0xFF, 0xFF, 0xFF),
    (0xCC, 0xCC, 0xCC),
    (0x99, 0x99, 0x99),
    (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33),
    (0x00, 0x00, 0x00),
    (0xE5, 0x3A, 0xA3),
    (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31),
    (0x1E, 0x93, 0xFF),
    (0x88, 0xD8, 0xF1),
    (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B),
    (0x92, 0x12, 0x31),
    (0x4F, 0xCC, 0x30),
    (0xA3, 0x56, 0xD6),
)

# Consecutive unrecognized scene-level transformations before the
# change grounds as an epistemic gap — a single unexplained frame is
# ordinary complex change, a streak is a pattern she can't explain.
_UNEXPLAINED_CHANGE_STREAK = 3

# Felt consequences per event kind: (chem, amount) pairs — real
# outcomes produce real impulses, bounded so a bad step is
# information, never punishment (Schultz 1998 reward prediction
# error).
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
    "part": ((CHEM_ACETYLCHOLINE, 0.03),),
    "part_lost": ((CHEM_NOREPINEPHRINE, 0.02),),
    "reshaped": ((CHEM_ACETYLCHOLINE, 0.02),),
}

# Objective kinds the world actuator can actually serve — a "learn"
# objective routes to unanswered affordance questions, the rest to
# coverage. "inspect" is the code actuator's kind and never gets
# routed into a grid step.
_WORLD_OBJECTIVE_KINDS = frozenset({"learn", "explore", "observe", "measure"})
# An objective held this long without a served verdict is revised
# back to pending so a fresher one can be elected — persistence,
# not fixation.
_OBJECTIVE_STALE_S = 120.0

# Visual-cortex sampling cadence in a turn-based world. The grid is
# still perceived every turn; the expensive occipital hierarchy is an
# attention sample, not a second decision channel. First frames and
# level boundaries always get through.
_WORLD_VISION_PERIOD = 4

# New concepts taught per frame — the whole alphabet arrives at
# once but teaching is a drip, not a batch.
_GLYPH_TEACH_MAX = 8

# Deliberation — vicarious trial and error. When the arbitrated
# field's top-two margin falls below this, the first choice was a
# near-guess: cognition settles once more while the forward model
# weighs each candidate's simulated consequence onto the field
# (theta sweeps at the choice point, Johnson & Redish 2007), and
# the basal ganglia arbitrates again with that evidence in.
_DELIBERATE_MARGIN = 0.05
# A simulated step's pull on the second arbitration — per cell of
# progress toward a valued end, plus a floor premium for a move the
# forward model says does anything at all.
_DELIBERATE_GAIN = 0.15
_DELIBERATE_MOVE = 0.05

# Impasse dynamics — an unchanging world is evidence, not just
# boredom. Confusion (the epistemic emotion of a violated model,
# D'Mello & Graesser) fires at _DRY_CONFUSE dry frames; the "stuck"
# verdict waits for _world_patience(), which stretches while
# untried actions still promise information — serotonin's
# confidence that patience will be rewarded (Miyazaki 2018).
_DRY_CONFUSE = 8
_DRY_BASE = 20
_DRY_LEARN_BONUS = 15

# Episodic recall cadence — a pass every this many world ticks, plus
# immediately on any salient event (the moments a decision is up for
# grabs). Bounded retrieval keeps the body loop cheap.
_RECALL_PERIOD = 8
_RECALL_LIMIT = 6  # candidates scanned per pass
_RECALL_FETCH = 4  # how many get full text retrieval
_RECALL_DIST = 20  # SimHash association threshold — the daemon's own

# Her own event vocabulary is the recall key: the outcome words she
# stored a memory with are the words that decode its valence.
_RECALL_OUTCOME_VALENCE = {
    "killed me": -1.0,
    "is the goal": 1.0,
    "was touching me": -0.3,
    "blocked me": -0.25,
    "changed me": 0.2,
}
_RECALL_ENTITY_RE = re.compile(
    r"the color-(\d+) thing (killed me|is the goal|was touching me"
    r"|blocked me|changed me)"
)


class EmbodimentMixin:
    """Mixin for :class:`Mind` — the environment-facing loop."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        def __getattr__(self, name: str) -> Any:
            """Tell type checkers that ``Mind`` supplies mixin attributes."""
            ...

    def _init_embodiment(self, seed: int | None = None) -> None:
        """Create the world model and the policy that acts on it."""
        from thalamus.bridge import WorldBridge, WorldBroadcastSink

        self.world_model = WorldModel()
        self._world_policy = WorldPolicy(self.world_model, seed=seed)
        # Hippocampal-entorhinal allocentric map — the world-centered
        # layer that persists where the egocentric model resets. The
        # organ lives in the temporal lobe (its anatomical region);
        # the world channel is one consumer, feeding grid coordinates
        # into the shared map's own context.
        self.cognitive_map = self.brain.temporal.cognitive_map
        # (action, predicted self-displacement) awaiting its outcome —
        # the intention half of the agency comparator.
        self._pending_world_action: tuple[str, Any] | None = None
        # The executive objective the chosen bid was serving — it is
        # fed the comparator's verdict when the outcome arrives, so
        # objectives learn from consequences the way actions do.
        self._pending_objective: Any = None
        # The objective the executive is currently holding, and the
        # time it was last served a verdict — an objective held long
        # without an answer is revised so a fresher one is elected.
        self._held_objective: Any = None
        self._held_since: float = 0.0
        # The executive's resting stop threshold — borrowed while a
        # world holds her so the competition's operating point is
        # the gate the executive owns; restored on disengage.
        self._exec_prev_inhibition: float | None = None
        # Decision journal — each turn's winning channel, its
        # weight, and where she stood. Her own trace of what she
        # did and why, kept across the episode so a policy that
        # circles is visible as circling.
        self._world_decision_log: deque[dict[str, Any]] = deque(maxlen=400)
        # Parietal imagery workspace — transform skills applied to
        # internal copies of the scene; recognition explains global
        # change that entity tracking cannot express.
        self._imagery = SpatialImagery(network=self.brain.frontal.network)
        self._prev_world_grid: Grid | None = None
        # Consecutive observations whose global change no known
        # transform explains — a persistent streak grounds as an
        # epistemic-gap concept for curiosity to find.
        self._unexplained_change_streak = 0
        # Whole-brain channel for world state: observations and
        # decisions broadcast through the global workspace (push) and
        # published as tensors on the daemon's slab (pull). The sink
        # feeds conscious world content back to Mind-level consumers
        # (volition context, the inner stream).
        workspace = getattr(self.brain.frontal, "global_workspace", None)
        language = getattr(self.brain.frontal, "language", None)
        self._world_bridge = WorldBridge(
            workspace=workspace,
            client=self.client,
            brain_waves=lambda: getattr(language, "current_brain_waves", None),
        )
        if workspace is not None:
            workspace.register_module(WorldBroadcastSink(self._note_world_broadcast))
        # The world content currently in the workspace — read by
        # volition context so autonomous urges know when a world is
        # already holding her attention.
        self._world_focus: dict[str, Any] | None = None

        # Hypothesis layer binding: the shared TaskCompetence keeps a
        # per-schema ledger of mechanic/goal hypotheses, verified and
        # falsified by this loop's own transitions.
        self._world_task: Any = None
        self._prev_world_state: dict[str, Any] | None = None
        self._world_episode_changes: tuple[str, ...] = ()

        # Engaged world — a world currently holding her attention
        # (a game in play). Engagement is *state*: volition context
        # reads it through world_engagement() and the appetitive
        # urges starve while it holds — the way absorption keeps a
        # person from wandering off — rather than the urges being
        # switched off upstream. The loop itself is wholly hers:
        # perceive, think, act — no delegate plays for her.
        self._engaged_world: str | None = None
        self._world_context: str | None = None
        self._world_seen: set[str] = set()
        self._world_levels = 0
        self._world_dry = 0
        self._world_deaths = 0
        # The action ids the engaged world last offered — patience
        # counts untried physics against this set.
        self._world_ids: list[str] = []
        self._world_voiced = 0
        self._world_last_voiced: str | None = None
        self._last_world_step = 0.0
        # Decision telemetry — the basis behind the last world
        # action (which drive won, how decisive, the contenders).
        # Harnesses attach it to each step as recorded reasoning.
        self.last_world_decision: dict[str, Any] | None = None
        # The motor field the last act() was arbitrated over — kept
        # so the step can read how much her own field values the
        # peak it returned (``_field_opinion``).
        self._world_motor: Any = None
        # Episodic recall bookkeeping — when the last pass fired and
        # what it remembered, for telemetry.
        self._world_recall_tick = -1
        self._world_last_recall: dict[int, float] = {}

    def observe(
        self,
        frame: Any = None,
        info: dict[str, Any] | None = None,
        background: int | None = None,
        through: list[Any] | None = None,
    ) -> list[WorldEvent]:
        """Take in one observation of the world; return what changed.

        ``frame`` may be a ``Scene`` (already perceived), a 2-D
        integer array (game pixels — quantized grids), a nested int
        list, or None to use the live camera frame. ``info`` carries
        environment-reported outcome flags (reset, win) when the
        environment supplies them. ``background`` is the empty-cell
        value when the caller knows it — a perceptual convention,
        like exposure — otherwise the frame's modal color is used.

        ``through`` carries the animation strip when an environment
        renders a consequence across several frames: each one is
        tracked as motion (the world model's pursuit channel) but
        only ``frame`` opens a trial — saccadic masking, done in the
        visual pathway rather than at the camera.
        """
        for mid in through or ():
            mid_scene = self._frame_to_scene(mid, background=background)
            if mid_scene is not None:
                self.world_model.pursue(mid_scene)
        scene = self._frame_to_scene(frame, background=background)
        if scene is None:
            return []
        # The action whose outcome this frame carries — captured
        # before _resolve_world_action clears it.
        pending = self._pending_world_action
        events = self.world_model.observe(scene, info=info)
        # A reported win/death or an inferred wholesale turnover reset
        # the model inside observe(); every policy commitment made
        # against that vanished episode is stale with it. Cached plans,
        # retrial entities, perseveration, and re-embodiment counts all
        # name the old world, not the one now in view.
        if any(event.kind in {"reset", "win"} for event in events):
            self._world_policy.reset()
        events.extend(self._recognize_scene_change(scene))
        self._map_world(scene)
        self._resolve_world_action()
        self._learn_world_hypotheses(pending, events)
        for event in events:
            self._feel_world_event(event)
            if event.salience >= _STORE_SALIENCE:
                self._store_world_event(event)
                self._emit_live_thought("world", event.text)
        self._publish_world_state(scene, events)
        return events

    def act(
        self,
        available_actions: list[Any],
        recommended: dict[str, float] | None = None,
    ) -> dict[str, Any] | None:
        """Choose the next action from the environment's whitelist.

        ``available_actions`` items are action ids — or descriptors
        ``{"action": id, "label": canonical_label}`` when the
        interface advertises conventional semantics (a labeled
        button). Labels are registered as priors on the world model;
        observed effects always override them.

        ``recommended`` is a specialist's disposition over the space
        (e.g. the game agent's navigational vote). It is written onto
        the action field — a voice in the arbitration, not a command:
        the basal ganglia still weighs it against the world model's
        own plan, hypothesis terms, and its calibration instinct.

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
        # Lay the decision out as distributed motor fields over the
        # grid and the offered actions — the fields bias the choice
        # below and are published for the daemon to read back.
        motor = self._compose_motor_fields(ids, recommended)
        self._world_motor = motor
        # The executive holds an objective into the competition:
        # select one when none is held — or when the held one has
        # gone stale unaddressed — so persistence is prefrontal,
        # not momentum. Its bid is a channel, never a command.
        executive = getattr(self.brain.frontal, "executive", None)
        if executive is not None:
            try:
                active = executive.active_intention
                if active is not self._held_objective:
                    self._held_objective = active
                    self._held_since = time.time()
                if active is None or time.time() - self._held_since > _OBJECTIVE_STALE_S:
                    if active is not None:
                        executive.revise_intention(active, reason="held too long unanswered")
                    executive.select_intention(
                        list(executive.actionable_intentions),
                        allowed_kinds=_WORLD_OBJECTIVE_KINDS,
                    )
                    self._held_objective = executive.active_intention
                    self._held_since = time.time()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"executive objective select failed: {e}")
        choice = self._world_policy.choose(ids, field=motor, executive=executive)
        self._pending_objective = None
        if choice is not None:
            action = str(choice["action"])
            expected = self.world_model.expected_effect(action)
            self._pending_world_action = (action, expected)
            self._pending_objective = (
                executive.active_intention
                if executive is not None and choice.get("source") == "intention"
                else None
            )
            # The cell rides with the action — a distal act is
            # attributable only if the model knows where it landed.
            self.world_model.note_action(action, target=choice.get("cell"))
            try:
                self.self_model.record_intention(
                    action,
                    intended_outcome=(f"move {expected}" if expected else f"act {action}"),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"agency intention record failed: {e}")
            self._broadcast_world_action(choice, motor)
        return choice

    def note_world_action(self, action: Any, cell: Any = None) -> None:
        """Record an externally-chosen action as her pending intention.

        For loops where a specialist (a game agent) picks the move
        but the embodied stack still owns the world model: registers
        the choice — and the model's predicted bodily effect — so the
        next ``observe`` attributes the consequences to it and the
        agency comparator compares prediction against reality. The
        action was made; the comparator's job is to feel whether the
        world obeyed.
        """
        action = str(action)
        expected = self.world_model.expected_effect(action)
        self._pending_world_action = (action, expected)
        self.world_model.note_action(action, target=cell)
        try:
            self.self_model.record_intention(
                action,
                intended_outcome=(f"move {expected}" if expected else f"act {action}"),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"agency intention record failed: {e}")

    def note_world_end(self, outcome: str | None = None) -> list[WorldEvent]:
        """The episode ended with no frame to carry the news.

        ``outcome`` is ``"win"``/``"hazard"`` when the harness knows
        the cause — the last things she was touching get the credit.
        ``None`` means the world ended unexplained: the same
        contiguity suspicion an unannounced reset applies. Either
        way the world is then forgotten (physics kept).
        """
        model = self.world_model
        if outcome in (OUT_WIN, OUT_HAZARD):
            events = model.attribute_terminal(outcome)
        else:
            events = model.suspect_terminal()
        events.append(WorldEvent("reset", "the world reset", salience=0.9))
        for event in events:
            self._feel_world_event(event)
            if event.salience >= _STORE_SALIENCE:
                self._store_world_event(event)
                self._emit_live_thought("world", event.text)
        self.reset_world(keep_learning=True)
        return events

    def reset_world(self, keep_learning: bool = True) -> None:
        """Episode boundary — forget the layout, keep the physics."""
        self.world_model.reset(keep_learning=keep_learning)
        self._world_policy.reset()
        # The allocentric map ends its fix but keeps its structure —
        # hippocampal spatial memory is long-term memory.
        self.cognitive_map.reset_episode()
        self._pending_world_action = None
        self._pending_objective = None
        self._held_objective = None
        self._held_since = 0.0
        self._prev_world_grid = None
        self._prev_world_state = None
        self._world_episode_changes = ()
        self._unexplained_change_streak = 0
        # The tick counter restarts with the episode — let the new
        # situation recall freely from its first frame.
        self._world_recall_tick = -1

    def world_summary(self) -> str:
        """What she currently understands about the outer world."""
        base = self.world_model.summary()
        return f"{base} — {self.cognitive_map.summary()} — {self._imagery.summary()}"

    # ─── Engaged world ──────────────────────────────────────────

    def engage_world(self, name: str = "the world") -> None:
        """Pin engagement: a world is now holding her attention.

        Focus as state, not amputation — while engagement holds,
        ``world_engagement`` keeps the appetitive urges starved in
        ``_volition_context`` (she is absorbed, the way a person
        mid-game doesn't wander off to audit code), and the bodily
        and defensive drives still get heard.
        """
        self._engaged_world = name
        # While the world is the task, the executive's stop gate
        # sits at the action competition's operating point — the
        # basal ganglia run the bidding, the executive owns what
        # "too weak to act" means here. Restored on disengage.
        executive = getattr(self.brain.frontal, "executive", None)
        if executive is not None:
            try:
                self._exec_prev_inhibition = executive.inhibition_threshold
                executive.set_inhibition_threshold(_SELECT_THRESHOLD)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"executive inhibition borrow failed: {e}")

    def disengage_world(self) -> None:
        """Release engagement — the world stops holding her.

        The specialist is dropped with the session: what it learned
        consolidated into ``TaskCompetence`` skills, which persist;
        the screen's particular layout does not — walking away and
        coming back, she remembers *how to play*, not that frame.
        """
        self._engaged_world = None
        self._world_seen.clear()
        self._world_levels = 0
        self._world_dry = 0
        self._world_deaths = 0
        self._world_voiced = 0
        self._world_last_voiced = None
        self._world_motor = None
        self._world_last_recall = {}
        self._world_recall_tick = -1
        self._last_world_step = 0.0
        self._pending_objective = None
        # The stop threshold returns to the executive's resting
        # value — the world's competition is over.
        if self._exec_prev_inhibition is not None:
            executive = getattr(self.brain.frontal, "executive", None)
            if executive is not None:
                try:
                    executive.set_inhibition_threshold(self._exec_prev_inhibition)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"executive inhibition restore failed: {e}")
            self._exec_prev_inhibition = None

    def world_engagement(self) -> float:
        """How held her attention is, [0, 1] — the focus signal.

        Two sources: an ignited world broadcast still ringing (a
        percept just fired — decays with ~10 s half-life) and a
        pinned engaged world (a live game session). Volition reads
        this so urges to wander off weigh engagement that already
        exists rather than treating her as idle.
        """
        engagement = 0.0
        focus = getattr(self, "_world_focus", None)
        if focus is not None:
            age_s = max(0.0, time.time() - focus["timestamp"] / 1000.0)
            engagement = focus["activation"] * pow(0.5, age_s / 10.0)
        if getattr(self, "_engaged_world", None) is not None:
            engagement = max(engagement, 0.9)
        return min(1.0, engagement)

    def world_episode_end(self, outcome: str, game: str = "the game") -> None:
        """An episode in the engaged world ended (``"win"``/``"over"``).

        Felt, broadcast, learned from — then the layout is forgotten
        and the physics kept. The specialist's segment closes with
        the same verdict so its outcome ledger stays honest.
        """
        win = outcome == "win"
        self._world_bridge.broadcast_outcome(
            outcome,
            f"episode ended {outcome}",
            0.9 if win else 0.7,
            metadata={"game": game, "deaths": self._world_deaths},
        )
        if win:
            self._impulse_world(CHEM_DOPAMINE, 0.02)
            self._world_say({"kind": "won", "game": game})
        else:
            self._world_deaths += 1
            self._impulse_world(CHEM_DOPAMINE, -0.008)
            self._impulse_world(CHEM_NOREPINEPHRINE, 0.008)
            self._impulse_world(CHEM_CORTISOL, 0.001 * self._world_deaths)
            self._world_say({"kind": "death", "game": game})
        try:
            self.world.it_acted(
                f"played {game}: episode ended {outcome}",
                detail=f"world episode {outcome}",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world act record failed: {e}")
        self.note_world_end("win" if win else None)
        # Checkpoint the learned world at every episode boundary —
        # a death or a win is exactly when new evidence lands.
        self._save_world_learning()

    @staticmethod
    def _world_learning_file(game: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", game)
        return default_data_dir() / "world_learning" / f"{safe}.json"

    @staticmethod
    def _competence_learning_file() -> Path:
        # Domain-general, not per-game: the TaskCompetence ledger is
        # deliberately shared across worlds — its schemas, lifted
        # operators, and skills are the part of game learning that is
        # meant to carry. It gets its own file so cross-game evidence
        # isn't stranded inside one world's physics snapshot.
        return default_data_dir() / "world_learning" / "_competence.json"

    def _save_world_learning(self) -> None:
        """Persist this world's learned physics and verdicts.

        Practice should compound: what one session's lives taught
        the model is the next session's ground, not a re-derivation.
        Advisory — a failed write must never interrupt play.
        """
        if not self._world_context:
            return
        if getattr(self, "_fresh_world_run", False):
            # A cold-start measurement is read-only both ways: her
            # banked experience stays untouched, not appended with
            # what the probe learned.
            return
        try:
            path = self._world_learning_file(self._world_context)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.world_model.export_learning()))
        except Exception:
            logger.debug("world learning save failed", exc_info=True)
        # Checkpoint the domain-general competence ledger at the same
        # boundary. The main state save only runs on shutdown or
        # sleep — a crash mid-run would otherwise take every operator
        # and skill the episode just earned with it.
        try:
            competence = self.brain.frontal.task_competence
        except AttributeError:
            return
        competence.save_checkpoint(self._competence_learning_file())

    def _load_world_learning(self) -> None:
        """Restore what past sessions learned about this world.

        Also answers the evidence question — how much experience did
        this run carry in? ``_world_learning_loaded`` reports it:
        ``"none"`` no file existed, ``"loaded"`` past lessons were
        restored, ``"suppressed"`` a fresh run refused them, and
        ``"unknown"`` a file was there but could not be read — the
        one state that must not masquerade as cold start.
        """
        if not self._world_context:
            self._world_learning_loaded = "none"
            self._world_competence_loaded = "none"
            return
        if getattr(self, "_fresh_world_run", False):
            # Cold-start evidence: the banked file is neither read
            # now nor overwritten later — the run is isolated both
            # directions.
            self._world_learning_loaded = "suppressed"
            self._world_competence_loaded = "suppressed"
            return
        loaded = "none"
        try:
            path = self._world_learning_file(self._world_context)
            if path.exists():
                self.world_model.import_learning(json.loads(path.read_text()))
                loaded = "loaded"
        except Exception:
            logger.debug("world learning load failed", exc_info=True)
            loaded = "unknown"
        self._world_learning_loaded = loaded
        self._restore_competence_checkpoint()

    def _restore_competence_checkpoint(self) -> None:
        """One-time recovery of the domain-general competence ledger.

        The shutdown save already restores ``task_competence`` — this
        file only matters when the process died mid-run, leaving the
        episode-end checkpoints newer than the last clean save.
        ``transition_count`` is cumulative evidence, so the checkpoint
        wins only when it strictly carries more; restoring an older
        file would erase learning the live mind already has. Runs once
        per session so a mid-session game switch can never clobber
        live schemas with a stale file. Status mirrors the
        ``_world_learning_loaded`` vocabulary for honest disclosure.
        """
        if getattr(self, "_world_competence_checked", False):
            return
        self._world_competence_checked = True
        self._world_competence_loaded = "none"
        try:
            competence = self.brain.frontal.task_competence
        except AttributeError:
            return
        self._world_competence_loaded = competence.restore_checkpoint(
            self._competence_learning_file()
        )

    def save_world_learning(self) -> None:
        """Public checkpoint for run ends that never switch games."""
        self._save_world_learning()

    def _begin_world(self, game: str, level: int) -> None:
        if self._world_context == game:
            return
        # The outgoing world's learning is banked before the reset
        # takes its episodic half; the incoming world's past lessons
        # are restored after it — order matters both ways.
        self._save_world_learning()
        self.reset_world(keep_learning=False)
        self._world_context = game
        self._load_world_learning()
        self._world_task = None
        self._world_levels = level
        self._world_dry = 0
        self._world_deaths = 0

    def world_step(
        self,
        frame: Any,
        available_actions: list[Any] | None = None,
        info: dict[str, Any] | None = None,
        through: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        """One full turn in the engaged world — perceive, think, act.

        The embodied analog of ``think()``: the frame becomes a
        percept through the whole stack (world model, cognitive map,
        imagery, episodic memory, the workspace broadcast), her
        cognition ticks on it (workspace decay/competition, wave
        refresh, executive candidates), and the basal ganglia emits
        the action over the composed motor field. The spatial
        specialist votes as a cortical skill — the choice is hers.

        Returns ``{"action": id, "x": .., "y": ..}`` — id 0 means
        the world should reset; ``x``/``y`` target coordinate
        actions. ``None`` when the world offered nothing to do.
        """
        info = info or {}
        game = str(info.get("game", "the game"))
        level = int(info.get("level", 0) or 0)
        self._begin_world(game, level)
        self.mark_user_activity()
        self._world_last_voiced = None
        self._world_onboard(game, level, info)

        # Levels advancing is the goal signal — the episode's own
        # reward marked before the new observation so the winning
        # segment closes intact. The ``win`` flag rides the frame
        # that follows the winning move: the model credits whatever
        # she was reaching when the world ended, then lets the old
        # place go while keeping what it taught her.
        advanced = level > self._world_levels
        if advanced:
            self._impulse_world(CHEM_DOPAMINE, 0.01)
            self._world_say({"kind": "scored", "game": game})
            self._world_dry = 0
        self._world_levels = level

        events = self._world_perceive(frame, game, level, info, advanced, through=through)

        # Think — the cognition slice a turn-based world would
        # otherwise never give her. And remember: the situation cues
        # whatever past experience it resembles before she commits.
        self._world_cognition_tick()
        self._recall_world(game, events)

        # Act — nothing outside her arbitrates: the motor field is
        # composed from her own world model and task competence, and
        # the basal ganglia's drives — goal plan, affordance probe,
        # frontier exploration, calibration — decide. She is her own
        # leader; there is no delegate to fall back on.
        avail = list(available_actions or [])
        if not avail:
            return None
        try:
            choice = self.act(avail)
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.world_step.act", e)
            choice = None
        if choice is None:
            return None
        # Deliberate — confidence gates effort, not the reverse. A
        # flat field margin means the first arbitration was a
        # near-guess: settle once more, let the forward model weigh
        # each candidate's imagined consequence onto the field, and
        # arbitrate again with that evidence present.
        if self._world_margin() < _DELIBERATE_MARGIN:
            disposition = self._deliberate_world(avail)
            if disposition:
                self._impulse_world(CHEM_ACETYLCHOLINE, 0.006)
                self._world_cognition_tick()
                try:
                    choice = self.act(avail, recommended=disposition) or choice
                except Exception as e:  # noqa: BLE001
                    note_swallowed("mind.world_step.deliberate", e)
        return self._world_commit(choice, avail, game)

    # ─── World-turn internals ───────────────────────────────────

    def _world_onboard(self, game: str, level: int, info: dict[str, Any]) -> None:
        """First contact with a game — engagement signature plus the
        world's spoken instruction broadcast as context, so her
        cognition meets the stream already knowing the genre."""
        from thalamus.brain_waves import BrainWave

        if game not in self._world_seen:
            self._world_seen.add(game)
            self._world_levels = level
            self._impulse_world(CHEM_ACETYLCHOLINE, 0.004)
            self._world_wave(BrainWave.BETA2, 0.15)
            self._world_say({"kind": "start", "game": game, "goal": info.get("instructions", "")})
            if info.get("instructions"):
                self._world_bridge.broadcast_observation(
                    f"{game} instructions: {info['instructions']}",
                    topics=["game", "instructions", game],
                    salience=0.9,
                )
                self._ground_world_words(str(info["instructions"]))

    def _world_perceive(
        self,
        frame: Any,
        game: str,
        level: int,
        info: dict[str, Any],
        advanced: bool,
        through: list[Any] | None = None,
    ) -> Any:
        """The frame enters through her embodied stack alone — the
        world model, the cognitive map, imagery, the workspace — and
        the change measure couples to dopamine (hers to compute, not
        a delegate's report). Returns the frame's events."""
        prev_grid = self._prev_world_grid
        events = self.observe(
            frame,
            info={**info, "game": game, "level": level, **({"win": True} if advanced else {})},
            through=through,
        )
        change = self._frame_change(prev_grid)
        vision_novel = (
            self._see_game_frame(frame)
            if (
                self.world_model.tick <= 1
                or advanced
                or self.world_model.tick % _WORLD_VISION_PERIOD == 0
            )
            else None
        )
        # The quantity channel rides the observation: how many
        # separate things the board holds is perceptual content her
        # cognition can reason about — subitized counts are exact,
        # larger ones arrive as the estimates they are.
        n_others = self.world_model.count_others()
        count_text = (
            str(n_others) if self.world_model.number.subitizes(n_others) else f"about {n_others}"
        )
        self._world_bridge.broadcast_observation(
            f"game {game}: level {level}/{info.get('win_levels', '?')}, "
            f"frame changed {change:.0%}, {count_text} "
            f"thing{'s' if n_others != 1 else ''} on the board",
            topics=["game", "spatial", game],
            salience=min(1.0, change * 3.0 + (0.5 if level > 0 else 0.0)),
            metadata={
                "game": game,
                "level": level,
                **({"vision_novel": vision_novel} if vision_novel is not None else {}),
            },
        )

        # The step's visible feedback couples to dopamine the same
        # way internal play couples its TD error — a changed world
        # is information.
        from thalamus.brain_waves import BrainWave

        impulse = max(-0.006, min(0.006, change * 0.04))
        if abs(impulse) >= 0.001:
            self._impulse_world(CHEM_DOPAMINE, impulse)
            self._world_wave(
                BrainWave.GAMMA if impulse > 0 else BrainWave.BETA2,
                min(0.25, abs(impulse) * 40),
            )
        if change < 0.005:
            self._world_dry += 1
            # Confusion before boredom: sustained impasse is
            # unexpected uncertainty — noradrenaline plus
            # acetylcholine (Yu & Dayan 2005), and the held
            # objective comes up for revision while the model is
            # still falsified.
            if self._world_dry == _DRY_CONFUSE:
                self._impulse_world(CHEM_NOREPINEPHRINE, 0.012)
                self._impulse_world(CHEM_ACETYLCHOLINE, 0.008)
                self._world_confuse(game)
            if self._world_dry == self._world_patience():
                self._world_say({"kind": "stuck", "game": game})
        else:
            self._world_dry = 0
        return events

    def _world_commit(self, choice: dict[str, Any], avail: list[Any], game: str) -> dict[str, Any]:
        """Ship her chosen action to the world — broadcast, record
        the decision basis, build the step descriptor."""
        action = int(choice["action"])
        if action == 0:
            self._world_bridge.broadcast_action(0, metadata={"game": game})
            self._record_world_decision("RESET", choice, avail)
            return {"action": 0}
        desc: dict[str, Any] = {"action": action}
        cell = choice.get("cell")
        if cell is not None:
            # A distal act carries its target — whichever actions
            # take coordinates is interface semantics the model
            # learned from labels, not an id this code knows.
            desc.update({"x": int(cell[1]), "y": int(cell[0])})
            self._world_bridge.broadcast_action(
                action,
                cell={"x": int(cell[1]), "y": int(cell[0])},
                metadata={"game": game},
            )
        self._record_world_decision(f"ACTION{action}", choice, avail)
        return desc

    def _world_cognition_tick(self) -> None:
        """The between-frames slice of cognition on world content.

        ``think()`` turns on an utterance; this turns on a frame —
        the wave state the broadcasts quote refreshes, and the
        workspace advances its dynamics (activation decay + the
        recurrent competition), so conscious game content settles or
        ignites by the same machinery as anything she attends. The
        hypothesis ledger and the executive feed already run inside
        ``observe`` and the heartbeat's round.
        """
        now = time.monotonic()
        dt = min(5.0, max(0.0, now - self._last_world_step)) if self._last_world_step else 0.2
        self._last_world_step = now
        try:
            language = getattr(self.brain.frontal, "language", None)
            if language is not None:
                language.current_brain_waves = self.brain_waves()
        except (OSError, ConnectionError) as e:
            note_swallowed("mind.world_step.brain_waves", e)
        workspace = getattr(self.brain.frontal, "global_workspace", None)
        if workspace is not None:
            try:
                workspace.tick(dt)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"workspace world tick failed: {e}")

    def _see_game_frame(self, frame: Any) -> bool | None:
        """Run a game grid through the visual cortex as an image.

        The palette-index grid becomes the RGB frame the occipital
        hierarchy expects, at native resolution. Returns the
        percept's novelty flag, or None when vision isn't online.
        """
        cortex = getattr(self, "visual_cortex", None)
        if cortex is None:
            return None
        try:
            import numpy as np

            rows = (
                frame.tolist()
                if hasattr(frame, "tolist")
                else getattr(frame, "rows", lambda: frame)()
            )
            grid = np.asarray(rows, dtype=np.int64) & 0xF
            rgb = np.asarray(_GAME_PALETTE, dtype=np.uint8)[grid]
            percept = cortex.see(rgb, learn=True)
            return bool(getattr(percept, "is_novel", False))
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.world_step.see_frame", e)
            return None

    def _world_say(self, moment: dict) -> None:
        """Her inner voice on a play moment — composed by her own
        language system from the event, never recited."""
        if self._world_voiced >= _WORLD_VOICE_CAP:
            return
        vocab = getattr(getattr(self.brain.frontal, "language", None), "vocabulary", None)
        if vocab is None:
            return
        try:
            text = vocab._compose_play_moment(moment, self.feel())
        except Exception:  # noqa: BLE001
            return
        if text:
            self._world_voiced += 1
            self._world_last_voiced = text
            inner = getattr(self, "inner_life", None)
            if inner is not None:
                try:
                    noted = inner.note_thought(
                        text,
                        trigger="play",
                        intent="play",
                        metadata={
                            "topic": moment.get("game", ""),
                            "moment": moment.get("kind", ""),
                        },
                    )
                    if noted is not None:
                        return
                except Exception as e:  # noqa: BLE001
                    note_swallowed("parietal_lobe.embodiment.play_moment", e)
            self._emit_live_thought("play", text)

    def _impulse_world(self, chem: int, amount: float) -> None:
        try:
            self._learner_neuro_impulse(chem, amount)
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.world_step._impulse", e)

    def _world_wave(self, wave: Any, amount: float) -> None:
        try:
            from thalamus.brain_waves import add_brain_wave_drive

            add_brain_wave_drive(wave, amount)
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.world_step._world_wave", e)

    def _record_world_decision(
        self,
        token: str,
        choice: dict[str, Any] | None,
        avail: list[Any],
    ) -> None:
        """Stash the decision basis behind this turn's action —
        which drive the striatum honored, how decisive her own
        disposition field was, what else contended. ``note`` carries
        her inner voice only when a moment was actually voiced."""
        rec = dict(self._world_motor.action_scores) if self._world_motor is not None else {}
        scored = sorted(
            (
                (
                    rec.get(str(a.get("action") if isinstance(a, dict) else a), 0.0),
                    a.get("action") if isinstance(a, dict) else a,
                )
                for a in avail
            ),
            reverse=True,
        )
        top = scored[0][0] if scored else 0.0
        second = scored[1][0] if len(scored) > 1 else 0.0
        self.last_world_decision = {
            "action": token,
            "source": (choice or {}).get("source", "unknown"),
            "weight": (choice or {}).get("weight"),
            "confidence": round(min(1.0, max(0.0, top - second) / 2.0), 3),
            "alternatives": [f"ACTION{a}" for score, a in scored[1:3] if score > 0],
            "note": self._world_last_voiced,
        }
        # Where she stood when she decided — the position half of
        # the journal, so a repeated loop shows as a repeated loop.
        pos: tuple[int, int] | None = None
        try:
            me = self.world_model.self_entity
            if me is not None:
                pos = (
                    round(me.centroid[0]),
                    round(me.centroid[1]),
                )
        except Exception:  # noqa: BLE001 — journal is advisory
            pos = None
        self._world_decision_log.append(
            {
                "action": token,
                "source": (choice or {}).get("source", "unknown"),
                "weight": (choice or {}).get("weight"),
                "pos": pos,
            }
        )

    def _world_margin(self) -> float:
        """Decisiveness of the last arbitration — the motor field's
        top-two separation, the same quantity the decision journal
        reports as confidence. A flat field is a guess, and a guess
        is what deliberation is for."""
        if self._world_motor is None:
            return 1.0
        scores = sorted(self._world_motor.action_scores.values(), reverse=True)
        top = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else 0.0
        return min(1.0, max(0.0, top - second) / 2.0)

    def _deliberate_world(self, avail: list[Any]) -> dict[str, float]:
        """Mental simulation at the choice point.

        Sweeps each offered action one step through learned physics
        and lets the imagined outcome bid — the forward-replay half
        of vicarious trial and error. Returns a disposition
        ``{action_id: pull}`` written onto the field for a second
        arbitration; empty when the model has nothing to imagine
        with, so early play and coordinate interfaces stay out of
        the loop.
        """
        try:
            start = self.world_model.start_state()
            if start is None:
                return {}
            valued = self._world_policy.valued_cells()

            def dist(cells: Any) -> float:
                if not valued:
                    return 0.0
                return min(abs(r - vr) + abs(c - vc) for r, c in cells for vr, vc in valued)

            base = dist(start.cells)
            out: dict[str, float] = {}
            for item in avail:
                aid = str(item.get("action") if isinstance(item, dict) else item)
                sim = self.world_model.simulate(start, aid)
                if sim is None or sim.cells == start.cells:
                    # Unknown physics, a refused move, or a hazard —
                    # the forward model offers it no voice.
                    continue
                gain = (base - dist(sim.cells)) * _DELIBERATE_GAIN
                gain += _DELIBERATE_MOVE
                if gain > 0:
                    out[aid] = gain
            return out
        except Exception as e:  # noqa: BLE001 — deliberation is advisory
            logger.debug(f"world deliberation failed: {e}")
            return {}

    def _world_patience(self) -> int:
        """How long a static world gets before she calls it stuck.

        Patience tracks remaining epistemic value: while untried
        actions still promise physics to learn, staying engaged is
        buying information — serotonin's confidence that effort now
        pays later. Once every offered action has a measured effect
        the base count stands.
        """
        ids = self._world_ids
        if not ids:
            return _DRY_BASE
        try:
            unlearned = sum(1 for a in ids if self.world_model.effect_of(a) is None)
        except Exception:  # noqa: BLE001 — patience is advisory
            return _DRY_BASE
        return _DRY_BASE + round(_DRY_LEARN_BONUS * unlearned / len(ids))

    def _world_confuse(self, game: str) -> None:
        """Impasse made felt — confusion, the epistemic emotion that
        says 'my model of this is wrong.'

        Broadcast so the workspace can recruit a restructure, and
        the executive's held objective put up for revision — the
        monitor (uncertainty) and the adjuster (strategy change) are
        different acts; this is the adjuster firing.
        """
        try:
            self._world_bridge.broadcast_observation(
                f"{game}: the world stopped answering {self._world_dry} "
                f"turns ago — my model of it may be wrong",
                topics=["game", game],
                salience=0.7,
                metadata={"game": game, "impasse": self._world_dry},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"confusion broadcast failed: {e}")
        executive = getattr(self.brain.frontal, "executive", None)
        held = self._held_objective
        if executive is None or held is None:
            return
        try:
            executive.revise_intention(held, reason="the world stopped answering — rethink")
            self._held_objective = None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"confusion revision failed: {e}")

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
        action = self._pending_world_action[0] if self._pending_world_action is not None else None
        entities = [
            (e.color, e.cells)
            for e in self.world_model.entities.values()
            if not e.gone
            and not e.is_self
            and not e.part_of_self
            and e.color not in self.world_model.terrain_kinds
        ]
        # When the allocentric fix is missing, self-reported velocity
        # feeds dead reckoning so the position estimate still tracks
        # her motion (corrected on the next fix).
        delta = me.velocity if position is None and me is not None and not me.gone else None
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
            self._emit_live_thought("world", event.text)
            if event.salience >= _STORE_SALIENCE:
                self._store_world_event(WorldEvent(event.kind, event.text, event.salience))

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

    def world_announce(self, text: str, *, topics: list[str] | None = None) -> bool:
        """The world's voice reaches her before the world does.

        A venue announcement — rules read by a proctor, a referee's
        call — is world content like the game's own instructions:
        broadcast to the workspace so every subscribed module hears it,
        and its words grounded into her concept network so they are
        things she can think about rather than sound that ignites
        nothing. What she does with a rule is hers; the channel only
        guarantees it arrived.
        """
        try:
            ok = self._world_bridge.broadcast_observation(
                text,
                topics=["announce", *(topics or [])],
                salience=0.95,
            )
        except Exception as e:  # noqa: BLE001 — a voice hiccup is not a crash
            note_swallowed("parietal_lobe.embodiment.world_announce", e)
            ok = False
        self._ground_world_words(text)
        return ok

    def _ground_world_words(self, text: str) -> None:
        """Words the world speaks to her become concepts.

        An instruction is the world using its own vocabulary; a
        content word she has no concept for is grounded as a
        low-confidence node so her language and thought machinery
        can at least hold it — "arrows" becomes a thing she can
        wonder about instead of sound that ignites nothing.
        """
        try:
            from temporal_lobe.concepts.classify import _FUNCTION_WORDS

            network = self.brain.frontal.network
            for raw in text.split():
                word = raw.strip(".,;:!?()\"'`").lower()
                if len(word) < 3 or word in _FUNCTION_WORDS:
                    continue
                if network._resolve(word) is not None:
                    continue
                network.add_concept(
                    word,
                    confidence=0.3,
                    origin="perception",
                    properties={"kind": "world_word"},
                )
        except Exception as e:  # noqa: BLE001 — grounding is advisory
            logger.debug(f"world-word grounding failed: {e}")

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
            self.brain.frontal.network.add_concept(
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

    def _frame_to_scene(self, frame: Any, background: int | None = None) -> Scene | None:
        """Normalize an observation into a Scene via the substrate."""
        from parietal_lobe.grid import Grid
        from parietal_lobe.scene import Scene, perceive

        if isinstance(frame, Scene):
            return frame
        if frame is None:
            return self.vision.perceive_scene(network=self.brain.frontal.network)
        # A 2-D integer frame (game pixels) is already a grid —
        # no sensor quantization needed.
        try:
            import numpy as np

            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    return perceive(
                        Grid.from_lists(frame.tolist()),
                        background=background,
                    )
                if frame.ndim == 3:
                    return self.vision.perceive_scene(frame, network=self.brain.frontal.network)
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
        objective = self._pending_objective
        self._pending_objective = None
        if objective is None:
            return
        try:
            executive = getattr(self.brain.frontal, "executive", None)
            if executive is None:
                return
            matched = observed == expected
            executive.observe_intention(
                objective,
                actual_outcome=f"move {observed}",
                success=matched,
            )
            if not matched:
                # A served objective that the world refused is
                # revised, not abandoned — its confidence drops so
                # arbitration elects fresher objectives while the
                # intention itself survives to be retried.
                executive.revise_intention(
                    objective,
                    reason="the world did not answer the attempt",
                )
            # The held objective was just served a verdict — the
            # staleness clock restarts.
            self._held_since = time.time()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"executive outcome record failed: {e}")

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
            tag = [float(chem.chemicals.get(CHEM_NAMES[i], 0.0)) for i in range(EMOTIONAL_TAG_SIZE)]
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

    # ─── Hypothesis learning ──────────────────────────────────

    def _world_task_context(self, state: dict[str, Any]) -> Any:
        """The shared task context for this world, recognized once."""
        if self._world_task is not None:
            return self._world_task
        try:
            competence = self.brain.frontal.task_competence
        except AttributeError:
            return None
        self._world_task = competence.recognize(
            domain="world.embodied",
            state=state,
            roles=("self", "object", "navigate"),
        )
        return self._world_task

    def _learn_world_hypotheses(
        self,
        pending: tuple[str, Any] | None,
        events: list[WorldEvent],
    ) -> None:
        """Feed this transition to the schema's hypothesis ledger.

        The pending action + the world-model's flat state are the same
        evidence stream ``SpatialAgent`` feeds on the terminal path —
        one ledger, two adapters. Terminal events close the episode
        for goal inference: a win's ending state abduces goal
        hypotheses, a reset's abduces avoid conditions.
        """
        try:
            state = self.world_model.transition_state()
        except Exception as e:  # noqa: BLE001 — learning must never
            logger.debug(f"world transition state failed: {e}")  # break the loop
            return
        ctx = self._world_task_context(state)
        if ctx is None:
            return
        competence = self.brain.frontal.task_competence
        action = pending[0] if pending is not None else None
        try:
            if self._prev_world_state is not None and action is not None:
                check = competence.record_transition(ctx, action, self._prev_world_state, state)
                self._world_episode_changes = check.observed_changes
            terminal = next((e for e in events if e.kind in ("win", "reset")), None)
            if terminal is not None:
                competence.record_episode(
                    ctx,
                    steps=(),
                    success=terminal.kind == "win",
                    verification_score=1.0 if terminal.kind == "win" else 0.0,
                    verification="external",
                    final_state=state,
                    terminal_changes=self._world_episode_changes,
                )
                self._world_episode_changes = ()
        except Exception as e:  # noqa: BLE001 — hypotheses are advisory
            logger.debug(f"world hypothesis update failed: {e}")
        self._prev_world_state = state

    def _recall_world(self, game: str, events: list[WorldEvent]) -> None:
        """Cue-driven episodic recall — the situation summons its past.

        Hippocampal pattern completion: the live situation (the game
        plus the world model's own summary) is the retrieval cue;
        whatever long-term memory associates to it gets re-read for
        the outcome it once carried. A remembered result lands on
        the kind it was about — *killed me* / *is the goal* /
        *blocked me* / *changed me* — weighted by how strongly the
        memory associates and how salient it was. The residue biases
        the next motor field: remembered goals pull, remembered
        killers push away. A voice, never a plan — the basal ganglia
        still arbitrates, and present-measured evidence outweighs.

        Fires on a cadence and immediately on salient events. The
        residue is rewritten whole each pass — recall is
        cue-dependent: a situation that stops cueing a memory stops
        feeling its pull.
        """
        tick = self.world_model.tick
        salient = any(e.salience >= _STORE_SALIENCE for e in events)
        if not salient and tick - self._world_recall_tick < _RECALL_PERIOD:
            return
        if tick == self._world_recall_tick:
            return
        self._world_recall_tick = tick
        try:
            cue = f"{game}: {self.world_model.summary()}"
            hits = self.client.find_similar(cue, limit=_RECALL_LIMIT)
            # Deep recall: the active index is shallow by design — a
            # situation whose only memories were compressed away still
            # counts. When no active hit lands inside the association
            # gate, reach into the archived tier.
            deep_ids: set[int] = set()
            if not any(h.hamming_distance <= _RECALL_DIST for h in hits):
                deep = self.client.find_similar_archived(cue, limit=_RECALL_LIMIT)
                deep_ids = {h.episode_id for h in deep}
                hits = [*hits, *deep]
        except Exception as e:  # noqa: BLE001 — recall is advisory
            logger.debug(f"world recall query failed: {e}")
            return
        valence: dict[int, float] = {}
        fetched = 0
        for hit in hits:
            if hit.hamming_distance > _RECALL_DIST or fetched >= _RECALL_FETCH:
                continue
            try:
                episode = self.client.retrieve_episode(hit.episode_id)
            except Exception:  # noqa: BLE001 — a lost memory is lost
                continue
            fetched += 1
            # Reconsolidation: a revived memory that actually biased
            # the field earns its way back into the active index.
            if hit.episode_id in deep_ids:
                try:
                    self.client.unarchive_episode(hit.episode_id)
                except Exception as e:  # noqa: BLE001 — advisory, not fatal
                    note_swallowed("mind.world_step.recall_world.unarchive", e)
            assoc = 1.0 - hit.hamming_distance / _RECALL_DIST
            for match in _RECALL_ENTITY_RE.finditer(episode.text):
                color = int(match.group(1))
                sign = _RECALL_OUTCOME_VALENCE[match.group(2)]
                weight = assoc * max(0.2, hit.salience) * sign
                valence[color] = max(-1.0, min(1.0, valence.get(color, 0.0) + weight))
        self._world_last_recall = dict(valence)
        self.world_model.recalled_valence = valence
        if valence:
            self._world_bridge.broadcast_observation(
                self.world_model.summary(),
                topics=["recall", "embodied", game],
                salience=min(0.6, 0.3 + 0.3 * max(map(abs, valence.values()))),
                metadata={"recalled_valence": dict(valence)},
            )

    # ─── Whole-brain world state ────────────────────────────────

    def _publish_world_state(self, scene: Scene, events: list[WorldEvent]) -> None:
        """Push the observed world into the shared channels.

        The frame goes onto the tensor slab as ``world.frame`` (pull:
        any region or daemon process can read the full grid); the
        structured percept goes to the global workspace (push:
        attention, memory, self, ToM, working memory, curiosity all
        receive the same observation). Routine frames stay subliminal;
        salient events ignite. Salient events also broadcast as
        outcomes so their arrival is itself cognitive content.
        """
        bridge = self._world_bridge
        grid = scene.grid
        bridge.publish_frame(grid.to_lists())

        salience = max((e.salience for e in events), default=0.1)
        topics = ["spatial", "embodied", *(e.kind for e in events)]
        relations: list[tuple[str, str, str]] = []
        for e in self.world_model.entities.values():
            if e.gone or e.is_self or e.part_of_self:
                continue
            aff = self.world_model.affordance_of(e.color)
            for outcome in (OUT_WIN, OUT_COLLECT, OUT_HAZARD):
                if aff.get(outcome, 0) > 0:
                    relations.append((f"entity:{e.color}", "affords", outcome))
                    break
            if len(relations) >= 4:
                break
        me = self.world_model.self_entity
        bridge.broadcast_observation(
            self.world_model.summary(),
            topics=topics,
            salience=salience,
            shape=grid.shape,
            metadata={
                "events": [{"kind": e.kind, "salience": e.salience} for e in events],
                "entities": sum(1 for e in self.world_model.entities.values() if not e.gone),
                "embodied": me is not None and not me.gone,
                "relations": relations,
            },
        )
        for event in events:
            if event.salience >= _STORE_SALIENCE:
                bridge.broadcast_outcome(event.kind, event.text, event.salience)
        self._ground_world_verdicts(events)

    def _ground_world_verdicts(self, events: list[WorldEvent]) -> None:
        """Terminal verdicts become thinkable concepts.

        The affordance ledger knows *that* a color is a hazard or a
        goal; edges make it *sayable* — the concept stream can then
        hold "the color-9 thing is a hazard" the way her body already
        holds the aversion. Only outcome kinds with a blamed entity
        ground an edge: contact verdicts, not suspicions.
        """
        _VERDICT_TARGETS = {
            OUT_HAZARD: "hazard",
            OUT_WIN: "goal",
            OUT_COLLECT: "goal",
            OUT_TRANSFORM: "transform",
        }
        try:
            network = self.brain.frontal.network
            for event in events:
                target = _VERDICT_TARGETS.get(event.kind)
                if target is None or event.other_id < 0:
                    continue
                entity = self.world_model.entities.get(event.other_id)
                if entity is None:
                    continue
                network.add_edge(
                    f"color-{entity.color}",
                    target,
                    RelationType.RELATED_TO,
                    0.6,
                    origin="perception",
                )
        except Exception as e:  # noqa: BLE001 — grounding is advisory
            logger.debug(f"world-verdict grounding failed: {e}")

    def _compose_motor_fields(
        self,
        ids: list[str],
        recommended: dict[str, float] | None = None,
    ) -> Any:
        """Project the decision onto the grid + action fields.

        The composed field does double duty: ``WorldPolicy.choose``
        reads it as a bias, and the bridge publishes it for the
        daemon's motor decode (``GET_MOTOR_TARGET``) so the decision
        is readable as distributed state, not just a returned token.
        """
        grid = self._prev_world_grid
        if grid is None or not ids:
            return None
        try:
            from basal_ganglia.motor_field import compose_motor_fields

            motor = compose_motor_fields(self.world_model, ids, grid.height, grid.width)
            self._world_ids = ids
            # Cognition writes onto the field before basal ganglia
            # read it: inferred goals add a disposition toward what
            # satisfies them, undecided mechanic hypotheses add pull
            # toward the action that would test them.
            ctx = self._world_task
            state = self._prev_world_state
            if ctx is not None and state is not None:
                competence = self.brain.frontal.task_competence
                for a in ids:
                    motor.action_scores[a] = (
                        motor.action_scores.get(a, 0.0)
                        + competence.predicted_goal_gain(ctx, a, state)
                        + competence.epistemic_bonus(ctx, a, state)
                    )
            if recommended:
                for a in ids:
                    motor.action_scores[a] = motor.action_scores.get(a, 0.0) + recommended.get(
                        a, 0.0
                    )
            self._world_bridge.publish_motor_fields(motor.spatial, motor.action_scores)
            return motor
        except Exception as e:  # noqa: BLE001 — fields are advisory
            logger.debug(f"motor field composition failed: {e}")
            return None

    def _frame_change(self, prev_grid: Any) -> float:
        """Fraction of cells the last ``observe`` saw change.

        The felt pulse of a step: what couples to dopamine, what
        feeds the stuck detector, what sets broadcast salience —
        measured off her own world-model grid, not reported by a
        delegate. A first frame is wholly new (1.0).
        """
        cur = self._prev_world_grid
        if cur is None:
            return 0.0
        if prev_grid is None:
            return 1.0
        h, w = cur.height, cur.width
        if prev_grid.height != h or prev_grid.width != w:
            return 1.0
        diff = sum(1 for r in range(h) for c in range(w) if prev_grid.at(r, c) != cur.at(r, c))
        return diff / (h * w) if h and w else 0.0

    def _broadcast_world_action(self, choice: dict[str, Any], motor: Any) -> None:
        """The chosen action becomes broadcast content — every region
        sees what she is doing, not just the policy that picked it."""
        cell = choice.get("cell")
        self._world_bridge.broadcast_action(
            choice["action"],
            cell=({"row": cell[0], "col": cell[1]} if cell is not None else None),
            metadata={
                "predicted_effect": (
                    list(self._pending_world_action[1])
                    if self._pending_world_action and self._pending_world_action[1] is not None
                    else None
                ),
                "field_peak": motor.peak if motor is not None else None,
            },
        )

    def _note_world_broadcast(self, item: Any) -> None:
        """Sink target: conscious world content, held for volition.

        The workspace's ``WorldBroadcastSink`` forwards ignited world
        items here; volition reads ``_world_focus`` so autonomous
        urges know when an outer world already holds her attention.
        """
        self._world_focus = {
            "kind": item.metadata.get("kind"),
            "topics": list(item.metadata.get("topics", [])),
            "activation": item.activation,
            "timestamp": item.timestamp,
        }
