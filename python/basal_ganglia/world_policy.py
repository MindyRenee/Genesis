"""World policy — action selection over the parietal world model.

The basal ganglia's job: given everything cortex knows, decide what
to do. The parietal ``WorldModel`` maintains what is true — where
things are, which one is her, what touching them does — and this
module runs *selection among competing channels* over that state.
Every drive bids concurrently, each at its own gain:

- **Motor babbling** — an action whose bodily effect is unknown bids
  near-top while the repertoire is uncomprehended; it must be tried
  before anything can be planned with it.
- **Re-embodiment** — when perception loses the body, the highest
  question is which visible thing answers the command; actions cycle
  systematically until contingency finds her again.
- **Goal pursuit** — a route through learned physics toward a cell
  whose kind collects, wins, or transforms bids high.
- **Constraint satisfaction** — a structured scene with a
  violation-reducing act bids high; a solve in hand is deliberate
  pursuit.
- **Affordance probing** — a thing never touched is an unanswered
  question; stepping into it IS the probe (a failed move is itself
  the observation). Mid gain.
- **Instrumental retrial** — a thing that refused a former body is
  an open question under this one; and while any gate refuses her,
  the kinds that changed her are the way to a body it hasn't. Mid
  gain, above blind probing.
- **Frontier coverage** — committed perseveration toward the
  frontier she is already pushing at; mid-low gain.
- **Baseline foraging** — novelty toward least-visited space, the
  always-on floor and weakest channel.

``choose`` disinhibits the strongest bid — there is no cascade for a
drive to hide inside. Returns action descriptors — ``{"action": id,
"cell": optional (r, c)}`` — for the caller to translate into
whatever protocol the environment speaks. The policy knows nothing
about any environment's action alphabet; ``available`` is ground
truth supplied per step.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from typing import TYPE_CHECKING, Any

from frontal_lobe.world_planner import WorldPlanner
from parietal_lobe.world_model import (
    OUT_BLOCK,
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_TRANSFORM,
    OUT_WIN,
)

if TYPE_CHECKING:
    from basal_ganglia.motor_field import MotorField
    from parietal_lobe.world_model import (
        Cell,
        SimState,
        WorldEntity,
        WorldModel,
    )

# Selection threshold — the basal ganglia's control pathway, per
# Gurney-Prescott-Redgrave: a channel must carry enough salience
# to *disinhibit* new behavior at all. Informed drives bid 0.35+;
# the forage floor sits at 0.25. When every channel is weak, the
# correct dynamics is not switching to the weakest noise — it is
# perseveration: the current effective behavior stays selected.
_SELECT_THRESHOLD = 0.30

# Goal pursuit is a hypothesis, not tenure: a route that is not
# closing distance to what she wants has stopped earning its gain.
# This many consecutive planned acts without a nearer best distance
# drops the bid below the question drives, so unanswered contacts
# get the turn the stalled route was spending.
_PLAN_STALL_DECAY = 6

# A resource meter below this fraction of its observed full scale
# reads as scarcity: pursuit narrows to the depleted kind until a
# pickup restores it. The drive is homeostatic — it says nothing
# about which things are fuel; the meter's kind does.
_LOW_RESERVE = 0.5


class WorldPolicy:
    """Selects actions from the environment's per-step whitelist."""

    def __init__(self, model: WorldModel, seed: int | None = None) -> None:
        """Bind the policy to a world model.

        ``model`` supplies the entities, affordances, and learned
        action physics the planner deliberates over; ``seed`` makes
        the RNG's tie-broken choices reproducible.
        """
        self._model = model
        self._planner = WorldPlanner(model)
        self._rng = random.Random(seed)
        # Distributed motor field composed for the in-flight choice —
        # set by ``choose`` for the duration of one selection. It is
        # a voice inside her scoring, never a crowned leader: she is
        # her own leader, so no field magnitude gets to decide in her
        # place.
        self._field: MotorField | None = None
        # The frontier cell exploration is currently committed to.
        # Perseveration is a drive, not a lock: it stands while the
        # target stays unvisited and reachable, and evidence against
        # it (blocked, refused, stood on) dissolves it.
        self._explore_goal: Cell | None = None
        # The committed frontier route: remaining (action, expected
        # state) pairs from the last sweep. Forward simulation is
        # reserved for decision points (the brain's theta sweeps
        # fire at choice points, not every step); between them the
        # cached route is executed while reality matches it.
        self._frontier_plan: list[tuple[str, SimState]] | None = None
        self._plan_expect: SimState | None = None
        # The persevered channel — (choice, expected SimState after
        # it). Below the selection threshold the current behavior
        # stays disinhibited; it ends when the prediction fails
        # (a bump, a transform — any deviation is a decision
        # point) or when a real bid clears the threshold.
        self._persist: tuple[dict[str, Any], SimState] | None = None
        # Conditional retrials also need prefrontal persistence:
        # recomputing the nearest open question every turn can put two
        # gates on opposite headings and finish neither. The entity
        # under test stays committed until evidence, disappearance, or
        # unreachability closes the question.
        self._retrial_entity: int | None = None
        # Affordance probing needs the same persistence: recomputing
        # the nearest unanswered entity every step can put two
        # questions on opposite headings and answer neither — the
        # classic symmetric-choice starvation. The entity under test
        # holds until it is answered, gone, or unreachable.
        self._probe_entity: int | None = None
        # Trials run while the body is unlocated. Global action counts
        # are lifelong physics evidence and can stay skewed from an old
        # level; re-embodiment needs balanced sampling *now* so one
        # blocked direction cannot starve the search for agency.
        self._reembody_tries: Counter[str] = Counter()
        # Safe waits are a timing channel too: when every mobile option
        # is fatal or already refuted, cycling the nonfatal blocked
        # commands is better than fossilizing one input forever.
        self._wait_tries: Counter[str] = Counter()
        # Stale-pursuit bookkeeping: consecutive planned choices that
        # failed to shrink the best-known distance to a valued cell.
        # Resets on approach progress, on a changed self-state (the
        # route's premises moved), on a changed destination (a new
        # route earns its own patience — stalls are evidence about
        # *that* route, not pursuit in general), and on episode reset.
        self._plan_stall = 0
        self._plan_best_dist = float("inf")
        self._plan_sig: Any = None
        self._plan_goal: frozenset[Cell] | None = None
        self._plan_route: int | None = None
        # The executive's held objective for the in-flight choice —
        # prefrontal persistence arrives as a channel in the
        # competition, not a command over it.
        self._objective: Any = None

    def reset(self) -> None:
        self._field = None
        self._explore_goal = None
        self._frontier_plan = None
        self._plan_expect = None
        self._persist = None
        self._retrial_entity = None
        self._probe_entity = None
        self._reembody_tries.clear()
        self._wait_tries.clear()
        self._plan_stall = 0
        self._plan_best_dist = float("inf")
        self._plan_sig = None
        self._plan_goal = None
        self._plan_route = None
        self._objective = None

    def choose(
        self,
        available: list[str],
        field: MotorField | None = None,
        executive: Any = None,
    ) -> dict[str, Any] | None:
        """Pick the next action, or ``None`` if nothing is offered.

        Selection is a *competition*, not a cascade. Every drive the
        world model's state supports bids at once — babbling, goal
        pursuit, constraint satisfaction, probing, frontier
        coverage, baseline foraging — and the strongest concurrent
        bid wins. There is no priority order: a drive loses when its
        evidence is weak and wins the moment it is the strongest
        thing she knows to want. The ``field`` (see
        ``basal_ganglia.motor_field``) is a voice inside the
        scoring — its dispositions break close calls, never a
        verdict in her place.

        Weights are drive *gain*, the way hunger and curiosity sit
        on different channels: motor babbling is imperative while
        the body is uncomprehended (1.0 down to 0.7 as the
        repertoire fills), a plan backed by learned value bids 0.8,
        an unanswered affordance question 0.5, committed frontier
        coverage 0.35, baseline foraging 0.25. Every bid carries
        evidence or it doesn't exist — silent channels can't win.

        ``executive`` — when the prefrontal layer is present — holds
        her objective into the competition as its own channel and
        owns the stop check on the winning impulse: the basal
        ganglia arbitrate, the executive supplies persistence and
        suppression.
        """
        if not available:
            return None
        self._field = field
        self._objective = (
            executive.active_intention if executive is not None else None
        )
        bids = [
            b
            for b in (
                self._bid_reembody(available),
                self._bid_calibrate(available),
                self._bid_plan(available),
                self._bid_board(available),
                self._bid_intention(available),
                self._bid_probe(available),
                self._bid_reprobe(available),
                self._bid_frontier(available),
                self._bid_forage(available),
            )
            if b is not None
        ]
        if not bids:
            return None
        # Outcome gating: a bid whose action her own forward model
        # already predicts will fail is not a real offer — issuing it
        # spends a move re-confirming evidence she has. The forward
        # model is consulted only where it has learned physics: an
        # unknown effect is still a question, a known-but-failing
        # transition is a veto.
        start = self._model.start_state()
        if start is not None:
            nonfatal = [
                b
                for b in bids
                if not self._model.action_fatal(
                    start, str(b["action"])
                )
            ]
            # A predicted death is a veto, not a fallback candidate.
            # If every *bid* is lethal the repertoire itself may still
            # contain an escape — the drives do not have to have asked
            # for it before survival gets to use it. When every offered
            # action is fatal, the least-explored one is still the only
            # experiment a mandatory-action world leaves open.
            if not nonfatal:
                escape_safe = [
                    a for a in available
                    if not self._model.action_fatal(start, str(a))
                ]
                escape = [
                    a for a in escape_safe
                    if not self._known_failure(start, str(a))
                ] or escape_safe
                if not escape:
                    # Some interfaces do not offer silence: returning
                    # nothing ends the attempt more surely than the
                    # predicted contact does. With every offered act
                    # condemned, the least-explored one is the only
                    # remaining experiment — uncertainty in the lethal
                    # forecast is preferable to certain forfeiture.
                    action = self._explore_action(
                        self._model.self_entity, available
                    )
                    forced = {
                        "action": action,
                        "cell": self._cell_for(action),
                        "source": "forced",
                        "weight": 0.05,
                    }
                    self._persist = self._expected(forced)
                    return forced
                action = self._explore_action(
                    self._model.self_entity, escape
                )
                nonfatal = [{
                    "action": action,
                    "cell": self._cell_for(action),
                    "source": "evade",
                    "weight": 0.35,
                }]
            bids = nonfatal
            viable = [
                b
                for b in bids
                if not self._known_failure(start, str(b["action"]))
            ]
            if viable:
                bids = viable
            else:
                alternates = [
                    a for a in available
                    if not self._model.action_fatal(start, str(a))
                    and not self._known_failure(start, str(a))
                ]
                if not alternates:
                    waits = [
                        a for a in available
                        if not self._model.action_fatal(start, str(a))
                    ]
                    if not waits:
                        return None
                    action = self._wait_action(waits)
                    bids = [{
                        "action": action,
                        "cell": self._cell_for(action),
                        "source": "wait",
                        "weight": _SELECT_THRESHOLD + 0.01,
                    }]
                else:
                    action = self._explore_action(
                        self._model.self_entity, alternates
                    )
                    bids = [{
                        "action": action,
                        "cell": self._cell_for(action),
                        "source": "evade",
                        "weight": 0.35,
                    }]
        winner = max(bids, key=self._strength)
        self._track_pursuit(winner)
        me = self._model.self_entity
        if me is None or me.gone:
            self._reembody_tries[str(winner["action"])] += 1
        if winner["source"] == "wait":
            self._wait_tries[str(winner["action"])] += 1
        else:
            self._wait_tries.clear()
        # Stop check: while an executive is present it owns the
        # suppression decision on the winning impulse — otherwise
        # the competition's own floor decides.
        suppressed = self._strength(winner) < _SELECT_THRESHOLD
        if not suppressed and executive is not None:
            try:
                suppressed = executive.inhibit_response(
                    self._strength(winner)
                ).inhibited
            except Exception:  # noqa: BLE001 — advisory machinery
                suppressed = False
        if suppressed:
            # No informed drive clears the control pathway's bar.
            # Perseverate: if the last action is still producing
            # its predicted effect, the current behavior continues —
            # a 0.25 floor does not get to redirect her because it
            # is the only voice left.
            kept = self._perseveration(available)
            if kept is not None:
                return kept
            self._persist = None
        self._persist = self._expected(winner)
        return winner

    def _known_failure(self, start: SimState, action: str) -> bool:
        """The action has a learned effect and it fails from here.

        Distal acts are excluded — their outcome lives off the
        body's transition function. A movement action whose effect
        is learned but whose simulated result is ``None`` is her
        physics saying *this will not move you* — the bump is
        already learned, spending it again teaches nothing.
        """
        model = self._model
        if action in model.coordinate_actions:
            return False
        effect = model.effect_of(action)
        if effect is None or effect == (0, 0):
            return False
        return model.simulate(
            start, action, explore_unknown=True
        ) is None

    def _perseveration(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Continue the running channel while its prediction holds.

        The check is her own forward model: the state the last
        action was simulated to produce against the state she now
        actually occupies. Cells and signature compared; distal
        acts and unavailable actions can't perseverate.
        """
        entry = self._persist
        if entry is None:
            return None
        choice, expected = entry
        if choice["action"] not in available:
            return None
        start = self._model.start_state()
        if (
            start is None
            or start.cells != expected.cells
            or start.signature != expected.signature
        ):
            return None
        # Still on script — extend the chain one step so a long
        # effective run perseverates rather than expiring after
        # a single step.
        nxt = self._model.simulate(start, choice["action"])
        if nxt is None:
            return None
        self._persist = (choice, nxt)
        return {
            "action": choice["action"],
            "cell": choice.get("cell"),
            "source": "persist",
            "weight": _SELECT_THRESHOLD,
        }

    def _expected(
        self, bid: dict[str, Any]
    ) -> tuple[dict[str, Any], SimState] | None:
        """The SimState the winning act should leave behind —
        the prediction perseveration checks against reality."""
        model = self._model
        if bid["action"] in model.coordinate_actions:
            return None  # distal effects aren't simulatable
        start = model.start_state()
        if start is None:
            return None
        nxt = model.simulate(start, bid["action"])
        if nxt is None:
            return None
        return (bid, nxt)

    def _track_pursuit(self, winner: dict[str, Any]) -> None:
        """Update the stale-pursuit counter from the winning bid.

        A signature change resets the ledger — a transform moves
        the route's premises — and so does any approach progress:
        pursuit counts as working while the nearest distance to a
        valued cell is still being beaten. Only planned acts tick
        the counter; another drive's turn is not the route's fault.
        """
        sig = self._model.self_signature()
        if sig != self._plan_sig:
            self._plan_sig = sig
            self._plan_stall = 0
            self._plan_best_dist = float("inf")
        if winner["source"] != "planned":
            return
        # Progress is measured against the route's destination. The
        # globally nearest valued cell can be unreachable — closing
        # on the real goal must not read as stalling on that one.
        goal = winner.get("goal")
        if goal is not None:
            goal = frozenset(goal)
        if goal != self._plan_goal:
            self._plan_goal = goal
            self._plan_best_dist = float("inf")
            self._plan_route = None
        dist = self._distance_to(
            winner.get("goal") or self._valued_cells()
        )
        # A route that must wrap a wall legitimately lengthens its
        # straight line to the goal; the corridor detour is still
        # progress. The plan's own remaining length is the honest
        # measure — a working route shortens every turn it runs.
        route = winner.get("route")
        progressed = dist < self._plan_best_dist or (
            route is not None
            and self._plan_route is not None
            and route < self._plan_route
        )
        if route is not None:
            self._plan_route = route
        if progressed:
            self._plan_best_dist = min(dist, self._plan_best_dist)
            self._plan_stall = 0
        else:
            self._plan_stall += 1

    def _distance_to(self, cells: set[Cell]) -> float:
        """BBox distance from her footprint to the nearest target cell."""
        fp = self._model.self_footprint()
        if not fp or not cells:
            return float("inf")
        top = min(r for r, _ in fp)
        bottom = max(r for r, _ in fp)
        left = min(c for _, c in fp)
        right = max(c for _, c in fp)
        return min(
            max(top - r, 0, r - bottom) + max(left - c, 0, c - right)
            for r, c in cells
        )

    def _strength(self, bid: dict[str, Any]) -> float:
        """Bid strength for selection: evidence weight plus the
        field's disposition as the close-call term, plus a hair of
        noise so identical bids don't fossilize an order."""
        field_term = 0.0
        if self._field is not None:
            field_term = self._field.action_scores.get(
                str(bid["action"]), 0.0
            )
        return (
            bid["weight"]
            + 0.05 * field_term
            + self._rng.random() * 0.01
        )

    def _bid_reembody(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Recover agency after the tracked body disappears.

        A reset, transform, or wholesale scene replacement can leave
        several visible candidates and no self. The answer cannot be
        inferred from appearance: it has to be measured by acting.
        Sampling must be balanced across the *current* loss of
        embodiment — lifelong try counts may already favor the very
        direction that is now blocked — so this channel cycles the
        available repertoire until some entity's contingent motion
        names her again.
        """
        me = self._model.self_entity
        if me is not None and not me.gone:
            self._reembody_tries.clear()
            return None
        non_reset = [
            a for a in available
            if self._model.action_labels.get(a) != "reset"
        ]
        candidates = non_reset or available
        least = min(self._reembody_tries[a] for a in candidates)
        choices = [
            a for a in candidates if self._reembody_tries[a] == least
        ]
        action = self._rng.choice(choices)
        return {
            "action": action,
            "cell": self._cell_for(action),
            "source": "reembody",
            "weight": 0.95 if non_reset else 0.15,
        }

    def _bid_calibrate(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Motor babbling — try an action whose bodily effect is
        still unknown.

        An untried direction can seal a corridor out of every plan
        the model ever makes, so while any of the repertoire is
        uncomprehended this bids near the top; a body is required —
        you cannot babble limbs you don't have. Interface-labeled
        reset actions babble last and cheap (restarting to satisfy
        curiosity forfeits real progress).
        """
        model = self._model
        me = model.self_entity
        if me is None or me.gone:
            return None
        untried = [
            a for a in available if not model.action_understood(a)
        ]
        if not untried:
            return None
        non_reset = [
            a for a in untried if model.action_labels.get(a) != "reset"
        ]
        candidates = non_reset or untried
        evidence = {
            a: sum(model.action_effects.get(a, Counter()).values())
            for a in candidates
        }
        least = min(evidence.values())
        choices = [a for a in candidates if evidence[a] == least]
        pick = self._rng.choice(choices)
        weight = (
            0.7 + 0.3 * len(non_reset) / len(available)
            if non_reset
            else 0.15
        )
        return {
            "action": pick,
            "cell": self._cell_for(pick),
            "source": "calibrate",
            "weight": weight,
        }

    def _bid_plan(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Goal pursuit — a route through learned physics toward a
        cell whose kind has learned value. Deliberate pursuit is a
        strong channel when it exists."""
        seq = self._planner.plan(
            self._valued_cells(), available
        )
        if not seq:
            return None
        action = seq[0][0]
        # A stalled pursuit bids below the question drives (probe
        # 0.5, reprobe 0.55) but above coverage: the goal is still
        # wanted, the route just has to wait for the physics it
        # is missing.
        weight = 0.45 if self._plan_stall >= _PLAN_STALL_DECAY else 0.8
        return {
            "action": action,
            "cell": self._cell_for(action),
            # The route's own destination, not the nearest valued
            # cell: the nearest one may be a walled-off decoy the
            # planner correctly skipped, and judging progress by
            # it would mark a working route as stalled.
            "goal": set(seq[-1][1].cells),
            "route": len(seq),
            "source": "planned",
            "weight": weight,
        }

    def _bid_board(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Constraint satisfaction — the structured-scene machinery
        holds a violation-reducing act. A solve in hand is
        deliberate pursuit: same league as a route to goal value."""
        cell = self._model.board_plan()
        if cell is None:
            return None
        actions = [
            a for a in self._model.coordinate_actions if a in available
        ]
        if not actions:
            return None
        # The control that has answered most is the one to press.
        action = max(
            actions,
            key=lambda a: (
                sum(
                    v for (act, _c), v in self._model.click_changes.items()
                    if act == a
                ),
                -self._model.action_tries[a],
            ),
        )
        return {
            "action": action,
            "cell": cell,
            "source": "planned",
            "weight": 0.8,
        }

    # Objective targets naming what her model has learned to want
    # route straight at the valued cells; targets naming what kills
    # her yield no route — avoidance is the passability gate's job,
    # not a pursuit bid's.
    _OBJECTIVE_GOALS = frozenset(
        {"goal", "win", "exit", "prize", "finish", "collect", "score"}
    )
    _OBJECTIVE_AVOID = frozenset(
        {"hazard", "death", "die", "danger", "lose", "losing", "kill"}
    )

    def _bid_intention(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Held objective — what the executive is pursuing on purpose.

        The prefrontal layer holds the objective; the basal ganglia
        route it through the same model-based planner every other
        channel uses. A held intention is a strong channel — above
        an affordance question, below a plan backed by learned
        value — because it is sustained deliberately, not on
        reflex.
        """
        intention = self._objective
        if intention is None:
            return None
        targets = self._objective_targets(intention)
        if not targets:
            return None
        action = self._planner.next_action(targets, available)
        if action is None:
            return None
        return {
            "action": action,
            "cell": self._cell_for(action),
            "source": "intention",
            "weight": 0.6,
        }

    def _objective_targets(self, intention: Any) -> set[Cell]:
        """Cells the held objective can mean.

        The objective's kind chooses which knowledge channel serves
        it — ``learn:`` wants unanswered affordance questions,
        ``observe:``/``explore:`` want coverage — and a target that
        names a valued thing ("goal", "exit") routes at what the
        model has learned to want. An aversive target returns no
        cells: she does not route toward what she means to avoid.
        """
        target = str(getattr(intention, "target", "") or "").lower()
        if target in self._OBJECTIVE_AVOID:
            return set()
        if target in self._OBJECTIVE_GOALS:
            valued = self._valued_cells()
            if valued:
                return valued
            # A goal named before she has valued anything is still
            # a question: the objective's kind picks the channel
            # that goes looking for it.
        kind = str(getattr(intention, "kind", "") or "").lower()
        if kind == "learn":
            return self._unprobed_cells()
        return set(self._frontier_targets())

    def _bid_probe(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Affordance probing — an unanswered question about what a
        thing does to her. Mid gain: beats wandering, loses to a
        goal or a babbling body."""
        probe = self._probe_unknown(available)
        if probe is None:
            return None
        probe["weight"] = 0.5
        return probe

    def _bid_frontier(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Committed frontier coverage — perseveration toward the
        frontier she is already pushing at. Loses to probe and
        goal evidence, beats raw novelty."""
        action = self._frontier_step(available)
        if action is None:
            return None
        return {
            "action": action,
            "cell": self._cell_for(action),
            "source": "frontier",
            "weight": 0.35,
        }

    def _bid_forage(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Baseline foraging — the always-on floor. Novelty and
        frontier scoring still pick the direction; the bid itself
        is the weakest channel, so she wanders only when nothing
        better-informed is on the table."""
        action = self._explore_action(
            self._model.self_entity, available
        )
        return {
            "action": action,
            "cell": self._cell_for(action),
            "source": "forage",
            "weight": 0.25,
        }

    def _cell_for(self, action: str) -> Cell | None:
        """Coordinate-typed actions need a target; simple ones don't.

        The interface labels which actions take a point — the same
        convention every participant sees. What the point *does* is
        still hers to learn. When a motor field was composed for this
        step its spatial peak is the target — the distributed decision
        supplies the coordinate.
        """
        target = (
            self._field.peak
            if self._field is not None and self._field.peak is not None
            else self._most_informative_cell()
        )
        if action in self._model.coordinate_actions:
            return target
        if self._model.self_entity is None:
            return target
        return None

    def _valued_cells(self) -> set[Cell]:
        """Cells of live entities whose kind is worth reaching.

        Collectibles and goal-kinds are ends. Transformers are a
        means — worth reaching only while her *current* self-state
        has been refused somewhere: the gate said no to what she
        is, and the thing that rewrites her is how that changes.
        """
        model = self._model
        sig = model.self_signature()
        # A refusal counts only while its verdict still binds — the
        # thing, or its kind, told this very signature no and nothing
        # unexplained has touched her since. The same predicate that
        # keeps a refused door out of pursuit is the evidence that
        # her state needs changing.
        needs_change = sig is not None and any(
            not e.gone and model.state_refused(e, sig)
            for e in model.entities.values()
        )
        pool = [
            e
            for e in model.entities.values()
            if not e.gone and not e.is_self and not e.part_of_self
            and e.color not in model.terrain_kinds
            and not model.is_display(e)
            and (
                model.affordance_of(e.color)[OUT_COLLECT] > 0
                or model.affordance_of(e.color)[OUT_WIN] > 0
                or (
                    needs_change
                    and (
                        model.affordance_of(e.color)[OUT_TRANSFORM] > 0
                        or e.color in model._state_changers
                    )
                )
            )
            # A door that already told this body no is a question for
            # the retest channels, not a destination: walking to it
            # again cannot succeed until her state has moved.
            and not model.state_refused(e, sig)
            and not model.hazard_entity(e)
        ]
        # Homeostasis, both directions of one threshold. A fuel
        # gauge running low narrows every pursuit channel to what
        # refills it: a starving animal does not finish the puzzle
        # before it eats — and in a step-budgeted world every route
        # is fuel-financed anyway. A metered kind that is NOT low
        # earns no steps at all: contact still pays the refill, but
        # steering toward it spends reserve to bank headroom the
        # walk may already have burned. Food is a destination when
        # hungry; it is scenery when fed. The scarce side falls
        # back to the full pool when the depleted kind has no
        # reachable thing left; the fed side spares an entity that
        # holds value beyond the refill.
        levels = model.resource_levels()
        scarce = {color for color, level in levels.items() if level < _LOW_RESERVE}
        fed = set(levels) - scarce
        if fed:
            pool = [
                e
                for e in pool
                if e.color not in fed
                or model.affordance_of(e.color)[OUT_WIN] > 0
                or model.affordance_of(e.color)[OUT_TRANSFORM] > 0
                or e.color in model._state_changers
            ]
        if scarce:
            fuel = [e for e in pool if e.color in scarce]
            if fuel:
                pool = fuel
        return {
            cell
            for e in pool
            for cell in e.cells
        }

    def _probe_unknown(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Move toward what contact would still teach her.

        Two frontiers: a kind never touched at all, and — once her
        own state can change — an entry-dependent thing whose answer
        for *this* self-state is unknown. A gate that refused her
        before stays a question every time she is remade.
        """
        unprobed = self._unprobed_entities()
        if not unprobed:
            self._probe_entity = None
            return None
        # As with retrials, a stationary thing is the measurable
        # question; a mover's reported cells are already history.
        # Static questions get first refusal of the route, then the
        # moving ones remain available if they are all she can test.
        stable = [
            e for e in unprobed
            if not e.spontaneous
            and e.velocity == (0.0, 0.0)
            and e.churn_ticks * 2 < e.n_seen
        ]
        candidates = stable or unprobed
        footprint = self._model.self_footprint()

        def distance(entity: WorldEntity) -> int:
            if not footprint:
                return 0
            top = min(r for r, _ in footprint)
            bottom = max(r for r, _ in footprint)
            left = min(c for _, c in footprint)
            right = max(c for _, c in footprint)
            return min(
                max(top - r, 0, r - bottom)
                + max(left - c, 0, c - right)
                for r, c in entity.cells
            )

        # Commitment first, then the nearest unanswered question:
        # the entity under test leads the ordering while it is still
        # a question, so two equidistant targets cannot trade the
        # bid back and forth every step.
        committed = next(
            (e for e in candidates if e.id == self._probe_entity),
            None,
        )
        if committed is None:
            self._probe_entity = None
        ordered = ([committed] if committed is not None else []) + [
            e for e in sorted(candidates, key=distance)
            if e is not committed
        ]
        for entity in ordered:
            target = set(entity.cells) | {
                (round(entity.centroid[0]), round(entity.centroid[1]))
            }
            action = self._planner.next_action(target, available)
            if action is not None and action in available:
                self._probe_entity = entity.id
                return {"action": action, "cell": None, "source": "probe"}
        self._probe_entity = None
        return None

    def _unprobed_entities(self) -> list[WorldEntity]:
        """Entities whose contact would still teach her something."""
        model = self._model
        sig = model.self_signature()
        return [
            e
            for e in model.entities.values()
            if not e.gone and not e.is_self and not e.part_of_self
            and e.color not in model.terrain_kinds
            and not model.is_display(e)
            and (
                not model.affordance_of(e.color)
                or (
                    sig is not None
                    and (
                        (
                            sig not in e.open_states
                            and sig not in e.blocked_states
                            and (
                                e.blocked_states
                                or e.open_states
                                or model.affordance_of(e.color)[OUT_BLOCK]
                                > 0
                                or model.affordance_of(e.color)[OUT_HAZARD]
                                > 0
                            )
                        )
                        or (
                            # It refused the self she was measured
                            # in — an epoch ago. What it answers now
                            # is again an open question.
                            sig in e.blocked_states
                            and e.blocked_epochs.get(sig)
                            not in (None, model._state_epoch)
                        )
                    )
                )
            )
        ]

    def _unprobed_cells(self) -> set[Cell]:
        """Cells whose contact would still teach her something."""
        return {
            cell
            for e in self._unprobed_entities()
            for cell in e.cells
        }

    def _bid_reprobe(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Instrumental retrial — the changed-body questions.

        Two targets, both conditional: a thing that refused a body
        her current signature has no verdict for (the gate that
        stopped a former self is the most informative place this
        body can go), and — while any kind refuses her current
        state — a kind that rewrote her before: transformers are
        the path to a body the gate hasn't refused. The first
        carries a direct question; the second manufactures the
        state the first needs.
        """
        model = self._model
        sig = model.self_signature()
        if sig is None:
            return None
        retrials = [
            e
            for e in model.entities.values()
            if not e.gone
            and not e.is_self
            and not e.part_of_self
            and e.blocked_states
            and (
                (
                    sig not in e.blocked_states
                    and sig not in e.open_states
                )
                or (
                    # Its verdict is from an older epoch — hidden
                    # state may have moved since it was taken.
                    sig in e.blocked_states
                    and e.blocked_epochs.get(sig)
                    not in (None, model._state_epoch)
                )
            )
            and not model.is_display(e)
            and not model.hazard_entity(e)
        ]
        if model.blocked_kinds:
            retrials += [
                e
                for e in model.entities.values()
                if not e.gone
                and (
                    e.color in model.transforms
                    # Kinds whose unexplained contact opened a fresh
                    # epoch — candidate hidden-state changers even
                    # when the rewrite never reached her surface.
                    or e.color in model._state_changers
                )
                and sig not in e.open_states
                and not model.is_display(e)
                and not model.hazard_entity(e)
                and e not in retrials
            ]
        if not retrials:
            return None
        # Stable questions first. A mobile entity's current cells are
        # already stale by the time a route is planned — aiming at
        # them becomes pursuit of a moving painting of the target.
        # Fixed gates, on the other hand, are exactly the conditional
        # measurements a changed body can take. Mobile questions stay
        # eligible when nothing static remains unanswered.
        stable = [
            e for e in retrials
            if not e.spontaneous
            and e.velocity == (0.0, 0.0)
            # A centroid that holds still while its silhouette
            # churns every sighting is not a fixed question — the
            # reported cells are a moving painting of the target.
            and e.churn_ticks * 2 < e.n_seen
        ]
        candidates = stable or retrials
        footprint = model.self_footprint()

        def distance(entity: WorldEntity) -> int:
            if not footprint:
                return 0
            top = min(r for r, _ in footprint)
            bottom = max(r for r, _ in footprint)
            left = min(c for _, c in footprint)
            right = max(c for _, c in footprint)
            return min(
                max(top - r, 0, r - bottom)
                + max(left - c, 0, c - right)
                for r, c in entity.cells
            )

        committed = next(
            (e for e in candidates if e.id == self._retrial_entity),
            None,
        )
        if committed is None:
            self._retrial_entity = None
        # When this very signature has been refused, re-measuring
        # gates it cannot pass is a wasted walk: the thing that
        # rewrites her manufactures the only state that can change
        # the answer. Producers rank first then; otherwise the
        # nearest open question is the cheap experiment.
        needs_change = any(
            not e.gone and model.state_refused(e, sig)
            for e in model.entities.values()
        )

        def rank(entity: WorldEntity) -> tuple[bool, int]:
            produces = (
                entity.color in model.transforms
                or entity.color in model._state_changers
            )
            return (needs_change and not produces, distance(entity))

        ordered = ([committed] if committed is not None else []) + [
            e for e in sorted(candidates, key=rank)
            if e is not committed
        ]
        for entity in ordered:
            # The centroid is the stable point inside a flickering
            # silhouette: a goal set that is only the current cells
            # repaints the destination every tick and paces between
            # two adjacent answers forever.
            target = set(entity.cells) | {
                (round(entity.centroid[0]), round(entity.centroid[1]))
            }
            action = self._planner.next_action(target, available)
            if action is None:
                continue
            self._retrial_entity = entity.id
            break
        else:
            self._retrial_entity = None
            return None
        return {
            "action": action,
            "cell": self._cell_for(action),
            "source": "reprobe",
            "weight": 0.55,
        }

    def _frontier_step(self, available: list[str]) -> str | None:
        """One step along the committed frontier route.

        Forward sweeps fire at decision points: on commitment, on
        commitment dissolution, and whenever the world deviates
        from the route's prediction. Between sweeps the cached
        plan is executed verbatim — the model predicted this path
        already, re-searching it every step is deliberation noise,
        not thought.
        """
        model = self._model
        goal = self._explore_goal
        if goal is not None and (
            model.visits[goal]
            or goal in model.known_blocked
            or goal in model.refused_cells
        ):
            goal = self._explore_goal = None
            self._frontier_plan = None
            self._plan_expect = None
        if goal is not None:
            action = self._plan_step(available)
            if action is not None:
                return action
            # Route spent or broken — this is a decision point:
            # dissolve and re-sweep below.
            self._explore_goal = None
            self._frontier_plan = None
            self._plan_expect = None
        for goal in self._frontier_targets():
            seq = self._planner.plan({goal}, available)
            if seq:
                self._explore_goal = goal
                self._frontier_plan = seq
                action = self._plan_step(available)
                if action is not None:
                    return action
        return None

    def _plan_step(self, available: list[str]) -> str | None:
        """Next action of the cached route, while it still predicts.

        Before each step the current state is compared against the
        state the previous action was simulated to leave behind —
        agreement means reality is still following the plan and no
        new deliberation is owed; deviation means the route's
        premises are stale and it is handed back to the sweep.
        """
        seq = self._frontier_plan
        if seq is None:
            return None
        if self._plan_expect is not None:
            start = self._model.start_state()
            if (
                start is None
                or start.cells != self._plan_expect.cells
                or start.signature != self._plan_expect.signature
            ):
                return None
            self._plan_expect = None
        while seq and seq[0][0] not in available:
            seq.pop(0)
        if not seq:
            return None
        action, expected = seq.pop(0)
        self._plan_expect = expected
        return action

    def _frontier_targets(self, limit: int = 6) -> list[Cell]:
        """Seen-but-unvisited cells, steered toward open questions.

        The frontier of an occupancy map is the cells she has seen
        free but never stood on — walking one reveals what sits
        beyond it, which is how a map grows. Pure coverage is
        nearest-first; but while anything remains unexplained —
        things never touched, kinds whose only answer was "nothing
        visible", wants no plan could reach — the frontier cells
        that matter are the ones *on the way to it*. A cell's pull
        is its map-distance to the nearest question, measured
        through the same open floor she would walk; a room no open
        path leads to pulls nothing.
        """
        model = self._model
        footprint = model.self_footprint()
        if not footprint:
            return []
        blocked = model.known_blocked | model.refused_cells
        open_map = model.known_free - blocked
        seen: set[Cell] = set(footprint)
        queue: deque[Cell] = deque(footprint)
        dist: dict[Cell, int] = dict.fromkeys(footprint, 0)
        out: list[Cell] = []
        expansions = 0
        while queue and expansions < 20_000:
            cell = queue.popleft()
            expansions += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt in seen or nxt in blocked or nxt not in open_map:
                    continue
                seen.add(nxt)
                dist[nxt] = dist[cell] + 1
                if model.visits[nxt] == 0:
                    out.append(nxt)
                queue.append(nxt)
        questions = self._question_cells()
        if not questions:
            return out[:limit]
        qdist = self._open_distances(questions, open_map)
        out.sort(key=lambda c: (qdist.get(c, 1 << 20), dist[c]))
        return out[:limit]

    def _question_cells(self) -> set[Cell]:
        """Cells of things contact has not yet answered for her.

        Three kinds of question: entities whose kind she has never
        touched (or whose answer depends on a self-state she no
        longer wears), members of kinds whose contacts remain
        unexplained — hidden-state candidates — and things she
        wants that no route currently reaches. The frontier should
        grow toward all of them; a question is only worth exploring
        toward while it stays unreached.
        """
        model = self._model
        cells: set[Cell] = set(self._valued_cells())
        for entity in self._unprobed_entities():
            cells |= entity.cells
        for entity in model.entities.values():
            if (
                entity.gone
                or entity.is_self
                or entity.part_of_self
                or entity.state_of_self
                or model.is_display(entity)
            ):
                continue
            if entity.color in model._state_changers:
                cells |= entity.cells
        return cells

    @staticmethod
    def _open_distances(
        seeds: set[Cell], open_map: set[Cell]
    ) -> dict[Cell, int]:
        """Map-distance outward from seed cells through open floor.

        Seeds may sit outside the open map — a question's own cells
        are occupied, not floor; their pull still propagates from
        whatever open cells surround them, so a thing ringed by
        walls radiates nothing.
        """
        dist: dict[Cell, int] = dict.fromkeys(seeds, 0)
        queue: deque[Cell] = deque(seeds)
        expansions = 0
        while queue and expansions < 20_000:
            cell = queue.popleft()
            expansions += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt in dist or nxt not in open_map:
                    continue
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
        return dist

    def _wait_action(self, waits: list[str]) -> str:
        """Choose the least-used safe time-advancing command."""
        least = min(self._wait_tries[a] for a in waits)
        return self._explore_action(
            self._model.self_entity,
            [a for a in waits if self._wait_tries[a] == least],
        )

    def _explore_action(
        self, me: WorldEntity | None, available: list[str]
    ) -> str:
        """Novelty-seeking with a frontier bias.

        Beyond preferring least-tried actions and least-visited
        destinations, an action is pulled toward the frontier — cells
        adjacent to space never visited — so coverage is systematic
        rather than a random walk. The motor field adds its
        disposition here as one term among the drives, not a verdict
        over them. With no tracked self the footprint terms go
        silent and the informed terms decide.
        """
        model = self._model
        best = available[0]
        best_score = -1.0
        fp = model.self_footprint() if me is not None else frozenset()
        for action in available:
            # The label prior is a hypothesis for exploration scoring;
            # observed effects replace it wholesale once they exist.
            effect = model.expected_effect(action)
            tries = model.action_tries[action]
            novelty = 0.0
            frontier = 0.0
            if effect is not None and fp:
                dest_cells = {
                    (r + effect[0], c + effect[1]) for r, c in fp
                }
                # A stride is refused by anything inside it — the
                # bump evidence lives on the path, not only on the
                # landing strip.
                path_cells = model._swept(fp, effect)
                if path_cells & model.known_blocked:
                    novelty = -1.0
                else:
                    novelty = sum(
                        1 for cell in dest_cells if model.visits[cell] == 0
                    ) / len(dest_cells)
                    unseen = sum(
                        1
                        for r, c in path_cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        for n in [(r + dr, c + dc)]
                        if n in model.known_free and model.visits[n] == 0
                    )
                    frontier = min(1.0, unseen / len(dest_cells))
            score = (
                novelty + frontier + 1.0 / (1 + tries)
                + self._rng.random() * 0.1
            )
            if self._field is not None:
                # The action disposition the motor field carries —
                # decisions laid out over the space feed back into
                # the discrete choice.
                score += self._field.action_scores.get(action, 0.0)
            if model.action_labels.get(action) == "reset":
                score -= 1.0  # restarting the episode is a last resort
            if score > best_score:
                best, best_score = action, score
        return best

    def _most_informative_cell(self) -> Cell | None:
        """A cell worth pointing at: the least-understood entity."""
        best: WorldEntity | None = None
        least = float("inf")
        for entity in self._model.entities.values():
            if (
                entity.gone
                or entity.color in self._model.terrain_kinds
                or self._model.is_display(entity)
            ):
                continue
            known = sum(self._model.affordance_of(entity.color).values())
            if known < least:
                best, least = entity, known
        if best is None:
            return None
        return (round(best.centroid[0]), round(best.centroid[1]))
