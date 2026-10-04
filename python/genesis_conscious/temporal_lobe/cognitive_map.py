"""Cognitive map — the hippocampal-entorhinal allocentric world model.

Medial temporal spatial memory: the world-centered complement to the
parietal lobe's egocentric stream. The parietal model answers "where
are things relative to me right now"; this map answers "how is this
place laid out" — structure that persists across episodes, is anchored
to landmarks rather than to her body, and is reinstated by pattern
completion when a familiar environment is re-entered.

The representation follows the neuroscience rather than any domain:

- **Grid code** (medial entorhinal cortex) — position is encoded as
  phases on hexagonal lattices at geometrically spaced scales and
  rotated orientations (Moser et al. 2008; Stensola et al. 2012).
  Each module m has scale λ_m and orientation θ_m; a position p has
  lattice-phase coordinates u_m = (B_m p) mod 1 where B_m is the
  reciprocal of the module's two 60°-separated basis vectors. Path
  integration is exact phase advance: u_m += B_m Δ. The phase code is
  the position signal; a grid cell with preferred phase φ reads it
  out as a sum-of-cosines hexagonal firing field.
- **Place cells** (hippocampus) — Gaussian fields over allocentric
  space; each visited location consolidates into a place node that
  re-activates when re-occupied within σ.
- **Boundary-vector cells** (subiculum / medial entorhinal) — a
  population tuned to (distance, direction) pairs, driven by the
  distance to the nearest known barrier along each ray (Barry et al.
  2006). They anchor place structure to the arena's geometry.
- **Transition graph** (Tolman 1948; Whittington et al. 2020, the
  Tolman-Eichenbaum machine) — places are nodes; observed traversals
  are directed edges labelled by the action taken. Structure (the
  graph) is factorized from content (what occupies each place), so
  layouts learned under one arrangement constrain new ones.
- **Successor representation** (Stachenfeld, Botvinick & Gershman
  2017) — the predictive map M = (I − γP)⁻¹ over the transition
  graph, where P is the row-normalized transition matrix under the
  observed action distribution. M[s, ·] is place s's expected
  discounted future occupancy — "from here, where am I headed."
- **Anchors** (retrosplenial / parahippocampal input) — entities
  that reappear at the same cell across episodes are promoted to
  landmarks; the map re-grounds to them rather than to drifted
  path integration.
- **Global remapping** — each distinct environment gets its own
  context graph. The opening observation's spatial signature is the
  retrieval cue: a matching stored context is reinstated (CA3-style
  pattern completion); an unfamiliar signature starts a new map.

Nothing here knows what environment it is mapping. Positions arrive
as coordinates, entities as (color, footprint) observations, actions
as opaque ids — the same substrate serves a maze, a grid world, or a
robot floor plan.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

Cell = tuple[int, int]

# Grid modules: geometric scale progression ratio ~1.6 (biological
# modules step ~1.4–1.7 from dorsal to ventral MEC — Stensola 2012),
# each at its own orientation so codes decorrelate across modules.
_GRID_SCALES = (3.0, 4.8, 7.7, 12.3)
_GRID_ORIENTATIONS = (0.0, 0.35, -0.2, 0.55)

# Place cells: positions within this distance of an existing place
# re-occupy it rather than forming a new field.
_PLACE_SIGMA = 1.5

# Boundary-vector population: preferred (distance, direction) pairs.
_BVC_DISTANCES = (0.5, 1.5, 3.0, 6.0)
_BVC_DIRECTIONS = 8  # rays spaced 45° apart
_BVC_SIGMA = 1.0

# Successor representation discount (Stachenfeld 2017 uses γ≈0.9+).
_SR_GAMMA = 0.9

# An opening observation must overlap a stored context's signature at
# least this much (Jaccard) to reinstate that map.
_CONTEXT_MATCH = 0.6


@dataclass
class MapEvent:
    """One change in the allocentric map worth noticing."""

    kind: str
    text: str
    salience: float = 0.3


@dataclass
class Place:
    """A hippocampal place node — a stable allocentric location."""

    id: int
    pos: tuple[float, float]
    visits: int = 0
    code: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Anchor:
    """A landmark: an entity kind re-observed at the same cell.

    Confirmed once it has appeared in more than one episode — a thing
    that keeps being where it was is a thing you can navigate by.
    """

    color: int
    cell: Cell
    episodes: set[int] = field(default_factory=set)

    @property
    def confirmed(self) -> bool:
        return len(self.episodes) >= 2


class GridCode:
    """Entorhinal position code: hexagonal-lattice phases per module.

    A position is represented by its fractional coordinates on each
    module's lattice — the "grid phase." Translation advances phases
    linearly (path integration), and the joint code across scales is
    locally unique (distinct positions in the arena map to distinct
    phase vectors), which is what lets the system represent position
    without any external coordinate frame.
    """

    def __init__(
        self,
        scales: tuple[float, ...] = _GRID_SCALES,
        orientations: tuple[float, ...] = _GRID_ORIENTATIONS,
    ) -> None:
        self.scales = scales
        # B_m maps position → lattice coordinates for module m. The
        # lattice basis is two vectors of length λ separated by 60°.
        self._B: list[np.ndarray] = []
        for lam, theta in zip(scales, orientations, strict=True):
            b1 = lam * np.array([math.cos(theta), math.sin(theta)])
            b2 = lam * np.array(
                [math.cos(theta + math.pi / 3), math.sin(theta + math.pi / 3)]
            )
            basis = np.stack([b1, b2], axis=1)
            self._B.append(np.linalg.inv(basis))

    def encode(self, pos: tuple[float, float]) -> np.ndarray:
        """Position → concatenated lattice phases, one pair per module."""
        p = np.array(pos, dtype=float)
        return np.concatenate([(B @ p) % 1.0 for B in self._B])

    def advance(
        self, code: np.ndarray, delta: tuple[float, float]
    ) -> np.ndarray:
        """Path integration: advance every module's phase by B·Δ."""
        d = np.array(delta, dtype=float)
        parts = []
        for i, B in enumerate(self._B):
            u = code[2 * i : 2 * i + 2]
            parts.append((u + B @ d) % 1.0)
        return np.concatenate(parts)

    def field(
        self, module: int, pos: tuple[float, float], offset: np.ndarray
    ) -> float:
        """One grid cell's activity: hex field at preferred phase.

        The firing field is a sum of three plane waves at 120° — in
        lattice-phase coordinates that is the three wave components
        du₀, du₁, du₀+du₁ — which produces the hexagonal array of
        firing peaks grid cells exhibit.
        """
        u = (self._B[module] @ np.array(pos, dtype=float)) % 1.0
        du = u - offset
        return float(
            np.mean(
                np.cos(
                    2 * math.pi * np.array([du[0], du[1], du[0] + du[1]])
                )
            )
        )


@dataclass
class ContextMap:
    """One environment's allocentric structure (a "rate map").

    Hippocampal place populations remap globally between environments;
    each context keeps its own places, transitions, anchors, and
    boundary set, and accumulates them across every episode spent in
    that same environment.
    """

    signature: frozenset[tuple[int, int, int, int]]
    bounds: tuple[int, int]
    places: dict[int, Place] = field(default_factory=dict)
    next_place_id: int = 1
    # Directed edges: (from_place, action) → counts over to_place.
    transitions: dict[tuple[int, str], Counter[int]] = field(
        default_factory=dict
    )
    # Object-location binding: what kind of thing occupies each place.
    contents: dict[int, Counter[int]] = field(default_factory=dict)
    anchors: dict[tuple[int, Cell], Anchor] = field(default_factory=dict)
    boundary: set[Cell] = field(default_factory=set)
    n_episodes: int = 1


class CognitiveMap:
    """The allocentric map — hippocampal formation of the organism.

    Consumed per observation: her allocentric position (or a
    dead-reckoning displacement when position is unknown), the
    currently visible entities, the action that produced this
    observation, and the egocentric model's known-blocked cells as
    boundary evidence.
    """

    def __init__(self) -> None:
        self.grid = GridCode()
        self.contexts: list[ContextMap] = []
        self._context: ContextMap | None = None
        self._episode = 0
        self._pos: tuple[float, float] | None = None
        self._code: np.ndarray | None = None
        self._place: Place | None = None

    # ─── Observation ────────────────────────────────────────────

    def observe(
        self,
        position: tuple[float, float] | None,
        entities: list[tuple[int, frozenset[Cell]]],
        action: str | None = None,
        blocked_cells: set[Cell] | None = None,
        bounds: tuple[int, int] | None = None,
        delta: tuple[float, float] | None = None,
    ) -> list[MapEvent]:
        """Integrate one allocentric observation.

        ``position`` is her centroid in world coordinates when the
        egocentric model knows where she is; ``action`` is the action
        that produced this observation (recorded on the transition
        edge); ``entities`` are the visible non-self things as
        (color, cells) pairs. ``bounds`` is the field size.
        ``delta`` is self-reported motion (e.g. velocity) used for
        dead reckoning when no allocentric fix is available.
        """
        events: list[MapEvent] = []
        if self._context is None:
            if bounds is None:
                return events
            self._context = self._resolve_context(entities, bounds)
            events.append(
                MapEvent(
                    "context",
                    "a familiar place" if self._context.n_episodes > 1
                    else "an unfamiliar place",
                    salience=0.5,
                )
            )
        ctx = self._context
        if blocked_cells:
            ctx.boundary |= blocked_cells

        # Anchors: every visible entity is a candidate landmark.
        for color, cells in entities:
            if not cells:
                continue
            r = round(sum(c[0] for c in cells) / len(cells))
            c = round(sum(c[1] for c in cells) / len(cells))
            key = (color, (r, c))
            anchor = ctx.anchors.get(key)
            if anchor is None:
                ctx.anchors[key] = Anchor(
                    color=color, cell=(r, c), episodes={self._episode}
                )
            else:
                anchor.episodes.add(self._episode)

        # Bind visible content to the places it occupies.
        for color, cells in entities:
            if not cells:
                continue
            er = sum(c[0] for c in cells) / len(cells)
            ec = sum(c[1] for c in cells) / len(cells)
            place = self._place_at(ctx, (er, ec), create=True)
            assert place is not None
            ctx.contents.setdefault(place.id, Counter())[color] += 1

        if position is not None:
            prev_place = self._place
            prev_pos = self._pos
            # Sensory observation corrects the integrated estimate —
            # landmark/boundary input dominates path integration.
            self._pos = position
            self._code = self.grid.encode(position)
            place = self._place_at(ctx, position, create=True)
            assert place is not None
            place.visits += 1
            self._place = place
            if (
                prev_place is not None
                and prev_pos is not None
                and place.id != prev_place.id
            ):
                edge = ctx.transitions.setdefault(
                    (prev_place.id, action or ""), Counter()
                )
                edge[place.id] += 1
            elif prev_place is not None and place.id == prev_place.id:
                # Stayed within one place — no edge.
                pass
        elif self._pos is not None and delta is not None:
            # No allocentric fix — dead-reckon from self-reported
            # motion; the estimate is corrected on the next fix.
            self.path_integrate(delta)
        return events

    def path_integrate(self, delta: tuple[float, float]) -> None:
        """Dead reckoning when no allocentric fix is available."""
        if self._pos is None:
            return
        self._pos = (
            self._pos[0] + delta[0],
            self._pos[1] + delta[1],
        )
        if self._code is not None:
            self._code = self.grid.advance(self._code, delta)

    @property
    def position_code(self) -> np.ndarray | None:
        """The grid-cell phase code of the current estimate."""
        return self._code

    def _resolve_context(
        self,
        entities: list[tuple[int, frozenset[Cell]]],
        bounds: tuple[int, int],
    ) -> ContextMap:
        """Pattern completion: reinstate a stored map or start one.

        The retrieval cue is the opening scene's spatial signature —
        the multiset of (color, size, rounded centroid) over entities
        plus the field dimensions. A sufficiently matching stored
        context is re-entered; otherwise a new map forms.
        """
        signature = frozenset(
            (
                color,
                len(cells),
                round(sum(c[0] for c in cells) / len(cells)),
                round(sum(c[1] for c in cells) / len(cells)),
            )
            for color, cells in entities
            if cells
        )
        best: ContextMap | None = None
        best_overlap = 0.0
        for ctx in self.contexts:
            if ctx.bounds != bounds:
                continue
            if signature == ctx.signature:
                overlap = 1.0  # identical opening scene — same place
            else:
                union = len(signature | ctx.signature)
                overlap = (
                    len(signature & ctx.signature) / union
                    if union
                    else 0.0
                )
            if overlap > best_overlap:
                best, best_overlap = ctx, overlap
        if best is not None and best_overlap >= _CONTEXT_MATCH:
            best.n_episodes += 1
            return best
        ctx = ContextMap(signature=signature, bounds=bounds)
        self.contexts.append(ctx)
        return ctx

    def _place_at(
        self,
        ctx: ContextMap,
        pos: tuple[float, float],
        create: bool,
    ) -> Place | None:
        """The place node covering a position — existing or new."""
        best: Place | None = None
        best_d = _PLACE_SIGMA
        for place in ctx.places.values():
            d = math.hypot(place.pos[0] - pos[0], place.pos[1] - pos[1])
            if d <= best_d and (best is None or d < best_d):
                best, best_d = place, d
        if best is not None or not create:
            return best
        place = Place(
            id=ctx.next_place_id,
            pos=pos,
            code=self.grid.encode(pos),
        )
        ctx.places[place.id] = place
        ctx.next_place_id += 1
        return place

    # ─── Queries ────────────────────────────────────────────────

    @property
    def current_place(self) -> Place | None:
        return self._place

    def transition_matrix(
        self, ctx: ContextMap | None = None
    ) -> tuple[list[int], np.ndarray]:
        """P — row-normalized transition probabilities over places.

        Marginalizes the observed action distribution: P(s'|s) =
        Σ_a P(a|s)·T(s,a,s'), with P(a|s) the empirical frequency of
        each action at s.
        """
        ctx = ctx or self._context
        if ctx is None or not ctx.places:
            return [], np.zeros((0, 0))
        ids = sorted(ctx.places)
        index = {pid: i for i, pid in enumerate(ids)}
        P = np.zeros((len(ids), len(ids)))
        action_counts: Counter[int] = Counter()
        for (src, _a), dsts in ctx.transitions.items():
            action_counts[src] += sum(dsts.values())
        for (src, _a), dsts in ctx.transitions.items():
            i = index[src]
            if action_counts[src] == 0:
                continue
            for dst, n in dsts.items():
                P[i, index[dst]] += n / action_counts[src]
        # Renormalize defensively (each row sums to ≤1).
        for i in range(len(ids)):
            s = P[i].sum()
            if s > 0:
                P[i] /= s
        return ids, P

    def successor_map(
        self, gamma: float = _SR_GAMMA
    ) -> tuple[list[int], np.ndarray]:
        """M = (I − γP)⁻¹ — the predictive occupancy map.

        Row s is place s's expected discounted future occupancy: the
        hippocampal "predictive map" of Stachenfeld et al. 2017.
        """
        ids, P = self.transition_matrix()
        if not ids:
            return ids, np.zeros((0, 0))
        n = len(ids)
        M = np.linalg.inv(np.eye(n) - gamma * P)
        return ids, M

    def route(
        self, from_place: int, to_place: int
    ) -> list[tuple[int, str]] | None:
        """Shortest world-centered route between two places.

        BFS over the transition graph — what frontal planning over
        the map consumes. Returns (place, action-taken) steps.
        """
        ctx = self._context
        if ctx is None:
            return None
        queue: deque[int] = deque([from_place])
        prev: dict[int, tuple[int, str]] = {}
        seen = {from_place}
        while queue:
            here = queue.popleft()
            if here == to_place:
                path: list[tuple[int, str]] = []
                node = to_place
                while node != from_place:
                    src, action = prev[node]
                    path.append((node, action))
                    node = src
                return list(reversed(path))
            for (src, action), dsts in ctx.transitions.items():
                if src != here:
                    continue
                for dst in dsts:
                    if dst in seen:
                        continue
                    seen.add(dst)
                    prev[dst] = (here, action)
                    queue.append(dst)
        return None

    def boundary_vector(
        self, pos: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Subicular boundary-vector activity at a position.

        For each preferred (distance, direction) pair: distance to the
        nearest boundary cell along that ray, Gaussian-tuned to the
        preferred distance — the BVC model of Barry et al. 2006.
        """
        ctx = self._context
        p = pos or self._pos
        if ctx is None or p is None:
            return np.zeros(len(_BVC_DISTANCES) * _BVC_DIRECTIONS)
        h, w = ctx.bounds
        out = np.zeros(len(_BVC_DISTANCES) * _BVC_DIRECTIONS)
        for di, d_pref in enumerate(_BVC_DISTANCES):
            for ai in range(_BVC_DIRECTIONS):
                theta = 2 * math.pi * ai / _BVC_DIRECTIONS
                dr, dc = math.sin(theta), math.cos(theta)
                # Walk the ray until a boundary cell or the field edge.
                d = 0.0
                steps = int(max(h, w)) + 1
                for step in range(1, steps):
                    r = round(p[0] + dr * step)
                    c = round(p[1] + dc * step)
                    if not (0 <= r < h and 0 <= c < w):
                        d = float(step)
                        break
                    if (r, c) in ctx.boundary:
                        d = float(step)
                        break
                else:
                    d = float(steps)
                if d == 0.0:
                    d = float(steps)
                out[di * _BVC_DIRECTIONS + ai] = math.exp(
                    -((d - d_pref) ** 2) / (2 * _BVC_SIGMA**2)
                )
        return out

    # ─── Episodes ───────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Episode boundary — end the fix, keep the map.

        Everything learned persists: places, transitions, anchors,
        boundaries. Only the egocentric estimate clears — the next
        observation re-anchors it, and if the opening signature
        matches a stored context the map is reinstated as a whole.
        """
        self._episode += 1
        self._context = None
        self._pos = None
        self._code = None
        self._place = None

    @property
    def context(self) -> ContextMap | None:
        return self._context

    def summary(self) -> str:
        ctx = self._context
        if ctx is None:
            return "no map of this place"
        confirmed = sum(1 for a in ctx.anchors.values() if a.confirmed)
        parts = [
            f"{len(ctx.places)} places",
            f"{sum(len(v) for v in ctx.transitions.values())} routes",
        ]
        if confirmed:
            parts.append(f"{confirmed} landmarks")
        if ctx.n_episodes > 1:
            parts.append(f"visited {ctx.n_episodes} times")
        return "; ".join(parts)
