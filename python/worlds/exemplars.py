"""Exemplar panels — detecting paired demonstrations in a scene.

Some worlds teach by showing: a wall of small framed panels, each a
``before`` beside its ``after``, and a separate panel posing the
problem. The parietal contribution to induction is finding that
structure — the bound relational objects — and handing pairs up to
the prefrontal inductor (``frontal_lobe.reasoning.induction``) as
exemplars. Nothing here knows what the pairs *mean*; it detects the
demonstration format itself:

- a *panel* is a component whose bounding box has a uniform border
  ring (a frame) around a non-uniform interior;
- panels of identical dimensions standing adjacent in one row-band
  are *pairs* — the left is the antecedent, the right the
  consequent (reading order is the only ordering the screen gives);
- unpaired panels of unusual width are *queries* — strips the
  world's demonstrator left unanswered.

The session re-parses every frame like the rest of the world model;
reinduction is the inductor's problem (it re-searches only when new
evidence contradicts the standing rule — the FPC rule-change
signal). When a query parses, ``expected`` holds the predicted
answer panel — the goal description a policy can aim at.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from frontal_lobe.reasoning.competence import ProcedureStep
from frontal_lobe.reasoning.induction import RuleStatus
from parietal_lobe.grid import Grid
from parietal_lobe.strips import orbit, segment_ex, tile_seq, unit_shape

from .boards import _background

if TYPE_CHECKING:
    from frontal_lobe.reasoning.induction import ExampleInductor

logger = logging.getLogger(__name__)

#: Panels worth reading are chunky but not the whole screen — a
#: multi-tile answer strip can run most of the width.
_MIN_PANEL = 3
_MAX_PANEL = 56
#: Two panels pair only if the gap between them is small relative
#: to their size — side-by-side demonstrations, not far scenery.
_MAX_GAP_RATIO = 3.0
#: A pair must be parsed on this many frames before it counts as a
#: demonstration — a demonstration persists; a one-frame alignment
#: is transition noise, not evidence. Without this, animation
#: frames inject junk pairs that poison both the store and the
#: frame-color schema.
_PAIR_COMMIT = 2

#: The format's domain name in the shared competence ledger —
#: paired-demonstration tasks across worlds are one task family.
#: Deliberately world-free: the signature binds which variable
#: names carry evidence, not which game they came from.
_EXEMPLAR_DOMAIN = "world.exemplar"
#: The format's semantic operator vocabulary. Positions in this
#: list are what consolidated procedures bind to — surface button
#: names are per-world bindings and never appear here.
_EX_ACTIONS = (
    "read_demonstrations",
    "induce_rule",
    "move",
    "write",
    "inert",
    "probe",
)
#: Schema-level role vocabulary — the cross-domain bridge.
_EX_ROLES = ("demonstration", "induce", "translate")

#: The format's working vocabulary. Nothing in this module *owns*
#: these names — on first evidence each is claimed into her concept
#: network (adopted when the word already exists there, minted with
#: a perception origin when it doesn't), and everything the session
#: reports — state variables, verbs, procedure steps — is routed
#: through the network's ids. The binding between a percept and its
#: name lives in her mind; this is just the list of roles the format
#: forces her to name. Roles are what she observed: ``object``
#: things on screen, ``verb`` things she does, ``event`` the
#: world's answers.
_EX_LEXICON: dict[str, str] = {
    "demonstration": "object",
    "pair": "object",
    "query": "object",
    "answer": "object",
    "tile": "object",
    "run": "object",
    "marker": "object",
    "slot": "object",
    "rule": "object",
    "write": "verb",
    "move": "verb",
    "inert": "verb",
    "probe": "verb",
    "read_demonstrations": "verb",
    "induce_rule": "verb",
    "effect": "event",
    "error": "event",
    "verdict": "event",
}


@dataclass
class _Panel:
    """A framed box found on the screen."""

    top: int
    left: int
    height: int
    width: int
    border: int  # the frame color
    interior: Grid  # contents inside the frame

    @property
    def row_band(self) -> int:
        return self.top


@dataclass
class ExemplarTarget:
    """A predicted answer for an observed query panel."""

    panel: _Panel
    predicted: Grid


def _is_frame(grid: Grid, top: int, left: int, h: int, w: int) -> int | None:
    """Color of the border ring of a bbox, if it is one solid frame."""
    border = grid.at(top, left)
    for c in range(left, left + w):
        if grid.at(top, c) != border or grid.at(top + h - 1, c) != border:
            return None
    for r in range(top, top + h):
        if grid.at(r, left) != border or grid.at(r, left + w - 1) != border:
            return None
    return border


def _color_components(grid: Grid, color: int) -> list[frozenset[tuple[int, int]]]:
    """8-connected regions of one color — a frame is a same-color ring."""
    seen: set[tuple[int, int]] = set()
    comps: list[frozenset[tuple[int, int]]] = []
    for r, c, v in grid.iter_cells():
        if v != color or (r, c) in seen:
            continue
        stack = [(r, c)]
        seen.add((r, c))
        cells: set[tuple[int, int]] = set()
        while stack:
            cr, cc = stack.pop()
            cells.add((cr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < grid.height
                        and 0 <= nc < grid.width
                        and (nr, nc) not in seen
                        and grid.at(nr, nc) == color
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
        comps.append(frozenset(cells))
    return comps


def _panels(grid: Grid) -> list[_Panel]:
    """Framed boxes: per-color components whose bbox edge is a ring."""
    out: list[_Panel] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for color in grid.colors() - {_background(grid)}:
        for cells in _color_components(grid, color):
            rs = [r for r, _c in cells]
            cs = [c for _r, c in cells]
            top, left = min(rs), min(cs)
            h = max(rs) - top + 1
            w = max(cs) - left + 1
            if not (_MIN_PANEL <= h <= _MAX_PANEL and _MIN_PANEL <= w <= _MAX_PANEL):
                continue
            if (top, left, h, w) in seen_boxes:
                continue
            # The component must fill its own border ring — a sparse
            # blob is not a framed panel.
            ring = {(r, c) for r in (top, top + h - 1) for c in range(left, left + w)} | {
                (r, c) for r in range(top, top + h) for c in (left, left + w - 1)
            }
            if not ring <= cells:
                continue
            border = _is_frame(grid, top, left, h, w)
            if border is None:
                continue
            interior = grid.subgrid(top + 1, left + 1, h - 2, w - 2)
            if len(interior.colors()) <= 1:
                continue  # an empty frame demonstrates nothing
            seen_boxes.add((top, left, h, w))
            out.append(_Panel(top, left, h, w, border, interior))
    return out


def _row_bands(panels: list[_Panel]) -> list[list[_Panel]]:
    """Group panels whose tops align (within a couple of cells)."""
    bands: list[list[_Panel]] = []
    for p in sorted(panels, key=lambda p: (p.top, p.left)):
        if bands and abs(p.top - bands[-1][0].top) <= 2:
            bands[-1].append(p)
        else:
            bands.append([p])
    return bands


def _gap_mark_color(grid: Grid, a: _Panel, b: _Panel, bg: int) -> int | None:
    """The ':' between two panels — a bar of one non-background color
    filling the gap across the band's height. Returns its color."""
    for c in range(a.left + a.width, b.left):
        vals = {grid.at(r, c) for r in range(a.top, a.top + a.height)}
        if len(vals) == 1:
            v = next(iter(vals))
            if v != bg:
                return v
    return None


def _pairs(bands: list[list[_Panel]], grid: Grid) -> list[tuple[_Panel, _Panel]]:
    """Adjacent panels pair left→right: antecedent run → consequent run.

    *Both* sides may be runs — a rule can read several tiles and emit
    several — so the size test is that both interiors tile at the
    scene's shared unit, not that either matches the other's whole
    footprint. When the scene uses ':' markers between panels, only
    marked adjacents are demonstrations (a consequent run sitting
    beside the *next* rule's antecedent is not a pair, and the marker
    is what says so). A real punctuation mark is *consistent* — one
    color used across multiple gaps; a lone uniform column is
    scenery, not a marker, and scenes without consistent marks fall
    back to proximity pairing.
    """
    bg = _background(grid)
    interiors = [p.interior for band in bands for p in band]
    unit = unit_shape(interiors)
    out: list[tuple[_Panel, _Panel]] = []
    for band in bands:
        band = sorted(band, key=lambda p: p.left)
        marks = [_gap_mark_color(grid, band[i], band[i + 1], bg) for i in range(len(band) - 1)]
        votes = Counter(c for c in marks if c is not None)
        marker = votes.most_common(1)[0][0] if votes and votes.most_common(1)[0][1] >= 2 else None
        i = 0
        while i + 1 < len(band):
            a, b = band[i], band[i + 1]
            gap = b.left - (a.left + a.width)
            ok = (
                a.height == b.height
                and 0 <= gap <= max(a.width, 1) * _MAX_GAP_RATIO
                and unit is not None
                and tile_seq(a.interior, unit) is not None
                and tile_seq(b.interior, unit) is not None
                and (marker is None or marks[i] == marker)
            )
            if ok:
                out.append((a, b))
                i += 2
            else:
                i += 1
    return out


class ExemplarSession:
    """Per-frame observer: parses paired panels, feeds the inductor.

    The session is world-side apparatus — the RLPFC inductor it feeds
    is injected by the organism, never grown here. Same honesty rule
    as ``boards``: when nothing demonstrative is on screen it is
    silent and changes nothing.
    """

    def __init__(self, inductor: ExampleInductor) -> None:
        self.inductor = inductor
        self.queries: list[_Panel] = []
        self.answer: _Panel | None = None
        self._paired: set[int] = set()
        self._seen_pairs: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
        # Pair candidates short of _PAIR_COMMIT sightings — evidence
        # accruing toward commitment, not yet demonstrations.
        self._pair_votes: dict[tuple[tuple[tuple[int, ...], ...], ...], int] = {}
        # Frame-color evidence: a query's border matches the
        # antecedents' borders; an answer area's matches the
        # consequents'. The demonstrator's own color scheme tells
        # which singleton is the question. Re-derived from each
        # frame's pairs — a new scene can reuse the last scene's
        # colors in the opposite role, and stale colors must not
        # shadow the live ones.
        self._ante_borders: set[int] = set()
        self._cons_borders: set[int] = set()
        self._tile_shape: tuple[int, int] | None = None
        #: Tile → the shape concept the ventral stream recognized for
        #: it — persistent positive recognitions only (misses stay
        #: ephemeral, retried after new concepts land). Owned here so
        #: recognition survives episode boundaries with the rest of
        #: the learned evidence.
        self.glyph_ids: dict[tuple[tuple[int, ...], ...], str] = {}
        self.writer = _PanelWriter()
        #: The shared competence substrate — injected by whoever owns
        #: the mind's frontal lobe (``None`` standalone). This
        #: session speaks the ledger's protocol for its own domain:
        #: recognize the format, report classified-verb transitions,
        #: consolidate verified episodes.
        self.competence: Any = None
        self._ctx: Any = None
        self._prev_state: dict[str, Any] | None = None
        self._last_action: str | None = None
        #: Per-level episode tally — what the level actually ran,
        #: consolidated into procedure steps at the boundary.
        self._episode: dict[str, Any] = {}
        #: Things worth remembering episodically, drained by whoever
        #: owns the long-term store (the session detects; it does not
        #: own the channel).
        self._milestones: list[tuple[str, float]] = []
        #: The rule object already noted as a milestone.
        self._noted_rule: Any = None
        #: The shared concept network — the session claims its
        #: vocabulary there so its own names are hers. Injected by
        #: the embodiment layer; the session stands alone without it.
        self.network: Any = None
        #: Vocabulary already claimed into ``network``.
        self._claimed: set[str] = set()
        #: Relations already landed — each edge is added once, at
        #: the site where its evidence arrived.
        self._grounded: set[str] = set()
        #: Verb classifications already witnessed — the diff a new
        #: classification lands edges against.
        self._v_known: dict[str, set[str]] = {"write": set(), "move": set(), "inert": set()}

    def observe(self, grid: Grid) -> None:
        panels = _panels(grid)
        pairs = _pairs(_row_bands(panels), grid)
        if len(pairs) >= 2:
            # The border schema is a property of the *format*, not of
            # one pair: demonstrations agree on which border colors
            # mean antecedent and which mean consequent. A lone
            # reversed pair — a mis-walked band, transition debris —
            # contradicts that consistency and is dropped before it
            # can blur the schema (overlapping border sets make the
            # query and answer indistinguishable).
            votes = Counter((a.border, b.border) for a, b in pairs)
            strong = {k for k, n in votes.items() if n >= 2}
            if strong:
                pairs = [p for p in pairs if (p[0].border, p[1].border) in strong]
            self._ante_borders = {a.border for a, _b in pairs}
            self._cons_borders = {b.border for _a, b in pairs}
            self._tile_shape = unit_shape([p.interior for pair in pairs for p in pair])
        self.inductor.begin_frame()
        self._paired = {id(p) for pair in pairs for p in pair}
        for a, b in pairs:
            key = (a.interior.rows(), b.interior.rows())
            if key in self._seen_pairs:
                continue
            seen = self._pair_votes.get(key, 0) + 1
            if seen < _PAIR_COMMIT:
                self._pair_votes[key] = seen
                continue
            del self._pair_votes[key]
            self._seen_pairs.add(key)
            self.inductor.observe(a.interior, b.interior)
            self._ante_borders.add(a.border)
            self._cons_borders.add(b.border)
            if self.inductor.last_replaced is not None:
                # A live demonstration just displaced an older entry.
                # Re-open the seen-gate for every stored pair sharing
                # that antecedent so the displaced entry — if it is
                # still real — can re-earn its slot within the commit
                # window instead of being shadowed permanently.
                ante = self.inductor.last_replaced.rows()
                self._seen_pairs = {k for k in self._seen_pairs if k[0] != ante}
                self._pair_votes = {k: n for k, n in self._pair_votes.items() if k[0] != ante}
                self.inductor.last_replaced = None
        self.queries = [p for p in panels if id(p) not in self._paired]
        self.answer = self._answer_panel(self.queries)
        self.writer.observe(grid, self.answer, self._tile_shape)
        self._ground_evidence()
        self._competence_tick()

    def _answer_panel(self, queries: list[_Panel]) -> _Panel | None:
        """The singleton whose frame matches consequent borders —
        the place the answer is written."""
        cons = [
            p
            for p in queries
            if p.border in self._cons_borders and p.border not in self._ante_borders
        ]
        if not cons:
            return None
        return max(cons, key=lambda p: p.width)

    def export_learning(self) -> dict[str, Any]:
        """The learned half as JSON-safe data — the same rule the
        world model keeps: evidence and skills persist, episode
        state re-derives from the first frames. The pair store is
        the persistent exemplar memory; the rule re-induces on load
        because induction over a handful of pairs is cheap."""
        return {
            "version": 1,
            "inductor": self.inductor.to_dict(),
            "writer": self.writer.export_dict(),
            # The frame-color schema — which border colors mean
            # antecedent versus consequent — is learned scene
            # evidence, not episode state.
            "ante_borders": sorted(self._ante_borders),
            "cons_borders": sorted(self._cons_borders),
            "tile_shape": list(self._tile_shape) if self._tile_shape else None,
            "seen_pairs": [[a, b] for a, b in self._seen_pairs],
            "pair_votes": [[a, n] for a, n in self._pair_votes.items()],
            # Ventral-stream recognitions — the tile→concept bindings
            # are learned evidence, same persistence class as the
            # pair store they belong to.
            "glyph_ids": [
                [[list(row) for row in tile], name] for tile, name in self.glyph_ids.items()
            ],
        }

    def import_learning(self, data: dict[str, Any], *, inductor: ExampleInductor) -> None:
        """Load what ``export_learning`` wrote — writer, the border
        schema, and the ventral glyph bindings. The caller supplies
        the restored inductor (``ExampleInductor.from_dict``); scene
        state (panels, margins, slots) stays empty — it re-derives
        from live frames."""
        self.inductor = inductor
        if len(self.inductor.store):
            # Evidence arrived without a rule — re-induce on the
            # next frame rather than trusting a stale candidate.
            self.inductor.needs_search = True
        w = data.get("writer")
        if w:
            self.writer.import_dict(w)
        self._ante_borders = set(data.get("ante_borders", []))
        self._cons_borders = set(data.get("cons_borders", []))
        ts = data.get("tile_shape")
        self._tile_shape = tuple(ts) if ts else None  # type: ignore[assignment]
        self._seen_pairs = {
            (tuple(tuple(row) for row in a), tuple(tuple(row) for row in b))
            for a, b in data.get("seen_pairs", [])
        }
        self._pair_votes = {
            tuple(tuple(tuple(row) for row in part) for part in a): int(n)  # type: ignore[misc]
            for a, n in data.get("pair_votes", [])
        }
        self.glyph_ids = {
            tuple(tuple(int(v) for v in row) for row in tile): str(name)
            for tile, name in data.get("glyph_ids", [])
        }

    def note_action(self, action: str) -> None:
        """Stash the act so the next frame can be read as its effect."""
        self._last_action = action
        self.writer.note_action(action)

    def act(self, available: list[str]) -> str | None:
        """The hand's next move toward the predicted answer.

        ``available`` is the action whitelist the world offered —
        the writer picks the verb it needs or probes an unclassified
        one. ``None`` when there is no target, no panel, nothing
        unwritten, or nothing it can do — the same silence
        ``boards.plan`` returns when it has no plan.
        """
        target = self.expected()
        if target is None or self.answer is None:
            return None
        # The prediction's walls came from the query parse, not the
        # answer panel's — let the segmenter discover them itself.
        seq = self.inductor.apply_seq(target.panel.interior)
        if seq is not None:
            return self.writer.act(seq, available)
        blocks: tuple[list[Grid], int | None] | None = segment_ex(target.predicted, None)
        if blocks is None and self._tile_shape is not None:
            parsed = tile_seq(target.predicted, self._tile_shape)
            if parsed is not None:
                blocks = (parsed[0], parsed[1])
        if blocks is None:
            return None
        return self.writer.act(blocks[0], available)

    def expected(self) -> ExemplarTarget | None:
        """The induced rule's answer to the widest unpaired panel.

        The widest singleton is the phrase; narrower singletons are
        answer areas and scenery. Returns None when no rule covers
        the evidence or the query can't be translated — the same
        silence the boards session returns when it has no rule.
        """
        if not self.queries:
            return None
        candidates = self.queries
        if self._ante_borders and self._cons_borders:
            framed = [
                p
                for p in candidates
                if p.border in self._ante_borders and p.border not in self._cons_borders
            ]
            if framed:
                candidates = framed
        query = max(candidates, key=lambda p: p.width)
        out = self.inductor.apply(query.interior)
        if out is None:
            return None
        if "answer" not in self._grounded:
            # A rule just answered a query — the relation between
            # them is now observed fact.
            self._grounded.add("answer")
            from temporal_lobe.concepts import RelationType

            self._edge("answer", "query", RelationType.RELATED_TO)
            self._edge("answer", "rule", RelationType.RELATED_TO)
        return ExemplarTarget(panel=query, predicted=out)

    def describe(self) -> str:
        return f"{len(self.queries)} queries | {self.inductor.describe()}"

    # ── Evidence surface for the shared competence layer ──────

    def engaged(self) -> bool:
        """Whether the scene is carrying demonstration evidence —
        the format exists only once pairs are committed."""
        return len(self.inductor.store) >= 2

    def _lex(self, name: str) -> str:
        """The concept id a format word resolves to in her network.

        Adopted when she already knows the word — her node gains the
        operational edges her evidence adds, so the verbal stub is
        grounded rather than duplicated. Minted with a perception
        origin when she doesn't. Either way, what the ledger records
        is a node she can inspect, link, and re-lexicalize — the
        binding between a percept and its name lives in her mind,
        not in this module's strings. Standalone (no network) the
        word itself passes through unchanged.
        """
        if self.network is None or name in self._claimed:
            return name
        try:
            concept = self.network.get_concept(name)
            if concept is None:
                concept = self.network.add_concept(
                    name,
                    origin="perception",
                    properties={
                        "domain": _EXEMPLAR_DOMAIN,
                        "role": _EX_LEXICON.get(name, "object"),
                    },
                )
            canon = getattr(concept, "id", None) or getattr(concept, "name", None)
            self._claimed.add(name)
            return canon or name
        except Exception:
            logger.debug("concept claim failed for %s", name, exc_info=True)
            return name

    def _edge(self, source: str, target: str, relation: Any) -> None:
        """A link between two of her concepts — called only at the
        site where the relation was observed, so every edge is
        evidence-backed."""
        if self.network is None:
            return
        try:
            self.network.add_edge(
                self._lex(source),
                self._lex(target),
                relation,
                origin="perception",
            )
        except Exception:
            logger.debug("concept edge failed", exc_info=True)

    def _ground_evidence(self) -> None:
        """Land the format's concepts and relations in her network.

        The vocabulary is claimed once the format is real (committed
        pairs), each relation at the moment its evidence arrived:
        pairs that committed, slots that parsed, verbs that
        classified. Nothing is asserted ahead of observation."""
        if self.network is None or not self.engaged():
            return
        from temporal_lobe.concepts import RelationType

        for word in _EX_LEXICON:
            self._lex(word)
        if "pairs" not in self._grounded:
            # Pairs committed — the demonstration relation itself
            # exists now, built of tiles.
            self._grounded.add("pairs")
            self._edge("pair", "demonstration", RelationType.IS_A)
            self._edge("pair", "tile", RelationType.PART_OF)
        unit = self._tile_shape
        if "runs" not in self._grounded and unit is not None:
            # A side that tiles multi at the shared unit makes runs
            # real: tiles are what runs are made of.
            multi = any(
                p.antecedent.width > unit[1] or p.consequent.width > unit[1]
                for p in self.inductor.store.pairs
            )
            if multi:
                self._grounded.add("runs")
                self._edge("tile", "run", RelationType.PART_OF)
        w = self.writer
        if w._slots and "slots" not in self._grounded:
            # The answer strip parsed into writable positions.
            self._grounded.add("slots")
            self._edge("slot", "answer", RelationType.PART_OF)
        # New verb classifications — the observed consequence of a
        # press names the verb's object.
        cur = {
            "write": set(w._writes),
            "move": set(w._moves),
            "inert": set(w._inert),
        }
        for cls, actions in cur.items():
            new = actions - self._v_known[cls]
            if new:
                if cls == "write":
                    self._edge("write", "slot", RelationType.RELATED_TO)
                elif cls == "move":
                    self._edge("move", "marker", RelationType.RELATED_TO)
                self._v_known[cls] |= new

    def state(self) -> dict[str, Any]:
        """The format's variables, flattened for the shared
        ``TaskCompetence`` layer — schema features, effect-model
        keys, and goal inference all speak this vocabulary. The
        names are her concept network's, the values the format's
        facts: a new world with the same format presents the same
        variables, and she can look up what each one means."""
        rule = self.inductor.rule
        w = self.writer
        return {
            f"{self._lex('pair')}.count": len(self.inductor.store),
            f"{self._lex('rule')}.kind": rule.kind if rule is not None else "none",
            f"{self._lex('rule')}.verified": self.inductor.status is RuleStatus.VERIFIED,
            f"{self._lex('answer')}.present": self.answer is not None,
            f"{self._lex('slot')}.count": len(w._slots or ()),
            f"{self._lex('marker')}.position": w._bracket if w._bracket is not None else -1,
            f"{self._lex('write')}.count": len(w._writes),
            f"{self._lex('move')}.count": len(w._moves),
            f"{self._lex('effect')}.mismatch": w._last_mismatch,
            f"{self._lex('error')}.count": w.errors,
        }

    def verb_of(self, action: str) -> str | None:
        """The semantic verb a pressed action was classified as —
        the concept in her network that names this verb class, not
        the world's button. None when the press hasn't been
        classified yet: an unclaimed button is a probe, also
        evidence."""
        w = self.writer
        if action in w._writes:
            return self._lex("write")
        if action in w._moves:
            return self._lex("move")
        if action in w._inert:
            return self._lex("inert")
        return None

    def glyph_tiles(self) -> list[Grid]:
        """The exemplar evidence as individual tiles — what the
        ventral stream should learn identities for.

        Only *committed* pair interiors count — teaching the visual
        hierarchy from unverified tiles would mint concepts for
        noise. Multi-tile runs decompose at the shared unit."""
        unit = self._tile_shape or unit_shape(
            [g for pair in self.inductor.store.pairs for g in (pair.antecedent, pair.consequent)]
        )
        seen: set[tuple[tuple[int, ...], ...]] = set()
        out: list[Grid] = []
        for pair in self.inductor.store.pairs:
            for interior in (pair.antecedent, pair.consequent):
                seq = tile_seq(interior, unit) if unit else None
                tiles = seq[0] if seq else [interior]
                for tile in tiles:
                    key = tile.rows()
                    if key not in seen:
                        seen.add(key)
                        out.append(tile)
        return out[:64]

    def glyph_pairs(self) -> list[tuple[Grid, Grid]]:
        """Committed 1-tile demonstrations as (ante, cons) tile pairs —
        the substitution relations a concept network can hold as
        edges. Runs (multi-tile sides) are skipped: a 2→3 rule is not
        a tilewise relation and must not assert one."""
        unit = self._tile_shape
        if unit is None:
            return []
        out: list[tuple[Grid, Grid]] = []
        for pair in self.inductor.store.pairs:
            a_seq = tile_seq(pair.antecedent, unit)
            c_seq = tile_seq(pair.consequent, unit)
            if a_seq and c_seq and len(a_seq[0]) == 1 and len(c_seq[0]) == 1:
                out.append((a_seq[0][0], c_seq[0][0]))
        return out

    def schema_state(self) -> dict[str, Any]:
        """The format's stable identity — what a paired-demonstration
        task *is*, independent of phase: pair count and tile geometry.
        Phase variables (rule found? slots parsed?) are bindings, not
        schema — they ride in ``state`` for transitions instead, so
        pre- and post-induction scenes stay one task family."""
        return {
            f"{self._lex('pair')}.count": len(self.inductor.store),
            f"{self._lex('tile')}.h": self._tile_shape[0] if self._tile_shape else 0,
            f"{self._lex('tile')}.w": self._tile_shape[1] if self._tile_shape else 0,
            f"{self._lex('answer')}.present": self.answer is not None,
        }

    # ── The shared competence protocol ────────────────────────

    def _competence_tick(self) -> None:
        """One frame's evidence to the shared competence ledger.

        Recognizes the format once (the schema persists across
        worlds); reports each pressed button under the verb the
        writer *classified* it as — the format-level name that
        transfers, not the per-world keycap."""
        if self.competence is None or not self.engaged():
            self._prev_state = None
            self._last_action = None
            return
        state = self.state()
        if self._ctx is None:
            self._ctx = self.competence.recognize(
                domain=_EXEMPLAR_DOMAIN,
                state=self.schema_state(),
                actions=_EX_ACTIONS,
                roles=_EX_ROLES,
            )
        if self._last_action is not None and self._prev_state is not None:
            verb = self.verb_of(self._last_action) or "probe"
            try:
                check = self.competence.record_transition(self._ctx, verb, self._prev_state, state)
                self._episode["changes"] = check.observed_changes
                self._episode[verb] = self._episode.get(verb, 0) + 1
            except Exception:
                logger.debug("exemplar transition record failed", exc_info=True)
        self._last_action = None
        self._prev_state = state
        self._note_rule_milestone()

    def _note_rule_milestone(self) -> None:
        """A rule reaching VERIFIED is worth remembering once."""
        rule = self.inductor.rule
        if (
            rule is not None
            and rule is not self._noted_rule
            and self.inductor.status is RuleStatus.VERIFIED
        ):
            self._noted_rule = rule
            from temporal_lobe.concepts import RelationType

            self._edge("rule", "pair", RelationType.RELATED_TO)
            self._milestones.append(
                (
                    f"induced {rule.kind} over {len(self.inductor.store)} demonstration pairs",
                    0.55,
                )
            )

    def _episode_steps(self) -> list[ProcedureStep]:
        """The episode as a procedure: what the level actually ran,
        in the order the format demands — read, induce, write. The
        steps' ``support``/counts are the readable evidence axis a
        foreign domain rebinds against."""
        steps: list[ProcedureStep] = []
        n_pairs = len(self.inductor.store)
        if n_pairs:
            steps.append(
                ProcedureStep(
                    action=self._lex("read_demonstrations"),
                    family="exemplar.parse",
                    parameters={"pairs": n_pairs},
                )
            )
        rule = self.inductor.rule
        if rule is not None:
            steps.append(
                ProcedureStep(
                    action=self._lex("induce_rule"),
                    family="exemplar.induction",
                    parameters={
                        "kind": rule.kind,
                        "rules": n_pairs,
                        "support": self.inductor.support,
                    },
                )
            )
        ep = self._episode
        if ep.get("write") or ep.get("move"):
            steps.append(
                ProcedureStep(
                    action=self._lex("write"),
                    family="exemplar.writer",
                    parameters={
                        "writes": ep.get("write", 0),
                        "moves": ep.get("move", 0),
                        "slots": len(self.writer._slots or ()),
                    },
                )
            )
        return steps

    def end_episode(self, *, success: bool) -> None:
        """Episode boundary — the world's own verdict, externally
        verified. A win consolidates the procedure into the shared
        skill ledger; a loss updates the schema's ledger only."""
        skill = None
        if self.competence is not None and self._ctx is not None:
            try:
                skill = self.competence.record_episode(
                    self._ctx,
                    steps=self._episode_steps(),
                    success=success,
                    verification_score=1.0 if success else 0.0,
                    verification="external",
                    final_state=self.state(),
                    terminal_changes=self._episode.get("changes", ()),
                )
            except Exception:
                logger.debug("exemplar episode record failed", exc_info=True)
        # The world's verdict is a fact whether or not a ledger
        # stands behind it — note it either way.
        from temporal_lobe.concepts import RelationType

        self._edge("verdict", "answer", RelationType.RELATED_TO)
        if success:
            self._milestones.append(
                (
                    f"wrote the answer — {len(self.writer._slots or ())} slots, and it held",
                    0.7,
                )
            )
            if skill is not None:
                self._milestones.append(
                    (
                        f"consolidated a procedure for this format ({skill.skill_id[:8]})",
                        0.55,
                    )
                )
        elif self._episode:
            self._milestones.append(
                (
                    f"the level ended unanswered — {len(self.inductor.store)} pairs known",
                    0.55,
                )
            )
        self._episode = {}

    def drain_milestones(self) -> list[tuple[str, float]]:
        """Drain episodic notes — the parietal session detects; the
        channel owner stores."""
        out, self._milestones = self._milestones, []
        return out


def _translated(old: set[tuple[int, int]], new: set[tuple[int, int]]) -> tuple[int, int] | None:
    """The shift taking `old` to `new`, if `new` is `old` under one
    rigid translation — the signature of a blob that moved, versus
    a thing that merely changed value in place."""
    vacated, appeared = old - new, new - old
    if not vacated or len(vacated) != len(appeared):
        return None
    # The marker may be several blobs that moved together — any
    # (vacated, appeared) pairing is a candidate shift; the true one
    # reconstructs the whole moved set.
    shifts = {(ar - r0, ac - c0) for r0, c0 in vacated for ar, ac in appeared}
    for dr, dc in shifts:
        if {(r + dr, c + dc) for r, c in vacated} == appeared:
            return dr, dc
    return None


def _norm_delta(d: int, n: int) -> int:
    """A signed step distance on a wrapping strip — a wrap jump of
    n-1 is one step the other way."""
    return d - n * round(d / n) if n else d


class _PanelWriter:
    """The hand — learns the answer panel's verbs, then writes.

    The premotor/cerebellar half of the exemplar circuit. Nothing
    about the controls is assumed:

    - **Verb classification is by observed effect.** Each act is
      stashed (``note_action``), the next frame is read as its
      effect: one slot's content changed → a *write* verb; no slot
      changed but the marker moved → a *move* verb; nothing on the
      panel changed → inert here.
    - **The bracket is whatever moves.** The selection marker is
      discovered from move-verb diffs (its color is never assumed);
      once learned, the bracket reads directly off the frame —
      the marker's column span names the selected slot.
    - **The alphabet is learned by writing.** Each slot's variant
      sequence is recorded as it is produced; a target block's
      distance in the cycle is then measured, not guessed.

    This is ideomotor learning — actions represented by their
    effects — the same inversion boards.py's click machinery uses,
    lifted from "click a tile" to "edit a slot in a strip".
    """

    #: How far outside the panel the selection marker may sit.
    MARGIN = 3

    def __init__(self) -> None:
        self.separator: int | None = None
        self._pending: str | None = None
        self._slots: list[Grid] | None = None
        self._slot_shape: tuple[int, int] = (0, 0)
        self._slot_cols: list[int] = []
        self._margin: dict[tuple[int, int], int] = {}
        self._marker_colors: set[int] = set()
        self._bracket: int | None = None
        self._writes: list[str] = []
        self._moves: dict[str, int] = {}  # action -> observed slot delta
        self._inert: set[str] = set()
        # Per-(verb, slot) transition graph: variant -> what the verb
        # produced next. The alphabet is learned as each verb's own
        # map, so a write pick is a measured path, not an assumed
        # forward/back direction.
        self._next: dict[tuple[str, int], dict[Grid, Grid]] = {}
        self._expect: tuple[str, int] | None = None  # (kind, slot)
        # Consequence ledger — every choice gets one: confirmed
        # predictions support a verb's claim, violated ones count
        # against. Same support/violation accounting the frontal
        # hypothesis ledger keeps.
        self.evidence: dict[str, list[int]] = {}
        self.errors = 0  # choices whose observed effect broke the prediction
        self._panel: _Panel | None = None
        self._tile_hint: tuple[int, int] | None = None
        #: Slot-to-target distance at the last decision — the
        #: "how far from done" signal the competence layer reads.
        self._last_mismatch = -1

    # ── Perception ────────────────────────────────────────────

    def note_action(self, action: str) -> None:
        self._pending = action

    def _margin_cells(self, grid: Grid, p: _Panel) -> dict[tuple[int, int], int]:
        bg = _background(grid)
        out: dict[tuple[int, int], int] = {}
        for r in range(p.top - self.MARGIN, p.top + p.height + self.MARGIN):
            for c in range(p.left - self.MARGIN, p.left + p.width + self.MARGIN):
                inside = p.top <= r < p.top + p.height and p.left <= c < p.left + p.width
                if not inside and 0 <= r < grid.height and 0 <= c < grid.width:
                    v = grid.at(r, c)
                    if v != bg:
                        out[(r, c)] = v
        return out

    def _parse(self, grid: Grid, panel: _Panel) -> bool:
        """Re-segment the answer strip; re-read the bracket. Returns
        whether the strip parsed — a false frame (win animation,
        transition noise) carries no honest effect to attribute."""
        parsed = None
        if self._tile_hint is not None:
            # A run tiles at fixed pitch — the pitch test ignores
            # uniform columns *inside* a glyph, which read as fake
            # walls to the column scan and break it.
            seq = tile_seq(panel.interior, self._tile_hint)
            if seq is not None:
                parsed = (seq[0], seq[1])
        if parsed is None:
            parsed = segment_ex(
                panel.interior,
                exemplar_shape=(None if self._slots is None else self._slot_shape),
            )
        if parsed is None and self._slots is not None:
            # The forced shape may be a stale level's geometry — a
            # scene change can resize the slots. Rediscover rather
            # than stay blind behind last level's tile size.
            parsed = segment_ex(panel.interior, exemplar_shape=None)
        ok = False
        if parsed is not None:
            blocks, wall = parsed
            if len(blocks) >= 2 and wall is not None:
                self.separator = wall
                self._slot_shape = blocks[0].shape
                # Column starts are uniform across blocks (equal pitch).
                self._slot_cols = self._slot_starts(panel, wall, len(blocks))
                self._slots = blocks
                ok = True
        self._margin = self._margin_cells(grid, panel)
        self._bootstrap_marker(grid, panel)
        self._read_bracket(panel)
        return ok

    def _bootstrap_marker(self, grid: Grid, panel: _Panel) -> None:
        """Before any move verb is seen, a marker can still be read:
        among margin colors foreign to background, border, and the
        slots' own palette, the *rarest* substantial one is the
        selection mark — a field color dominates the margin, a
        bracket is a small coherent blob. Ambiguity waits for
        move-verb evidence instead."""
        if self._marker_colors or not self._slots:
            return
        own = {panel.border, _background(grid)} | {
            s.at(r, c) for s in self._slots for r in range(s.height) for c in range(s.width)
        }
        foreign = set(self._margin.values()) - own
        if len(foreign) == 1:
            self._marker_colors = foreign
            return
        counts = {c: sum(1 for v in self._margin.values() if v == c) for c in foreign}
        substantial = {c for c, n in counts.items() if n >= 4}
        if substantial:
            low = min(counts[c] for c in substantial)
            rarest = [c for c in substantial if counts[c] == low]
            if len(rarest) == 1:
                self._marker_colors = {rarest[0]}

    def _slot_starts(self, panel: _Panel, wall: int, n: int) -> list[int]:
        """Absolute x-start of each slot — where its block's columns
        sit inside the panel interior."""
        interior = panel.interior
        starts: list[int] = []
        c = 0
        step = self._slot_shape[1] or 1
        while len(starts) < n and c < interior.width:
            col_vals = {interior.at(r, c) for r in range(interior.height)}
            if col_vals != {wall}:
                starts.append(panel.left + 1 + c)
                c += step
            else:
                c += 1
        return starts

    def _read_bracket(self, panel: _Panel) -> None:
        """The marker's column centroid names the selected slot."""
        if not self._marker_colors or not self._slot_cols:
            return
        xs = [c for (_r, c), v in self._margin.items() if v in self._marker_colors]
        if not xs:
            return
        cx = sum(xs) / len(xs)
        w = self._slot_shape[1] if self._slots else 5
        best = min(
            range(len(self._slot_cols)),
            key=lambda i: abs((self._slot_cols[i] + w / 2) - cx),
        )
        self._bracket = best

    # ── Effect attribution ────────────────────────────────────

    def observe(
        self,
        grid: Grid,
        panel: _Panel | None,
        tile_shape: tuple[int, int] | None = None,
    ) -> None:
        self._tile_hint = tile_shape
        if panel is None:
            self._slots = None
            return
        old_slots = list(self._slots or [])
        old_margin = dict(self._margin)
        self._panel = panel
        parsed_ok = self._parse(grid, panel)
        if self._pending is None:
            return
        action, self._pending = self._pending, None
        if not parsed_ok or self._slots is None:
            # The frame didn't segment — a transition burst teaches
            # nothing and breaks no prediction.
            self._expect = None
            return
        new_slots = self._slots
        diffs = [i for i, (o, n) in enumerate(zip(old_slots, new_slots, strict=False)) if o != n]
        if len(diffs) == 1 and len(old_slots) == len(new_slots):
            i = diffs[0]
            if action not in self._writes:
                self._writes.append(action)
            # An act has one panel meaning — write evidence withdraws
            # a stale move claim (levels where verbs change jobs).
            self._moves.pop(action, None)
            self._bracket = i
            self._next.setdefault((action, i), {})[old_slots[i]] = new_slots[i]
        elif not diffs and self._margin != old_margin:
            # A marker *relocates*: the cells that left a color and
            # the cells that joined it are the same blob under one
            # translation. A counter or decoration that changes in
            # place has no such translation and is ignored.
            for c in set(self._margin.values()) | set(old_margin.values()):
                old_pos = {k for k, v in old_margin.items() if v == c}
                new_pos = {k for k, v in self._margin.items() if v == c}
                if _translated(old_pos, new_pos) is not None:
                    self._marker_colors.add(c)
                    old_b = self._bracket_at([cc for _r, cc in old_pos])
                    new_b = self._bracket_at([cc for _r, cc in new_pos])
                    if old_b is not None and new_b is not None and old_b != new_b:
                        n = len(self._slot_cols)
                        self._moves[action] = _norm_delta(new_b - old_b, n)
                        if action in self._writes:
                            self._writes.remove(action)
                        self._bracket = new_b
        elif not diffs and self._margin == old_margin:
            self._inert.add(action)
        self._check_expectation(action, diffs)

    def _check_expectation(self, action: str, diffs: list[int]) -> None:
        """Efference-copy check: the verb's predicted effect versus
        what just happened. A hit supports the verb's claim; a miss
        counts against it and withdraws the claim — the next act
        re-probes or falls back instead of trusting a model the
        world just falsified. Consequences either way."""
        if self._expect is None:
            return
        kind, detail = self._expect
        self._expect = None
        ev = self.evidence.setdefault(action, [0, 0])
        if kind == "write":
            if len(diffs) == 1 and diffs[0] == detail and action in self._writes:
                ev[0] += 1
            else:
                ev[1] += 1
                self.errors += 1
                if action in self._writes:
                    self._writes.remove(action)
        elif action in self._moves:
            ev[0] += 1
        else:
            ev[1] += 1
            self.errors += 1

    def _bracket_at(self, xs: list[int]) -> int | None:
        """The slot index under a marker column span."""
        if not self._slot_cols or not xs:
            return None
        cx = sum(xs) / len(xs)
        w = self._slot_shape[1] or 5
        return min(
            range(len(self._slot_cols)),
            key=lambda i: abs((self._slot_cols[i] + w / 2) - cx),
        )

    # ── The hand ──────────────────────────────────────────────

    def act(self, blocks: list[Grid], available: list[str]) -> str | None:
        slots = self._slots
        if slots is None or len(slots) != len(blocks):
            return None
        # A slot matches its target under tilt — the alphabet renders
        # each rune in its own orientation, and the world's check is
        # tilt-aware, same as the dictionary matching is.
        views = [orbit(t) for t in blocks]
        mismatch = [
            i for i, (s, views_i) in enumerate(zip(slots, views, strict=False)) if s not in views_i
        ]
        self._last_mismatch = len(mismatch)
        if not mismatch:
            return None
        i = mismatch[0]

        # Selected and writable → press the write verb whose measured
        # alphabet path reaches the target soonest.
        if self._bracket == i and self._writes:
            self._expect = ("write", i)
            return self._write_pick(i, blocks[i])
        # Writable but not selected → move. A verb's direction is its
        # measured slot delta, normalized for the strip's wrap.
        if self._bracket is not None and self._moves:
            need = _norm_delta(i - self._bracket, len(slots))
            for a, d in self._moves.items():
                if d and (d > 0) == (need > 0):
                    self._expect = ("move", self._bracket)
                    return a
        # Something needed is still unlearned — probe the action whose
        # panel effect has never been seen (scoped motor babble). An
        # unseen verb beats a known move the wrong way — the wrap
        # always reaches the target eventually, but the alphabet of
        # verbs is small and worth knowing whole.
        for a in available:
            if a not in self._writes and a not in self._moves and a not in self._inert:
                return a
        if self._bracket is not None and self._moves:
            self._expect = ("move", self._bracket)
            return next(iter(self._moves))
        return None

    # ── Persistence — the learned half, not the episode ───────

    def export_dict(self) -> dict[str, Any]:
        """Skills, not scene state: verb meanings, the marker's
        color, the consequence ledger, and the measured alphabet
        transitions. Bracket/slots/margin re-derive from frames."""
        return {
            "writes": list(self._writes),
            "moves": dict(self._moves),
            "inert": sorted(self._inert),
            "marker_colors": sorted(self._marker_colors),
            "separator": self.separator,
            "evidence": {a: list(ev) for a, ev in self.evidence.items()},
            "errors": self.errors,
            "next": [
                [a, slot, f.to_lists(), t.to_lists()]
                for (a, slot), nxt in self._next.items()
                for f, t in nxt.items()
            ],
        }

    def import_dict(self, data: dict[str, Any]) -> None:
        """Load what ``export_dict`` wrote — the learned verb
        vocabulary, verb classifications, marker colors, the
        consequence ledger, and the measured transition maps."""
        self._writes = list(data.get("writes", []))
        self._moves = {a: int(d) for a, d in data.get("moves", {}).items()}
        self._inert = set(data.get("inert", []))
        self._marker_colors = set(data.get("marker_colors", []))
        self.separator = data.get("separator")
        self.evidence = {a: [int(x) for x in ev] for a, ev in data.get("evidence", {}).items()}
        self.errors = int(data.get("errors", 0))
        for a, slot, f, t in data.get("next", []):
            self._next.setdefault((str(a), int(slot)), {})[Grid.from_lists(f)] = Grid.from_lists(t)

    def _write_pick(self, slot: int, target: Grid) -> str:
        """The write verb whose learned transition map reaches the
        target in fewest presses — measured paths, so either a
        forward/back pair or any cycle structure the world uses is
        priced by evidence rather than assumed."""
        slots = self._slots
        if slots is None:
            return self._writes[0]
        cur = slots[slot]
        best: str | None = None
        best_d = 16
        best_ev = -1
        for w in self._writes:
            nxt = self._graph_for(w, slot)
            g, d = cur, 0
            while g != target and g in nxt and d < 16:
                g, d = nxt[g], d + 1
            if g == target and d:
                # Fewest presses wins; a tie goes to the verb whose
                # predictions have been confirmed most — evidence,
                # not recency.
                ev = self.evidence.get(w, [0, 0])[0]
                if d < best_d or (d == best_d and ev > best_ev):
                    best, best_d, best_ev = w, d, ev
        return best or self._writes[0]

    def _graph_for(self, w: str, slot: int) -> dict[Grid, Grid]:
        """The verb's measured transition map for a slot — its own if
        seen, else the union of what the verb did on *other* slots.
        Where slots share one alphabet the union is the same map; a
        contradiction between slots means the alphabets genuinely
        differ and no map is honestly claimed."""
        g = self._next.get((w, slot))
        if g:
            return g
        merged: dict[Grid, Grid] = {}
        for (a, _s), nx in self._next.items():
            if a != w:
                continue
            for f, t in nx.items():
                if f in merged and merged[f] != t:
                    return {}
                merged[f] = t
        return merged
