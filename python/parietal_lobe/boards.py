"""Constraint-board specialist for click-driven play.

Some environments hand the agent a screen of tiled boards: small
uniform blocks that change when clicked, sitting beside patterned
blocks that do not. The patterned blocks are *markers*: each carries
a key color at its center and a ring of constraint cells around it,
and the level ends when the relation every marker asserts between
its key and its neighbours holds everywhere at once.

Nothing here knows those semantics in advance. What the specialist
knows is a general inference:

- screens sometimes show several boards side by side, and when they
  do, the ones that differ only in dressing (a frame, a bracket) are
  the work — the rest are demonstrations of *done*;
- a marker's border value predicts the relation it demands between
  its centre and the neighbour on that side, and that mapping can be
  read straight off the demonstration boards;
- once the mapping is known, a violating tile is fixed by clicking
  it until it satisfies, watching the frame each step — so unknown
  cycle lengths and click propagation need no prior model.

The induced rule is learned game mechanics and persists across
levels the way her learned physics does; the parse is re-done every
frame so propagation, re-layouts, and fresh colour cycles all fall
out of observation rather than assumption. When no consistent rule
can be induced the specialist stays silent and general exploration
keeps the turn.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from parietal_lobe.grid import Grid

_BORDER_OFFSETS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]


@dataclass
class _Element:
    """One clickable cell or marker block found on the screen."""

    cells: frozenset[tuple[int, int]]
    uniform: int | None  # the block's colour when single-coloured
    micro: tuple[tuple[int, ...], ...] | None  # 3x3 pattern when patterned

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return min(rs), min(cs), max(rs), max(cs)

    @property
    def centroid(self) -> tuple[float, float]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    @property
    def key(self) -> int | None:
        """The colour a constraint sees: the block itself when plain,
        its centre cell when patterned."""
        if self.uniform is not None:
            return self.uniform
        if self.micro is not None:
            return self.micro[1][1]
        return None

    @property
    def border_values(self) -> set[int]:
        if self.micro is None:
            return set()
        return {self.micro[o[0] + 1][o[1] + 1] for o in _BORDER_OFFSETS}

    @property
    def is_marker(self) -> bool:
        """A marker displays a reference colour at its centre that its
        constraint ring never repeats. A patterned tile (sparse dots
        over one tile colour) fails this: its centre is the tile colour,
        which always appears on the ring too."""
        return self.micro is not None and self.micro[1][1] not in self.border_values


@dataclass
class _Marker:
    row: int
    col: int
    centre: int
    borders: dict[tuple[int, int], int]


@dataclass
class _Board:
    """A lattice of elements: rows × cols of tiles and markers."""

    positions: dict[tuple[int, int], _Element]
    framed: bool
    markers: list[_Marker] = field(default_factory=list)

    @property
    def marker_positions(self) -> set[tuple[int, int]]:
        return {(m.row, m.col) for m in self.markers}


def _background(grid: Grid) -> int:
    return Counter(v for _r, _c, v in grid.iter_cells()).most_common(1)[0][0]


def _components(
    grid: Grid, background: int
) -> list[frozenset[tuple[int, int]]]:
    """8-connected non-background regions (blocks, frames, bars)."""
    seen: set[tuple[int, int]] = set()
    comps: list[frozenset[tuple[int, int]]] = []
    for r, c, v in grid.iter_cells():
        if v == background or (r, c) in seen:
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
                        and grid.at(nr, nc) != background
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
        comps.append(frozenset(cells))
    return comps


def _micro_grid(
    grid: Grid, cells: frozenset[tuple[int, int]]
) -> tuple[tuple[int, ...], ...] | None:
    """Sample a block as a 3×3 pattern; None when it is uniform."""
    rs = sorted(r for r, _c in cells)
    cs = sorted(c for _, c in cells)
    top, left, bot, right = rs[0], cs[0], rs[-1], cs[-1]
    h, w = bot - top + 1, right - left + 1
    if h < 3 or w < 3:
        return None
    rows: list[tuple[int, ...]] = []
    colours: set[int] = set()
    for i in range(3):
        band_r = range(top + (i * h) // 3, top + ((i + 1) * h) // 3)
        out: list[int] = []
        for j in range(3):
            band_c = range(left + (j * w) // 3, left + ((j + 1) * w) // 3)
            counts = Counter(
                grid.at(r, c)
                for r in band_r
                for c in band_c
                if 0 <= r < grid.height and 0 <= c < grid.width
            )
            colour = counts.most_common(1)[0][0]
            out.append(colour)
            colours.add(colour)
        rows.append(tuple(out))
    if len(colours) <= 1:
        return None
    return tuple(rows)


def _is_element(h: int, w: int) -> bool:
    """Board blocks are chunky; frames and bars are thin or huge."""
    return 3 <= h <= 10 and 3 <= w <= 10 and 0.5 <= h / w <= 2.0


def _to_element(grid: Grid, cells: frozenset[tuple[int, int]]) -> _Element:
    micro = _micro_grid(grid, cells)
    uniform = None
    if micro is None:
        uniform = Counter(grid.at(r, c) for r, c in cells).most_common(1)[0][0]
    return _Element(cells=cells, uniform=uniform, micro=micro)


def _lattice(elements: list[_Element], framed: bool) -> _Board | None:
    """Clustered elements -> board positions on their natural pitch."""
    if len(elements) < 4:
        return None
    rows = sorted({round(e.centroid[0]) for e in elements})
    cols = sorted({round(e.centroid[1]) for e in elements})

    # Merge centres that belong to the same lattice line.
    def lines(vals: list[int]) -> list[int]:
        out: list[int] = []
        for v in vals:
            if not out or v - out[-1] > 4:
                out.append(v)
        return out

    row_lines = lines(rows)
    col_lines = lines(cols)
    positions: dict[tuple[int, int], _Element] = {}
    for e in elements:
        cr, cc = e.centroid
        ri = min(range(len(row_lines)), key=lambda i: abs(row_lines[i] - cr))
        ci = min(range(len(col_lines)), key=lambda i: abs(col_lines[i] - cc))
        positions[(ri, ci)] = e
    board = _Board(positions=positions, framed=framed)
    for (ri, ci), e in positions.items():
        if e.is_marker and e.micro is not None:
            board.markers.append(
                _Marker(
                    row=ri,
                    col=ci,
                    centre=e.micro[1][1],
                    borders={
                        off: e.micro[off[0] + 1][off[1] + 1]
                        for off in _BORDER_OFFSETS
                    },
                )
            )
    return board


def _boards(grid: Grid) -> list[_Board]:
    """Segment the screen into element lattices.

    Blocks touching or nearly touching belong to one board. A region
    wrapped in decoration (a bracket, an outline) is a *framed* board
    — the world's own way of saying *this* one is work — and its own
    separator colour takes over background duty inside the frame, so
    the tiles nested in it still resolve.
    """
    background = _background(grid)
    raw = _components(grid, background)
    elements: list[_Element] = []
    framed_boards: list[_Board] = []
    for comp in raw:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        top, left = min(rs), min(cs)
        h, w = max(rs) - top + 1, max(cs) - left + 1
        if _is_element(h, w) and len(comp) >= 9:
            elements.append(_to_element(grid, comp))
            continue
        # Large structured regions may be frames: re-segment their
        # interior against the frame's own dominant colour.
        if h < 10 or w < 10:
            continue
        separator = Counter(grid.at(r, c) for r, c in comp).most_common(1)[0][0]
        edge_colours = {
            grid.at(r, c)
            for r, c in comp
            if r in (top, top + h - 1) or c in (left, left + w - 1)
        }
        inner: list[_Element] = []
        seen: set[tuple[int, int]] = set()
        for r in range(top, top + h):
            for c in range(left, left + w):
                if (r, c) in seen or grid.at(r, c) in (background, separator):
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
                                top <= nr < top + h
                                and left <= nc < left + w
                                and (nr, nc) not in seen
                                and grid.at(nr, nc) not in (background, separator)
                            ):
                                seen.add((nr, nc))
                                stack.append((nr, nc))
                crs = [cr for cr, _ in cells]
                ccs = [cc for _, cc in cells]
                ch = max(crs) - min(crs) + 1
                cw = max(ccs) - min(ccs) + 1
                if (
                    _is_element(ch, cw)
                    and len(cells) >= 9
                    # Frame fragments (bracket corners, stray edges)
                    # wear the rim's colours — a real board element
                    # always shows at least one other.
                    and not {grid.at(r, c) for r, c in cells} <= edge_colours
                ):
                    inner.append(_to_element(grid, frozenset(cells)))
        board = _lattice(inner, framed=True)
        if board is not None:
            framed_boards.append(board)

    # Union-find clustering by bbox proximity — neighbours sit one
    # small gap apart, boards sit a clear margin apart.
    parent = list(range(len(elements)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(elements):
        at, al, ab, ar = a.bbox
        for j in range(i + 1, len(elements)):
            bt, bl, bb, br = elements[j].bbox
            # Gap between bboxes in each axis; adjacent elements are
            # separated by a couple of background cells at most.
            gap = max(at - bb, bt - ab, al - br, bl - ar, 0)
            if gap <= 4:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    clusters: dict[int, list[_Element]] = {}
    for i, e in enumerate(elements):
        clusters.setdefault(find(i), []).append(e)

    boards = framed_boards
    for cluster in clusters.values():
        board = _lattice(cluster, framed=False)
        if board is not None:
            boards.append(board)
    return boards


def _induce_rule(boards: list[_Board]) -> dict[int, str] | None:
    """Read the border-value → relation mapping off exemplar boards.

    Every marker border that points at a real neighbour contributes
    one observation: (border value, neighbour key equals centre?).
    A mapping is learned only when every value is self-consistent —
    contradicting evidence means the parse is wrong, and silence is
    safer than a confident error.
    """
    tally: dict[int, list[bool]] = {}
    observations = 0
    for board in boards:
        if board.framed:
            continue
        marked = board.marker_positions
        for marker in board.markers:
            for off, value in marker.borders.items():
                pos = (marker.row + off[0], marker.col + off[1])
                if pos in marked:
                    continue
                neighbour = board.positions.get(pos)
                if neighbour is None or neighbour.key is None:
                    continue
                tally.setdefault(value, []).append(neighbour.key == marker.centre)
                observations += 1
    if observations < 3:
        return None
    rule: dict[int, str] = {}
    for value, hits in tally.items():
        if all(hits):
            rule[value] = "equal"
        elif not any(hits):
            rule[value] = "differ"
        else:
            return None
    return rule or None


@dataclass
class _Violation:
    row: int
    col: int
    target: int | None  # colour to reach, or None for "anything but"


def _nonzero_relation(rule: dict[int, str] | None) -> str | None:
    """Generalization over unobserved border values.

    When every observed nonzero border demands the same relation,
    unseen nonzero values most plausibly demand it too (a zero code is
    a distinct symbol, never folded in). Mixed evidence — some nonzero
    values equal, some differ — means no safe generalization.
    """
    if not rule:
        return None
    rels = {rel for value, rel in rule.items() if value != 0}
    return rels.pop() if len(rels) == 1 else None


class ConstraintSession:
    """The board specialist's state across one play session.

    The induced rule, the click-propagation link colour, and the
    colour-transition map are learned mechanics — they survive level
    boundaries the way learned physics does. Everything positional is
    reparsed from each frame, because layouts, palettes, and
    propagation patterns all change.
    """

    def __init__(self) -> None:
        self.rule: dict[int, str] | None = None
        self._nonzero_rel: str | None = None
        self._link_colour: int | None = None
        self._transitions: dict[int, int] = {}
        self._signature: tuple | None = None
        self._satisfied_frames = 0
        self._planned_clicks = 0
        self._tile_clicks: dict[tuple[int, int], int] = {}
        self._disabled = False
        self._boards: list[_Board] = []
        self._last_plan: tuple[tuple[int, int], tuple[int, int], _Board] | None = None
        self._pre_click: tuple[tuple[int, int], _Board] | None = None

    def observe(self, grid: Grid) -> None:
        boards = _boards(grid)
        signature = tuple(
            sorted(
                (b.framed, len(b.positions), len(b.markers))
                for b in boards
            )
        )
        new_layout = signature != self._signature
        if new_layout:
            # A new layout is a new level of work — counts reset, the
            # induced rule and link colour (learned mechanics) carry
            # forward. The colour cycle does not: palettes change per
            # level, so transitions are re-learned per layout.
            self._signature = signature
            self._satisfied_frames = 0
            self._planned_clicks = 0
            self._tile_clicks.clear()
            self._transitions.clear()
            self._disabled = False
        if self.rule is None:
            self.rule = _induce_rule(boards)
            self._nonzero_rel = _nonzero_relation(self.rule)
        self._boards = boards
        if self._pre_click is not None and not new_layout:
            self._learn_from_click()
        self._pre_click = None

    def _query(self) -> _Board | None:
        query = next((b for b in self._boards if b.framed), None)
        if query is None:
            singles = [b for b in self._boards if b.markers]
            query = singles[0] if len(singles) == 1 else None
        return query

    def _learn_from_click(self) -> None:
        """Attribute the frame diff to the click that caused it.

        A click that changes more than its own tile reveals
        propagation: the micro colour whose cells point at the changed
        neighbours is the link. Every colour change observed anywhere
        teaches the shared transition map.
        """
        pos, prev_board = self._pre_click  # type: ignore[misc]
        query = self._query()
        if query is None:
            return
        changed: dict[tuple[int, int], tuple[int, int]] = {}
        for p, element in query.positions.items():
            old = prev_board.positions.get(p)
            if old is None or old.key is None or element.key is None:
                continue
            if element.key != old.key:
                changed[p] = (old.key, element.key)
        if not changed:
            return
        for old, new in changed.values():
            self._transitions[old] = new
        if len(changed) < 2:
            return
        clicked = prev_board.positions.get(pos)
        if clicked is None or clicked.micro is None:
            return
        spread = {
            (p[0] - pos[0], p[1] - pos[1]) for p in changed if p != pos
        }
        if not spread:
            return
        best: tuple[int, int] | None = None
        for colour in clicked.border_values | {clicked.micro[1][1]}:
            dirs = {
                (o[0], o[1])
                for o in _BORDER_OFFSETS
                if clicked.micro[o[0] + 1][o[1] + 1] == colour
            }
            if spread <= dirs and (best is None or len(dirs) < best[1]):
                best = (colour, len(dirs))
        if best is not None:
            self._link_colour = best[0]

    def _relation_for(self, value: int) -> str | None:
        rel = self.rule.get(value) if self.rule else None
        if rel is not None:
            return rel
        if value != 0:
            return self._nonzero_rel
        return None

    def _prop_set(
        self, board: _Board, pos: tuple[int, int]
    ) -> set[tuple[int, int]]:
        """Tiles a click on ``pos`` moves: itself plus the neighbours
        its link-coloured cells point at (once the link is learned)."""
        out = {pos}
        element = board.positions.get(pos)
        if element is None or element.micro is None or self._link_colour is None:
            return out
        marked = board.marker_positions
        for off in _BORDER_OFFSETS:
            if element.micro[off[0] + 1][off[1] + 1] != self._link_colour:
                continue
            q = (pos[0] + off[0], pos[1] + off[1])
            if q in board.positions and q not in marked:
                out.add(q)
        return out

    def plan(self) -> tuple[int, int] | None:
        """The screen cell to click next, or None when it can't help.

        Read-only: nothing is charged until ``note_click`` reports the
        click actually left the agent.
        """
        self._last_plan = None
        if self._disabled or self.rule is None:
            return None
        query = self._query()
        if query is None or not query.markers:
            return None
        if self._planned_clicks > 96:
            self._disabled = True
            return None

        violations = self._violations(query)
        if not violations:
            # The board looks solved; if the level does not advance,
            # the parse or the rule is wrong — stop spending clicks.
            self._satisfied_frames += 1
            if self._satisfied_frames > 6:
                self._disabled = True
            return None
        self._satisfied_frames = 0

        violating = {(v.row, v.col) for v in violations}
        marked = query.marker_positions

        # Exact solve first: once the colour cycle and propagation
        # link are learned, each tile's final colour is its start plus
        # the count of clicks propagating into it, mod the cycle —
        # a linear system over GF(L) that escapes propagation
        # deadlocks a greedy walk cannot. Solve-directed clicks are
        # trusted past the naive per-tile cap (a parity solution can
        # legitimately revisit a tile); the session budget and the
        # stuck-solve detector still bound a wrong model.
        solved = self._solve_clicks(query)
        if solved:
            for pos in sorted(solved):
                if solved[pos] > 0 and self._tile_clicks.get(pos, 0) <= 12:
                    element = query.positions.get(pos)
                    if element is None:
                        continue
                    cr, cc = element.centroid
                    cell = (round(cr), round(cc))
                    self._last_plan = (pos, cell, query)
                    return cell

        # A click helps when its propagation set covers violating
        # tiles — its own tile, or neighbours its link cells point at.
        # Coverage, then fewest side-effects, then least-tried.
        best_score: tuple | None = None
        best_pos: tuple[int, int] | None = None
        for pos in sorted(query.positions):
            if pos in marked or self._tile_clicks.get(pos, 0) > 5:
                continue
            affected = self._prop_set(query, pos)
            hits = len(affected & violating)
            if not hits:
                continue
            score = (-hits, len(affected), self._tile_clicks.get(pos, 0), pos)
            if best_score is None or score < best_score:
                best_score, best_pos = score, pos
        if best_pos is None:
            return None
        element = query.positions.get(best_pos)
        if element is None:
            return None
        cr, cc = element.centroid
        cell = (round(cr), round(cc))
        self._last_plan = (best_pos, cell, query)
        return cell

    def note_click(self, cell: tuple[int, int]) -> None:
        """A planned click was emitted — charge it to the tile and
        remember the pre-click board so the next frame diff can be
        attributed to it."""
        if self._last_plan is None or self._last_plan[1] != cell:
            return
        key, _cell, board = self._last_plan
        self._tile_clicks[key] = self._tile_clicks.get(key, 0) + 1
        self._planned_clicks += 1
        self._pre_click = (key, board)

    def _constraints(
        self, board: _Board
    ) -> dict[tuple[int, int], list[tuple[str, int]]]:
        """Every (relation, key) demand marker borders place on tiles."""
        marked = board.marker_positions
        needs: dict[tuple[int, int], list[tuple[str, int]]] = {}
        for marker in board.markers:
            for off, value in marker.borders.items():
                rel = self._relation_for(value)
                if rel is None:
                    continue
                pos = (marker.row + off[0], marker.col + off[1])
                if pos in marked or pos not in board.positions:
                    continue
                needs.setdefault(pos, []).append((rel, marker.centre))
        return needs

    def _violations(self, board: _Board) -> list[_Violation]:
        """Tiles whose colour breaks a marker constraint, row-major."""
        out: list[_Violation] = []
        for (r, c), constraints in sorted(self._constraints(board).items()):
            current = board.positions[(r, c)].key
            equal_targets = {
                centre
                for rel, centre in constraints
                if rel == "equal" and centre != current
            }
            differ_centres = {
                centre
                for rel, centre in constraints
                if rel == "differ" and centre == current
            }
            if len(equal_targets) == 1:
                out.append(_Violation(r, c, equal_targets.pop()))
            elif len(equal_targets) > 1:
                continue  # contradictory demands — leave it alone
            elif differ_centres:
                out.append(_Violation(r, c, None))
        return out

    def _cycle_order(self) -> list[int] | None:
        """The colour cycle implied by observed transitions, if it
        closes into a clean loop."""
        if len(self._transitions) < 2:
            return None
        start = next(iter(self._transitions))
        order: list[int] = []
        seen: set[int] = set()
        cur: int | None = start
        while cur is not None and cur not in seen:
            seen.add(cur)
            order.append(cur)
            cur = self._transitions.get(cur)
        if cur != start or len(order) < 2:
            return None
        return order

    def _solve_clicks(self, board: _Board) -> dict[tuple[int, int], int] | None:
        """Exact click vector for the current board, or None.

        With the colour cycle known and propagation sets readable via
        the link colour, each constrained tile yields one modular
        equation: start + sum of clicks reaching it ≡ target. The
        system is solved over GF(L) (prime cycle lengths); a tile
        whose demand can't be linearized ('differ' with more than one
        remaining option) stays out of the system for the greedy tail
        to handle.
        """
        order = self._cycle_order()
        if order is None or not _is_prime(len(order)):
            return None
        size = len(order)
        index = {colour: i for i, colour in enumerate(order)}
        marked = board.marker_positions
        variables = [p for p in sorted(board.positions) if p not in marked]
        vindex = {p: i for i, p in enumerate(variables)}

        rows: list[list[int]] = []
        for pos, constraints in self._constraints(board).items():
            element = board.positions[pos]
            start = element.key
            if start not in index:
                continue
            allowed: set[int] | None = None
            for rel, centre in constraints:
                if centre not in index:
                    allowed = set()
                    break
                options = (
                    {centre}
                    if rel == "equal"
                    else {c for c in order if c != centre}
                )
                allowed = options if allowed is None else allowed & options
            if not allowed or len(allowed) > 1:
                continue  # unsatisfiable or not linearizable
            target = allowed.pop()
            delta = (index[target] - index[start]) % size
            if delta == 0:
                continue  # already satisfied
            row = [0] * (len(variables) + 1)
            for var, vi in vindex.items():
                if pos in self._prop_set(board, var):
                    row[vi] = 1 % size
            row[-1] = delta
            rows.append(row)
        if not rows:
            return None

        # Gaussian elimination over GF(size).
        nvars = len(variables)
        pivot_col: dict[int, int] = {}
        r = 0
        for c in range(nvars):
            piv = next(
                (i for i in range(r, len(rows)) if rows[i][c] % size),
                None,
            )
            if piv is None:
                continue
            rows[r], rows[piv] = rows[piv], rows[r]
            inv = pow(rows[r][c], -1, size)
            rows[r] = [(v * inv) % size for v in rows[r]]
            for i in range(len(rows)):
                if i != r and rows[i][c] % size:
                    factor = rows[i][c] % size
                    rows[i] = [
                        (a - factor * b) % size
                        for a, b in zip(rows[i], rows[r], strict=True)
                    ]
            pivot_col[r] = c
            r += 1
            if r == len(rows):
                break
        for row in rows:
            if all(v % size == 0 for v in row[:-1]) and row[-1] % size:
                return None  # inconsistent — model is wrong somewhere
        out = dict.fromkeys(variables, 0)
        for ri, c in pivot_col.items():
            if rows[ri][-1] % size:
                out[variables[c]] = rows[ri][-1] % size
        return out


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True
