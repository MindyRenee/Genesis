"""Assembly puzzles — piece arrangement as a task family.

A jigsaw in miniature: pieces carry four edge codes (N, E, S, W) and
must be placed on a board so every shared edge carries an equal code
and every border-facing edge is flat (code 0). The model is honest —
it reports mismatches but never pre-filters illegal placements, so the
agent has to *discover* that matching edges is the goal rather than
being told.

Generation starts from a solved arrangement, so every generated puzzle
has at least one solution — but ``complete()`` verifies the board's own
constraint satisfaction, not equality with the generator's layout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# Edge codes on generated puzzles run 1.._MAX_CODE; 0 is the flat
# border edge. A small alphabet leaves room for ambiguous matches —
# two internal edges may share a code — which is realistic: matching
# is necessary, not sufficient.
_MAX_CODE = 8


@dataclass(frozen=True)
class Piece:
    """One puzzle piece: four edge codes, rotatable."""

    piece_id: int
    edges: tuple[int, int, int, int]  # N, E, S, W

    def rotated(self, turns: int) -> Piece:
        e = self.edges
        for _ in range(turns % 4):
            # Clockwise: new N = old W, new E = old N, etc.
            e = (e[3], e[0], e[1], e[2])
        return Piece(self.piece_id, e)


@dataclass
class Placement:
    """One candidate or applied move."""

    piece_id: int
    row: int
    col: int
    turns: int
    fits: int  # constraints satisfied (border + occupied neighbors)


class PiecePuzzle:
    """An edge-matching board: place pieces so adjacencies agree."""

    def __init__(self, rows: int, cols: int, pieces: list[Piece]) -> None:
        self.rows = rows
        self.cols = cols
        self.pool: dict[int, Piece] = {p.piece_id: p for p in pieces}
        self.board: dict[tuple[int, int], Piece] = {}

    @classmethod
    def generate(
        cls, rows: int, cols: int, rng: random.Random
    ) -> PiecePuzzle:
        """Build a puzzle with a guaranteed solution.

        Internal edges get random codes shared by both sides; border
        edges are flat (0). Pieces are created in solved orientation and
        shuffled — the solver never sees the generator's assignment.
        """
        h_edges = [
            [rng.randint(1, _MAX_CODE) for _ in range(cols - 1)]
            for _ in range(rows)
        ]
        v_edges = [
            [rng.randint(1, _MAX_CODE) for _ in range(cols)]
            for _ in range(rows - 1)
        ]
        pieces = []
        for r in range(rows):
            for c in range(cols):
                edges = (
                    v_edges[r - 1][c] if r > 0 else 0,
                    h_edges[r][c] if c < cols - 1 else 0,
                    v_edges[r][c] if r < rows - 1 else 0,
                    h_edges[r][c - 1] if c > 0 else 0,
                )
                pieces.append(Piece(r * cols + c, edges))
        rng.shuffle(pieces)
        return cls(rows, cols, pieces)

    # ── World queries ─────────────────────────────────────────

    def mismatches(self) -> int:
        """Constraint violations on the current board.

        Counts each broken shared edge once, plus every nonzero edge
        facing off the board. Zero + full board = solved.
        """
        bad = 0
        for (r, c), piece in self.board.items():
            e = piece.edges
            if r == 0 and e[0] != 0:
                bad += 1
            if c == self.cols - 1 and e[1] != 0:
                bad += 1
            if r == self.rows - 1 and e[2] != 0:
                bad += 1
            if c == 0 and e[3] != 0:
                bad += 1
            east = self.board.get((r, c + 1))
            if east is not None and e[1] != east.edges[3]:
                bad += 1
            south = self.board.get((r + 1, c))
            if south is not None and e[2] != south.edges[0]:
                bad += 1
        return bad

    def complete(self) -> bool:
        return len(self.board) == self.rows * self.cols and (
            self.mismatches() == 0
        )

    def frontier(self) -> list[tuple[int, int]]:
        """Empty slots worth considering: adjacent to a placed piece,
        or every slot when the board is empty."""
        if not self.board:
            return [
                (r, c) for r in range(self.rows) for c in range(self.cols)
            ]
        out = set()
        for r, c in self.board:
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and (nr, nc) not in self.board
                ):
                    out.add((nr, nc))
        return sorted(out) or [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.board
        ]

    def candidate_fits(
        self, piece: Piece, row: int, col: int, turns: int
    ) -> int:
        """How many *existing* constraints this placement satisfies."""
        p = piece.rotated(turns)
        e = p.edges
        fits = 0
        if row == 0:
            fits += e[0] == 0
        elif (row - 1, col) in self.board:
            fits += e[0] == self.board[(row - 1, col)].edges[2]
        if col == self.cols - 1:
            fits += e[1] == 0
        elif (row, col + 1) in self.board:
            fits += e[1] == self.board[(row, col + 1)].edges[3]
        if row == self.rows - 1:
            fits += e[2] == 0
        elif (row + 1, col) in self.board:
            fits += e[2] == self.board[(row + 1, col)].edges[0]
        if col == 0:
            fits += e[3] == 0
        elif (row, col - 1) in self.board:
            fits += e[3] == self.board[(row, col - 1)].edges[1]
        return fits

    def candidates(self) -> list[Placement]:
        """Every legal placement: frontier slots × pool × rotations."""
        out = []
        for r, c in self.frontier():
            for piece in self.pool.values():
                for turns in range(4):
                    out.append(
                        Placement(
                            piece.piece_id,
                            r,
                            c,
                            turns,
                            self.candidate_fits(piece, r, c, turns),
                        )
                    )
        return out

    def violations(
        self,
    ) -> tuple[
        list[tuple[tuple[int, int], tuple[int, int]]],
        list[tuple[int, int]],
    ]:
        """The broken constraints themselves: adjacent slot pairs whose
        shared edges disagree, plus slots violating a border edge.

        Blame lives here, not in per-slot counts — a mismatch is a
        property of the *pair*, so repair needs the pair.
        """
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        unary: list[tuple[int, int]] = []
        for (r, c), piece in self.board.items():
            e = piece.edges
            if (r == 0 and e[0] != 0) or (c == 0 and e[3] != 0):
                unary.append((r, c))
            if (r == self.rows - 1 and e[2] != 0) or (
                c == self.cols - 1 and e[1] != 0
            ):
                unary.append((r, c))
            for dr, dc, mine, theirs in ((0, 1, 1, 3), (1, 0, 2, 0)):
                other = self.board.get((r + dr, c + dc))
                if other is not None and e[mine] != other.edges[theirs]:
                    pairs.append(((r, c), (r + dr, c + dc)))
        return pairs, unary

    # ── Actions ───────────────────────────────────────────────

    def place(
        self, piece_id: int, row: int, col: int, turns: int
    ) -> dict[str, Any] | None:
        """Place a piece. Returns the observed outcome, None if illegal."""
        piece = self.pool.get(piece_id)
        if piece is None or (row, col) in self.board:
            return None
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        before = self.mismatches()
        fits = self.candidate_fits(piece, row, col, turns)
        self.board[(row, col)] = piece.rotated(turns)
        del self.pool[piece_id]
        return {
            "piece_id": piece_id,
            "row": row,
            "col": col,
            "turns": turns,
            "fits": fits,
            "delta_mismatches": self.mismatches() - before,
        }

    def remove(self, row: int, col: int) -> dict[str, Any] | None:
        """Lift a placed piece back into the pool (placements are
        tentative — real jigsaws let you retry)."""
        piece = self.board.get((row, col))
        if piece is None:
            return None
        before = self.mismatches()
        del self.board[(row, col)]
        self.pool[piece.piece_id] = piece
        return {
            "piece_id": piece.piece_id,
            "row": row,
            "col": col,
            "delta_mismatches": self.mismatches() - before,
        }
