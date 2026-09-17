"""Spatial practice — Genesis's gated puzzle curriculum.

This is the puzzle analogue of canvas.py: an *ability* she owns.
The curriculum is a set of ARC-style puzzles ordered so that each
one must be mastered before the next unlocks. Nobody drives her
through it — when her volition engine raises the ``puzzle`` urge,
she takes a single attempt at her current puzzle.

Protocol — no teaching, no correction:

  1. Her current puzzle is the first unmastered one in the
     curriculum. Each has a one-line hint — a nudge, never the
     answer.
  2. An attempt runs her SpatialReasoner on the training pairs and
     scores her guesses against the held-out test output. Mastery
     is the best cell-accuracy achieved, 0.0 → 1.0.
  3. Mastery 1.0 (an exact solve) unlocks the next puzzle.
     Progress persists in ``<data_dir>/spatial_practice.json`` so
     sessions compound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .grid import Grid
from .solver import SpatialReasoner

_PROGRESS_FILE = "spatial_practice.json"


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


# The gated curriculum: ordered; each puzzle must be mastered before
# the next unlocks. The hint is one line — a nudge, not the answer.
CURRICULUM: list[dict] = [
    {
        "name": "gravity_intro",
        "hint": "Make each object fall straight down until it reaches the bottom.",
        "train": [
            (_g([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 1, 0]])),
            (_g([[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 3], [0, 0, 0], [0, 0, 0]]),
                  _g([[0, 0, 0], [0, 0, 0], [0, 0, 3]]))],
    },
    {
        "name": "fill_intro",
        "hint": "Fill the empty space completely enclosed by the ring.",
        "train": [
            (_g([[2, 2, 2], [2, 0, 2], [2, 2, 2]]),
             _g([[2, 2, 2], [2, 4, 2], [2, 2, 2]])),
        ],
        "test": [(_g([[5, 5, 5], [5, 0, 5], [5, 5, 5]]),
                  _g([[5, 5, 5], [5, 4, 5], [5, 5, 5]]))],
    },
    {
        "name": "gravity_variation",
        "hint": "Each object falls to the bottom, keeping its own color.",
        "train": [
            (_g([[0, 0, 0], [0, 6, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 6, 0]])),
        ],
        "test": [(_g([[7, 0], [0, 0], [0, 0]]),
                  _g([[0, 0], [0, 0], [7, 0]]))],
    },
    {
        "name": "fill_margin",
        "hint": "Fill only the hole fully enclosed by the shape; change nothing else.",
        "train": [
            (_g([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 0, 3, 0],
                 [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 8, 3, 0],
                 [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0],
                      [0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0],
                      [0, 0, 0, 0]]))],
    },
    {
        "name": "silhouette_stretch",
        "hint": "Repair the damaged region so the shape becomes whole again.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0], [0, 6, 6, 6, 6, 0],
                 [0, 6, 0, 1, 1, 0], [0, 6, 6, 1, 1, 0],
                 [0, 6, 6, 6, 6, 0], [0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0], [0, 6, 6, 6, 6, 0],
                 [0, 6, 0, 6, 6, 0], [0, 6, 6, 6, 6, 0],
                 [0, 6, 6, 6, 6, 0], [0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 0],
                      [0, 4, 2, 2, 4, 0], [0, 4, 4, 4, 4, 0],
                      [0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 0],
                      [0, 4, 4, 4, 4, 0], [0, 4, 4, 4, 4, 0],
                      [0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "gravity_recall",
        "hint": "Each object falls straight down until it lands.",
        "train": [
            (_g([[0, 0, 0], [0, 0, 0], [0, 0, 8], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 8]])),
        ],
        "test": [(_g([[0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0], [0, 0, 0, 0], [0, 5, 0, 0]]))],
    },
    # Uniform-line lessons — scaffold toward arc:25d8a9c8.
    {
        "name": "uniform_row_intro",
        "hint": "Find the row where every cell is the same color.",
        "train": [
            (_g([[1, 2, 1], [3, 3, 3], [2, 4, 2]]),
             _g([[0, 0, 0], [5, 5, 5], [0, 0, 0]])),
            (_g([[6, 6, 6], [1, 4, 1], [2, 1, 2]]),
             _g([[5, 5, 5], [0, 0, 0], [0, 0, 0]])),
        ],
        "test": [(_g([[7, 1, 7], [2, 2, 8], [9, 9, 9]]),
                  _g([[0, 0, 0], [0, 0, 0], [5, 5, 5]]))],
    },
    {
        "name": "uniform_row_marks",
        "hint": "Mark every row made of a single color.",
        "train": [
            (_g([[4, 4, 4], [1, 5, 1], [8, 8, 8]]),
             _g([[7, 7, 7], [0, 0, 0], [7, 7, 7]])),
        ],
        "test": [(_g([[2, 3, 2], [6, 6, 6], [1, 9, 1]]),
                  _g([[0, 0, 0], [7, 7, 7], [0, 0, 0]]))],
    },
    {
        "name": "uniform_column",
        "hint": "Now do the same for columns.",
        "train": [
            (_g([[2, 1, 2], [2, 3, 2], [2, 4, 2]]),
             _g([[5, 0, 5], [5, 0, 5], [5, 0, 5]])),
        ],
        "test": [(_g([[8, 1, 3], [8, 2, 3], [8, 4, 3]]),
                  _g([[5, 0, 5], [5, 0, 5], [5, 0, 5]]))],
    },
    # Fixed-shift lessons — scaffold toward arc:25ff71a9.
    {
        "name": "shift_down_intro",
        "hint": "Move the whole pattern down one row.",
        "train": [
            (_g([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [2, 0, 0], [0, 0, 0]])),
            (_g([[0, 3, 0], [0, 3, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 3, 0], [0, 3, 0]])),
        ],
        "test": [(_g([[0, 0, 0], [0, 0, 4], [0, 0, 0]]),
                  _g([[0, 0, 0], [0, 0, 0], [0, 0, 4]]))],
    },
    {
        "name": "shift_right",
        "hint": "Move the whole pattern one step to the right.",
        "train": [
            (_g([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0]]),
             _g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0],
                      [0, 0, 2, 0]]),
                  _g([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2],
                      [0, 0, 0, 2]]))],
    },
    {
        "name": "shift_up",
        "hint": "Move the whole pattern up one row.",
        "train": [
            (_g([[0, 0, 0], [0, 5, 0], [0, 5, 0]]),
             _g([[0, 5, 0], [0, 5, 0], [0, 0, 0]])),
            (_g([[0, 0, 0], [0, 0, 0], [0, 7, 7]]),
             _g([[0, 0, 0], [0, 7, 7], [0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0], [0, 0, 6], [0, 0, 0]]),
                  _g([[0, 0, 6], [0, 0, 0], [0, 0, 0]]))],
    },
    # Self-stencil lessons — scaffold toward arc:007bbfb7.
    {
        "name": "self_tile_intro",
        "hint": "Every filled cell holds a copy of the whole pattern.",
        "train": [
            (_g([[1, 0], [1, 1]]),
             _g([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0],
                 [1, 1, 1, 1]])),
            (_g([[0, 2], [2, 2]]),
             _g([[0, 0, 0, 2], [0, 0, 2, 2], [0, 2, 0, 2],
                 [2, 2, 2, 2]])),
        ],
        "test": [(_g([[3, 3], [0, 3]]),
                  _g([[3, 3, 3, 3], [0, 3, 0, 3], [0, 0, 3, 3],
                      [0, 0, 0, 3]]))],
    },
    {
        "name": "self_tile_sparse",
        "hint": "Same stencil idea — the pattern stamps itself where it is filled.",
        "train": [
            (_g([[4, 0], [0, 4]]),
             _g([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0],
                 [0, 0, 0, 4]])),
        ],
        "test": [(_g([[0, 6], [6, 0]]),
                  _g([[0, 0, 0, 6], [0, 0, 6, 0], [0, 6, 0, 0],
                      [6, 0, 0, 0]]))],
    },
    {
        "name": "self_tile_large",
        "hint": "Same stencil idea, on a bigger pattern.",
        "train": [
            (_g([[5, 0, 0], [0, 0, 0], [0, 0, 5]]),
             _g([[5, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 5, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 5, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 5]])),
        ],
        "test": [(_g([[0, 8, 0], [0, 0, 0], [0, 0, 0]]),
                  _g([[0, 0, 0, 0, 8, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]]))],
    },
    # Diagonal-marking lessons — scaffold toward arc:10fcaaa3.
    {
        "name": "diagonal_intro",
        "hint": "Put 8s on the diagonal corners of each mark.",
        "train": [
            (_g([[0, 3, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 3, 0], [8, 0, 8], [0, 0, 0]])),
            (_g([[0, 0, 0], [0, 0, 0], [0, 0, 5]]),
             _g([[0, 0, 0], [0, 8, 0], [0, 0, 5]])),
        ],
        "test": [(_g([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
                  _g([[8, 0, 8], [0, 2, 0], [8, 0, 8]]))],
    },
    {
        "name": "diagonal_color",
        "hint": "The diagonal marks can be any color.",
        "train": [
            (_g([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[1, 0, 0], [0, 4, 0], [0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 7], [0, 0, 0], [0, 0, 0]]),
                  _g([[0, 0, 7], [0, 4, 0], [0, 0, 0]]))],
    },
    {
        "name": "tile_then_mark",
        "hint": "Copy the pattern to fill the grid, then mark the diagonal corners.",
        "train": [
            (_g([[0, 5], [0, 0]]),
             _g([[0, 5, 0, 5], [8, 0, 8, 0], [0, 5, 0, 5],
                 [8, 0, 8, 0]])),
        ],
        "test": [(_g([[6, 0], [0, 0]]),
                  _g([[6, 0, 6, 0], [0, 8, 0, 8], [6, 0, 6, 0],
                      [0, 8, 0, 8]]))],
    },
    # Cyclic-extension lessons — scaffold toward arc:017c7c7b.
    {
        "name": "extend_down_intro",
        "hint": "Repeat the pattern below to make it taller.",
        "train": [
            (_g([[1, 0], [0, 1], [1, 0]]),
             _g([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]])),
            (_g([[2, 0], [0, 2]]),
             _g([[2, 0], [0, 2], [2, 0], [0, 2]])),
        ],
        "test": [(_g([[0, 3], [3, 0]]),
                  _g([[0, 3], [3, 0], [0, 3], [3, 0]]))],
    },
    {
        "name": "recolor_extend",
        "hint": "Recolor, then repeat the pattern below.",
        "train": [
            (_g([[1, 1], [0, 1]]),
             _g([[3, 3], [0, 3], [3, 3], [0, 3]])),
        ],
        "test": [(_g([[0, 1], [1, 1]]),
                  _g([[0, 3], [3, 3], [0, 3], [3, 3]]))],
    },
    # Divider-intersection lessons — scaffold toward arc:0520fde7.
    {
        "name": "intersect_intro",
        "hint": "The 5s split the grid — mark where both halves are filled.",
        "train": [
            (_g([[1, 0, 1, 5, 0, 0, 1],
                 [0, 1, 0, 5, 0, 1, 0],
                 [0, 0, 1, 5, 1, 0, 0]]),
             _g([[0, 0, 2], [0, 2, 0], [0, 0, 0]])),
            (_g([[1, 1, 0, 5, 1, 0, 0],
                 [0, 1, 0, 5, 0, 1, 1],
                 [0, 0, 1, 5, 1, 0, 1]]),
             _g([[2, 0, 0], [0, 2, 0], [0, 0, 2]])),
        ],
        "test": [(_g([[0, 0, 1, 5, 0, 0, 1],
                      [1, 0, 0, 5, 1, 1, 0],
                      [0, 1, 0, 5, 0, 1, 1]]),
                  _g([[0, 0, 2], [2, 0, 0], [0, 2, 0]]))],
    },
    {
        "name": "intersect_rows",
        "hint": "The same idea, stacked vertically.",
        "train": [
            (_g([[1, 0, 0], [0, 0, 1], [5, 5, 5], [1, 0, 0],
                 [0, 1, 0]]),
             _g([[2, 0, 0], [0, 0, 0]])),
        ],
        "test": [(_g([[0, 1, 0], [0, 0, 0], [5, 5, 5], [1, 1, 0],
                      [0, 0, 1]]),
                  _g([[0, 2, 0], [0, 0, 0]]))],
    },
    # Shape-classification lessons — scaffold toward arc:27a28665.
    {
        "name": "classify_intro",
        "hint": "The shape of the pattern decides the color — the ink color doesn't matter.",
        "train": [
            (_g([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), _g([[6]])),
            (_g([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), _g([[2]])),
            (_g([[0, 4, 0], [4, 4, 4], [0, 4, 0]]), _g([[6]])),
        ],
        "test": [(_g([[0, 3, 0], [3, 3, 3], [0, 3, 0]]), _g([[6]]))],
    },
    {
        "name": "classify_more",
        "hint": "Same idea — a third shape has its own label.",
        "train": [
            (_g([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), _g([[6]])),
            (_g([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), _g([[2]])),
            (_g([[1, 1, 1], [0, 1, 0], [0, 1, 0]]), _g([[4]])),
        ],
        "test": [(_g([[7, 7, 7], [0, 7, 0], [0, 7, 0]]), _g([[4]]))],
    },
    # Cell-settling lessons — scaffold toward arc:1e0a9b12.
    {
        "name": "settle_down_intro",
        "hint": "Let the pieces sink to the bottom of each column, keeping their order.",
        "train": [
            (_g([[1, 0, 0], [0, 2, 0], [3, 0, 0]]),
             _g([[0, 0, 0], [1, 0, 0], [3, 2, 0]])),
            (_g([[0, 4, 0], [0, 0, 5], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 4, 5]])),
        ],
        "test": [(_g([[0, 0, 6], [7, 0, 0], [0, 0, 0]]),
                  _g([[0, 0, 0], [0, 0, 0], [7, 0, 6]]))],
    },
    {
        "name": "settle_up",
        "hint": "Now let them float to the top.",
        "train": [
            (_g([[0, 0, 3], [0, 4, 0], [5, 0, 0]]),
             _g([[5, 4, 3], [0, 0, 0], [0, 0, 0]])),
        ],
        "test": [(_g([[0, 2, 0], [0, 0, 0], [0, 0, 8]]),
                  _g([[0, 2, 8], [0, 0, 0], [0, 0, 0]]))],
    },
    {
        "name": "settle_right",
        "hint": "Let them drift to the right edge of each row.",
        "train": [
            (_g([[1, 0, 2], [3, 0, 0], [0, 4, 0]]),
             _g([[0, 1, 2], [0, 0, 3], [0, 0, 4]])),
        ],
        "test": [(_g([[0, 6, 0], [9, 0, 7], [0, 0, 0]]),
                  _g([[0, 0, 6], [0, 9, 7], [0, 0, 0]]))],
    },
    # Bridge lessons — scaffold toward arc:29c11459.
    {
        "name": "bridge_intro",
        "hint": "Draw a line between the two marks — each color reaches the middle.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0, 0, 3],
                 [0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0],
                 [2, 2, 2, 5, 3, 3, 3],
                 [0, 0, 0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [4, 0, 0, 0, 0, 0, 1]]),
             _g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [4, 4, 4, 5, 1, 1, 1]])),
        ],
        "test": [(_g([[6, 0, 0, 0, 0, 0, 8],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
                  _g([[6, 6, 6, 5, 8, 8, 8],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "bridge_multi",
        "hint": "Every row with two marks gets its own bridge.",
        "train": [
            (_g([[1, 0, 0, 0, 0, 0, 9],
                 [0, 0, 0, 0, 0, 0, 0],
                 [7, 0, 0, 0, 0, 0, 3]]),
             _g([[1, 1, 1, 5, 9, 9, 9],
                 [0, 0, 0, 0, 0, 0, 0],
                 [7, 7, 7, 5, 3, 3, 3]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0],
                      [2, 0, 0, 0, 0, 0, 4],
                      [0, 0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0, 0],
                      [2, 2, 2, 5, 4, 4, 4],
                      [0, 0, 0, 0, 0, 0, 0]]))],
    },
    # Bar-ranking lessons — scaffold toward arc:08ed6ac7.
    {
        "name": "rank_intro",
        "hint": "Recolor each bar by its height — tallest becomes 1.",
        "train": [
            (_g([[0, 5, 0, 0],
                 [0, 5, 0, 0],
                 [0, 5, 0, 5],
                 [0, 5, 0, 5]]),
             _g([[0, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 0, 2],
                 [0, 1, 0, 2]])),
        ],
        "test": [(_g([[0, 0, 0, 5],
                      [0, 0, 0, 5],
                      [0, 5, 0, 5],
                      [0, 5, 0, 5]]),
                  _g([[0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 2, 0, 1],
                      [0, 2, 0, 1]]))],
    },
    {
        "name": "rank_three",
        "hint": "Rank all the bars — tallest is 1, next is 2, and so on.",
        "train": [
            (_g([[5, 0, 0, 0, 0],
                 [5, 0, 0, 0, 0],
                 [5, 0, 5, 0, 0],
                 [5, 0, 5, 0, 5],
                 [5, 0, 5, 0, 5]]),
             _g([[1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0],
                 [1, 0, 2, 0, 0],
                 [1, 0, 2, 0, 3],
                 [1, 0, 2, 0, 3]])),
        ],
        "test": [(_g([[0, 0, 5, 0, 0],
                      [0, 0, 5, 0, 0],
                      [0, 5, 5, 0, 5],
                      [0, 5, 5, 0, 5],
                      [0, 5, 5, 0, 5]]),
                  _g([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 2, 1, 0, 3],
                      [0, 2, 1, 0, 3],
                      [0, 2, 1, 0, 3]]))],
    },
    # Cross-rays lessons — scaffold toward arc:23581191.
    {
        "name": "cross_intro",
        "hint": "Each dot draws its row and column in its color; crossings become 2.",
        "train": [
            (_g([[0, 0, 0, 0, 0],
                 [0, 8, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 7, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[0, 8, 0, 7, 0],
                 [8, 8, 8, 2, 8],
                 [0, 8, 0, 7, 0],
                 [7, 2, 7, 7, 7],
                 [0, 8, 0, 7, 0]])),
        ],
        "test": [(_g([[0, 0, 8, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [7, 0, 0, 0, 0]]),
                  _g([[2, 8, 8, 8, 8],
                      [7, 0, 8, 0, 0],
                      [7, 0, 8, 0, 0],
                      [7, 0, 8, 0, 0],
                      [7, 7, 2, 7, 7]]))],
    },
    {
        "name": "cross_colors",
        "hint": "Each dot draws its row and column in its color; crossings become 2.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 6, 0, 0, 0, 0]]),
             _g([[0, 6, 0, 0, 3, 0],
                 [0, 6, 0, 0, 3, 0],
                 [3, 2, 3, 3, 3, 3],
                 [0, 6, 0, 0, 3, 0],
                 [0, 6, 0, 0, 3, 0],
                 [6, 6, 6, 6, 2, 6]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0],
                      [3, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 6, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),
                  _g([[3, 0, 0, 6, 0, 0],
                      [3, 3, 3, 2, 3, 3],
                      [3, 0, 0, 6, 0, 0],
                      [3, 0, 0, 6, 0, 0],
                      [2, 6, 6, 6, 6, 6],
                      [3, 0, 0, 6, 0, 0]]))],
    },
    # Empty-halves lessons — scaffold toward arc:1b2d62fb.
    {
        "name": "empty_intro",
        "hint": "The 1s split the grid; mark where both sides are empty.",
        "train": [
            (_g([[9, 0, 0, 1, 0, 9, 0],
                 [0, 9, 0, 1, 0, 0, 9],
                 [9, 9, 9, 1, 9, 9, 9]]),
             _g([[0, 0, 8],
                 [8, 0, 0],
                 [0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 9, 1, 9, 0, 0],
                      [9, 0, 0, 1, 0, 0, 9],
                      [0, 9, 0, 1, 9, 0, 0]]),
                  _g([[0, 8, 0],
                      [0, 8, 0],
                      [0, 0, 8]]))],
    },
    {
        "name": "empty_rows",
        "hint": "The 1s split the grid; mark where both sides are empty.",
        "train": [
            (_g([[9, 0, 9],
                 [0, 0, 0],
                 [1, 1, 1],
                 [0, 9, 0],
                 [9, 0, 0]]),
             _g([[0, 0, 0],
                 [0, 8, 8]])),
        ],
        "test": [(_g([[0, 0, 9],
                      [9, 0, 0],
                      [1, 1, 1],
                      [9, 0, 0],
                      [0, 9, 9]]),
                  _g([[0, 8, 0],
                      [0, 0, 0]]))],
    },
    # Quadrant-picking lessons — scaffold toward arc:2dc579da.
    {
        "name": "quad_intro",
        "hint": "The cross splits the grid into four; keep the quarter holding the odd cell.",
        "train": [
            (_g([[7, 4, 5, 4, 4],
                 [4, 4, 5, 4, 4],
                 [5, 5, 5, 5, 5],
                 [4, 4, 5, 4, 4],
                 [4, 4, 5, 4, 4]]),
             _g([[7, 4],
                 [4, 4]])),
        ],
        "test": [(_g([[4, 4, 5, 4, 4],
                      [4, 4, 5, 4, 4],
                      [5, 5, 5, 5, 5],
                      [4, 4, 5, 4, 4],
                      [4, 4, 5, 4, 7]]),
                  _g([[4, 4],
                      [4, 7]]))],
    },
    {
        "name": "quad_offset",
        "hint": "The cross splits the grid into four; keep the quarter holding the odd cell.",
        "train": [
            (_g([[6, 6, 6, 6, 6, 2, 6],
                 [6, 8, 6, 6, 6, 2, 6],
                 [6, 6, 6, 6, 6, 2, 6],
                 [2, 2, 2, 2, 2, 2, 2],
                 [6, 6, 6, 6, 6, 2, 6]]),
             _g([[6, 6, 6, 6, 6],
                 [6, 8, 6, 6, 6],
                 [6, 6, 6, 6, 6]])),
        ],
        "test": [(_g([[4, 4, 2, 4, 4],
                      [4, 4, 2, 4, 4],
                      [4, 4, 2, 4, 4],
                      [2, 2, 2, 2, 2],
                      [4, 4, 2, 7, 4],
                      [4, 4, 2, 4, 4]]),
                  _g([[7, 4],
                      [4, 4]]))],
    },
    # Corner-shear lessons — scaffold toward arc:025d127b.
    {
        "name": "shear_intro",
        "hint": "Slide each shape one step right; its bottom row and right edge stay put.",
        "train": [
            (_g([[0, 3, 0, 0, 0],
                 [0, 3, 0, 0, 0],
                 [0, 3, 3, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[0, 0, 3, 0, 0],
                 [0, 0, 3, 0, 0],
                 [0, 3, 3, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 7, 0, 0, 0],
                      [0, 7, 0, 0, 0],
                      [0, 7, 7, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]),
                  _g([[0, 0, 7, 0, 0],
                      [0, 0, 7, 0, 0],
                      [0, 7, 7, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]))],
    },
    {
        "name": "shear_multi",
        "hint": "Slide each shape one step right; its bottom row and right edge stay put.",
        "train": [
            (_g([[5, 0, 0, 0, 0, 0],
                 [0, 5, 0, 0, 0, 0],
                 [0, 5, 0, 0, 0, 0],
                 [0, 0, 0, 8, 8, 0],
                 [0, 0, 0, 8, 0, 0],
                 [0, 0, 0, 8, 0, 0]]),
             _g([[0, 5, 0, 0, 0, 0],
                 [0, 5, 0, 0, 0, 0],
                 [0, 5, 0, 0, 0, 0],
                 [0, 0, 0, 0, 8, 0],
                 [0, 0, 0, 0, 8, 0],
                 [0, 0, 0, 8, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 2, 0, 0],
                      [0, 0, 0, 2, 2, 0],
                      [0, 0, 0, 2, 0, 0],
                      [4, 4, 0, 0, 0, 0],
                      [4, 0, 0, 0, 0, 0],
                      [4, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 2, 0, 0],
                      [0, 4, 0, 0, 0, 0],
                      [0, 4, 0, 0, 0, 0],
                      [4, 0, 0, 0, 0, 0]]))],
    },
    # Dual-frame lessons — scaffold toward arc:1bfc4729.
    {
        "name": "dualframe_intro",
        "hint": "Each dot's row becomes a full bar; each half of the frame takes its dot's color.",
        "train": [
            (_g([[0, 0, 0, 0, 0],
                 [0, 4, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 7, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[4, 4, 4, 4, 4],
                 [4, 4, 4, 4, 4],
                 [4, 0, 0, 0, 4],
                 [7, 7, 7, 7, 7],
                 [7, 7, 7, 7, 7]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0],
                      [0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 6, 0, 0, 0],
                      [0, 0, 0, 0, 0]]),
                  _g([[2, 2, 2, 2, 2],
                      [2, 2, 2, 2, 2],
                      [2, 0, 0, 0, 2],
                      [6, 6, 6, 6, 6],
                      [6, 6, 6, 6, 6]]))],
    },
    # Align-tops lessons — scaffold toward arc:1caeab9d.
    {
        "name": "align_intro",
        "hint": "Slide every shape so its top lines up with the color-1 shape's top.",
        "train": [
            (_g([[0, 5, 0, 0, 0, 0],
                 [0, 5, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 5, 0, 0, 1, 0],
                 [0, 5, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 8, 0, 0, 0],
                      [0, 0, 8, 0, 0, 0]]),
                  _g([[0, 0, 8, 0, 0, 1],
                      [0, 0, 8, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "align_three",
        "hint": "Slide every shape so its top lines up with the color-1 shape's top.",
        "train": [
            (_g([[0, 2, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 4],
                 [0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 2, 0, 0, 1, 0, 4],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [7, 0, 0, 0, 0, 6, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0, 0],
                      [7, 0, 0, 1, 0, 6, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]))],
    },
    # Unwind-tile lessons — scaffold toward arc:2013d3e2.
    {
        "name": "unwind_intro",
        "hint": "The grid is one tile rotated four ways; recover the original tile.",
        "train": [
            (_g([[5, 0, 3, 5],
                 [3, 3, 3, 0],
                 [0, 3, 3, 3],
                 [5, 3, 0, 5]]),
             _g([[5, 0],
                 [3, 3]])),
        ],
        "test": [(_g([[0, 7, 7, 0],
                      [7, 7, 7, 7],
                      [7, 7, 7, 7],
                      [0, 7, 7, 0]]),
                  _g([[0, 7],
                      [7, 7]]))],
    },
    {
        "name": "unwind_repair",
        "hint": "The grid is one tile rotated four ways; recover the tile even if a cell is wrong.",
        "train": [
            (_g([[2, 8, 0, 2],
                 [0, 2, 2, 0],
                 [0, 2, 2, 0],
                 [2, 0, 0, 2]]),
             _g([[2, 0],
                 [0, 2]])),
        ],
        "test": [(_g([[3, 0, 0, 3],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [3, 0, 0, 3]]),
                  _g([[3, 0],
                      [0, 0]]))],
    },
    # Stamp-recolor lessons — scaffold toward arc:321b1fc6.
    {
        "name": "stamp_intro",
        "hint": "The multicolor tile is a stamp — repaint every same-shaped block with it.",
        "train": [
            (_g([[7, 6, 0, 0, 0],
                 [9, 4, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 8, 8, 0],
                 [0, 0, 8, 8, 0]]),
             _g([[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 7, 6, 0],
                 [0, 0, 9, 4, 0]])),
        ],
        "test": [(_g([[0, 0, 8, 8, 0],
                      [0, 0, 8, 8, 0],
                      [0, 0, 0, 0, 0],
                      [0, 3, 5, 0, 0],
                      [0, 2, 1, 0, 0]]),
                  _g([[0, 0, 3, 5, 0],
                      [0, 0, 2, 1, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]))],
    },
    {
        "name": "stamp_shape",
        "hint": "The multicolor tile is a stamp — repaint every same-shaped block with it.",
        "train": [
            (_g([[6, 6, 0, 0, 0, 0, 0, 0],
                 [4, 4, 4, 0, 8, 8, 0, 0],
                 [0, 0, 0, 0, 8, 8, 8, 0],
                 [0, 0, 0, 0, 0, 0, 0, 8],
                 [0, 0, 0, 0, 0, 0, 0, 8]]),
             _g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 6, 6, 0, 0],
                 [0, 0, 0, 0, 4, 4, 4, 0],
                 [0, 0, 0, 0, 0, 0, 0, 8],
                 [0, 0, 0, 0, 0, 0, 0, 8]])),
        ],
        "test": [(_g([[0, 0, 0, 8, 8, 0, 0],
                      [0, 0, 0, 8, 8, 8, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0],
                      [5, 5, 5, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 5, 5, 5, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]))],
    },
    # Link-verdict lessons — scaffold toward arc:239be575.
    {
        "name": "link_intro",
        "hint": "Output 8 if a trail of 8s touches both red blocks, otherwise 0.",
        "train": [
            (_g([[2, 2, 0, 0, 0],
                 [2, 2, 0, 0, 0],
                 [0, 0, 8, 0, 0],
                 [0, 0, 0, 2, 2],
                 [0, 0, 0, 2, 2]]),
             _g([[8]])),
            (_g([[2, 2, 0, 0, 0],
                 [2, 2, 0, 0, 0],
                 [8, 0, 0, 0, 0],
                 [0, 0, 0, 2, 2],
                 [0, 0, 0, 2, 2]]),
             _g([[0]])),
        ],
        "test": [(_g([[0, 0, 0, 2, 2],
                      [0, 8, 0, 2, 2],
                      [0, 0, 8, 0, 0],
                      [2, 2, 8, 0, 0],
                      [2, 2, 0, 0, 0]]),
                  _g([[8]]))],
    },
    {
        "name": "link_chain",
        "hint": "Output 8 if a trail of 8s touches both red blocks, otherwise 0.",
        "train": [
            (_g([[2, 2, 0, 0, 0, 0, 0],
                 [2, 2, 0, 0, 0, 0, 0],
                 [0, 0, 8, 0, 0, 0, 0],
                 [0, 0, 0, 8, 0, 0, 0],
                 [0, 0, 0, 0, 8, 0, 0],
                 [0, 0, 0, 0, 0, 2, 2],
                 [0, 0, 0, 0, 0, 2, 2]]),
             _g([[8]])),
        ],
        "test": [(_g([[2, 2, 0, 0, 0, 0, 0],
                      [2, 2, 0, 0, 0, 0, 0],
                      [0, 0, 8, 0, 0, 0, 0],
                      [0, 0, 0, 0, 8, 0, 0],
                      [0, 0, 0, 0, 8, 0, 0],
                      [0, 0, 0, 0, 0, 2, 2],
                      [0, 0, 0, 0, 0, 2, 2]]),
                  _g([[0]]))],
    },
    # Halo lessons — scaffold toward arc:0ca9ddb6.
    {
        "name": "halo_intro",
        "hint": "Each colored cell sprouts a fixed pattern of neighbors around it.",
        "train": [
            (_g([[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 2, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0],
                 [0, 4, 0, 4, 0],
                 [0, 0, 2, 0, 0],
                 [0, 4, 0, 4, 0],
                 [0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0],
                 [0, 0, 7, 0, 0],
                 [0, 7, 1, 7, 0],
                 [0, 0, 7, 0, 0],
                 [0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0],
                      [0, 2, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0]]),
                  _g([[4, 0, 4, 0, 0],
                      [0, 2, 0, 0, 0],
                      [4, 0, 4, 7, 0],
                      [0, 0, 7, 1, 7],
                      [0, 0, 0, 7, 0]]))],
    },
    {
        "name": "halo_mixed",
        "hint": "Each colored cell sprouts a fixed pattern of neighbors around it.",
        "train": [
            (_g([[3, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0]]),
             _g([[3, 0, 0, 0, 0, 0],
                 [0, 4, 0, 4, 0, 0],
                 [0, 0, 2, 0, 0, 0],
                 [0, 4, 0, 4, 7, 0],
                 [0, 0, 0, 7, 1, 7],
                 [0, 0, 0, 0, 7, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [8, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 4, 0, 4],
                      [0, 0, 0, 0, 2, 0],
                      [0, 7, 0, 4, 0, 4],
                      [7, 1, 7, 0, 0, 0],
                      [0, 7, 0, 0, 0, 0],
                      [8, 0, 0, 0, 0, 0]]))],
    },
    # Block-counting lessons — scaffold toward arc:1fad071e.
    {
        "name": "count_intro",
        "hint": "Count the solid 2-by-2 blocks of one color; answer with that many cells.",
        "train": [
            (_g([[1, 1, 0, 2, 2],
                 [1, 1, 0, 2, 2],
                 [0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0],
                 [0, 0, 1, 1, 0]]),
             _g([[1, 1, 0]])),
            (_g([[0, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 0]]),
             _g([[1, 0, 0]])),
        ],
        "test": [(_g([[1, 1, 0, 0, 1, 1],
                      [1, 1, 0, 0, 1, 1],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0]]),
                  _g([[1, 1, 1]]))],
    },
    {
        "name": "count_distractors",
        "hint": "Count the solid 2-by-2 blocks of one color; answer with that many cells.",
        "train": [
            (_g([[2, 2, 0, 0, 0],
                 [2, 2, 0, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 1, 1, 0, 0]]),
             _g([[1, 0, 0]])),
        ],
        "test": [(_g([[1, 1, 0, 2, 2],
                      [1, 1, 0, 2, 2],
                      [0, 0, 0, 0, 0],
                      [2, 2, 0, 1, 1],
                      [2, 2, 0, 1, 1]]),
                  _g([[1, 1, 0]]))],
    },
    # Diff-halves lessons — scaffold toward arc:3428a4f5.
    {
        "name": "diff_intro",
        "hint": "The two halves compare cell by cell — mark where they differ.",
        "train": [
            (_g([[1, 0, 2],
                 [4, 4, 4],
                 [0, 0, 2]]),
             _g([[3, 0, 0]])),
            (_g([[2, 4, 0],
                 [2, 4, 2],
                 [0, 4, 0]]),
             _g([[3],
                 [0],
                 [0]])),
        ],
        "test": [(_g([[2, 0],
                      [4, 4],
                      [0, 2]]),
                  _g([[3, 3]]))],
    },
    {
        "name": "diff_rows",
        "hint": "The two halves compare cell by cell — mark where they differ.",
        "train": [
            (_g([[2, 0, 2],
                 [0, 2, 0],
                 [4, 4, 4],
                 [0, 0, 2],
                 [2, 2, 0]]),
             _g([[3, 0, 0],
                 [3, 0, 0]])),
            (_g([[2, 0],
                 [0, 2],
                 [4, 4],
                 [2, 0],
                 [0, 2]]),
             _g([[0, 0],
                 [0, 0]])),
        ],
        "test": [(_g([[2, 2, 0],
                      [0, 0, 2],
                      [4, 4, 4],
                      [0, 2, 2],
                      [2, 0, 0]]),
                  _g([[3, 0, 3],
                      [3, 0, 3]]))],
    },
    # Line-casting lessons — scaffold toward arc:178fcbfb.
    {
        "name": "lines_intro",
        "hint": "Each colored cell draws a straight line through its whole row or column.",
        "train": [
            (_g([[0, 0, 0, 0, 0],
                 [0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[0, 2, 0, 0, 0],
                 [0, 2, 0, 0, 0],
                 [0, 2, 0, 0, 0],
                 [1, 1, 1, 1, 1],
                 [0, 2, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 3],
                 [0, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[3, 3, 3, 3, 3],
                 [2, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 3, 0, 0, 0]]),
                  _g([[0, 0, 0, 2, 0],
                      [0, 0, 0, 2, 0],
                      [1, 1, 1, 1, 1],
                      [0, 0, 0, 2, 0],
                      [3, 3, 3, 3, 3]]))],
    },
    {
        "name": "lines_cross",
        "hint": "Each colored cell draws a straight line through its whole row or column.",
        "train": [
            (_g([[0, 0, 2, 0],
                 [0, 0, 0, 0],
                 [0, 3, 0, 0],
                 [0, 0, 0, 0]]),
             _g([[0, 0, 2, 0],
                 [0, 0, 2, 0],
                 [3, 3, 3, 3],
                 [0, 0, 2, 0]])),
        ],
        "test": [(_g([[0, 2, 0, 0],
                      [0, 0, 0, 0],
                      [3, 0, 0, 0],
                      [0, 0, 0, 2],
                      [0, 0, 0, 0]]),
                  _g([[0, 2, 0, 2],
                      [0, 2, 0, 2],
                      [3, 3, 3, 3],
                      [0, 2, 0, 2],
                      [0, 2, 0, 2]]))],
    },
    # Symmetry-completion lessons — scaffold toward arc:11852cab.
    {
        "name": "sym_intro",
        "hint": "Complete the pattern so it looks the same when rotated.",
        "train": [
            (_g([[0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0],
                 [0, 3, 0, 3, 0],
                 [0, 0, 3, 0, 0],
                 [0, 0, 0, 0, 0]])),
            (_g([[0, 5, 0],
                 [5, 0, 0],
                 [0, 0, 0]]),
             _g([[5, 5, 0],
                 [5, 5, 0],
                 [0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0],
                      [0, 8, 0, 0],
                      [0, 0, 0, 0],
                      [0, 8, 0, 0]]),
                  _g([[0, 0, 0, 0],
                      [0, 8, 0, 0],
                      [8, 0, 8, 0],
                      [0, 8, 0, 0]]))],
    },
    {
        "name": "sym_pattern",
        "hint": "Complete the pattern so it looks the same when rotated.",
        "train": [
            (_g([[3, 0, 1, 0, 3],
                 [0, 0, 8, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [3, 0, 0, 0, 3]]),
             _g([[3, 0, 1, 0, 3],
                 [0, 0, 8, 0, 0],
                 [1, 8, 0, 8, 1],
                 [0, 0, 8, 0, 0],
                 [3, 0, 1, 0, 3]])),
        ],
        "test": [(_g([[2, 0, 0, 0, 2],
                      [0, 0, 5, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 2]]),
                  _g([[2, 0, 0, 0, 2],
                      [0, 0, 5, 0, 0],
                      [0, 5, 0, 5, 0],
                      [0, 0, 5, 0, 0],
                      [2, 0, 0, 0, 2]]))],
    },
    # Edge-ray lessons — scaffold toward arc:1f642eb9.
    {
        "name": "ray_intro",
        "hint": "Each lone cell shines its color onto the edge of the big block it faces.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 5, 0, 0, 0, 0],
                 [0, 0, 8, 8, 0, 0, 0],
                 [0, 0, 8, 8, 0, 0, 6],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 5, 0, 0, 0, 0],
                 [0, 0, 5, 8, 0, 0, 0],
                 [0, 0, 8, 6, 0, 0, 6],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0],
                 [0, 8, 8, 0, 0],
                 [4, 8, 8, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 9, 0, 0]]),
             _g([[0, 0, 0, 0, 0],
                 [0, 8, 8, 0, 0],
                 [4, 4, 9, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 9, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 2, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 8, 8, 8, 0, 6],
                      [7, 0, 8, 8, 8, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0]]),
                  _g([[0, 0, 0, 2, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 8, 2, 6, 0, 6],
                      [7, 0, 7, 8, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0]]))],
    },
    {
        "name": "ray_multi",
        "hint": "Each lone cell shines its color onto the edge of the big block it faces.",
        "train": [
            (_g([[0, 5, 0, 9, 0, 0],
                 [0, 8, 8, 8, 0, 0],
                 [0, 8, 8, 8, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]),
             _g([[0, 5, 0, 9, 0, 0],
                 [0, 5, 8, 9, 0, 0],
                 [0, 8, 8, 8, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0],
                      [0, 8, 8, 8, 0, 0],
                      [0, 8, 8, 8, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 4, 0, 6, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0],
                      [0, 8, 8, 8, 0, 0],
                      [0, 4, 8, 6, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 4, 0, 6, 0, 0]]))],
    },
    # Nearest-border lessons — scaffold toward arc:2204b7a8.
    {
        "name": "border_intro",
        "hint": "Each mark takes the color of the border line closest to it.",
        "train": [
            (_g([[1, 0, 0, 0, 0, 2],
                 [1, 0, 3, 0, 0, 2],
                 [1, 0, 0, 0, 3, 2],
                 [1, 0, 0, 0, 0, 2],
                 [1, 0, 0, 0, 0, 2]]),
             _g([[1, 0, 0, 0, 0, 2],
                 [1, 0, 1, 0, 0, 2],
                 [1, 0, 0, 0, 2, 2],
                 [1, 0, 0, 0, 0, 2],
                 [1, 0, 0, 0, 0, 2]])),
            (_g([[4, 4, 4, 4, 4],
                 [0, 0, 3, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [7, 7, 7, 7, 7]]),
             _g([[4, 4, 4, 4, 4],
                 [0, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 7, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [7, 7, 7, 7, 7]])),
        ],
        "test": [(_g([[9, 0, 0, 0, 6],
                      [9, 0, 3, 0, 6],
                      [9, 3, 0, 0, 6],
                      [9, 0, 0, 3, 6],
                      [9, 0, 0, 0, 6]]),
                  _g([[9, 0, 0, 0, 6],
                      [9, 0, 9, 0, 6],
                      [9, 9, 0, 0, 6],
                      [9, 0, 0, 6, 6],
                      [9, 0, 0, 0, 6]]))],
    },
    {
        "name": "border_rows",
        "hint": "Each mark takes the color of the border line closest to it.",
        "train": [
            (_g([[5, 5, 5, 5, 5, 5],
                 [0, 0, 0, 3, 0, 0],
                 [0, 3, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0],
                 [0, 0, 0, 0, 0, 0],
                 [8, 8, 8, 8, 8, 8]]),
             _g([[5, 5, 5, 5, 5, 5],
                 [0, 0, 0, 5, 0, 0],
                 [0, 5, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 8, 0],
                 [0, 0, 0, 0, 0, 0],
                 [8, 8, 8, 8, 8, 8]])),
        ],
        "test": [(_g([[5, 5, 5, 5, 5, 5],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 3, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [8, 8, 8, 8, 8, 8]]),
                  _g([[5, 5, 5, 5, 5, 5],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [8, 8, 8, 8, 8, 8]]))],
    },
    # Ghost-pair lessons — scaffold toward arc:22233c11. Touching
    # diagonal-aligned twins echo copies at the diamond lattice points.
    {
        "name": "ghost_domino",
        "hint": "Each pair of touching identical marks echoes a "
        "copy of itself further along the pattern.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 8, 0, 0],
                 [0, 0, 3, 0, 0, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0],
                 [0, 8, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 8, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0, 8, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 3, 0, 0, 0, 0],
                      [0, 0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 8, 0, 0],
                      [0, 0, 3, 0, 0, 0, 0],
                      [0, 0, 0, 3, 0, 0, 0],
                      [0, 8, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "ghost_blocks",
        "hint": "Each pair of touching identical marks echoes a "
        "copy of itself further along the pattern.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3, 0, 0, 0],
                 [0, 0, 3, 3, 0, 0, 0, 0, 0],
                 [0, 0, 3, 3, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[8, 8, 0, 0, 0, 0, 0, 0, 0],
                 [8, 8, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3, 0, 0, 0],
                 [0, 0, 0, 0, 3, 3, 0, 0, 0],
                 [0, 0, 3, 3, 0, 0, 0, 0, 0],
                 [0, 0, 3, 3, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 8, 8, 0],
                 [0, 0, 0, 0, 0, 0, 8, 8, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 5, 5, 0, 0, 0],
                 [0, 0, 0, 0, 5, 5, 0, 0, 0],
                 [0, 0, 5, 5, 0, 0, 0, 0, 0],
                 [0, 0, 5, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [8, 8, 0, 0, 0, 0, 0, 0, 0],
                 [8, 8, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 5, 5, 0, 0, 0],
                 [0, 0, 0, 0, 5, 5, 0, 0, 0],
                 [0, 0, 5, 5, 0, 0, 0, 0, 0],
                 [0, 0, 5, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 8, 8, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 6, 6, 0, 0, 0, 0],
                      [0, 0, 0, 6, 6, 0, 0, 0, 0],
                      [0, 6, 6, 0, 0, 0, 0, 0, 0],
                      [0, 6, 6, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                  _g([[8, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 6, 6, 0, 0, 0, 0],
                      [0, 0, 0, 6, 6, 0, 0, 0, 0],
                      [0, 6, 6, 0, 0, 0, 0, 0, 0],
                      [0, 6, 6, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 8, 8, 0, 0],
                      [0, 0, 0, 0, 0, 8, 8, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]]))],
    },
    # Axes-stamp lessons — scaffold toward arc:2281f1f4. A template
    # line's pattern is painted onto every line the selector marks.
    {
        "name": "stamp_rows",
        "hint": "One line holds a pattern and another marks which "
        "lines receive it; paint the pattern on every marked line.",
        "train": [
            (_g([[0, 5, 0, 0, 5, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 5],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 5],
                 [0, 0, 0, 0, 0, 0]]),
             _g([[0, 5, 0, 0, 5, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 6, 0, 0, 6, 5],
                 [0, 0, 0, 0, 0, 0],
                 [0, 6, 0, 0, 6, 5],
                 [0, 0, 0, 0, 0, 0]])),
            (_g([[4, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 4, 0, 0, 4, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [4, 0, 0, 0, 0, 0]]),
             _g([[4, 6, 0, 0, 6, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 4, 0, 0, 4, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [4, 6, 0, 0, 6, 0]])),
        ],
        "test": [(_g([[0, 0, 3, 0, 0, 3],
                      [0, 0, 0, 0, 0, 0],
                      [0, 3, 0, 0, 0, 0],
                      [0, 3, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 3, 0, 0, 3],
                      [0, 0, 0, 0, 0, 0],
                      [0, 3, 6, 0, 0, 6],
                      [0, 3, 6, 0, 0, 6],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "stamp_cols",
        "hint": "One line holds a pattern and another marks which "
        "lines receive it; paint the pattern on every marked line.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0],
                 [5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [5, 0, 0, 0, 0, 0],
                 [0, 0, 5, 5, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0],
                 [5, 0, 6, 6, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [5, 0, 6, 6, 0, 0],
                 [0, 0, 5, 5, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0],
                      [2, 0, 0, 0, 0, 2],
                      [0, 0, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),
                  _g([[6, 0, 0, 0, 2, 6],
                      [0, 0, 0, 0, 0, 0],
                      [2, 0, 0, 0, 0, 2],
                      [6, 0, 0, 0, 2, 6],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]))],
    },
    # Singleton-ring lessons — scaffold toward arc:31aa019c. The
    # one-of-a-kind color marks where a ring goes; all else erases.
    {
        "name": "ring_intro",
        "hint": "The one-of-a-kind color marks a center; erase "
        "everything else and surround it with a ring.",
        "train": [
            (_g([[0, 3, 0, 0, 0, 0, 5],
                 [0, 0, 0, 0, 0, 3, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 4, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [3, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 5]]),
             _g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 2, 2, 0, 0],
                 [0, 0, 2, 4, 2, 0, 0],
                 [0, 0, 2, 2, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0, 0, 0],
                 [0, 9, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 7, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 7],
                 [0, 0, 0, 0, 7, 0, 0]]),
             _g([[2, 2, 2, 0, 0, 0, 0],
                 [2, 9, 2, 0, 0, 0, 0],
                 [2, 2, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0],
                      [3, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 3, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 8, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 2, 2, 2, 0, 0, 0],
                      [0, 2, 8, 2, 0, 0, 0],
                      [0, 2, 2, 2, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "ring_edge",
        "hint": "The one-of-a-kind color marks a center; erase "
        "everything else and surround it with a ring.",
        "train": [
            (_g([[7, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 3, 0, 0, 0, 0]]),
             _g([[7, 4, 0, 0, 0, 0],
                 [4, 4, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]])),
            (_g([[2, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1]]),
             _g([[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 4, 4],
                 [0, 0, 0, 0, 4, 1]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 6],
                      [0, 0, 0, 0, 0, 0],
                      [0, 3, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 3, 0, 0],
                      [0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 4, 6],
                      [0, 0, 0, 0, 4, 4],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]))],
    },
    # Slide-to-anchor lessons — scaffold toward arc:05f2a901. The
    # lone object moves straight toward the anchor until they touch.
    {
        "name": "attract_down",
        "hint": "The object slides toward the anchor until they touch.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 0, 0, 0, 0, 0],
                 [0, 0, 2, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 8, 8, 0, 0, 0],
                 [0, 0, 0, 8, 8, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 0, 0, 0, 0, 0],
                 [0, 0, 2, 2, 0, 0, 0, 0],
                 [0, 0, 0, 8, 8, 0, 0, 0],
                 [0, 0, 0, 8, 8, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])),
            (_g([[0, 0, 0, 0, 6, 0, 0, 0],
                 [0, 0, 0, 0, 6, 6, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 8, 8, 0, 0],
                 [0, 0, 0, 0, 8, 8, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 6, 0, 0, 0],
                 [0, 0, 0, 0, 6, 6, 0, 0],
                 [0, 0, 0, 0, 8, 8, 0, 0],
                 [0, 0, 0, 0, 8, 8, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 8, 8, 0, 0, 0],
                      [0, 0, 0, 8, 8, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 4, 4, 0, 0, 0],
                      [0, 0, 0, 0, 4, 0, 0, 0]]),
                  _g([[0, 0, 0, 8, 8, 0, 0, 0],
                      [0, 0, 0, 8, 8, 0, 0, 0],
                      [0, 0, 0, 4, 4, 0, 0, 0],
                      [0, 0, 0, 0, 4, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "attract_side",
        "hint": "The object slides toward the anchor until they touch.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 8, 8, 0, 0, 0, 3, 0],
                 [0, 8, 8, 0, 0, 0, 3, 3],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 8, 8, 3, 0, 0, 0, 0],
                 [0, 8, 8, 3, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 4, 0, 0, 0, 8, 8, 0],
                      [4, 4, 0, 0, 0, 8, 8, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 4, 8, 8, 0],
                      [0, 0, 0, 4, 4, 8, 8, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]))],
    },
    # Stamp-at-marks lessons — scaffold toward arc:363442ee. The
    # block by the divider is stamped centered on every mark.
    {
        "name": "stamp_marks_intro",
        "hint": "Copy the block by the divider onto every mark.",
        "train": [
            (_g([[2, 1, 2, 5, 0, 0, 0, 0, 0],
                 [1, 1, 2, 5, 0, 0, 0, 0, 0],
                 [2, 2, 1, 5, 0, 0, 0, 1, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0]]),
             _g([[2, 1, 2, 5, 0, 0, 0, 0, 0],
                 [1, 1, 2, 5, 0, 0, 2, 1, 2],
                 [2, 2, 1, 5, 0, 0, 1, 1, 2],
                 [0, 0, 0, 5, 0, 0, 2, 2, 1],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[3, 4, 4, 5, 0, 0, 0, 0, 0],
                      [4, 3, 4, 5, 0, 0, 0, 0, 0],
                      [3, 4, 3, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 1, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0]]),
                  _g([[3, 4, 4, 5, 0, 0, 0, 0, 0],
                      [4, 3, 4, 5, 0, 0, 0, 0, 0],
                      [3, 4, 3, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 3, 4, 4],
                      [0, 0, 0, 5, 0, 0, 4, 3, 4],
                      [0, 0, 0, 5, 0, 0, 3, 4, 3]]))],
    },
    {
        "name": "stamp_marks_multi",
        "hint": "Copy the block by the divider onto every mark.",
        "train": [
            (_g([[6, 6, 1, 5, 0, 0, 0, 0, 0],
                 [6, 1, 6, 5, 0, 0, 0, 1, 0],
                 [1, 6, 6, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0, 1, 0],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0]]),
             _g([[6, 6, 1, 5, 0, 0, 6, 6, 1],
                 [6, 1, 6, 5, 0, 0, 6, 1, 6],
                 [1, 6, 6, 5, 0, 0, 1, 6, 6],
                 [0, 0, 0, 5, 0, 0, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 6, 6, 1],
                 [0, 0, 0, 5, 0, 0, 6, 1, 6],
                 [0, 0, 0, 5, 0, 0, 1, 6, 6]])),
        ],
        "test": [(_g([[9, 2, 9, 5, 0, 0, 0, 0, 0],
                      [2, 9, 2, 5, 0, 0, 0, 0, 0],
                      [9, 2, 2, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 1, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 1, 0]]),
                  _g([[9, 2, 9, 5, 0, 0, 0, 0, 0],
                      [2, 9, 2, 5, 0, 0, 0, 0, 0],
                      [9, 2, 2, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 9, 2, 9, 0, 0],
                      [0, 0, 0, 5, 2, 9, 2, 0, 0],
                      [0, 0, 0, 5, 9, 2, 9, 2, 9],
                      [0, 0, 0, 5, 0, 0, 2, 9, 2]]))],
    },
    # radial_map: objects are compressed onto a 3x3 map by the sign
    # of their offset from the nearest marker cell.
    {
        "name": "radial_intro",
        "hint": "Draw each object around the marker's center by the "
                "direction it lies.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 2, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 5, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 2, 2],
                 [0, 5, 0],
                 [0, 0, 0]])),
            (_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 5, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[1, 0, 1],
                 [0, 5, 0],
                 [0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 5, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 8, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 8, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0],
                      [0, 5, 0],
                      [0, 8, 8]]))],
    },
    {
        "name": "radial_multi",
        "hint": "Draw each object around the marker's center by the "
                "direction it lies.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 4, 4, 0, 0, 0, 0],
                 [0, 0, 5, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 5, 0, 0],
                 [0, 0, 0, 0, 0, 3, 0, 3, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 4],
                 [0, 5, 0],
                 [3, 0, 3]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 9, 0, 9, 0],
                      [0, 0, 0, 0, 0, 0, 5, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 7, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 7, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 5, 0, 0, 0, 0, 0, 0, 0]]),
                  _g([[9, 0, 7],
                      [0, 5, 0],
                      [0, 0, 7]]))],
    },
    # fill_busiest: the block holding the most marks floods its
    # whole cell; every other mark is erased.
    {
        "name": "busy_intro",
        "hint": "Fill the cell that holds the most marks.",
        "train": [
            (_g([[2, 0, 0, 5, 0, 0, 0],
                 [0, 2, 0, 5, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0],
                 [5, 5, 5, 5, 5, 5, 5],
                 [0, 0, 0, 5, 0, 0, 0],
                 [2, 0, 0, 5, 0, 2, 0],
                 [0, 0, 0, 5, 0, 0, 0]]),
             _g([[2, 2, 2, 5, 0, 0, 0],
                 [2, 2, 2, 5, 0, 0, 0],
                 [2, 2, 2, 5, 0, 0, 0],
                 [5, 5, 5, 5, 5, 5, 5],
                 [0, 0, 0, 5, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 5, 0, 0, 4],
                      [0, 0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0],
                      [5, 5, 5, 5, 5, 5, 5],
                      [4, 0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0],
                      [0, 4, 0, 5, 0, 0, 0]]),
                  _g([[0, 0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0],
                      [5, 5, 5, 5, 5, 5, 5],
                      [4, 4, 4, 5, 0, 0, 0],
                      [4, 4, 4, 5, 0, 0, 0],
                      [4, 4, 4, 5, 0, 0, 0]]))],
    },
    {
        "name": "busy_tie",
        "hint": "Fill the cell that holds the most marks.",
        "train": [
            (_g([[1, 0, 1, 5, 0, 0, 0],
                 [0, 0, 0, 5, 0, 0, 0],
                 [0, 0, 0, 5, 0, 1, 0],
                 [5, 5, 5, 5, 5, 5, 5],
                 [0, 0, 0, 5, 0, 0, 0],
                 [0, 0, 0, 5, 0, 1, 0],
                 [0, 0, 0, 5, 0, 0, 1]]),
             _g([[1, 1, 1, 5, 0, 0, 0],
                 [1, 1, 1, 5, 0, 0, 0],
                 [1, 1, 1, 5, 0, 0, 0],
                 [5, 5, 5, 5, 5, 5, 5],
                 [0, 0, 0, 5, 1, 1, 1],
                 [0, 0, 0, 5, 1, 1, 1],
                 [0, 0, 0, 5, 1, 1, 1]])),
        ],
        "test": [],
    },
    # fill_lanes: a row or column whose only marks are its two ends
    # gets its interior painted.
    {
        "name": "lane_intro",
        "hint": "Paint the empty stretch between the two ends of a "
                "clear row or column.",
        "train": [
            (_g([[8, 0, 0, 0, 2],
                 [0, 0, 0, 0, 0],
                 [6, 0, 0, 0, 6],
                 [0, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0]]),
             _g([[8, 3, 3, 3, 2],
                 [3, 0, 0, 0, 3],
                 [6, 3, 3, 3, 6],
                 [0, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [7, 0, 0, 0, 7],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]]),
                  _g([[0, 1, 0, 0, 0],
                      [0, 3, 0, 0, 0],
                      [7, 3, 3, 3, 7],
                      [0, 3, 0, 0, 0],
                      [0, 1, 0, 0, 0]]))],
    },
    {
        "name": "lane_busy",
        "hint": "Paint the empty stretch between the two ends of a "
                "clear row or column.",
        "train": [
            (_g([[2, 0, 0, 0, 0, 8],
                 [0, 0, 0, 0, 0, 0],
                 [5, 0, 0, 9, 0, 5],
                 [0, 0, 0, 0, 0, 0],
                 [2, 0, 0, 0, 0, 8]]),
             _g([[2, 3, 3, 3, 3, 8],
                 [0, 0, 0, 0, 0, 0],
                 [5, 0, 0, 9, 0, 5],
                 [0, 0, 0, 0, 0, 0],
                 [2, 3, 3, 3, 3, 8]])),
        ],
        "test": [],
    },
    # eye_ray: the lone odd cell inside a shape shoots a beam across
    # the shape toward its far side.
    {
        "name": "eye_ray_side",
        "hint": "The odd cell inside the shape shoots a beam out "
                "through the shape.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 4, 0, 0, 0, 0, 0],
                 [0, 0, 4, 4, 0, 0, 0, 0],
                 [0, 0, 1, 4, 4, 0, 0, 0],
                 [0, 0, 4, 4, 0, 0, 0, 0],
                 [0, 0, 4, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 4, 0, 0, 0, 0, 0],
                 [0, 0, 4, 4, 0, 0, 0, 0],
                 [0, 0, 1, 4, 4, 1, 1, 1],
                 [0, 0, 4, 4, 0, 0, 0, 0],
                 [0, 0, 4, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 7, 7, 7, 0, 0],
                      [0, 0, 0, 0, 7, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 7, 7, 7, 0, 0],
                      [0, 0, 0, 0, 7, 0, 0, 0],
                      [0, 0, 0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 0, 3, 0, 0, 0]]))],
    },
    {
        "name": "eye_ray_up",
        "hint": "The odd cell inside the shape shoots a beam out "
                "through the shape.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 6, 0, 0, 0],
                 [0, 0, 0, 6, 6, 6, 0, 0],
                 [0, 0, 6, 6, 2, 6, 6, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 6, 0, 0, 0],
                 [0, 0, 0, 6, 6, 6, 0, 0],
                 [0, 0, 6, 6, 2, 6, 6, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])),
        ],
        "test": [],
    },
    # cross_fill: the divider grid's middle cross is painted with a
    # fixed palette (N=2, W=4, center=6, E=3, S=1).
    {
        "name": "cross_fill_intro",
        "hint": "Paint the divider grid's cross: the arms each take "
                "their own color.",
        "train": [
            (_g([[0, 0, 8, 0, 8, 0, 0],
                 [0, 0, 8, 0, 8, 0, 0],
                 [8, 8, 8, 8, 8, 8, 8],
                 [0, 0, 8, 0, 8, 0, 0],
                 [8, 8, 8, 8, 8, 8, 8],
                 [0, 0, 8, 0, 8, 0, 0],
                 [0, 0, 8, 0, 8, 0, 0]]),
             _g([[0, 0, 8, 2, 8, 0, 0],
                 [0, 0, 8, 2, 8, 0, 0],
                 [8, 8, 8, 8, 8, 8, 8],
                 [4, 4, 8, 6, 8, 3, 3],
                 [8, 8, 8, 8, 8, 8, 8],
                 [0, 0, 8, 1, 8, 0, 0],
                 [0, 0, 8, 1, 8, 0, 0]])),
        ],
        "test": [(_g([[0, 8, 0, 8, 0],
                      [8, 8, 8, 8, 8],
                      [0, 8, 0, 8, 0],
                      [8, 8, 8, 8, 8],
                      [0, 8, 0, 8, 0]]),
                  _g([[0, 8, 2, 8, 0],
                      [8, 8, 8, 8, 8],
                      [4, 8, 6, 8, 3],
                      [8, 8, 8, 8, 8],
                      [0, 8, 1, 8, 0]]))],
    },
]


def _cell_match(pred: Grid, expected: Grid) -> float:
    """Fraction of cells a guess gets right — the 0→1 mastery scale."""
    if pred.shape != expected.shape:
        return 0.0
    total = pred.height * pred.width
    if total == 0:
        return 0.0
    same = sum(
        1
        for r in range(pred.height)
        for c in range(pred.width)
        if pred.at(r, c) == expected.at(r, c)
    )
    return same / total


_ARC_DIR = (
    Path(__file__).resolve().parents[3] / "evals" / "arc_tasks"
)
# Attempts without mastery before a task parks and the next unlocks.
# Parked tasks stay retryable — nothing is ever forced or closed.
_PARK_AFTER = 8
_ARC_HINT = "Transform each input grid the way the examples show."

# Per-task instructions — one sentence describing the goal for tasks
# where the generic hint undersells the structure.
_ARC_HINTS = {
    "22233c11": (
        "The objects lie on invisible diagonal lines; paint copies "
        "of their shape, in the new color, where each line continues."
    ),
}


def _load_arc_tasks() -> list[dict]:
    """Append real ARC-AGI-1 tasks to the curriculum, easy-first.

    Ordering is by total training cells — a neutral size heuristic,
    not a difficulty ranking tuned to her solver.
    """
    tasks: list[dict] = []
    if not _ARC_DIR.is_dir():
        return tasks
    for path in sorted(_ARC_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            train = [
                (_g(p["input"]), _g(p["output"])) for p in data["train"]
            ]
            test = [
                (_g(p["input"]), _g(p["output"]))
                for p in data["test"]
                if "output" in p
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if train and test:
            size = sum(g.height * g.width for g, _ in train)
            tasks.append({
                "name": f"arc:{path.stem}",
                "hint": _ARC_HINTS.get(path.stem, _ARC_HINT),
                "train": train,
                "test": test,
                "_size": size,
            })
    tasks.sort(key=lambda t: t["_size"])
    for t in tasks:
        del t["_size"]
    return tasks


CURRICULUM.extend(_load_arc_tasks())


@dataclass
class PracticeAttempt:
    """The result of one attempt at the current puzzle."""

    task: str
    hint: str
    score: float          # this attempt's cell accuracy
    best: float           # running mastery after this attempt
    solved: bool          # exact match this attempt
    mastered: bool        # reached 1.0 — next puzzle unlocked
    rule: str             # her hypothesis description (or "none")
    nodes: int            # search effort this attempt
    total_attempts: int   # lifetime attempts on this puzzle
    failure: str          # why it failed ("" when solved or unknown)


@dataclass
class SpatialPractice:
    """Genesis's persistent puzzle curriculum.

    Lives inside her Mind: owns the gating and mastery state, runs
    attempts through her SpatialReasoner when her volition raises the
    puzzle urge.
    """

    data_dir: str
    _progress_path: Path = field(init=False)
    mastery: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    # Rules already proven wrong per task — verified-on-train rules
    # whose test guess missed (overfit counterexamples) plus the best
    # near-miss. Retries exclude them so each attempt explores
    # genuinely new hypothesis space instead of re-deriving the same
    # failure. This is the persistent half of learning from mistakes.
    failed: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._progress_path = Path(self.data_dir) / _PROGRESS_FILE
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._progress_path.read_text())
            self.mastery = {
                str(k): float(v)
                for k, v in data.get("mastery", {}).items()
            }
            self.attempts = {
                str(k): int(v)
                for k, v in data.get("attempts", {}).items()
            }
            self.failed = {
                str(k): [str(r) for r in v]
                for k, v in data.get("failed", {}).items()
            }
        except (OSError, json.JSONDecodeError, ValueError):
            self.mastery = {}
            self.attempts = {}
            self.failed = {}

    def _save(self) -> None:
        tmp = self._progress_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(
                {
                    "mastery": self.mastery,
                    "attempts": self.attempts,
                    "failed": self.failed,
                },
                indent=2,
            ))
            tmp.replace(self._progress_path)
        except OSError:
            pass

    def _unlocked(self, index: int) -> bool:
        """A task unlocks when every earlier task is mastered or parked."""
        for t in CURRICULUM[:index]:
            n = t["name"]
            if (
                self.mastery.get(n, 0.0) < 1.0
                and self.attempts.get(n, 0) < _PARK_AFTER
            ):
                return False
        return True

    def current_task(self) -> dict | None:
        """The puzzle she'd attempt now — None when all are done.

        Prefers the earliest unlocked task that isn't parked; when
        everything open is parked, offers the earliest parked one so
        she can revisit it whenever she wants.
        """
        parked: dict | None = None
        for i, task in enumerate(CURRICULUM):
            name = task["name"]
            if self.mastery.get(name, 0.0) >= 1.0:
                continue
            if not self._unlocked(i):
                continue
            if self.attempts.get(name, 0) < _PARK_AFTER:
                return task
            if parked is None:
                parked = task
        return parked

    def has_pending(self) -> bool:
        """Whether an unmastered puzzle remains — a volition signal."""
        return self.current_task() is not None

    def locked_count(self) -> int:
        """Puzzles still locked behind the current one."""
        cur = self.current_task()
        if cur is None:
            return 0
        return len(CURRICULUM) - CURRICULUM.index(cur) - 1

    def attempt(
        self, reasoner: SpatialReasoner, time_budget: float = 30.0
    ) -> PracticeAttempt | None:
        """One attempt at the current puzzle.

        Runs her reasoner on the puzzle's training pairs, scores her
        best guess against the held-out test output, and persists
        mastery. Returns None when the curriculum is complete.
        """
        task = self.current_task()
        if task is None:
            return None

        name = str(task["name"])
        tests = [i for i, _ in task["test"]]
        expected = [o for _, o in task["test"]]
        sol = reasoner.solve(
            task["train"],
            tests,
            time_budget=time_budget,
            exclude_rules=set(self.failed.get(name, [])),
        )

        guess_sets = (
            sol.guesses or ([sol.predictions] if sol.predictions else [])
        )
        score = 0.0
        solved = False
        for guesses in guess_sets:
            acc = min(
                (
                    _cell_match(p, e)
                    for p, e in zip(guesses, expected, strict=True)
                ),
                default=0.0,
            )
            score = max(score, acc)
            if all(
                p == e for p, e in zip(guesses, expected, strict=True)
            ):
                solved = True
                score = 1.0

        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)

        # Learn from the miss: rules that verified on the training
        # pairs but guessed wrong on test are proven counterexamples —
        # record them so the next attempt can't walk the same path.
        # When nothing verified, the best near-miss is recorded so a
        # retry pushes past it rather than stalling on it again.
        failure = ""
        if not solved:
            proven_wrong = list(sol.verified_rules)
            if sol.hypothesis is not None:
                proven_wrong.append(sol.hypothesis.describe())
            known = self.failed.setdefault(name, [])
            for rule in proven_wrong:
                if rule not in known:
                    known.append(rule)
            del known[32:]
            if sol.verified_rules:
                failure = (
                    f"overfit: {sol.verified_rules[0]} verified on "
                    f"train but missed test"
                )
            elif sol.failure is not None:
                failure = sol.failure.describe()
        self._save()

        return PracticeAttempt(
            task=name,
            hint=str(task["hint"]),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=sol.hypothesis.describe() if sol.hypothesis else "none",
            nodes=sol.nodes_explored,
            total_attempts=self.attempts[name],
            failure=failure,
        )

    def status(self) -> dict[str, float]:
        """Mastery per puzzle — the same 0→1 progression as her art."""
        return {t["name"]: self.mastery.get(t["name"], 0.0)
                for t in CURRICULUM}
