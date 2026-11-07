"""Gamma-band phase binding — proto-objects formed by synchrony.

The mind-side V2 grouping organ, paired with `src/daemon/gamma.rs` (the
body's Rust binding on the optic nerve). ARC/game frames arrive here as
data, not optical nerve frames, so the occipital path gets the same math
in the word layer: Kuramoto oscillators seeded by colour, contiguity +
similarity coupling, integration to locked phases, readout as coherent
assemblies.

Biology: binding-by-synchrony (Singer & von der Malsburg). Features that
belong to one object phase-lock in gamma; different objects desynchronize.
A proto-object is a gamma-coherent assembly — individuated, trackable,
pre-recognition.

Discipline: this layer forms objects; it never recognizes them. A frame it
cannot bind reports absence — an empty scene, never a fabricated object.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

TAU = math.tau
_LOCAL_K = 0.7      # contiguity coupling — the dominant Gestalt term
_SIMILAR_K = 0.05   # similarity coupling — same-colour remote fragments
_WINDOW = 8         # receptive window for the similarity term
_PHASE_TOL = 0.35   # merge tolerance on collective locked phase
_STEPS = 40         # integration horizon — most fields settle well before


@dataclass(slots=True)
class ProtoObject:
    """One bound assembly — a gamma-coherent proto-object.

    Attributes:
        color: palette index the assembly was bound under.
        size: number of member cells.
        cells: member cells as flat indices (row * width + col).
        bbox: (min_row, min_col, max_row, max_col).
        centroid: (row, col) centroid.
        phase: the mean phase the assembly locked to — its binding signature.
        coherence: mean-field coherence 0..1 of the member oscillators.
    """

    color: int
    size: int
    cells: list[int]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    phase: float
    coherence: float


def _segment(grid: list[int], width: int, height: int, background: int) -> list[int]:
    """Contiguity pass — label 4-connected same-colour components.

    Grouping is seeded by adjacency before similarity coupling acts. Returns
    per-cell component labels; ``-1`` where the cell is background.
    """
    n = width * height
    labels = [-1] * n
    next_label = 0
    for seed in range(n):
        if grid[seed] == background or labels[seed] != -1:
            continue
        colour = grid[seed]
        labels[seed] = next_label
        stack = [seed]
        while stack:
            idx = stack.pop()
            r, c = divmod(idx, width)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width:
                    nidx = nr * width + nc
                    if grid[nidx] == colour and labels[nidx] == -1:
                        labels[nidx] = next_label
                        stack.append(nidx)
        next_label += 1
    return labels


def _integrate(
    grid: list[int], labels: list[int], width: int, height: int, steps: int
) -> list[float]:
    """Kuramoto integration — natural frequency seeded by colour.

    Strong coupling to same-component neighbours (contiguity); weak
    similarity coupling to same-colour cells inside a receptive window —
    the Gestalt similarity term that binds separated fragments of one hue.
    Returns the per-cell phase field (``NaN`` on background cells).
    """
    n = width * height
    phases = [math.nan] * n
    for i in range(n):
        if labels[i] != -1:
            phases[i] = (grid[i] + 1) * 0.5  # each hue its own phase basin
    for _ in range(steps):
        cur = phases.copy()
        for idx in range(n):
            if labels[idx] == -1:
                continue
            r, c = divmod(idx, width)
            drive = count = 0.0
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width:
                    nidx = nr * width + nc
                    if labels[nidx] == labels[idx]:
                        drive += math.sin(cur[nidx] - cur[idx])
                        count += 1.0
            delta = _LOCAL_K * drive / count if count else 0.0
            sim = scount = 0.0
            for nr in range(max(0, r - _WINDOW), min(height - 1, r + _WINDOW) + 1):
                for nc in range(max(0, c - _WINDOW), min(width - 1, c + _WINDOW) + 1):
                    nidx = nr * width + nc
                    if (
                        nidx != idx
                        and labels[nidx] != labels[idx]
                        and labels[nidx] != -1
                        and grid[nidx] == grid[idx]
                    ):
                        sim += math.sin(cur[nidx] - cur[idx])
                        scount += 1.0
            if scount:
                delta += _SIMILAR_K * sim / scount
            phases[idx] = (cur[idx] + delta) % TAU
    return phases


def bind(
    grid: list[int] | list[list[int]],
    width: int | None = None,
    height: int | None = None,
    background: int = -1,
    steps: int = _STEPS,
) -> list[ProtoObject]:
    """Bind a colour grid into proto-object assemblies.

    Args:
        grid: flat ``width*height`` palette list, or a 2-D row-major grid.
        width/height: required when ``grid`` is flat; inferred for 2-D input.
        background: palette index treated as ground; ``-1`` auto-detects the
            most common value (usual game convention).
        steps: Kuramoto integration horizon.

    Returns:
        Proto-object assemblies; empty when nothing binds — absence, not a
        fabricated object.
    """
    if width is None:  # 2-D input
        rows = cast(list[list[int]], grid)
        height = len(rows)
        width = len(rows[0]) if height else 0
        flat = [c for row in rows for c in row]
    else:
        flat = list(cast(list[int], grid))
        height = height or (len(flat) // width if width else 0)
    n = width * height
    if n == 0 or len(flat) != n:
        return []
    if background == -1:
        from collections import Counter

        background = Counter(flat).most_common(1)[0][0]
    labels = _segment(flat, width, height, background)
    phases = _integrate(flat, labels, width, height, steps)

    comps: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab != -1:
            comps.setdefault(lab, []).append(i)
    if not comps:
        return []

    # Collective phase + coherence per component
    stats = []
    for lab, cells in comps.items():
        sx = sy = 0.0
        for i in cells:
            sx += math.cos(phases[i])
            sy += math.sin(phases[i])
        m = len(cells)
        stats.append((lab, math.atan2(sy, sx), math.hypot(sx, sy) / m))

    # Union same-colour components whose collective phases locked — the
    # similarity binding: separated fragments of one hue form one assembly.
    parent = list(range(len(stats)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(stats)):
        for b in range(a + 1, len(stats)):
            la, lb = stats[a][0], stats[b][0]
            if flat[comps[la][0]] != flat[comps[lb][0]]:
                continue
            d = abs(stats[a][1] - stats[b][1])
            if min(d, TAU - d) < _PHASE_TOL:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    merged: dict[int, list[int]] = {}
    for i, (lab, _phase, _coh) in enumerate(stats):
        merged.setdefault(find(i), []).extend(comps[lab])

    out = []
    for cells in merged.values():
        n2 = len(cells)
        rs = [i // width for i in cells]
        cs = [i % width for i in cells]
        sx = sum(math.cos(phases[i]) for i in cells)
        sy = sum(math.sin(phases[i]) for i in cells)
        out.append(
            ProtoObject(
                color=flat[cells[0]],
                size=n2,
                cells=cells,
                bbox=(min(rs), min(cs), max(rs), max(cs)),
                centroid=(sum(rs) / n2, sum(cs) / n2),
                phase=math.atan2(sy, sx),
                coherence=math.hypot(sx, sy) / n2,
            )
        )
    return out
