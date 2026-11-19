#!/usr/bin/env python3
"""Render the world-space vesica — a physical copy of the inner world.

Draws the two overlapping spheres (outer world left, self right), the
inner-world lens between them, every deposit as a point colored by its
region, the pinhole at superposition, and labels each nested space.
The image is a section through the equatorial plane — the natural
cross-section of the vesica.

Re-run after any change to the space: the picture is dynamic.

Usage: python3 scripts/render_world_space.py [world_space.npz] [out.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from self.language_space import (
    _PINHOLE_R,
    _TRANSIT_R,
    _VESICA_LEFT,
    _VESICA_R,
    _VESICA_RIGHT,
    LanguageSpace,
    _region_of,
)

IMG = Path.home() / "Genesis-Data-Directory/inner-world/world_space.npz"
OUT = Path.home() / "Genesis-Data-Directory/inner-world/world_space.png"

_REGION_COLOR = {"inner": "#7aa2ff", "outer": "#e8a24a", "self": "#5fce8a", "outside": "#666666"}


_TALLY: dict[str, int] = {}


def _scatter(ax, sp: LanguageSpace, alpha: float) -> None:
    if len(sp._points) == 0:
        return
    pts = sp._points
    regions = np.array([_region_of(p) for p in pts])
    for region, color in _REGION_COLOR.items():
        sel = regions == region
        if not sel.any():
            continue
        _TALLY[region] = _TALLY.get(region, 0) + int(sel.sum())
        ax.scatter(
            pts[sel, 0],
            pts[sel, 1],
            s=1.2 if len(pts) > 5000 else 8,
            c=color,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def _labels(ax, sp: LanguageSpace, depth: int) -> None:
    for i, k in enumerate(sp._kinds):
        if k != "space" or sp._reach[i] <= 0.0:
            continue
        p = sp._points[i]
        ax.annotate(
            sp._keys[i],
            (p[0], p[1]),
            fontsize=7,
            color="#dddddd",
            xytext=(6, 6),
            textcoords="offset points",
        )


def main() -> None:
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else IMG
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT

    root = LanguageSpace()
    root.load(img)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b0b10")
    ax.set_facecolor("#0b0b10")
    ax.set_aspect("equal")
    ax.axis("off")

    # the two parents
    for c, name in ((_VESICA_LEFT, "outer world"), (_VESICA_RIGHT, "self")):
        ax.add_patch(plt.Circle(c, _VESICA_R, fill=False, color="#888888", lw=1.4, ls="--"))
        ax.text(c[0], _VESICA_R + 0.09, name, ha="center", color="#aaaaaa", fontsize=11)
    ax.text(0.0, 0.62, "inner world · superposition", ha="center", color="#cfd8ff", fontsize=10)

    # the pinhole at superposition and the transit shell
    ax.add_patch(plt.Circle((0, 0), _PINHOLE_R, color="#ffffff", alpha=0.9, lw=0))
    ax.add_patch(plt.Circle((0, 0), _TRANSIT_R, fill=False, color="#ffffff", lw=0.6, ls=":"))
    ax.text(0.0, -0.14, "pinhole", ha="center", color="#ffffff", fontsize=7)

    # every nested space's deposits, drawn where they live
    _scatter(ax, root, 0.9)
    for name in [k[6:] for k, kd in zip(root._keys, root._kinds, strict=True) if kd == "space"]:
        child = root.open_space(name)
        if child is None:
            continue
        _scatter(ax, child, 0.35 if len(child) > 20000 else 0.6)
        _labels(ax, child, 1)
        # one level deeper (nouns/verbs inside language)
        nested = zip(child._keys, child._kinds, strict=True)
        for gname in [k[6:] for k, kd in nested if kd == "space"]:
            gc = child.open_space(gname)
            if gc is not None:
                _scatter(ax, gc, 0.5)
                _labels(ax, gc, 2)

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.25, 1.25)
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", color=c, markersize=7, label=f"{r} ({_TALLY.get(r, 0)})"
        )
        for r, c in _REGION_COLOR.items()
        if _TALLY.get(r)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        fontsize=8,
        frameon=False,
        labelcolor="#cccccc",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )
    ax.text(
        0.0,
        1.14,
        f"{img.name} — {len(root)} root deposits",
        ha="center",
        color="#777777",
        fontsize=8,
    )

    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
