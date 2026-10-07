"""Parietal spatial imagery — the depictive workspace.

This module is the mental-imagery faculty for spatial structure:
known transforms are *skills* she can apply to an internal copy of a
grid and re-perceive, exactly the way Soar's Spatial/Visual System
exposes imagery operations to an agent's general decision procedure
(Lathrop, Wintermute & Laird 2011; Wintermute 2012). Three design
commitments, each grounded in the literature:

- **Imagery reuses perceptual-motor machinery offline.** Imagined
  transforms are actions simulated on the depictive buffer —
  motor-imagery analog (Jeannerod 2001; Caligiore et al. 2013) — and
  the generative direction of predictive coding: higher levels
  synthesize the low-level representation a transform would produce
  (Breedlove et al. 2020; Dijkstra et al. 2018).
- **Recognition is not search.** ``recognize_change`` asks which
  single known skill explains an observed input→output change —
  one step per candidate, parallel in spirit (Copycat's codelet
  style, Hofstadter & Mitchell 1994). Composition and sequencing
  belong to the general planner, not to this module; humans likewise
  imagine one step and re-perceive rather than enumerating programs.
- **Skills are knowledge.** The vocabulary lives in
  ``spatial/transforms.py``; each name grounds as a
  ``spatial_skill:*`` concept so spreading activation can make
  relevant skills salient (the Slipnet hook). Skills learned later
  register through ``register_skill`` — the repertoire grows with
  experience instead of being fixed at birth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..spatial.grid import Grid
from ..spatial.transforms import (
    INFER_OP_NAMES,
    INFER_OPS,
    Op,
    parameterized_ops,
    simple_ops,
)

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

logger = logging.getLogger(__name__)

# Infer-ops read a transform's parameters off an observed
# input→output pair — perceptual inference, not search — so they
# participate in recognition alongside the fixed vocabulary. Display
# names drop the ``infer_`` prefix ("infer_recolor_map" → "recolor_map").
_INFER_SKILLS = tuple(
    (INFER_OP_NAMES[fn].removeprefix("infer_"), fn) for fn in INFER_OPS
)


class SpatialImagery:
    """Depictive workspace: apply and recognize spatial skills."""

    def __init__(self, network: ConceptNetwork | None = None) -> None:
        # The innate vocabulary — motor primitives of grid space.
        self._skills: dict[str, Op] = dict(simple_ops())
        # Skills acquired after birth (teaching, discovered rules).
        self._learned: dict[str, Op] = {}
        self._grounded = False
        if network is not None:
            self._ground_skills(network)

    @property
    def known_skills(self) -> list[str]:
        """Skill names she can currently imagine, in fixed order."""
        return [*self._skills, *self._learned]

    def imagine(self, grid: Grid, skill: str) -> Grid | None:
        """Apply one skill to an internal copy of ``grid``.

        The forward-model half of imagery: predict what this transform
        would produce, one step, without touching perception. Returns
        ``None`` for unknown skills and inapplicable ones alike —
        inapplicability is data, not an error.
        """
        op: Op | None = self._skills.get(skill) or self._learned.get(skill)
        if op is None:
            # Parameterized families are per-input (keep_color_N) —
            # resolve the name against this grid's own colors.
            op = dict(parameterized_ops(grid)).get(skill)
        if op is None:
            return None
        try:
            return op(grid)
        except Exception as e:  # noqa: BLE001 — an op must never break imagery
            logger.debug(f"imagine {skill} failed: {e}")
            return None

    def applicable_skills(self, grid: Grid) -> list[str]:
        """Skills whose preconditions hold on this grid — affordances
        of the imagery workspace."""
        return [
            name for name in self.known_skills
            if self.imagine(grid, name) is not None
        ]

    def recognize_change(self, before: Grid, after: Grid) -> list[str]:
        """Which known skills explain ``before`` → ``after``.

        Single-step recognition: each candidate is imagined and its
        result compared to what was actually observed — the generative
        model run backward. Returns every single-skill explanation,
        empty when the change is complex or unfamiliar. Unfamiliar
        change is an epistemic gap for the caller to route (curiosity,
        learning), not a failure this module papers over.
        """
        if before == after:
            return []
        hits: list[str] = []
        candidates = dict(self._skills)
        candidates.update(self._learned)
        candidates.update(parameterized_ops(before))
        for name, op in candidates.items():
            try:
                if op(before) == after:
                    hits.append(name)
            except Exception:  # noqa: BLE001 — one bad op loses its candidacy
                continue
        for name, infer in _INFER_SKILLS:
            try:
                inferred = infer([(before, after)])
                if inferred is not None and inferred(before) == after:
                    hits.append(name)
            except Exception:  # noqa: BLE001
                continue
        return hits

    def register_skill(self, name: str, op: Op) -> None:
        """Add a learned transform to the repertoire."""
        self._learned[name] = op

    def summary(self) -> str:
        n = len(self._skills) + len(self._learned)
        return f"{n} spatial skills"

    def _ground_skills(self, network: ConceptNetwork) -> None:
        """Write the vocabulary into the concept network once, so the
        skills are real concepts activation can spread to."""
        if self._grounded:
            return
        self._grounded = True
        for name in self._skills:
            try:
                network.add_concept(
                    f"spatial_skill:{name}",
                    origin="perception",
                    properties={"kind": "spatial_transform"},
                )
            except Exception as e:  # noqa: BLE001 — grounding is advisory
                logger.debug(f"skill grounding failed for {name}: {e}")
