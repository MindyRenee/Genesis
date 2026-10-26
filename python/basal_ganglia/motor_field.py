"""Distributed motor fields — a decision as a field, not a symbol.

The basal ganglia resolve competing drives into an action. That
resolution doesn't have to collapse to a single token the moment it
forms: here it is also laid out over the world grid as a *field* —
where to act, how strongly, what to avoid — and over the offered
actions as a disposition vector.

Two consumers read the same fields:

- The policy itself — the action field biases exploration scoring and
  the spatial field supplies coordinate-action targets, so the
  distributed decision feeds back into the discrete one.
- The daemon — the fields are published onto the tensor slab
  (``motor.spatial``, ``motor.actions``, ``motor.action_ids``), where
  ``src/field/motor.rs`` decodes the winners back onto the canonical
  64×64 coordinate grid. What the mind decided is readable by any
  process that shares the slab.

Field composition draws only on the parietal world model's learned
state — affordances, hazard marks, visit counts, action effects — so
it generalizes with the model instead of assuming a game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from parietal_lobe.world_model import (
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_WIN,
)

if TYPE_CHECKING:
    from parietal_lobe.world_model import WorldModel

# The canonical coordinate grid the action interface addresses —
# mirrored in `src/field/motor.rs` as MOTOR_FIELD_WIDTH/HEIGHT. The
# daemon scales field-local peaks onto this grid, so fields may be
# authored at the frame's own resolution.
MOTOR_FIELD_WIDTH = 64
MOTOR_FIELD_HEIGHT = 64

# Field amplitudes — relative weights of the drives the field carries.
_W_WIN = 1.0  # touching this kind won before
_W_COLLECT = 0.7  # this kind is consumable value
_W_HAZARD = 1.0  # this kind killed her
_W_UNPROBED = 0.25  # an entity whose affordance is still unknown
_W_RESPONSIVE = 0.5  # a kind whose distal acts the world has answered
_W_FRONTIER = 0.15  # visited space adjacent to unseen space
_W_BLOCKED = 0.8  # known-solid cells
_W_UNTRIED = 0.3  # an action whose effect is unlearned
_W_RESET = 0.8  # interface-labeled episode-ending actions
_W_RECALL = 0.6  # episodic recall's associative valence on a kind


@dataclass(slots=True)
class MotorField:
    """One step's distributed motor state.

    Attributes:
        spatial: H×W field over the world grid — positive cells
            attract, negative repel.
        action_scores: action token → disposition (may be negative).
        peak: ``(row, col)`` argmax of ``spatial``, or None.
    """

    spatial: list[list[float]]
    action_scores: dict[str, float]
    peak: tuple[int, int] | None = field(default=None)


def compose_spatial_field(model: WorldModel, height: int, width: int) -> list[list[float]]:
    """Project the world model's learned values onto the grid.

    Nested (row, col) accumulation over each live entity's cells —
    goal/value attract, hazards repel, the merely-unprobed pull
    weakly — plus map-level terms: known-blocked cells repel and the
    frontier (known-free cells next to never-visited space) attracts.
    """
    f = [[0.0] * width for _ in range(height)]

    for entity in model.entities.values():
        if (
            entity.gone
            or entity.is_self
            or entity.part_of_self
            or entity.state_of_self
            or entity.color in model.terrain_kinds
        ):
            continue
        aff = model.affordance_of(entity.color)
        strength = (
            _W_WIN * aff.get(OUT_WIN, 0)
            + _W_COLLECT * aff.get(OUT_COLLECT, 0)
            - _W_HAZARD * aff.get(OUT_HAZARD, 0)
        )
        # A kind the world answers when she acts at it keeps its
        # pull — a control worth pressing again. Normalized so a
        # press that moves ~8 cells saturates the term.
        responsive = min(1.0, model.click_responsiveness(entity.color) / 8.0)
        strength += _W_RESPONSIVE * responsive
        # Associative recall: a kind her memory remembers as a goal
        # attracts, as a killer repels — a voice in the field, not a
        # plan. Present-measured affordances still outweigh it.
        strength += _W_RECALL * model.recalled_valence_of(entity.color)
        if strength == 0.0 and not aff:
            trials = sum(
                n for (_action, color), n in model.click_tries.items() if color == entity.color
            )
            strength = _W_UNPROBED / (1.0 + trials)
        for r, c in entity.cells:
            if 0 <= r < height and 0 <= c < width:
                f[r][c] += strength

    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell in model.known_blocked:
                f[r][c] -= _W_BLOCKED
            elif cell in model.known_free and model.visits[cell] == 0:
                # Frontier pull only matters beside somewhere seen.
                if any(
                    model.visits[(r + dr, c + dc)] > 0
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if (r + dr, c + dc) in model.known_free
                ):
                    f[r][c] += _W_FRONTIER
    return f


def compose_action_field(
    model: WorldModel,
    available: list[str],
    spatial: list[list[float]],
) -> dict[str, float]:
    """Dispositions over the offered actions.

    An action with a learned self-displacement inherits the field
    value of the cell it would reach — moving toward a valued peak
    scores like the peak. Untried actions carry a calibration bonus
    (their effect is unlearned; trying them is how the field learns).
    Interface-labeled "reset" actions are suppressed — restarting is
    a last resort, not a drive.
    """
    height = len(spatial)
    width = len(spatial[0]) if height else 0
    me = model.self_entity
    fp = model.self_footprint() if me is not None else frozenset()
    scores: dict[str, float] = {}
    for action in available:
        score = 0.0
        effect = model.expected_effect(action) if me is not None else None
        if effect is not None and (effect[0] or effect[1]) and fp:
            dest_cells = {(r + effect[0], c + effect[1]) for r, c in fp}
            field_cells = [(r, c) for r, c in dest_cells if 0 <= r < height and 0 <= c < width]
            if field_cells:
                score += sum(spatial[r][c] for r, c in field_cells) / len(field_cells)
            if dest_cells & model.known_blocked:
                score -= _W_BLOCKED
        if not model.action_understood(action):
            score += _W_UNTRIED
        if model.action_labels.get(action) == "reset":
            score -= _W_RESET
        scores[action] = score
    return scores


def compose_motor_fields(
    model: WorldModel, available: list[str], height: int, width: int
) -> MotorField:
    """Compose both fields for one decision step."""
    spatial = compose_spatial_field(model, height, width)
    peak: tuple[int, int] | None = None
    best = float("-inf")
    for r in range(height):
        for c in range(width):
            if spatial[r][c] > best:
                best = spatial[r][c]
                peak = (r, c)
    if peak is not None and best <= 0.0:
        peak = None  # no positive attractor — nothing pulls
    actions = compose_action_field(model, available, spatial)
    return MotorField(spatial=spatial, action_scores=actions, peak=peak)
