"""Parietal world model — embodied physics and outcome learning.

Covers the mechanisms a world resets must not break:

- velocity is the rigid translation that explains a footprint, not
  a centroid delta — occlusion and re-segmentation cannot invent
  sub-tile motion and poison ``action_effects``;
- terminal outcomes (win / death / unexplained reset) credit the
  contacts of the world that *ended*, then reset positions while
  keeping learned physics;
- hazard suspicion is weighed against everyday exposure — a wall
  she leans on all day is not a killer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import pytest
from parietal_lobe.grid import Grid
from parietal_lobe.scene import PerceivedObject, Scene, perceive
from parietal_lobe.world_model import (
    OUT_BLOCK,
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_NOTHING,
    OUT_TRANSFORM,
    OUT_WIN,
    SimState,
    WorldModel,
)


def _scene(regions: list[tuple[int, set[tuple[int, int]]]],
           h: int = 16, w: int = 16) -> Scene:
    """Paint color → cells regions on a 0-background grid and perceive."""
    grid = [[0] * w for _ in range(h)]
    for color, cells in regions:
        for r, c in cells:
            grid[r][c] = color
    return perceive(Grid.from_lists(grid), background=0)


_ME = 3
_ME_CELLS = {(4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4),
             (6, 2), (6, 3), (6, 4)}  # 3x3 block


def _world_with_self() -> WorldModel:
    """A world where the color-3 block has been identified as her."""
    model = WorldModel()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
         (9, {(12, 10), (12, 11)})]
    me = _ME_CELLS
    model.observe(_scene([(_ME, {(r + 4, c) for r, c in me}), *other]))
    model.note_action("up")
    model.observe(_scene([(_ME, {(r + 2, c) for r, c in me}), *other]))
    model.note_action("up")
    model.observe(_scene([(_ME, set(me)), *other]))
    assert model.self_id is not None
    return model


# ─── rigid translation ──────────────────────────────────────


def test_occluded_self_does_not_invent_motion():
    """A half-occluded footprint measures as unmoved, not a jitter."""
    model = _world_with_self()
    me = model.self_entity
    assert me is not None and set(me.cells) == _ME_CELLS

    # Action B: an occluder covers her lower rows — the visible
    # remainder is one row, which the old centroid delta read as a
    # (-1, 0) move. The rigid answer is (0, 0).
    model.note_action("push")
    occluded = {(4, 2), (4, 3), (4, 4)}
    occluder = {(r, c) for r, c in _ME_CELLS - occluded}
    model.observe(_scene([
        (_ME, occluded),
        (7, occluder),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]))

    assert model.effect_of("push") == (0, 0)
    assert model.action_noops["push"] == 1
    # The clean "up" effect is unpolluted.
    assert model.effect_of("up") == (-2, 0)


def test_modal_effect_survives_blocked_trials():
    """Blocked trials count as no-ops, never as rival effects."""
    model = _world_with_self()
    # Two clean "up" trials already sit in the counter. Add three
    # blocked (fully occluded → no motion) trials for "sideways":
    # under centroid measurement their jitter became the mode.
    for _ in range(3):
        model.note_action("side")
        model.observe(_scene([
            (_ME, {(4, 2), (4, 3), (4, 4)}),     # top row visible
            (7, {(5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)}),
            (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
            (9, {(12, 10), (12, 11)}),
        ]))
    assert model.action_noops["side"] == 3
    assert model.effect_of("side") == (0, 0)


def test_prediction_events_name_the_real_effect():
    """An honest mismatch reports the learned vector, not jitter."""
    model = _world_with_self()
    # "up" expects (-2, 0); hold her still and see the surprise.
    model.note_action("up")
    events = model.observe(_scene([
        (_ME, set(_ME_CELLS)),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]))
    surprises = [e for e in events if e.kind == "unexpected"]
    assert surprises, "a refused move should surprise the model"
    assert "expected (-2, 0)" in surprises[0].text


def test_self_swap_mid_trial_writes_no_effect():
    """A self re-adoption inside one trial is noise, not physics.

    She issued the action as one body and the frame answered with a
    different body's displacement — the ls20 composite avatar fails
    exactly this way, the swapped-in entity's stride landing in the
    effect table as a phantom. The trial is discarded: the swap
    itself stands (the new body is still self), only the
    unattributable motion is refused.
    """
    model = WorldModel()
    a0 = {(4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4),
          (6, 2), (6, 3), (6, 4)}
    b0 = {(10, 10), (10, 11), (11, 10), (11, 11)}
    hud = (9, {(0, 0), (0, 1)})
    model.observe(_scene([(_ME, a0), (7, b0), hud]))
    a, b = a0, b0
    for _ in range(2):
        model.note_action("up")
        a = {(r - 2, c) for r, c in a}
        b = {(r - 2, c) for r, c in b}
        model.observe(_scene([(_ME, a), (7, b), hud]))
    assert model.self_id is not None
    acting_self = model.self_id

    # Pending "sideways" is unlearned — any rigid delta counts as an
    # effect-move, so B's jump is enough to win the self flag the
    # moment A is gone.
    model.note_action("sideways")
    b = {(r + 5, c) for r, c in b}
    model.observe(_scene([(7, b), hud]))

    assert model.self_id is not None
    assert model.self_id != acting_self
    assert "sideways" not in model.action_effects


def test_cerebellar_comparator_flags_stale_action():
    """Sustained reafferent error re-opens an action for babbling.

    "up" is learned; when the world keeps answering it with a
    different displacement, the cerebellum's running error on the
    action crosses the miscalibration line and ``action_understood``
    stops vouching for it — the same verdict a body draws when a
    practiced move stops landing where it should.
    """
    from cerebellum.motor import SensorimotorLayer

    model = _world_with_self()
    assert model.action_understood("up")
    model.motor_comparator = SensorimotorLayer()

    others = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
              (9, {(12, 10), (12, 11)})]
    # "up" predicts (-2, 0); the world answers with a different
    # displacement each trial — every one maximal error, and no
    # rival delta accumulates enough votes to take over the mode.
    for dr, dc in [(0, -2), (1, -2), (0, -1), (1, -1)]:
        model.note_action("up")
        model.observe(_scene([
            (_ME, {(r + dr, c + dc) for r, c in _ME_CELLS}), *others
        ]))
        model.observe(_scene([
            (_ME, set(_ME_CELLS)), *others
        ]))

    assert model.motor_comparator.error_of("up") > 0.5
    assert not model.action_understood("up")


# ─── terminal outcomes ──────────────────────────────────────


def test_win_credits_last_contacts_and_resets():
    """info['win'] credits whatever she touched when the world ended."""
    model = _world_with_self()
    # The door is a tracked entity — she saw it, moved into contact.
    goal = {(4, 5), (4, 6), (5, 5)}
    model.observe(_scene([
        (_ME, set(_ME_CELLS)),
        (7, goal),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]))
    model.known_free.add((15, 15))
    events = model.observe(
        _scene([
            (_ME, set(_ME_CELLS)),
            (7, goal),
            (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
            (8, {(15, 15)}),
            (9, {(12, 10), (12, 11)}),
        ]),
        info={"win": True},
    )
    assert model.affordance_of(7)[OUT_WIN] == 1
    assert any(e.kind == OUT_WIN and "goal" in e.text for e in events)
    # The world that ended is forgotten — everything tracked now
    # was first seen this tick — while learned physics persist.
    assert model.entities
    assert all(e.first_seen == model.tick
               for e in model.entities.values())
    assert model.effect_of("up") == (-2, 0)
    # A new level is not a retry: cell-map verdicts from the solved
    # layout must not become phantom walls in this one.
    assert (15, 15) not in model.known_free


def test_death_retry_keeps_the_spatial_map():
    """Dying in one maze remembers its walls; winning leaves them."""
    model = _world_with_self()
    model.known_free.add((15, 15))
    model.known_blocked.add((15, 14))
    model.reset(keep_learning=True, keep_map=True)
    assert (15, 15) in model.known_free
    assert (15, 14) in model.known_blocked


def test_wholesale_turnover_is_an_unannounced_reset():
    """Zero entity continuity means the world was replaced under her."""
    model = _world_with_self()
    # Give her a visible occluder in contact first — the suspect.
    model.observe(_scene([
        (_ME, {(4, 2), (4, 3), (4, 4)}),
        (7, {(5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)}),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]))
    # A brand-new world in brand-new colors — nothing matches.
    events = model.observe(_scene([
        (2, {(0, 5), (0, 6)}),
        (11, {(8, 8), (9, 8), (9, 9)}),
        (13, {(14, 1), (14, 2), (14, 3)}),
    ]))
    assert any(e.kind == "reset" for e in events)
    assert any(e.kind == "suspect" for e in events)
    # The occluder she was pressed against earns the suspicion.
    assert model.death_suspects[7] == 1


def test_unannounced_reset_suspects_the_pending_path():
    """A stride's killer may live inside the path, never in contact.

    A reset can replace the whole scene before the occupied frame
    exists. Contiguity then has no chance to testify; the pending
    motion's swept cells are the only place the cause can still be
    measured.
    """
    model = _world_with_self()          # learned "up" is (-2, 0)
    model.observe(_scene([
        (_ME, set(_ME_CELLS)),
        (7, {(2, 2), (2, 3)}),          # inside the swept strip
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]))
    model.note_action("up")
    events = model.observe(_scene([
        (2, {(0, 5), (0, 6)}),
        (11, {(8, 8), (9, 8), (9, 9)}),
    ]))
    assert any(e.kind == "reset" for e in events)
    assert model.death_suspects[7] == 1
    assert any("path" in e.text for e in events if e.kind == "suspect")


def test_partial_turnover_is_not_a_reset():
    """Ordinary motion — some entities persist — is not a reset."""
    model = _world_with_self()
    events = model.observe(_scene([
        (_ME, set(_ME_CELLS)),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),          # wall persists
        (8, {(13, 13), (14, 13)}),                      # gem replaced
    ]))
    assert not any(e.kind == "reset" for e in events)


# ─── hazard judgment ────────────────────────────────────────


def test_hazard_kind_weighs_suspicion_against_exposure():
    """A wall she leans on is not a killer; a rare fatal touch is."""
    model = WorldModel()
    # Confirmed kill always counts.
    model.affordances.setdefault(4, Counter())[OUT_HAZARD] += 1
    assert model.hazard_kind(4)
    # Ubiquitous presence at endings ≠ danger.
    model.contact_ticks[5] = 500
    model.death_suspects[5] = 4
    assert not model.hazard_kind(5)
    # A kind seen almost only at endings convicts.
    model.contact_ticks[6] = 2
    model.death_suspects[6] = 3
    assert model.hazard_kind(6)


def test_collect_kind_survives_adjacent_death():
    """A collectible next to her at a budget death stays valuable."""
    model = _world_with_self()
    # A pickup she keeps touching, then one budget death nearby.
    model.contact_ticks[9] = 120
    model.death_suspects[9] = 2
    model.affordances.setdefault(9, Counter())[OUT_COLLECT] += 2
    assert not model.hazard_kind(9)


# ─── conditional passability ────────────────────────────────


def test_blocked_state_pair_is_not_kind_wide():
    """A refusal is recorded against (entity, self-state), not the kind."""
    model = _world_with_self()
    me = model.self_entity
    assert me is not None and me.color == _ME
    sig = model.self_signature()
    assert sig is not None
    door = model._spawn(PerceivedObject(
        index=99, color=7, cells=frozenset({(4, 5), (4, 6)})
    ))
    door.blocked_states.add(sig)
    model.affordances.setdefault(7, Counter())[OUT_BLOCK] += 1
    # Same state: refused. A remade self may enter.
    assert not model.passable(door, sig)
    assert model.passable(door, ((11, sig[0][1]),))


# ─── composite self ─────────────────────────────────────────


_PART = {(4, 5), (5, 5)}          # color-9 block attached to her right edge


def _composite_world() -> WorldModel:
    """Her body is two pieces — a color-3 core and a color-9 part.

    The part rides at her right edge and translates with her; two
    consecutive rigid co-moves earn annexation. She ends back at
    her original cells for a stable reference frame.
    """
    model = WorldModel()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
             (9, {(12, 10), (12, 11)})]
    model.observe(_scene([(_ME, set(_ME_CELLS)),
                          (9, set(_PART)), *other]))
    for step in (1, 2):              # down: rows 4-6 → 6-8 → 8-10
        model.note_action("down")
        model.observe(_scene([
            (_ME, {(r + 2 * step, c) for r, c in _ME_CELLS}),
            (9, {(r + 2 * step, c) for r, c in _PART}),
            *other,
        ]))
    for step in (1, 2):              # up: back to rows 4-6
        model.note_action("up")
        model.observe(_scene([
            (_ME, {(r + 2 * (2 - step), c) for r, c in _ME_CELLS}),
            (9, {(r + 2 * (2 - step), c) for r, c in _PART}),
            *other,
        ]))
    assert model.self_id is not None
    part = next(e for e in model.entities.values() if e.part_of_self)
    assert part.color == 9
    return model


def test_rigid_companion_is_annexed_as_part():
    """A thing translating with her contiguously is part of her body."""
    model = _composite_world()
    sig = model.self_signature()
    assert sig is not None and len(sig) == 2
    colors = sorted(s[0] for s in sig)
    assert colors == [_ME, 9]
    fp = model.self_footprint()
    assert fp == frozenset(_ME_CELLS | _PART)


def test_pacing_neighbor_is_not_a_part():
    """A thing that keeps pace but drifts is never annexed."""
    model = WorldModel()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)})]
    start = {(r + 4, c) for r, c in _ME_CELLS}   # rows 8-10
    model.observe(_scene([(_ME, start),
                          (9, {(r + 4, c) for r, c in _PART}), *other]))
    # The neighbor moves half as far — close, but it does not ride.
    part_cells = {(r + 4, c) for r, c in _PART}
    for step in range(1, 4):
        model.note_action("up")
        part_cells = {(r - 1, c) for r, c in part_cells}
        model.observe(_scene([
            (_ME, {(r - 2 * step, c) for r, c in start}),
            (9, part_cells),
            *other,
        ]))
    assert all(not e.part_of_self for e in model.entities.values())


def test_part_transform_is_learned_and_simulated():
    """A pad rewriting her part yields a simulatable from→to rule."""
    model = _composite_world()
    part = next(e for e in model.entities.values() if e.part_of_self)
    sig_before = model.self_signature()
    assert sig_before is not None
    from_sig = next(s for s in sig_before if s[0] == 9)
    to_sig = (11, from_sig[1])
    # A color-7 pad beside the part. Paint order: pad first so the
    # part occludes it — contact is adjacency, as in the real world.
    pad_cells = {(6, 5), (7, 5)}
    model.note_action("down")
    model.observe(_scene([
        (7, pad_cells),
        (_ME, {(r + 1, c) for r, c in _ME_CELLS}),
        (11, {(r + 1, c) for r, c in _PART}),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]))
    # Recolored mid-move and still riding: identity kept by adoption.
    assert part.color == 11 and part.part_of_self
    assert model.transforms.get(7, {}).get(from_sig, Counter()
                                           )[to_sig] >= 1
    assert model.affordance_of(7)[OUT_TRANSFORM] >= 1
    # Simulation applies the learned rewrite to her old state.
    state = model.start_state()
    assert state is not None
    pre = SimState(cells=state.cells, signature=sig_before,
                   gone=state.gone)
    nxt = model.simulate(pre, "down")
    assert nxt is not None and to_sig in nxt.signature


def test_part_camouflage_drops_but_self_survives():
    """A part melting into same-colored ground is lost, not the floor."""
    model = _composite_world()
    part = next(e for e in model.entities.values() if e.part_of_self)
    # The part's landing cells are inside a large same-colored
    # expanse — one connected region, far too big to adopt.
    floor = {(r, c) for r in range(4, 10) for c in range(5, 15)}
    floor |= {(r + 1, c) for r, c in _PART}
    model.note_action("down")
    model.observe(_scene([
        (9, floor),
        (_ME, {(r + 1, c) for r, c in _ME_CELLS}),
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
    ]))
    # The huge same-colored expanse was not adopted as the part.
    assert not part.part_of_self
    sig = model.self_signature()
    assert sig is not None and len(sig) == 1
    assert model.self_entity is not None


def test_signature_conditional_door():
    """A gate learns per-signature: refused as one body, open as another."""
    model = _composite_world()
    sig_a = model.self_signature()
    assert sig_a is not None
    door = model._spawn(PerceivedObject(
        index=98, color=6, cells=frozenset({(4, 7)})
    ))
    door.blocked_states.add(sig_a)
    model.affordances.setdefault(6, Counter())[OUT_BLOCK] += 1
    assert not model.passable(door, sig_a)
    sig_b = tuple((c, s) if c != 9 else (11, s) for c, s in sig_a)
    assert model.passable(door, sig_b)


# ─── terrain ────────────────────────────────────────────────


def _terrain_world() -> WorldModel:
    """She walks across a large floor region that reclaims her cells.

    The floor (color 8) fills the room; her color-3 block moves
    through it. Each vacated footprint reappears as floor — the
    medium absorbing the space she leaves.
    """
    floor_cells = {(r, c) for r in range(2, 14) for c in range(0, 14)}
    model = WorldModel()
    start = {(r + 4, c) for r, c in _ME_CELLS}   # rows 8-10
    model.observe(_scene([
        (_ME, start),
        (8, floor_cells - start),
    ]))
    cur = start
    for _ in range(3):
        nxt = {(r - 1, c) for r, c in cur}
        model.note_action("up")
        model.observe(_scene([
            (_ME, nxt),
            (8, floor_cells - nxt),
        ]))
        cur = nxt
    assert model.self_id is not None
    return model, cur


def test_vacated_medium_is_terrain_not_a_blocker():
    """The ground that reclaims her cells is terrain, never a wall."""
    model, cur = _terrain_world()
    assert 8 in model.terrain_kinds
    # She bumps a wall — the floor sits under her target too, but
    # the medium is never credited as what stopped her.
    wall = {(min(r for r, _ in cur) - 1, c) for c in (2, 3, 4)}
    floor_cells = {(r, c) for r in range(2, 14) for c in range(0, 14)}
    model.note_action("up")
    model.observe(_scene([
        (_ME, set(cur)),                      # refused: unmoved
        (6, wall),
        (8, floor_cells - cur - wall),
    ]))
    assert model.affordance_of(6)[OUT_BLOCK] == 1
    assert model.affordance_of(8)[OUT_BLOCK] == 0


def test_terrain_does_not_forbid_simulated_moves():
    """A move through medium-covered cells still simulates."""
    model, cur = _terrain_world()
    state = model.start_state()
    assert state is not None
    nxt = model.simulate(state, "up")
    assert nxt is not None
    assert nxt.cells == frozenset((r - 1, c) for r, c in cur)


def test_unobserved_void_is_not_free_or_plannable():
    """Unclaimed pixels are not a map of where she can go.

    The empty margin around an observed medium is merely unseen
    space: marking it free invents routes through walls and voids.
    Crossing it is how its refusal is learned; the map must not
    pre-commit to its admission.
    """
    model, cur = _terrain_world()
    assert 8 in model.terrain_kinds
    floor = {(r, c) for r in range(2, 14) for c in range(14)}
    assert floor <= model.known_free
    assert (0, 0) not in model.known_free
    # Walk to the northern edge; the background beyond it is not a
    # route until the world has admitted or refused it.
    for _ in range(3):
        nxt = {(r - 1, c) for r, c in cur}
        model.note_action("up")
        model.observe(_scene([
            (_ME, nxt),
            (8, floor - nxt),
        ]))
        cur = nxt
    assert (1, 2) not in model.known_free
    assert model.simulate(model.start_state(), "up") is None


def test_refusal_without_entity_marks_space_not_things():
    """A failed move into empty space refuses the cells, not a kind."""
    model, cur = _terrain_world()
    # She pushes up into unclaimed space beyond the floor — no
    # entity is there to blame, the space itself refuses.
    floor_cells = {(r, c) for r in range(2, 14) for c in range(0, 14)}
    for _ in range(4):                       # walk to the floor's top edge
        nxt = {(r - 1, c) for r, c in cur}
        model.note_action("up")
        model.observe(_scene([
            (_ME, nxt),
            (8, floor_cells - nxt),
        ]))
        cur = nxt
    # A refused strip is a disjunction — *some* cell inside it
    # refused — so a multi-cell destination earns suspicion, not
    # condemnation. Convergence convicts: the second refusal of
    # the same cells is the wall showing under the strip again.
    for _ in range(2):
        model.note_action("up")
        model.observe(_scene([
            (_ME, set(cur)),                 # refused: off the floor
            (8, floor_cells - cur),
        ]))
    # The cells beyond the medium carry the refusal — no entity
    # was there to take the blame. Her own cells are excluded:
    # a cell she still occupies was never refused her.
    refused = {(r - 1, c) for r, c in cur} - floor_cells - cur
    assert refused and refused <= model.refused_cells
    assert refused <= model.known_blocked
    # And the planner cannot simulate through them.
    state = model.start_state()
    assert state is not None
    assert model.simulate(state, "up") is None


def test_refused_cells_do_not_reopen_as_free():
    """The map's free marking must not resurrect refused space."""
    model, cur = _terrain_world()
    floor_cells = {(r, c) for r in range(2, 14) for c in range(0, 14)}
    for _ in range(4):
        nxt = {(r - 1, c) for r, c in cur}
        model.note_action("up")
        model.observe(_scene([
            (_ME, nxt),
            (8, floor_cells - nxt),
        ]))
        cur = nxt
    # Two refusals — suspicion converges to conviction.
    for _ in range(2):
        model.note_action("up")
        model.observe(_scene([
            (_ME, set(cur)),
            (8, floor_cells - cur),
        ]))
    refused = {(r - 1, c) for r, c in cur} - floor_cells - cur
    # Another observation cycle: free-space marking runs again —
    # the refused cells must stay blocked.
    model.observe(_scene([
        (_ME, set(cur)),
        (8, floor_cells - cur),
    ]))
    assert refused <= model.known_blocked
    assert not refused & model.known_free


# ─── strides ────────────────────────────────────────────────


def _stride_world() -> tuple[WorldModel, set]:
    """A world where her learned "up" is a five-cell leap.

    ARC leaps move her several cells atomically — she never stands
    on the cells between takeoff and landing, yet a wall inside the
    leap still refuses it. Two observed moves teach the effect.
    """
    model = WorldModel()
    cur = {(r + 30, c) for r, c in _ME_CELLS}      # rows 30-32
    model.observe(_scene([(_ME, cur)], h=48, w=16))
    for _ in range(3):
        nxt = {(r - 5, c) for r, c in cur}
        model.note_action("up")
        model.observe(_scene([(_ME, nxt)], h=48, w=16))
        cur = nxt
    assert model.self_id is not None
    assert model.effect_of("up") == (-5, 0)
    return model, cur


def test_stride_simulates_over_open_path():
    """An unobstructed leap lands five rows up, as learned."""
    model, cur = _stride_world()                    # cur rows 20-22
    state = model.start_state()
    assert state is not None
    nxt = model.simulate(state, "up")
    assert nxt is not None
    assert nxt.cells == frozenset((r - 5, c) for r, c in cur)


def test_stride_bump_blames_the_wall_inside_the_leap():
    """The blocker inside a stride is convicted — not the landing."""
    model, cur = _stride_world()                    # cur rows 19-21
    # The wall sits inside the stride at row 17 — the landing
    # strip (rows 14-16) is open; the blocker lives mid-leap.
    wall = {(17, c) for c in (2, 3, 4)}
    model.note_action("up")
    model.observe(_scene([
        (_ME, set(cur)),                            # refused: unmoved
        (6, wall),
    ], h=48, w=16))
    assert model.affordance_of(6)[OUT_BLOCK] == 1
    assert wall <= model.known_blocked
    # And the model now refuses the leap in imagination, even
    # though the destination strip itself is open.
    state = model.start_state()
    assert state is not None
    assert model.simulate(state, "up") is None


def test_stride_bump_spares_proven_floor_in_the_leap():
    """Walked-on ground inside the path takes none of the blame."""
    model, cur = _stride_world()                    # cur rows 19-21
    # A color-7 patch inside the stride she has already crossed
    # without consequence — proven-open surface.
    pad = {(18, c) for c in (2, 3, 4)}
    model.affordances.setdefault(7, Counter())[OUT_NOTHING] += 1
    wall = {(17, c) for c in (2, 3, 4)}
    model.note_action("up")
    model.observe(_scene([
        (_ME, set(cur)),
        (6, wall),
        (7, pad),
    ], h=48, w=16))
    assert model.affordance_of(6)[OUT_BLOCK] == 1
    assert model.affordance_of(7)[OUT_BLOCK] == 0


def test_stride_crossed_ground_counts_as_covered():
    """Cells inside a completed leap are visited, never stood on."""
    model, _ = _stride_world()
    # The third leap ran under an owned self (24-26 → 19-21):
    # row 22 lay strictly inside it — passed over, never stood.
    assert model.visits[(22, 3)] >= 1
    assert (22, 3) in model.known_free
    # A row beyond every leap so far is still unvisited.
    assert model.visits[(18, 3)] == 0


def test_planner_no_longer_jumps_the_wall():
    """After a stride bump, planning refuses the same leap."""
    from frontal_lobe.world_planner import WorldPlanner

    model, cur = _stride_world()
    wall = {(17, c) for c in (2, 3, 4)}
    model.note_action("up")
    model.observe(_scene([
        (_ME, set(cur)),
        (6, wall),
    ], h=48, w=16))
    # Goal beyond the wall: "up" is the only learned move and it
    # now fails in simulation, so no plan exists.
    planner = WorldPlanner(model)
    assert planner.next_action({(5, 3)}, ["up"]) is None


def test_win_credits_the_cell_she_was_entering():
    """A goal consumed by the winning move is unreachable by contact.

    She strides onto the goal tile and the level ends — the contact
    is never observed, so credit has to come from where her pending
    move was carrying her. The floor under the destination takes no
    blame and no prize; only the goal kind does.
    """
    model, cur = _stride_world()                    # cur rows 19-21
    goal = {(14, 2), (14, 3), (14, 4),
            (15, 2), (15, 3), (15, 4),
            (16, 2), (16, 3), (16, 4)}
    model.observe(_scene([(_ME, set(cur)), (7, goal)], h=48, w=16))
    # The winning leap carries her onto the goal — the next frame
    # shows her there with the goal already gone.
    model.note_action("up")
    landing = {(r - 5, c) for r, c in cur}
    events = model.observe(
        _scene([(_ME, landing)], h=48, w=16), info={"win": True})
    assert model.affordance_of(7)[OUT_WIN] == 1
    assert any(e.kind == OUT_WIN and "goal" in e.text for e in events)


def test_refusal_releases_after_unexplained_contact():
    """A refusal verdict binds only within the epoch it was measured.

    Part of her state is hidden from her own sensors, and touching
    the unexplained may have rewritten it — so a verdict taken
    before that contact stops binding until re-proven. A real wall
    earns its certainty back by refusing again; the gate that was
    only state-wrong simply never does.
    """
    model, cur = _stride_world()        # me color-3 rows 19-21, up = (-5, 0)
    gate = {(14, 2), (14, 3), (14, 4),
            (15, 2), (15, 3), (15, 4),
            (16, 2), (16, 3), (16, 4)}  # inside the leap, not adjacent
    wall = {(15, 6), (15, 7), (16, 7)}  # color-7: a plain wall
    pad = {(25, 6), (25, 7)}            # color-1: the unexplained
    model.observe(_scene([
        (_ME, set(cur)), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    # Leap up — the gate inside the path refuses.
    model.note_action("up")
    model.observe(_scene([
        (_ME, set(cur)), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    # Shift over and leap up into the color-7 wall — refused too.
    moved = {(r, c + 4) for r, c in cur}
    model.observe(_scene([
        (_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    model.note_action("up")
    model.observe(_scene([
        (_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    sig = model.self_signature()
    gate_e = next(e for e in model.entities.values() if e.color == 6)
    wall_e = next(e for e in model.entities.values() if e.color == 7)
    assert not model.passable(gate_e, sig)
    assert not model.passable(wall_e, sig)
    # She walks onto the unexplained tile — the scene shows only
    # her there; the tracked pad still overlaps her footprint.
    stand = {(24, 6), (24, 7), (24, 8),
             (25, 6), (25, 7), (25, 8),
             (26, 6), (26, 7), (26, 8)}
    model.observe(_scene([
        (_ME, stand), (6, gate), (7, wall)], h=48, w=16))
    # Both old verdicts are stale now — her state may have changed
    # in ways nothing reported.
    assert model.passable(gate_e, sig)
    assert model.passable(wall_e, sig)
    # Re-measured in the new epoch the wall refuses again — its
    # verdict binds afresh; the un-retested gate stays eligible.
    model.observe(_scene([
        (_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    model.note_action("up")
    model.observe(_scene([
        (_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    assert not model.passable(wall_e, sig)
    assert model.passable(gate_e, sig)


def test_planner_targets_entire_body_not_only_anchor():
    from frontal_lobe.world_planner import WorldPlanner

    model = _world_with_self()
    model.action_effects["down"] = Counter({(1, 0): 3})
    model.action_effects["left"] = Counter({(0, -1): 3})
    plan = WorldPlanner(model).plan({(7, 4), (4, 0)}, ["down", "left"])
    assert plan is not None
    assert [action for action, _ in plan] == ["down"]


def test_retrial_prefers_stable_gates_over_moving_things():
    """A conditional wall is a measurable question; a mover is not.

    Hidden-state changes make old refusals worth retesting. If a
    mobile entity happens to sit nearer than the stable gate, aiming
    at its current cells just chases it — every step repaints the
    target. Static retrials get the route first; the mover can still
    be learned from when encountered, or pursued when nothing stable
    remains unanswered.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    model.action_effects["down"] = Counter({(5, 0): 3})
    model.action_tries["down"] = 3
    model.action_effects["right"] = Counter({(0, 3): 3})
    model.action_tries["right"] = 3
    sig = model.self_signature()
    assert sig is not None
    epoch = model._state_epoch
    mobile = model._spawn(PerceivedObject(
        index=90, color=8, cells=frozenset({(5, 6)})
    ))
    mobile.velocity = (1.0, 0.0)
    mobile.blocked_states.add(sig)
    mobile.blocked_epochs[sig] = epoch
    stable = model._spawn(PerceivedObject(
        index=91, color=9, cells=frozenset({(10, 3)})
    ))
    stable.blocked_states.add(sig)
    stable.blocked_epochs[sig] = epoch
    other = model._spawn(PerceivedObject(
        index=92, color=10, cells=frozenset({(12, 6)})
    ))
    other.blocked_states.add(sig)
    other.blocked_epochs[sig] = epoch
    model._state_epoch += 1

    policy = WorldPolicy(model, seed=0)
    asked: list[set[tuple[int, int]]] = []
    original = policy._planner.next_action
    def spy(goal_cells, available):
        asked.append(set(goal_cells))
        return original(goal_cells, available)
    policy._planner.next_action = spy
    choice = policy.choose(["right", "down"])
    assert choice is not None
    assert choice["source"] == "reprobe"
    assert choice["action"] == "down"
    assert asked[-1] in (set(stable.cells), set(other.cells))
    policy._bid_reprobe(["right", "down"])
    assert asked[-1] == asked[-2]


def test_unembodied_policy_cycles_actions_to_recover_self():
    """A lost body is found by balanced motor probing, not habit.

    Lifelong try counts can stay skewed from the previous level, and
    the direction the old body favored may be the wall the new body
    cannot cross. While no self is identified, each available command
    must get a contingent trial before any is retried — appearance
    alone cannot say which visible thing is hers.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = WorldModel()
    model.observe(_scene([
        (_ME, set(_ME_CELLS)),
        (7, {(10, 10), (10, 11)}),
    ]))
    model.action_tries.update({"left": 80, "down": 20, "right": 3})
    policy = WorldPolicy(model, seed=0)
    available = ["left", "right", "up", "down"]
    chosen = []
    for _ in available:
        bid = policy.choose(available)
        assert bid is not None
        assert bid["source"] == "reembody"
        chosen.append(bid["action"])
    assert set(chosen) == set(available)


def test_repeated_observed_noop_is_not_replanned():
    """A failed transition is evidence even when no wall is named.

    Entity blame can be ambiguous — substrate, occlusion, and exact
    swept ownership can all hide the blocker — but two identical
    state-actions that left her unmoved are still a measured failed
    transition. The forward model must stop spending moves on it.
    """
    model = _world_with_self()
    state = model.start_state()
    assert state is not None
    effect = model.effect_of("up")
    assert effect is not None
    model.known_free.update(model._swept(state.cells, effect))
    assert model.simulate(state, "up") is not None
    model._move_failures[("up", state.cells, state.signature)] = (
        2,
        model.tick,
    )
    assert model.simulate(state, "up") is None
    model.tick += 20
    assert model.simulate(state, "up") is not None


@pytest.mark.parametrize("effect", [(-5, 0), (10, 0), (0, -3), (0, 12)])
def test_simulation_does_not_plan_body_outside_observed_grid(effect):
    model = _world_with_self()
    model.action_effects["move"] = Counter({effect: 3})
    assert model.simulate(model.start_state(), "move") is None


def test_new_world_forgets_interface_but_retry_retains_it():
    model = _world_with_self()
    model.learn_action_label("press", "coordinate")
    model.learn_action_label("up", "up")
    model.reset(keep_learning=True)
    assert model.coordinate_actions == {"press"}
    assert model.action_priors["up"] == (-1, 0)
    model.reset(keep_learning=False)
    assert not model.coordinate_actions
    assert not model.action_priors
    assert not model.action_labels


def test_new_game_clears_local_physics_but_preserves_cognitive_systems(tmp_path):
    from mind import Mind

    subject = Mind(str(tmp_path / "genesis.sock"), offline=True)
    network = subject.brain.frontal.network
    competence = subject.brain.frontal.task_competence
    subject._begin_world("first", 0)
    subject.world_model.action_effects["1"] = Counter({(0, 5): 3})
    subject.world_model.learn_action_label("6", "coordinate")
    subject._world_policy._explore_goal = (5, 5)
    subject._world_levels = 4
    subject._world_deaths = 2
    subject._begin_world("first", 4)
    assert subject.world_model.effect_of("1") == (0, 5)
    subject._begin_world("second", 0)
    assert subject.world_model.effect_of("1") is None
    assert not subject.world_model.coordinate_actions
    assert subject._world_policy._explore_goal is None
    assert subject._world_levels == subject._world_deaths == 0
    assert subject.brain.frontal.network is network
    assert subject.brain.frontal.task_competence is competence


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
