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

from parietal_lobe.grid import Grid
from parietal_lobe.scene import PerceivedObject, Scene, perceive
from parietal_lobe.world_model import (
    OUT_BLOCK,
    OUT_COLLECT,
    OUT_HAZARD,
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
    events = model.observe(
        _scene([
            (_ME, set(_ME_CELLS)),
            (7, goal),
            (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
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
    model.note_action("up")
    model.observe(_scene([
        (_ME, set(cur)),                     # refused: off the floor
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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
