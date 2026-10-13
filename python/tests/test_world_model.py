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
    _MOVE_FAILURE_COOLDOWN,
    OUT_BLOCK,
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_NOTHING,
    OUT_TRANSFORM,
    OUT_WIN,
    SimState,
    WorldModel,
)


def _scene(regions: list[tuple[int, set[tuple[int, int]]]], h: int = 16, w: int = 16) -> Scene:
    """Paint color → cells regions on a 0-background grid and perceive."""
    grid = [[0] * w for _ in range(h)]
    for color, cells in regions:
        for r, c in cells:
            grid[r][c] = color
    return perceive(Grid.from_lists(grid), background=0)


_ME = 3
_ME_CELLS = {(4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)}  # 3x3 block


def _world_with_self() -> WorldModel:
    """A world where the color-3 block has been identified as her."""
    model = WorldModel()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}), (9, {(12, 10), (12, 11)})]
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
    model.observe(
        _scene(
            [
                (_ME, occluded),
                (7, occluder),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )

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
        model.observe(
            _scene(
                [
                    (_ME, {(4, 2), (4, 3), (4, 4)}),  # top row visible
                    (7, {(5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)}),
                    (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                    (9, {(12, 10), (12, 11)}),
                ]
            )
        )
    assert model.action_noops["side"] == 3
    assert model.effect_of("side") == (0, 0)


def test_prediction_events_name_the_real_effect():
    """An honest mismatch reports the learned vector, not jitter."""
    model = _world_with_self()
    # "up" expects (-2, 0); hold her still and see the surprise.
    model.note_action("up")
    events = model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
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
    a0 = {(4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)}
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

    others = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}), (9, {(12, 10), (12, 11)})]
    # "up" predicts (-2, 0); the world answers with a different
    # displacement each trial — every one maximal error, and no
    # rival delta accumulates enough votes to take over the mode.
    for dr, dc in [(0, -2), (1, -2), (0, -1), (1, -1)]:
        model.note_action("up")
        model.observe(_scene([(_ME, {(r + dr, c + dc) for r, c in _ME_CELLS}), *others]))
        model.observe(_scene([(_ME, set(_ME_CELLS)), *others]))

    assert model.motor_comparator.error_of("up") > 0.5
    assert not model.action_understood("up")


def test_explained_refusal_is_confirmation_not_miscalibration():
    """A bump the model predicted is confirmation, not surprise.

    The cerebellum used to grade observed-versus-modal misses, so a
    command learned physics already refuses from this state read as
    a maximal error every time — and the estimate only accumulated,
    never recovered, because matches never fed it. The prediction
    graded is the contextual one: a refused command expects (0,0),
    and every confidently-measured trial feeds the EMA.
    """
    from cerebellum.motor import SensorimotorLayer

    model = _world_with_self()
    model.motor_comparator = SensorimotorLayer()
    others = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}), (9, {(12, 10), (12, 11)})]
    # The cells "up" would sweep carry a refusal verdict — the mark
    # a real bump leaves (unclaimed free-looking cells get re-freed
    # by the map updater before the next observe).
    blocked = {(r, c) for r in (2, 3) for c in (2, 3, 4)}
    model.refused_cells.update(blocked)
    model.known_free.difference_update(blocked)

    model.note_action("up")
    model.observe(_scene([(_ME, set(_ME_CELLS)), *others]))
    # observed (0,0) matched the contextual prediction: nothing was
    # surprising, so nothing counted against calibration.
    assert model.motor_comparator.error_of("up") == 0.0

    # A genuine surprise — "up" lands her left instead.
    model.note_action("up")
    model.observe(_scene([(_ME, {(r, c - 2) for r, c in _ME_CELLS}), *others]))
    surprised = model.motor_comparator.error_of("up")
    assert surprised > 0

    # Back in place, another predicted bump: a confirmation is a
    # trial too, and it pays the estimate down.
    model.observe(_scene([(_ME, set(_ME_CELLS)), *others]))
    model.note_action("up")
    model.observe(_scene([(_ME, set(_ME_CELLS)), *others]))
    assert model.motor_comparator.error_of("up") < surprised


# ─── terminal outcomes ──────────────────────────────────────


def test_win_credits_last_contacts_and_resets():
    """info['win'] credits whatever she touched when the world ended."""
    model = _world_with_self()
    # The door is a tracked entity — she saw it, moved into contact.
    goal = {(4, 5), (4, 6), (5, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, goal),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.known_free.add((15, 15))
    events = model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, goal),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (8, {(15, 15)}),
                (9, {(12, 10), (12, 11)}),
            ]
        ),
        info={"win": True},
    )
    assert model.affordance_of(7)[OUT_WIN] == 1
    assert any(e.kind == OUT_WIN and "goal" in e.text for e in events)
    # The world that ended is forgotten — everything tracked now
    # was first seen this tick — while learned physics persist.
    assert model.entities
    assert all(e.first_seen == model.tick for e in model.entities.values())
    assert model.effect_of("up") == (-2, 0)
    # A new level is not a retry: cell-map verdicts from the solved
    # layout must not become phantom walls in this one.
    assert (15, 15) not in model.known_free


def test_terminal_credit_is_an_object_not_the_medium():
    """A win credits the goal kind once — never floor or walls.

    The winning contact can leave her pressed against terrain, an
    oversized wall, and two goal objects of one kind all at once.
    Only the object-sized non-terrain kind caused the ending; the
    medium and substrate were where she stood, and two witnesses of
    one kind are still one verdict.
    """
    model, cur = _terrain_world()  # me rows 5-7, color 8 is terrain
    floor = {(r, c) for r in range(2, 14) for c in range(14)}
    wall = {(r, 5) for r in range(2, 12)}  # 10 cells > her 9
    goals = {(4, 2), (4, 3), (8, 2), (8, 3)}  # two color-7 objects
    model.observe(
        _scene(
            [
                (8, floor - cur - wall - goals),
                (6, wall),
                (7, goals),
                (_ME, set(cur)),
            ]
        )
    )
    events = model.observe(_scene([(_ME, set(cur))]), info={"win": True})

    assert model.affordance_of(7)[OUT_WIN] == 1
    assert model.affordance_of(8)[OUT_WIN] == 0
    assert model.affordance_of(6)[OUT_WIN] == 0
    assert sum(e.kind == OUT_WIN and "goal" in e.text for e in events) == 1


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
    model.observe(
        _scene(
            [
                (_ME, {(4, 2), (4, 3), (4, 4)}),
                (7, {(5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)}),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    # A brand-new world in brand-new colors — nothing matches.
    events = model.observe(
        _scene(
            [
                (2, {(0, 5), (0, 6)}),
                (11, {(8, 8), (9, 8), (9, 9)}),
                (13, {(14, 1), (14, 2), (14, 3)}),
            ]
        )
    )
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
    model = _world_with_self()  # learned "up" is (-2, 0)
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, {(2, 2), (2, 3)}),  # inside the swept strip
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")
    events = model.observe(
        _scene(
            [
                (2, {(0, 5), (0, 6)}),
                (11, {(8, 8), (9, 8), (9, 9)}),
            ]
        )
    )
    assert any(e.kind == "reset" for e in events)
    assert model.death_suspects[7] == 1
    assert any("path" in e.text for e in events if e.kind == "suspect")


def test_losing_only_her_body_suspects_the_fatal_path():
    """A persistent world can still kill the tracked body.

    ARC deaths often respawn the avatar while the maze itself keeps
    continuity. That is not a wholesale reset, but the kind sitting
    inside the failed move is still causal evidence — exactly the
    suspicion a terminal reset would have recorded.
    """
    model = _world_with_self()  # learned "up" is (-2, 0)
    hazard = {(2, 2), (2, 3)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")
    events = model.observe(
        _scene(
            [
                (7, hazard),  # she is gone, world stays
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    assert not any(e.kind == "reset" for e in events)
    assert model.death_suspects[7] == 1
    assert any("path" in e.text for e in events if e.kind == "suspect")


def test_fatal_path_counts_a_killer_that_vanishes_with_her():
    """A hazard that disappears on the fatal tick still testifies.

    A projectile, gate, or consuming tile can be gone from the same
    frame that lacks her body. The cause is not exonerated by its own
    disappearance: if it occupied the path of the pending move, it
    was there when the ending was caused.
    """
    model = _world_with_self()  # learned "up" is (-2, 0)
    hazard = {(2, 2), (2, 3)}
    others = [
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]
    model.observe(_scene([(_ME, set(_ME_CELLS)), (7, hazard), *others]))
    model.note_action("up")
    events = model.observe(_scene(others))
    assert model.death_suspects[7] == 1
    assert any("path" in e.text for e in events if e.kind == "suspect")


def test_fatal_contact_counts_a_killer_that_vanishes_with_her():
    """A disappearing contact is still terminal evidence."""
    model = _world_with_self()
    hazard = {(4, 5), (5, 5)}
    others = [
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]
    model.observe(_scene([(_ME, set(_ME_CELLS)), (7, hazard), *others]))
    events = model.observe(_scene(others))
    assert model.death_suspects[7] == 1
    assert any("touching me" in e.text for e in events if e.kind == "suspect")


def test_terminal_suspicion_counts_each_kind_once():
    """Two touched suspects of one kind are one ending's evidence."""
    model = _world_with_self()
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, {(4, 5), (4, 6)}),
                (7, {(7, 2), (7, 3)}),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    events = model.suspect_terminal()
    assert model.death_suspects[7] == 1
    assert sum(e.kind == "suspect" for e in events) == 1


def test_vanished_body_part_is_not_a_terminal_suspect():
    """A composite avatar's pieces die together; they do not kill her.

    The lower half can vanish in the same frame as the core while the
    rest of the maze persists. Marking it gone must not evict it from
    the self cluster before terminal suspicion runs — otherwise every
    death teaches her that her own color-9 part is a hazard.
    """
    model = _composite_world()
    part = next(e for e in model.entities.values() if e.part_of_self)
    assert part.color == 9
    others = [
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]

    model.note_action("down")
    events = model.observe(_scene(others))

    assert part.gone
    assert model.death_suspects[9] == 0
    assert model.death_signatures[(part.color, part.shape)] == 0
    assert not any(e.kind == "suspect" and e.other_id == part.id for e in events)


def test_terminal_suspicion_uses_the_pending_action_without_a_frame():
    """A declared GAME_OVER carries no new frame; the last command does."""
    model = _world_with_self()
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, {(2, 2)}),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")

    events = model.suspect_terminal()

    assert model.death_suspects[7] == 1
    assert any("path" in event.text for event in events)


def test_refused_self_sends_the_state_changer_ahead_of_the_gate():
    """When her signature is refused, producers outrank measurers.

    Re-walking a gate the current body cannot pass measures nothing
    new; the thing that rewrites her manufactures the only state
    that can change the answer. Producer-class retrials rank first
    while her own signature stands refused — distance decides only
    among questions of the same class.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    for action, delta in (("down", (2, 0)), ("right", (0, 2))):
        model.action_effects.setdefault(action, Counter())[delta] += 2
        model.action_tries[action] += 2
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (9, {(4, 6), (4, 7)}),  # the gate — near
                (7, {(12, 13), (12, 14)}),  # the producer — far
                (8, {(12, 1), (12, 2)}),  # the thing that refused this sig
            ]
        )
    )
    sig = model.self_signature()
    gate = next(e for e in model.entities.values() if e.color == 9)
    gate.blocked_states.add((("other",),))  # it refused a former self
    model.blocked_kinds.setdefault(8, set()).add(sig)  # this sig refused
    model._state_changers.add(7)
    model.known_free.update((r, c) for r in range(16) for c in range(16))

    bid = WorldPolicy(model, seed=0)._bid_reprobe(["down", "right"])

    assert bid is not None and bid["action"] == "down"


def test_empty_resource_meter_explains_the_ending():
    """Starvation is its own cause; contact is just where she stood.

    When a fuel gauge visibly ran out, the ending has an account
    already — crediting whatever she was touching would convict an
    innocent thing and poison the channels that feed her.
    """
    model = _world_with_self()
    model.affordances.setdefault(11, Counter())[OUT_COLLECT] += 1

    def frame(bar: int) -> Scene:
        return _scene(
            [
                (_ME, set(_ME_CELLS)),
                (11, {(15, c) for c in range(bar)}),  # the meter
                (11, {(8, 8), (8, 9)}),  # the pickup
                (7, {(4, 5), (4, 6)}),  # touching, innocent
                (9, {(12, 10), (12, 11)}),
            ]
        )

    for n in (16, 14, 12, 10, 8, 4, 1):
        model.observe(frame(n))
    assert model.resource_levels()[11] <= 0.1

    events = model.suspect_terminal()

    assert [e.kind for e in events] == ["depleted"]
    assert model.death_suspects[7] == 0
    assert not model.death_signatures


def test_fatal_suspicion_marks_the_shape_not_the_kind():
    """A same-color goal need not die for a same-color hazard.

    One fatal contact is too little evidence to condemn a whole color,
    but it is enough to avoid the exact silhouette that was there. A
    different color-7 object remains approachably unknown.
    """
    model = _world_with_self()
    hazard = {(4, 5), (4, 6)}
    decoy = {(10, 10)}
    sig = model.self_signature()
    assert sig is not None
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (7, decoy),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    events = model.observe(
        _scene(
            [
                (7, hazard),
                (7, decoy),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    assert any(e.kind == "suspect" for e in events)
    assert not model.hazard_kind(7)

    entities = {tuple(sorted(e.cells)): e for e in model.entities.values()}
    killer = entities[tuple(sorted(hazard))]
    innocent = entities[tuple(sorted(decoy))]
    assert model.hazard_entity(killer)
    assert not model.hazard_entity(innocent)
    # Adjacency alone did not acquit it; only occupying the thing is
    # evidence her body may enter it safely.
    assert model.signature_contacts[(7, killer.shape)] == 0
    assert not model.passable(killer, sig)
    assert model.passable(innocent, sig)


def test_partial_turnover_is_not_a_reset():
    """Ordinary motion — some entities persist — is not a reset."""
    model = _world_with_self()
    events = model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),  # wall persists
                (8, {(13, 13), (14, 13)}),  # gem replaced
            ]
        )
    )
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


def test_suspected_hazard_forbids_contact_not_just_occupation():
    """A lethal contact is unsafe from beside the thing too."""
    model = _world_with_self()
    hazard = {(1, 4), (1, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    model.known_free.update(model._swept(state.cells, (-2, 0)))
    # The stride does not enter the hazard's cells; it passes beside
    # them. That is still the fatal contact the ending measured.
    assert model.action_fatal(state, "up")
    assert model.simulate(state, "up") is None


def test_stalled_pursuit_yields_to_unanswered_questions():
    """A route that stops closing on the goal loses gain to probing.

    Pursuit is a hypothesis: while each planned act still shortens
    the distance to something valued it keeps its 0.8 gain; when
    several planned acts buy no approach the bid decays below the
    question drives and an unprobed entity takes the turn.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    # Teach "right" (+0,+2) outright so the planner has a usable
    # stride and calibration does not outbid the plan.
    model.action_effects.setdefault("right", Counter())[(0, 2)] += 2
    model.action_tries["right"] += 2
    goal_cells = {(4, 14), (5, 14)}
    unknown_cells = {(4, 10), (5, 10)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (9, goal_cells),
                (6, unknown_cells),
            ]
        )
    )
    model.affordances.setdefault(9, Counter())[OUT_WIN] += 1
    model.known_free.update((r, c) for r in range(16) for c in range(16))
    policy = WorldPolicy(model, seed=0)

    first = policy.choose(["right"])
    assert first is not None
    assert first["source"] == "planned" and first["weight"] == 0.8

    # No observation ever moves her — every planned act repeats the
    # same position, so pursuit stalls and the probe earns the turn.
    last = first
    for _ in range(8):
        last = policy.choose(["right"])
    assert last is not None
    assert last["source"] == "probe" and last["weight"] == 0.5


def test_pursuit_progress_follows_the_route_not_the_nearest_decoy():
    """Closing on the real goal is progress even past a decoy.

    The stall ledger used to measure distance to the nearest valued
    cell. When the nearest one sits behind a wall and the planner
    correctly routes to a farther reachable goal, every approach
    step read as no-progress and the bid decayed mid-route. The
    ledger follows the route's own destination instead.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    for action, delta in (("down", (2, 0)), ("up", (-2, 0))):
        model.action_effects.setdefault(action, Counter())[delta] += 2
        model.action_tries[action] += 2
    # A map-evidenced wall column, not an entity: unclaimed blocked
    # cells refuse the sweep outright in simulation.
    wall = {(r, 8) for r in range(30)}
    decoy = {(4, 10), (5, 10)}
    goal = {(24, 2), (24, 3)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (9, decoy),
                (9, goal),
            ],
            h=30,
        )
    )
    model.affordances.setdefault(9, Counter())[OUT_WIN] += 1
    model.known_free.update((r, c) for r in range(30) for c in range(16))
    model.known_free.difference_update(wall | decoy | goal)
    model.known_blocked.update(wall)
    policy = WorldPolicy(model, seed=0)

    bid = policy._bid_plan(["down", "up"])
    assert bid is not None and bid["action"] == "down"
    # The route commits to the reachable goal — the decoy stays
    # nearer by raw distance but the planner does not route to it.
    assert min(c for _, c in bid["goal"]) < 8
    policy._track_pursuit(bid)

    # She strides toward the goal while the decoy — still nearer
    # than the goal — recedes behind her. Under the nearest-cell
    # metric every one of these steps counted as no-progress.
    for _ in range(3):
        moved = {(r + 4, c) for r, c in model.self_footprint()}
        model.observe(
            _scene(
                [
                    (_ME, moved),
                    (9, decoy),
                    (9, goal),
                ],
                h=30,
            )
        )
        nxt = policy._bid_plan(["down", "up"])
        assert nxt is not None
        policy._track_pursuit(nxt)
    assert policy._plan_stall == 0


def test_detour_route_shortening_counts_as_progress():
    """A corridor that wraps a wall must not read as stalled.

    The straight line to the goal grows while she walks the detour
    leg, so crow-fly distance alone would mark a working route
    stalled and hand her remaining fuel to wandering. The plan's own
    remaining length is the honest measure — it shrinks every turn
    the route actually runs.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    policy = WorldPolicy(model, seed=0)
    goal = {(24, 2), (24, 3)}
    # The first bid seeds the ledger; the same distance then repeats
    # (the detour is not yet closing) while the route itself
    # shortens each step — progress by the only measure that can see
    # around walls.
    for route in (20, 19, 18, 17, 16, 15, 14, 13):
        policy._track_pursuit({"source": "planned", "goal": goal, "route": route})
    assert policy._plan_stall == 0
    # And a route that stops shortening still stalls — the counter's
    # evidence, not a leash.
    for _ in range(3):
        policy._track_pursuit({"source": "planned", "goal": goal, "route": 13})
    assert policy._plan_stall == 3


def test_scarce_resource_meter_narrows_pursuit_to_its_kind():
    """A depleting collect-kind readout is a fuel gauge.

    The meter shares its kind with the pickups that refill it, so
    its shrinking is the bodily evidence of running out. While it
    reads full the whole valued pool stays in pursuit; run low,
    only the depleted kind remains a destination — a starving
    animal does not finish the puzzle before it eats.
    """
    from basal_ganglia.world_policy import _LOW_RESERVE, WorldPolicy

    model = _world_with_self()
    model.affordances.setdefault(11, Counter())[OUT_COLLECT] += 1
    model.affordances.setdefault(9, Counter())[OUT_WIN] += 1
    policy = WorldPolicy(model, seed=0)

    def frame(bar: int) -> Scene:
        return _scene(
            [
                (_ME, set(_ME_CELLS)),
                (11, {(15, c) for c in range(bar)}),  # the meter
                (11, {(8, 8), (8, 9)}),  # the pickup
                (9, {(12, 10), (12, 11)}),  # a goal-kind
            ]
        )

    # A readout earns its label by churning on schedule; the first
    # sightings are just a shrinking edge thing.
    for n in (8, 7, 6, 5, 4):
        model.observe(frame(n))
    assert not model.resource_levels()
    valued = policy._valued_cells()
    assert (8, 8) in valued and (12, 10) in valued

    model.observe(frame(3))
    assert model.resource_levels()[11] < _LOW_RESERVE
    valued = policy._valued_cells()
    assert (8, 8) in valued and (8, 9) in valued
    assert (12, 10) not in valued and (12, 11) not in valued


def test_full_resource_meter_spares_the_pickup():
    """A topped-off gauge makes its refill a null errand.

    Walking to a pickup the meter cannot absorb spends the reserve
    to gain nothing — the pickup would vanish while the walk still
    costs fuel. Pursuit leaves it alone and keeps what can still
    be won.
    """
    from basal_ganglia.world_policy import _LOW_RESERVE, WorldPolicy

    model = _world_with_self()
    model.affordances.setdefault(11, Counter())[OUT_COLLECT] += 1
    model.affordances.setdefault(9, Counter())[OUT_WIN] += 1
    policy = WorldPolicy(model, seed=0)

    def frame(bar: int) -> Scene:
        return _scene(
            [
                (_ME, set(_ME_CELLS)),
                (11, {(15, c) for c in range(bar)}),  # the meter
                (11, {(8, 8), (8, 9)}),  # the pickup
                (9, {(12, 10), (12, 11)}),  # a goal-kind
            ]
        )

    # The gauge earns its label by churning; a refill then restores
    # it to its observed peak — the reading a just-spawned or just-
    # refilled body sees.
    for n in (8, 7, 6, 5, 6, 7, 8):
        model.observe(frame(n))
    assert model.resource_levels()[11] >= _LOW_RESERVE

    valued = policy._valued_cells()
    assert (8, 8) not in valued and (8, 9) not in valued
    assert (12, 10) in valued and (12, 11) in valued


def test_exported_learning_restores_the_learned_world():
    """Practice should compound — learned physics survive a fresh mind.

    The export carries exactly what keep_learning preserves; a
    brand-new model that imports it reads the same verdicts the
    earner wrote — affordances, refused kinds, transforms, meter
    scales — as JSON-safe data end to end.
    """
    import json

    model = _world_with_self()
    sig = ((7, ((0, 0), (0, 1))),)
    model.affordances.setdefault(11, Counter())[OUT_COLLECT] += 2
    model.affordances.setdefault(9, Counter())[OUT_WIN] += 1
    model.action_effects.setdefault("down", Counter())[(2, 0)] += 4
    model.action_tries["down"] += 4
    model.blocked_kinds.setdefault(9, set()).add(sig)
    model.transforms.setdefault(1, {}).setdefault((7, ((0, 0), (0, 1))), Counter())[
        (9, ((1, 1), (1, 2)))
    ] += 1
    model._state_changers.add(1)
    model._kind_peak[11] = 84
    model.terrain_kinds.add(5)
    model.death_suspects[8] += 1
    # A measured refusal on a former board's cells: the count says it
    # was earned, but the stamp belongs to a former clock — on this
    # one it must read as long-ago, not bind the new board.
    cells = frozenset({(4, 4), (4, 5)})
    model._move_failures[("down", cells, sig)] = (3, 500)
    model.tick = 1

    restored = WorldModel()
    assert restored.import_learning(json.loads(json.dumps(model.export_learning())))
    assert restored.affordances[11][OUT_COLLECT] == 2
    assert restored.action_effects["down"][(2, 0)] == 4
    # A refusal measured in a former context is a prior to re-ask,
    # never a verdict: the membership survives (it arms the hunt
    # for what rewrites her) while its stamp sits behind the
    # current epoch, so it cannot exclude contact unmeasured.
    assert sig in restored.blocked_kinds[9]
    assert restored._blocked_epoch[(9, sig)] < restored._state_epoch
    from types import SimpleNamespace

    entity = SimpleNamespace(color=9, open_states=set(), blocked_states=set(), blocked_epochs={})
    assert not restored.state_refused(entity, sig)
    assert restored.transforms[1][(7, ((0, 0), (0, 1)))][(9, ((1, 1), (1, 2)))] == 1
    assert 1 in restored._state_changers
    assert restored._kind_peak[11] == 84
    # Terrain is a map verdict, not kind physics — the color that was
    # floor on the old board may be wall on this one, and it has no
    # epoch to re-ask. It re-derives from the first cells she
    # vacates, so the import leaves it unlearned.
    assert 5 not in restored.terrain_kinds
    assert restored.death_suspects[8] == 1
    # The imported failure keeps its count — the move was measured —
    # but its stamp sits a cooldown behind this clock, so the cell it
    # condemned on the old board cannot refuse a route here until it
    # is re-measured.
    count, last = restored._move_failures[("down", cells, sig)]
    assert count == 3
    assert restored.tick - last >= _MOVE_FAILURE_COOLDOWN
    assert not restored.import_learning({"version": 99})
    assert not restored.import_learning(None)


def test_remembered_body_is_recognized_not_rediscovered():
    """A familiar body confirms on one obedient answer.

    Discovery prices "which thing is me?" for a scene full of
    strangers; recognition only asks "is it still mine?". After a
    self has been named, its color rides the exported learning —
    a fresh model that sees the remembered color answer one
    action names it, while a body that does *not* match still has
    to earn the full contingency count.
    """
    import json

    model = _world_with_self()
    assert model._self_prior_color == _ME

    restored = WorldModel()
    assert restored.import_learning(json.loads(json.dumps(model.export_learning())))
    assert restored._self_prior_color == _ME

    # One contingent move by the remembered color names it.
    restored.observe(_scene([(_ME, _ME_CELLS), (7, {(8, 8), (8, 9)})]))
    restored.note_action("up")
    restored.observe(_scene([(_ME, {(r - 2, c) for r, c in _ME_CELLS}), (7, {(8, 8), (8, 9)})]))
    me = restored.self_entity
    assert me is not None and me.color == _ME

    # A body that does not match the prior is discovery, not
    # recognition: one answer is not enough.
    novel = WorldModel()
    novel._self_prior_color = 4
    novel.observe(_scene([(_ME, _ME_CELLS)]))
    novel.note_action("up")
    novel.observe(_scene([(_ME, {(r - 2, c) for r, c in _ME_CELLS})]))
    assert novel.self_entity is None
    novel.note_action("up")
    novel.observe(_scene([(_ME, {(r - 4, c) for r, c in _ME_CELLS})]))
    assert novel.self_entity is not None


def test_probe_commits_to_its_question_until_answered():
    """Two questions on opposite headings must not starve each other.

    Recomputing the nearest unanswered thing every step lets a newly
    nearer question steal the bid each turn — she paces between them
    and answers neither. The entity under test holds until it is
    answered, gone, or unreachable.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    for action, delta in (("up", (-2, 0)), ("down", (2, 0))):
        model.action_effects.setdefault(action, Counter())[delta] += 2
        model.action_tries[action] += 2
    # Everything already answered except the two questions.
    model.affordances.setdefault(5, Counter())[OUT_NOTHING] += 1
    model.affordances.setdefault(9, Counter())[OUT_NOTHING] += 1
    north = {(2, 2), (2, 3)}
    south = {(10, 2), (10, 3)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
                (6, north),
                (7, south),
            ]
        )
    )
    model.known_free.update((r, c) for r in range(16) for c in range(16))
    policy = WorldPolicy(model, seed=0)

    bid = policy._probe_unknown(["up", "down"])
    assert bid is not None and bid["action"] == "up"
    north_id = policy._probe_entity
    assert north_id is not None

    # The southern question moves adjacent — strictly nearer than the
    # committed one. Recomputing afresh would flip the bid; the held
    # question keeps it.
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
                (6, north),
                (7, {(7, 2), (7, 3)}),
            ]
        )
    )
    bid = policy._probe_unknown(["up", "down"])
    assert bid is not None and bid["action"] == "up"
    assert policy._probe_entity == north_id


def test_frontier_grows_toward_unanswered_questions():
    """Coverage is steered by what contact has not yet answered.

    Uniform nearest-first exploration spends a bounded budget on
    anonymous floor; while an untouched thing exists, the frontier
    cells on the way to it outrank nearer cells that lead nowhere.
    """
    model = _world_with_self()
    wall = {(r, 8) for r in range(13)}
    question = {(10, 11), (10, 12)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (5, wall),
                (7, question),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.known_free.update((r, c) for r in range(16) for c in range(16))
    model.known_free.difference_update(wall | question)
    from basal_ganglia.world_policy import WorldPolicy

    targets = WorldPolicy(model, seed=0)._frontier_targets()

    assert targets
    # Everything past column 8 is behind the wall's gap; a cell
    # chosen toward the questions there beats closer floor that
    # leads away from them.
    assert targets[0][1] > 8


def test_frontier_is_nearest_first_without_questions():
    """With nothing left unexplained, coverage is pure distance."""
    model = _world_with_self()
    model.observe(_scene([(_ME, set(_ME_CELLS))]))
    model.known_free.update((r, c) for r in range(16) for c in range(16))
    from basal_ganglia.world_policy import WorldPolicy

    targets = WorldPolicy(model, seed=0)._frontier_targets()

    assert targets
    footprint = model.self_footprint()
    for cell in targets:
        assert min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in footprint) <= 2


def test_policy_uses_a_forced_move_when_every_offered_act_is_fatal():
    """In a mandatory-action world, no answer is the only sure loss."""
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    hazard = {(1, 4), (1, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    model.death_signatures[(killer.color, killer.shape)] += 1
    choice = WorldPolicy(model, seed=0).choose(["up"])
    assert choice is not None
    assert choice["source"] == "forced"
    assert choice["action"] == "up"


def test_policy_uses_an_unbid_safe_escape_when_bids_are_fatal(
    monkeypatch: pytest.MonkeyPatch,
):
    """A lethal winning bid must not hide the one safe offered move."""
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    model.action_effects["left"] = Counter({(0, -2): 3})
    hazard = {(1, 4), (1, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    model.death_signatures[(killer.color, killer.shape)] += 1
    policy = WorldPolicy(model, seed=0)
    for channel in (
        "_bid_reembody",
        "_bid_calibrate",
        "_bid_plan",
        "_bid_board",
        "_bid_intention",
        "_bid_probe",
        "_bid_reprobe",
        "_bid_frontier",
    ):
        monkeypatch.setattr(policy, channel, lambda available: None)
    monkeypatch.setattr(
        policy,
        "_bid_forage",
        lambda available: {
            "action": "up",
            "cell": None,
            "source": "forage",
            "weight": 0.25,
        },
    )
    choice = policy.choose(["up", "left"])
    assert choice is not None
    assert choice["source"] == "evade"
    assert choice["action"] == "left"


def test_policy_uses_an_unbid_route_when_bids_are_measured_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    """Known-failure bids do not get kept as a meaningless fallback."""
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    model.action_effects["left"] = Counter({(0, -2): 3})
    policy = WorldPolicy(model, seed=0)
    for channel in (
        "_bid_reembody",
        "_bid_calibrate",
        "_bid_plan",
        "_bid_board",
        "_bid_intention",
        "_bid_probe",
        "_bid_reprobe",
        "_bid_frontier",
    ):
        monkeypatch.setattr(policy, channel, lambda available: None)
    monkeypatch.setattr(
        policy,
        "_bid_forage",
        lambda available: {
            "action": "up",
            "cell": None,
            "source": "forage",
            "weight": 0.25,
        },
    )
    monkeypatch.setattr(
        policy,
        "_known_failure",
        lambda start, action: action == "up",
    )
    choice = policy.choose(["up", "left"])
    assert choice is not None
    assert choice["source"] == "evade"
    assert choice["action"] == "left"


def test_policy_waits_with_a_safe_blocked_move_when_trapped(
    monkeypatch: pytest.MonkeyPatch,
):
    """A nonfatal bump is a legal wait; a fatal fallback is not."""
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    model.action_effects["left"] = Counter({(0, -2): 3})
    policy = WorldPolicy(model, seed=0)
    for channel in (
        "_bid_reembody",
        "_bid_calibrate",
        "_bid_plan",
        "_bid_board",
        "_bid_intention",
        "_bid_probe",
        "_bid_reprobe",
        "_bid_frontier",
    ):
        monkeypatch.setattr(policy, channel, lambda available: None)
    monkeypatch.setattr(
        policy,
        "_bid_forage",
        lambda available: {
            "action": "up",
            "cell": None,
            "source": "forage",
            "weight": 0.25,
        },
    )
    monkeypatch.setattr(policy, "_known_failure", lambda start, action: True)
    first = policy.choose(["up", "left"])
    second = policy.choose(["up", "left"])
    assert first is not None and second is not None
    assert first["source"] == second["source"] == "wait"
    assert {first["action"], second["action"]} == {"up", "left"}


def test_refused_move_is_judged_where_she_stays():
    """A bump beside an untraveled hazard is a wait, not a death."""
    model = _world_with_self()
    hazard = {(1, 4), (1, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    # The attempted stride would pass beside the lethal silhouette,
    # but the first cell of that stride is already a measured wall.
    # Issuing the command leaves her footprint where it is; danger
    # must be evaluated there, not along the path she never takes.
    model.known_blocked.add((3, 2))
    assert model.action_refused(state, "up")
    assert not model.action_fatal(state, "up")


def test_policy_can_wait_on_a_refused_move_beside_a_hazard(
    monkeypatch: pytest.MonkeyPatch,
):
    """A known bump can spend the turn when every mobile bid is fatal."""
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    hazard = {(1, 4), (1, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    model.death_signatures[(killer.color, killer.shape)] += 1
    model.known_blocked.add((3, 2))
    policy = WorldPolicy(model, seed=0)
    for channel in (
        "_bid_reembody",
        "_bid_calibrate",
        "_bid_plan",
        "_bid_board",
        "_bid_intention",
        "_bid_probe",
        "_bid_reprobe",
        "_bid_frontier",
        "_bid_forage",
    ):
        monkeypatch.setattr(policy, channel, lambda available: None)
    monkeypatch.setattr(
        policy,
        "_bid_probe",
        lambda available: {
            "action": "up",
            "cell": None,
            "source": "probe",
            "weight": 0.5,
        },
    )
    choice = policy.choose(["up"])
    assert choice is not None
    assert choice["source"] == "wait"
    assert choice["action"] == "up"


@pytest.mark.parametrize("exact", [True, False])
def test_moving_hazard_is_fatal_where_it_will_be_not_only_where_it_is(
    exact: bool,
):
    """A lethal mover is predicted through the same tick as the move."""
    model = _world_with_self()
    hazard = {(0, 4), (0, 5)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.velocity = (1.0, 0.0)
    killer.velocity_exact = exact
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    model.known_free.update(model._swept(state.cells, (-2, 0)))
    # Its current cells are outside the action's contact envelope;
    # its measured next tick is not.
    assert model.action_fatal(state, "up")
    assert model.simulate(state, "up") is None


def test_hazard_position_transition_overrides_its_last_velocity():
    """A patrol's next move is learned from the place it turns.

    The trace that exposed this had a lethal mover arrive at a junction
    moving right, then leave upward. Trusting the incoming velocity called
    the escape safe; the previous body's transition from the same cells
    said otherwise.
    """
    model = _world_with_self()
    start = {(7, 4), (7, 5)}
    others = [
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]
    model.observe(_scene([(_ME, set(_ME_CELLS)), (7, start), *others]))
    killer = next(e for e in model.entities.values() if e.color == 7 and set(e.cells) == start)
    killer.velocity = (0.0, 2.0)  # stale/wrong for the move that matters
    model._record_motion_transition(killer, frozenset(start), (-2, 0), (-2, 0))
    model.action_effects["left"] = Counter({(0, -2): 3})
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    model.known_free.update(model._swept(state.cells, (-2, 0)))
    model.known_free.update(model._swept(state.cells, (0, -2)))
    # Learned motion from this exact position carries it into the upward
    # escape's contact envelope, whatever its incoming velocity says.
    assert model.action_fatal(state, "up")
    assert model.simulate(state, "up") is None
    # Sideways separation is not condemned by the mover's swept path:
    # after the tick it is beside the path, not touching her landing.
    assert not model.action_fatal(state, "left")


def test_hazard_prefers_the_matching_motion_phase_when_known():
    """Cross-phase options are uncertainty, not a verdict forever."""
    model = _world_with_self()
    hazard = {(4, 6), (4, 7)}
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.velocity = (0.0, 2.0)
    model._record_motion_transition(killer, frozenset(hazard), (0, 2), (-2, 0))
    model._record_motion_transition(killer, frozenset(hazard), (-2, 0), (0, -2))
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    model.known_free.update(model._swept(state.cells, (-2, 0)))
    # The matching phase says it turns upward; a leftward departure
    # learned under another phase must not condemn this lane forever.
    assert not model.action_fatal(state, "up")


def test_uncertain_hazard_phase_remembers_its_other_moves():
    """A stochastic patrol phase cannot hide a learned turn primitive."""
    model = _world_with_self()
    hazard = {
        (4, 5),
        (4, 6),
        (5, 5),
        (5, 6),
        (6, 5),
        (6, 6),
    }
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.velocity = (0.0, 2.0)
    # This exact footprint+phase has already answered differently —
    # it is a junction, not a deterministic transition.
    model._record_motion_transition(killer, frozenset(hazard), (0, 2), (0, 2))
    model._record_motion_transition(killer, frozenset(hazard), (0, 2), (0, 0))
    # Elsewhere on the patrol the same silhouette has stepped left.
    other_cells = frozenset((r, c + 2) for r, c in hazard)
    model._record_motion_transition(killer, other_cells, (-2, 0), (0, -2))
    model.action_effects["left"] = Counter({(0, -2): 3})
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    # The unlisted-but-learned left departure brings it into contact
    # with where her leftward stride lands.
    assert model.action_fatal(state, "left")


def test_matching_phase_elsewhere_beats_other_footprint_departures():
    """An unseen local phase is inferred from the same patrol leg."""
    model = _world_with_self()
    hazard = {
        (4, 5),
        (4, 6),
        (5, 5),
        (5, 6),
        (6, 5),
        (6, 6),
    }
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.velocity = (0.0, 2.0)
    # This footprint has turned left, but only after arriving upward.
    model._record_motion_transition(killer, frozenset(hazard), (-2, 0), (0, -2))
    # This rightward leg has been seen continuing right elsewhere.
    other_cells = frozenset((r, c + 2) for r, c in hazard)
    model._record_motion_transition(killer, other_cells, (0, 2), (0, 2))
    # A distant rightward leg turned left; it is weaker evidence than
    # the adjacent rightward leg that continued right.
    far_cells = frozenset((r, c + 10) for r, c in hazard)
    model._record_motion_transition(killer, far_cells, (0, 2), (0, -2))
    model.action_effects["left"] = Counter({(0, -2): 3})
    model.death_signatures[(killer.color, killer.shape)] += 1
    state = model.start_state()
    assert state is not None
    # Continuing right carries the escort away from her left landing;
    # the left turn belongs to a different phase and must not veto it.
    assert not model.action_fatal(state, "left")


def test_action_contingent_hazard_is_fatal_in_its_fatal_direction():
    """A command-following escort is judged by this command's effect."""
    model = _world_with_self()
    hazard = {
        (7, 2),
        (7, 3),
        (7, 4),
        (8, 2),
        (8, 3),
        (8, 4),
        (9, 2),
        (9, 3),
        (9, 4),
    }
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.steps_after_action = 3
    killer.action_moves = 3
    killer.action_effect_moves = 3
    model.death_signatures[(killer.color, killer.shape)] += 1
    model.death_motion_deltas[(killer.color, killer.shape, (-3, 0))] += 1
    model.action_effects["up"] = Counter({(-3, 0): 3})
    model.action_effects["right"] = Counter({(0, 3): 3})
    state = model.start_state()
    assert state is not None
    # Up repeats the fatal chase direction; right is a known lateral
    # escort move, not the observed kill vector.
    assert model.action_fatal(state, "up")
    assert not model.action_fatal(state, "right")


def test_action_contingent_hazard_uses_an_observed_refusal():
    """A follower that has stopped for this command may stay lethal."""
    model = _world_with_self()
    hazard = {
        (4, 5),
        (4, 6),
        (5, 5),
        (5, 6),
        (6, 5),
        (6, 6),
    }
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.steps_after_action = 3
    killer.action_moves = 3
    killer.action_effect_moves = 3
    model.death_signatures[(killer.color, killer.shape)] += 1
    model.death_motion_deltas[(killer.color, killer.shape, (-3, 0))] += 1
    model.action_effects["right"] = Counter({(0, 3): 3})
    model._record_motion_transition(killer, frozenset(hazard), (0, 3), (0, 0))
    state = model.start_state()
    assert state is not None
    assert model._predicted_motion(killer, (0, 3)) == {(0, 0)}
    # The unrelated kill vector cannot clear the observed stopped
    # outcome: staying in her landing footprint is still lethal.
    assert model.action_fatal(state, "right")


def test_autonomous_hazard_keeps_uncertain_motion_despite_fatal_direction():
    """A patrol's unproven directions are not acquitted by one kill vector."""
    model = _world_with_self()
    hazard = {
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 5),
        (6, 6),
        (6, 7),
    }
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, hazard),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    killer = next(e for e in model.entities.values() if set(e.cells) == hazard)
    killer.velocity = (2.0, 0.0)
    model._record_motion_transition(killer, frozenset(hazard), (2, 0), (-2, 0))
    model._record_motion_transition(killer, frozenset(hazard), (2, 0), (2, 0))
    model.death_signatures[(killer.color, killer.shape)] += 1
    model.death_motion_deltas[(killer.color, killer.shape, (0, 2))] += 1
    state = model.start_state()
    assert state is not None
    # Her upward stride is beside the patrol's learned upward turn. An
    # autonomous hazard cannot be reduced to the one direction evidence
    # happened to record at a previous ending.
    assert model.action_fatal(state, "up")


# ─── conditional passability ────────────────────────────────


def test_blocked_state_pair_is_not_kind_wide():
    """A refusal is recorded against (entity, self-state), not the kind."""
    model = _world_with_self()
    me = model.self_entity
    assert me is not None and me.color == _ME
    sig = model.self_signature()
    assert sig is not None
    door = model._spawn(PerceivedObject(index=99, color=7, cells=frozenset({(4, 5), (4, 6)})))
    door.blocked_states.add(sig)
    model.affordances.setdefault(7, Counter())[OUT_BLOCK] += 1
    # Same state: refused. A remade self may enter.
    assert not model.passable(door, sig)
    assert model.passable(door, ((11, sig[0][1]),))


# ─── composite self ─────────────────────────────────────────


_PART = {(4, 5), (5, 5)}  # color-9 block attached to her right edge


def _composite_world() -> WorldModel:
    """Her body is two pieces — a color-3 core and a color-9 part.

    The part rides at her right edge and translates with her; two
    consecutive rigid co-moves earn annexation. She ends back at
    her original cells for a stable reference frame.
    """
    model = WorldModel()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)}), (9, {(12, 10), (12, 11)})]
    model.observe(_scene([(_ME, set(_ME_CELLS)), (9, set(_PART)), *other]))
    for step in (1, 2):  # down: rows 4-6 → 6-8 → 8-10
        model.note_action("down")
        model.observe(
            _scene(
                [
                    (_ME, {(r + 2 * step, c) for r, c in _ME_CELLS}),
                    (9, {(r + 2 * step, c) for r, c in _PART}),
                    *other,
                ]
            )
        )
    for step in (1, 2):  # up: back to rows 4-6
        model.note_action("up")
        model.observe(
            _scene(
                [
                    (_ME, {(r + 2 * (2 - step), c) for r, c in _ME_CELLS}),
                    (9, {(r + 2 * (2 - step), c) for r, c in _PART}),
                    *other,
                ]
            )
        )
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


def test_same_color_decoy_cannot_steal_a_body_part_identity():
    """A landing near same-colored scenery must not swap limb identity.

    ls20 puts the avatar's color-9 half directly beside other color-9
    scenery. Nearest-centroid tracking can then prefer the stationary
    decoy's old position for the *moved* part. The pending action's
    learned effect names the part's expected landing cells, so body
    continuity wins over a merely closer stranger.
    """
    model = _composite_world()
    part = next(e for e in model.entities.values() if e.part_of_self)
    decoy_cells = {(4, 15), (5, 15)}
    others = [
        (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
        (9, {(12, 10), (12, 11)}),
    ]
    model.note_action("right")
    model.observe(
        _scene(
            [
                (_ME, {(r, c + 4) for r, c in _ME_CELLS}),
                (9, {(r, c + 4) for r, c in _PART}),
                *others,
            ]
        )
    )

    model.note_action("right")
    model.observe(
        _scene(
            [
                (_ME, {(r, c + 8) for r, c in _ME_CELLS}),
                (9, {(r, c + 8) for r, c in _PART}),
                (9, set(decoy_cells)),
                *others,
            ]
        )
    )
    decoy = next(
        e
        for e in model.entities.values()
        if e.id != part.id and e.color == 9 and set(e.cells) == decoy_cells
    )

    assert part.part_of_self
    assert set(part.cells) == {(r, c + 8) for r, c in _PART}
    assert not decoy.part_of_self
    assert set(decoy.cells) == decoy_cells


def test_pacing_neighbor_is_not_a_part():
    """A thing that keeps pace but drifts is never annexed."""
    model = WorldModel()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)})]
    start = {(r + 4, c) for r, c in _ME_CELLS}  # rows 8-10
    model.observe(_scene([(_ME, start), (9, {(r + 4, c) for r, c in _PART}), *other]))
    # The neighbor moves half as far — close, but it does not ride.
    part_cells = {(r + 4, c) for r, c in _PART}
    for step in range(1, 4):
        model.note_action("up")
        part_cells = {(r - 1, c) for r, c in part_cells}
        model.observe(
            _scene(
                [
                    (_ME, {(r - 2 * step, c) for r, c in start}),
                    (9, part_cells),
                    *other,
                ]
            )
        )
    assert all(not e.part_of_self for e in model.entities.values())


def test_autonomous_neighbor_cannot_be_annexed_later():
    """A mover that already answered its own clock is not a limb."""
    model = _world_with_self()
    other = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)})]
    part_cells = set(_PART)
    model.observe(_scene([(_ME, set(_ME_CELLS)), (9, part_cells), *other]))
    # It moves while she does not — autonomous motion, even though
    # a command was pending and every later stride matches hers.
    model.note_action("hold")
    part_cells = {(r + 1, c) for r, c in part_cells}
    model.observe(_scene([(_ME, set(_ME_CELLS)), (9, part_cells), *other]))
    me_cells = set(_ME_CELLS)
    for _ in range(2):
        model.note_action("down")
        me_cells = {(r + 2, c) for r, c in me_cells}
        part_cells = {(r + 2, c) for r, c in part_cells}
        model.observe(_scene([(_ME, me_cells), (9, part_cells), *other]))
    mover = max(
        (e for e in model.entities.values() if e.color == 9),
        key=lambda e: e.id,
    )
    assert mover.independent_moves > 0
    assert not mover.part_of_self


def test_suspected_hazard_cannot_remain_a_body_part():
    """A lethal co-mover is a threat riding beside her, not a limb."""
    model = _composite_world()
    part = next(e for e in model.entities.values() if e.part_of_self)
    model.death_signatures[(part.color, part.shape)] += 1
    events = model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (9, set(_PART)),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    assert not part.part_of_self
    assert any(e.kind == "part_lost" for e in events)


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
    model.observe(
        _scene(
            [
                (7, pad_cells),
                (_ME, {(r + 1, c) for r, c in _ME_CELLS}),
                (11, {(r + 1, c) for r, c in _PART}),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    # Recolored mid-move and still riding: identity kept by adoption.
    assert part.color == 11 and part.part_of_self
    assert model.transforms.get(7, {}).get(from_sig, Counter())[to_sig] >= 1
    assert model.affordance_of(7)[OUT_TRANSFORM] >= 1
    # Simulation applies the learned rewrite to her old state.
    state = model.start_state()
    assert state is not None
    pre = SimState(cells=state.cells, signature=sig_before, gone=state.gone)
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
    model.observe(
        _scene(
            [
                (9, floor),
                (_ME, {(r + 1, c) for r, c in _ME_CELLS}),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
            ]
        )
    )
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
    door = model._spawn(PerceivedObject(index=98, color=6, cells=frozenset({(4, 7)})))
    door.blocked_states.add(sig_a)
    model.affordances.setdefault(6, Counter())[OUT_BLOCK] += 1
    assert not model.passable(door, sig_a)
    sig_b = tuple((c, s) if c != 9 else (11, s) for c, s in sig_a)
    assert model.passable(door, sig_b)


def test_respawn_preserves_kind_refusals():
    """Same body after death, same measured answers.

    A respawn is not hidden-state evidence: the signature itself is
    the measure of what changed. A kind that refused this signature
    before the episode reset still refuses it after — otherwise
    every new life spends its steps re-walking to doors that already
    said no.
    """
    model = _world_with_self()
    sig = model.self_signature()
    assert sig is not None
    door = model._spawn(PerceivedObject(index=98, color=6, cells=frozenset({(4, 7)})))
    door.blocked_states.add(sig)
    door.blocked_epochs[sig] = model._state_epoch
    model.blocked_kinds.setdefault(6, set()).add(sig)
    model._blocked_epoch[(6, sig)] = model._state_epoch
    model.affordances.setdefault(6, Counter())[OUT_BLOCK] += 1
    model.affordances[6][OUT_WIN] += 1
    assert model.state_refused(door, sig)

    model.reset(keep_learning=True)
    # Re-embody: the same motion-correlation the episode needs before
    # a signature exists to compare against.
    model.observe(
        _scene(
            [
                (_ME, {(r + 4, c) for r, c in _ME_CELLS}),
                (6, {(4, 7)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")
    model.observe(
        _scene(
            [
                (_ME, {(r + 2, c) for r, c in _ME_CELLS}),
                (6, {(4, 7)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (6, {(4, 7)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    door2 = next(e for e in model.entities.values() if e.color == 6)
    assert model.self_signature() == sig
    assert model.state_refused(door2, sig)

    # A refused goal is a question, not a destination: it leaves the
    # pursuit set until something changes her state.
    from basal_ganglia.world_policy import WorldPolicy

    valued = WorldPolicy(model, seed=0)._valued_cells()
    assert not door2.cells & valued

    # ...and returns to it for the state that has never been refused.
    sig_b = tuple((c, s) if c != _ME else (11, s) for c, s in sig)
    assert not model.state_refused(door2, sig_b)


def test_state_changer_is_valued_while_signature_refused():
    """The thing that can move her state is a means while this state
    has been refused — even when the refusal evidence survives only
    at kind granularity across an episode reset."""
    model = _world_with_self()
    sig = model.self_signature()
    assert sig is not None
    door = model._spawn(PerceivedObject(index=98, color=6, cells=frozenset({(4, 7)})))
    door.blocked_states.add(sig)
    door.blocked_epochs[sig] = model._state_epoch
    model.blocked_kinds.setdefault(6, set()).add(sig)
    model._blocked_epoch[(6, sig)] = model._state_epoch
    model.affordances.setdefault(6, Counter())[OUT_BLOCK] += 1
    model.affordances[6][OUT_WIN] += 1
    model._state_changers.add(8)

    model.reset(keep_learning=True)
    model.observe(
        _scene(
            [
                (_ME, {(r + 4, c) for r, c in _ME_CELLS}),
                (6, {(4, 7)}),
                (8, {(10, 10)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")
    model.observe(
        _scene(
            [
                (_ME, {(r + 2, c) for r, c in _ME_CELLS}),
                (6, {(4, 7)}),
                (8, {(10, 10)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    model.note_action("up")
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (6, {(4, 7)}),
                (8, {(10, 10)}),
                (9, {(12, 10), (12, 11)}),
            ]
        )
    )
    from basal_ganglia.world_policy import WorldPolicy

    valued = WorldPolicy(model, seed=0)._valued_cells()
    pad = next(e for e in model.entities.values() if e.color == 8)
    door2 = next(e for e in model.entities.values() if e.color == 6)
    assert pad.cells <= valued
    assert not door2.cells & valued


def test_unoccluded_object_revives_its_own_entity_not_self():
    """A thing resurfacing under her belongs to its own lineage.

    She stands on a pad; her body vanishes (death, teleport); the
    pad's pixels are seen again where she stood. Without a
    same-color lineage check, the recolor pass claims them for her
    self record — binding "me" to a stationary wall for as long as
    the silence timer allows.
    """
    model = _world_with_self()
    me = model.self_entity
    assert me is not None
    pad = {(4, 5), (5, 5)}
    others = [(7, pad), (5, {(0, 0), (0, 1), (1, 0), (1, 1)})]
    model.observe(_scene([(_ME, set(_ME_CELLS)), *others]))
    pad_e = next(e for e in model.entities.values() if set(e.cells) == pad)

    # Her footprint covers the pad: it goes unobserved underneath
    # her and is marked gone while she stands on it.
    model.note_action("down")
    model.observe(
        _scene(
            [
                (_ME, {(r, c + 1) for r, c in _ME_CELLS} | {(4, 5), (5, 5)}),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
            ]
        )
    )
    # Death: her body teleports away; the pad is seen again — same
    # cells, same color — while her old record is still looking.
    model.note_action("down")
    model.observe(
        _scene(
            [
                (
                    _ME,
                    {
                        (10, 1),
                        (10, 2),
                        (11, 1),
                        (11, 2),
                        (12, 1),
                        (12, 2),
                        (13, 1),
                        (13, 2),
                        (13, 3),
                    },
                ),
                (7, pad),
                (5, {(0, 0), (0, 1), (1, 0), (1, 1)}),
            ]
        )
    )

    assert not pad_e.gone
    assert set(pad_e.cells) == pad
    assert pad_e.color == 7
    # Her record was not re-skinned to the pad: if self survived the
    # frame at all it is not the pad fragment.
    self_ent = model.entities.get(model.self_id or -1)
    assert self_ent is None or set(self_ent.cells) != pad


def test_contact_linked_readout_joins_self_signature():
    """A remote indicator can be self-state without being body space.

    The causal signature is the same one embodied transformers use:
    a stable object away from her footprint rewrites itself only when
    she touches something. One coincidence is a vote; a repeated
    contact-linked rewrite binds that seat's readout into the
    signature while keeping it out of the body and out of target sets.
    """
    model = _world_with_self()
    pad = {(4, 5), (5, 5)}
    icon_cells = {(12, 10), (12, 11)}
    others = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)})]
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, pad),
                (9, icon_cells),
                *others,
            ]
        )
    )
    icon = next(e for e in model.entities.values() if set(e.cells) == icon_cells)
    icon_from = (10, icon.shape)
    icon_to = (11, icon.shape)

    model.note_action("push")
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, pad),
                (10, icon_cells),
                *others,
            ]
        )
    )
    assert not icon.state_of_self

    model.note_action("push")
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, pad),
                (11, icon_cells),
                *others,
            ]
        )
    )

    assert icon.state_of_self
    assert model.is_display(icon)
    sig = model.self_signature()
    assert sig is not None and icon_to in sig
    assert not icon.cells & model.self_footprint()
    assert model.transforms[7][icon_from][icon_to] == 1
    assert model.affordance_of(7)[OUT_TRANSFORM] == 1

    # A learned readout is not a world object to pursue just because
    # its color happens to be valuable elsewhere.
    model.affordances.setdefault(11, Counter())[OUT_WIN] += 1
    goal = model._spawn(PerceivedObject(index=97, color=11, cells=frozenset({(14, 14)})))
    from basal_ganglia.world_policy import WorldPolicy

    valued = WorldPolicy(model, seed=0)._valued_cells()
    assert set(goal.cells) <= valued
    assert not icon.cells & valued

    # The seat survives retries: the next attempt's readout is
    # recognized as self evidence even before it changes.
    model.reset(keep_learning=True)
    model.observe(_scene([(_ME, set(_ME_CELLS)), (9, icon_cells)]))
    restored = next(e for e in model.entities.values() if set(e.cells) == icon_cells)
    assert restored.state_of_self


def test_hud_churn_is_not_a_self_state_candidate():
    """A per-tick counter is a display, not a state readout.

    Churners that rewrite themselves every frame pass through
    ``is_display`` after enough observations; once classified they
    must not collect self-state votes, or a step gauge would end up
    inside the signature and make every state look new.
    """
    model = _world_with_self()
    pad = {(4, 5), (5, 5)}
    others = [(5, {(0, 0), (0, 1), (1, 0), (1, 1)})]
    # Bottom-edge readout whose cells shuffle a little every frame —
    # the shape a step counter takes. Two alternating 3-cell patterns
    # keep identity via centroid while churning every tick.
    small_a = {(15, 5), (15, 6), (15, 7)}
    small_b = {(15, 5), (15, 7), (15, 8)}
    for i in range(6):
        model.note_action("push")
        model.observe(
            _scene(
                [
                    (_ME, set(_ME_CELLS)),
                    (7, pad),
                    (9, small_a if i % 2 == 0 else small_b),
                    *others,
                ]
            )
        )
    counter = next(
        e for e in model.entities.values() if e.color == 9 and any(r >= 15 for r, _ in e.cells)
    )
    assert model.is_display(counter)

    # Now it makes big rewrites while she is in causal contact —
    # the exact pattern that would earn a seat vote for a stable
    # readout. A display must still be refused.
    big_a = {(15, 5), (15, 6), (15, 7), (15, 8)}
    big_b = {(15, 10), (15, 11), (15, 12), (15, 13)}
    for i in range(2):
        model.note_action("push")
        model.observe(
            _scene(
                [
                    (_ME, set(_ME_CELLS)),
                    (7, pad),
                    (9, big_a if i % 2 == 0 else big_b),
                    *others,
                ]
            )
        )
    assert not counter.state_of_self
    assert not any(seat & counter.cells for seat in model._self_state_votes)
    sig = model.self_signature()
    assert sig is not None and (9, counter.shape) not in sig


def test_remote_state_transform_can_unlock_goal_route():
    """A state readout makes gate-changing plans representable.

    Without the remote part the signature does not change on the pad,
    so the goal's blocked verdict keeps applying. With it, contact
    rewrites the signature and the same learned physics can plan
    "transform, then enter the goal".
    """
    from frontal_lobe.world_planner import WorldPlanner

    model, cur = _stride_world()
    icon_cells = {(44, 0), (44, 1)}
    pad = {(10, 2)}
    goal = {(5, 2)}
    model.observe(
        _scene(
            [
                (_ME, set(cur)),
                (7, pad),
                (8, goal),
                (9, icon_cells),
            ],
            h=48,
            w=16,
        )
    )
    icon = next(e for e in model.entities.values() if set(e.cells) == icon_cells)
    icon.state_of_self = True
    model._self_state_sites.add(model._self_state_seat(icon.cells))
    sig_a = model.self_signature()
    assert sig_a is not None and (9, icon.shape) in sig_a
    pad_e = next(e for e in model.entities.values() if set(e.cells) == pad)
    goal_e = next(e for e in model.entities.values() if set(e.cells) == goal)
    goal_e.blocked_states.add(sig_a)
    goal_e.blocked_epochs[sig_a] = model._state_epoch
    model.affordances.setdefault(7, Counter())[OUT_TRANSFORM] += 1
    model.transforms.setdefault(7, {}).setdefault((icon.color, icon.shape), Counter())[
        (10, icon.shape)
    ] += 1
    model.affordances.setdefault(8, Counter())[OUT_WIN] += 1
    model.known_free.update((r, c) for r in range(48) for c in range(16))

    plan = WorldPlanner(model).plan(set(goal), ["up"])

    assert plan is not None
    assert [action for action, _ in plan] == ["up", "up", "up"]
    assert pad_e.color == 7
    assert goal_e.color == 8


# ─── terrain ────────────────────────────────────────────────


def _terrain_world() -> WorldModel:
    """She walks across a large floor region that reclaims her cells.

    The floor (color 8) fills the room; her color-3 block moves
    through it. Each vacated footprint reappears as floor — the
    medium absorbing the space she leaves.
    """
    floor_cells = {(r, c) for r in range(2, 14) for c in range(0, 14)}
    model = WorldModel()
    start = {(r + 4, c) for r, c in _ME_CELLS}  # rows 8-10
    model.observe(
        _scene(
            [
                (_ME, start),
                (8, floor_cells - start),
            ]
        )
    )
    cur = start
    for _ in range(3):
        nxt = {(r - 1, c) for r, c in cur}
        model.note_action("up")
        model.observe(
            _scene(
                [
                    (_ME, nxt),
                    (8, floor_cells - nxt),
                ]
            )
        )
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
    model.observe(
        _scene(
            [
                (_ME, set(cur)),  # refused: unmoved
                (6, wall),
                (8, floor_cells - cur - wall),
            ]
        )
    )
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
        model.observe(
            _scene(
                [
                    (_ME, nxt),
                    (8, floor - nxt),
                ]
            )
        )
        cur = nxt
    assert (1, 2) not in model.known_free
    state = model.start_state()
    assert state is not None
    assert model.simulate(state, "up") is None
    # Policy gating may still spend a real step on the unanswered edge;
    # only demonstrated refusals are a known failure. Planning remains
    # bound to proven floor.
    assert model.simulate(state, "up", explore_unknown=True) is not None
    model.refused_cells.update(model._swept(state.cells, (-1, 0)))
    assert model.simulate(state, "up", explore_unknown=True) is None


def test_refusal_without_entity_marks_space_not_things():
    """A failed move into empty space refuses the cells, not a kind."""
    model, cur = _terrain_world()
    # She pushes up into unclaimed space beyond the floor — no
    # entity is there to blame, the space itself refuses.
    floor_cells = {(r, c) for r in range(2, 14) for c in range(0, 14)}
    for _ in range(4):  # walk to the floor's top edge
        nxt = {(r - 1, c) for r, c in cur}
        model.note_action("up")
        model.observe(
            _scene(
                [
                    (_ME, nxt),
                    (8, floor_cells - nxt),
                ]
            )
        )
        cur = nxt
    # A refused strip is a disjunction — *some* cell inside it
    # refused — so a multi-cell destination earns suspicion, not
    # condemnation. Convergence convicts: the second refusal of
    # the same cells is the wall showing under the strip again.
    for _ in range(2):
        model.note_action("up")
        model.observe(
            _scene(
                [
                    (_ME, set(cur)),  # refused: off the floor
                    (8, floor_cells - cur),
                ]
            )
        )
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
        model.observe(
            _scene(
                [
                    (_ME, nxt),
                    (8, floor_cells - nxt),
                ]
            )
        )
        cur = nxt
    # Two refusals — suspicion converges to conviction.
    for _ in range(2):
        model.note_action("up")
        model.observe(
            _scene(
                [
                    (_ME, set(cur)),
                    (8, floor_cells - cur),
                ]
            )
        )
    refused = {(r - 1, c) for r, c in cur} - floor_cells - cur
    # Another observation cycle: free-space marking runs again —
    # the refused cells must stay blocked.
    model.observe(
        _scene(
            [
                (_ME, set(cur)),
                (8, floor_cells - cur),
            ]
        )
    )
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
    cur = {(r + 30, c) for r, c in _ME_CELLS}  # rows 30-32
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
    model, cur = _stride_world()  # cur rows 20-22
    state = model.start_state()
    assert state is not None
    nxt = model.simulate(state, "up")
    assert nxt is not None
    assert nxt.cells == frozenset((r - 5, c) for r, c in cur)


def test_stride_bump_blames_the_wall_inside_the_leap():
    """The blocker inside a stride is convicted — not the landing."""
    model, cur = _stride_world()  # cur rows 19-21
    # The wall sits inside the stride at row 17 — the landing
    # strip (rows 14-16) is open; the blocker lives mid-leap.
    wall = {(17, c) for c in (2, 3, 4)}
    model.note_action("up")
    model.observe(
        _scene(
            [
                (_ME, set(cur)),  # refused: unmoved
                (6, wall),
            ],
            h=48,
            w=16,
        )
    )
    assert model.affordance_of(6)[OUT_BLOCK] == 1
    assert wall <= model.known_blocked
    # And the model now refuses the leap in imagination, even
    # though the destination strip itself is open.
    state = model.start_state()
    assert state is not None
    assert model.simulate(state, "up") is None


def test_stride_bump_spares_proven_floor_in_the_leap():
    """Walked-on ground inside the path takes none of the blame."""
    model, cur = _stride_world()  # cur rows 19-21
    # A color-7 patch inside the stride she has already crossed
    # without consequence — proven-open surface.
    pad = {(18, c) for c in (2, 3, 4)}
    model.affordances.setdefault(7, Counter())[OUT_NOTHING] += 1
    wall = {(17, c) for c in (2, 3, 4)}
    model.note_action("up")
    model.observe(
        _scene(
            [
                (_ME, set(cur)),
                (6, wall),
                (7, pad),
            ],
            h=48,
            w=16,
        )
    )
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
    model.observe(
        _scene(
            [
                (_ME, set(cur)),
                (6, wall),
            ],
            h=48,
            w=16,
        )
    )
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
    model, cur = _stride_world()  # cur rows 19-21
    goal = {(14, 2), (14, 3), (14, 4), (15, 2), (15, 3), (15, 4), (16, 2), (16, 3), (16, 4)}
    model.observe(_scene([(_ME, set(cur)), (7, goal)], h=48, w=16))
    # The winning leap carries her onto the goal — the next frame
    # shows her there with the goal already gone.
    model.note_action("up")
    landing = {(r - 5, c) for r, c in cur}
    events = model.observe(_scene([(_ME, landing)], h=48, w=16), info={"win": True})
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
    model, cur = _stride_world()  # me color-3 rows 19-21, up = (-5, 0)
    gate = {
        (14, 2),
        (14, 3),
        (14, 4),
        (15, 2),
        (15, 3),
        (15, 4),
        (16, 2),
        (16, 3),
        (16, 4),
    }  # inside the leap, not adjacent
    wall = {(15, 6), (15, 7), (16, 7)}  # color-7: a plain wall
    pad = {(25, 6), (25, 7)}  # color-1: the unexplained
    model.observe(_scene([(_ME, set(cur)), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    # Leap up — the gate inside the path refuses.
    model.note_action("up")
    model.observe(_scene([(_ME, set(cur)), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    # Shift over and leap up into the color-7 wall — refused too.
    moved = {(r, c + 4) for r, c in cur}
    model.observe(_scene([(_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    model.note_action("up")
    model.observe(_scene([(_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    sig = model.self_signature()
    gate_e = next(e for e in model.entities.values() if e.color == 6)
    wall_e = next(e for e in model.entities.values() if e.color == 7)
    assert not model.passable(gate_e, sig)
    assert not model.passable(wall_e, sig)
    # She walks onto the unexplained tile — the scene shows only
    # her there; the tracked pad still overlaps her footprint.
    stand = {(24, 6), (24, 7), (24, 8), (25, 6), (25, 7), (25, 8), (26, 6), (26, 7), (26, 8)}
    model.observe(_scene([(_ME, stand), (6, gate), (7, wall)], h=48, w=16))
    # Both old verdicts are stale now — her state may have changed
    # in ways nothing reported.
    assert model.passable(gate_e, sig)
    assert model.passable(wall_e, sig)
    # Re-measured in the new epoch the wall refuses again — its
    # verdict binds afresh; the un-retested gate stays eligible.
    model.observe(_scene([(_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
    model.note_action("up")
    model.observe(_scene([(_ME, moved), (6, gate), (7, wall), (1, pad)], h=48, w=16))
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
    mobile = model._spawn(PerceivedObject(index=90, color=8, cells=frozenset({(5, 6)})))
    mobile.velocity = (1.0, 0.0)
    mobile.blocked_states.add(sig)
    mobile.blocked_epochs[sig] = epoch
    stable = model._spawn(PerceivedObject(index=91, color=9, cells=frozenset({(10, 3)})))
    stable.blocked_states.add(sig)
    stable.blocked_epochs[sig] = epoch
    other = model._spawn(PerceivedObject(index=92, color=10, cells=frozenset({(12, 6)})))
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


def test_calibration_babbles_the_least_measured_action():
    """A no-op in one state is a question, not a dead limb.

    The trace pathology: a direction fails at the first wall, earns
    an all-zero effect, and then calibration wastes the run randomly
    re-testing controls that already have many trials. Distinct
    refusal contexts can convict the action; one position cannot —
    and babbling spends its next trial on the least-measured control.
    """
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    model.action_effects["left"] = Counter({(0, -2): 3})
    model.action_tries["left"] = 10
    model.action_effects["right"] = Counter({(0, 2): 3})
    model.action_tries["right"] = 10
    model.action_effects["down"] = Counter({(0, 0): 3})
    model.action_tries["down"] = 3
    model.action_noop_contexts["down"] = {frozenset(_ME_CELLS)}
    assert not model.action_understood("down")
    model.action_noop_contexts["down"].update(
        frozenset({(r + i, c) for r, c in _ME_CELLS}) for i in range(1, 4)
    )
    assert model.action_understood("down")
    model.action_noop_contexts["down"] = {frozenset(_ME_CELLS)}

    bid = WorldPolicy(model, seed=0)._bid_calibrate(["left", "right", "down"])
    assert bid is not None
    assert bid["action"] == "down"
    assert bid["source"] == "calibrate"


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
    model.observe(
        _scene(
            [
                (_ME, set(_ME_CELLS)),
                (7, {(10, 10), (10, 11)}),
            ]
        )
    )
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


def test_gone_self_is_unembodied_not_a_planning_state():
    """A corpse's old cells cannot veto the search for a new body."""
    from basal_ganglia.world_policy import WorldPolicy

    model = _world_with_self()
    me = model.self_entity
    assert me is not None
    me.gone = True
    assert model.start_state() is None
    bid = WorldPolicy(model, seed=0).choose(["left", "right"])
    assert bid is not None
    assert bid["source"] == "reembody"


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


def test_world_transition_clears_policy_episode_state(tmp_path):
    """A win/reset inside observe retires the old episode's policy.

    The model forgets a replaced layout inside ``observe``; the basal
    ganglia cannot keep aiming at plans, entities, or re-embodiment
    counts measured against the vanished world.
    """
    from mind import Mind

    subject = Mind(str(tmp_path / "genesis.sock"), offline=True)
    policy = subject._world_policy
    state = SimState(
        cells=frozenset(_ME_CELLS),
        signature=(),
        gone=frozenset(),
    )
    policy._explore_goal = (5, 5)
    policy._frontier_plan = [("up", state)]
    policy._persist = ({"action": "up"}, state)
    policy._retrial_entity = 99
    policy._reembody_tries["up"] = 3

    subject.observe(_scene([(_ME, set(_ME_CELLS))]), info={"win": True})

    assert policy._explore_goal is None
    assert policy._frontier_plan is None
    assert policy._persist is None
    assert policy._retrial_entity is None
    assert not policy._reembody_tries


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
