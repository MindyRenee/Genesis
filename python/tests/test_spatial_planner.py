"""Spatial deliberation tests — collision-learned terrain and BFS.

Two layers of honesty:

- the organic test drives ``observe()`` through real frame
  transitions, so avatar identity, step vectors, and blocked cells
  are all *learned* — exactly the evidence path live play uses;
- the seeded tests then unit-test the planner against that learned
  state directly, so each property (routing, frontier, hazards) is
  checked in isolation.

Nothing here knows anything about any particular game: the planner
consumes displacement statistics, collision memory, and visitation —
self-generated evidence only.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])

from frontal_lobe.reasoning import TaskCompetence
from parietal_lobe.grid import Grid
from parietal_lobe.scene import perceive
from worlds.spatial_agent import SpatialAgent

# Steps are the moves a world would teach: 1=up 2=down 3=left 4=right.
_STEPS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _world(rows: int = 16, cols: int = 16) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def _place(g: list[list[int]], r: int, c: int, v: int) -> None:
    g[r][c] = v


def _to_grid(g: list[list[int]]) -> Grid:
    return Grid.from_lists(g)


def _step_frames(
    g: list[list[int]], at: tuple[int, int], action: int
) -> tuple[list[list[int]], tuple[int, int]]:
    """Apply a move honestly: blocked cells refuse the move."""
    dr, dc = _STEPS[action]
    nr, nc = at[0] + dr, at[1] + dc
    nxt = [row[:] for row in g]
    if 0 <= nr < len(g) and 0 <= nc < len(g[0]) and nxt[nr][nc] == 0:
        nxt[at[0]][at[1]], nxt[nr][nc] = 0, 9
        return nxt, (nr, nc)
    return nxt, at  # bumped — world says no


def test_learned_end_to_end_plan():
    """Avatar, strides, and walls are all earned through observe()."""
    agent = SpatialAgent(task_competence=TaskCompetence(), perceive=perceive)
    g = _world()
    at = (8, 4)
    _place(g, *at, 9)
    agent.observe(_to_grid(g))

    # Earn the stride model: one free move per direction first —
    # a blocked cell only teaches terrain once she knows which way
    # she was trying to go.
    for action in (4, 4, 1, 1, 3, 2):
        agent._last_action = action
        g, at = _step_frames(g, at, action)
        agent.observe(_to_grid(g))
    assert agent.avatar_color == 9
    steps = agent._step_vectors([1, 2, 3, 4])
    assert steps[4] == (0, 1) and steps[1] == (-1, 0)
    assert steps[3] == (0, -1) and steps[2] == (1, 0)

    # Earn a wall: put one left of her, let her see it, then push
    # into it twice. A colored refusal is a gate suspect — two
    # witnessed bumps hard-block it for the planner while a state
    # change can still reopen it.
    g[at[0]][at[1] - 1] = 4
    agent.observe(_to_grid(g))
    for _ in range(2):
        agent._last_action = 3
        g, at = _step_frames(g, at, 3)
        agent.observe(_to_grid(g))
    assert (at[0], at[1] - 1) in agent._blocked_cells | agent._gated_cells

    # Now she plans: goal is up-right, wall learned — the first step
    # must not be into the wall.
    action = agent._plan(_to_grid(g), [1, 2, 3, 4], (4.0, 12.0))
    assert action != 3 and action is not None


def test_step_vectors_use_successful_moves_only():
    agent = SpatialAgent(task_competence=TaskCompetence(), perceive=perceive)
    agent.avatar_color = 9
    st = agent._stat(1)
    st.moves = 5
    st.dr = -10.0  # attempts blocked at walls don't dilute stride
    assert agent._step_vectors([1, 2, 3, 4])[1] == (-2, 0)
    # Untried actions offer no vector — they stay exploration-only.
    assert agent._step_vectors([7]) == {}


def test_collision_marks_destination_blocked():
    agent = SpatialAgent(task_competence=TaskCompetence(), perceive=perceive)
    agent.avatar_color = 9
    st = agent._stat(4)
    st.moves = 10
    st.dc = 10.0
    g = _world()
    g[5][5], g[5][6] = 9, 4
    prev = _to_grid(g)
    agent._learn_blocked(prev, prev, 4)
    # A colored refusal is a gate suspect: the edge is recorded as
    # refused while the cell stays terrain-optimistic until the
    # refusal repeats under a changed state.
    assert (5, 5, 4) in agent._bumped_gates
    assert (5, 6) not in agent._blocked_cells

    # A background-colored refusal is terrain outright — the void
    # opens for no state.
    g = _world()
    g[5][5], g[5][6] = 9, 0
    prev = _to_grid(g)
    agent._learn_blocked(prev, prev, 4)
    assert (5, 6) in agent._blocked_cells
    assert (5, 5, 4) in agent._bumped_moves


def test_no_collision_no_blocked():
    agent = SpatialAgent(task_competence=TaskCompetence(), perceive=perceive)
    agent.avatar_color = 9
    st = agent._stat(4)
    st.moves = 10
    st.dc = 10.0
    g = _world()
    g[5][5] = 9
    prev = _to_grid(g)
    g[5][5], g[5][6] = 0, 9
    agent._learn_blocked(prev, _to_grid(g), 4)
    assert not agent._blocked_cells


def _seeded_agent(at: tuple[int, int], g: list[list[int]]) -> SpatialAgent:
    agent = SpatialAgent(task_competence=TaskCompetence(), perceive=perceive)
    agent.avatar_color = 9
    for a, (dr, dc) in _STEPS.items():
        st = agent._stat(a)
        st.moves = 10
        st.dr, st.dc = dr * 10.0, dc * 10.0
    g[at[0]][at[1]] = 9
    return agent


def test_bfs_routes_around_wall():
    # Wall column at c=7 rows 0..12; gap at row 13. Greedy fails, BFS
    # must route down and around.
    g = _world()
    agent = _seeded_agent((4, 5), g)
    for r in range(13):
        g[r][7] = 4
        agent._blocked_cells.add((r, 7))
    action = agent._plan(_to_grid(g), [1, 2, 3, 4], (4.0, 10.0))
    assert action == 2  # first step is down toward the gap


def test_unreachable_goal_pursues_frontier():
    # Fully walled-in goal → planner falls back to nearest unvisited
    # passable anchor instead of ramming the wall.
    g = _world()
    agent = _seeded_agent((8, 8), g)
    for c in range(16):
        g[4][c] = 4
        agent._blocked_cells.add((4, c))
    agent._visited_cells.add((8, 8))
    action = agent._plan(_to_grid(g), [1, 2, 3, 4], (2.0, 8.0))
    # Goal unreachable → she pursues the nearest frontier instead of
    # returning nothing: any expansion move is the honest answer.
    assert action in (1, 2, 3, 4)


def test_fully_explored_returns_none():
    g = _world()
    agent = _seeded_agent((8, 8), g)
    for c in range(16):
        g[4][c] = 4
        agent._blocked_cells.add((4, c))
    for r in range(5, 16):
        for c in range(16):
            agent._visited_cells.add((r, c))
    assert agent._plan(_to_grid(g), [1, 2, 3, 4], None) is None


def test_hazard_cells_are_hard():
    g = _world()
    agent = _seeded_agent((5, 5), g)
    g[5][7] = 3  # hazard color between avatar and goal at (5,10)
    agent._hazard_colors.add(3)
    action = agent._plan(_to_grid(g), [1, 2, 3, 4], (5.0, 10.0))
    assert action != 4  # stepping into the hazard is never the plan
