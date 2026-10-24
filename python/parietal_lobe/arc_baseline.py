"""Non-agent baselines for ARC practice — the denominator.

A completed level means nothing alone: an environment where uniform
random play wins is not evidence of learning, and one where it never
moves is where every step she saves becomes information. These
policies answer the question a score can't answer for itself — what
does the floor look like on the same accounting?

They run through the *same* ``ArcPractice`` loop she does: same frame
translation, same legality checks, same step/reset/per-level
bookkeeping, same scorecard lifecycle. The only difference is what
sits behind ``world_step`` — a named reference policy wearing the
mind's interface instead of her cognition. It is not her: no model,
no learning, no game mode of its own, no readiness gate — the stub
answers ``ok`` so the shared loop runs, and its
``baseline:<policy>`` source stamp on every move keeps the recording
honest about who moved.

    /arc ls20-9607627b --baseline uniform --steps 400
    /arc --all --baseline first --steps 400
"""

from __future__ import annotations

import random
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from .arc import ArcEvaluation, ArcPractice, ArcResult

POLICIES = ("uniform", "first")

# frame, offered actions, rng -> action descriptor (None = abstain)
PolicyFn = Callable[[Any, list[dict[str, Any]], random.Random], dict[str, Any] | None]


def _grid_shape(frame: Any) -> tuple[int, int]:
    """(height, width) of the current grid — click bounds."""
    rows = frame if isinstance(frame, list) else getattr(frame, "rows", [])
    h = len(rows)
    w = len(rows[0]) if h else 0
    return h or 64, w or 64


def _uniform_policy(frame: Any, actions: list[dict], rng: random.Random) -> dict | None:
    """Uniform random over offered actions; uniform cell on distal acts."""
    if not actions:
        return None
    item = rng.choice(actions)
    desc: dict[str, Any] = {"action": item["action"]}
    if item.get("label") == "coordinate":
        h, w = _grid_shape(frame)
        desc["x"], desc["y"] = rng.randrange(w), rng.randrange(h)
    return desc


def _first_policy(frame: Any, actions: list[dict], rng: random.Random) -> dict | None:
    """Always the first offered action; the origin cell on distal acts."""
    if not actions:
        return None
    item = actions[0]
    desc: dict[str, Any] = {"action": item["action"]}
    if item.get("label") == "coordinate":
        desc["x"], desc["y"] = 0, 0
    return desc


_POLICY_FNS = {"uniform": _uniform_policy, "first": _first_policy}


class _BaselineMind:
    """A named policy answering the mind's contract — not her.

    Implements only what the adapter consumes: ``is_gaming`` so the
    game-mode contract is satisfied, ``readiness`` always ok (a
    reference measurement has no physiology to honor),
    ``world_step`` delegating to the policy, ``world_episode_end``
    a no-op, and ``game_decision`` stamped with the policy name so
    ``decision_sources`` tallies and recording provenance attribute
    every move to the baseline, never to her.
    """

    def __init__(self, name: str, fn: PolicyFn) -> None:
        self.is_gaming = True
        self._name = name
        self._fn = fn
        self._rng = random.Random(0)

    def readiness(self, *, require_body: bool = True) -> SimpleNamespace:
        return SimpleNamespace(ok=True, reasons=())

    def world_step(
        self,
        frame: Any,
        available_actions: list[Any] | None = None,
        info: dict[str, Any] | None = None,
        through: list[Any] | None = None,
    ) -> dict[str, Any] | None:
        # The reference policy is a fixed function of the settled
        # frame — no motion channel, so the strip is unused.
        return self._fn(frame, available_actions or [], self._rng)

    def world_episode_end(self, outcome: str, game: str = "the game") -> None:
        pass

    @property
    def game_decision(self) -> dict[str, str]:
        return {"action": self._name, "source": f"baseline:{self._name}"}


class ArcBaseline:
    """A named reference policy over the same environments and loop.

    ``policy`` is one of ``POLICIES``: ``uniform`` picks uniformly
    among offered actions (and uniformly over cells for
    coordinate-taking actions); ``first`` always takes the first
    offered action — the deterministic floor. Neither chooses RESET
    (it is never offered) and neither learns anything.
    """

    def __init__(
        self,
        policy: str = "uniform",
        environments_dir: str | None = None,
        recordings_dir: str | None = None,
    ) -> None:
        if policy not in _POLICY_FNS:
            raise ValueError(f"unknown baseline {policy!r} — expected one of {POLICIES}")
        self.policy = policy
        self._practice = ArcPractice(
            _BaselineMind(policy, _POLICY_FNS[policy]),
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
            instructions="",
            require_live=False,
        )

    def play(self, game_id: str, max_steps: int = 200) -> ArcResult:
        return self._practice.play(game_id, max_steps=max_steps)

    def play_all(self, max_steps: int = 200) -> ArcEvaluation:
        return self._practice.play_all(max_steps=max_steps)
