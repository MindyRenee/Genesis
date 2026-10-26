"""GamePlayer — the one player.

Every earlier game machinery hardcoded a solver per game. This is
the honest version: a single player over the GameEnvironment
interface, the way one brain plays solitaire, Pac-Man, or a puzzle
it's never seen. It knows nothing about the game's rules — it
perceives the state, tries the actions the world exposes, and
learns which actions pay, through the shared competence substrate.

The learning is real and general: tabular value learning over
perceived states (Q-learning — Watkins & Dayan, 1992). The state
key is a structural hash of the grid parse, so situations that
*look the same* share values; the reward is the environment's own
verdict, not a hint. Exploration is epsilon-greedy with a novelty
bonus for actions never tried in a state — curiosity before
exploitation, the way a person pokes a new game's controls.

Persistent value memory lives in ``TaskCompetence``'s per-family
skill records — "how she plays pacman" is a skill like any other,
and it survives across sessions the same way.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from worlds.environment import GameEnvironment, GameObservation

from genesis_client.swallow import note_swallowed


def _state_key(obs: GameObservation) -> str:
    """A structural hash of the perceived state.

    The grid parse is the perception — identical-looking boards are
    the same situation to her. Text state joins the key when the
    world reports any.
    """
    h = hashlib.blake2b(digest_size=12)
    if obs.grid is not None:
        for row in obs.grid.rows():
            h.update(bytes(b % 256 for b in row))
    if obs.text:
        h.update(obs.text.encode("utf-8", "replace")[:200])
    return h.hexdigest()


@dataclass
class PlayResult:
    """One episode: how long she played, how it went, what she felt."""

    game: str
    steps: int
    score: float
    reward: float  # total environment reward
    won: bool = False
    cause: str = ""  # why the episode ended
    learned: int = 0  # state→action values updated this episode


@dataclass
class PlayStep:
    """One move and the world's answer — the unit play is felt in.

    Play engages a brain at event granularity, not episode
    granularity: in-game events (a pickup, a hit, a narrow escape)
    each drive discriminable distributed responses — reward,
    salience, and defensive circuitry (event-related fMRI of retro
    gameplay, PMC13228873). ``td_error`` is the reward prediction
    error the Q update just applied — the same teaching signal
    midbrain dopamine encodes (Schultz 1998; Koepp et al., Nature
    1998, showed real play releases striatal dopamine). ``threat``
    is the environment's report of danger imminence — the gradient
    behind the predatory-imminence response (Fanselow; Mobbs et al.,
    Science 2007). Emitted per step so her chemistry, oscillations,
    and inner voice can follow the game live.
    """

    step: int
    action: str
    why: str  # "explore" | "exploit" — her reason for the move
    state: str
    reward: float
    td_error: float
    score: float
    threat: float = 0.0  # danger imminence the world reports
    terminal: bool = False
    events: tuple[str, ...] = ()  # "death", "escaped", "scored"...


@dataclass
class GamePlayer:
    """A general game player — perceive, choose, act, learn.

    One value table across every environment she plays: the key is
    (game, state-hash), so skills stay per-game while the player —
    the capability — stays singular.
    """

    epsilon: float = 0.25  # exploration rate
    alpha: float = 0.3  # learning rate
    gamma: float = 0.95  # discount — future rewards count
    _values: dict[str, dict[str, float]] = field(default_factory=dict)

    def value(self, game: str, state: str, action: str) -> float:
        return self._values.get(f"{game}:{state}", {}).get(action, 0.0)

    def _update(self, game: str, state: str, action: str, target: float) -> bool:
        """One Q update. Returns True when the value actually moved."""
        key = f"{game}:{state}"
        row = self._values.setdefault(key, {})
        old = row.get(action, 0.0)
        new = old + self.alpha * (target - old)
        if abs(new - old) < 1e-9:
            return False
        row[action] = new
        return True

    def choose(
        self,
        game: str,
        obs: GameObservation,
        actions: list[str],
        *,
        rng: random.Random,
    ) -> tuple[str, str, str]:
        """Pick an action — exploit known value, else explore.

        Returns (action, state_key, why). Novel actions in a state
        get an optimistic tie-break so untried moves draw her before
        familiar ones do.
        """
        state = _state_key(obs)
        known = self._values.get(f"{game}:{state}", {})
        if actions and rng.random() < self.epsilon:
            return rng.choice(actions), state, "explore"
        best_val = max((known.get(a, 0.0) for a in actions), default=0.0)
        # Untried actions read as 0 — tie with the incumbent best on
        # a fresh state, then novelty wins by preferring them.
        untried = [a for a in actions if a not in known]
        tied = [a for a in actions if known.get(a, 0.0) == best_val]
        pool = [a for a in untried if a in tied] or tied or actions
        return (
            (rng.choice(pool) if pool else actions[0]),
            state,
            ("explore" if pool == untried else "exploit"),
        )

    def play(
        self,
        env: GameEnvironment,
        *,
        max_steps: int = 400,
        rng: random.Random | None = None,
        time_budget: float = 120.0,
        on_step: Any = None,
        should_stop: Any = None,
    ) -> PlayResult:
        """One episode — play until the world says it's over.

        The loop is the whole capability: observe → choose → act →
        feel the world's answer → remember. No rule knowledge enters
        it; everything she knows about this game comes back through
        reward.

        ``on_step`` receives a :class:`PlayStep` after every move —
        the live event stream her chemistry, oscillations, and inner
        voice ride on. ``should_stop`` is the interrupt: polled
        between moves, a true answer ends the episode at the next
        step boundary (cause ``"interrupted"``) — she finishes the
        current move and puts the game down.
        """
        chooser = rng or random.Random()
        game = env.name
        obs = env.reset()
        total = 0.0
        learned = 0
        steps = 0
        cause = "finished"
        import time as _t

        deadline = _t.monotonic() + time_budget
        prev_threat = 0.0
        while steps < max_steps and not env.done():
            if _t.monotonic() > deadline:
                cause = "time"
                break
            if should_stop is not None and should_stop():
                cause = "interrupted"
                break
            action, state, why = self.choose(game, obs, env.actions(), rng=chooser)
            outcome = env.act(action)
            steps += 1
            total += outcome.reward
            obs = outcome.observation

            # Q(s,a) ← r + γ·max Q(s',·) — the value of what just
            # happened plus the best the next state promises.
            next_state = _state_key(obs)
            nxt = self._values.get(f"{game}:{next_state}", {})
            future = max(nxt.values()) if nxt else 0.0
            target = outcome.reward + (0.0 if outcome.terminal else self.gamma * future)
            old = self._values.get(f"{game}:{state}", {}).get(action, 0.0)
            td_error = target - old
            if self._update(game, state, action, target):
                learned += 1

            if on_step is not None:
                prev_threat = self._notify_step(
                    on_step,
                    step=steps,
                    action=action,
                    why=why,
                    state=state,
                    outcome=outcome,
                    td_error=td_error,
                    score=env.score(),
                    prev_threat=prev_threat,
                )

        won = outcome.info.get("won", False) if steps else False
        if env.done() and cause == "finished":
            cause = "game_over"
        return PlayResult(
            game=game,
            steps=steps,
            score=env.score(),
            reward=total,
            won=won,
            cause=cause,
            learned=learned,
        )

    def _notify_step(
        self,
        on_step,
        *,
        step: int,
        action,
        why: str,
        state,
        outcome,
        td_error: float,
        score: float,
        prev_threat: float,
    ) -> float:
        """Emit the PlayStep for the move just taken; returns the
        threat to carry into the next step."""
        threat = float(outcome.info.get("threat", 0.0))
        events: list[str] = []
        if outcome.info.get("death") or (outcome.terminal and outcome.reward < 0):
            events.append("death")
        elif prev_threat >= 0.55 and threat <= 0.25:
            # The danger was closing in and passed — averted
            # punishment, which the brain rewards as relief.
            events.append("escaped")
        if outcome.reward >= 0.2:
            events.append("scored")
        if outcome.terminal:
            events.append("won" if outcome.info.get("won") else "over")
        try:
            on_step(
                PlayStep(
                    step=step,
                    action=action,
                    why=why,
                    state=state,
                    reward=outcome.reward,
                    td_error=td_error,
                    score=score,
                    threat=threat,
                    terminal=outcome.terminal,
                    events=tuple(events),
                )
            )
        except Exception as e:  # noqa: BLE001 — a listener must
            note_swallowed(  # never kill her game
                "worlds.player.listener", e
            )
        return threat

    def skills(self, game: str) -> int:
        """How many situations she has valued for this game."""
        prefix = f"{game}:"
        return sum(1 for k in self._values if k.startswith(prefix))
