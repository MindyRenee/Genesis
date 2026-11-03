"""Genesis plays ARC-AGI-3.

This adapter hands every decision to Genesis' own ARC-AGI-3 policy
(``parietal_lobe.arc_policy.ARC3ExplorationPolicy`` from
https://github.com/MindyRenee/Genesis, branch ``genesis3``): a math-only
state-graph explorer with volatile-pixel masking, basal-ganglia action
release gated by TD dopamine RPE, GABA no-op inhibition, the EM field ->
brain waves -> chemicals loop (``ArcNeuroLoop``), and hippocampal replay of
the previous level's shortest solution. No language model, no internet, no
per-game code.

The Genesis source is bundled into the notebook and unpacked to
``GENESIS_SRC`` (default ``/tmp/genesis_src``) before this module loads.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

_GENESIS_SRC = os.environ.get("GENESIS_SRC", "/tmp/genesis_src")
for _path in (os.path.join(_GENESIS_SRC, "python"), _GENESIS_SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from parietal_lobe.arc_play import transition_reward  # noqa: E402
from parietal_lobe.arc_policy import ARC3ExplorationPolicy  # noqa: E402

logger = logging.getLogger()

RESET_STATES = (GameState.NOT_PLAYED, GameState.GAME_OVER)


class MyAgent(Agent):
    """Genesis: graph exploration driven by her basal ganglia and neurochemistry."""

    MAX_ACTIONS = int(os.environ.get("GENESIS_MAX_ACTIONS", "2500"))
    MAX_SECONDS = float(os.environ.get("GENESIS_MAX_SECONDS", "1800"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy = ARC3ExplorationPolicy()
        self._previous: FrameData | None = None
        self._previous_action: GameAction | None = None
        self._previous_data: dict[str, Any] | None = None
        self._started = time.time()

    @property
    def name(self) -> str:
        return f"{super().name}.genesis.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return time.time() - self._started > self.MAX_SECONDS

    def _observe(self, latest_frame: FrameData) -> None:
        previous, action = self._previous, self._previous_action
        if previous is None or action is None or action is GameAction.RESET:
            return
        if previous.state in RESET_STATES:
            return
        state_after = getattr(latest_frame.state, "name", str(latest_frame.state))
        reward = transition_reward(
            previous.levels_completed or 0, latest_frame.levels_completed or 0, state_after
        )
        try:
            self.policy.observe_transition(
                previous,
                action.name,
                latest_frame,
                state_after,
                data=self._previous_data,
                reward=reward,
            )
        except Exception:  # never let a learning error stop play
            logger.exception("genesis observe_transition failed")

    def _remember(self, latest_frame: FrameData, action: GameAction, data: Any) -> GameAction:
        self._previous = latest_frame
        self._previous_action = action
        self._previous_data = dict(data) if data else None
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        self._observe(latest_frame)
        if latest_frame.state in RESET_STATES:
            action = GameAction.RESET
            action.reasoning = {"policy": "genesis", "reason": "start_or_recover"}
            return self._remember(latest_frame, action, None)

        legal = [GameAction.from_id(int(a)).name for a in (latest_frame.available_actions or [])]
        legal = [name for name in legal if name != "RESET"]
        if not legal:
            legal = [a.name for a in GameAction if a is not GameAction.RESET]
        try:
            decision = self.policy.choose_action(latest_frame, legal, [])
            name, data = decision.action, decision.data
            reasoning = decision.reasoning
        except Exception:
            logger.exception("genesis choose_action failed; resetting")
            name, data, reasoning = "RESET", None, {"policy": "genesis", "reason": "error"}

        action = GameAction.from_name(name)
        if action.is_complex():
            data = data or {"x": 32, "y": 32}
            action.set_data({"x": int(data["x"]), "y": int(data["y"])})
        else:
            data = None
        action.reasoning = {
            "policy": "genesis",
            "mode": (reasoning or {}).get("mode"),
            "dopamine_rpe": (reasoning or {}).get("dopamine_rpe"),
        }
        return self._remember(latest_frame, action, data)
