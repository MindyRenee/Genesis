"""Play ARC-AGI-3 games with Genesis' exploration policy on one shared scorecard.

All games in a run share one toolkit ``Arcade`` (one API key and one
scorecard), and the scorecard is closed at the end so the run produces a
final score.
"""
from __future__ import annotations

import argparse
import json

from parietal_lobe.arc_play import run_arc3_episode
from parietal_lobe.arc_policy import ARC3ExplorationPolicy
from parietal_lobe.arc_sources import (
    arc3_arcade,
    close_arc3_scorecard,
    list_arc3_games,
    open_arc3_game,
)


def main() -> int:
    """Run the policy on the requested games and print per-game results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", nargs="*", help="game ids (default: every game)")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-recording", action="store_true")
    args = parser.parse_args()

    arcade = arc3_arcade(offline=args.offline)
    try:
        game_ids = args.games or [g["game_id"] for g in list_arc3_games(arcade=arcade)]
        for game_id in game_ids:
            env, observation = open_arc3_game(
                game_id, arcade=arcade, save_recording=not args.no_recording
            )
            policy = ARC3ExplorationPolicy()
            episode = run_arc3_episode(
                env,
                game_id=game_id,
                initial_observation=observation,
                choose_action=policy.choose_action,
                max_steps=args.max_steps,
            )
            print(json.dumps({
                "game_id": game_id,
                "actions": episode.actions_taken,
                "resets": episode.resets,
                "levels_completed": episode.levels_completed,
                "win_levels": episode.win_levels,
                "final_state": episode.final_state,
                "stopped_reason": episode.stopped_reason,
                "distinct_states": len({t["next_state_signature"] for t in episode.transitions}),
            }), flush=True)
    finally:
        scorecard = close_arc3_scorecard(arcade)
        print(json.dumps({"scorecard": getattr(scorecard, "card_id", None),
                          "score": getattr(scorecard, "score", None)}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
