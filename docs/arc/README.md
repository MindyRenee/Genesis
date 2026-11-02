# Official ARC resources for Genesis

This integration supports two different benchmark formats:

- **ARC-AGI-1** is a static grid-transformation benchmark. Tasks contain demonstration input/output grids and one or more test inputs. Answers must match the exact grid, including dimensions, colors, and cell positions.
- **ARC-AGI-3** is an interactive environment benchmark. Genesis receives observations, chooses actions, and uses feedback to learn the environment and make progress.

## Official upstream sources

- ARC-AGI-1 dataset and human testing interface: https://github.com/fchollet/ARC-AGI
- ARC-AGI-1/2 format guide: https://arcprize.org/guide/1
- ARC-AGI-3 toolkit and environment API: https://github.com/arcprize/ARC-AGI
- ARC-AGI-3 documentation: https://docs.arcprize.org/

The ARC-AGI-3 Python toolkit is installed as an optional dependency pinned in pyproject.toml. It is a game/environment API, not a reasoning model. Genesis' reasoning and action-selection logic remains separate.

## Install

Run from the Genesis repository root:

    bash scripts/install_arc_resources.sh

The script clones the official ARC-AGI-1 repository to data/arc/ARC-AGI-1. On desktop Python it installs Genesis with the optional official ARC-AGI-3 toolkit; in Termux it installs the toolkit directly to avoid forcing all desktop-oriented base dependencies. The dataset is kept outside source control; do not commit downloaded benchmark data.

Set GENESIS_ARC1_DIR if the dataset clone should live elsewhere. Set ARC_API_KEY to a key from https://three.arcprize.org if you need additional ARC-AGI-3 games available to registered users. The toolkit can use an anonymous key when none is set, but available games may be limited.

## ARC-AGI-1 solving

ARC-AGI-1 is connected to Genesis' existing `SpatialReasoner`; this is not a separate model. From the repository root with the Python package installed:

    from parietal_lobe.arc_solver import solve_arc1_task
    from parietal_lobe.arc_sources import list_arc1_tasks

    task_path = list_arc1_tasks("training")[0]
    result = solve_arc1_task(task_path)
    print(result.solved, result.rule, result.predictions)

The solver converts the demonstration grids to Genesis `Grid` objects and searches the existing symbolic transformation space. It returns predictions only when the selected rule exactly reproduces every training pair. An unsolved task returns diagnostic information rather than claiming success. Search limits can be configured through `max_depth`, `beam_width`, `max_nodes`, and `time_budget`.

You can run the solver directly from the repository root, including in Termux:

    python scripts/arc1_solve.py
    python scripts/arc1_solve.py --split training --index 0
    python scripts/arc1_solve.py data/arc/ARC-AGI-1/data/training/007bbfb7.json
    python scripts/arc1_benchmark.py --split evaluation --time-budget 3

`arc1_solve.py` reports every guess (`--num-guesses`, default 2, matching the official two-attempt rule) and, when the task file includes test outputs, whether the first guess and any guess are correct. `arc1_benchmark.py` runs a whole split and reports pass@1, pass@2, unsolved tasks, and verified-but-wrong answers (rules that fit every demonstration but miss the test). Among rules that fit the demonstrations equally, the solver prefers the fewest transforms and then the smallest parameter description, counting every entry of a lookup table, so memorised colour maps and shape tables rank below compact rules.

The command prints JSON with the exact-verification status, rule, predictions, search cost, and failure diagnostics. Exit status is 0 for a verified solve, 1 for an unsolved task, and 2 for an input/setup error. The default search budget is deliberately bounded; an unsolved result does not prove a task is impossible.

## Python API

    from parietal_lobe.arc_sources import (
        list_arc1_tasks, load_arc1_task, arc1_problem_view,
        list_arc3_games, open_arc3_game, step_arc3_game,
    )
    from parietal_lobe.arc_play import run_arc3_episode
    from parietal_lobe.arc_policy import ARC3ExplorationPolicy

    task_path = list_arc1_tasks("training")[0]
    task = load_arc1_task(task_path)
    problem = arc1_problem_view(task)  # test outputs omitted by default

    games = list_arc3_games()
    environment, observation = open_arc3_game(games[0]["game_id"])
    # Graph-frontier exploration over observed boards; exploratory, not a
    # task-solving guarantee.
    policy = ARC3ExplorationPolicy()
    episode = run_arc3_episode(
        environment,
        game_id=games[0]["game_id"],
        initial_observation=observation,
        choose_action=policy.choose_action,
        max_steps=100,
        outer_world=mind.world,  # routes observations/actions through OuterWorld
    )

## Integrity and evaluation rules

- ARC-AGI-1 grids are validated as rectangular grids with integer colors 0..9 and maximum dimensions 30 × 30.
- Test outputs are omitted by default from arc1_problem_view so the agent-facing view does not reveal answers.
- Keep training and evaluation splits distinct. Develop on training tasks; reserve evaluation tasks for controlled measurements. Repeatedly tuning against evaluation scores makes them unsuitable as unbiased final measurements.
- Retain ARC-AGI-3 observations, actions, and recordings for reproducibility. Never expose API credentials in source files, recordings, logs, or commits.
- The bounded ARC-AGI-3 loop checks state-dependent legal actions, retains per-step transitions (board, action, coordinates, level counts, reward), recovers from `GAME_OVER` with `RESET` (configurable via `reset_on_game_over` / `max_resets`), and stops on `WIN` or a step budget.

## ARC-AGI-3 play

Run games from the repository root (Python 3.12+, toolkit installed):

    PYTHONPATH=python python scripts/arc3_play.py ls20 ft09 --max-steps 300

All games in one run share a single toolkit `Arcade`, so one API key and one scorecard; the scorecard is closed at the end and its id and score are printed. Pass `arcade=` to `list_arc3_games` / `open_arc3_game` to do the same from Python, and close it with `close_arc3_scorecard`.

What the integration relies on from the toolkit (`arc-agi==0.9.9`):

- **Board.** Observations are `arcengine.FrameDataRaw`. The board is the private `frame` field (a list of 64x64 NumPy layers, colours 0..15) and is not part of the public model fields, so generic serializers drop it. `observation_grid`, `observation_frames`, `observation_view`, and `observation_percept` in `arc_sources` extract it; the policy signature, Genesis `OuterWorld` percepts (`{"grid": ...}`), ASCII transition records, and phone-bridge JSON (`board`) all use it. Session ids, timestamps, and echoed action inputs are excluded from state identity.
- **States.** `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`. `WIN` means every level is cleared. After `GAME_OVER` the only valid action is `RESET`, which restarts the current level; the episode loop and the bridge both handle this.
- **Actions.** `ACTION1`..`ACTION5` and `ACTION7` take no data. `ACTION6` is a click and requires `{"x": column, "y": row}` with integers 0..63; `step_arc3_game` rejects it without coordinates (the toolkit would otherwise click (0, 0)).
- **Scoring.** Official ARC-AGI-3 scoring is per completed level, `(human_actions / agent_actions)^2` capped at 1.15, weighted toward later levels; every submitted action counts, `RESET` included. Episodes therefore record `levels_completed`, `win_levels`, `resets`, and `actions_taken`, and the loop rewards each transition with +1 per level completed, +1 for `WIN`, and -1 for `GAME_OVER`.

`ARC3ExplorationPolicy` builds a graph of observed board states. In each state it tries untested actions first (each click target counts as its own action; targets are visible objects, smallest first, then the centres of an 8x8 tiling for games that react to clicks inside large regions), weighting actions by how often they changed the board and by observed reward. When the current state has nothing left to try it walks the shortest known path to a state that still has untested actions.

## Her brain during ARC-AGI-3 play

The policy is pixel and graph math only; these existing organs take part:

- **Outer world** — `worlds.arc_arena.ArcArena` plays each game inside her `OuterWorld`: boards arrive via `perceive`, moves leave via `it_acted`, and both update her experiential latent space. `scripts/arc3_play.py --no-world` opts out.
- **Basal ganglia** — `basal_ganglia.BasalGanglia` releases the final action from the scored candidates, gated by her dopamine RPE (TD(0) on masked board states). Moves known to change nothing are GABA-inhibited.
- **EM field -> brain waves -> chemicals** — `parietal_lobe.arc_neuro.ArcNeuroLoop` runs one closed loop per move: game events release dopamine/norepinephrine/GABA, `thalamus.brain_waves.BrainWaveOscillator` turns chemicals into band powers, and `hypothalamus.transduction.wave_impulses` lets the bands release their peptides. An optional field source holds measured EM levels on the oscillator (`em_field: absent` otherwise). Wave `focus` scales how much she trusts learned state values.
- **Procedural replay** — at each new level she first retries the shortest solution of the previous level.

Every decision's `reasoning` carries this telemetry (`mode`, `dopamine_rpe`, `striatal_disinhibition`, `gaba_inhibited`, `neuro`).
