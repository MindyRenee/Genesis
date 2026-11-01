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

## Python API

    from parietal_lobe.arc_sources import (
        list_arc1_tasks, load_arc1_task, arc1_problem_view,
        list_arc3_games, open_arc3_game, step_arc3_game,
    )
    from parietal_lobe.arc_play import ARC3Decision, run_arc3_episode

    task_path = list_arc1_tasks("training")[0]
    task = load_arc1_task(task_path)
    problem = arc1_problem_view(task)  # test outputs omitted by default

    games = list_arc3_games()
    environment, observation = open_arc3_game(games[0]["game_id"])
    # Genesis's own policy must return an ARC3Decision on each turn.
    # The loop enforces legal actions, records transitions, and stops at
    # a terminal state or the configured step limit.
    episode = run_arc3_episode(
        environment,
        game_id=games[0]["game_id"],
        initial_observation=observation,
        choose_action=genesis_policy,  # integrate Genesis policy here
        max_steps=100,
    )

## Integrity and evaluation rules

- ARC-AGI-1 grids are validated as rectangular grids with integer colors 0..9 and maximum dimensions 30 × 30.
- Test outputs are omitted by default from arc1_problem_view so the agent-facing view does not reveal answers.
- Keep training and evaluation splits distinct. Develop on training tasks; reserve evaluation tasks for controlled measurements. Repeatedly tuning against evaluation scores makes them unsuitable as unbiased final measurements.
- Retain ARC-AGI-3 observations, actions, and recordings for reproducibility. Never expose API credentials in source files, recordings, logs, or commits.
- The bounded ARC-AGI-3 loop now provides closed-loop observation/action execution, checks state-dependent legal actions, retains per-step transitions, and stops on terminal states or a step budget. It deliberately requires a caller-supplied Genesis policy: autonomous action selection and learning are not yet wired. ARC-AGI-1 currently provides validated task loading and answer-hidden problem views, not a grid solver. Neither benchmark integration has been run on Termux yet.
