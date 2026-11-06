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
    # This baseline autonomously selects legal controls by state/action novelty.
    # It is exploratory, not a task-solving guarantee.
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
- The bounded ARC-AGI-3 loop provides closed-loop observation/action execution, checks state-dependent legal actions, retains per-step transitions, and stops on terminal states or a step budget. Supplying `outer_world=mind.world` routes each observation through `OuterWorld.perceive` and each executed action through `OuterWorld.it_acted`, which notifies the registered cognitive event listener. Each post-action observation, including terminal feedback, carries metadata linking it to the action and state transition. If OuterWorld rejects perception (for example, because Genesis is asleep), the loop stops rather than silently continuing without a recorded observation. Grid-shaped observations are exposed in the format the existing scene parser accepts. The loop accepts a caller-supplied policy and now includes `ARC3ExplorationPolicy`, a bounded model-free baseline that autonomously chooses among current legal controls using episode-local state/action visit counts. Its scores mean exploration priority, not success probability; it does not infer game rules, optimize a game objective, or guarantee a win. It uses recorded transitions to avoid repeating previously tried controls in the same observed state. When the bound policy supports `observe_transition`, the loop passes explicit `WIN` and `GAME_OVER` outcomes back to it; the baseline records state/action-specific terminal feedback and uses it to adjust later action priorities. Nonterminal transitions are not assigned invented rewards. This is a minimal empirical adaptation mechanism, not a learned game model or a guarantee of strategic progress. ARC-AGI-1 has a bridge to the existing spatial reasoner, but neither benchmark integration nor these changes have been runtime-tested in Termux.


## Binary Mathematical Language substrate

`parietal_lobe.binary_language` adds an experimental six-bit symbolic substrate inspired by the *shape* of Braille cells, without importing Braille's human-assigned letter mapping. Each cell is a value from 0 to 63; cells can compose ordered, typed frames (observation, entity, property, action, relation, agent, patient, time, negation, question, unknown). `GroundingMemory` records competing pattern-to-referent hypotheses, contexts, supporting observations, and counterexamples. Its support ratio is bookkeeping, not a calibrated probability.

This is an encoding and grounding primitive, not a natural-language system by itself. It does not yet connect to Genesis' live language production, concept network, perception loop, or telemetry. A runtime integration must supply actual perceptual evidence, preserve the distinction between observation and hypothesis, and test whether learned associations generalize to held-out contexts before this is treated as language learning.
