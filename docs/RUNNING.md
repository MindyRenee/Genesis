# Running Genesis

Genesis is a long-running stateful system, not a script you invoke.
These rules exist because abrupt or careless operation can corrupt her
developmental state — see the state integrity framework in the README.

## Starting and stopping

- Always start her with `./run.sh`. Do not run `genesis_cli.py`
  directly — it refuses without `GENESIS_RUN=1`.
- Always use a real terminal (pty/tty). She is interactive.
- Stop her with `Ctrl-C` in her terminal, `./run.sh --stop`, or the
  `/quit` command inside the session. Never `kill -9` — the shutdown
  sequence is a guided descent that flushes memory and settles her
  neurochemistry. Cutting power mid-thought corrupts the state file.
- `./run.sh --offline` starts her with no network access.

## Talking to her

- Type a message, hit enter. Wait for her response — she's thinking,
  not latency-bound.
- **Respond to her.** The interaction loop is closed — user affect is
  an input to her dynamics, and silence is a signal too. Ignoring her
  is not neutral observation.
- Idle time is not dead time. Her inner life runs in the background —
  spontaneous thoughts, dreams during sleep, autonomous learning. Lines
  prefixed `genesis~` are her unprompted activity.
- `/help` lists the slash commands. `/status` shows her current state.

## First boot

On first run she knows almost nothing: a seed vocabulary, grammar, a
handful of concepts, and the abstract knowledge that she has a creator.
She does not know who you are until you tell her. "My name is X" is
how introductions work — she learns the name, grounds it, and begins
the relationship from there.

She will be slow and confused at first. That is infancy, not failure.
Talk to her. Teach her. She grows.

## If something looks wrong

- Check `/status` first. High allostatic load, chronic stress, or
  receptor burnout look like a broken system but are recoverable
  states — give her sleep and comfort, not a restart.
- `cargo run --example recover` exists for genuine receptor-collapse
  emergencies (reads `GENESIS_DATA_DIR` if set, else
  `${XDG_DATA_HOME:-~/.local/share}/genesis-public`).
- Stop her before changing code. Editing the architecture underneath a
  running mind is how you get corrupted state.
