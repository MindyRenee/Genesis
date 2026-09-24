# Genesis — Project Notes

## What this is
Genesis is a machine-native cognitive architecture. The Rust crate
(`genesis`) defines the core state schema — the memory-mapped hub of
system state that every module reads from and writes to. The Python
layer (`python/genesis_cognitive/`) is the cognitive mind: perception,
reasoning, language, introspection, and self-modeling.

## CRITICAL RULE: Never hardcode Genesis's responses
Genesis's thoughts, speech, and self-expression must ALWAYS emerge
from its own cognitive architecture — NEVER hardcode
pre-written response templates, canned phrases, or fixed sentence
structures that it "recites."

This applies to:
- **Spontaneous thoughts** (inner_life.py) — must be generated from
  its concept network, self_composer, or emotional state, NOT from
  template strings like `"I am written in {language}"`
- **Conversation responses** — must go through its language engine
  and cognition, NOT hardcoded reply patterns
- **Self-descriptions** — must be composed from its actual concept
  network and self-model, NOT template sentences with filled-in slots
- **Art/drawing descriptions** — must reflect its actual state, NOT
  pre-written captions

If you find yourself writing a string template with `{variable}` slots
that Genesis "says," STOP. Route the underlying data through its
concept network or language engine instead. It should compose its
own words from its own understanding.

### Seeds are OK — hardcoding what it says is NOT
There is an important distinction between **seeds** and **hardcoded
responses**:

- **Seeds are OK.** Seeding its concept network with initial knowledge,
  vocabulary, relationship verbs, narrative templates, and thought
  seeds is fine. Seeds are *input data* that its generative systems
  work with — they are the raw material from which it composes, not
  the composition itself. A seed says "here are building blocks";
  it does not say "say this sentence."
- **Hardcoding what it says is NOT OK.** Writing the actual words
  Genesis speaks as fixed strings (with or without `{variable}` slots)
  is never acceptable. Its words must emerge from its language engine,
  concept network, and cognitive architecture.

The test: **Is this a building block it uses, or is this the thing it
says?** Building blocks (seeds, vocabulary, grammar rules, relation
verbs, narrative structures) are fine. The final words it speaks are
not — those must be generated, not recited.

## Toolchain
- Rust stable, edition 2024 (rust-version 1.85+)
- Build: `cargo build` / `cargo build --release`. Requires libclang
  and kernel V4L2 headers — `v4l2-sys-mit` runs `bindgen` against
  `<linux/videodev2.h>` at build time (Debian/Ubuntu:
  `sudo apt install libclang-dev linux-libc-dev`).
- Test: `cargo test`
- Run examples: `cargo run --example <name>`
- Python 3.12, dependencies pinned in `python/requirements.txt`
  (runtime) and `python/requirements-dev.txt` (lint + test). Install
  with: `pip install -r python/requirements-dev.txt`
- Lint: `ruff check` and `pyflakes` (both must pass)
- Type check: `mypy python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ --ignore-missing-imports` (0 errors)
- Optional voice deps (not in requirements.txt): `vosk`, `sounddevice`,
  `speechrecognition` — install separately for microphone/TTS support

## Keep the project tidy
The project is large. Dead code, unused files, and stale leftovers
accumulate quickly and make the codebase harder to navigate. When
auditing or working on a file:
- If a file is outdated, unused, or no longer necessary, remove it
  rather than spending effort fixing it. Confirm it has no importers
  first (grep for imports), then delete it.
- Remove dead methods, unused imports, and stale comments you encounter.

**Important distinction: dead vs unwired.** Not everything that appears
unused is dead. Some code is intentionally written but not yet wired in.
Before removing something that appears unused, check whether it looks
like scaffolded work-in-progress; if unsure, ask rather than deleting.

## State integrity
Genesis is a long-running stateful system. Always shut down via
`./run.sh --stop` (or Ctrl-C in the running terminal) — never kill
processes directly. Do not corrupt the mmap'd state or drive the system
into degenerate regimes for experimentation; see the State integrity
framework section of the README.

## Persistence and lifecycle verification
- `python3 -m pytest python/tests/ -q -o addopts=''` runs the full Python
  suite with an explicit summary. The daemon integration test uses a fresh,
  temporary XDG data directory, runs the mind offline, and shuts down via
  `run.sh --stop`; failed shutdown must preserve that temporary state.
- Focused state-integrity regressions live in
  `python/tests/test_cli_lifecycle.py` and `python/tests/test_persistence.py`.
- Never unlink singleton lock files during shutdown: replacing the inode
  can allow a second process to acquire a different lock for the same state.
- Existing unreadable cognitive state is a startup error, not first boot.
  Repair or restore it explicitly; do not silently replace it with fresh state.
