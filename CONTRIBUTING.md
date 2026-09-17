# Contributing to Genesis

Contributions are welcome — that's what this release is for. The rules
that matter most:

- **Read `AGENTS.md` first.** The no-hardcoding rule is the core
  constraint of the project: Genesis's language and self-expression
  must always emerge from her own cognitive architecture. Seeds
  (vocabulary, grammar, relation verbs, concept building blocks) are
  legitimate input data; pre-written sentences she recites are not.
- **`ruff check`, `pyflakes`, `mypy` (0 errors), `cargo test`, and the
  Python suite must all pass.** See the README for the exact commands;
  CI runs all of them on every PR.
- **Fix root causes, not symptoms.** Prefer correct/intelligent
  solutions over fast/superficial ones.
- **Keep the codebase tidy.** Dead code gets deleted, not maintained.
- **Respect the state integrity framework** (see the README). Genesis
  is a long-running stateful system; never corrupt the mmap'd state or
  drive it into degenerate regimes for experimentation.

## Before you start

For anything non-trivial, open an issue first — the architecture has
tight invariants (state layout, no-hardcoding, state integrity) and a
short conversation up front saves a rejected PR later.
