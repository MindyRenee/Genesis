# ARC grid-play evidence — live state

Generated 2026-10-15 13:41 from `/home/mindy/.local/share/genesis` at repo `605598a` by `scripts/arc_play_evidence.py`.

Genesis plays public **ARC-AGI-1 training tasks** inside her games curriculum (`python/frontal_lobe/games/`). Each task is offered with its train pairs and test inputs; her spatial reasoner searches compositions of grid transforms, verifies candidates on train, and guesses on the held-out test input. The stored test answer is used for scoring only — `solve()` receives train pairs and test inputs (`play.py:1366-1375`); `expected` never enters the search — it only scores the guesses afterward (`play.py:1380-1388`). Mastery 1.00 requires every test cell exact.

**Scope, stated plainly:** these are *training-set* tasks — the easiest public ARC split — solved in live play, not under the official evaluator. This is evidence of real program synthesis and skill transfer, not a claim that Genesis passed ARC-AGI.

Offered: **25** · solved: **3** · attempted but unsolved: **5** · unplayed: **17**

## Results

| task | outcome | tries | mastery | winning/best rule | nodes | last played | id-copy | copy-train0 | 1-color |
|---|---|---|---|---|---|---|---|---|---|---|
| 496994bd | solved | 1 | 1.00 | `skill:bf05eadfd36dbc6d(skill=bf05eadfd36dbc6d)` | 39 | 2026-10-15 13:19 | 70% | 0% | 40% |
| 7468f01a | solved | 1 | 1.00 | `crop_to_content → reflect_h` | 343 | 2026-10-14 22:41 | 0% | 0% | 84% |
| f25ffba3 | solved | 1 | 1.00 | `symmetrize_h(axis=h)` | 53 | 2026-10-15 13:14 | 65% | 20% | 30% |
| 6455b5f5 | missed | 1 | 0.00 | `fill_enclosed(color=1) → skill:cc5196f52cfc7f…` | 1857 | 2026-10-15 13:09 | 67% | 0% | 42% |
| 7f4411dc | missed | 1 | 0.00 | `attract_marks → identity` | 1094 | 2026-10-15 13:03 | 90% | 0% | 79% |
| 8a004b2b | missed | 1 | 0.00 | `select_color(color=4) → fill_enclosed(color=1)` | 507 | 2026-10-14 22:21 | 0% | 0% | 60% |
| af902bf9 | missed | 1 | 0.00 | `fill_enclosed(color=2) → fill_enclosed(color=2)` | 1475 | 2026-10-15 13:27 | 87% | 76% | 79% |
| b91ae062 | missed | 2 | 0.00 | `tile(reps=(2, 2)) → gravity_up(direction=up)` | 254 | 2026-10-14 22:00 | 0% | 0% | 44% |
| 007bbfb7 | unplayed | 0 | 0.00 | `—` | 0 | — | 0% | 35% | 56% |
| 0dfd9992 | unplayed | 0 | 0.00 | `—` | 0 | — | 88% | 37% | 16% |
| 1e32b0e9 | unplayed | 0 | 0.00 | `—` | 0 | — | 81% | 50% | 53% |
| 23b5c85d | unplayed | 0 | 0.00 | `—` | 0 | — | 0% | 0% | 100% |
| 2dc579da | unplayed | 0 | 0.00 | `—` | 0 | — | 0% | 0% | 97% |
| 3aa6fb7a | unplayed | 0 | 0.00 | `—` | 0 | — | 92% | 61% | 67% |
| 42a50994 | unplayed | 0 | 0.00 | `—` | 0 | — | 95% | 0% | 82% |
| 5582e5ca | unplayed | 0 | 0.00 | `—` | 0 | — | 33% | 0% | 100% |
| 6cdd2623 | unplayed | 0 | 0.00 | `—` | 0 | — | 79% | 0% | 89% |
| 93b581b8 | unplayed | 0 | 0.00 | `—` | 0 | — | 67% | 28% | 56% |
| 9dfd6313 | unplayed | 0 | 0.00 | `—` | 0 | — | 61% | 0% | 64% |
| a68b268e | unplayed | 0 | 0.00 | `—` | 0 | — | 0% | 38% | 62% |
| c3f564a4 | unplayed | 0 | 0.00 | `—` | 0 | — | 85% | 6% | 12% |
| cf98881b | unplayed | 0 | 0.00 | `—` | 0 | — | 0% | 31% | 44% |
| d511f180 | unplayed | 0 | 0.00 | `—` | 0 | — | 56% | 0% | 31% |
| dc433765 | unplayed | 0 | 0.00 | `—` | 0 | — | 98% | 0% | 98% |
| e8593010 | unplayed | 0 | 0.00 | `—` | 0 | — | 72% | 53% | 72% |

Floor columns are trivial predictors scored on the same held-out answer: output=input, output=first train output, and best single-color fill *at the true output shape* (a generous upper bound on guessing one color).

## Verbatim traces

- `496994bd` 2026-10-15 13:19: score=1.00 `skill:bf05eadfd36dbc6d(skill=bf05eadfd36dbc6d)` (39 nodes, felt=satisfied)
- `7468f01a` 2026-10-14 22:41: score=1.00 `crop_to_content → reflect_h` (343 nodes, felt=satisfied)
- `f25ffba3` 2026-10-15 13:14: score=1.00 `symmetrize_h(axis=h)` (53 nodes, felt=satisfied)
- `6455b5f5` 2026-10-15 13:09: score=0.00 `fill_enclosed(color=1) → skill:cc5196f52cfc7f1d(skill=cc5196f52cfc7f1d)` (1857 nodes, felt=determined, why=fill_enclosed(color=1) → skill:cc5196f52cfc7f1d(skill=cc5196f52cfc7f1d) broke on pair 1: 80 diffs (0->1×72, 0->8×8))
- `7f4411dc` 2026-10-15 13:03: score=0.00 `attract_marks → identity` (1094 nodes, felt=determined, why=attract_marks → identity broke on pair 2: 3 diffs (5->0×3))
- `8a004b2b` 2026-10-14 22:21: score=0.00 `select_color(color=4) → fill_enclosed(color=1)` (507 nodes, felt=determined, why=select_color(color=4) → fill_enclosed(color=1) broke on pair 0: 24 diffs (0->1×24))
- `af902bf9` 2026-10-15 13:27: score=0.00 `fill_enclosed(color=2) → fill_enclosed(color=2)` (1475 nodes, felt=determined, why=fill_enclosed(color=2) → fill_enclosed(color=2) broke on pair 1: 16 diffs (0->2×16))
- `b91ae062` 2026-10-14 22:00: score=0.00 `tile(reps=(2, 2)) → gravity_up(direction=up)` (254 nodes, felt=determined, why=tile(reps=(2, 2)) → gravity_up(direction=up) broke on pair 2: wrong shape)
- `b91ae062` 2026-10-14 23:02: score=0.00 `gravity_up(direction=up) → tile(reps=(2, 2))` (254 nodes, felt=determined, why=gravity_up(direction=up) → tile(reps=(2, 2)) broke on pair 2: wrong shape)

## Learning curve — the curriculum underneath

Same-task replays, oldest first. Falling node counts are her search getting cheaper; `skill:` in later rules is a learned procedure replacing search outright — the mechanism that let her solve `496994bd` cold.

| game | family | plays | nodes searched per play | skill reuses |
|---|---|---|---|---|
| abab_beads | sequence | 3 | 3 → 7 → 3 | 0 |
| color_sorter | sorter | 3 | 7 → 5 → 8 | 0 |
| directions | relations | 3 | 5 → 4×2 | 0 |
| fill_margin | grid | 2 | 98 → 21 | 1 |
| gravity_intro | grid | 6 | 48 → 24 → 26 → 24 → 25 → 24 | 0 |
| gravity_variation | grid | 3 | 44 → 16×2 | 2 |
| jigsaw_rows | assembly | 4 | 60 → 42 → 98 → 34 | 0 |
| letters_ab | sequence | 2 | 3×2 | 0 |
| make_five | quantities | 2 | 6 → 2 | 0 |
| make_nine | quantities | 3 | 12 → 3 → 2 | 0 |
| mixed_beads | sequence | 3 | 10 → 7 → 6 | 0 |
| pacman | system | 2 | 0 → 10 | 0 |
| pinwheel | grid | 3 | 28×3 | 0 |
| silhouette_stretch | grid | 88 | 522 → 526×2 → 547×2 → 646×3 → 651×2 → 522 → 24×2 → 472 → 437×33 → 22 → 437×6 → 473 → 437×10 → 473 → 437×22 | 2 |

## What this is not

- Not the official ARC-AGI-1 evaluation split or evaluator.
- Not ARC-AGI-3 — that boundary lives in `docs/ARC-AGI-3.md`.
- Not a cherry-pick: every offered task is listed, including misses and unplayed ones, with failed candidate rules retained in `gameplay.json` (`failed`, `failed_predictions` ledgers).
