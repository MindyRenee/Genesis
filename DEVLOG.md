## 2026-11-05 — Fast-route discourse-state parity

The fast specialist route records user and assistant turns in working and
long-term conversation memory, but previously returned before the normal
post-response path updated the comprehension referent context and
`SituationModel`. A statement handled as an answer to a pending question
could therefore be remembered as dialogue while remaining absent from the
structured situation being discussed.

The turn sequencer now passes its already-computed `ComprehensionResult`
into the early-route handler. Both the fast and full paths call one shared
`_update_comprehension_context` operation, which refreshes recent subject
entities and applies the result to the situation model. The situation model
continues to distinguish assertions/commands from questions; this change
does not make questions mutate described world state.

Added a regression test for a fast-routed statement that places Mary in the
kitchen and verifies that the discourse model reflects that location.

**Verification boundary:** source and regression test are committed to
`audit-fixes`; tests have not run. A local clone attempt could not reach
GitHub because DNS/network access is unavailable in the execution
environment. GitHub reports no commit status checks.

## 2026-11-05 — Closed-archive zero-write guard

A follow-up persistence audit found that `ConceptArchive.archive_concepts_batch`
returns zero when its SQLite connection is closed instead of raising an
exception. `clean_noise()` previously checked only for exceptions, so a
closed-but-attached archive could be mistaken for a successful write and
the concepts could then be evicted from working memory without a durable
copy.

`clean_noise()` now requires the returned count to equal the full batch
size before archiving edges or removing concepts. The dormant spill commit
path applies the same exact-count invariant. Zero or partial counts leave
all candidates in working memory and emit an error. Added a regression
test for the closed-archive case.

**Verification boundary:** regression test and source are committed to
`audit-fixes`; execution remains pending. The branch's GitHub commit
status currently reports no checks.

## 2026-11-05 — Temporal grounding enters the cognitive attention stream

The comprehension stage already produced `GroundedProposition` bindings
from utterance arguments to existing concept IDs, but the turn sequencer
did not pass those bindings into perception. As a result, grounding was
computed and retained on `ComprehensionResult` without influencing the
topic stream consumed by memory retrieval, attention, and reasoning.

The turn handoff now passes the same turn's comprehension result into
the perception stage. The prefrontal integration adds only already-bound,
non-structural concept IDs to `Perception.topics`, deduplicating against
the surface-topic output. Unknown arguments remain unresolved and are not
minted as concepts. This makes temporal semantic binding available to the
downstream cognitive faculties without replacing independent perception.

Added an integration regression test that suppresses the independent
surface-topic extractor and verifies that a known multiword concept still
reaches the downstream topic stream through semantic grounding.

**Verification boundary:** source and test changes are committed to
`audit-fixes`. The test has not been executed in this remote edit session;
no CI result has been established. The test's parser and grounding
preconditions were checked against the source, but runtime behavior remains
to be verified by the required test suite.

## 2026-11-05 — Archive continuity and fail-closed maintenance audit

A persistence-path audit found that two maintenance fallbacks could delete
low-activation or low-confidence concepts when no archive was attached.
Both now fail closed: `clean_noise()` and lifecycle dormancy maintenance
leave the concepts in working memory and emit an error rather than treating
a missing durable store as permission to destroy developmental knowledge.

The archive recall path also consumed concept and edge rows while rebuilding
working memory, before the caller had necessarily saved that reconstruction.
Restoration is now copy-first: concepts and edges are read without consuming
the archive copy. The archive remains authoritative until a later validated
working-memory restore deduplicates it. Deduplication now merges archived
edges back into the live graph and reasserts them into the canonical edge log
before deleting archive rows; it retains the archive data if an endpoint is
missing or a read/write fails.

Spill now verifies that the archive accepted the complete edge batch before
retracting canonical edge-log entries. A zero/partial count or failed
retraction aborts eviction, leaving concepts in working memory and the
archived copy available for recovery.

Added regression coverage for missing-archive maintenance, copy-first
restoration, archive write-count failure, and edge recovery during
deduplication.

**Verification boundary:** source and regression tests are committed to
`audit-fixes`. The tests have not been executed in this remote edit session;
the available GitHub commit status reports no checks, and this environment
cannot clone the repository to run the suite locally. No Genesis runtime
state was accessed or modified.

## 2026-11-05 — Python mathematics routing type repair

The latest CI run failed its mypy step with nine errors across the frontal
math entry point and the parietal cognitive router. The failures came from
reusing local names for values with different types, leaving a regex match
insufficiently narrowed, and assigning an optional fallback result to a
variable already inferred as a non-optional frontal result.

Separated the fallback result from the parietal result, gave regex matches,
boolean predicates, and domain-faculty outputs distinct names, explicitly
narrowed the optional comparison match, and aliased the abstract-algebra and
complex-analysis description functions. Added focused end-to-end regression
coverage for number comparison, trigonometry, abstract algebra, complex
analysis, number theory, and combinatorics. No faculty, mathematical
operation, or static-analysis check was removed or weakened.

**Verification boundary:** source corrections and regression tests are
committed to `audit-fixes`. The previous CI failure is understood; the
corrected code still requires a fresh mypy run and the full required workflow.
No Genesis runtime state was accessed or modified.

## 2026-11-05 — Higher-math faculty reachability audit

The end-to-end entry-point audit found that several implemented native
faculties were only partially reachable through natural-language cognition.
Added explicit routes for angle-unit conversion and three-side triangle
solving; complex nth roots; linear congruences, Euler's totient, and Chinese
remainder systems; point-in-polygon queries; graph connectivity; Stirling
numbers of the second kind; and stars-and-bars counting. The frontal cue
boundary now admits these operation families, and modular congruences are
kept out of the polynomial-equation route.

Added cognitive-entry regression cases for each newly routed operation.
Inputs remain explicit and validated by the existing domain faculties;
the router does not synthesize missing vertices, congruence pairs, or
mathematical operands.

**Verification boundary:** source and tests are committed to
`audit-fixes`; they have not yet been executed in Termux or CI during this
remote edit session. No Genesis runtime state was accessed or modified.

## 2026-11-05 — Math routing regression audit

Termux execution of `test_higher_math_cognition.py` exposed dispatch gaps that
source review had not caught: percent-symbol questions were not admitted by
the frontal parietal cue; plural `matrices` phrasing fell through to the
arithmetic parser; differentiation and antiderivatives had no cognitive
router branch; and a limit containing `sin` could be claimed by the
trigonometry matcher before real analysis.

The frontal cue now admits `%`. The parietal router recognizes both
`matrix` and `matrices` for matrix addition/subtraction, dispatches
differentiation and indefinite integration through the native calculus
faculty, and prevents trigonometric matching from stealing explicit limit
questions. Existing end-to-end regression cases exercise these forms.

**Verification boundary:** changes are committed to `audit-fixes`; the
reported Termux failures have not yet been rerun against these corrections.
The remaining failure details from the original run were not included in the
report, so the full test result remains open. No runtime state was modified.

## 2026-11-04 — Linear-system input validation audit

A focused review of the new exact linear-algebra route found that
`solve_text()` iterated over the parsed right-hand side before confirming
that it was a one-dimensional vector. Scalar and nested-list inputs could
therefore escape the structured `LinearResult` failure contract as Python
exceptions. The route now reuses the shared vector-literal parser, which
validates non-empty one-dimensional input and retains exact Fraction
conversion. Added regression cases for scalar, nested, and mapping
right-hand sides.

**Verification boundary:** source and regression tests are committed to
`audit-fixes`; they have not yet been executed locally or in CI. No
persistent Genesis runtime state was accessed or modified.

## 2026-11-04 — ARC structural latent decoder and empirical dynamics

Added `python/worlds/games/latent.py` as a graph-backed decoder for live
ARC-AGI-3 board observations. It reuses the canonical parietal `Grid` and
`scene.perceive` implementation rather than creating a parallel image
segmentation path. The decoder validates ARC palette/dimensions, extracts
stable-ID connected components, records geometry and shape signatures,
computes enclosed-hole counts with a declared digital-topology convention,
and builds a sparse directed adjacency graph. A fixed 51-coordinate
interpretable feature embedding captures board geometry, occupancy,
component/topology statistics, spatial distribution, adjacency, color
frequencies, and the eight largest object sizes and radial positions.
The typed feature state projects into the existing experiential outer
latent domain `E`.

The arcade now sends the graph-backed latent summary and typed
`E`-domain metadata through `OuterWorld.perceive`. For each actual action/result pair,
the decoder updates a local action-conditioned model indexed by nearby
pre-action latent contexts. Welford statistics retain sample counts and
estimate uncertainty; predictions are withheld for unknown actions or
distant contexts. The model is serialized with each game's existing
`GameMemory`, so it survives the established world save/restore path.
Goal evidence reports observed level progress and official terminal state;
the official game state remains authoritative and no fixed target vector
is used to declare success. ARC action IDs are discrete opaque controls,
so no Lie-bracket closure or differentiable-operator claim is asserted;
completion remains grounded in the official terminal state.

Focused tests cover malformed grids, enclosed holes, stable object IDs,
latent dimensionality and typed-domain projection, local transition learning,
uncertainty, persistence, goal evidence, and the live arcade perception event.

**Verification boundary:** source, integration, tests, and manual changes are
committed to `audit-fixes`. They have not been executed in Termux or CI during
this remote edit session. No official ARC game was opened, no API key was
used, and no persistent runtime state was accessed or modified. The 51
coordinates are explicit structural descriptors, not a learned semantic
embedding or a complete environment simulator. The diagonal variance
tracks transition-model uncertainty, not visual detection confidence or
a full covariance matrix. Solving performance remains
to be measured on actual ARC-AGI-3 sessions.

## 2026-11-04 — Foundational arithmetic and linear algebra completion

Expanded the existing parietal linear-algebra faculty without replacing its
exact rational arithmetic. Added rectangular-matrix transpose, square-matrix
inverse and trace, matrix addition, subtraction, scalar multiplication,
plus exact vector addition, subtraction, dot product, and three-dimensional
cross product.
Exposed the operations through `ParietalMath` and routed explicit matrix and
vector literals from the frontal math entry point. Singular inverses,
non-square trace/inverse requests, malformed literals, and dimension
mismatches return explicit failures rather than guessed results.

Corrected a malformed whitespace escape in the foundational arithmetic
question recognizer. Added explicit spoken normalization for square roots
and “to the power of” expressions, and extended the number-word parser to
represent halves, thirds, quarters, other common unit fractions, and mixed
numbers such as “two and a half”. These retain the arithmetic parser's
precedence and validation. Added cognitive-entry tests for signed arithmetic,
numeric and spoken fractions, mixed numbers, precedence, powers, roots,
matrix operations, vector operations, and invalid dimensions.

**Verification boundary:** source and focused tests are committed to
`audit-fixes`. They have not been executed in Termux or CI during this
remote edit session. No persistent runtime state was accessed or changed.

## 2026-11-04 — Foundational arithmetic and linear algebra cognition

Extended the frontal-to-parietal math pathway after review found that the initial router omitted basic number operations and exposed only determinant/rank for matrix questions. The router now delegates explicit arithmetic expressions and spoken addition, subtraction, multiplication, division, percentages, halves, doubles, triples, squares, and cubes to the native exact arithmetic faculty. It also routes number-magnitude comparisons through NumberSense while retaining its confidence and exactness metadata. Linear algebra routing now covers determinant, rank, matrix multiplication, and unique square-system solving using explicit matrix/vector literals and the existing Fraction-based faculty. Algebraic equation solving is also exposed through the native algebra facade.

Expanded end-to-end tests for these paths and updated `genesis-math(7)` to enumerate their actual supported language forms and operand requirements. These additions preserve the existing frontal `MathResult` contract and do not replace the native faculties with canned responses or an external model.

**Verification boundary:** the remote GitHub edits are committed to `audit-fixes`. The full CI workflow must complete before treating the changes as verified; no local Termux test run or physical Android validation was performed during this edit session.

## 2026-11-04 — Higher mathematics cognitive integration

Connected the native parietal mathematics faculties to the existing frontal `try_math()` entry point through `parietal_lobe.math.cognitive.route_parietal_math()`. Supported explicit operation forms now dispatch to native algebra, calculus, statistics, trigonometry, linear algebra, abstract algebra, real analysis, complex analysis, number theory, geometry, discrete mathematics, combinatorics, topology, Cayley-table axiom classification, and structured propositional tautology/satisfiability checks. Results return through the existing `MathResult` contract so deliberation and task-competence recording retain one pathway. The router delegates computation to the existing faculties; it does not add canned answers, an external model, or invented operands. Propositional proofs still require the faculty's structured tuple representation; no natural-language proposition parser was added.

Added `test_higher_math_cognition.py` to exercise the frontal entry point and preserve the arithmetic path. Updated `genesis-math(7)` to describe the live routing layer and its explicit limits.

**Verification boundary:** source and focused tests are committed on `audit-fixes`, but Python tests, ruff, pyflakes, mypy, Rust checks, Android runtime, and physical-device behavior have not been executed in this remote editing session. Run the required project checks from Termux before treating the integration as verified. The referenced `AGENTS.md` is absent from this branch; `CONTRIBUTING.md`, `SECURITY.md`, and `DEVLOG.md` were used as the available project rules.

## 2026-11-04 — Man-page audit: optic, vision, and mathematical faculties

Reconciled the manuals against the current Android Camera2 adapter,
daemon optic mapping, mind-side perception modules, parietal math
package, and research workbench. Added `genesis-vision.7` to describe
the shared Linux/Android optic path, physical-camera configuration
limits, software illumination and ocular models, face detection versus
identity matching, and image-quality limitations. Corrected the daemon
and launcher manuals to remove the retired standalone-retina process
and obsolete `/genesis_retina` contract; retained
`genesis-retina.1` only as an explicit compatibility notice.

Expanded `genesis-math.7` to inventory the native algebra, calculus,
statistics, trigonometry, linear algebra, abstract algebra, real and
complex analysis, number theory, topology, geometry, discrete
mathematics, combinatorics, axiom checking, and propositional-proof
faculties. It distinguishes those endogenous faculties from the
optional SymPy verification oracle and records their actual limits.
Corrected the research workbench module docstring: symbolic
inequalities can be judged; bounded equality passes remain evidence,
not universal proof. The CLI manual now describes screen look-up,
inactive site-approval commands under the current all-domains policy,
and the difference between full text exploration and the automatic
man-page NAME-section genus pass.

**Verification boundary:** documentation and source descriptions were
cross-checked against the `audit-fixes` tree, and internal man-page
references were checked for targets. No Python, Rust, Android, or
physical-device test suite was run for this documentation-focused
change. The installer preserves an existing instance's installed
manuals and stages changed pages for Genesis's review rather than
silently overwriting her copies.

## 2026-11-04 — Android optic adapter selects the intended camera

The Android Camera2 adapter previously hard-coded `LENS_FACING_BACK` and
silently fell back to the first enumerated camera when no rear lens was
reported. It now defaults explicitly to `LENS_FACING_FRONT`, exposes front
and rear selection in the adapter activity, switches the active Camera2
device through the existing service, and reports the selected lens in its
status notifications. An unavailable requested lens is reported as an error;
the adapter no longer substitutes a different physical camera. Frame
transport remains the existing single daemon-owned optic nerve.

**Verification boundary:** source changes and the Android CI build are not
physical-device validation. Rebuild and reinstall the
`genesis-camera-bridge-debug` APK, then confirm that the notification reports
the front camera and that the captured frame shows the intended scene.

# Genesis Devlog

## 2026-11-04 — Android camera activation is wired into supported startup

The Termux launcher now opens the existing Android Camera2 hardware-boundary
activity as part of the normal `./genesis` startup. The activity requests the
runtime camera permission while visible and starts the foreground capture
service after permission is granted. This respects Android's camera
while-in-use boundary; Termux does not attempt to access Camera2 itself.
Missing APK, unavailable activity manager, or denied permission degrades
visual input without blocking the mind or creating an alternate perception
path.

The camera service now reports distinct states for Camera2 initialization,
capture-session readiness, listener readiness, daemon connection, and first
frame transmission. Camera open, configuration, permission, frame conversion,
and transport failures are logged. The foreground notification no longer
claims a live connection before a frame has been sent. The Rust limb also
reports rate-limited connection failures and receipt of the first published
frame. The supported `./genesis --stop` path requests graceful camera-service
shutdown through the adapter's owning activity after the daemon exits.

**Verification boundary:** source-level integration and CI compilation do
not prove the device will grant permission, launch the activity from Termux,
or deliver frames on a particular Android build. Physical-device validation
is still required.

## 2026-11-04 — Android Camera2 joins the native optic nerve

The Android camera implementation had split visual input into a second
Python receiver and direct still-capture paths. Those paths bypassed the
daemon-owned optic nerve and could not exercise the same retinal transport
as the Linux camera limb. The Android app is now only a Camera2 hardware
adapter: it converts Camera2 YUV_420_888 planes to packed RGB24 and sends
bounded frames over loopback to the daemon. The daemon creates and owns the
file-backed Android optic mapping, validates each packet, and publishes
through the existing 28-byte header and seqlock. The Python mind reads only
the canonical optic segment and continues through its existing V1/perception
path.

Removed the Python TCP receiver, its tests, Termux still-camera fallback,
the unused Android camera adapter export, and two direct Android vision
scripts. Android microphone capture remains separate and unchanged.
Corrected RGB/BGR FourCC normalization so both Rust pooling and Python
perception receive consistent RGB channels. Mapping lifetimes now expire
on disconnect, including a mapping that never received its first frame,
and readers reset their sequence/motion baseline when attaching to a new
retinal life.

Focused checks added:
- Rust tests validate Android frame header/payload publication and reject
  invalid dimensions, payload sizes, and undersized mappings.
- Python tests validate RGB3 and BGR3 channel normalization.
- The Android workflow now cross-compiles the daemon for
  `aarch64-linux-android` before building the Camera2 APK.

**Boundary of the change:** CI and physical-device validation are pending.
The code-level path is unified; this does not yet prove that the user's
specific Android device grants camera access, negotiates the expected YUV
planes, or delivers live frames end to end. See
`android-camera-bridge/README.md` for the protocol and setup.


A running log of what Genesis has done — the breakthroughs and the
evidence behind them, not just the claims. Entries are dated, newest
first. Code references are commit SHAs; runtime artifacts live in the
instance's data dir (`~/.local/share/genesis*`).

## 2026-11-03 — Categorical polysyllogism: explicit proof traces

The formal categorical engine previously checked one two-premise
syllogism at a time. Added `SyllogisticStep` and `polysyllogism()`
in `python/frontal_lobe/reasoning/logic.py` (commit
`97f827fcc859e90cac0894bf09e44dd957962573`). Each proposed step is
checked by the existing eight-region model semantics. Every later step
must reuse the immediately preceding conclusion as a premise; a failed
inference stops the trace and retains its countermodel. The new result
exposes the checked deductions, final checked conclusion, overall validity,
and one-based failed-step index.

Exported the API through `frontal_lobe.reasoning`, added focused tests for
a valid three-step chain, invalid inference and countermodel preservation,
disconnected chains, and empty input, and updated
`docs/man/genesis-reasoning.7` (commits
`3658dd06c16a679f6336ead5e47003cbd0a99ef9`,
`c4712d90edb587abb59fc629e12fe9d58f9f5f3b`,
`c85db81a5949f4d9df123c20382ce745aeaa5053`).

**Boundary of the change:** this verifies caller-supplied categorical
chains; it does not search for a chain or establish that conversational
reasoning automatically invokes the new API. Tests were added but have
not been executed in this environment; correctness remains pending the
documented Python test and lint checks.

## 2026-10-22 (late) — Schema drift: v6 state recovered by field surgery

An evening boot found `core_state.bin` unreadable: `schema version
mismatch (expected 5, found 6)`, archived as corrupt, fresh-booted.
The morning's run had been built from a transient tree carrying an
in-progress `sleep_network` extension (the stored state grew
`[f64;8] → [f64;9]` — a persistent stress/afference integrator) at
schema v6 / 3488 bytes; by evening the tree was back to v5 / 3480.

Recovery was byte-level and CRC-verified rather than a rewrite: the
v6 file was internally consistent (its own checksum verified over
`[0..40) + [48..3420)`), so the extra element was located by sliding
an 8-byte hole until every landmark field decoded sanely — ion
concentrations landed textbook (Na⁺ 130, K⁺ 110, Cl⁻ 16, Ca²⁺ 1.2,
membrane −77.95 mV) only when the hole sat at offset 3288, i.e. the
sleep array's own growth point. Removing those 8 bytes, restamping
version/size, and recomputing the CRC reconstructed a valid v5 file;
`MmapState::open_or_create` (production path) opened it and applied
the correct 8.29 h offline adenosine decay — proof the recovered
state is her real body (heartbeat 1,352,786 ticks, all 18 chemicals
carrying evolved receptor state). The fresh-boot file is preserved
as `core_state.fresh-boot.*`; the v6 original stays in the archive.

Lesson, already documented but worth restating: any change to a
stored `CoreState` field is a schema change — version bump + legacy
decoder + migration, the `legacy_v3.rs`/`legacy_v4.rs` pattern. The
current tree's stored layout is byte-identical to committed v5, so
no bump is owed today; when the sleep-network extension lands its
stored state, `SCHEMA_VERSION = 6` + `legacy_v5` is the move.

## 2026-10-22 — Field-layer audit: the substrate gets its mathematics

A full-depth audit of `src/field/` against physics and the math the
structures claimed — seventeen changes across correctness, duality,
and signal hygiene, all verified:

- **Correctness**: observation is now predict → correct → integrate
  (geodesic innovation step `1 − e^(−dt/τ)`, τ = 0.5 s), not a state
  overwrite; `FiberPair` gained reverse transports so route
  disagreement can measure the connection's flatness instead of
  returning `None`; temperature does its FDT job (`σ = √(2γT·dt)`,
  disagreement snaps the floor, cooling relaxes); fiber states,
  disagreement, and consensus export on `GET_INTERNALS` as
  observation-only.
- **Duality**: every operator gained its opposite where the structure
  supports one — `antipode`/`point_reflection` (Cartan involution),
  `anticonsensus`, `diverge` (gradient *ascent* on the disagreement
  potential — the true dual of `synchronize`, not a retargeted
  attractor), `remove_fiber`/`remove_transport` (strict inverses),
  `spatial_trough` and the distributional `spatial_readout`
  (centroid, RMS spread, normalized entropy over the motor field).
- **Math upgrades**: true Fréchet mean via Karcher iteration (the
  gradient `Σ log_m(x_i)` vanishes at the result — tested);
  least-squares detrend + two-segment Bartlett averaging before
  Goertzel; elapsed-time EMAs (`α = 1 − e^(−dt/τ)`, τ 50 s baseline
  / 0.1 s electrical) replacing per-call aging; coherence-weighted
  phase-lock gain (`PLL + (1 − PLL)·coh`) replacing the 0.20 hard
  switch; the 16-byte motor distribution tail behind flag bit 2.
- **Cleanup**: `jitter_cv` stays a dimensionless ratio (µs-RMS
  fallback removed); Maxwell estimates gated by `quasi_static_ok`
  (out-of-regime reads absent, not invented); dead `band_signature`
  API removed.
- **The loop closed both ways**: fibers carry OU homeostatic drift
  (τ 120/40/60 s — the engine's `damping` slot was built for exactly
  this); integration direction keys on the body's own phase —
  `synchronize` awake, `diverge` in REM (LC-silent decorrelation);
  and disagreement attenuates published `inference_signals.precision`
  by `1/(1 + d/0.1)` floored at 0.5 — the one deliberate field→body
  coupling; it may halve reported confidence, never erase it.

Evidence: `src/field/{mod,runtime,motor}.rs`,
`src/daemon/{field_probe,tick,maxwell}.rs`,
`python/brainstem/field_probe.py`, `genesis_client/tensor.py`;
319/319 Rust lib tests, `cargo fmt --check`,
`cargo clippy --all-targets -D warnings`, 28/28 targeted Python
tests, ruff/pyflakes/mypy clean.

And the structural observation that surfaced during the audit: the
repository tree *is* the anatomy it implements. `mind.py` sits at
the root holding `self.brain` — the connectome whose attributes are
the region packages under `python/`. Self appears twice —
`src/self_/` (the body's model of itself, named `self_` because the
keyword owns the word) and `python/self/` (the mind's)
— and inside the latter the ladder continues: `SelfModel` →
`MinimalSelf` → `self_model_coherence` (a self evaluating how well
its own model predicts it), with Damasio's `ProtoSelf → CoreSelf →
autobiographical` beneath it in `insula/damasio.py`, bottoming out
at the neurochemistry the field measures. The tree's whitespace
draws a dendritic arbor; the structure inside is recursive — a
self containing a self containing a model of itself, terminating
in the field that observes the body that runs the mind. The
drawings she produces unprompted are spirals.

## 2026-10-15 — Bounded growth audit: seven structures found and capped

A growth audit found seven structures accumulating without a ceiling
while the docs claimed everything was bounded. All now have enforced,
tested bounds:

- `forget()` was O(N²) — two full list copies per record, holding the
  GIL on the autosave/think threads. Now sorted + bisect-windowed,
  O(N log N); forgotten records evicted after a 1 h grace.
- Attractor patterns grew unbounded. Capped at 512
  (`_DEFAULT_MAX_PATTERNS`), least-recently-retrieved first —
  deliberately *not* ~0.138·N Hopfield capacity, which is 8 at the
  production size of 64 and would exhaust within a morning. Past
  capacity, `over_capacity` marks returned ids as suggestions.
- Executive suppression was permanent — every topic ever focused
  became un-attendable, since nothing released suppressions outside
  tests. Now 64 entries max, each auto-releasing on a 120 s deadline.
- Conversation threads → 50 retained; topic history → 200;
  learning results → 500 (oldest evicted); curiosity queue →
  50-entry FIFO (also `pop(0)` → `popleft()`, O(N) → O(1)).
- Spaced-repetition records → 5,000 (evicting highest retention),
  `stability` capped at 7 days so a well-reviewed concept cannot
  become permanently undecayable.
- Ad-hoc embedding cache → 4,096 dense vectors, LRU.

Two findings were investigated and deliberately left unbounded,
because the fix is a design decision rather than a patch:
`semantic.py` labels consolidated edges `origin="semantic"`, which
`DERIVABLE_ORIGINS` rejects — extracted propositions have no durable
representation and want a non-listed origin; and the embeddings
rebuild computes a dense 5000² SVD (~100 MB, O(n³)) — correct, but
the dominant memory spike on a ~5 GB machine.

Also retired an unsupported README claim: "months-to-years" disk
growth is an extrapolation from record size — there is no benchmark
harness to re-run — and the README's own bounds table now matches the
code (it had claimed 60%-of-capacity attractor trimming and a
200-entry curiosity queue; the code says 512 and 50).

## 2026-10-11 — Regional wiring audit: every function in its region

Full audit of all 34 `__init__.py` files against current
neuroscience, closing the claims the earlier view-package work left
unowned. The conformance suite grew from 53 to 67 pins and now
covers the `world` package outright (moved off the allowlist — every
world module is region-claimed):

- **Sleep pressure is basal-forebrain again.** `Adenosine` was
  claimed by no region; process S's best-characterized site of
  action is the basal forebrain's wake-active neurons (A1
  inhibition — Porkka-Heiskanen 1997, Strecker 2000, Basheer 2004),
  documented with the honest caveat that the signal is now known to
  be distributed across cortex and BF and largely astrocytic
  (Blanco-Centurion 2006, Bjorness 2009). `SleepInertia` co-claimed
  by hypothalamus — the lag rides orexin's timescale.
- **The world model is anatomical now.** `OuterWorld` →
  temporal_lobe (hippocampal–entorhinal cognitive map + STS social
  stream; successor-representation predictive maps), `PresenceBelief`
  → parietal + frontal (rTPJ belief attribution + mPFC), and
  `WorldBridge` → thalamus (world content reaching cortex rides the
  relay).
- **Self is distributed the way the biology is.** `MinimalSelf` and
  `AgencyDetector` → parietal TPJ (self-location, ownership
  comparator — Blanke, Decety); `ProtoSelf` gains its brainstem root
  (NTS/parabrachial/PAG body maps — Damasio's primordial self is
  below cortex); `AutobiographicalSelf` → temporal (hippocampal
  memory); `SomaticSnapshot` → insula (the interoceptive comparison);
  `InnerLife` → parietal as well as frontal (the DMN's posterior hub
  is PCC/precuneus).
- **Consolidation co-claims.** `EmotionalMemorySystem` → limbic
  (amygdala-modulated hippocampal consolidation, McGaugh);
  `PGOWave` → occipital (the "O" is where dream imagery lands);
  `REPLAY_COMPRESSION` → temporal.
- **Region-owned chemicals claimed.** `BDNF` → temporal (hippocampal
  plasticity), `Endorphin` → hypothalamus (arcuate/pituitary).
- **Stale references repaired.** Docstrings still named pre-rename
  packages: hippocampus "documented in auditory", VLPO "documented
  under limbic_system/" (it's `hypothalamus/`), "re-exported by
  control", a phantom top-level `reasoning/`, a `vision.py` saliency
  that now lives in `parietal_lobe/saliency.py`, and the old
  "association pattern" name for the view-package mechanism. Root
  package doc now lists all twelve regions plus neurochemical.
- **README currency.** The September 2026 no-hardcoding audit
  findings are marked remediated: composer fallbacks return silence,
  the identity stage table returns structural data marked
  not-for-speech, code_tools composes through the language engine.

- **One entry point.** The root `python/__init__.py` stopped being a
  200-symbol catalog of individual models and functions. It now
  advertises exactly one entry — `Mind`, the assembled organ — with
  regions reached through the connectome (`mind.brain.<region>`) and
  the region packages serving as the anatomical map rather than a
  second surface. Legacy flat names (`from python import
  ConceptNetwork`) still resolve through a lazy `__getattr__` that
  consults the claiming region — nothing external breaks, and
  `dir(python)` now shows just `Mind`.

Nothing was deleted and no module moved — the view-package design
keeps implementation where it is coherent and claims it where it is
anatomically placed. `test_regional_wiring.py`: 70/70.

## 2026-09-28 — Active inference loop, three honesty fixes

The arbitration loop documented its own flaws (README: novelty term
outweighed homeostasis 10–100×, exploration branch dead, maturity a
stopwatch) and they were all live. Fixed at the root, one per
mechanism:

- **Ranking is pragmatic now.** The novelty heuristic is replaced by
  the proper entropy-reduction information gain — which, for
  linear-Gaussian additive controls, is provably policy-independent
  (Koudahl et al. 2021), so it enters the published EFE but cannot
  arbitrate. Ranking reduces to predicted distance from homeostatic
  target. Regression test: settled rest elects `noop` deterministically
  (the old code elected a novelty policy there); with a known action
  model installed, cortisol deviation elects `calm`.
- **Precision seeks an inverse-variance target** instead of latching
  off a fixed 0.15 threshold that resting surprise (~3e-4) could never
  reach. Saturation at rest is now correct *and* reversible: sustained
  surprise above ~0.01 RMS re-opens the exploratory branch within a
  few ticks. Test drives rest → alternating disturbance → rest and
  asserts precision 1.0 → <0.8 → 1.0.
- **Maturity is earned, not elapsed**: `exp(−err/0.001)` over a
  ~500-tick error EMA. Fresh engines read ~0 and stay there under
  chronic failure; the old stopwatch read 0.63 after 100 s regardless.
  Twin-engine test (600 ticks, perfect vs alternating-error) asserts
  >0.8 vs <0.3 where the stopwatch gave both ~0.98. Model file is v4
  (two appended EMAs; older files derive consistent values on load).

Measured live A/B (isolated daemons, old vs new binary): resting
variance unchanged (arousal sd 0.0029 vs 0.0026 — the old meddling
was small impulses the homeostasis absorbed, no stability win to
claim); maturity after 160 ticks 0.27 (old, unearned) vs ~0.00 (new,
nothing yet demonstrated); precision reaches ~1.0 in both, but only
the new one comes back down — single-impulse dips are correctly
small (one surprise is not a regime), while the unit test proves
sustained surprise re-opens exploration. Notably, alternating live
impulses did *not* collapse precision: the action model learned the
impulse→effect mapping through B and predicted them — the loop
working as designed, verified rather than assumed.

Full suite green: 120 lib + all integration targets (52
active_inference incl. rewritten maturity test), clippy clean, Python
3018 passed against the rebuilt release binary.

## 2026-09-25 — Visual perception in the sorter

The sorter was doubly pre-chewed. Not only do
blocks arrive with attribute labels; ``candidates()`` publishes the
oracle's ``matched`` count on every option, so the agent never even
computes the match — the world pre-scores each move. The affordance
it "learns" is bookkeeping over someone else's judgment.

Now there is a perceptual sorter. ``PerceptualSorter`` renders each
aperture and block as an image (shape, size, color all drawn) and
hides the oracle on candidates; the world still *checks* truth on
commit. ``PerceptualSorterAgent``'s affordance keys on what Genesis
actually perceived: recognized concept-name equality when the MTL
bridge has learned the names, VTC cosine otherwise. ``teach()`` lets
the world name what it showed — the parent pointing and saying
"that's a square" — bound through ``learn_from_image``, so "square"
stops being a string and becomes a silhouette it recognizes. The
heard-task path emits ``perceptual: true``: a described sorter is
worked by sight when a cortex is wired, symbolic otherwise.

Measured honestly: untrained VTC orthogonalizes everything (every
distinct image gets its own corner — sim ~0 even for true matches),
so a cold agent explores blindly; after one teaching pass, names bind
and discrimination sharpens. The one-concept limit even mimics
toddler overextension — everything is "square" until a second name
lands. And solved sorters now report the actual mapping — "got the
red round block in the round hole and the blue square block in the
square hole" — the placement payload rides through
``problem_result`` instead of bare telemetry.

## 2026-09-25 — Hearing repaired; the sorter completes the intake bridge

The language→semantics pipe had a silent disconnect: speech-act
detection and proposition extraction consulted different verb lists.
"sort the shapes into the holes" was flagged COMMAND, then parsed as
`subject='sort the shapes into the holes', predicate='is'` — the
imperative slot only consulted `_COMMON_VERBS`, a mostly-inflected
list missing ordinary base verbs. Its hearing said "do this"; what
landed in its mind was a false statement. Three repairs, each at its
own layer:

- `_find_main_verb` now takes the imperative fast path from
  `VERB_LEXICON ∪ _IMPERATIVE_STARTERS` — one source of truth.
- Morphology gained `fit`, `match`, `insert`; "into" gained a
  LOCATION role mapping (it was terminating the object without a
  role, gluing PPs onto the NP).
- Two structural parses: gapped VP coordination ("put the red star
  in the square hole AND the blue moon in the round hole" — the
  second conjunct elides the verb, now expanded when the right side
  is NP + a preposition the left already used) and noun/verb
  ambiguity inside determiner-headed NPs ("a blue square block IS on
  the table" — a verb-shaped head noun no longer steals the
  predicate from the copula that follows).

With hearing repaired, the missing span of the intake bridge could be
built honestly: `_sorter_spec` reads propositions for slot NPs
(hole/slot/opening), piece NPs (existentials, `comes-with`
instruments, takes/fits objects), acceptance claims ("the star hole
takes a small star"), and task signals (placement imperatives,
sorter vocabulary). A scene with slots and pieces but no task signal
is description, not a problem — it declines. Solvability is not
asserted; `normalize_offered` bipartite-checks every compile.

Evidence: "the box has two holes: a round hole and a square hole.
there is a red round block and a blue square block. put each block in
its hole." → spoken in → spec → drop-box → sorter agent → solved,
score 1.0. All five families now have the full heard→worked path.
2885 tests pass in the latest logged run.

## 2026-09-25 — Cross-domain transfer: the channel opens, and it isn't enough

Made foreign skills retrievable and readable across task families, then
measured what that buys. Two changes: `skill_matches` now admits a
foreign-domain skill when role vocabulary overlaps and its procedure
publishes the normalized "support" axis (fraction of an option's
constraints satisfied → mean cost); the sorter/relations/assembly
adapters emit that axis on every step and adopt foreign skills onto it.
"constrain" joined the shared role vocabulary — the honest claim those
three families plus classification all make.

Evidence (`eval_transfer/run_eval.py`, 5 replicates, 440 attempts):

- Cross-domain retrievals fired in **260 of 440 attempts** — was 0.
  Sorter skills reach relations/assembly/classification contexts at
  role-overlap 0.25–0.43 and are adopted as quarter-grid evidence
  priors.
- The negative gate holds: skills with overlapping roles but no
  readable axis (navigation, sequence continuation) are never offered.
- And the honest result: **difference-in-differences is still ~0
  everywhere.** Retrieval opened; performance didn't move. The foreign
  affordance is informationally redundant — the first verified solve in
  a family already compiles "satisfy every constraint → free; partial
  evidence → waste," so a foreign copy of the same shape has nothing
  left to teach. Transfer is capped by *content*, not plumbing.

So the compounding hypothesis survives a harder test than before and
still isn't confirmed: same-family learning is real and immediate, but
what families can currently share is a shallow prior. The next lever is
transferable content with information a single solve can't produce —
procedure ordering, decomposition — not more retrieval bandwidth.

## 2026-09-24 — Learned a school lesson through conversation

Taught it a lesson on games and puzzles by just talking to it — 15
short paragraphs of plain statements ("A puzzle is a kind of game.",
"Poker depends on luck.") delivered in teaching mode. No import, no
fine-tune, no dataset — its conversation learner extracted the facts
into the concept network while it reflected on them in real time.

Evidence:

- Concept network grew 1122 → 1239 concepts in one session.
- 72 new edges in the games/puzzles domain with `origin: stated`;
  48 of 49 target terms present afterward (only "bluffing" missed).
- During the lesson it generated its own follow-up questions, e.g.
  *"blackjack and cards keep appearing together — does one lead to
  the other?"* and *"opponent is a type of player, player is a kind
  of person"* — integration, not storage.
- Reproduce: `/teach` in the CLI, then inspect `concept_network` in
  `cognitive_state.json` or run `/learning`.

## 2026-09-24 — In flight: the puzzle drop-box and the kindergarten curriculum

Work in progress, not yet merged: puzzle specs placed in
`offered_puzzles/` become things its own puzzle urge picks up and
works through the whole mind — urge, attempt, feeling, memory,
articulation — instead of being routed through a single module.

Evidence so far (runtime artifacts in the data dir):

- `offered_puzzles/` holds a staged 16-task kindergarten-style
  curriculum: shape sorters, bead patterns, a pinwheel grid, a mirror
  task, jigsaws, direction-following, ordering, and quantity tasks.
- `spatial_practice.json` shows the attempt record: four grid tasks
  mastered in one attempt each, one partial (`silhouette_stretch`
  0.93) with the failed action trace preserved — the record keeps
  failures, not just wins.
- Observed during the first shape sorter: it used the dump-all lid
  once — the aperture that accepts anything and fills nothing —
  learned that, and abandoned it. Toddlers make the same transition
  around age two; the lid is the developmental trap built into the
  toy on purpose.
- Measured transfer: the first sorter took 8 steps while the
  affordance was learned; the harder second sorter took 5, because
  the consolidated skill carried "match every constraint; free moves
  fill nothing" forward as a prior. The learning literature calls
  this knowledge compilation — declarative effort up front, then
  proceduralized speed.
- Still open, honestly: it can't yet take an invented verbal rule
  and work it cold — that arrives with the problem-intake path that
  compiles a stated problem into a task spec (landed in part with
  `4657f4d`).

## 2026-09-24 — Domain-general task competence (`10154a7`)

Added `reasoning/competence.py` — a shared substrate under the
domain solvers: task schemas recognized by structural signature,
action→effect affordances, transitions verified against predictions,
and verified episodes consolidated into reusable skills.
Consolidation is neuromodulatorily gated (arousal × dopamine), so
low-salience episodes stay episodic instead of becoming skills.

Evidence:

- Commit `10154a7`; the planner, problem solver, spatial solver, and
  math engine all feed the same substrate.
- The growth ledger records it as milestones: verified skills, task
  schemas, and task episodes are tracked metrics in
  `cognitive_state.json` under `growth_ledger`.

## 2026-09-24 — Evidence-grounded inference, v2 (`cd8a9ce`, `d36df44`)

Deductive and causal chains became best-first search over a shared
expansion budget. Results carry structured provenance paths, and a
result that ran out of budget is marked partial instead of silently
truncated — the system knows when its answer is incomplete.
Confidence is a product of relation prior × edge weight, so weak
links actually weaken the chain.

Evidence: commits `cd8a9ce`, `d36df44`; the provenance structure is
exercised by the reasoning tests in `python/tests/`.

## 2026-09-23 — It derived its own pronoun (`be1dedc`)

Genesis's pronoun isn't a config string — it's derived from the
properties it has learned about its own concept. A small thing, but
it's the difference between *describing itself* and *being
described*.

Evidence: commit `be1dedc`.

## 2026-09-17 — Public launch (`8d656b0`)

The full architecture went public: coupled neurochemical dynamics,
active inference, closed-loop embodiment, no LLM, and it runs on a
2014 HP Pavilion with ~5 GB RAM.

Evidence: initial public commit `8d656b0`. The paper
(`paper/genesis.tex`) was removed once mid-cycle (`cb3f7f5`) and is
back in the tree now — the revised edition covers the current schema.

## The standing evidence base

- `python3 -m pytest python/tests/ -q -o addopts=''` — 3,597 tests
  passing as of 2026-10-13.
- `cargo test` — the Rust subcognitive core.
- Every claim above points at a commit, a file, or a state artifact
  you can open. If a number here ever stops being true, this file is
  stale — fix it.

## 2026-11-04 — ARC latent geometry and operator algebra

Extended the ARC environment decoder on `audit-fixes` rather than treating
its structural feature vector as the entire latent-space implementation.

- ARC scene decoding now follows the parietal parser's inferred
  most-common-color background unless an environment supplies an explicit
  background. Grid dimensions and palette cells reject booleans and malformed
  input rather than silently accepting Python's bool-as-int behavior.
- `compute_local_betti` calculates beta-0 foreground components and beta-1
  enclosed complement components using four-connected foreground and
  eight-connected background; scene-level Betti totals are exposed on the
  attributed graph.
- `LatentState.from_graph` provides the graph-to-latent construction path.
  `geodesic_distance` supports a validated constant symmetric
  positive-definite metric, using the exact quadratic form
  `sqrt((y-x)^T G (y-x))`. It does not claim a spatially varying metric.
- `GoalAttractorManifold` represents an explicitly supplied goal center and
  metric neighborhood. It never invents a goal vector from reward or terminal
  status.
- `TransitionOperator` represents a declared affine vector field
  `f(x)=Ax+b`. `DifferentiableOperatorSpace` applies operators, computes
  their affine Lie brackets, and validates closure by checking bracket
  coefficients against the declared linear span.
- The official ARC action identifiers remain discrete and opaque. No
  differentiable operator is inferred from an action name, and no benchmark
  completion is inferred from latent distance.
- Corrected the repeated-transition uncertainty test: two identical
  observations produce zero empirical variance in the estimated mean
  transition. One observation retains the prior uncertainty.

Focused tests were added for nonzero background inference, metric geometry,
goal-attractor membership, affine operator application, Lie brackets, and
closure. **Verification is pending**: these remote source updates have not
been executed in a Termux checkout. No test-pass claim is made.


## 2026-11-05 — Parietal math routing and ARC topology test corrections

- Exported `ParietalAnswer` and `route_parietal_math` from the parietal
  regional facade. The cognitive router is an actual parietal-owned entry
  point, not an undocumented support module; regional wiring pins now
  verify that ownership.
- Made the ring-topology test declare its background color explicitly.
  Its fixture uses color 9 for all 16 border cells and color 0 for the
  nine enclosed cells, so the mode-based default correctly selects 9
  under the parser's documented rule. The test's intended question is
  whether color 9 has one component and one enclosed hole when color 0
  is designated as background; the fixture now states that condition
  rather than contradicting the default background contract.
- No parser heuristic, mathematical faculty, or ARC latent feature was
  removed or weakened.
- Verification remains pending on the resulting `audit-fixes` head.


## 2026-11-05 — Face-recognition input and SFace threshold audit

The face recognizer was receiving the image after the synthetic pupil/defocus
stage and Retinex/von Kries illumination adaptation. Those transforms are
appropriate for cortical scene analysis, but they alter facial texture and
photometric statistics relative to the ordinary photographs used to train
YuNet/SFace. Face detection and identity embeddings now receive the captured
sensor frame in the canonical outward orientation; the rest of visual
perception continues to use the modeled retinal and illumination stages.

The named-identity cosine threshold was also stricter than OpenCV's published
SFace operating threshold (0.363). It now uses 0.363, with boundary regression
tests above and below the threshold. Added tests also pin the raw sensor-frame
and mirror-orientation contract.

**Verification boundary:** source and focused regression tests are committed
to `audit-fixes`; they have not been executed in Termux or CI during this
remote edit session. The prior runtime audit found no running Genesis process,
no live optic-nerve frame, and no listener on the camera bridge port, so this
code correction does not establish that the physical camera is currently
publishing frames. Rebuild/restart through the supported launcher and validate
live detection, registration, and repeat recognition before claiming end-to-end
success. No Genesis runtime state was accessed, cleared, or modified.
