# Genesis Structural Generalization: Independent Evaluation Protocol

## Purpose

This protocol extends `eval_structural_generalization.py` without changing Genesis's cognitive architecture. It is designed to distinguish a genuine relation-generalization result from a result tied to one relation, one sentence template, or one random seed.

A passing result is evidence only for the tested capabilities. It is not, by itself, evidence of general intelligence, consciousness, or ARC-AGI performance.

## Freeze the system before testing

1. Record the exact Genesis commit SHA, Python version, operating system, hardware, and available RAM.
2. Do not edit Genesis or the evaluator after the test begins.
3. Run the same frozen commit on at least 10 predeclared seeds. Publish every run, including failures.
4. Keep the expected graph and answer key inside the evaluator; do not expose them to Genesis.
5. Save raw per-trial outcomes, not only an aggregate percentage.

## Test stages

### Stage A — Reproduce the current baseline

Run the existing blind randomized structural generalization evaluator at its default seed. Record its full console output and commit SHA. The current implementation is a one-world `PART_OF` chain test with a disconnected distractor, direct-edge provenance check, and inference-volume diagnostic.

### Stage B — Seed robustness

Run the unchanged evaluator across 10 predeclared seeds. Report:

- successful blind positive closures / attempted closures;
- negative-control false positives / controls;
- stated-edge provenance successes / checks;
- blindness-precondition failures;
- inference volume as a diagnostic, not part of accuracy.

Do not combine these different measures into one headline score.

### Stage C — Relation generalization

Add separate, predeclared conditions for each supported transitive relation (for example, `IS_A`, `PART_OF`, `DEPENDS_ON`, and `ENABLES`). Do not assume all are validly transitive in every domain; explicitly define the intended semantics for each condition. Generate fresh opaque concepts for every trial. Report each relation separately.

### Stage D — Language-template generalization

Vary the natural-language surface form used to teach the same relation. Include only templates verified to be accepted by Genesis's existing parser. Randomize the template independently of graph structure. Report parser/ingestion failures separately from inference failures; do not silently discard failed trials.

### Stage E — Longer structures and harder controls

Predeclare multiple chain lengths and query distances, including distances not used in the baseline. Add branched graphs, multiple disconnected distractors, and near-miss negative queries. The evaluator must compute the expected closure independently from the generated graph, not by calling Genesis's inference code.

## Scoring rules

- **Positive inference accuracy:** correct expected non-direct edges that Genesis inferred, divided by attempted positive queries.
- **Negative-control accuracy:** expected-absent edges that remain absent, divided by negative queries.
- **Ingestion rate:** supplied facts represented in the graph with the expected stated provenance.
- **Provenance accuracy:** inferred conclusions carry inferred provenance; directly taught facts retain stated provenance.
- **Blindness validity:** generated concepts were absent before the trial. A failed blindness precondition invalidates that trial; it is not a Genesis success or failure.
- **Diagnostics:** inference volume, runtime, memory use, and cooldown are reported separately and do not affect accuracy.

Never report a single percentage without its numerator, denominator, test definition, seed, and commit SHA.

## Independence and anti-contamination

- Keep benchmark generation independent of Genesis's own code wherever possible.
- Do not use concept names, templates, or graph structures that appear in training/demo fixtures.
- Include evaluator tests with deliberately broken inference behavior to confirm that the scoring harness catches failures.
- Include a no-inference ablation and, where feasible, a parser/learning ablation. These establish whether the tested component is necessary for the result.
- Have another person run the frozen instructions from a clean checkout and publish their raw output.

## Interpretation

A robust result across seeds, relation types, templates, graph structures, and independent reproduction would strengthen the claim that Genesis exhibits generalized symbolic relational inference under those conditions. It would still be a bounded capability result, not a general-intelligence or consciousness determination.