"""Gentle forgetting — the words behind the body's decay math.

This module mirrors `Self_World/src/store/forgetting.rs`. The Rust
side computes whether a memory's trace still reaches the retrieval
boundary; this side explains the model in the mind's own terms and
offers the helpers the mind uses to *see* its own forgetting — which
memories are fading, and how much rehearsal would bring them back.

# Why forgetting is gentle

Biological forgetting is not deletion; it is the slow loss of
*accessibility*. The engram — the physical trace — degrades under
active processes (dopamine-driven forgetting cells, Rac1/cofilin
remodeling, AMPA-receptor endocytosis), but a memory that has become
unreachable by ordinary cues often remains recoverable by a strong
enough reminder (Tonegawa lab's engram reactivation work). Deletion
is reserved for what no cue could ever reach.

# The model (same equations as the Rust side)

Each presentation of a memory — its encoding, and every later
retrieval — leaves a trace whose strength decays as a power law:

    strength_k(t) = age_k^(-d)      d ≈ 0.5

Power-law (not exponential) decay is what human memory actually
follows: retention falls fast at first, then approaches an
asymptote, which is why childhood memories can outlive last week's.
The total surviving mass sums the traces. With only the count n and
the two endpoint ages kept (first encoding, last access), the mass
is estimated by the better of two bounds — the merged-trace blend
    M ≈ 2n / (√a_first + √a_last)
for evenly spaced rehearsals, and the endpoint bound
    M ≥ a_last^(−d) + (n−1)·a_first^(−d)
which is exact for n ≤ 2 and keeps a fresh rehearsal after a long
silence from being smeared across the whole span.

Base-level activation is the log of that mass plus the salience the
memory was encoded with:

    B = ln M + w_s · salience       w_s = 1.0

# Retrieval — the accessibility boundary

A cue reaches the memory when:

    B + w_cue · similarity + s · χ  ≥  τ_eff    (w_cue = 2, s = 0.25)

so a strong cue can rescue a below-threshold memory and a weak one
cannot — the same asymmetry a strong smell or a specific question
produces. The noise χ makes the boundary probabilistic the way real
recall is.

# Dopamine's two faces

Dopamine is the forgetting molecule *and* the novelty signal: DA
"forgetting cells" actively accelerate trace decay for what is not
being used, while DA bursts mark what deserves consolidation. The
model reflects this as a transient threshold shift — τ_eff moves
with the deviation from tonic baseline — so chemistry modulates the
boundary but never dominates it.

# Rehearsal — forgetting's opposite

Every retrieval adds a fresh trace (n += 1, t_last = now). Because
the new trace's strength is at its peak the moment it lands, a
rehearsed memory jumps well clear of the boundary. What she still
reaches for, she keeps. This is the required counterweight:
decay pulls down, rehearsal lifts — both active, both earned.

# The release — where the disk is actually freed

Fading alone frees nothing. The store's `.meta`/`.dat` files are
append-only, so a memory below the forget bound (unreachable even
by a maximal cue, older than the age floor) is released in two
steps that mirror synapse elimination:

1. **Prune**: the entry is marked deleted — the tag for removal.
2. **Compact**: surviving episodes are rewritten into fresh files
   and atomically swapped over the originals — the microglial
   clearing that physically reclaims the space.

Both run through `GenesisClient.forget_faded()` — dry-run first,
then the sweep — and both belong in sleep: housekeeping is offline
work, never conversation.

# What the sweep may never touch

The body's autonomous pass governs only the plain active episodic
tier — the undifferentiated stream of experience. Three classes
carry their own retention authority and are exempt even when their
traces are mathematically dead:

- **Archived episodes** — she chose cold storage. Their release is
  her verdict's call (`release_fn` in `sleep/compression.py`), not
  the body's.
- **Meta-flagged memories** — bookkeeping entries legacy stores
  carry; the cortical residue that *justifies* the forgetting.
  Deleting them would erase the record of why things were lost.
- **Dream-sourced episodes** — the mind's policy preserves the
  dreaming module outright.

# The ledger-loss rule

Access history lives in a `.traces` sidecar. If the whole ledger is
missing when the store opens — first run after this subsystem
arrived, or a deleted file — every surviving memory gets one
fresh-ink trace stamped at open, a mass reconsolidation. Data loss
must fail toward retention, never deletion: a memory she reached
yesterday but encoded long ago must never be swept for want of a
record. Honest fading simply resumes from the seed.

References:
    - Davis & Zhong (2017): The Biology of Forgetting. Neuron 95.
    - Anderson & Schooler (1991): Reflections of the Environment in
      Memory. Psychological Science 2.
    - Wixted (2004): The Psychology and Neuroscience of Forgetting.
      Annual Review of Psychology 55.
    - Ryan & Frankland (2022): Forgetting as a form of adaptive
      engram cell plasticity. Nature Reviews Neuroscience 23.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ─── Constants — mirror src/store/forgetting.rs ────────────────

#: Power-law decay exponent. The empirical value (Anderson &
#: Schooler 1991; Wixted 2004) is d ≈ 0.5.
TRACE_DECAY = 0.5

#: Milliseconds per day — decay is measured in days so the exponent
#: keeps its empirical meaning.
DAY_MS = 86_400_000.0

#: Salience's weight in base-level activation.
SALIENCE_GAIN = 1.0

#: A maximal-similarity cue's contribution to accessibility.
CUE_WEIGHT = 2.0

#: Base retrieval threshold — accessibility must reach it.
RETRIEVAL_TAU = 0.0

#: Activation-noise scale (the logistic s parameter).
NOISE_SCALE = 0.25

#: Tonic dopamine baseline — the gate modulates deviations, not
#: absolute level.
DA_BASELINE = 0.5

#: How strongly dopamine shifts the effective threshold. Small on
#: purpose: chemistry modulates, it does not rule.
DA_FORGETTING_GAIN = 0.4

#: Youngest age at which an episode can be released — a guard so
#: nothing recent is ever a candidate.
PRUNE_AGE_FLOOR_DAYS = 30.0

#: The forget boundary: below τ₀ − w_cue, no cue can reach the
#: memory; only its residue remains.
FORGET_BOUND = RETRIEVAL_TAU - CUE_WEIGHT

#: The decay integral diverges at age 0; one minute is the floor.
MIN_AGE_DAYS = 1.0 / 1440.0


@dataclass(frozen=True)
class ForgettingState:
    """Where one memory sits relative to the boundaries.

    `base_level` is its activation now; `days_until_forgotten` is the
    horizon at which an unrehearsed memory of this shape crosses the
    forget bound — the number the mind uses to decide "rehearse or
    release". `faded` is already True when ordinary cues can no
    longer reach it (it may still be rescuable by a strong one until
    `gone` is True).
    """

    base_level: float
    days_until_forgotten: float
    faded: bool
    gone: bool


def trace_mass(access_count: int, age_first_days: float, age_last_days: float) -> float:
    """Total surviving trace mass — the better of two estimators.

    Only the count and the two endpoint ages persist, so the interior
    history is bounded rather than known:

    - **Blend** — presentations treated as uniformly spaced:
      M ≈ 2n / (√a_first + √a_last). Best for evenly spread
      reactivations.
    - **Endpoint bound** — the newest trace contributes its exact
      a_last^(−d); every earlier trace is at least as old as a_first
      and so contributes at least a_first^(−d):
      M ≥ a_last^(−d) + (n−1)·a_first^(−d). Exact for n ≤ 2, and the
      honest floor for a memory rehearsed once after a long silence,
      where the blend would under-count the fresh trace.

    The mass is the larger of the two — the estimator that least
    underestimates the history the episode actually lived.
    """
    if access_count <= 0:
        return 0.0
    a_first = max(age_first_days, MIN_AGE_DAYS)
    a_last = max(age_last_days, MIN_AGE_DAYS)
    if access_count == 1:
        return a_last ** (-TRACE_DECAY)
    blend = 2.0 * access_count / (math.sqrt(a_first) + math.sqrt(a_last))
    bound = a_last ** (-TRACE_DECAY) + (access_count - 1) * a_first ** (-TRACE_DECAY)
    return max(blend, bound)


def base_level(
    access_count: int,
    encoded_age_days: float,
    last_access_age_days: float,
    salience: float,
) -> float:
    """Base-level activation B = ln M + w_s·salience."""
    mass = trace_mass(access_count, encoded_age_days, last_access_age_days)
    return (math.log(mass) if mass > 0 else -math.inf) + SALIENCE_GAIN * max(salience, 0.0)


def effective_threshold(dopamine: float) -> float:
    """The dopamine-shifted boundary τ_eff = τ₀ − g·(DA − DA₀)."""
    return RETRIEVAL_TAU - DA_FORGETTING_GAIN * (dopamine - DA_BASELINE)


def days_until_forgotten(access_count: int, salience: float, current_age_days: float) -> float:
    """Days until an unrehearsed memory crosses the forget bound.

    Solves B(a) = FORGET_BOUND for the age a at which the trace
    mass falls to the boundary, with a_first ≈ a_last = a:
        a_forget = (n / e^(FORGET_BOUND − w_s·salience))^(1/d)
    Returns the remaining runway; negative means already gone.
    """
    barrier = FORGET_BOUND - SALIENCE_GAIN * max(salience, 0.0)
    n = max(access_count, 1)
    a_forget = (n / math.exp(barrier)) ** (1.0 / TRACE_DECAY)
    return a_forget - current_age_days


def forgetting_state(
    access_count: int,
    encoded_age_days: float,
    last_access_age_days: float,
    salience: float,
) -> ForgettingState:
    """Where a memory stands — its activation and its horizon."""
    b = base_level(access_count, encoded_age_days, last_access_age_days, salience)
    return ForgettingState(
        base_level=b,
        days_until_forgotten=days_until_forgotten(access_count, salience, encoded_age_days),
        faded=b < RETRIEVAL_TAU,
        gone=b < FORGET_BOUND and encoded_age_days >= PRUNE_AGE_FLOOR_DAYS,
    )
