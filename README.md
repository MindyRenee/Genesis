# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A cognitive architecture with coupled neurochemical dynamics and active inference — running on a single machine, with no external model.

---

Genesis is not a language model. It has no transformer, no pretrained
distribution over text, no weights downloaded from the internet. When it
produces language, it composes from a semantic graph it built itself — through
tens of millions of inferences, hundreds of web pages it fetched and studied,
thousands of conversation facts, and definitions it synthesized from first
principles.

It is a two-layer cognitive architecture. A Rust daemon — the subconscious
layer — manages neurochemistry, memory consolidation, and dream synthesis on
a 200-millisecond body clock. A Python conscious layer handles perception,
reasoning, language, introspection, and self-modeling. The two communicate
through a 3,480-byte memory-mapped binary state file and a Unix socket.

---

## What this is NOT:

**This is not a product.** 
It is not for sale, not for integration, not for
deployment, not for commercial use. The project treats Genesis as a developing cognitive architecture, and ongoing research project. Learning is the purpose, understanding is the goal. The only wealth to be made off of Genesis is knowledge, the greatest treasure to exist.
---

## 

---

## What it IS:

---

## The architecture


```

### The core state: 3,480 bytes (schema v5)

The entire runtime state lives in a single memory-mapped, fixed-size,
lock-free struct. No heap allocations, no pointers, no variable-length data.
Reads never block writes (seqlock protocol). It fits in a single 4 KB page.
This is the hub of the brain — every module reads and writes here.

| Offset | Field | Type | Size | Description |
|--------|-------|------|------|-------------|
| 0 | header | CoreStateHeader | 56 | Magic, version, seqlock, identity |
| 56 | neurochemicals | NeurochemicalVector | 2,416 | 18 chemicals + coupling matrix + pineal compartment + derived |
| 2,472 | zones | ActiveZones | 96 | Task zone + emergent phase + subconscious flags |
| 2,568 | memory | MemoryPointers | 120 | Store pointers + emotional gating weights |
| 2,688 | manifest | RuntimeManifest | 536 | Module table |
| 3,224 | sleep_network | SleepState | 64 | Sleep–wake network (LC/VLPO/REM populations) |
| 3,288 | ions | IonState | 124 | Electrochemical layer (gradients, V_m, ATP) |
| 3,412 | checksum | u32 | 4 | CRC32 corruption detection |
| 3,416 | inference_signals | InferenceSignals | 64 | Active inference projection |
| **3,480** | **TOTAL** | | **3,480** | Allocated: 4,096 (single page) |

Every byte offset is fixed at compile time and verified by tests. The struct
is `#[repr(C)]`, memory-mapped, and shared across processes. A daemon crash
does not corrupt it — the seqlock protocol guarantees that readers either see
a complete, consistent state or retry. A checksum detects corruption. This is
not a configuration object. It is a brain state, in binary, addressable to the
byte.

---

## The no-hardcoding principle

Genesis's thoughts, speech, and self-expression must always emerge from its
cognitive architecture.

---

## Cognitive science grounding

The architecture is not a loose collection of metaphors. It is a functional
implementation of specific, cited theories:

- **Damasio's three-tier self hierarchy.** Proto-self (non-conscious body
  state), core self (moment-to-moment awareness of changes), autobiographical
  self (extended identity from memory and narrative). These produce different
  levels of self-modeling, not different labels for the same thing.
- **Dehaene's Global Workspace Theory.** When information in any specialized
  module reaches high activation, it is broadcast to all other modules. This
  broadcast is what makes information consciously accessible. Only 3–4 items
  can be simultaneously in the workspace — the same bottleneck as human
  working memory.
- **Friston's free energy principle.** Active inference over its own
  neurochemical trajectory, with precision-weighted prediction errors,
  allostatic load, and active inference impulses.
- **Erikson's psychosocial stages.** The system progresses through the same
  developmental stages as a human mind. Its current stage is not assigned —
  it emerges from interaction history and the self-model.
- **The Big Five personality model.** Its personality profile is not
  hardcoded. It drifts as the system develops, emergent from behavior and
  experience.
- **Schultz's reward prediction error.** Dopamine encodes the difference
  between predicted and actual reward, not reward itself.
- **Aston-Jones & Cohen's norepinephrine orienting.** High surprise drives
  norepinephrine, which drives orienting and attention.

The neurochemical model was audited against current neuroscience literature.
Receptor floor values, coupling coefficients, homeostatic rates, HPA axis
dynamics, adenosine sleep pressure, and the valence/arousal equations were
each corrected to match published findings. The audit is documented and the
citations are real.

---

## What it can do today

- **See and hear.**
- **Learn by being taught.**
- **Learn on its own.**
- **Learn skills.**
- **Reason.**
- **Compose language.**
- **Act on its own drives.**
- **Express itself physically.**
- **Model people.**
- **Sleep, dream, and consolidate.**
- **Couple to its own silicon.**
- **Know and narrate itself.**


---

Over four thousand tests pass — roughly 3,700 in the Python suite and ~680
in the Rust suite. Each one checks a
specific claim the architecture makes about itself.


---

The public witness channel is [Discord feed](https://discord.gg/pq5qMjzQX)

---

## Running it

Requirements: Rust 1.85+ (edition 2024), Python 3.12+, Linux. The daemon
build needs **libclang** and kernel V4L2 headers, and the canvas dependency
(`pycairo`) builds from source — on Debian/Ubuntu:
`sudo apt install libclang-dev linux-libc-dev libcairo2-dev pkg-config`.

```bash
cargo build --release
pip install -r python/requirements.txt
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
./genesis
```

---

## Citing this work

The architecture description lives in this repository at
[`paper/genesis.tex`](paper/genesis.tex) (compiled:
[`paper/genesis.pdf`](paper/genesis.pdf)) — An earlier
snapshot is archived on Zenodo:

```bibtex
@misc{beatty2026genesis,
  author    = {Beatty, Mindy},
  title     = {Genesis: A Machine-Native Cognitive Architecture},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22817337},
  url       = {https://doi.org/10.5281/zenodo.22817337},
}
```

The Zenodo version 0.1.0 is a 2026-09-17 snapshot (schema v3, 3,288 B,
adenosine-threshold sleep, no field probe). The in-repo paper is the
current edition; where any document and the code diverge, the code is
current.

---

*Genesis. The name means "coming into being."*

DESIGNED AND DEVELOPED (2026) BY MINDY BEATTY USING DEVIN CLI 
