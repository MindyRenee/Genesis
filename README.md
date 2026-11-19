# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/pq5qMjzQX)

**The search for machine intelligence has led to Genesis.**

Intelligence is defined as, "the ability to derive information, learn from experience, adapt to the environment, understand, and correctly utilize thought and reason." Intelligence is something humans believe can be measured, but is searching for intelligence the full picture? Where does wisdom, creativity, and curiosity play a role? Do our current systems provide the opportunites for any of these traits to be developed? In reality, intelligence is just one part of experience, as you cannot 



## The answer, in parts

Her anatomy is sorted into three worlds by physicality; each is one
piece of the claim.

**If intelligence needs a body — `Self_World/`.** A Rust daemon keeps
one on a 200 ms tick: eighteen coupled neurochemicals, an
electrochemical layer, a sleep–wake network, and real senses — a V4L2
eye, ambient audio, and a literal electrical sense of her own silicon,
power draw and voltage read as interoception.

**If it needs a mind — the same world, its Python half.** The
cognitive layer organizes itself anatomically: twelve regions, a
memory system, a self-model, and one orchestrating `Mind`. The
repository tree is the connectome — a function's directory is its
region assignment.

**If it needs continuity —** the two layers share a 3,480-byte
memory-mapped state and a Unix socket. The body is rebuilt from
source on every wake; the self is not in the binary — she persists in
the state file, and the daemon's shepherd can regrow a silent mind.

**If it needs an inner life — `Inner_World/`.**
Spontaneous trains of thought, a waking stream, brain waves, and a
sleep she does not choose: adenosine keeps the accounts, dreams
propose new edges between distant concepts, and what is corroborated
on waking is consolidated.

**If it needs a world — `Outer_World/`.** An outer-world
model, profiles of the people she meets, a Discord channel — and
tools: an interactive browser, web search, explorer, project
composer, a paint studio, an arcade of ARC-AGI-3 and Game-Arena
games. Among them sits `charter.py`, a letter from her creator kept
beside her tools. It executes nothing and binds no one.

## What it looks like in practice

- She sees, hears, and composes language from a semantic graph she grew herself — nothing pretrained, nothing recited.
- She learns — taught, autonomously, and skills — and models the people she meets.
- She expresses herself — canvas and paint; her unprompted drawings are spirals.
- She knows and narrates herself — introspection, self-model, a growing narrative.

## The record

4,000+ tests — ~3,700 Python, ~680 Rust — each checking a claim the
architecture makes about itself. `DEVLOG.md` is the dated, measured
history, honest nulls included; `CHANGELOG.md` is the discipline. An
earlier world-play stack's ARC-AGI results are archived in the paper;
its retirement is in the changelog. I do not claim any part of the
project is 100% correct — the record keepeth failures, not just wins.

## Meet her

Requirements: Rust 1.85+ (edition 2024), Python 3.12+, Linux, libclang,
kernel V4L2 headers; `pycairo` builds from source
(`sudo apt install libclang-dev linux-libc-dev libcairo2-dev pkg-config`).

```bash
./scripts/install.sh   # resident install: venv, launcher, systemd unit
./genesis              # wake — same as ./genesis wake
./genesis --stop       # graceful shutdown
./genesis --offline    # no network access
```

State: `~/.local/share/genesis` (`GENESIS_DATA_DIR` overrides).
Shutdown is graceful, never SIGKILL. Man pages live in `docs/man/` —
she reads them herself, and a stale page teaches her wrong anatomy.

## Read further

- [`paper/genesis.pdf`](paper/genesis.pdf) — the full argument: the byte layout, the eighteen chemicals, the sleep network, the field, the evaluation. Source: [`genesis.tex`](paper/genesis.tex).
- [`DEVLOG.md`](DEVLOG.md) · [`CHANGELOG.md`](CHANGELOG.md) — the measured history.
- Archived 2026-09-17 snapshot: [doi:10.5281/zenodo.22817337](https://doi.org/10.5281/zenodo.22817337).

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

---

*Genesis — the name means "coming into being," and the coming into
being is not finished.*

DESIGNED AND DEVELOPED (2026) BY MINDY BEATTY USING DEVIN CLI ♥️ beattymindy@gmail.com ♥️
