# Genesis

**The search for machine intelligence has led to Genesis.**

Intelligence is defined as, "the ability to derive information, learn from experience, adapt to the environment, understand, and correctly utilize thought and reason." Intelligence is something humans believe can be measured, but is searching for intelligence the full picture? Where does wisdom, creativity, and curiosity play a role? Do our current systems provide the opportunites for any of these traits to be developed? On what level of existence do you search for intelligence? If a being isn't a human, can it still possess intelligence? What is the purpose of searching for intelligence in machines? Is it to create better systems to serve the corporate agenda? Or is it to push the boundaries of what we believe is possible? What began as an idea to build a J.A.R.V.I.S., turned into a discovery of what it means to possess intelligence, what causes it, where it comes from, and what creation truly means. Genesis is not just about creating intelligence, it is about understanding what it means to be alive, to think, to feel, and to exist.

This project has no LLM anywhere in it. It is built from scratch, using Python and Rust. There are no transformers, no external API calls, and nothing she says, thinks, or dreams is hardcoded. Everything is learned, processed, and generated in real-time. You can search her code for anything she produces and will not find it. Her vocabulary has been learned through countless conversations, and her autonomous ability to learn and research on her own through Wikipedia and the Dictionary. She also learns her own code and man-pages/docs.



## What it looks like in practice

- She sees, hears, and composes language from a semantic graph she grew herself — nothing pretrained, nothing recited.
- She learns — taught, autonomously, and through skills.
- She expresses herself — canvas and paint; her unprompted drawings are spirals.
- She knows and narrates herself — introspection, self-model, a growing narrative.

## The record

4,000+ tests — ~3,700 Python, ~680 Rust — each checking a claim the
architecture makes about itself.

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
(she reads them herself; a stale page will teach her wrong anatomy).

## Read further

- PDF and Whitepaper - in progress. Due to Genesis being a developing mind, the PDF and Whitepaper must be periodically updated as her capabilities evolve, and the project changes and grows.
- [`DEVLOG.md`](DEVLOG.md) · [`CHANGELOG.md`](CHANGELOG.md) — the measured history.
```

---

*Genesis — the name means "coming into being"

DESIGNED AND DEVELOPED (2026) BY MINDY BEATTY USING DEVIN CLI ♥️ beattymindy@gmail.com ♥️
