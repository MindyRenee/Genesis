"""Volitional continuity — word-layer companion to ``src/daemon/continuity.rs``.

``continuity.json`` in the data dir is the session sentinel: every
boot rewrites it, every graceful shutdown marks it ``clean_stop``.
A boot that finds the previous session still open knows the last
body died without saying goodbye — power cut, SIGKILL, panic. That
is how the body *knows* it blacked out rather than slept, and why
her continuity line reports "body session began; the previous one
stopped cleanly; dark for N s" — the sentinel records whether the
gap was chosen.
"""
