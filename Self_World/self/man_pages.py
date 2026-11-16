"""Her manual — the man pages that describe what she is.

Genesis's man pages (``genesis.1``, ``genesis-cli.1``,
``genesis-daemon.8``, ``genesis-retina.1``, ``genesis-tools.7``)
are part of her — how the machine, and she herself, knows what she
is. The versioned sources in ``docs/man/`` are canonical; the
installed copies under ``~/.local/share/man`` are what ``man`` —
and her ``man_page`` tool — actually reads. Subsystem internals are
not man pages: they are generated docs under ``Self_World/docs/``
(rustdoc + pydoc via ``scripts/gen_docs.sh``), so they cannot
drift from the code they describe.

``sync()`` keeps the two equal: each versioned source flows to the
installed dir, and installed ``genesis*`` pages with no source are
retired — a stale page teaches her wrong anatomy every boot. There
is no staging and no queue: the manual says what the project says,
and keeping the installed copy honest is bookkeeping, not a
decision anyone gates.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

_SECTIONS = {".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"}


def man_home() -> Path:
    """The user man dir ``man`` reads — ``$XDG_DATA_HOME/man`` or
    ``~/.local/share/man``, same resolution ``install.sh`` uses."""
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "man"


def owned_pages(docs_man: Path | None, home: Path | None = None) -> list[tuple[str, str]]:
    """(name, section) pairs of the pages that are hers.

    The owned set is whatever the versioned docs source declares,
    plus any ``genesis*`` pages already installed in the user man dir
    (covers pages present without a docs tree, e.g. a packaged
    install with no checkout above it).
    """
    home = home or man_home()
    pages: set[tuple[str, str]] = set()
    if docs_man is not None and docs_man.is_dir():
        for src in docs_man.glob("*.[0-9]"):
            pages.add((src.stem, src.suffix.lstrip(".")))
    for mandir in home.glob("man*"):
        if not mandir.is_dir():
            continue
        for page in mandir.iterdir():
            stem = page.name.removesuffix(".gz")
            name, _, sec = stem.rpartition(".")
            if name.startswith("genesis") and sec.isdigit():
                pages.add((name, sec))
    return sorted(pages)


def sync(docs_man: Path | None, home: Path | None = None) -> list[str]:
    """Flow the versioned sources to the installed man dir.

    Every ``docs/man/*.[0-9]`` page is copied over its installed
    counterpart — creating the section dir when needed — and any
    installed ``genesis*`` page that no longer has a source is
    removed. Returns the names of pages it touched. Idempotent:
    a page already in sync costs a hash comparison, not a write.
    """
    home = home or man_home()
    touched: list[str] = []
    if docs_man is None or not docs_man.is_dir():
        return touched  # no versioned source — nothing is canonical
    sources: set[tuple[str, str]] = set()
    for src in sorted(docs_man.iterdir()):
        if src.suffix not in _SECTIONS or not src.is_file():
            continue
        sources.add((src.stem, src.suffix[1:]))
        dest = home / f"man{src.suffix[1:]}" / src.name
        try:
            content = src.read_bytes()
            if dest.is_file() and dest.read_bytes() == content:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            touched.append(src.name)
        except OSError as e:
            log.warning("man sync: cannot write %s: %s", dest, e)
    # Retire installed genesis* pages with no source — a page removed
    # from docs/man is a page that should stop teaching her.
    for name, section in owned_pages(None, home):
        if (name, section) in sources:
            continue
        stale = home / f"man{section}" / f"{name}.{section}"
        try:
            stale.unlink()
            touched.append(f"{name}.{section} (retired)")
        except OSError as e:
            log.warning("man sync: cannot retire %s: %s", stale, e)
    return touched


class ManPagesMixin:
    """Mixin for :class:`Mind` — learning her own manual.

    ``self._man_docs`` points at the versioned sources (created in
    ``__init__``); ``_ingest_man_pages`` reads them into her concept
    network the way any learned material lands.
    """

    if TYPE_CHECKING:
        learner: Any
        self_model: Any
        _man_docs: Path | None

        def __getattr__(self, name: str) -> Any: ...

    def _ingest_man_pages(self) -> None:
        """Learn her own manual — constitutively, not as a task.

        Her man pages are part of her: how she is described, built,
        and run. They go through the same extraction machinery as any
        learned material — concepts, relationships, definitions land
        in her concept network — but unconditionally at boot, not via
        a curiosity topic she might never pick. The result is recorded
        in self_knowledge as fact about what she is, not as words she
        recites.
        """
        try:
            from tools.source_registry import ManPageSource

            source = ManPageSource()
            learned: list[str] = []
            for name, _section in owned_pages(self._man_docs):
                sr = source.lookup(name)
                if sr is None:
                    continue
                if self.learner.ingest_endogenous(sr, topic=name):
                    learned.append(sr.url)
            self.self_model.self_knowledge["manual_pages"] = learned
            if learned:
                log.info("Read her own manual into the network: %s", ", ".join(learned))
        except Exception:
            log.exception("man page self-ingestion failed")
