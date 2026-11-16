"""Man page sync tests — the manual's bookkeeping contract.

The versioned sources in ``docs/man/`` are canonical: ``sync`` flows
them to the installed man dir and retires installed ``genesis*``
pages that no longer have a source — in both plain and gzip forms.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from self import man_pages


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs" / "man"
    docs.mkdir(parents=True)
    home = tmp_path / "share" / "man"
    return docs, home


def test_sync_installs_and_is_idempotent(tmp_path: Path) -> None:
    docs, home = _tree(tmp_path)
    (docs / "genesis.1").write_text(".TH genesis 1")
    (docs / "genesis-tools.7").write_text(".TH genesis-tools 7")

    touched = man_pages.sync(docs, home)
    assert sorted(touched) == ["genesis-tools.7", "genesis.1"]
    assert (home / "man1" / "genesis.1").read_text() == ".TH genesis 1"
    assert (home / "man7" / "genesis-tools.7").read_text() == ".TH genesis-tools 7"

    # Second pass: content already equal — nothing touched.
    assert man_pages.sync(docs, home) == []


def test_sync_retires_gz_pages(tmp_path: Path) -> None:
    """A compressed stale page must retire like a plain one.

    ``owned_pages`` strips the ``.gz`` suffix when enumerating
    installed pages; retirement used to build only the plain
    filename, so a ``genesis-old.1.gz`` survived every sync and kept
    being ingested at boot.
    """
    docs, home = _tree(tmp_path)
    (docs / "genesis.1").write_text(".TH genesis 1")
    (home / "man1").mkdir(parents=True)
    (home / "man1" / "genesis.1").write_text(".TH genesis 1")
    (home / "man1" / "genesis-old.1").write_bytes(b"stale")
    (home / "man1" / "genesis-old.1.gz").write_bytes(b"stale")

    touched = man_pages.sync(docs, home)
    assert "genesis-old.1 (retired)" in touched
    assert "genesis-old.1.gz (retired)" in touched
    assert sorted(p.name for p in (home / "man1").iterdir()) == ["genesis.1"]


def test_sync_without_source_retires_nothing(tmp_path: Path) -> None:
    """No docs/man tree means nothing is canonical — installed pages
    are left alone rather than mass-retired."""
    home = tmp_path / "share" / "man"
    (home / "man1").mkdir(parents=True)
    (home / "man1" / "genesis.1").write_text(".TH genesis 1")

    assert man_pages.sync(None, home) == []
    assert man_pages.sync(tmp_path / "absent", home) == []
    assert (home / "man1" / "genesis.1").is_file()


def test_owned_pages_covers_both_forms(tmp_path: Path) -> None:
    docs, home = _tree(tmp_path)
    (docs / "genesis.1").write_text(".TH genesis 1")
    (docs / "genesis-daemon.8").write_text(".TH genesis-daemon 8")
    (home / "man3").mkdir(parents=True)
    (home / "man3" / "genesis-fake.3.gz").write_bytes(b"gz")
    (home / "man1").mkdir(parents=True)
    (home / "man1" / "ls.1").write_text("not hers")

    pages = man_pages.owned_pages(docs, home)
    assert ("genesis", "1") in pages
    assert ("genesis-daemon", "8") in pages
    assert ("genesis-fake", "3") in pages
    assert ("ls", "1") not in pages
