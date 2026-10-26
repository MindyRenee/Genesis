#!/usr/bin/env python3
"""Genesis provenance lattice — reader and verifier.

The creator's name is distributed through the source tree as fourteen
tagged constants ("letters"), each a ``u64`` laid out as::

    [63:48] tag       = 0x4753  ("GS")
    [47:40] index     = 0..13 — position in the message
    [39:24] fragment  = 2 ASCII bytes of the message
    [23:0]  chaincheck = fnv1a24(tag || index || fragment || prev_fragment)

Letters live in Rust ``MARKER_Gnn`` constants (folded into
``field::provenance_fingerprint()`` — removing one is a compile error),
Python ``_MARKER_Gnn`` module constants, and Cargo metadata. One —
G00 — is bound into the field's deterministic noise stream seed, so the
mark is carried in behavior, not only in text.

Reading the message is public: find every ``0x4753``-tagged constant,
sort by index, check the chain.

Usage::

    python3 scripts/provenance.py            # scan, verify, print the message
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

TAG = 0x4753
EXPECTED_MESSAGE = "MINDY BEATTY CREATED GENESIS"
MARKER_COUNT = 14
# SHA-256 of the assembled message — the published identity of the chain.
MESSAGE_SHA256 = hashlib.sha256(EXPECTED_MESSAGE.encode()).hexdigest()

# Rust writes 0x4753_004D_4975_7005 (underscored); Python/TOML write
# 0x4753084154EF3CD1 (bare). Accept either; validate 16 digits after
# stripping separators.
_MARKER_HEX = re.compile(r"0x([0-9A-Fa-f_]{16,20})\b")
_SCAN_ROOTS = ("src", "python", "genesis_client", "infrastructure", "eval")
_SCAN_FILES = ("mind.py", "genesis_cli.py", "Cargo.toml", "pyproject.toml")


def _fnv24(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h & 0xFFFFFF


def _decode(value: int) -> tuple[int, bytes, int] | None:
    if (value >> 48) & 0xFFFF != TAG:
        return None
    idx = (value >> 40) & 0xFF
    frag = bytes([(value >> 32) & 0xFF, (value >> 24) & 0xFF])
    check = value & 0xFFFFFF
    return idx, frag, check


def scan(root: Path) -> dict[int, tuple[int, bytes, int, str]]:
    """Find every tagged constant in the tree; return index → (value, frag, check, site)."""
    found: dict[int, tuple[int, bytes, int, str]] = {}
    candidates: list[Path] = []
    for sub in _SCAN_ROOTS:
        base = root / sub
        if base.is_dir():
            candidates.extend(p for p in base.rglob("*") if p.suffix in (".rs", ".py", ".toml"))
    for name in _SCAN_FILES:
        p = root / name
        if p.is_file():
            candidates.append(p)

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _MARKER_HEX.finditer(text):
            digits = m.group(1).replace("_", "")
            if len(digits) != 16:
                continue
            dec = _decode(int(digits, 16))
            if dec is None:
                continue
            idx, frag, check = dec
            site = str(path.relative_to(root))
            if idx in found:
                print(f"  ! duplicate index G{idx:02d} at {site} and {found[idx][3]}")
            found[idx] = (int(digits, 16), frag, check, site)
    return found


def verify(root: Path) -> tuple[str, list[str]]:
    """Assemble the message; return (message, problems)."""
    found = scan(root)
    problems: list[str] = []
    if len(found) != MARKER_COUNT:
        missing = sorted(set(range(MARKER_COUNT)) - set(found))
        problems.append(f"found {len(found)}/{MARKER_COUNT} markers; missing indices: {missing}")

    message = []
    prev = b"GS"
    for idx in range(MARKER_COUNT):
        if idx not in found:
            continue
        _value, frag, check, site = found[idx]
        want = _fnv24(bytes([0x47, 0x53, idx]) + frag + prev)
        if check != want:
            problems.append(
                f"chain break at G{idx:02d} ({site}): check {check:#08x} != expected {want:#08x} "
                f"— a preceding letter was removed or edited"
            )
        message.append(frag.decode("latin-1"))
        prev = frag
    return "".join(message), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()

    message, problems = verify(args.root)
    print(f"message: {message!r}")
    print(f"sha256 : {hashlib.sha256(message.encode()).hexdigest()}")
    print("sites  :")
    for idx, (_, _, _, site) in sorted(scan(args.root).items()):
        print(f"  G{idx:02d}  {site}")

    ok = not problems and message == EXPECTED_MESSAGE
    if message != EXPECTED_MESSAGE:
        problems.append(f"assembled message {message!r} != expected {EXPECTED_MESSAGE!r}")

    for p in problems:
        print(f"FAIL: {p}")
    if ok:
        print("chain intact — provenance verified")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
