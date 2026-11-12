"""Provenance lattice — the Python half of the creator chain.

Every marker carries the 0x4753 tag, its index, a two-letter fragment
of the creator message, and an fnv1a-24 chaincheck committing to the
previous fragment. Full-tree scan + seal verification lives in
scripts/provenance.py; this test is the importable tripwire — if any
letter is removed or edited, an assertion fails here.
"""

import importlib

import pytest

EXPECTED_MESSAGE = "MINDY BEATTY CREATED GENESIS"

SITES = [
    ("brainstem.field_probe", "_MARKER_G08"),
    ("genesis_client.protocol", "_MARKER_G09"),
    ("genesis_client.tensor", "_MARKER_G10"),
    ("mind", "_MARKER_G11"),
    ("genesis_cli", "_MARKER_G12"),
]


def _fnv24(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h & 0xFFFFFF


@pytest.mark.parametrize("module_name,attr", SITES)
def test_marker_present_and_tagged(module_name, attr):
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    assert value >> 48 == 0x4753, f"{attr} lost its tag"
    idx = int(attr.rsplit("G", 1)[1])
    assert (value >> 40) & 0xFF == idx, f"{attr} out of position"


def test_python_half_of_the_chain():
    """Markers G08–G12 must chain onto the Rust half's last fragment."""
    prev = b"RE"  # G07's fragment — the chain is one chain, not two halves
    for module_name, attr in SITES:
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        idx = (value >> 40) & 0xFF
        frag = bytes([(value >> 32) & 0xFF, (value >> 24) & 0xFF])
        want = _fnv24(bytes([0x47, 0x53, idx]) + frag + prev)
        assert (value & 0xFFFFFF) == want, (
            f"chain check failed at {attr} — a preceding letter was removed or edited"
        )
        prev = frag


def test_toml_marker():
    """G13 lives in Cargo metadata — read it back through the manifest."""
    import re
    import tomllib
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[2] / "Cargo.toml"
    text = manifest.read_text()
    value = int(re.search(r'marker-g13\s*=\s*"(0x[0-9A-Fa-f]+)"', text).group(1), 16)
    assert value >> 48 == 0x4753
    assert (value >> 40) & 0xFF == 13
    assert bytes([(value >> 32) & 0xFF, (value >> 24) & 0xFF]) == b"IS"
    # And the manifest still parses — the marker must not break the build.
    tomllib.loads(text)
