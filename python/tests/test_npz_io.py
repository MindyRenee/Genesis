"""Tests for atomic, pickle-free ``.npz`` persistence.

Genesis keeps several large learned artifacts as ``.npz`` files. They
previously shared two defects: writes went straight to the target, so
a crash part-way through left a truncated file; and string lists were
stored with ``dtype=object``, which forces ``allow_pickle=True`` on
load — arbitrary code execution for anyone who can write the file, and
never necessary, since NumPy's native ``<U`` dtype stores the same
strings as plain UTF-8.
"""

import os
import pickle
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_cognitive.infrastructure._npz_io import load_npz, save_npz, str_array


def test_round_trip_preserves_arrays():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        save_npz(
            path,
            names=str_array(["dog", "water", "hippocampus"]),
            vectors=np.arange(12, dtype=np.float32).reshape(4, 3),
            count=np.array(42),
        )
        data = load_npz(path)
        assert [str(n) for n in data["names"]] == ["dog", "water", "hippocampus"]
        assert np.array_equal(data["vectors"], np.arange(12, dtype=np.float32).reshape(4, 3))
        assert int(data["count"]) == 42


def test_saved_file_needs_no_pickle():
    """The whole point: a fresh file must open with allow_pickle=False.

    If this regresses, every load silently regains an arbitrary-code-
    execution surface for anyone who can write the state directory.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        save_npz(path, names=str_array(["dog", "water"]), v=np.ones(2, np.float32))
        with np.load(path, allow_pickle=False) as data:
            assert [str(n) for n in data["names"]] == ["dog", "water"]


def test_str_array_handles_empty_without_becoming_float():
    """An empty list must not infer float64 — that silently corrupts type."""
    arr = str_array([])
    assert arr.dtype.kind == "U", arr.dtype
    arr2 = str_array(["only"])
    assert arr2.dtype.kind == "U", arr2.dtype


def test_save_is_atomic_no_temp_files_left():
    """A successful save leaves exactly the target, no debris."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.npz")
        save_npz(path, x=np.array([1, 2, 3]))
        entries = sorted(os.listdir(d))
        assert entries == ["a.npz"], entries


def test_failed_save_leaves_previous_file_intact(tmp_path):
    """A crash mid-write must not destroy the last good artifact.

    The un-serialisable value makes np.savez fail after the temp file
    exists, which is exactly the window a real crash occupies. The
    previous file must still open and still hold its old contents.
    """
    path = str(tmp_path / "a.npz")
    save_npz(path, x=np.array([1, 2, 3]))
    # np.savez silently coerces a dict, so it does not fail; a lambda
    # cannot be pickled and does, after the temp file exists.
    unserializable = lambda: 1  # noqa: E731
    # The exact type comes from numpy's npz writer (PicklingError from
    # pickle, or AttributeError from the array-format path depending on
    # version). What matters is that it propagates and the invariant
    # below holds, not which of them it is.
    with pytest.raises((pickle.PicklingError, TypeError, AttributeError)):
        save_npz(path, x=unserializable)
    data = load_npz(path)
    assert np.array_equal(data["x"], np.array([1, 2, 3]))
    # ...and the failed attempt cleaned up its temp file.
    assert sorted(os.listdir(tmp_path)) == ["a.npz"]


def test_save_errors_propagate_rather_than_being_swallowed(tmp_path):
    """A failed save must be visible to the caller.

    These sites used to catch the error and log at DEBUG, so a
    truncated or unwritten artifact went unnoticed while the in-memory
    state carried on diverging from disk.
    """
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)  # read+execute: cannot create files
    try:
        with pytest.raises(OSError):
            save_npz(str(unwritable / "x.npz"), x=np.array([1]))
    finally:
        unwritable.chmod(0o700)


def test_legacy_pickled_file_still_loads(tmp_path):
    """A pre-existing object-dtype file must remain readable.

    The fallback exists so this change does not invalidate every
    existing state directory. It warns once; the window closes when
    every installation has been rewritten.
    """
    path = str(tmp_path / "legacy.npz")
    np.savez(
        path,
        names=np.array(["dog", "water"], dtype=object),
        v=np.ones(2, np.float32),
    )
    # Confirm the fixture really needs pickle. Note numpy raises on
    # *access*, not on open — which is why load_npz() materialises the
    # arrays inside its try block rather than just opening the archive.
    with pytest.raises(ValueError):
        with np.load(path, allow_pickle=False) as probe:
            _ = probe["names"]
    data = load_npz(path)
    assert [str(n) for n in data["names"]] == ["dog", "water"]
