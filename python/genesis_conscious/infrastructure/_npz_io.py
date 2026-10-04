"""Atomic, pickle-free persistence for the ``.npz`` artifact files.

Genesis keeps several large learned artifacts as NumPy ``.npz``:
the concept embedding matrix, the holographic graph, the VQ codebook,
and the two visual-cortex layers. They all shared two defects.

**Non-atomic writes.** ``np.savez(path, ...)`` writes straight to the
target. A crash, an OOM kill, or a full disk part-way through leaves a
truncated file that will not open — and these are large learned
structures whose loss costs a full rebuild. The rest of the codebase
already does this correctly (``persistence._atomic_write_state``,
``edge_log.EdgeLog.compact``), so these sites were the outliers.

**Unnecessary pickling.** String lists were stored as
``np.array(names, dtype=object)``, which serialises via pickle, which
forced ``allow_pickle=True`` on load. Pickle on load is arbitrary code
execution for anyone who can write the file, and it was never needed:
NumPy's native ``<U`` dtype stores the same strings as plain UTF-8.

The legacy problem: files written before this change contain pickled
object arrays, so ``allow_pickle=False`` cannot open them. Rather than
either breaking every existing state directory or leaving the RCE
vector open, :func:`load_npz` tries the safe path first and falls back
to the legacy one, logging a warning so the window is visible. Once
every installation has been rewritten, the fallback can go.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import numpy as np

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

__all__ = ["legacy_pickle_warned", "load_npz", "save_npz", "str_array"]


# One warning per process, not per load: these files are opened on
# every startup and a repeated warning would drown the log.
legacy_pickle_warned = False


def str_array(values: Any) -> np.ndarray:
    """A string array NumPy can save without pickling.

    ``np.array(list_of_str)`` infers the native fixed-width unicode
    dtype (``<U``), which round-trips through ``.npz`` as plain UTF-8
    and therefore needs no pickle. An empty list still needs an
    explicit dtype or NumPy defaults to ``float64``.
    """
    if len(values) == 0:
        return np.array([], dtype=np.str_)
    return np.array(list(values))


def save_npz(path: str | os.PathLike[str], **arrays: Any) -> None:
    """Write a ``.npz`` atomically.

    Serialises to a temporary file in the *same directory* (so
    ``os.replace`` is atomic — a cross-device rename is not), fsyncs
    it, then renames over the target. A reader therefore sees either
    the complete previous file or the complete new one, never a
    truncated mixture.

    Propagates errors. A failed save means the in-memory learned state
    and the on-disk state have diverged, and the caller needs to know;
    swallowing it (as these sites previously did, at DEBUG) is how a
    silently-truncated artifact goes unnoticed for months.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError as unlink_err:
            note_swallowed(
                "genesis_conscious.infrastructure._npz_io.save_npz",
                unlink_err,
            )
        raise


def load_npz(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a ``.npz``, preferring the pickle-free path.

    Falls back to ``allow_pickle=True`` for files written before
    :func:`save_npz` existed, and warns once when it does. That
    fallback is the only remaining pickle surface in the project and
    should be removable once every state directory has been rewritten.
    """
    global legacy_pickle_warned
    path = os.fspath(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except ValueError as e:
        # numpy raises ValueError when a pickled object array is
        # present and allow_pickle is False.
        if not legacy_pickle_warned:
            legacy_pickle_warned = True
            logger.warning(
                "loading %s required pickle (legacy format written before "
                "atomic npz persistence). It will be rewritten in the "
                "native format on the next save. Source: %s",
                os.path.basename(path),
                e,
            )
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
