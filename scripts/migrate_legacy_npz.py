#!/usr/bin/env python3
"""Convert a trusted legacy NPZ to a pickle-free file while the organism is stopped.

The input is unpickled and may execute code. Run this only on state files
whose provenance you trust. The output must be a new path; the original
file is never replaced by this script.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infrastructure._npz_io import load_npz, save_npz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.is_file() or args.destination.exists() or args.destination.is_symlink():
        parser.error("source must exist and destination must not exist")
    arrays = load_npz(args.source, allow_legacy_pickle=True)
    for name, array in arrays.items():
        if array.dtype.hasobject:
            if not all(isinstance(value, str) for value in array.flat):
                parser.error(f"{name} contains non-string objects and cannot be converted")
            arrays[name] = array.astype(np.str_)
    fd, tmp_name = tempfile.mkstemp(
        dir=args.destination.parent, prefix=args.destination.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        save_npz(tmp, **arrays)
        os.link(tmp, args.destination)
    finally:
        tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
