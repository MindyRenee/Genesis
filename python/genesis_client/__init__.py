"""Genesis client library — the cognitive mind's bridge to the subcognitive.

This package provides a Python client for communicating with the Genesis
subcognitive daemon over a Unix domain socket. It implements the binary
IPC protocol defined in the Rust crate's `daemon::ipc` module.

# Quick start

    import os
    from genesis_client import GenesisClient

    # The daemon's socket lives in the data dir (default: ~/.local/share/genesis-public/)
    with GenesisClient(os.path.expanduser("~/.local/share/genesis-public/genesis.sock")) as client:
        summary = client.get_neuro_summary()
        print(f"Genesis is feeling: {summary.phase_name}")
        print(f"  arousal: {summary.arousal:.2f}")
        print(f"  valence: {summary.valence:.2f}")

# Public API

The top-level package re-exports the two symbols most callers need:
``GenesisClient`` (the client class) and ``NeuroSummary`` (the most
commonly read return type). For everything else — protocol constants,
wire types, exceptions — import directly from the relevant submodule:

    from genesis_client.protocol import PHASE_FLOW, CHEM_DOPAMINE
    from genesis_client.types import InferenceSummary, Episode
    from genesis_client.exceptions import EpisodeNotFound
"""

from .client import GenesisClient
from .types import NeuroSummary

__all__ = [
    "GenesisClient",
    "NeuroSummary",
]
