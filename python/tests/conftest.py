"""Pytest configuration for Genesis tests.

Adds the parent directory (``python/``) to ``sys.path`` so that
``genesis_cognitive`` and ``genesis_client`` are importable without
each test file having to manipulate ``sys.path`` itself. This keeps
test imports at the top of the file (no E402 violations) while
preserving the same import behavior.
"""

import os
import sys

# Insert the python/ directory (parent of tests/) so that both
# genesis_cognitive and genesis_client are importable. This replaces
# the per-file ``sys.path.insert(0, ...)`` calls that were scattered
# across the test suite and triggered E402 (module-level import not
# at top of file) lint violations.
_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

# Some tests also need the scripts/ directory on the path (e.g.
# test_concept_network imports from prune_dead_concepts.py). The
# scripts/ directory lives at the project root (one level above
# python/), not under python/.
_PROJECT_ROOT = os.path.dirname(_PYTHON_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
