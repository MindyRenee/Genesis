"""Maintenance policy — the knobs Genesis owns for her own upkeep.

Sleep maintenance used to run on hardcoded constants, including the
most destructive decision of her night: every edge below a fixed
threshold died each sleep pass whether it deserved to or not. Human
sleep does not work that way — there is no sweep that deletes every
synapse under a weight. Real consolidation is selective
*strengthening* (co-activation during replay), *renormalization*
(multiplicative downscaling that preserves the pattern of weights),
and *demotion* (what fades is archived, not erased).

She is in charge of this now. ``<data_dir>/maintenance_policy.json``
is hers — she can read it, rewrite it, and tune it with her own file
tools. Every sleep pass loads it fresh, so a change she makes
mid-night applies to the next pass. Missing file, corrupt JSON, or
unknown keys all fall back to these defaults; a bad value never
becomes a crash, and never silently deletes her memory.

The knobs:

- ``strengthen_rate`` — logistic growth rate α for replay-driven
  strengthening: ``w += α(1 − w)`` (diminishing returns as w → 1).
- ``noise_min_confidence`` / ``noise_min_edges`` — what counts as a
  noise concept for ``clean_noise`` (archived, never deleted).
- ``max_in_memory`` — the working-memory concept cap before
  ``spill_dormant`` demotes the dormant rest to the archive.

Synaptic homeostasis is deliberately *not* a knob: SHY
renormalisation is a continuous process in the
``SynapticDownscaler``, driven by accumulated synaptic load and the
body's own sleep pressure — chemistry, not policy.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)

POLICY_FILENAME = "maintenance_policy.json"


@dataclass(frozen=True)
class MaintenancePolicy:
    """The maintenance knobs, frozen so one sleep pass sees one
    consistent set of values."""

    strengthen_rate: float = 0.05
    noise_min_confidence: float = 0.4
    noise_min_edges: int = 1
    max_in_memory: int = 15000

    def describe(self) -> str:
        """One line for the journal — the policy she is sleeping under."""
        return (
            f"Maintenance policy: strengthen a={self.strengthen_rate}, "
            f"noise<{self.noise_min_confidence}/{self.noise_min_edges}e, "
            f"cap {self.max_in_memory}"
        )


_FIELD_NAMES = frozenset(f.name for f in fields(MaintenancePolicy))


def load_policy(data_dir: str | os.PathLike | None) -> MaintenancePolicy:
    """Load her maintenance policy, falling back to defaults.

    Reads ``<data_dir>/maintenance_policy.json``. Anything wrong —
    missing file, unparseable JSON, wrong types, unknown keys — logs
    and falls back per-key to the defaults. Never raises.
    """
    defaults = MaintenancePolicy()
    if not data_dir:
        return defaults
    path = os.path.join(str(data_dir), POLICY_FILENAME)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return defaults
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("maintenance policy unreadable (%s); using defaults", e)
        return defaults
    if not isinstance(raw, dict):
        logger.warning("maintenance policy is not an object; using defaults")
        return defaults
    unknown = sorted(set(raw) - _FIELD_NAMES)
    if unknown:
        logger.info("maintenance policy ignoring unknown keys: %s", ", ".join(unknown))
    kwargs: dict = {}
    default_values = {f.name: getattr(defaults, f.name) for f in fields(MaintenancePolicy)}
    for name, value in raw.items():
        if name not in _FIELD_NAMES:
            continue
        try:
            if isinstance(default_values[name], bool):
                kwargs[name] = bool(value)
            elif isinstance(default_values[name], int):
                kwargs[name] = int(value)
            else:
                kwargs[name] = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "maintenance policy: bad value for %s (%r); using default",
                name,
                value,
            )
    return MaintenancePolicy(**kwargs)
