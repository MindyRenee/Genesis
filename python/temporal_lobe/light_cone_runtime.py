"""Bridge grounded world events into the adaptive light-cone manifold.

Only existing concept vectors are accepted; missing vectors are a no-op.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from temporal_lobe.events import ExternalEvent
from temporal_lobe.light_cones import AdaptiveLightConeManifold, SpacetimeEvent, WorldDomain


class LightConeExperienceBridge:
    """Thread-safe adapter from grounded concepts to the shared Self apex."""

    def __init__(self, *, max_dimensions: int = 256) -> None:
        if isinstance(max_dimensions, bool) or not isinstance(max_dimensions, int):
            raise ValueError("max_dimensions must be an integer")
        if max_dimensions < 1:
            raise ValueError("max_dimensions must be positive")
        self.manifold = AdaptiveLightConeManifold()
        self.max_dimensions = max_dimensions
        self._sequence = 0
        self._epoch_origin: float | None = None
        self._lock = threading.RLock()

    def process(self, event: ExternalEvent, embeddings: Any) -> bool:
        """Process an event when its declared topics have actual vectors.

        Position remains the origin because this adapter receives semantic
        vectors, not calibrated physical coordinates. Timestamps are epoch
        seconds and are monotonically clamped to tolerate delayed callbacks.
        """
        if not event.inbound:
            return False
        vectors: list[np.ndarray] = []
        for topic in event.topics:
            try:
                vector = embeddings.get_concept_vector(topic)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if vector is None:
                continue
            try:
                values = np.asarray(vector, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if values.size == 0 or not np.all(np.isfinite(values)):
                continue
            vectors.append(values[: self.max_dimensions])
        if not vectors:
            return False

        width = min(self.max_dimensions, max(vector.size for vector in vectors))
        aligned = [np.pad(vector, (0, width - vector.size)) for vector in vectors]
        latent = np.mean(np.stack(aligned, axis=0), axis=0)
        if not np.all(np.isfinite(latent)):
            return False

        timestamp = float(event.timestamp)
        salience = float(event.salience)
        if not math.isfinite(timestamp) or not math.isfinite(salience):
            return False
        with self._lock:
            if self._epoch_origin is None:
                self._epoch_origin = timestamp
            # Convert wall-clock epoch seconds to a session-relative timeline;
            # the causal model requires one consistent time origin.
            timestamp -= self._epoch_origin
            apex = self.manifold.apex
            if apex is not None:
                timestamp = max(timestamp, apex.time_s)
            self._sequence += 1
            event_id = (
                f"world:{event.source}:{event.kind.value}:"
                f"{timestamp:.9f}:{self._sequence}"
            )
            experience = SpacetimeEvent(
                event_id=event_id,
                time_s=timestamp,
                position=(0.0, 0.0, 0.0),
                domain=WorldDomain.OUTER,
                confidence=max(0.0, min(1.0, salience)),
            )
            self.manifold.process_experience(
                experience,
                (complex(float(value), 0.0) for value in latent),
            )
        return True
