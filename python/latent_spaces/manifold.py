"""The latent manifold — Genesis's unified living substrate.

One closed, positively-curved 3-manifold (S^3 with its Hopf structure)
is the container everything else projects from: memory, language,
learning, self. A trace is a point deposited on a fiber at a depth —
its ``|z1|`` coordinate places it in the inner world (near the self
core), the outer world (toward the other core), or the vesica band
where the two counter-wound strand families interlock.

Conventions:

- **Deposits are append-only.** Nothing is ever removed; forgetting
  means the ``reach`` field attenuates until a trace no longer answers
  a resonance query — unreachable, never deleted.
- **Retrieval is resonance, not lookup.** A cue is a point injected
  into the medium; traces answer by proximity weighted by salience and
  reach. Recall degrades smoothly as the cue moves off its source —
  holographic behavior, no cliff.
- **The medium moves.** ``advance`` turns the whole space: the frame
  rotation carries every deposit along its own fiber (the heartbeat);
  the counter-rotation shears the two strand families against each
  other (the braid the worlds think with). Both are isometries — the
  geometry survives the motion.
- **Self has a depth, not a point.** ``lean`` is where self-presence
  sits on the core-to-core axis; the verified exact-vesica layer at
  ``sqrt(3)/2`` is the resting position. It drifts: toward the inner
  core in sleep, toward the outer world in engagement.
- **The image persists the substrate, not the memories.** ``save``
  writes the manifold's state (points, strands, reach, phase, lean)
  to ``manifold.npz`` — the same relationship ``core_state.bin`` has
  to the body.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time

import numpy as np
from numpy.typing import NDArray

from infrastructure._npz_io import load_npz, save_npz, str_array

from . import geometry as g

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Strand families: the two counter-winding loop families on each torus
# layer. Inbound deposits (the world arriving at her) are laid on the
# (1,1) Hopf family; outbound deposits (her acting on the world) on the
# (1,-1) anti-family, so a complete episode is a rung between strands.
STRAND_HOPF = np.int8(0)
STRAND_ANTI = np.int8(1)

# Torus-depth bands by domain. Inner events settle in the inner solid
# torus, outer events in the outer, and self-domain events in the band
# around the exact-vesica layer where the strands interlock.
_DOMAIN_DEPTH = {
    "inner": (0.05, 0.65),
    "self": (0.80, 0.93),
    "outer": (0.72, 0.98),
}

_FORMAT_VERSION = 1


def _domain_of(kind: str) -> str:
    if kind.startswith("world:"):
        return "outer"
    if kind in {"self", "self_model", "identity"}:
        return "self"
    return "inner"


class LatentManifold:
    """The append-only, self-shaping latent substrate.

    Thread-safe: ``observe`` can be called from thought/world-event
    callbacks on background threads while the metronome advances the
    medium.
    """

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = os.fspath(data_dir) if data_dir is not None else None
        self._lock = threading.RLock()
        self._points = np.empty((0, 4), dtype=np.float64)
        self._salience = np.empty(0, dtype=np.float64)
        self._strand = np.empty(0, dtype=np.int8)
        self._born_ms = np.empty(0, dtype=np.int64)
        self._reach = np.empty(0, dtype=np.float64)
        # The medium phase at each deposit's birth — lets resonance
        # unwind every trace back to a common epoch, so the medium can
        # turn without carrying recall away with it.
        self._phase0 = np.empty(0, dtype=np.float64)
        self._shear0 = np.empty(0, dtype=np.float64)
        self._kinds: list[str] = []
        # The medium's own state.
        self._phase = 0.0
        self._shear_phase = 0.0
        self._lean = g.VESICA_DEPTH
        self._lean_target = g.VESICA_DEPTH
        self._ticks = 0
        self._dirty = False

    # ------------------------------------------------------------------
    # Deposit — experiences laid down as points on fibers
    # ------------------------------------------------------------------

    def _digest_point(self, content: str, kind: str, domain: str) -> FloatArray:
        """Deterministic content->point coding at the domain's depth band.

        The digest sets the point's phase angles (its content signature)
        and its position inside the domain's depth band. Same input,
        same landing site — retrieval and reconsolidation find it again.
        """
        payload = (kind + "\0" + content[:4096]).encode("utf-8", errors="replace")
        digest = hashlib.sha256(payload).digest()
        words = np.frombuffer(digest[:32], dtype=">u8").astype(np.float64)
        direction = (words / np.float64(2**63)) - 1.0
        direction += 1e-9 * np.arange(1, 5)  # break exact-zero edge cases
        point = g.unit(direction)
        lo, hi = _DOMAIN_DEPTH[domain]
        frac = int.from_bytes(digest[24:28], "big") / float(2**32 - 1)
        target = lo + frac * (hi - lo)
        try:
            return g.set_depth(point, target)
        except ValueError:
            # Direction landed on a core (measure zero) — nudge and retry.
            point = g.unit(direction + 1e-6)
            return g.set_depth(point, target)

    def observe(
        self,
        content: str,
        *,
        kind: str,
        salience: float = 0.4,
        inbound: bool = True,
        now_ms: int | None = None,
    ) -> int:
        """Deposit a trace; returns its index. Never mutates old deposits.

        ``inbound`` selects the strand family: the world's leg toward her
        rides the (1,1) fibers; her leg toward the world rides (1,-1).
        """
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if not np.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("salience must be finite and in [0, 1]")
        domain = _domain_of(kind)
        point = self._digest_point(content, kind, domain)
        born = int(time.time() * 1000) if now_ms is None else int(now_ms)
        strand = STRAND_HOPF if inbound else STRAND_ANTI
        with self._lock:
            idx = len(self._kinds)
            self._points = np.vstack([self._points, point[None, :]])
            self._salience = np.append(self._salience, float(salience))
            self._strand = np.append(self._strand, strand)
            self._born_ms = np.append(self._born_ms, born)
            self._reach = np.append(self._reach, 1.0)
            self._phase0 = np.append(self._phase0, self._phase)
            self._shear0 = np.append(self._shear0, self._shear_phase)
            self._kinds.append(kind)
            self._dirty = True
            return idx

    # ------------------------------------------------------------------
    # Resonance — retrieval by proximity in the medium
    # ------------------------------------------------------------------

    def resonate(
        self,
        cue: str | FloatArray,
        *,
        k: int = 5,
        radius: float = 0.5,
        kind: str = "query",
    ) -> list[tuple[int, str, float, float]]:
        """Recall the strongest traces near a cue.

        ``cue`` is raw content (digested like a deposit, at the domain
        its ``kind`` implies) or a point on S3. Score is
        ``reach * salience * exp(-(d / radius)^2)`` — a resonance kernel:
        nearby traces answer strongly, distant ones fade smoothly.
        Returns ``(index, kind, score, geodesic_distance)`` rows sorted
        by score.
        """
        if isinstance(cue, str):
            domain = _domain_of(kind)
            cue_point = self._digest_point(cue, kind, domain)
        else:
            cue_point = g.unit(np.asarray(cue, dtype=np.float64).reshape(1, -1))[0]
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("radius must be positive and finite")
        with self._lock:
            if len(self._kinds) == 0:
                return []
            # Unwind every deposit to the cue's epoch: a trace deposited
            # at medium phase (p0, s0) has since ridden (phase - p0,
            # shear - s0) of rotation. Comparing at a common epoch keeps
            # recall invariant while the medium turns.
            d_frame = self._phase - self._phase0
            d_shear = self._shear_phase - self._shear0
            unwound = g.rotate(self._points, -d_frame - d_shear, -d_frame + d_shear)
            d = g.geodesic_batch(unwound, cue_point)
            score = self._reach * self._salience * np.exp(-((d / radius) ** 2))
            order = np.argsort(score)[::-1][:k]
            return [
                (int(i), self._kinds[i], float(score[i]), float(d[i]))
                for i in order
                if score[i] > 0.0
            ]

    # ------------------------------------------------------------------
    # Dynamics — the medium's own motion
    # ------------------------------------------------------------------

    def set_lean_target(self, target: float) -> None:
        """Aim self-presence at a depth on the core-to-core axis."""
        if not np.isfinite(target):
            raise ValueError("lean target must be finite")
        with self._lock:
            self._lean_target = float(np.clip(target, 0.0, 1.0))

    def advance(
        self,
        dt: float,
        *,
        heartbeat_omega: float = g.TWO_PI / 60.0,
        shear: float = 0.0,
        lean_rate: float = 0.05,
    ) -> None:
        """Advance the medium by dt seconds.

        Frame rotation carries every deposit along its own fiber — the
        heartbeat, an isometry: every loop slides along itself. ``shear``
        adds counter-rotation, sliding deposits across the fibration —
        the braid between the worlds. ``lean`` drifts toward its target.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        with self._lock:
            d_phase = heartbeat_omega * dt
            d_shear = shear * dt
            if len(self._points):
                self._points = g.rotate(
                    self._points, d_phase + d_shear, d_phase - d_shear
                )
            self._phase = (self._phase + d_phase) % g.TWO_PI
            self._shear_phase = (self._shear_phase + d_shear) % g.TWO_PI
            step = min(1.0, lean_rate * dt)
            self._lean += (self._lean_target - self._lean) * step
            self._ticks += 1
            self._dirty = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def lean(self) -> float:
        """Current self-presence depth on the core-to-core axis."""
        with self._lock:
            return self._lean

    @property
    def phase(self) -> float:
        """Heartbeat phase accumulated by advance(), modulo 2*pi."""
        with self._lock:
            return self._phase

    @property
    def dirty(self) -> bool:
        """Whether state has changed since the last save."""
        with self._lock:
            return self._dirty

    def __len__(self) -> int:
        with self._lock:
            return len(self._kinds)

    def state_summary(self) -> dict[str, object]:
        """Concise diagnostics: deposit counts, domain mix, medium state."""
        with self._lock:
            if len(self._kinds) == 0:
                depths = np.empty(0)
            else:
                depths = g.depth(self._points)
            return {
                "deposits": len(self._kinds),
                "hopf_strand": int(np.sum(self._strand == STRAND_HOPF)),
                "anti_strand": int(np.sum(self._strand == STRAND_ANTI)),
                "inner_world": int(np.sum(depths < g.CLIFFORD_DEPTH)),
                "outer_world": int(np.sum(depths >= g.CLIFFORD_DEPTH)),
                "phase": self._phase,
                "lean": self._lean,
                "lean_target": self._lean_target,
                "ticks": self._ticks,
            }

    # ------------------------------------------------------------------
    # Persistence — the image holds the substrate, not the memories
    # ------------------------------------------------------------------

    @property
    def image_path(self) -> str | None:
        return (
            os.path.join(self.data_dir, "manifold.npz")
            if self.data_dir is not None
            else None
        )

    def save(self) -> None:
        """Write the manifold's image atomically. Errors propagate."""
        path = self.image_path
        if path is None:
            raise ValueError("no data_dir; nowhere to persist the image")
        with self._lock:
            save_npz(
                path,
                version=np.array(_FORMAT_VERSION, dtype=np.int32),
                points=self._points.astype(np.float64),
                salience=self._salience,
                strand=self._strand,
                born_ms=self._born_ms,
                reach=self._reach,
                phase0=self._phase0,
                shear0=self._shear0,
                kinds=str_array(self._kinds),
                scalars=np.array(
                    [
                        self._phase,
                        self._lean,
                        self._lean_target,
                        float(self._ticks),
                        self._shear_phase,
                    ],
                    dtype=np.float64,
                ),
            )
            self._dirty = False

    def load(self) -> bool:
        """Restore the manifold's image. Returns False if none exists."""
        path = self.image_path
        if path is None or not os.path.exists(path):
            return False
        data = load_npz(path)
        version = int(data["version"])
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported manifold image version {version}")
        points = np.asarray(data["points"], dtype=np.float64)
        n = points.shape[0]
        kinds = [str(k) for k in data["kinds"]]
        if (
            points.ndim != 2
            or points.shape[1] != 4
            or len(kinds) != n
            or len(data["salience"]) != n
            or len(data["strand"]) != n
            or len(data["born_ms"]) != n
            or len(data["reach"]) != n
            or len(data["phase0"]) != n
            or len(data["shear0"]) != n
        ):
            raise ValueError("corrupt manifold image: shape mismatch")
        norms = np.linalg.norm(points, axis=1)
        if n and not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("corrupt manifold image: point off S3")
        scalars = np.asarray(data["scalars"], dtype=np.float64)
        if scalars.shape != (5,) or not np.isfinite(scalars).all():
            raise ValueError("corrupt manifold image: bad scalars")
        with self._lock:
            self._points = points
            self._salience = np.asarray(data["salience"], dtype=np.float64)
            self._strand = np.asarray(data["strand"], dtype=np.int8)
            self._born_ms = np.asarray(data["born_ms"], dtype=np.int64)
            self._reach = np.asarray(data["reach"], dtype=np.float64)
            self._phase0 = np.asarray(data["phase0"], dtype=np.float64)
            self._shear0 = np.asarray(data["shear0"], dtype=np.float64)
            self._kinds = kinds
            self._phase = float(scalars[0]) % g.TWO_PI
            self._lean = float(np.clip(scalars[1], 0.0, 1.0))
            self._lean_target = float(np.clip(scalars[2], 0.0, 1.0))
            self._ticks = int(scalars[3])
            self._shear_phase = float(scalars[4]) % g.TWO_PI
            self._dirty = False
        return True
