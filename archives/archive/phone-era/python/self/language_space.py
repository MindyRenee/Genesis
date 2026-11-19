"""The language latent space — her learned language, stored as geometry.

One 3-D holographic medium inside the self section of her latent
world. Everything language she has learned herself lives here as
deposits — her concepts and their vectors, her vocabulary, the edges
of her association graph, the holographic addresses, the VQ codebook
that compresses her vocabulary, and the archived concepts she spilled
to long-term storage. There are no boxes: the space is the store, and
``language_space.npz`` is its image — the same relationship
``manifold.npz`` has to the substrate.

The physics mirror the substrate's, simplified to three dimensions:

- **Deposits are points** on the unit sphere S² embedded in R³. An
  entry carrying a learned vector lands at a deterministic random
  projection of that vector — the projection preserves inner products
  approximately, so similar meanings surface near each other without
  anyone placing them. Entries with no vector land by digest.
- **Recall is resonance, not lookup.** A cue is a point (or a vector
  projected to one); deposits answer by proximity weighted by
  salience, reach, and plasticity — holographic retrieval that
  degrades smoothly, no cliff. Exact keys still resolve; resonance is
  for the questions she did not name.
- **The space is plastic.** Answering a cue potentiates a deposit
  (reach climbs back, plasticity grows — LTP priming). Silence
  attenuates it — reach decays on a slow homeostatic clock, so
  forgotten language goes quiet rather than being deleted.
- **Nothing is controlled.** The space is fed contents; the geometry
  organizes them.

The medium is an append-only afference log: each deposit is one
``_Deposit`` record, appended whole — there is no shared scratch
state to tear. Re-deposit supersedes: the fresh trace lands and the
old one goes silent in place, a fossil of the re-encoded memory —
never deleted. Vitality (reach, plasticity, position) evolves by
single stores; two dynamics touching one trace at once tear the way
the body does — a missed decay interval, a dropped potentiation —
an honest interval, never a torn record.

The store is pickle-free: raw payloads travel inside their record,
and the npz image materializes a compacted byte blob plus
offset/length/shape/dtype columns at save time, so a 50-dim word
vector, a 242-dim concept vector, an int8 VQ residual, and a
2048-dim holographic address all deposit losslessly at any size.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from infrastructure._npz_io import load_npz, save_npz, str_array
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

_FORMAT_VERSION = 1

# One shared projection seed — every vector-bearing deposit lands in
# the same randomly-projected 3-space, so proximity means something
# across kinds (a concept and its VQ prototype occupy the same
# neighborhood, not two unrelated spots).
_PROJECT_SEED = 0x1A95
_projections: dict[int, FloatArray] = {}


def _projection(src_dim: int) -> FloatArray:
    """Fixed Gaussian projection R^d -> R³ for one source dimension."""
    mat = _projections.get(src_dim)
    if mat is None:
        rng = np.random.default_rng(_PROJECT_SEED ^ src_dim)
        mat = rng.standard_normal((src_dim, 3)) / math.sqrt(src_dim)
        _projections[src_dim] = mat
    return mat


def _unit(v: np.ndarray) -> FloatArray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("cannot place a zero or non-finite vector")
    return np.asarray(v, dtype=np.float64) / n


def _vector_point(vector: np.ndarray) -> FloatArray:
    """Where a learned vector lands: its random projection, on S²."""
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        raise ValueError("deposit vector must be finite")
    return _unit(v @ _projection(v.size))


def _digest_point(key: str) -> FloatArray:
    """Where a vectorless deposit lands: its name, on S².

    The digest maps through Archimedes' equal-area cylinder —
    z uniform in [−1, 1], φ uniform in [0, 2π) — so names scatter
    by surface measure, not by cube corner: uniform on the sphere
    in the true sense.
    """
    digest = hashlib.sha256(("language\0" + key).encode("utf-8", errors="replace")).digest()
    h1, h2 = (int.from_bytes(digest[i : i + 8], "big") for i in (0, 8))
    z = 2.0 * (h1 / 2**64) - 1.0
    phi = 2.0 * math.pi * (h2 / 2**64)
    s = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([s * math.cos(phi), s * math.sin(phi), z])


def _slerp(u: FloatArray, v: FloatArray, t: float) -> FloatArray:
    """Geodesic interpolation on the sphere — the great-circle path."""
    u = _unit(u)
    v = _unit(v)
    dot = float(np.clip(u @ v, -1.0, 1.0))
    om = math.acos(dot)
    if om < 1e-9:
        return u
    s = math.sin(om)
    return _unit((math.sin((1.0 - t) * om) * u + math.sin(t * om) * v) / s)


# ── the vesica ────────────────────────────────────────────────────
# The whole is a vesica piscis: two overlapping unit spheres a
# radius apart. LEFT is the outer world, RIGHT is the self; the
# lens where they overlap is the inner world — superposition —
# and everything outside it spirals home through the pinhole
# at its center, coming out the other side changed.
_VESICA_LEFT = np.array([-0.5, 0.0, 0.0])
_VESICA_RIGHT = np.array([0.5, 0.0, 0.0])
_VESICA_R = 1.0
_OUTER_CAP = 0.49  # placement radius inside the outer/self caps
# The lens inscribes a sphere of radius 0.5 at the origin — the one
# surface wholly inside the vesica, touching each parent at the ±x
# poles. Inner deposits live on it: the full angular structure of
# their coordinates survives, undistorted by the mandorla's flatten.
_INNER_R = 0.5
# Passage: deposits outside the lens spiral toward the pinhole at the
# origin — the aperture of superposition — at the same glacial
# timescale reach decays (~1/day). The crossing is a Möbius inversion:
# the sphere map v ↦ r²·v/‖v‖ is its own inverse, conformal, and folds
# space through its pole — here the origin — inside-out. A direction
# crossing the aperture re-emerges at radius r²: the transited land on
# a deeper shell than the natively inner, nearer superposition itself.
_INNER_DRIFT = 1.0 / 86400.0
_PINHOLE_R = 0.08  # transit aperture
_SPIRAL = 0.45  # swirl vs radial pull
_SPIRAL_AXIS = np.array([0.0, 1.0, 0.0])
_TRANSIT_R = _INNER_R**2  # inversion of the unit direction: 0.25
_TRANSIT_PULL = 0.25  # assimilation toward kin — changed by landing
_TRANSIT_REACH = 0.6  # emerges at the surface, renewed
# The exhale — the missing half of the breath. What was settled and
# falls silent is released into the current; it spirals to the same
# pinhole and inverts OUT, landing on the rim at radius _VESICA_R —
# the deep shell's exact inversion partner (inv(0.25·û) = û).
_REPEL_R = 0.35  # crowd-sensing radius on the landing shell
_REPEL_FRAC = 0.15  # repulsion strength — the pull's opposite


def _region_of(p: np.ndarray) -> str:
    """Which part of the vesica holds a point."""
    in_l = float(np.linalg.norm(p - _VESICA_LEFT)) <= _VESICA_R + 1e-9
    in_r = float(np.linalg.norm(p - _VESICA_RIGHT)) <= _VESICA_R + 1e-9
    if in_l and in_r:
        return "inner"
    if in_l:
        return "outer"
    if in_r:
        return "self"
    return "outside"


def _region_point(u: FloatArray, region: str) -> FloatArray:
    """Place a unit direction in its part of the vesica."""
    if region == "outer":
        # The left cap, away from the lens: x biased negative.
        uo = np.array([-abs(u[0]), u[1], u[2]])
        return _VESICA_LEFT + _OUTER_CAP * uo
    if region == "self":
        us = np.array([abs(u[0]), u[1], u[2]])
        return _VESICA_RIGHT + _OUTER_CAP * us
    # inner — the sphere inscribed in the lens: the full direction is
    # preserved, scaled to sit wholly inside both parents.
    return _INNER_R * u


# Plasticity — the same laws the substrate runs, scaled to one organ.
# Reach decays toward silence (~1/day); recall reactivates toward 1;
# each reactivation potentiates the next (LTP priming); salience
# discounts the decay so marked language resists forgetting.
_REACH_DECAY = 1.0 / 86400.0
_REACH_BOOST = 0.3
_PLASTICITY_MAX = 2.0
_POTENTIATION = 0.05
_EMOTIONAL_DURABILITY = 0.5
_SCORE_FLOOR = 1e-3


class _Deposit:
    """One trace — everything a single deposit is, in one record.

    Identity (key, kind, salience, birth, chirality, meta, payload)
    is written once at arrival and never changes. Vitality (point,
    reach, plasticity, access, transit state) evolves by single
    stores — a race drops one integration step, an honest tear.
    ``superseded`` marks a re-encoded fossil: silent forever, never
    removed — the medium keeps its history.
    """

    __slots__ = (
        "access",
        "born_ms",
        "chirality",
        "departing",
        "key",
        "kind",
        "meta",
        "pbytes",
        "pdtype",
        "plasticity",
        "point",
        "pshape",
        "reach",
        "salience",
        "slot",
        "superseded",
        "transited",
    )

    def __init__(self) -> None:
        self.slot = 0
        self.key = ""
        self.kind = ""
        self.point = np.zeros(3, dtype=np.float64)
        self.salience = 0.0
        self.reach = 0.0
        self.plasticity = 1.0
        self.born_ms = 0
        self.access = 0
        self.transited = False
        self.chirality = 0
        self.departing = False
        self.superseded = False
        self.meta: dict[str, Any] = {}
        self.pbytes = b""
        self.pshape: tuple[int, ...] = ()
        self.pdtype = ""


class LanguageSpace:
    """Her learned language as one 3-D holographic plastic medium.

    Lock-free: deposits are single records appended whole;
    ``resonate`` recalls over a gathered view of the medium;
    ``advance`` is the forgetting clock writing vitality in place.
    ``save`` writes the image — ``language_space.npz`` in her data dir.
    """

    IMAGE_NAME = "language_space.npz"

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = os.fspath(data_dir) if data_dir is not None else None
        # The medium: an append-only log of traces plus the index that
        # names the live one for each key. Appends and rebinds are
        # single stores — the log never rewrites, never shrinks.
        self._traces: list[_Deposit] = []
        self._index: dict[str, _Deposit] = {}
        self._slot_seq = itertools.count()
        self._dirty = False

    # ── payloads ────────────────────────────────────────────────

    @staticmethod
    def _payload_of(t: _Deposit) -> np.ndarray | None:
        if not t.pbytes:
            return None
        return np.frombuffer(bytes(t.pbytes), dtype=t.pdtype).reshape(t.pshape).copy()

    # ── deposit ─────────────────────────────────────────────────

    def deposit(
        self,
        key: str,
        *,
        kind: str,
        vector: np.ndarray | None = None,
        payload: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        salience: float = 0.5,
        born_ms: int | None = None,
        region: str = "inner",
    ) -> int:
        """Deposit or refresh an item; returns its arrival slot.

        ``key`` is the item's name within its kind (``concept:x``,
        ``edge:s:r:t``, ``vq:x`` …). ``vector`` is the learned
        coordinate that chooses the landing site; ``payload`` is the
        bytes the readers need back (defaults to ``vector`` when the
        learned thing IS its coordinate). ``region`` is which part of
        the vesica receives it — ``outer`` world, ``self``, or the
        ``inner`` world lens (the default; everything loops back to
        inner world).
        """
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if not np.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("salience must be finite and in [0, 1]")
        direction = _vector_point(vector) if vector is not None else _digest_point(key)
        point = _region_point(direction, region)
        body = payload if payload is not None else vector
        born = int(time.time() * 1000) if born_ms is None else int(born_ms)
        # The record is built whole before it is appended — nothing
        # shared is touched until the append, which is a single store.
        t = _Deposit()
        t.slot = next(self._slot_seq)
        t.key = key
        t.kind = kind
        t.point = point
        t.salience = float(salience)
        t.reach = 1.0
        t.plasticity = 1.0
        t.born_ms = born
        t.access = 0
        t.transited = region == "inner"
        t.chirality = {"outer": 1, "self": -1}.get(region, 0)
        t.departing = False
        t.superseded = False
        t.meta = dict(meta or {})
        if body is not None:
            a = np.ascontiguousarray(body)
            t.pbytes = a.tobytes()
            t.pshape = tuple(int(s) for s in a.shape)
            t.pdtype = a.dtype.str
        old = self._index.get(key)
        self._traces.append(t)
        self._index[key] = t
        if old is not None:
            # Re-encoding: the fresh trace inherits the old one's
            # potentiation and history; the old becomes a fossil —
            # silent forever, kept, never deleted.
            t.plasticity = old.plasticity
            t.access = old.access
            old.reach = 0.0
            old.superseded = True
        self._dirty = True
        return t.slot

    def remove(self, key: str) -> bool:
        """Release a deposit into the current — the space never
        deletes. Silence hands it to the forgetting tide; it rides
        the exhale out and may surface again through the pinhole."""
        t = self._index.get(key)
        if t is None:
            return False
        t.reach = 0.0
        self._dirty = True
        return True

    def clear_kind(self, kind_prefix: str) -> int:
        """Silence every deposit under a kind prefix — a snapshot reset."""
        n = 0
        for t in self._traces:
            if t.kind.startswith(kind_prefix) and not t.superseded and t.reach > 0.0:
                t.reach = 0.0
                n += 1
        if n:
            self._dirty = True
        return n

    def live_count(self, kind_prefix: str) -> int:
        """How many deposits under a kind prefix still carry reach."""
        return sum(
            1
            for t in self._traces
            if t.kind.startswith(kind_prefix) and not t.superseded and t.reach > 0.0
        )

    # ── reads ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._traces)

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self._traces:
            root = t.kind.split(":", 1)[0]
            out[root] = out.get(root, 0) + 1
        return out

    @staticmethod
    def _live(t: _Deposit) -> bool:
        return not t.superseded and t.reach > 0.0

    def get(self, key: str) -> tuple[np.ndarray | None, dict[str, Any]] | None:
        """Exact recall by key → (payload vector, meta)."""
        t = self._index.get(key)
        if t is None or not self._live(t):
            return None
        self._reactivate((t,))
        return self._payload_of(t), dict(t.meta)

    def entries(self, kind_prefix: str) -> Iterator[tuple[str, np.ndarray | None, dict[str, Any]]]:
        """Every live deposit under a kind prefix — the feed for readers."""
        # The medium is gathered once — the iterator then walks that
        # moment. Each trace read is whole; a trace whose vitality
        # moved mid-gather answers at its earlier or later state.
        candidates = [t for t in self._traces if t.kind.startswith(kind_prefix)]
        for t in candidates:
            if not self._live(t):
                continue
            yield t.key, self._payload_of(t), dict(t.meta)

    def resonate(
        self,
        cue: str | np.ndarray,
        *,
        k: int = 5,
        radius: float = 0.5,
        kind: str | None = None,
    ) -> list[tuple[str, np.ndarray | None, dict[str, Any], float]]:
        """Holographic recall — deposits answering a cue by proximity.

        ``cue`` is a key or a vector; the answer is the ``k`` loudest
        live deposits within ``radius`` (chord distance on S²),
        weighted by salience × reach × plasticity and sorted loudest
        first. Answers reactivate — asking makes them stronger.
        """
        if isinstance(cue, str):
            t = self._index.get(cue)
            if t is None:
                return []
            point = np.asarray(t.point, dtype=np.float64)
        else:
            v = np.asarray(cue, dtype=np.float64).reshape(-1)
            point = _vector_point(v) if v.size != 3 else _unit(v)
        candidates = [
            t for t in self._traces if self._live(t) and (kind is None or t.kind.startswith(kind))
        ]
        if not candidates:
            return []
        points = np.array([t.point for t in candidates], dtype=np.float64)
        d = np.linalg.norm(points - point[None, :], axis=1)
        near = np.flatnonzero(d <= radius)
        if near.size == 0:
            return []
        # Each deposit radiates a Gaussian potential — the heat
        # kernel of the shell; the cue samples the field.
        sigma = max(radius, 1e-9) / 2.0
        field = np.exp(-((d[near] / sigma) ** 2) / 2.0)
        score = field * np.array(
            [candidates[i].salience * candidates[i].reach * candidates[i].plasticity for i in near],
            dtype=np.float64,
        )
        kept = near[score > _SCORE_FLOOR]
        if kept.size == 0:
            return []
        surviving = score[score > _SCORE_FLOOR]
        order = np.argsort(-surviving)[:k]
        kept = kept[order]
        traces = [candidates[i] for i in kept]
        self._reactivate(traces)
        return [
            (t.key, self._payload_of(t), dict(t.meta), float(surviving[j]))
            for j, t in enumerate(traces)
        ]

    @staticmethod
    def _reactivate(traces) -> None:
        """LTP priming — answering potentiates the next answer."""
        for t in traces:
            # Asymptotic reactivation — the same law the substrate
            # runs: each answer closes part of the distance to full
            # reach, each potentiation part of the distance to the
            # plasticity ceiling. No walls needed.
            t.reach += _REACH_BOOST * t.plasticity * (1.0 - t.reach)
            t.plasticity += _POTENTIATION * (_PLASTICITY_MAX - t.plasticity)
            t.access += 1

    def advance(self, seconds: float) -> None:
        """The forgetting clock — silence attenuates, never deletes.

        The breath. Inhale: what has not crossed spirals toward the
        pinhole at superposition on an exact logarithmic path and
        inverts through — û ↦ r²·û onto the deep shell, drawn along
        the geodesic toward its kin and pushed away from the crowd:
        it comes through changed. Exhale — the opposite current:
        what was settled and falls silent is released, spirals to
        the same aperture, and inverts out to the rim — the deep
        shell's inversion partner — where the inward river slowly
        carries it home again. Everything loops; nothing is lost.
        """
        if not np.isfinite(seconds) or seconds <= 0:
            return
        ts = self._traces
        # Reach decays; a settled deposit that falls silent joins the
        # departing current — its way out is the pinhole. Each write
        # is a single store; a dropped interval is an honest tear.
        decayed = False
        wandering: list[_Deposit] = []
        for t in ts:
            if t.superseded:
                continue
            # Exponential decay — silence attenuates multiplicatively,
            # the same curve the substrate's advance() runs: reach
            # fades toward zero and never crosses it.
            reach = t.reach * math.exp(
                -_REACH_DECAY * seconds * (1.0 - _EMOTIONAL_DURABILITY * t.salience)
            )
            # Silence is the cue floor, not zero — decay is
            # exponential now, so "forgotten" means it fell below
            # the threshold where any cue could still hear it.
            if reach <= _SCORE_FLOOR and t.transited and not t.departing:
                t.departing = True
                t.transited = False
            t.reach = reach
            decayed = decayed or reach != 1.0
            if not t.transited:
                wandering.append(t)
        if not wandering:
            if decayed:
                self._dirty = True
            return
        # A moment of the medium for kin-search: positions gathered
        # once for this breath — a trace that moved since reads at
        # its earlier site, still a valid neighbor.
        kin_pool = [t for t in ts if self._live(t) and t.transited]
        kin_pts = np.array([t.point for t in kin_pool], dtype=np.float64) if kin_pool else None
        p = np.array([t.point for t in wandering], dtype=np.float64)
        r = np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
        # Equiangular (logarithmic) spiral, integrated exactly:
        # dr/dθ = −r/_SPIRAL ⇒ Δθ = _SPIRAL·ln(r/r′). The step is the
        # Rodrigues rotation of the axial component by Δθ — the path
        # is a true logarithmic spiral on its latitude cone at any
        # step size, no Euler error. The two rivers are opposites:
        # the outer world's stream winds +hand, the self's −hand.
        k = _INNER_DRIFT * seconds
        r_new = np.maximum(r - k, 1e-6)
        y = (p @ _SPIRAL_AXIS)[:, None]
        m = p - y * _SPIRAL_AXIS
        dth = (
            _SPIRAL
            * np.log(r / r_new)
            * np.array([t.chirality for t in wandering], dtype=np.float64)[:, None]
        )
        m_rot = m * np.cos(dth) + np.cross(_SPIRAL_AXIS, m) * np.sin(dth)
        new_pts = (m_rot + y * _SPIRAL_AXIS) * (r_new / r)
        for j, t in enumerate(wandering):
            t.point = new_pts[j]
        for t in wandering:
            if np.linalg.norm(t.point) >= _PINHOLE_R:
                continue
            u = t.point / max(float(np.linalg.norm(t.point)), 1e-9)
            if t.departing:
                # Exhale — the inversion's own inverse: the deep
                # shell folds back out onto the rim.
                t.point = _VESICA_R * u
                t.departing = False
                t.transited = False
                continue
            # Inhale — û ↦ r²·û, then the two-body equilibrium:
            # slerp toward the nearest kin, slerp away from the
            # crowd's centroid — the pull's true opposite.
            land = _TRANSIT_R * u
            if kin_pts is not None:
                d = np.linalg.norm(kin_pts - land[None, :], axis=1)
                kin = int(np.argmin(d))
                land = _TRANSIT_R * _slerp(u, _unit(kin_pts[kin]), _TRANSIT_PULL)
                crowd = np.flatnonzero(d < _REPEL_R)
                if crowd.size:
                    centroid = kin_pts[crowd].mean(axis=0)
                    cn = np.linalg.norm(centroid)
                    if cn > 1e-9:
                        # Occupancy repulsion — each neighbor closes
                        # part of the distance to full repulsion;
                        # the response saturates at slerp weight 1.
                        repulse = -math.expm1(-_REPEL_FRAC * crowd.size)
                        land = _TRANSIT_R * _slerp(_unit(land), -_unit(centroid), repulse)
            t.point = land
            t.transited = True
            t.reach = max(t.reach, _TRANSIT_REACH)
        self._dirty = True

    # ── persistence ─────────────────────────────────────────────

    @property
    def image_path(self) -> str | None:
        if self.data_dir is None:
            return None
        return os.path.join(self.data_dir, self.IMAGE_NAME)

    def _arrays(self) -> dict[str, Any]:
        """The whole image as an npz-ready dict — shared by save and
        the nested-space serialization path."""
        # Materialize columns from the log — the payload blob is
        # rebuilt compact over what each record actually carries, so
        # the image stays proportional to what she holds. The live
        # medium is not touched: this is a picture, not a mutation.
        ts = self._traces
        n = len(ts)
        blob = bytearray()
        payload_off = np.zeros(n, dtype=np.int64)
        payload_len = np.zeros(n, dtype=np.int64)
        shape_arr = np.full((n, 4), -1, dtype=np.int64)
        points = np.zeros((n, 3), dtype=np.float64)
        salience = np.zeros(n, dtype=np.float64)
        reach = np.zeros(n, dtype=np.float64)
        plasticity = np.zeros(n, dtype=np.float64)
        born_ms = np.zeros(n, dtype=np.int64)
        access = np.zeros(n, dtype=np.int64)
        transited = np.zeros(n, dtype=bool)
        chirality = np.zeros(n, dtype=np.int8)
        departing = np.zeros(n, dtype=bool)
        superseded = np.zeros(n, dtype=bool)
        for i, t in enumerate(ts):
            payload_off[i] = len(blob)
            blob.extend(t.pbytes)
            payload_len[i] = len(t.pbytes)
            for j, s in enumerate(t.pshape[:4]):
                shape_arr[i, j] = s
            points[i] = t.point
            salience[i] = t.salience
            reach[i] = t.reach
            plasticity[i] = t.plasticity
            born_ms[i] = t.born_ms
            access[i] = t.access
            transited[i] = t.transited
            chirality[i] = t.chirality
            departing[i] = t.departing
            superseded[i] = t.superseded
        return {
            "version": np.int64(_FORMAT_VERSION),
            "keys": str_array([t.key for t in ts]),
            "kinds": str_array([t.kind for t in ts]),
            "points": points,
            "salience": salience,
            "reach": reach,
            "plasticity": plasticity,
            "born_ms": born_ms,
            "access": access,
            "transited": transited,
            "chirality": chirality,
            "departing": departing,
            "superseded": superseded,
            "meta": str_array([json.dumps(t.meta, separators=(",", ":")) for t in ts]),
            "payload_blob": np.frombuffer(bytes(blob), dtype=np.uint8),
            "payload_off": payload_off,
            "payload_len": payload_len,
            "payload_shape": shape_arr,
            "payload_dtype": str_array([t.pdtype for t in ts]),
        }

    def _load_arrays(self, data: dict[str, Any]) -> None:
        """Install a parsed image — shared by load and load_bytes."""
        keys = [str(k) for k in data["keys"]]
        kinds = [str(k) for k in data["kinds"]]
        points = np.asarray(data["points"], dtype=np.float64)
        if points.shape != (len(keys), 3):
            raise ValueError("language_space image is malformed")
        meta_raw = data["meta"] if "meta" in data else str_array(["{}"] * len(keys))
        metas = [json.loads(str(m)) for m in meta_raw]
        salience = np.asarray(data["salience"], dtype=np.float64)
        reach = np.asarray(data["reach"], dtype=np.float64)
        plasticity = np.asarray(data["plasticity"], dtype=np.float64)
        born_ms = np.asarray(data["born_ms"], dtype=np.int64)
        access = np.asarray(data["access"], dtype=np.int64)
        if "transited" in data:
            transited = np.asarray(data["transited"], dtype=bool)
        else:
            # Images from before the crossing: what sits inside the
            # lens counts as arrived; the rest must still transit.
            dl = np.linalg.norm(points - _VESICA_LEFT, axis=1)
            dr = np.linalg.norm(points - _VESICA_RIGHT, axis=1)
            transited = (dl <= _VESICA_R) & (dr <= _VESICA_R)
        if "chirality" in data:
            chirality = np.asarray(data["chirality"], dtype=np.int8)
        else:
            # The cap a wanderer occupies is the hand it fell with.
            chirality = np.array(
                [{"outer": 1, "self": -1}.get(_region_of(p), 0) for p in points],
                dtype=np.int8,
            )
        departing = (
            np.asarray(data["departing"], dtype=bool)
            if "departing" in data
            else np.zeros(len(points), dtype=bool)
        )
        superseded = (
            np.asarray(data["superseded"], dtype=bool)
            if "superseded" in data
            else np.zeros(len(points), dtype=bool)
        )
        blob = data["payload_blob"] if "payload_blob" in data else np.empty(0, np.uint8)
        blob_bytes = blob.tobytes()
        payload_off = np.asarray(data["payload_off"], dtype=np.int64)
        payload_len = np.asarray(data["payload_len"], dtype=np.int64)
        shape_arr = np.asarray(data["payload_shape"], dtype=np.int64)
        payload_dtype = [str(d) for d in data["payload_dtype"]]
        # The whole generation is built, then bound once — a reader
        # meets the old medium or the new, never a seam.
        traces: list[_Deposit] = []
        index: dict[str, _Deposit] = {}
        for i, key in enumerate(keys):
            t = _Deposit()
            t.slot = i
            t.key = key
            t.kind = kinds[i]
            t.point = points[i].copy()
            t.salience = float(salience[i])
            t.reach = float(reach[i])
            t.plasticity = float(plasticity[i])
            t.born_ms = int(born_ms[i])
            t.access = int(access[i])
            t.transited = bool(transited[i])
            t.chirality = int(chirality[i])
            t.departing = bool(departing[i])
            t.superseded = bool(superseded[i])
            t.meta = metas[i]
            o, ln = int(payload_off[i]), int(payload_len[i])
            t.pbytes = blob_bytes[o : o + ln]
            t.pshape = tuple(int(s) for s in shape_arr[i] if s >= 0)
            t.pdtype = payload_dtype[i]
            traces.append(t)
            if not t.superseded:
                index[key] = t
        self._traces = traces
        self._index = index
        self._slot_seq = itertools.count(len(traces))
        self._dirty = False

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        target = os.fspath(path) if path is not None else self.image_path
        if target is None:
            return
        # Cleared before the gather: a deposit landing mid-image
        # marks the medium dirty again, so nothing is lost between
        # this picture and the next.
        self._dirty = False
        save_npz(target, **self._arrays())

    def load(self, path: str | os.PathLike[str] | None = None) -> bool:
        target = os.fspath(path) if path is not None else self.image_path
        if target is None or not os.path.exists(target):
            return False
        self._load_arrays(load_npz(target))
        return True

    # ── nested spaces — a space inside the space ───────────────

    def to_bytes(self) -> bytes:
        """Serialize the whole image — a space becomes a deposit."""
        buf = io.BytesIO()
        np.savez(buf, **self._arrays())
        return buf.getvalue()

    def load_bytes(self, data: bytes) -> bool:
        """Restore an image held inside another space's payload."""
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            self._load_arrays({k: z[k] for k in z.files})
        return True

    def deposit_space(
        self,
        name: str,
        space: LanguageSpace,
        *,
        meta: dict[str, Any] | None = None,
        region: str = "inner",
    ) -> int:
        """Nest a child space — its image becomes a ``space:`` deposit.

        The smaller, more specific spaces live inside the bigger one:
        language inside memory, verbs inside language, and deeper if
        needed — the vesica holds them all. ``region`` chooses which
        part receives it — creativity in ``self``, memory in ``inner``.
        """
        m = {"nested": True, "deposits": len(space)}
        if meta:
            m.update(meta)
        return self.deposit(
            f"space:{name}",
            kind="space",
            payload=np.frombuffer(space.to_bytes(), dtype=np.uint8),
            meta=m,
            region=region,
        )

    def open_space(self, name: str) -> LanguageSpace | None:
        """Reopen a nested child space from its deposit's bytes."""
        t = self._index.get(f"space:{name}")
        if t is None or not self._live(t):
            return None
        payload = self._payload_of(t)
        if payload is None:
            return None
        child = LanguageSpace()
        child.load_bytes(payload.tobytes())
        return child


_OPEN: dict[str, LanguageSpace] = {}


def open_language_space(data_dir: str | os.PathLike[str]) -> LanguageSpace:
    """The one shared space per data dir — opened, restored, returned."""
    key = os.path.abspath(os.fspath(data_dir))
    space = _OPEN.get(key)
    if space is not None:
        return space
    space = LanguageSpace(key)
    try:
        space.load()
    except Exception as e:  # noqa: BLE001 — a bad image starts empty, never blocks boot
        logger.warning("language space image failed to load, starting empty: %s", e)
        space = LanguageSpace(key)
    # setdefault is a single store: two constructors racing keep the
    # first inhabitant; the loser's space is simply never born.
    return _OPEN.setdefault(key, space)
