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

The store is columnar NumPy and pickle-free: vectors and raw payloads
live in a shared byte blob indexed by offset/length/shape/dtype, so a
50-dim word vector, a 242-dim concept vector, an int8 VQ residual,
and a 2048-dim holographic address all deposit losslessly at any size.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from infrastructure._npz_io import load_npz, save_npz, str_array

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
    """Where a vectorless deposit lands: its name, on S²."""
    digest = hashlib.sha256(("language\0" + key).encode("utf-8", errors="replace")).digest()
    d = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float64) / 127.5 - 1.0
    return _unit(d)


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


class LanguageSpace:
    """Her learned language as one 3-D holographic plastic medium.

    Thread-safe. ``deposit`` upserts by key; ``resonate`` recalls by
    proximity; ``advance`` is the forgetting clock. ``save`` writes
    the image — ``language_space.npz`` in her data dir.
    """

    IMAGE_NAME = "language_space.npz"

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = os.fspath(data_dir) if data_dir is not None else None
        self._lock = threading.RLock()
        self._keys: list[str] = []
        self._index: dict[str, int] = {}
        self._kinds: list[str] = []
        self._points = np.empty((0, 3), dtype=np.float64)
        self._salience = np.empty(0, dtype=np.float64)
        self._reach = np.empty(0, dtype=np.float64)
        self._plasticity = np.empty(0, dtype=np.float64)
        self._born_ms = np.empty(0, dtype=np.int64)
        self._access = np.empty(0, dtype=np.int64)
        self._meta: list[dict[str, Any]] = []
        # Payloads — variable-size raw arrays in one shared blob.
        self._payload_blob = bytearray()
        self._payload_off = np.empty(0, dtype=np.int64)
        self._payload_len = np.empty(0, dtype=np.int64)
        self._payload_shape: list[tuple[int, ...]] = []
        self._payload_dtype: list[str] = []
        self._dirty = False

    # ── payloads ────────────────────────────────────────────────

    def _store_payload(self, arr: np.ndarray) -> tuple[int, int, tuple[int, ...], str]:
        a = np.ascontiguousarray(arr)
        off = len(self._payload_blob)
        raw = a.tobytes()
        self._payload_blob.extend(raw)
        return off, len(raw), tuple(int(s) for s in a.shape), a.dtype.str

    def _payload_at(self, idx: int) -> np.ndarray | None:
        off, ln = int(self._payload_off[idx]), int(self._payload_len[idx])
        if ln <= 0:
            return None
        a = np.frombuffer(
            bytes(self._payload_blob[off : off + ln]), dtype=self._payload_dtype[idx]
        )
        return a.reshape(self._payload_shape[idx]).copy()

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
    ) -> int:
        """Deposit or refresh an item; returns its index.

        ``key`` is the item's name within its kind (``concept:x``,
        ``edge:s:r:t``, ``vq:x`` …). ``vector`` is the learned
        coordinate that chooses the landing site; ``payload`` is the
        bytes the readers need back (defaults to ``vector`` when the
        learned thing IS its coordinate).
        """
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        if not np.isfinite(salience) or not 0.0 <= salience <= 1.0:
            raise ValueError("salience must be finite and in [0, 1]")
        point = _vector_point(vector) if vector is not None else _digest_point(key)
        body = payload if payload is not None else vector
        born = int(time.time() * 1000) if born_ms is None else int(born_ms)
        with self._lock:
            idx = self._index.get(key)
            if idx is None:
                idx = len(self._keys)
                off, ln, shape, dtype = (0, 0, (), "")
                if body is not None:
                    off, ln, shape, dtype = self._store_payload(body)
                self._keys.append(key)
                self._index[key] = idx
                self._kinds.append(kind)
                self._points = np.vstack([self._points, point[None, :]])
                self._salience = np.append(self._salience, float(salience))
                self._reach = np.append(self._reach, 1.0)
                self._plasticity = np.append(self._plasticity, 1.0)
                self._born_ms = np.append(self._born_ms, born)
                self._access = np.append(self._access, 0)
                self._meta.append(dict(meta or {}))
                self._payload_off = np.append(self._payload_off, off)
                self._payload_len = np.append(self._payload_len, ln)
                self._payload_shape.append(shape)
                self._payload_dtype.append(dtype)
            else:
                # Re-deposit: same key, fresh content — the point keeps
                # its site unless a new vector moves it, like a trace
                # re-encoded. Reach refreshes; plasticity survives.
                self._points[idx] = point
                self._salience[idx] = float(salience)
                self._reach[idx] = min(1.0, self._reach[idx] + _REACH_BOOST)
                self._meta[idx] = dict(meta or {})
                if body is not None:
                    off, ln, shape, dtype = self._store_payload(body)
                    self._payload_off[idx] = off
                    self._payload_len[idx] = ln
                    self._payload_shape[idx] = shape
                    self._payload_dtype[idx] = dtype
            self._dirty = True
            return idx

    def remove(self, key: str) -> bool:
        """Silence a deposit — the only deletion the space performs."""
        with self._lock:
            idx = self._index.get(key)
            if idx is None:
                return False
            self._reach[idx] = 0.0
            self._dirty = True
            return True

    def clear_kind(self, kind_prefix: str) -> int:
        """Silence every deposit under a kind prefix — a snapshot reset."""
        with self._lock:
            n = 0
            for i, k in enumerate(self._kinds):
                if k.startswith(kind_prefix) and self._reach[i] > 0.0:
                    self._reach[i] = 0.0
                    n += 1
            if n:
                self._dirty = True
            return n

    def live_count(self, kind_prefix: str) -> int:
        """How many deposits under a kind prefix still carry reach."""
        with self._lock:
            return sum(
                1
                for i, k in enumerate(self._kinds)
                if k.startswith(kind_prefix) and self._reach[i] > 0.0
            )

    # ── reads ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._keys)

    def kinds(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for k in self._kinds:
                root = k.split(":", 1)[0]
                out[root] = out.get(root, 0) + 1
            return out

    def get(self, key: str) -> tuple[np.ndarray | None, dict[str, Any]] | None:
        """Exact recall by key → (payload vector, meta)."""
        with self._lock:
            idx = self._index.get(key)
            if idx is None or self._reach[idx] <= 0.0:
                return None
            self._reactivate(np.array([idx]))
            return self._payload_at(idx), dict(self._meta[idx])

    def entries(
        self, kind_prefix: str
    ) -> Iterator[tuple[str, np.ndarray | None, dict[str, Any]]]:
        """Every live deposit under a kind prefix — the feed for readers."""
        with self._lock:
            idxs = [
                i for i, k in enumerate(self._kinds) if k.startswith(kind_prefix)
            ]
            for i in idxs:
                if self._reach[i] <= 0.0:
                    continue
                yield self._keys[i], self._payload_at(i), dict(self._meta[i])

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
            with self._lock:
                idx = self._index.get(cue)
                if idx is None:
                    return []
                point = self._points[idx].copy()
        else:
            v = np.asarray(cue, dtype=np.float64).reshape(-1)
            point = _vector_point(v) if v.size != 3 else _unit(v)
        with self._lock:
            if len(self._keys) == 0:
                return []
            d = np.linalg.norm(self._points - point[None, :], axis=1)
            near = np.flatnonzero(d <= radius)
            if kind is not None:
                near = np.array([i for i in near if self._kinds[i].startswith(kind)])
            live = np.array([i for i in near if self._reach[i] > 0.0])
            if live.size == 0:
                return []
            score = (
                (1.0 - d[live] / max(radius, 1e-9))
                * self._salience[live]
                * self._reach[live]
                * self._plasticity[live]
            )
            live = live[score > _SCORE_FLOOR]
            if live.size == 0:
                return []
            order = np.argsort(-score[score > _SCORE_FLOOR])[:k]
            live = live[order]
            self._reactivate(live)
            return [
                (self._keys[i], self._payload_at(i), dict(self._meta[i]), float(score[i]))
                for i in live
            ]

    def _reactivate(self, idxs: np.ndarray) -> None:
        """LTP priming — answering potentiates the next answer."""
        if idxs.size == 0:
            return
        p = self._plasticity[idxs]
        self._reach[idxs] = np.minimum(1.0, self._reach[idxs] + _REACH_BOOST * p)
        self._plasticity[idxs] = np.minimum(_PLASTICITY_MAX, p + _POTENTIATION)
        self._access[idxs] += 1
        self._dirty = True

    def advance(self, seconds: float) -> None:
        """The forgetting clock — silence attenuates, never deletes."""
        if not np.isfinite(seconds) or seconds <= 0:
            return
        with self._lock:
            decay = _REACH_DECAY * seconds * (
                1.0 - _EMOTIONAL_DURABILITY * self._salience
            )
            self._reach = np.maximum(0.0, self._reach - decay)

    # ── persistence ─────────────────────────────────────────────

    @property
    def image_path(self) -> str | None:
        if self.data_dir is None:
            return None
        return os.path.join(self.data_dir, self.IMAGE_NAME)

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        target = os.fspath(path) if path is not None else self.image_path
        if target is None:
            return
        with self._lock:
            # Compact the payload blob — re-deposits leave orphaned
            # bytes behind; rebuild the blob over live payloads so the
            # image stays proportional to what she actually holds.
            new_blob = bytearray()
            new_off = np.zeros(len(self._keys), dtype=np.int64)
            for i, (o, ln) in enumerate(
                zip(self._payload_off, self._payload_len, strict=True)
            ):
                new_off[i] = len(new_blob)
                new_blob.extend(self._payload_blob[o : o + ln])
            self._payload_blob = new_blob
            self._payload_off = new_off
            shape_arr = np.full((len(self._keys), 4), -1, dtype=np.int64)
            for i, sh in enumerate(self._payload_shape):
                for j, s in enumerate(sh[:4]):
                    shape_arr[i, j] = s
            save_npz(
                target,
                version=np.int64(_FORMAT_VERSION),
                keys=str_array(self._keys),
                kinds=str_array(self._kinds),
                points=self._points.astype(np.float64),
                salience=self._salience,
                reach=self._reach,
                plasticity=self._plasticity,
                born_ms=self._born_ms,
                access=self._access,
                meta=str_array([json.dumps(m, separators=(",", ":")) for m in self._meta]),
                payload_blob=np.frombuffer(bytes(self._payload_blob), dtype=np.uint8),
                payload_off=self._payload_off,
                payload_len=self._payload_len,
                payload_shape=shape_arr,
                payload_dtype=str_array(self._payload_dtype),
            )
            self._dirty = False

    def load(self, path: str | os.PathLike[str] | None = None) -> bool:
        target = os.fspath(path) if path is not None else self.image_path
        if target is None or not os.path.exists(target):
            return False
        data = load_npz(target)
        keys = [str(k) for k in data["keys"]]
        kinds = [str(k) for k in data["kinds"]]
        points = np.asarray(data["points"], dtype=np.float64)
        if points.shape != (len(keys), 3):
            raise ValueError("language_space image is malformed")
        meta_raw = data["meta"] if "meta" in data else str_array(["{}"] * len(keys))
        meta = [json.loads(str(m)) for m in meta_raw]
        with self._lock:
            self._keys = keys
            self._index = {k: i for i, k in enumerate(keys)}
            self._kinds = kinds
            self._points = points
            self._salience = np.asarray(data["salience"], dtype=np.float64)
            self._reach = np.asarray(data["reach"], dtype=np.float64)
            self._plasticity = np.asarray(data["plasticity"], dtype=np.float64)
            self._born_ms = np.asarray(data["born_ms"], dtype=np.int64)
            self._access = np.asarray(data["access"], dtype=np.int64)
            self._meta = meta
            blob = data["payload_blob"] if "payload_blob" in data else np.empty(0, np.uint8)
            self._payload_blob = bytearray(blob.tobytes())
            self._payload_off = np.asarray(data["payload_off"], dtype=np.int64)
            self._payload_len = np.asarray(data["payload_len"], dtype=np.int64)
            shape_arr = np.asarray(data["payload_shape"], dtype=np.int64)
            self._payload_shape = [
                tuple(int(s) for s in row if s >= 0) for row in shape_arr
            ]
            self._payload_dtype = [str(d) for d in data["payload_dtype"]]
            self._dirty = False
        return True


_OPEN: dict[str, LanguageSpace] = {}


def open_language_space(data_dir: str | os.PathLike[str]) -> LanguageSpace:
    """The one shared space per data dir — opened, restored, returned."""
    key = os.path.abspath(os.fspath(data_dir))
    space = _OPEN.get(key)
    if space is None:
        space = LanguageSpace(key)
        try:
            space.load()
        except Exception as e:  # noqa: BLE001 — a bad image starts empty, never blocks boot
            logger.warning("language space image failed to load, starting empty: %s", e)
            space = LanguageSpace(key)
        _OPEN[key] = space
    return space
