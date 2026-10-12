"""Multidimensional tensors shared with the subcognitive daemon.

The daemon hosts a named tensor slab (``src/daemon/tensor_slab.rs``)
reachable over four IPC commands — ``PUT_TENSOR``, ``GET_TENSOR``,
``LIST_TENSORS``, ``DROP_TENSOR`` — plus ``GET_MOTOR_TARGET``, which
decodes the published motor fields into a concrete action/grid target
on the daemon's canonical 64×64 coordinate grid
(``src/field/motor.rs``).

Wire layout (little-endian, C row-major payload)::

    tensor        := [u8 dtype] [u8 ndim] [ndim × u32 dims] [payload]
    put request   := [u8 name_len] [name utf8] tensor
    get request   := [u8 name_len] [name utf8]
    get response  := [u8 found] (tensor iff found == 1)
    list response := [u16 count] (per entry: [u8 name_len] [name]
                     [u8 dtype] [u8 ndim] [ndim × u32 dims])
    drop request  := [u8 name_len] [name utf8]
    motor target  := [u8 flags][u8 action_id][f32 a_strength]
                     [u16 x][u16 y][f32 cell_strength]

The constants and limits mirror the Rust side; ``protocol.py`` carries
the opcode ids.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

__all__ = [
    "DTYPE_STRUCT",
    "MAX_DIMS",
    "MAX_NAME_LEN",
    "MAX_TENSOR_BYTES",
    "DType",
    "MotorTarget",
    "Tensor",
    "TensorMeta",
    "pack_name",
    "unpack_list",
    "unpack_motor_target",
]


class DType(IntEnum):
    """Tensor element dtypes — codes mirror ``dtype::*`` in
    ``src/daemon/tensor_slab.rs`` and are part of the wire protocol."""

    U8 = 0
    I32 = 1
    F32 = 2
    F64 = 3


#: struct format per dtype — single-element, little-endian.
DTYPE_STRUCT: dict[DType, str] = {
    DType.U8: "B",
    DType.I32: "i",
    DType.F32: "f",
    DType.F64: "d",
}

#: Mirrors of the Rust slab limits — checked client-side too, so a
#: bad call fails fast instead of consuming a socket round-trip.
MAX_DIMS = 8
MAX_NAME_LEN = 128
MAX_TENSOR_BYTES = 4 << 20

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_F32 = struct.Struct("<f")


@dataclass(slots=True)
class Tensor:
    """A dense, C-order tensor: dtype, shape, and raw LE payload.

    ``payload`` holds ``dtype_width × prod(shape)`` bytes. Construct
    via :meth:`from_lists` / :meth:`from_array`, or directly from
    already-packed bytes.
    """

    dtype: DType
    shape: tuple[int, ...]
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_STRUCT:
            raise ValueError(f"unknown dtype {self.dtype!r}")
        if not self.shape or len(self.shape) > MAX_DIMS:
            raise ValueError(f"rank must be 1..{MAX_DIMS}, got {len(self.shape)}")
        expected = self.elements * self.itemsize
        if expected != len(self.payload):
            raise ValueError(
                f"payload {len(self.payload)}B != "
                f"{self.elements} elements × {self.itemsize}B"
            )
        if len(self.payload) > MAX_TENSOR_BYTES:
            raise ValueError("tensor payload exceeds MAX_TENSOR_BYTES")

    @property
    def elements(self) -> int:
        """Number of elements (``prod(shape)``; 0 when any extent is 0)."""
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return struct.calcsize("<" + DTYPE_STRUCT[self.dtype])

    # ── Construction ──────────────────────────────────────────

    @classmethod
    def from_lists(
        cls, data: Sequence[Any], dtype: DType | None = None
    ) -> Tensor:
        """Build a tensor from nested lists, inferring the shape.

        ``dtype`` defaults to F32 when any element is a float, I32
        otherwise; pass it explicitly to force a width (e.g. U8 for
        color-indexed grids).
        """
        shape = _infer_shape(data)
        flat: list[Any] = []
        _flatten(data, shape, 0, flat)
        if dtype is None:
            dtype = DType.F32 if any(isinstance(v, float) for v in flat) else DType.I32
        payload = struct.pack(
            "<" + DTYPE_STRUCT[dtype] * len(flat), *flat
        ) if flat else b""
        return cls(dtype=dtype, shape=shape, payload=payload)

    @classmethod
    def from_array(cls, arr: Any) -> Tensor:
        """Build a tensor from a numpy array (C-order cast)."""
        import numpy as np

        a = np.ascontiguousarray(arr)
        dt = {
            np.dtype("uint8"): DType.U8,
            np.dtype("int32"): DType.I32,
            np.dtype("float32"): DType.F32,
            np.dtype("float64"): DType.F64,
        }.get(a.dtype)
        if dt is None:
            a = a.astype(np.float32)
            dt = DType.F32
        return cls(dtype=dt, shape=tuple(a.shape) or (1,), payload=a.tobytes())

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: DType = DType.F32) -> Tensor:
        """A zeroed field of the given shape."""
        n = 1
        for d in shape:
            n *= d
        return cls(
            dtype=dtype,
            shape=tuple(shape),
            payload=b"\x00" * (n * struct.calcsize("<" + DTYPE_STRUCT[dtype])),
        )

    # ── Views ─────────────────────────────────────────────────

    def values(self) -> list[int | float]:
        """Flat element list."""
        if not self.payload:
            return []
        return list(
            struct.unpack(
                "<" + DTYPE_STRUCT[self.dtype] * self.elements, self.payload
            )
        )

    def to_lists(self) -> list[Any]:
        """Nested lists matching ``shape``."""
        return _nest(self.values(), self.shape)

    def to_array(self) -> Any:
        """numpy array view (copies into a fresh array)."""
        import numpy as np

        np_dtype = {
            DType.U8: np.uint8,
            DType.I32: np.int32,
            DType.F32: np.float32,
            DType.F64: np.float64,
        }[self.dtype]
        return np.frombuffer(self.payload, dtype=np_dtype).reshape(self.shape)

    # ── Wire codec ────────────────────────────────────────────

    def pack(self) -> bytes:
        """``[dtype][ndim][dims][payload]``."""
        out = bytearray([int(self.dtype), len(self.shape)])
        for d in self.shape:
            out += _U32.pack(d)
        out += self.payload
        return bytes(out)

    @classmethod
    def unpack(cls, buf: bytes | bytearray | memoryview) -> Tensor:
        """Parse ``[dtype][ndim][dims][payload]`` — the whole buffer is
        the tensor (no trailing bytes allowed)."""
        tensor, used = cls._unpack_prefix(buf)
        if used != len(buf):
            raise ValueError("trailing bytes after tensor")
        return tensor

    @classmethod
    def _unpack_prefix(
        cls, buf: bytes | bytearray | memoryview
    ) -> tuple[Tensor, int]:
        """Parse a tensor from the front of ``buf``; returns it plus the
        byte offset just past it."""
        if len(buf) < 2:
            raise ValueError("tensor header too short")
        try:
            dtype = DType(buf[0])
        except ValueError as e:
            raise ValueError(f"unknown dtype {buf[0]}") from e
        ndim = buf[1]
        if not 1 <= ndim <= MAX_DIMS:
            raise ValueError(f"bad rank {ndim}")
        header = 2 + ndim * 4
        if len(buf) < header:
            raise ValueError("tensor dims truncated")
        shape = tuple(
            _U32.unpack_from(buf, 2 + i * 4)[0] for i in range(ndim)
        )
        n = 1
        for d in shape:
            n *= d
        nbytes = n * struct.calcsize("<" + DTYPE_STRUCT[dtype])
        if len(buf) < header + nbytes:
            raise ValueError("tensor payload truncated")
        payload = bytes(buf[header : header + nbytes])
        return cls(dtype=dtype, shape=shape, payload=payload), header + nbytes


@dataclass(slots=True)
class TensorMeta:
    """One row of ``LIST_TENSORS`` — identity without payload."""

    name: str
    dtype: DType
    shape: tuple[int, ...]


@dataclass(slots=True)
class MotorTarget:
    """Decode of ``GET_MOTOR_TARGET``: what the published motor fields
    currently point at on the daemon's canonical 64×64 grid."""

    action_id: int | None = None
    action_strength: float = 0.0
    x: int | None = None
    y: int | None = None
    cell_strength: float = 0.0

    @property
    def has_action(self) -> bool:
        return self.action_id is not None

    @property
    def has_cell(self) -> bool:
        return self.x is not None


def pack_name(name: str) -> bytes:
    """``[u8 name_len][name utf8]`` — the request prefix shared by
    PUT/GET/DROP."""
    raw = name.encode("utf-8")
    if not raw or len(raw) > MAX_NAME_LEN:
        raise ValueError(f"tensor name must be 1..{MAX_NAME_LEN} bytes")
    return _U8.pack(len(raw)) + raw


def unpack_list(resp: bytes) -> list[TensorMeta]:
    """Parse a ``LIST_TENSORS`` response."""
    if len(resp) < 2:
        raise ValueError("list response too short")
    count = _U16.unpack_from(resp, 0)[0]
    out: list[TensorMeta] = []
    off = 2
    for _ in range(count):
        nlen = resp[off]
        off += 1
        name = bytes(resp[off : off + nlen]).decode("utf-8")
        off += nlen
        dtype = DType(resp[off])
        ndim = resp[off + 1]
        off += 2
        shape = tuple(
            _U32.unpack_from(resp, off + i * 4)[0] for i in range(ndim)
        )
        off += ndim * 4
        out.append(TensorMeta(name=name, dtype=dtype, shape=shape))
    return out


def unpack_motor_target(resp: bytes) -> MotorTarget:
    """Parse the fixed 14-byte ``GET_MOTOR_TARGET`` response."""
    if len(resp) < 14:
        raise ValueError("motor target response too short")
    flags = resp[0]
    target = MotorTarget()
    if flags & 1:
        target.action_id = resp[1]
        target.action_strength = _F32.unpack_from(resp, 2)[0]
    if flags & 2:
        target.x = _U16.unpack_from(resp, 6)[0]
        target.y = _U16.unpack_from(resp, 8)[0]
        target.cell_strength = _F32.unpack_from(resp, 10)[0]
    return target


# ─── List helpers ─────────────────────────────────────────────


def _infer_shape(data: Sequence[Any]) -> tuple[int, ...]:
    """Shape of a nested sequence; ragged rows are an error."""
    if not isinstance(data, (list, tuple)):
        return ()
    n = len(data)
    if n == 0:
        return (0,)
    inner = _infer_shape(data[0])
    for row in data[1:]:
        if _infer_shape(row) != inner:
            raise ValueError("ragged nested lists can't form a tensor")
    return (n, *inner)


def _flatten(
    data: Sequence[Any], shape: tuple[int, ...], dim: int, out: list[Any]
) -> None:
    if dim == len(shape):
        out.append(data)
        return
    for item in data:
        _flatten(item, shape, dim + 1, out)


def _nest(flat: list[Any], shape: tuple[int, ...]) -> Any:
    if not shape:
        return flat[0] if flat else None
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = 1
    for d in shape[1:]:
        step *= d
    return [
        _nest(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])
    ]
