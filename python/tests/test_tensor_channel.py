"""Tensor channel tests — the Rust↔Python multidimensional path.

The codec tests pin the wire layout shared with
``src/daemon/tensor_slab.rs``; the client tests run against an
in-process fake daemon on a Unix socketpair that implements the same
wire semantics, so the round trip exercises the real request framing
(``_request`` → ``write_message`` → socket → handler).
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from genesis_client.client import GenesisClient
from genesis_client.protocol import (
    DROP_TENSOR,
    GET_MOTOR_TARGET,
    GET_TENSOR,
    HANDSHAKE,
    LIST_TENSORS,
    PROTOCOL_VERSION,
    PUT_TENSOR,
    read_message,
    write_message,
)
from genesis_client.tensor import (
    DType,
    Tensor,
    pack_name,
    unpack_motor_target,
)

# ─── Codec ────────────────────────────────────────────────────


def test_tensor_roundtrip_2d_grid():
    """An ARC-style u8 grid survives encode→decode with shape intact."""
    rows = [[r * 8 + c for c in range(8)] for r in range(8)]
    t = Tensor.from_lists(rows, DType.U8)
    assert t.shape == (8, 8)
    back = Tensor.unpack(t.pack())
    assert back == t
    assert back.to_lists() == rows


def test_tensor_roundtrip_3d_and_4d():
    t = Tensor.from_lists(
        [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], DType.F32
    )
    assert t.shape == (2, 2, 2)
    back = Tensor.unpack(t.pack())
    assert back == t

    deep = Tensor.from_lists([[[[1, 2], [3, 4]]]], DType.I32)
    back = Tensor.unpack(deep.pack())
    assert back.shape == (1, 1, 2, 2)
    assert back.to_lists() == [[[[1, 2], [3, 4]]]]


def test_tensor_empty_extent():
    t = Tensor(dtype=DType.F32, shape=(0, 4), payload=b"")
    back = Tensor.unpack(t.pack())
    assert back.elements == 0
    assert back.shape == (0, 4)


def test_tensor_rejects_ragged_and_bad_rank():
    with pytest.raises(ValueError):
        Tensor.from_lists([[1, 2], [3]])
    with pytest.raises(ValueError):
        Tensor(dtype=DType.U8, shape=(2, 2), payload=b"\x00" * 3)
    with pytest.raises(ValueError):
        Tensor(dtype=DType.U8, shape=(), payload=b"\x00")


def test_tensor_numpy_bridge():
    np = pytest.importorskip("numpy")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Tensor.from_array(arr)
    assert (t.dtype, t.shape) == (DType.F32, (3, 4))
    assert (t.to_array() == arr).all()


def test_pack_name_bounds():
    assert pack_name("world.frame") == b"\x0bworld.frame"
    with pytest.raises(ValueError):
        pack_name("")
    with pytest.raises(ValueError):
        pack_name("x" * 200)


# ─── In-process fake daemon ───────────────────────────────────


def _fake_daemon(sock: socket.socket) -> None:
    """Mirror of the Rust slab's wire semantics (tensor_slab.rs)."""
    slab: dict[str, Tensor] = {}
    try:
        while True:
            cmd, payload = read_message(sock)
            if cmd == HANDSHAKE:
                write_message(sock, cmd, bytes([PROTOCOL_VERSION]))
                continue
            if cmd == PUT_TENSOR:
                nlen = payload[0]
                name = payload[1 : 1 + nlen].decode()
                slab[name] = Tensor.unpack(payload[1 + nlen :])
                write_message(sock, cmd, b"\x01")
            elif cmd == GET_TENSOR:
                nlen = payload[0]
                name = payload[1 : 1 + nlen].decode()
                t = slab.get(name)
                write_message(sock, cmd, b"\x00" if t is None else b"\x01" + t.pack())
            elif cmd == LIST_TENSORS:
                out = bytearray(struct.pack("<H", len(slab)))
                for name in sorted(slab):
                    t = slab[name]
                    out += bytes([len(name)]) + name.encode()
                    out += bytes([int(t.dtype), len(t.shape)])
                    for d in t.shape:
                        out += struct.pack("<I", d)
                write_message(sock, cmd, bytes(out))
            elif cmd == DROP_TENSOR:
                nlen = payload[0]
                slab.pop(payload[1 : 1 + nlen].decode(), None)
                write_message(sock, cmd, b"\x01")
            elif cmd == GET_MOTOR_TARGET:
                resp = _motor_target(slab)
                write_message(sock, cmd, resp)
            else:
                write_message(sock, cmd, b"\x85")  # INVALID_VALUE
    except Exception:  # noqa: BLE001 — the fake daemon dies with its socket
        return


def _motor_target(slab: dict[str, Tensor]) -> bytes:
    """Mirror of `src/field/motor.rs::current_motor_target`'s decode —
    14 bytes: `[flags][action_id][f32][u16 x][u16 y][f32]`."""
    action_id, a_strength = 0xFF, 0.0
    scores = slab.get("motor.actions")
    ids = slab.get("motor.action_ids")
    if (
        scores is not None
        and scores.dtype == DType.F32
        and len(scores.shape) == 1
        and scores.shape[0] > 0
    ):
        vals = scores.values()
        i = max(range(len(vals)), key=lambda i: vals[i])
        action_id = (
            ids.payload[i]
            if ids is not None and ids.dtype == DType.U8 and ids.shape == scores.shape
            else i
        )
        a_strength = vals[i]
        flags_a = 1
    else:
        flags_a = 0
    x = y = 0
    s_strength = 0.0
    field = slab.get("motor.spatial")
    flags_c = 0
    if (
        field is not None
        and field.dtype == DType.F32
        and len(field.shape) == 2
        and all(d > 0 for d in field.shape)
    ):
        h, w = field.shape
        vals = field.values()
        i = max(range(len(vals)), key=lambda i: vals[i])
        fx, fy = i % w, i // w  # index → coordinate (row-major)
        x, y = min(fx * 64 // w, 63), min(fy * 64 // h, 63)
        s_strength = vals[i]
        flags_c = 2
    return (
        bytes([flags_a | flags_c, action_id])
        + struct.pack("<f", a_strength)
        + struct.pack("<HH", x, y)
        + struct.pack("<f", s_strength)
    )


@pytest.fixture()
def daemon_client(tmp_path):
    """A GenesisClient wired to a live fake daemon."""
    path = tmp_path / "genesis.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    client = GenesisClient(str(path), request_timeout=5.0)
    thread = threading.Thread(target=_accept_loop, args=(server,), daemon=True)
    thread.start()
    client.connect()
    yield client
    client.disconnect()
    server.close()


def _accept_loop(server: socket.socket) -> None:
    while True:
        try:
            conn, _ = server.accept()
        except OSError:
            return
        threading.Thread(target=_fake_daemon, args=(conn,), daemon=True).start()


def test_client_put_get_roundtrip(daemon_client):
    frame = [[0, 1], [2, 3]]
    t = Tensor.from_lists(frame, DType.U8)
    assert daemon_client.put_tensor("world.frame", t)
    back = daemon_client.get_tensor("world.frame")
    assert back == t
    assert daemon_client.get_tensor("absent") is None


def test_client_list_and_drop(daemon_client):
    daemon_client.put_tensor("a", Tensor.from_lists([1.0, 2.0], DType.F32))
    daemon_client.put_tensor("b", Tensor.from_lists([[1]], DType.I32))
    metas = {m.name: m for m in daemon_client.list_tensors()}
    assert metas["a"].shape == (2,)
    assert metas["b"].shape == (1, 1)
    assert daemon_client.drop_tensor("a")
    assert daemon_client.drop_tensor("a")  # idempotent
    assert [m.name for m in daemon_client.list_tensors()] == ["b"]


def test_client_motor_target_decode(daemon_client):
    """Decision → field → daemon-side decode on the 64×64 grid."""
    # Spatial bump at field-local (x=4, y=2) of an 8×8 field → (32, 16).
    field = [[0.0] * 8 for _ in range(8)]
    field[2][4] = 1.0
    daemon_client.put_tensor("motor.spatial", Tensor.from_lists(field, DType.F32))
    daemon_client.put_tensor(
        "motor.actions", Tensor.from_lists([0.1, 0.9, 0.4], DType.F32)
    )
    daemon_client.put_tensor(
        "motor.action_ids", Tensor.from_lists([1, 3, 5], DType.U8)
    )
    target = daemon_client.get_motor_target()
    assert target.has_action and target.action_id == 3
    assert target.has_cell and (target.x, target.y) == (32, 16)
    assert target.cell_strength == pytest.approx(1.0)


def test_motor_target_empty_slab():
    resp = unpack_motor_target(b"\x00\xff" + b"\x00" * 12)
    assert not resp.has_action and not resp.has_cell
