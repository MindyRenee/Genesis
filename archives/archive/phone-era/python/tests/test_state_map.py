"""Ambient field reads — CoreStateMap and the client's mmap fast path.

The tests fabricate a schema-v5 ``core_state.bin`` image the way the
daemon publishes it (header, derived blocks, CRC over the covered
ranges) and verify verified reads, staleness, inode-following across
rebirth, the NeuroSummary synthesis, the channel lock, and the
socket-free ``get_state`` / ``get_neuro_summary`` fast paths — no
daemon required, which is the point: the field is ambient.
"""

import os
import socket
import struct
import threading
import time
import zlib

import pytest
from genesis_client.client import GenesisClient
from genesis_client.exceptions import ConnectionError as GenesisConnectionError
from genesis_client.protocol import PING, read_message, write_message
from genesis_client.state_map import SCHEMA_VERSION, STATE_FILENAME, STATE_SIZE, CoreStateMap
from genesis_client.types import NeuroSummary

# Schema-v5 layout (src/state/core_state.rs): the pieces the decoders read.
# Header: magic, version, size, pad, created_at, last_updated, heartbeat, seq_lock, instance_id.
_HEADER_FMT = "<4sII4sQQQQQ"
_EFF_LEVELS = 2360  # neuro vector effective_levels[18] (f32 each)
_GLOBAL_TONE = 2432
_AROUSAL = 2436
_VALENCE = 2440
_PLASTICITY = 2444
_ZONE = 2472
_PHASE = 2473
_STM_COUNT = 2584
_LTM_COUNT = 2600
_ENC_W = 2632
_CON_W = 2636
_RET_W = 2640
_SLEEP_H = 3256
_CHECKSUM = 3412
_DA_TONIC = 96  # chemicals[0] at 56, tonic_level at +40
_DA_PHASIC = 100  # phasic_level at +44


def _image(
    *,
    heartbeat: int = 7,
    last_updated: int | None = None,
    arousal: float = 0.6,
    valence: float = 0.2,
    global_tone: float = 0.9,
    plasticity: float = 0.7,
    dopamine: float = 0.4,
    da_tonic: float = 0.3,
    da_phasic: float = 0.05,
    phase: int = 1,
    enc: float = 0.6,
    con: float = 0.5,
    ret: float = 0.4,
    sleep_h: float = 40.0,
    version: int = SCHEMA_VERSION,
    crc_ok: bool = True,
) -> bytes:
    """A 4096-byte state-file page holding a valid schema-v5 image."""
    now = int(time.time() * 1000)
    buf = bytearray(4096)
    struct.pack_into(
        _HEADER_FMT,
        buf,
        0,
        b"GNSX",
        version,
        STATE_SIZE,
        b"\0" * 4,
        now,
        last_updated if last_updated is not None else now,
        heartbeat,
        0,
        1,
    )
    struct.pack_into("<f", buf, _EFF_LEVELS, dopamine)
    struct.pack_into("<f", buf, _GLOBAL_TONE, global_tone)
    struct.pack_into("<f", buf, _AROUSAL, arousal)
    struct.pack_into("<f", buf, _VALENCE, valence)
    struct.pack_into("<f", buf, _PLASTICITY, plasticity)
    buf[_ZONE] = 2
    buf[_PHASE] = phase
    struct.pack_into("<I", buf, _STM_COUNT, 3)
    struct.pack_into("<Q", buf, _LTM_COUNT, 11)
    struct.pack_into("<f", buf, _ENC_W, enc)
    struct.pack_into("<f", buf, _CON_W, con)
    struct.pack_into("<f", buf, _RET_W, ret)
    struct.pack_into("<d", buf, _SLEEP_H, sleep_h)
    struct.pack_into("<f", buf, _DA_TONIC, da_tonic)
    struct.pack_into("<f", buf, _DA_PHASIC, da_phasic)
    crc = zlib.crc32(bytes(buf[0:40]))
    crc = zlib.crc32(bytes(buf[48:_CHECKSUM]), crc)
    struct.pack_into("<I", buf, _CHECKSUM, crc if crc_ok else crc ^ 0xFF)
    return bytes(buf)


def _write(path, image: bytes) -> None:
    with open(path, "wb") as f:
        f.write(image)


class TestCoreStateMap:
    def test_read_returns_verified_image(self, tmp_path):
        path = tmp_path / STATE_FILENAME
        image = _image(heartbeat=42)
        _write(path, image)
        m = CoreStateMap(str(path))
        assert m.read() == image[:STATE_SIZE]
        m.close()

    def test_read_state_decodes(self, tmp_path):
        path = tmp_path / STATE_FILENAME
        _write(path, _image(arousal=0.6, dopamine=0.4, sleep_h=80.76))
        m = CoreStateMap(str(path))
        state = m.read_state()
        assert state is not None
        assert state.arousal == pytest.approx(0.6)
        assert state.chemicals["dopamine"] == pytest.approx(0.4)
        assert state.heartbeat == 7
        assert state.sleep_pressure == pytest.approx(80.76 / 323.88)

    def test_missing_file_returns_none(self, tmp_path):
        m = CoreStateMap(str(tmp_path / STATE_FILENAME))
        assert m.read() is None

    def test_bad_crc_returns_none(self, tmp_path):
        path = tmp_path / STATE_FILENAME
        _write(path, _image(crc_ok=False))
        assert CoreStateMap(str(path)).read() is None

    def test_wrong_version_returns_none(self, tmp_path):
        path = tmp_path / STATE_FILENAME
        _write(path, _image(version=4))
        assert CoreStateMap(str(path)).read() is None

    def test_stale_field_returns_none(self, tmp_path):
        path = tmp_path / STATE_FILENAME
        old = int(time.time() * 1000) - 60_000
        _write(path, _image(last_updated=old))
        assert CoreStateMap(str(path)).read() is None

    def test_fresh_enough_under_custom_limit(self, tmp_path):
        path = tmp_path / STATE_FILENAME
        old = int(time.time() * 1000) - 500
        _write(path, _image(last_updated=old))
        m = CoreStateMap(str(path), max_age_ms=100)
        assert m.read() is None
        assert m.read(max_age_ms=60_000) is not None

    def test_inode_change_follows_rebirth(self, tmp_path):
        """A recreated file (reborn body) replaces the inode — the map
        must follow the path, not the old mapping."""
        path = tmp_path / STATE_FILENAME
        _write(path, _image(heartbeat=1))
        m = CoreStateMap(str(path))
        assert m.read() is not None
        _write(tmp_path / "next.bin", _image(heartbeat=99))
        os.replace(tmp_path / "next.bin", path)
        buf = m.read()
        assert buf is not None
        assert struct.unpack_from("<Q", buf, 32)[0] == 99


class TestNeuroSummaryFromState:
    def test_fields_match_daemon_packing(self):
        s = NeuroSummary.from_state(
            _image(
                arousal=0.6,
                valence=0.2,
                global_tone=0.9,
                plasticity=0.7,
                enc=0.6,
                con=0.5,
                ret=0.4,
                phase=3,
                da_tonic=0.3,
                da_phasic=0.05,
            )
        )
        assert s.arousal == pytest.approx(0.6)
        assert s.valence == pytest.approx(0.2)
        assert s.global_tone == pytest.approx(0.9)
        assert s.plasticity_gate == pytest.approx(0.7)
        assert s.encoding_weight == pytest.approx(0.6)
        assert s.consolidation_weight == pytest.approx(0.5)
        assert s.retrieval_weight == pytest.approx(0.4)
        assert s.phase == 3
        assert s.dopamine_tonic == pytest.approx(0.3)
        assert s.dopamine_phasic == pytest.approx(0.05)

    def test_daemon_clamps_applied(self):
        """finite_clamp semantics: over-range clamps, non-finite → lower bound."""
        s = NeuroSummary.from_state(_image(arousal=1.4, valence=-3.0))
        assert s.arousal == 1.0
        assert s.valence == -1.0


class TestClientFastPath:
    def test_get_state_without_daemon(self, tmp_path):
        """The field is ambient — readable with no socket and no connect()."""
        _write(tmp_path / STATE_FILENAME, _image(arousal=0.33))
        client = GenesisClient(str(tmp_path / "genesis.sock"))
        state = client.get_state()
        assert state.arousal == pytest.approx(0.33)
        assert not client.is_connected

    def test_get_neuro_summary_without_daemon(self, tmp_path):
        _write(tmp_path / STATE_FILENAME, _image(arousal=0.5, phase=2))
        client = GenesisClient(str(tmp_path / "genesis.sock"))
        summary = client.get_neuro_summary()
        assert summary.arousal == pytest.approx(0.5)
        assert summary.phase == 2

    def test_no_field_falls_back_to_ipc_error(self, tmp_path):
        """No file and no daemon: the old error contract is preserved."""
        client = GenesisClient(str(tmp_path / "genesis.sock"))
        with pytest.raises(GenesisConnectionError):
            client.get_state()

    def test_per_thread_axons_never_braid(self, tmp_path):
        """Each thread carries its own wire; replies can't cross.

        Every worker gets its own socketpair with an echo server;
        because the wires are private by construction, no caller can
        read another thread's frame — the same guarantee the old
        channel lock enforced, without the lock.
        """
        pairs = [socket.socketpair() for _ in range(6)]
        stop = threading.Event()

        def serve(srv: socket.socket) -> None:
            srv.settimeout(0.2)
            while not stop.is_set():
                try:
                    cmd, payload = read_message(srv)
                except OSError:
                    continue
                try:
                    write_message(srv, cmd, payload)
                except OSError:
                    return

        servers = [threading.Thread(target=serve, args=(srv,), daemon=True) for srv, _ in pairs]
        for t in servers:
            t.start()

        client = GenesisClient(str(tmp_path / "genesis.sock"))
        client._ever_connected = True

        errors: list[str] = []

        def worker(n: int, wire: socket.socket) -> None:
            # The worker's own axon — bound on its own thread.
            client._read_sock = wire
            for i in range(30):
                want = f"req-{n}-{i}".encode()
                try:
                    got = client._request(PING, want)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"thread {n}: {e}")
                    return
                if got != want:
                    errors.append(f"thread {n}: sent {want!r}, got {got!r}")
                    return

        threads = [
            threading.Thread(target=worker, args=(n, cli)) for n, (_, cli) in enumerate(pairs)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)

        stop.set()
        for t in servers:
            t.join(timeout=5)
        for srv, cli in pairs:
            srv.close()
            cli.close()
        assert errors == []
