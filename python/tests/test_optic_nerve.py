"""Optic nerve tests — the body's retinal afference and the mind's
thalamic gate honouring.

The Rust daemon measures the retinal segment itself (subcortical
channels: SCN light, collicular transients, TRN relay gate); these
tests pin the wire decode and the mind-side behaviour that consumes
it: a look through a closed relay is a blind look, and cortical
engagement feeds back onto the relay it looked through.
"""

import struct
import time

import pytest

from genesis_client.types import OpticState


def _optic_wire(
    *,
    flags: int = 3,
    illuminance: float = 0.4,
    melanopic: float = 0.5,
    motion: float = 0.1,
    orienting: float = 0.2,
    relay_gain: float = 0.8,
    cortical_gain: float = 0.5,
    retinal_lux: float = 700.0,
    frame_seq: int = 1234,
    frame_age_ms: int = 33,
    pupil_mm: float = 4.2,
    accommodation_dpt: float = 2.0,
    lid: float = 0.5,
) -> bytes:
    """Build the GET_OPTIC_STATE body exactly as the daemon sends it:
    [u8 ack][u8 flags][u8 pad][7 × f32][u64 seq][i64 age_ms]
    [f32 pupil_mm][f32 accommodation_dpt][f32 lid]."""
    return (
        bytes([1, flags, 0])
        + struct.pack(
            "<7f",
            illuminance,
            melanopic,
            motion,
            orienting,
            relay_gain,
            cortical_gain,
            retinal_lux,
        )
        + struct.pack("<Q", frame_seq)
        + struct.pack("<q", frame_age_ms)
        + struct.pack("<f", pupil_mm)
        + struct.pack("<f", accommodation_dpt)
        + struct.pack("<f", lid)
    )


class TestOpticStateUnpack:
    def test_full_decode(self) -> None:
        state = OpticState.unpack(_optic_wire())
        assert state is not None
        assert state.eye_present is True
        assert state.frames_fresh is True
        assert abs(state.illuminance - 0.4) < 1e-6
        assert abs(state.melanopic - 0.5) < 1e-6
        assert abs(state.motion - 0.1) < 1e-6
        assert abs(state.orienting - 0.2) < 1e-6
        assert abs(state.relay_gain - 0.8) < 1e-6
        assert abs(state.cortical_gain - 0.5) < 1e-6
        assert abs(state.retinal_lux - 700.0) < 1e-3
        assert state.frame_seq == 1234
        assert state.frame_age_ms == 33
        assert abs(state.pupil_mm - 4.2) < 1e-6
        assert abs(state.accommodation_dpt - 2.0) < 1e-6
        assert state.lid == pytest.approx(0.5)
        # 2 D of lens power is a 50 cm focus distance.
        assert state.focus_m == pytest.approx(0.5)

    def test_legacy_body_without_pupil_decodes_zero(self) -> None:
        """A daemon that predates the pupil field sends the 46-byte
        body — pupil_mm reads as unreported, not a decode failure."""
        state = OpticState.unpack(_optic_wire()[:47])
        assert state is not None
        assert state.pupil_mm == 0.0
        assert state.accommodation_dpt == 0.0
        assert state.focus_m is None
        assert state.lid == 1.0

    def test_pupil_only_body_decodes_without_lens(self) -> None:
        """The 50-byte body (pupil but no lens) is additive-legacy:
        accommodation reads as unreported."""
        state = OpticState.unpack(_optic_wire()[:51])
        assert state is not None
        assert state.pupil_mm == pytest.approx(4.2)
        assert state.accommodation_dpt == 0.0
        assert state.focus_m is None
        assert state.lid == 1.0

    def test_lens_only_body_decodes_without_lid(self) -> None:
        """The 54-byte body (pupil + lens but no lid) is
        additive-legacy: an unreported lid reads as open, not shut —
        a legacy daemon cannot look asleep to the optics layer."""
        state = OpticState.unpack(_optic_wire()[:55])
        assert state is not None
        assert state.pupil_mm == pytest.approx(4.2)
        assert state.accommodation_dpt == pytest.approx(2.0)
        assert state.lid == 1.0

    def test_infinity_focus_reports_no_distance(self) -> None:
        """0 D is the relaxed lens — focus at infinity, no distance."""
        state = OpticState.unpack(_optic_wire(accommodation_dpt=0.0))
        assert state is not None
        assert state.focus_m is None

    def test_absent_eye_decodes_as_a_reading(self) -> None:
        state = OpticState.unpack(_optic_wire(flags=0, frame_seq=0, frame_age_ms=-1))
        assert state is not None
        assert state.eye_present is False
        assert state.frames_fresh is False
        assert state.frame_age_ms == -1

    def test_short_or_unacked_is_no_reading(self) -> None:
        assert OpticState.unpack(b"") is None
        assert OpticState.unpack(b"\x01") is None
        assert OpticState.unpack(_optic_wire()[:20]) is None
        assert OpticState.unpack(b"\x80" + _optic_wire()[1:]) is None


class _OpticClient:
    """A client stub carrying only what Vision.see() reads."""

    def __init__(self, optic: OpticState | None) -> None:
        self._optic = optic

    def get_optic_state(self) -> OpticState | None:
        return self._optic


class TestThalamicGate:
    def test_closed_gate_returns_gated_without_reading_the_frame(self, monkeypatch) -> None:
        """NREM/REM: the TRN blocks transmission upstream of cortex —
        see() must report "gated" and never touch the frame buffer."""
        import occipital_lobe.perception.vision as vision_mod
        from occipital_lobe.perception.vision import Vision

        def _unread_frame(copy: bool = True):
            raise AssertionError("the gate must close before the frame read")

        monkeypatch.setattr(vision_mod, "latest_frame", _unread_frame)
        nrem = OpticState.unpack(_optic_wire(relay_gain=0.02))
        scene = Vision().see(client=_OpticClient(nrem))
        assert scene.status == "gated"
        assert scene.relay_gate == pytest.approx(0.02, abs=1e-6)

    def test_open_gate_records_the_relay(self, monkeypatch) -> None:
        """An awake relay passes seeing through and the scene carries
        the gate it was read under."""
        import occipital_lobe.perception.vision as vision_mod
        from occipital_lobe.perception.vision import Vision

        vis = Vision()
        monkeypatch.setattr(vision_mod, "latest_frame", lambda copy=True: None)
        vis._available = True
        awake = OpticState.unpack(_optic_wire(relay_gain=0.8))
        # Frame read returns mid_update — the gate itself was honoured;
        # what matters is seeing proceeded past it.
        scene = vis.see(client=_OpticClient(awake))
        assert scene.status == "mid_update"

    def test_no_optic_channel_sees_ungated(self, monkeypatch) -> None:
        """A daemon that predates GET_OPTIC_STATE reports no gate —
        seeing proceeds exactly as it always has."""
        import occipital_lobe.perception.vision as vision_mod
        from occipital_lobe.perception.vision import Vision

        vis = Vision()
        monkeypatch.setattr(vision_mod, "latest_frame", lambda copy=True: None)
        vis._available = True
        scene = vis.see(client=_OpticClient(None))
        assert scene.status == "mid_update"
        assert scene.relay_gate is None

    def test_client_absent_sees_ungated(self, monkeypatch) -> None:
        import occipital_lobe.perception.vision as vision_mod
        from occipital_lobe.perception.vision import Vision

        vis = Vision()
        monkeypatch.setattr(vision_mod, "latest_frame", lambda copy=True: None)
        vis._available = True
        scene = vis.see()
        assert scene.status == "mid_update"
        assert scene.relay_gate is None


class TestCorticothalamicGain:
    def test_decays_with_a_half_life(self) -> None:
        from occipital_lobe.perception.vision import Vision

        vis = Vision()
        assert vis.corticothalamic_gain() == 0.0
        vis._optic_engagement = 0.8
        vis._optic_engagement_at = time.monotonic()
        assert 0.7 < vis.corticothalamic_gain() <= 0.8
        # 40 s out = two 20 s half-lives → 0.8 / 4.
        vis._optic_engagement_at = time.monotonic() - 40.0
        assert abs(vis.corticothalamic_gain() - 0.2) < 0.03

    def test_gated_status_composes_words(self) -> None:
        """ "gated" must be a status the vocabulary can describe —
        a blind look must not surface a raw status word."""
        from limbic_system.emotion import EmotionalState
        from temporal_lobe.vocabulary import Vocabulary

        vocab = Vocabulary(seed=42)
        emotion = EmotionalState(
            label="calm",
            nuance="baseline",
            cognitive_style="steady",
            valence=0.0,
            alertness=0.5,
            plasticity=0.5,
            creativity=0.5,
            caution=0.3,
            openness_to_engage=0.7,
        )
        content = vocab._compose_vision_status_content("gated", emotion)
        assert content is not None and len(content) > 4
