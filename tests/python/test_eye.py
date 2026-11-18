"""The eye tools — organ resolution over live-advertised dials."""

from __future__ import annotations

import tools.eye as eye_mod
from tools.eye import register_eye_tools
from tools.framework import ToolRegistry

_CTRLS = """User Controls

         brightness 0x00980900 (int)    : min=0 max=255 value=128
          contrast 0x00980901 (int)    : min=0 max=255 value=32
        saturation 0x00980902 (int)    : min=0 max=100 value=64
         sharpness 0x0098091b (int)    : min=0 max=7 value=0
 exposure_time_absolute 0x009a0902 (int)    : min=2 max=1250 value=156
"""


def _registry(monkeypatch, client=None, raw=None):
    """A registry with the eye's subprocess layer stubbed."""
    calls = []

    def fake_raw(args):
        calls.append(args)
        return "" if "--set-ctrl" in args else _CTRLS

    monkeypatch.setattr(eye_mod, "_raw", raw or fake_raw)
    reg = ToolRegistry()
    register_eye_tools(reg, client=client)
    return reg, calls


class _Client:
    """The body's optic channel, answering for the tests."""

    def __init__(self, take=True):
        self.take = take
        self.sent = []

    def get_optic_state(self):
        return None

    def set_lid(self, v):
        self.sent.append(("lid", v))
        return self.take

    def set_focus(self, m):
        self.sent.append(("focus", m))
        return self.take

    def set_optic_gain(self, g):
        self.sent.append(("gain", g))
        return self.take


def test_anatomy_names_organs_and_muscles(monkeypatch):
    reg, _ = _registry(monkeypatch)
    r = reg.run("eye_anatomy")
    assert r.success
    assert "chroma: saturation" in r.output
    assert "acuity: sharpness" in r.output
    assert "pupil: exposure_time_absolute" in r.output


def test_organ_maps_fraction_onto_device_range(monkeypatch):
    reg, calls = _registry(monkeypatch)
    r = reg.run("eye_chroma", fraction=1.0)
    assert r.success
    assert ["--set-ctrl", "saturation=100"] in calls
    r = reg.run("eye_acuity", fraction=0.5)
    assert r.success
    # sharpness range 0–7: half of the muscle's own range, not a clamp.
    assert ["--set-ctrl", "sharpness=4"] in calls


def test_absent_organ_reports_no_muscle(monkeypatch):
    def fewer(args):
        return (
            ""
            if "--set-ctrl" in args
            else " saturation 0x00980902 (int)    : min=0 max=100 value=64"
        )

    reg, _ = _registry(monkeypatch, raw=fewer)
    r = reg.run("eye_acuity", fraction=0.5)
    assert not r.success
    assert "no acuity muscle" in r.error


def test_raw_dial_passthrough(monkeypatch):
    reg, calls = _registry(monkeypatch)
    r = reg.run("eye_set", dial="gamma", value="110")
    assert r.success
    assert ["--set-ctrl", "gamma=110"] in calls


def test_body_organs_reach_the_client(monkeypatch):
    client = _Client()
    reg, _ = _registry(monkeypatch, client=client)
    assert reg.run("eye_lid", openness=0.3).success
    assert reg.run("eye_focus", meters=2.0).success
    assert reg.run("eye_gain", value=1.5).success
    assert client.sent == [("lid", 0.3), ("focus", 2.0), ("gain", 1.5)]


def test_body_organs_unwired_report(monkeypatch):
    reg, _ = _registry(monkeypatch, client=None)
    r = reg.run("eye_lid", openness=0.5)
    assert not r.success
    assert "unwired" in r.error


def test_body_refusal_reported(monkeypatch):
    reg, _ = _registry(monkeypatch, client=_Client(take=False))
    r = reg.run("eye_focus", meters=2.0)
    assert not r.success
    assert "did not take" in r.error
