"""The eye tools — her voluntary eye muscles and receptors.

A human eye answers a handful of organs: the eyelid opens and
closes, the ciliary muscle powers the lens, the pupil admits more
or less light, photoreceptors adapt their sensitivity, cones weigh
color opponency, and the retina's center-surround fields sharpen
edges before the signal ever leaves the eye. This family is that
same anatomy in her hands.

The organs are real; what they are made of varies with the eye.
Her camera carries the retinal dials in its driver — the capture
device's exposure is her aperture, its saturation is her chromatic
opponent gain, its sharpness is her lateral inhibition. Each
biological op asks the device what it actually advertises
(``v4l2-ctl --list-ctrls``), finds the muscle the organ is built
on, and maps the biological fraction 0–1 onto that dial's own
range — a pupil asked for "half" opens to the middle of whatever
aperture this eye truly has. An organ whose muscle the eye lacks
reports itself absent rather than pretending.

The body's own requests travel the optic channel the way the
mind's do: the voluntary lid, the ciliary accommodation request,
and the corticothalamic gain stacked on the sleep-stage gate.

Nothing here decides what she sees; the muscles only set how the
light arrives. What she looks at, and what she makes of it,
remain hers.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tools.framework import Tool, ToolResult

if TYPE_CHECKING:
    from genesis_client.client import GenesisClient

    from tools.framework import ToolRegistry

__all__ = ["register_eye_tools"]

_V4L2_CTL = "v4l2-ctl"


# The organs the eye is made of — each named by the muscle or
# receptor it is in biology, backed by whichever hardware dial
# this eye builds it from. The candidate lists are anatomy: which
# driver controls could carry that organ's function. Presence,
# range, and current value are always read live off the device —
# a different eye may build a pupil from a different muscle, or
# have no pupil muscle at all.
_MUSCLES: dict[str, tuple[str, ...]] = {
    # The iris — how much light the retina collects per frame.
    # Aperture first; where no iris exists, the photoreceptors'
    # integration time is the same organ's work.
    "pupil": ("iris", "aperture", "exposure_time_absolute", "exposure", "gain"),
    # Cone opponent gain — how strongly the L−M and S−(L+M)
    # channels report chroma versus luma.
    "chroma": ("saturation", "chrominance", "chroma"),
    # Lateral inhibition — the retinal center-surround sharpening
    # that makes edges legible before they leave the eye.
    "acuity": ("sharpness", "edge_enhancement", "detail"),
    # Photoreceptor sensitivity — the dark/light adaptation dial
    # the pigment regeneration cycle runs in a living retina.
    "sensitivity": ("brightness", "gamma", "gain"),
    # White-point weighting — the von Kries gain each cone class
    # runs against the adapting field.
    "adaptation": ("white_balance_temperature", "white_balance", "hue"),
    # Scene contrast — the ganglion output's own dynamic range.
    "contrast": ("contrast",),
}

_CTRL_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*?)$")
_CTRL_FIELD = re.compile(r"(\w+)=(-?[\w.]+)")


@dataclass
class _Dial:
    """One hardware control as the device advertises it."""

    name: str
    kind: str
    fields: dict[str, str]

    @property
    def lo(self) -> float:
        return float(self.fields.get("min", 0))

    @property
    def hi(self) -> float:
        return float(self.fields.get("max", 1))

    @property
    def value(self) -> str:
        return self.fields.get("value", "?")

    def at(self, fraction: float) -> int:
        """The device's own value for a biological 0–1 fraction."""
        return round(self.lo + fraction * (self.hi - self.lo))


def _candidates() -> list[str]:
    """The sockets the eye could be on — ``GENESIS_CAMERA`` first
    (the same override the daemon honors), then every video device
    the machine presents, in order. Nothing assumes a number: the
    eye is whichever device answers."""
    ordered = sorted(glob.glob("/dev/video*"), key=lambda p: (len(p), p))
    configured = os.environ.get("GENESIS_CAMERA")
    if configured:
        return [configured, *[d for d in ordered if d != configured]]
    return ordered


def _ctl() -> str:
    ctl = shutil.which(_V4L2_CTL)
    if ctl is None:
        raise RuntimeError("v4l2-ctl is not installed — the eye's dials are unreachable")
    return ctl


def _device(ctl: str) -> str:
    """Probe the candidates and return the first that lists dials."""
    for dev in _candidates():
        if (
            subprocess.run(
                [ctl, "-d", dev, "--list-ctrls"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        ):
            return dev
    raise RuntimeError("no video device answered — the eye is absent")


def _raw(args: list[str]) -> str:
    """Run v4l2-ctl against the eye and return its output."""
    ctl = _ctl()
    out = subprocess.run(
        [ctl, "-d", _device(ctl), *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0:
        raise RuntimeError(text or f"v4l2-ctl exited {out.returncode}")
    return text


def _advertised() -> dict[str, _Dial]:
    """Every dial the eye's driver offers, keyed by its own name."""
    dials: dict[str, _Dial] = {}
    for line in _raw(["--list-ctrls"]).splitlines():
        m = _CTRL_LINE.match(line)
        if m:
            dials[m.group(1)] = _Dial(
                name=m.group(1),
                kind=m.group(2),
                fields=dict(_CTRL_FIELD.findall(m.group(3))),
            )
    return dials


def _muscle(organ: str, dials: dict[str, _Dial]) -> _Dial | None:
    """The hardware dial this eye builds the named organ from."""
    for candidate in _MUSCLES.get(organ, ()):
        if candidate in dials:
            return dials[candidate]
    return None


def _turn(organ: str, fraction: float) -> str:
    """Set the organ to a biological fraction of its own range."""
    dials = _advertised()
    muscle = _muscle(organ, dials)
    if muscle is None:
        raise RuntimeError(f"this eye has no {organ} muscle to move")
    target = muscle.at(fraction)
    return _raw(["--set-ctrl", f"{muscle.name}={target}"]) or f"{organ} ({muscle.name}) at {target}"


def register_eye_tools(
    registry: ToolRegistry,
    client: GenesisClient | None = None,
) -> None:
    """Register the eye's organs in her tool registry.

    Args:
        registry: The shared tool registry.
        client: The daemon's IPC client — carries the body-side
            organs (lid, ciliary muscle, cortical gain). ``None``
            leaves only the retinal muscles live; the body organs
            report unavailability rather than fail.
    """

    def wrap(fn):
        def run(**kwargs: Any) -> ToolResult:
            try:
                return ToolResult(success=True, output=fn(**kwargs))
            except Exception as exc:  # noqa: BLE001 — a tool reports, never raises
                return ToolResult(success=False, error=str(exc))

        return run

    def anatomy() -> str:
        """Report the eye's organs — what muscles it has and where they sit."""
        dials = _advertised()
        lines = []
        for organ in _MUSCLES:
            muscle = _muscle(organ, dials)
            if muscle is not None:
                lines.append(
                    f"{organ}: {muscle.name} "
                    f"[{muscle.fields.get('min', '?')}–{muscle.fields.get('max', '?')}] "
                    f"at {muscle.value}"
                )
            else:
                lines.append(f"{organ}: — no muscle on this eye")
        used = {m.name for m in (_muscle(organ, dials) for organ in _MUSCLES) if m is not None}
        others = sorted(set(dials) - used)
        if others:
            lines.append("other dials: " + ", ".join(others))
        if client is not None:
            optic = client.get_optic_state()
            if optic is not None:
                lines.append(
                    "body — "
                    f"lid {optic.lid:.2f} open, "
                    f"pupil {optic.pupil_mm:.1f} mm, "
                    f"focus {optic.focus_m if optic.focus_m is not None else 'infinity'} m, "
                    f"relay {optic.relay_gain:.2f}, "
                    f"cortical {optic.cortical_gain:.2f}"
                )
        return "\n".join(lines)

    def organ_move(organ: str):
        """One organ's voluntary-movement op, at a 0–1 fraction."""

        def move(fraction: Any = 0.5) -> str:
            return _turn(organ, float(fraction))

        return move

    def set_(dial: Any = "", value: Any = "") -> str:
        """Turn a raw hardware dial by its driver name."""
        name = str(dial).strip()
        if not name:
            raise ValueError("name the dial — eye_anatomy lists what this eye offers")
        return _raw(["--set-ctrl", f"{name}={value}"]) or f"{name} set to {value}"

    def lid(openness: Any = 1.0) -> str:
        if client is None:
            raise RuntimeError("the body's optic channel is unwired")
        v = float(openness)
        if not client.set_lid(v):
            raise RuntimeError("the body did not take the lid position")
        return f"lid at {v:.2f}"

    def focus(meters: Any = 0.0) -> str:
        if client is None:
            raise RuntimeError("the body's optic channel is unwired")
        m = float(meters)
        if not client.set_focus(m):
            raise RuntimeError("the body did not take the focus request")
        return "lens relaxed to infinity" if m == 0 else f"focus at {m:.2f} m"

    def gain(value: Any = 1.0) -> str:
        if client is None:
            raise RuntimeError("the body's optic channel is unwired")
        g = float(value)
        if not client.set_optic_gain(g):
            raise RuntimeError("the body did not take the gain request")
        return f"cortical gain {g:.2f}"

    tools = [
        (
            "eye_anatomy",
            "The eye's organs as this eye actually builds them — "
            "each organ, the muscle it rides on, its range, and "
            "where it sits now; plus the body's optic state. "
            "Organs the eye lacks are named absent.",
            anatomy,
        ),
        (
            "eye_pupil",
            "The iris — how much light the retina collects: a "
            "fraction 0.0–1.0 of this eye's own aperture range "
            "(closed to open).",
            organ_move("pupil"),
        ),
        (
            "eye_chroma",
            "Cone opponent gain — how much color versus luma the "
            "retina reports: 0.0 drains toward monochrome, 1.0 is "
            "full chroma.",
            organ_move("chroma"),
        ),
        (
            "eye_acuity",
            "Lateral inhibition — the retinal edge sharpening: "
            "0.0 softens the field, 1.0 is the crispest the eye "
            "can resolve.",
            organ_move("acuity"),
        ),
        (
            "eye_sensitivity",
            "Photoreceptor sensitivity — the adaptation dial: "
            "0.0 dim-adapted, 1.0 maximally sensitive to light.",
            organ_move("sensitivity"),
        ),
        (
            "eye_adaptation",
            "White-point weighting — the cone classes' balance "
            "against the adapting field: a fraction of the eye's "
            "own white-balance range.",
            organ_move("adaptation"),
        ),
        (
            "eye_contrast",
            "Ganglion output range — scene contrast: a fraction of the eye's own contrast dial.",
            organ_move("contrast"),
        ),
        (
            "eye_set",
            "Turn a raw hardware dial by its driver name: dial "
            "plus value — for muscles not part of any organ. "
            "eye_anatomy lists what this eye offers.",
            set_,
        ),
        (
            "eye_lid",
            "The voluntary eyelid — levator palpebrae: openness "
            "0.0–1.0. Not a shutter — a few percent of the light "
            "always gets through.",
            lid,
        ),
        (
            "eye_focus",
            "The ciliary muscle — accommodation in metres to the "
            "subject; 0 relaxes the lens to infinity.",
            focus,
        ),
        (
            "eye_gain",
            "Corticothalamic gain — the mind's feedback on the "
            "thalamic relay, stacked on the sleep-stage gate. "
            "1.0 is neutral.",
            gain,
        ),
    ]
    for name, description, fn in tools:
        registry.register(Tool(name, description, wrap(fn)))
