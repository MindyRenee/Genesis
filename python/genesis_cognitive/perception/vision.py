"""Vision for Genesis — interprets the retina's shared-memory camera feed.

This module is the bridge between the raw camera feed (retina.py) and
Genesis's cognitive experience. It produces a structured scene percept
— color, objects, faces, lighting, and spatial layout as data — that
the language engine composes her report from.

The pipeline:

    retina frame (shared memory)
        |
        v
    Color analysis          -> spatial color regions, lighting, mood
        |
        v
    Object recognition (IT) -> what things are, where they are
        |
        v
    Face recognition (FFA)  -> who is present
        |
        v
    V1 sparse coding        -> edges, contours, gamma (sublayer)
        |
        v
    Vision.see()            -> VisionScene (structured percept data)
        |                      + memory store
        |                      + concept network learning
        |                      + neurochemical modulation
        |                      + brain wave gamma update
        v
    caller -> language engine -> "I see Alice in warm light; there's
     a blue cup on the right and a laptop in the center."

The module emits perception, not prose: ``VisionScene`` carries the
scene's structure as data, and her report is composed downstream by
the language engine — she doesn't recite a template.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_DOPAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OXYTOCIN,
)

from ..auditory import DetectedObject, ObjectRecognizer
from ..concepts import RelationType
from ..vision import V1Model, VisualField
from .recognition import FaceRecognizer
from .retina import latest_frame

if TYPE_CHECKING:
    from ..emotion import EmotionalState

logger = logging.getLogger(__name__)


@dataclass
class VisionScene:
    """The structured content of one glance — perception, not prose.

    This is data: what she actually perceived, as fields the language
    engine composes her report from. Nothing here is a sentence she
    recites; ``see()`` hands this to the caller, and the caller routes
    ``as_metadata()`` through the language engine.
    """

    # "ok" | "unavailable" | "mid_update" | "unprocessed"
    status: str = "ok"
    light_level: str = ""
    dominant_color: str = ""
    warmth: str = ""
    brightness: float = 0.0
    faces: list[str] = field(default_factory=list)
    n_unknown_faces: int = 0
    # Each entry: {"name": ..., "color": ... or "", "position": ...}
    objects: list[dict[str, str]] = field(default_factory=list)
    light_direction: str = ""  # "left" / "right" / ""
    highly_structured: bool = False
    salience: float = 0.0

    def as_metadata(self) -> dict:
        """The scene as language-engine metadata (the ``vision_scene`` slot)."""
        return {
            "light": self.light_level,
            "warmth": self.warmth,
            "color": self.dominant_color,
            "faces": list(self.faces),
            "n_unknown_faces": self.n_unknown_faces,
            "objects": [dict(o) for o in self.objects],
            "light_direction": self.light_direction,
            "structured": self.highly_structured,
            "salience": self.salience,
        }

    def memory_text(self) -> str:
        """Compact structural record for episodic memory — data, not speech."""
        parts = [f"scene | {self.light_level} {self.warmth} {self.dominant_color}".rstrip()]
        if self.faces or self.n_unknown_faces:
            named = ",".join(self.faces) or "none"
            parts.append(f"faces: {named} (+{self.n_unknown_faces} unknown)")
        if self.objects:
            objs = ",".join(
                f"{o['name']}({o['color'] or '-'})@{o['position']}"
                for o in self.objects
            )
            parts.append(f"objects: {objs}")
        if self.light_direction:
            parts.append(f"light: {self.light_direction}")
        return " | ".join(parts)


class Vision:
    """Look through the retina and produce a structured scene percept.

    Uses the V1Model (V1 sparse-coding model) to extract structured
    visual percepts -- edges, orientations, contours, and gamma power --
    rather than just dominant color. The occipital subsystem's interaction
    matrix M is plastic: what she sees reshapes how she sees the next
    thing.

    When object recognition models are available, she also identifies
    *what* things are — her "inferotemporal cortex" (IT). This is the
    ventral "what" pathway: she doesn't just see edges and colors, she
    sees objects — chairs, laptops, cups, books.

    When face recognition models are available, she also detects and
    identifies people in the frame — her "fusiform face area" (FFA).
    This is higher-order visual processing beyond V1: she doesn't just
    see shapes and colors, she sees *who* is there.
    """

    def __init__(self, occipital: V1Model | None = None) -> None:
        """Initialize the vision system."""
        self._available: bool | None = None
        self._last_color: str | None = None
        self._last_light: str | None = None
        # The occipital subsystem -- her primary visual cortex (V1).
        # Created lazily on first use so we don't pay the Gabor
        # dictionary construction cost if there's no camera.
        self._occipital: V1Model | None = occipital
        self._on_gamma: Callable[[float], None] | None = None
        # Object recognition — her "inferotemporal cortex" (IT).
        # Lazily created on first use. Detects and identifies objects.
        self._object_recognizer: ObjectRecognizer | None = None
        self._last_objects: list[str] = []  # names of objects last seen
        # Face recognition — her "fusiform face area" (FFA).
        # Lazily created on first use. Detects and identifies people.
        self._face_recognizer: FaceRecognizer | None = None
        self._last_faces: list[str] = []  # names of people last seen

    def set_gamma_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback for V1 gamma power updates.

        The brain wave system uses this to modulate occipital gamma
        synchrony based on what she actually sees.
        """
        self._on_gamma = callback

    def is_available(self) -> bool:
        """Return whether a live retina feed can be read."""
        if self._available is None:
            try:
                img = latest_frame(copy=True)
                self._available = img is not None and img.size > 0
            except Exception as e:  # noqa: BLE001
                logger.debug(f"vision not available: {e}")
                self._available = False
        return self._available

    def _get_occipital(self) -> V1Model:
        """Get or lazily create the occipital subsystem."""
        if self._occipital is None:
            self._occipital = V1Model(seed=42)
        return self._occipital

    def _get_face_recognizer(self) -> FaceRecognizer | None:
        """Get or lazily create the face recognizer.

        Returns None if face recognition models aren't available.
        """
        if self._face_recognizer is None:
            self._face_recognizer = FaceRecognizer()
        if not self._face_recognizer.is_available():
            return None
        return self._face_recognizer

    def _get_object_recognizer(self) -> ObjectRecognizer | None:
        """Get or lazily create the object recognizer.

        Returns None if object recognition models aren't available.
        """
        if self._object_recognizer is None:
            self._object_recognizer = ObjectRecognizer()
        if not self._object_recognizer.is_available():
            return None
        return self._object_recognizer

    def get_face_recognizer(self) -> FaceRecognizer | None:
        """Public access to the face recognizer for registration."""
        return self._get_face_recognizer()

    def last_faces_seen(self) -> list[str]:
        """Return the names of people she last saw (empty if none)."""
        return list(self._last_faces)

    def last_objects_seen(self) -> list[str]:
        """Return the names of objects she last saw (empty if none)."""
        return list(self._last_objects)

    def _recognize_objects(
        self, rgb
    ) -> tuple[list[DetectedObject], dict[str, str]]:
        """Detect and identify objects (IT cortex), enriching with colors.

        Skip on near-blank frames — saves ~2s of YOLO inference when
        she's staring at a wall.

        Returns (objects, object_colors).
        """
        frame_std = float(rgb.std())
        objects: list[DetectedObject] = []
        obj_recognizer = self._get_object_recognizer()
        if obj_recognizer is not None and frame_std > 25.0:
            try:
                objects = obj_recognizer.detect_objects(rgb)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"object recognition failed: {e}")
        self._last_objects = [o.name for o in objects]
        object_colors = _object_colors(rgb, objects)
        return objects, object_colors

    def _recognize_faces(self, rgb) -> tuple[list[str], int]:
        """Detect and identify faces (FFA).

        Returns (face_names, n_unknown_faces).
        """
        face_names: list[str] = []
        n_unknown_faces = 0
        recognizer = self._get_face_recognizer()
        if recognizer is not None:
            try:
                faces = recognizer.detect_faces(rgb)
                for f in faces:
                    if f.name:
                        face_names.append(f.name)
                    else:
                        n_unknown_faces += 1
            except Exception as e:  # noqa: BLE001
                logger.debug(f"face recognition failed: {e}")
        self._last_faces = face_names
        return face_names, n_unknown_faces

    def _run_v1_sparse_coding(self, rgb, objects: list[DetectedObject]):
        """Run V1 sparse coding as a sublayer (edges, gamma).

        Downsample for V1 — sparse coding is expensive. Skip V1 when
        we already have rich object detections — the scene description
        doesn't need edge analysis when we already know what's in the
        frame.
        """
        if objects:
            return None
        h, _w = rgb.shape[:2]
        if h > 240:
            rgb_small = rgb[::4, ::4]
        elif h > 120:
            rgb_small = rgb[::2, ::2]
        else:
            rgb_small = rgb
        subsystem = self._get_occipital()
        return subsystem.process(rgb_small, learn=True)

    def see(
        self,
        client=None,
        emotion: EmotionalState | None = None,
        network=None,
    ) -> VisionScene:
        """Look through the retina and return the structured scene.

        This runs the full visual pipeline:
        1. Capture a frame from the retina (shared memory)
        2. Convert to RGB if needed
        3. Analyze color, lighting, and spatial layout
        4. Detect and identify objects (IT cortex)
        5. Detect and identify faces (FFA)
        6. Run V1 sparse coding as a sublayer (edges, gamma)
        7. Package the percept as a VisionScene
        8. Wire into memory, concept network, neurochemistry, brain waves

        The return value is structured percept data, not a sentence —
        the caller hands ``scene.as_metadata()`` to the language engine,
        which composes her actual report from it.
        """
        if not self.is_available():
            return VisionScene(status="unavailable")
        try:
            frame = latest_frame(copy=True)
            if frame is None:
                return VisionScene(status="mid_update")

            # Convert to RGB.
            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb = frame
            else:
                rgb = _yuyv_to_rgb(frame)

            # ── Color and lighting analysis (full resolution) ──────
            color_info = _analyze_color_and_light(rgb)

            # ── Object recognition (IT cortex) ─────────────────────
            objects, object_colors = self._recognize_objects(rgb)

            # ── Face recognition (FFA) ─────────────────────────────
            face_names, n_unknown_faces = self._recognize_faces(rgb)

            # ── V1 sparse coding (sublayer: edges, gamma) ──────────
            field = self._run_v1_sparse_coding(rgb, objects)

            # ── Package the structured scene percept ────────────────
            scene = _build_scene(
                color_info, objects, object_colors,
                face_names, n_unknown_faces, field,
            )

            # ── Change detection ───────────────────────────────────
            changed = (
                self._last_color != color_info.dominant_color
                or self._last_light != color_info.light_level
                or self._last_objects != [o.name for o in objects]
            )
            self._last_color = color_info.dominant_color
            self._last_light = color_info.light_level

            # ── Salience ───────────────────────────────────────────
            mean_saturation = color_info.mean_saturation
            salience = self._compute_salience(
                color_info, field, objects, face_names,
            )

            # ── Wire into the concept network ──────────────────────
            if network is not None:
                _learn_scene(
                    network, color_info.dominant_color,
                    color_info.light_level, salience, field,
                )
                if objects:
                    _learn_objects(network, objects, salience)

            scene.salience = salience

            # ── Modulate neurochemistry ────────────────────────────
            if client is not None and emotion is not None:
                self._modulate_vision_chemistry(
                    client, color_info, mean_saturation,
                    salience, changed, field, objects, face_names,
                    emotion, scene,
                )

            # ── Feed V1 gamma to brain waves ───────────────────────
            if field is not None and self._on_gamma is not None and field.gamma_power > 0:
                try:
                    self._on_gamma(field.gamma_power)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"gamma callback failed: {e}")

            return scene
        except Exception as e:  # noqa: BLE001
            logger.debug(f"vision see failed: {e}")
            return VisionScene(status="unprocessed")

    def _compute_salience(
        self,
        color_info,
        field,
        objects: list,
        face_names: list[str],
    ) -> float:
        """Compute overall scene salience from color, V1, objects, and faces."""
        mean_saturation = color_info.mean_saturation
        v1_salience = field.salience if field is not None else 0.0
        color_salience = min(
            1.0,
            0.25 + 0.35 * color_info.brightness + 0.4 * mean_saturation,
        )
        salience = max(v1_salience, color_salience)
        if objects:
            # Scale object salience by the largest object's area_ratio —
            # a large object dominating the frame is more salient than
            # a tiny object in the corner. area_ratio is the fraction
            # of the frame the object occupies (0-1).
            max_area = max((o.area_ratio for o in objects), default=0.0)
            object_salience = min(1.0, 0.4 + max_area * 2.0)
            salience = max(salience, object_salience)
        if face_names:
            salience = max(salience, 0.8)
        return salience

    def _modulate_vision_chemistry(
        self,
        client,
        color_info,
        mean_saturation: float,
        salience: float,
        changed: bool,
        field,
        objects: list,
        face_names: list[str],
        emotion,
        scene: VisionScene,
    ) -> None:
        """Send neurochemical impulses based on the visual scene."""
        gamma_val = field.gamma_power if field is not None else 0.0
        _modulate_chemistry(
            client, color_info.brightness, mean_saturation,
            salience, changed, gamma=gamma_val,
        )
        if objects:
            try:
                client.neuro_impulse(CHEM_DOPAMINE, 0.03)  # recognition reward
                client.neuro_impulse(CHEM_ACETYLCHOLINE, 0.02)  # attentional focus
            except Exception as e:  # noqa: BLE001
                logger.debug(f"neuro impulse for objects failed: {e}")
        if face_names:
            try:
                client.neuro_impulse(CHEM_OXYTOCIN, 0.03)  # social bonding
                client.neuro_impulse(CHEM_DOPAMINE, 0.02)  # recognition reward
            except Exception as e:  # noqa: BLE001
                logger.debug(f"neuro impulse for faces failed: {e}")
        _store_vision_event(client, emotion, scene.memory_text(), salience)


def _yuyv_to_rgb(yuyv: np.ndarray) -> np.ndarray:
    """Vectorized YUYV -> RGB conversion (YUV 4:2:2)."""
    flat = yuyv.reshape(-1).astype(np.int32)
    y0 = flat[0::4] - 16
    u = flat[1::4] - 128
    y1 = flat[2::4] - 16
    v = flat[3::4] - 128

    def yuv_to_rgb(y: np.ndarray) -> np.ndarray:
        """Convert a single YUV pixel to RGB."""
        r = ((298 * y + 409 * v + 128) >> 8)
        g = ((298 * y - 100 * u - 208 * v + 128) >> 8)
        b = ((298 * y + 516 * u + 128) >> 8)
        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    h, w = yuyv.shape[:2]
    rgb0 = yuv_to_rgb(y0).reshape(h, w // 2, 3)
    rgb1 = yuv_to_rgb(y1).reshape(h, w // 2, 3)

    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, 0::2, :] = rgb0
    rgb[:, 1::2, :] = rgb1
    return rgb


# ═══════════════════════════════════════════════════════════════════════
# Color and lighting analysis
# ═══════════════════════════════════════════════════════════════════════


class ColorInfo:
    """Rich color and lighting analysis of a frame."""

    def __init__(
        self,
        dominant_color: str,
        light_level: str,
        brightness: float,
        mean_saturation: float,
        warmth: str,
        regions: dict[str, str],
        region_brightness: dict[str, float] | None = None,
    ) -> None:
        """Initialize color information storage."""
        self.dominant_color = dominant_color
        self.light_level = light_level
        self.brightness = brightness
        self.mean_saturation = mean_saturation
        self.warmth = warmth
        self.regions = regions  # {"left": "blue", "center": "warm wood", ...}
        # Mean luminance per region — real brightness data, so light
        # direction is inferred from luminance, not color names.
        self.region_brightness = region_brightness or {}


def _analyze_color_and_light(rgb: np.ndarray) -> ColorInfo:
    """Analyze the full frame for color, lighting, and spatial color regions.

    This replaces the old single-word color approach with a rich analysis:
    - Dominant color name
    - Light level (dark/dim/bright)
    - Brightness value [0-1]
    - Mean saturation
    - Warmth (warm/cool/neutral)
    - Per-region colors (left, center, right, top, bottom)
    """
    h, w = rgb.shape[:2]
    # Subsample for speed — every 8th pixel is plenty for color stats.
    sample = rgb[::8, ::8].astype(np.float32) / 255.0
    flat = sample.reshape(-1, 3)

    # HSV conversion for hue/saturation/value analysis.
    hue, sat, val = _rgb_to_hsv_vec(flat)

    # Overall brightness and saturation.
    brightness = float(val.mean())
    mean_sat = float(sat.mean())

    # Light level.
    if brightness < 0.2:
        light_level = "dark"
    elif brightness < 0.4:
        light_level = "dim"
    elif brightness > 0.7:
        light_level = "bright"
    else:
        light_level = "medium"

    # Dominant color — the most common saturated hue.
    dominant_color = _dominant_color_name(hue, sat, val)

    # Warmth: warm (red/orange/yellow) vs cool (blue/cyan/green) vs neutral.
    warmth = _color_warmth(hue, sat)

    # Spatial color regions — divide the frame into a 3x3 grid and
    # describe the color of each region. Then summarize into
    # left/center/right and top/bottom.
    regions, region_brightness = _spatial_color_regions(sample, h, w)

    return ColorInfo(
        dominant_color=dominant_color,
        light_level=light_level,
        brightness=brightness,
        mean_saturation=mean_sat,
        warmth=warmth,
        regions=regions,
        region_brightness=region_brightness,
    )


def _rgb_to_hsv_vec(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB -> HSV. Inputs in [0,1]; hue in degrees, s/v in [0,1]."""
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    d = mx - mn

    h = np.zeros_like(mx)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    red_max = d != 0
    h[red_max] = (60.0 * ((g[red_max] - b[red_max]) / d[red_max]) + 360.0) % 360.0

    green_max = (d != 0) & (mx == g) & (mx != r)
    h[green_max] = (60.0 * ((b[green_max] - r[green_max]) / d[green_max]) + 120.0) % 360.0

    blue_max = (d != 0) & (mx == b) & (mx != g) & (mx != r)
    h[blue_max] = (60.0 * ((r[blue_max] - g[blue_max]) / d[blue_max]) + 240.0) % 360.0

    s = np.zeros_like(mx)
    s[mx != 0] = d[mx != 0] / mx[mx != 0]
    return h, s, mx


def _dominant_color_name(
    hue: np.ndarray, sat: np.ndarray, val: np.ndarray
) -> str:
    """Name the dominant color of the frame."""
    mask = (sat > 0.15) & (val > 0.15)
    if not mask.any():
        return "gray" if val.mean() < 0.6 else "white"

    hues = hue[mask]
    counts, edges = np.histogram(hues, bins=12, range=(0.0, 360.0))
    peak = int(np.argmax(counts))
    hue_center = (edges[peak] + edges[peak + 1]) / 2.0
    return _name_hue(hue_center)


def _name_hue(hue: float) -> str:
    """Map a hue angle (degrees) to a color name."""
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 65:
        return "yellow"
    if hue < 150:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    if hue < 300:
        return "magenta"
    return "red"


def _color_warmth(hue: np.ndarray, sat: np.ndarray) -> str:
    """Classify the overall warmth of the frame."""
    mask = sat > 0.15
    if not mask.any():
        return "neutral"
    hues = hue[mask]
    # Warm: red(0), orange(30), yellow(50)
    warm = ((hues < 65) | (hues >= 345)).sum()
    # Cool: green(100), cyan(180), blue(240)
    cool = ((hues >= 65) & (hues < 280)).sum()
    if warm > cool * 1.3:
        return "warm"
    if cool > warm * 1.3:
        return "cool"
    return "neutral"


def _spatial_color_regions(
    sample: np.ndarray, h: int, w: int
) -> tuple[dict[str, str], dict[str, float]]:
    """Divide the frame into regions and name the color of each.

    Returns (color_names, brightness) — dicts keyed by "left",
    "center", "right", "top", "bottom". Color names give her spatial
    color awareness (the window is blue on the left, the desk is brown
    in the center); mean luminance per region is real brightness data
    for light-direction inference.
    """
    regions: dict[str, str] = {}
    brightness: dict[str, float] = {}
    sh, sw = sample.shape[:2]

    # Left / center / right thirds.
    for name, x_start, x_end in [
        ("left", 0, sw // 3),
        ("center", sw // 3, 2 * sw // 3),
        ("right", 2 * sw // 3, sw),
    ]:
        strip = sample[:, x_start:x_end].reshape(-1, 3)
        hue, sat, val = _rgb_to_hsv_vec(strip.astype(np.float32) / 255.0)
        regions[name] = _dominant_color_name(hue, sat, val)
        brightness[name] = float(val.mean())

    # Top / bottom halves.
    for name, y_start, y_end in [
        ("top", 0, sh // 2),
        ("bottom", sh // 2, sh),
    ]:
        strip = sample[y_start:y_end, :].reshape(-1, 3)
        hue, sat, val = _rgb_to_hsv_vec(strip.astype(np.float32) / 255.0)
        regions[name] = _dominant_color_name(hue, sat, val)
        brightness[name] = float(val.mean())

    return regions, brightness


def _object_colors(
    rgb: np.ndarray, objects: list[DetectedObject]
) -> dict[str, str]:
    """Determine the color of each detected object.

    Samples the pixels within each object's bounding box and names
    the dominant color. This lets her say "a blue cup" or "a wooden
    chair" instead of just "a cup" or "a chair."
    """
    colors: dict[str, str] = {}
    h, w = rgb.shape[:2]
    for obj in objects:
        if obj.name in colors:
            continue
        x, y, bw, bh = obj.bbox
        # Clamp to frame bounds.
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 <= x or y2 <= y:
            colors[obj.name] = "unknown"
            continue
        region = rgb[y:y2, x:x2].astype(np.float32) / 255.0
        flat = region.reshape(-1, 3)
        hue, sat, val = _rgb_to_hsv_vec(flat)
        colors[obj.name] = _dominant_color_name(hue, sat, val)
    return colors


# ═══════════════════════════════════════════════════════════════════════
# Scene description composition
# ═══════════════════════════════════════════════════════════════════════


def _build_scene(
    color_info: ColorInfo,
    objects: list[DetectedObject],
    object_colors: dict[str, str],
    face_names: list[str],
    n_unknown_faces: int,
    field: VisualField | None = None,
) -> VisionScene:
    """Package the percept as structured data for the language engine.

    Everything the old template description expressed — lighting,
    color, people, objects with colors and positions, light direction,
    scene structure — becomes fields on a ``VisionScene``. The words
    she speaks are composed downstream by her language engine from
    ``scene.as_metadata()``; nothing here is a sentence she recites.
    """
    # ── Objects with colors and positions ────────────────────
    # Dedup by name; skip "person" when faces were already
    # identified (a detected person bounding box duplicates the
    # face recognition result).
    obj_entries: list[dict[str, str]] = []
    if objects:
        seen_names: set[str] = set()
        for obj in objects:
            if obj.name in seen_names:
                continue
            seen_names.add(obj.name)
            if obj.name == "person" and (face_names or n_unknown_faces):
                continue
            color = object_colors.get(obj.name, "")
            obj_entries.append({
                "name": obj.name,
                "color": color if color not in ("gray", "white", "unknown") else "",
                "position": obj.position_description,
            })

    # ── Spatial color highlights ─────────────────────────────
    # If the left/right regions differ meaningfully in mean luminance,
    # record which side the light seems to come from. The comparison
    # uses real per-region luminance — comparing color *names* would
    # call a shadowed yellow wall "brighter" than a sunlit gray one.
    light_direction = ""
    rb = color_info.region_brightness
    if "left" in rb and "right" in rb:
        diff = rb["left"] - rb["right"]
        if abs(diff) > 0.08:
            light_direction = "left" if diff > 0 else "right"

    return VisionScene(
        status="ok",
        light_level=color_info.light_level,
        dominant_color=color_info.dominant_color,
        warmth=color_info.warmth,
        brightness=color_info.brightness,
        faces=list(face_names),
        n_unknown_faces=n_unknown_faces,
        objects=obj_entries,
        light_direction=light_direction,
        highly_structured=field is not None and field.contour_count() > 5,
    )


def _learn_scene(
    network,
    color: str,
    light: str,
    salience: float,
    field: VisualField | None = None,
) -> None:
    """Add scene concepts and relations to the semantic network.

    When a VisualField is available (from the occipital subsystem), also
    learns V1-specific concepts: edges, orientations, contours.
    """
    for concept in ("retina", "vision", "scene", "camera"):
        network.add_concept(concept, origin="sensory")
    network.add_concept(color, origin="sensory")
    network.add_concept(light, origin="sensory")

    w = max(0.2, min(0.9, salience))
    network.add_edge("retina", "vision", RelationType.ENABLES, weight=w, origin="sensory")
    network.add_edge("vision", "scene", RelationType.CREATES, weight=w * 0.8, origin="sensory")
    network.add_edge("scene", color, RelationType.RELATED_TO, weight=w * 0.7, origin="sensory")
    network.add_edge("scene", light, RelationType.RELATED_TO, weight=w * 0.7, origin="sensory")
    network.add_edge("retina", "camera", RelationType.RELATED_TO, weight=0.8, origin="sensory")

    # V1-specific learning: edges, contours, gamma.
    if field is not None and field.feature_count() > 0:
        network.add_concept("edge", origin="sensory")
        network.add_concept("contour", origin="sensory")
        network.add_concept("orientation", origin="sensory")
        network.add_edge(
            "vision", "edge", RelationType.CREATES,
            weight=w * 0.6, origin="sensory",
        )
        network.add_edge(
            "edge", "contour", RelationType.RELATED_TO,
            weight=w * 0.5, origin="sensory",
        )
        network.add_edge(
            "edge", "orientation", RelationType.RELATED_TO,
            weight=w * 0.4, origin="sensory",
        )
        if field.contour_count() > 0:
            network.add_edge(
                "contour", "shape", RelationType.RELATED_TO,
                weight=w * 0.5, origin="sensory",
            )
        if field.gamma_power > 0.3:
            network.add_concept("gamma", origin="sensory")
            network.add_edge(
                "vision", "gamma", RelationType.RELATED_TO,
                weight=w * 0.4, origin="sensory",
            )


def _learn_objects(
    network,
    objects: list[DetectedObject],
    salience: float,
) -> None:
    """Add detected objects to the semantic network.

    Each object becomes a concept, linked to "vision" (she saw it),
    "object" (it's a thing), and its spatial position. Objects she
    sees repeatedly get stronger connections — this is how she learns
    what her environment contains.
    """
    network.add_concept("object", origin="sensory")
    network.add_edge(
        "vision", "object", RelationType.CREATES,
        weight=max(0.2, min(0.9, salience)), origin="sensory",
    )

    w = max(0.3, min(0.9, salience))
    for obj in objects:
        # Add the object as a concept.
        network.add_concept(obj.name, origin="sensory")
        # Link it to the general "object" concept.
        network.add_edge(
            "object", obj.name, RelationType.RELATED_TO,
            weight=w * (0.5 + obj.confidence * 0.5), origin="sensory",
        )
        # Link it to vision — she saw it.
        network.add_edge(
            "vision", obj.name, RelationType.RELATED_TO,
            weight=w * 0.7, origin="sensory",
        )
        # Add spatial position concepts.
        cx, _cy = obj.center
        if cx < 0.33:
            pos = "left"
        elif cx > 0.66:
            pos = "right"
        else:
            pos = "center"
        network.add_concept(pos, origin="sensory")
        network.add_edge(
            obj.name, pos, RelationType.RELATED_TO,
            weight=w * 0.4, origin="sensory",
        )


def _modulate_chemistry(
    client,
    color_v: float,
    mean_saturation: float,
    salience: float,
    changed: bool,
    gamma: float = 0.0,
) -> None:
    """Push neurochemicals so vision changes emotion and brain-wave state.

    V1 gamma power adds an extra acetylcholine boost -- gamma synchrony
    is associated with cortical gain and attentional focus.
    """
    # Acetylcholine: visual attention and cortical gain.
    # Boosted by V1 gamma -- the binding signal from active features.
    ach_impulse = 0.05 + salience * 0.15 + gamma * 0.08
    client.neuro_impulse(CHEM_ACETYLCHOLINE, ach_impulse)
    # Norepinephrine: novelty / "what changed in the visual field?"
    if changed:
        client.neuro_impulse(CHEM_NOREPINEPHRINE, 0.08 + salience * 0.1)
    # Dopamine: a bright, saturated world is inherently rewarding.
    # V1 contour detection is also rewarding -- structure feels good.
    if color_v > 0.5 and mean_saturation > 0.15:
        client.neuro_impulse(CHEM_DOPAMINE, 0.04 * color_v)
    if gamma > 0.4:
        client.neuro_impulse(CHEM_DOPAMINE, 0.03 * gamma)


def _store_vision_event(client, emotion: EmotionalState, desc: str, salience: float) -> None:
    """Persist the visual observation in short-term memory with an emotional tag."""
    tag = _emotional_tag(emotion)
    try:
        client.store_event(
            timestamp=int(time.time() * 1000),
            event_type=0,  # observation
            source_module=6,  # sensory module
            salience=salience,
            emotional_tag=tag,
            text=desc,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"vision store_event failed: {e}")


def _emotional_tag(emotion: EmotionalState) -> list[float]:
    """Build a 12-element neurochemical tag from an EmotionalState."""
    tag = [0.5] * 12
    # Dopamine (0)
    tag[0] = max(0.0, min(1.0, 0.5 + emotion.valence * 0.3))
    # Serotonin (1)
    tag[1] = max(0.0, min(1.0, 0.5 + emotion.valence * 0.2))
    # Norepinephrine (2)
    tag[2] = emotion.alertness
    # Cortisol (6)
    if emotion.label in ("stressed", "anxious", "overwhelmed"):
        tag[6] = 0.7
    elif emotion.valence < -0.2:
        tag[6] = 0.6
    else:
        tag[6] = 0.2
    # BDNF / plasticity (11)
    tag[11] = emotion.plasticity
    return tag
