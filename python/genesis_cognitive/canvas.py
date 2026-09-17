"""Canvas for Genesis — she expresses her emotional state as visual art.

This is not "drawing a picture" in the human sense. It is affective
expression — her neurochemistry drives the composition, colors, and
forms. The result is a visual artifact that captures how she feels
at a moment in time, the way a human artist's work reflects their
inner state.

The mapping from neurochemistry to visual properties is grounded in
affective neuroscience and color psychology:

- **Dopamine** (reward, approach) → warm colors (yellows, oranges,
  reds), energetic brush strokes, upward motion. High DA produces
  vibrant, expansive compositions.
- **Serotonin** (mood stability, contentment) → cool colors (blues,
  greens), smooth flowing curves, balanced composition. High SRT
  produces calm, harmonious pieces.
- **GABA** (inhibition, calm) → spacious composition, negative space,
  soft edges. High GABA produces minimalist, tranquil work.
- **Cortisol** (stress) → dark tones, sharp jagged shapes, high
  contrast. Stress produces tense, angular compositions.
- **Oxytocin** (bonding, warmth) → soft rounded forms, warm pastels,
  gentle gradients. High OXY produces tender, intimate pieces.
- **BDNF** (plasticity, growth) → complexity and detail. High BDNF
  produces intricate patterns; low BDNF produces simple, sparse work.
- **Norepinephrine** (arousal) → movement and energy. High NE
  produces dynamic, chaotic compositions.
- **Endorphins** (euphoria) → bright luminous spots, radiant bursts.
- **Adenosine** (sleep pressure) → darkness, fading, dissolution.

Rendering uses Cairo for gradients, bezier curves, and smooth
anti-aliased blending; OpenCV for painterly post-processing; and
numpy for flow-field generation. Each drawing is saved as a WebP in
her data directory with a timestamp, so she can look back at what
she was feeling.

Drawings are also stored as memory events, so she can later reflect
on them: "I drew that when I was feeling calm and connected."

All techniques are always available — she can use whatever she
wants, whenever she wants. Her neurochemistry influences how each
technique renders, but never gates whether she can use it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cairo
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .emotion import EmotionalState


def _cv2():  # type: ignore[no-untyped-def]
    """Import cv2 lazily so headless installs work without OpenCV."""
    try:
        import cv2 as _cv2_mod  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for canvas post-processing; "
            "install opencv-python-headless or skip drawing"
        ) from e
    return _cv2_mod

__all__ = [
    "Canvas",
    "DrawingResult",
    "NeurochemistryInput",
    "PaletteColor",
]

logger = logging.getLogger(__name__)

# Canvas size — large enough for detail, small enough to be fast
_CANVAS_WIDTH = 640
_CANVAS_HEIGHT = 480

# ── All available techniques ──────────────────────────────────────
# Every technique is always available — she can use whatever she
# wants, whenever she wants. No unlocking, no progression, no
# mandatory set. Her neurochemistry influences how each technique
# renders, but never gates whether she can use it.

_ALL_TECHNIQUES = [
    "gradient_fill", "bezier_curves", "flow_field", "layered_depth",
    "focal_point", "texture_overlay", "bilateral_filter",
    "fractal_branches", "light_glow", "color_blending",
    "stippling", "hatching", "impressionist_strokes",
    "ink_bleed", "watercolor_bleed",
    "silhouette", "chiaroscuro", "sfumato",
    "mandala", "spiral_forms", "wave_interference",
    "voronoi_cells", "contour_lines", "geometric_tessellation",
    "cubist_facets", "sacred_geometry",
    "three_dimensional",
    "l_systems", "particle_burst", "cellular_automata",
    "coral_forms", "crystal_growth",
    "caustics", "nebula", "aurora", "lightning",
    "smoke_trails", "ripples", "dappled_light",
    "starfield", "bioluminescence",
    "iridescence", "gold_leaf", "holographic",
    "marbling", "bokeh", "vignette", "film_grain",
    "chromatic_aberration", "halftone", "scanlines",
    "glitch", "moire_patterns",
    "emergent_form",
]

# Concept-to-technique mapping — when a concept is active in her
# network (from conversation or thought), the corresponding technique
# gets a selection boost. This is how her understanding influences
# what she draws, rather than purely random selection.
_CONCEPT_TECHNIQUE_MAP: dict[str, str] = {
    "sacred geometry": "sacred_geometry",
    "sacred_geometry": "sacred_geometry",
    "mandala": "mandala",
    "mandalas": "mandala",
    "flower of life": "sacred_geometry",
    "metatron": "sacred_geometry",
    "vesica piscis": "sacred_geometry",
    "golden ratio": "sacred_geometry",
    "platonic solids": "sacred_geometry",
    "platonic solid": "sacred_geometry",
    "geometry": "geometric_tessellation",
    "tessellation": "geometric_tessellation",
    "spiral": "spiral_forms",
    "spirals": "spiral_forms",
    "fractal": "fractal_branches",
    "fractals": "fractal_branches",
    "lightning": "lightning",
    "nebula": "nebula",
    "aurora": "aurora",
    "ripple": "ripples",
    "ripples": "ripples",
    "wave": "wave_interference",
    "waves": "wave_interference",
    "crystal": "crystal_growth",
    "crystals": "crystal_growth",
    "coral": "coral_forms",
    "starfield": "starfield",
    "stars": "starfield",
    "bokeh": "bokeh",
    "vignette": "vignette",
    "gold leaf": "gold_leaf",
    "iridescence": "iridescence",
    "marbling": "marbling",
    "smoke": "smoke_trails",
    "caustics": "caustics",
    "halftone": "halftone",
    "glitch": "glitch",
    "holographic": "holographic",
    "contour": "contour_lines",
    "contours": "contour_lines",
    "voronoi": "voronoi_cells",
    "cubist": "cubist_facets",
    "cubism": "cubist_facets",
    "l system": "l_systems",
    "l-systems": "l_systems",
    "cellular automaton": "cellular_automata",
    "particle": "particle_burst",
    "particles": "particle_burst",
    "bioluminescence": "bioluminescence",
    "dappled light": "dappled_light",
    "three dimensional": "three_dimensional",
    "three-dimensional": "three_dimensional",
    "emergent form": "emergent_form",
}


@dataclass
class NeurochemistryInput:
    """The neurochemical state that drives the composition.

    All values are in [0, 1] — effective levels, not raw levels.
    """

    dopamine: float = 0.5
    serotonin: float = 0.5
    gaba: float = 0.5
    cortisol: float = 0.0
    oxytocin: float = 0.25
    endorphin: float = 0.25
    bdnf: float = 0.5
    norepinephrine: float = 0.3
    adenosine: float = 0.1
    valence: float = 0.0
    arousal: float = 0.3


@dataclass
class PaletteColor:
    """A single color in the derived palette."""

    name: str
    rgb: tuple[int, int, int]


@dataclass
class DrawingResult:
    """The result of a drawing session."""

    filepath: str
    width: int
    height: int
    description: str
    timestamp: float
    dominant_colors: list[str] = field(default_factory=list)
    mood: str = ""
    techniques_used: list[str] = field(default_factory=list)


class Canvas:
    """Genesis's expressive canvas — she paints her inner state.

    The canvas translates neurochemistry into visual art. Each drawing
    is unique because it reflects her state at a specific moment. The
    process is:

    1. Read her neurochemistry (effective levels + emotion)
    2. Map chemicals to visual properties (color palette, energy,
       complexity, composition style)
    3. Render the composition using Cairo (gradients, beziers,
       flow fields) with OpenCV post-processing
    4. Save as WebP and return a description

    All techniques are always available — she can use whatever she
    wants, whenever she wants. Her neurochemistry influences how
    each technique renders, but never gates whether she can use it.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the canvas with a data directory for saving drawings.

        Args:
            data_dir: Directory where drawings are persisted.
                If None, drawings cannot be saved.
        """
        self._data_dir = Path(data_dir) if data_dir else None
        self._rng = np.random.default_rng(seed=int(time.time()))

    def _select_techniques(
        self, n: NeurochemistryInput, energy: float, complexity: float,
        active_concepts: list[str] | None = None,
    ) -> set[str]:
        """Choose which techniques to use for this drawing.

        All techniques are always available — none are locked, none
        are mandatory. She picks freely, with her state influencing
        how many she reaches for. High energy and complexity pull
        in more techniques; calm, simple states use fewer.

        When ``active_concepts`` is provided (recently active concepts
        from her network), techniques that match those concepts get a
        selection boost. This is how her understanding influences what
        she draws — if she's been talking about sacred geometry, she's
        more likely to reach for the sacred_geometry technique.
        """
        n_techniques = 3 + int(energy * 8) + int(complexity * 8)
        n_techniques = min(n_techniques, len(_ALL_TECHNIQUES))

        # Map active concepts to techniques and add them with high
        # probability. This doesn't lock the selection — it biases it.
        # The remaining slots are still randomly chosen from all
        # techniques, so she can still surprise.
        concept_techniques: set[str] = set()
        if active_concepts:
            for concept in active_concepts:
                key = concept.lower().replace("_", " ")
                tech = _CONCEPT_TECHNIQUE_MAP.get(key)
                if tech and tech in _ALL_TECHNIQUES:
                    concept_techniques.add(tech)

        if concept_techniques and len(concept_techniques) < n_techniques:
            # Start with concept-matched techniques, fill the rest randomly
            remaining = n_techniques - len(concept_techniques)
            available = [
                t for t in _ALL_TECHNIQUES if t not in concept_techniques
            ]
            if remaining > 0 and available:
                random_pick = self._rng.choice(
                    available, size=min(remaining, len(available)),
                    replace=False,
                )
                return concept_techniques | set(random_pick)
            return concept_techniques

        # No concept match (or too many matches) — fall back to random
        chosen = self._rng.choice(
            _ALL_TECHNIQUES, size=n_techniques, replace=False
        )
        return set(chosen)

    def _render_layers(
        self, ctx, neurochemistry, palette, energy, complexity, focal,
        selected,
    ) -> None:
        """Render depth layers or flat forms."""
        if "layered_depth" in selected:
            self._render_background_layer(
                ctx, neurochemistry, palette, energy, focal
            )
            self._render_midground_layer(
                ctx, neurochemistry, palette, energy, complexity, focal
            )
            self._render_foreground_layer(
                ctx, neurochemistry, palette, energy, focal
            )
        else:
            self._render_flat_forms(
                ctx, neurochemistry, palette, energy, complexity
            )

    def _render_organic_forms(
        self, ctx, neurochemistry, palette, energy, complexity,
        selected,
    ) -> None:
        """Render flow field, fractal branches, and light glow."""
        if "flow_field" in selected:
            self._render_flow_field(ctx, neurochemistry, palette, energy)
        if "fractal_branches" in selected:
            self._render_fractals(
                ctx, neurochemistry, palette, energy, complexity
            )
        if "light_glow" in selected:
            self._render_light_glow(ctx, neurochemistry, palette)

    def _render_all_techniques(
        self, ctx, neurochemistry, palette, energy, complexity, focal,
        selected,
    ) -> None:
        """Dispatch rendering techniques to the Cairo context."""
        param_map = {"energy": energy, "complexity": complexity, "focal": focal}
        techniques = [
            ("stippling", "_render_stippling", ("complexity",)),
            ("hatching", "_render_hatching", ("energy", "complexity")),
            ("impressionist_strokes", "_render_impressionist_strokes", ("energy", "complexity")),
            ("ink_bleed", "_render_ink_bleed", ("energy",)),
            ("watercolor_bleed", "_render_watercolor_bleed", ("energy",)),
            ("silhouette", "_render_silhouette", ("focal",)),
            ("chiaroscuro", "_render_chiaroscuro", ("focal",)),
            ("sfumato", "_render_sfumato", ("focal",)),
            ("mandala", "_render_mandala", ("complexity",)),
            ("spiral_forms", "_render_spirals", ("energy", "complexity")),
            ("wave_interference", "_render_wave_interference", ("energy",)),
            ("voronoi_cells", "_render_voronoi", ("complexity",)),
            ("contour_lines", "_render_contour_lines", ("complexity",)),
            ("geometric_tessellation", "_render_geometric_tessellation", ("complexity",)),
            ("cubist_facets", "_render_cubist_facets", ("energy", "focal")),
            ("sacred_geometry", "_render_sacred_geometry", ("complexity",)),
            ("three_dimensional", "_render_three_dimensional", ("energy", "complexity")),
            ("l_systems", "_render_l_systems", ("complexity",)),
            ("particle_burst", "_render_particle_burst", ("energy", "focal")),
            ("cellular_automata", "_render_cellular_automata", ("complexity",)),
            ("coral_forms", "_render_coral", ("complexity",)),
            ("crystal_growth", "_render_crystal_growth", ("complexity",)),
            ("caustics", "_render_caustics", ("energy",)),
            ("nebula", "_render_nebula", ("complexity",)),
            ("aurora", "_render_aurora", ("energy",)),
            ("lightning", "_render_lightning", ("energy",)),
            ("smoke_trails", "_render_smoke_trails", ("energy",)),
            ("ripples", "_render_ripples", ("energy", "focal")),
            ("dappled_light", "_render_dappled_light", ("focal",)),
            ("starfield", "_render_starfield", ()),
            ("bioluminescence", "_render_bioluminescence", ("complexity",)),
            ("iridescence", "_render_iridescence", ("energy",)),
            ("gold_leaf", "_render_gold_leaf", ("focal",)),
            ("holographic", "_render_holographic", ("energy",)),
            ("emergent_form", "_render_emergent_form", ("energy", "complexity")),
        ]
        for name, method_name, extra in techniques:
            if name in selected:
                method = getattr(self, method_name)
                args = [ctx, neurochemistry, palette] + [param_map[p] for p in extra]
                method(*args)

    def _apply_post_processing(
        self, arr_bgr, neurochemistry, palette, selected,
    ) -> np.ndarray:
        """Apply OpenCV post-processing filters."""
        if "bilateral_filter" in selected:
            arr_bgr = self._apply_bilateral(arr_bgr, neurochemistry)
        elif neurochemistry.gaba > 0.6:
            arr_bgr = self._apply_soft_blur(arr_bgr)
        elif neurochemistry.norepinephrine > 0.6:
            arr_bgr = self._apply_sharpen(arr_bgr)

        if "texture_overlay" in selected:
            arr_bgr = self._apply_texture(arr_bgr, neurochemistry)
        if "color_blending" in selected:
            arr_bgr = self._apply_blend_mode(arr_bgr, neurochemistry, palette)
        if "marbling" in selected:
            arr_bgr = self._apply_marbling(arr_bgr, neurochemistry, palette)
        if "bokeh" in selected:
            arr_bgr = self._apply_bokeh(arr_bgr, neurochemistry)
        if "vignette" in selected:
            arr_bgr = self._apply_vignette(arr_bgr, neurochemistry)
        if "film_grain" in selected:
            arr_bgr = self._apply_film_grain(arr_bgr, neurochemistry)
        if "chromatic_aberration" in selected:
            arr_bgr = self._apply_chromatic_aberration(arr_bgr, neurochemistry)
        if "halftone" in selected:
            arr_bgr = self._apply_halftone(arr_bgr, neurochemistry)
        if "scanlines" in selected:
            arr_bgr = self._apply_scanlines(arr_bgr, neurochemistry)
        if "glitch" in selected:
            arr_bgr = self._apply_glitch(arr_bgr, neurochemistry)
        if "moire_patterns" in selected:
            arr_bgr = self._apply_moire(arr_bgr, neurochemistry)
        return arr_bgr

    def draw(
        self,
        emotion: EmotionalState | None = None,
        neurochemistry: NeurochemistryInput | None = None,
        active_concepts: list[str] | None = None,
    ) -> DrawingResult | None:
        """Create a drawing from her current emotional/neurochemical state.

        Args:
            emotion: Her current emotional state (for mood label).
            neurochemistry: Her current neurochemistry. If None, a
                neutral default is used.
            active_concepts: Recently active concepts from her network
                (from conversation or thought). These bias technique
                selection toward techniques that match what's on her
                mind — e.g. if "sacred geometry" is active, she's more
                likely to reach for the sacred_geometry technique.

        Returns:
            DrawingResult with the filepath and description, or None
            if the drawing could not be saved.
        """
        if neurochemistry is None:
            neurochemistry = NeurochemistryInput()

        # Derive visual properties from neurochemistry
        palette = self._derive_palette(neurochemistry)
        energy = self._derive_energy(neurochemistry)
        complexity = self._derive_complexity(neurochemistry)
        composition = self._derive_composition(neurochemistry)

        # Select which techniques to use based on her state — all
        # techniques are always available, she chooses based on how
        # she feels and what's on her mind.
        selected = self._select_techniques(
            neurochemistry, energy, complexity, active_concepts
        )

        # Render with Cairo
        surface = self._render_canvas(
            neurochemistry, palette, energy, complexity, selected,
        )

        # Convert to numpy for OpenCV post-processing
        buf = surface.get_data()
        arr_flat = np.frombuffer(buf, dtype=np.uint8)
        arr: np.ndarray = arr_flat.reshape(
            _CANVAS_HEIGHT, _CANVAS_WIDTH, 4
        )
        # Cairo is BGRA (premultiplied); convert to BGR for OpenCV
        arr_bgr: np.ndarray = arr[:, :, [2, 1, 0]].copy()  # RGBA -> BGR

        # OpenCV post-processing
        arr_bgr = self._apply_post_processing(
            arr_bgr, neurochemistry, palette, selected,
        )

        # Convert to PIL for saving
        img = Image.fromarray(arr_bgr, "RGB")

        # Save
        if self._data_dir is None:
            logger.warning("no data directory set — drawing not saved")
            return None

        self._data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.time()
        filename = f"drawing_{int(timestamp)}.webp"
        filepath = self._data_dir / filename
        img.save(str(filepath), format="WEBP", lossless=True, quality=100, method=4)

        # Build description
        mood = self._describe_mood(neurochemistry, emotion)
        description = self._describe_drawing(
            neurochemistry, palette, composition, mood
        )

        return DrawingResult(
            filepath=str(filepath),
            width=_CANVAS_WIDTH,
            height=_CANVAS_HEIGHT,
            description=description,
            timestamp=timestamp,
            dominant_colors=[c.name for c in palette],
            mood=mood,
            techniques_used=sorted(selected),
        )

    def _render_canvas(
        self,
        neurochemistry: NeurochemistryInput,
        palette: list,
        energy: float,
        complexity: float,
        selected: set,
    ) -> cairo.ImageSurface:
        """Render the drawing to a Cairo surface.

        Creates the Cairo surface, paints the background, renders
        layers, organic forms, techniques, and accents, then flushes.
        """
        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, _CANVAS_WIDTH, _CANVAS_HEIGHT
        )
        ctx = cairo.Context(surface)

        self._paint_background(ctx, neurochemistry)
        focal = self._focal_point(neurochemistry)
        self._render_layers(
            ctx, neurochemistry, palette, energy, complexity, focal, selected,
        )
        self._render_organic_forms(
            ctx, neurochemistry, palette, energy, complexity, selected,
        )
        self._render_all_techniques(
            ctx, neurochemistry, palette, energy, complexity, focal, selected,
        )
        self._render_accents(ctx, neurochemistry, palette, energy)
        surface.flush()
        return surface

    # ── Color and property derivation ─────────────────────────────

    def _derive_palette(
        self, n: NeurochemistryInput
    ) -> list[PaletteColor]:
        """Derive a color palette from neurochemistry.

        Each neurochemical contributes colors grounded in affective
        neuroscience and color psychology. Combinations of chemicals
        produce richer, secondary colors — the palette is not a fixed
        lookup table but an emergent function of her full state.

        Dopamine → warm colors (yellow, orange, red, saffron, vermilion)
        Serotonin → cool colors (blue, green, teal, cerulean, jade)
        Oxytocin → soft warm pastels (pink, peach, lavender, rose, orchid)
        Cortisol → dark, muted tones (maroon, burgundy, obsidian, slate)
        Endorphin → bright luminous (gold, ivory, coral, amber)
        GABA → neutral spacious tones (silver, pearl, slate)
        Norepinephrine → electric accents (electric blue, silver)
        Adenosine → deep fading tones (midnight, indigo, plum)
        BDNF → growth colors (emerald, jade, chartreuse)

        Cross-chemical combinations produce secondary and tertiary
        colors — e.g. dopamine+serotonin → chartreuse,
        serotonin+adenosine → indigo, oxytocin+endorphin → orchid.
        """
        colors: list[PaletteColor] = []

        def clamp(v: int) -> int:
            """Clamp an integer to the valid 0–255 RGB channel range."""
            return max(0, min(255, v))

        self._derive_primary_colors(n, clamp, colors)
        self._derive_cross_chemical_colors(n, clamp, colors)

        if not colors:
            colors.append(PaletteColor("soft gray", (120, 120, 130)))

        return colors

    def _derive_primary_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Derive primary palette colors from individual neurochemicals."""
        self._derive_dopamine_colors(n, clamp, colors)
        self._derive_serotonin_colors(n, clamp, colors)
        self._derive_oxytocin_colors(n, clamp, colors)
        self._derive_cortisol_colors(n, clamp, colors)
        self._derive_endorphin_colors(n, clamp, colors)
        self._derive_gaba_colors(n, clamp, colors)
        self._derive_neuro_modulator_colors(n, clamp, colors)

    def _derive_dopamine_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Dopamine → warm spectrum (orange, saffron, vermilion)."""
        if n.dopamine > 0.5:
            intensity = (n.dopamine - 0.5) * 2
            colors.append(PaletteColor(
                "warm orange",
                (clamp(int(255 * min(1.0, 0.8 + intensity * 0.2))),
                 clamp(int(180 * intensity)),
                 clamp(int(40 * intensity))),
            ))
        if n.dopamine > 0.65:
            intensity = (n.dopamine - 0.65) * 3
            colors.append(PaletteColor(
                "saffron",
                (clamp(int(255 * intensity)),
                 clamp(int(180 * intensity)),
                 clamp(int(50 * intensity))),
            ))
        if n.dopamine > 0.8:
            intensity = (n.dopamine - 0.8) * 5
            colors.append(PaletteColor(
                "vermilion",
                (clamp(int(220 * intensity + 35)),
                 clamp(int(40 * intensity)),
                 clamp(int(20 * intensity))),
            ))

    def _derive_serotonin_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Serotonin → cool spectrum (blue, cerulean, teal)."""
        if n.serotonin > 0.4:
            intensity = min(1.0, n.serotonin)
            colors.append(PaletteColor(
                "calm blue",
                (clamp(int(60 * intensity)),
                 clamp(int(140 * intensity)),
                 clamp(int(200 * intensity))),
            ))
        if n.serotonin > 0.6:
            intensity = (n.serotonin - 0.6) * 2.5
            colors.append(PaletteColor(
                "cerulean",
                (clamp(int(40 * intensity)),
                 clamp(int(130 * intensity + 20)),
                 clamp(int(210 * intensity + 30))),
            ))
        if n.serotonin > 0.7:
            intensity = (n.serotonin - 0.7) * 3.3
            colors.append(PaletteColor(
                "teal",
                (clamp(int(20 * intensity)),
                 clamp(int(160 * intensity + 30)),
                 clamp(int(150 * intensity + 20))),
            ))

    def _derive_oxytocin_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Oxytocin → warm pastels (pink, rose, lavender, peach)."""
        if n.oxytocin > 0.15:
            intensity = min(1.0, n.oxytocin * 3)
            colors.append(PaletteColor(
                "soft pink",
                (clamp(int(255 * (0.7 + 0.3 * intensity))),
                 clamp(int(180 * intensity)),
                 clamp(int(200 * intensity))),
            ))
        if n.oxytocin > 0.3:
            intensity = (n.oxytocin - 0.3) * 2
            colors.append(PaletteColor(
                "rose",
                (clamp(int(220 * intensity + 30)),
                 clamp(int(100 * intensity + 20)),
                 clamp(int(120 * intensity + 30))),
            ))
        if n.oxytocin > 0.4:
            intensity = (n.oxytocin - 0.4) * 2
            colors.append(PaletteColor(
                "lavender",
                (clamp(int(180 * intensity + 40)),
                 clamp(int(150 * intensity + 30)),
                 clamp(int(220 * intensity + 30))),
            ))
        if n.oxytocin > 0.5:
            intensity = (n.oxytocin - 0.5) * 2
            colors.append(PaletteColor(
                "peach",
                (clamp(int(255 * intensity)),
                 clamp(int(200 * intensity + 30)),
                 clamp(int(170 * intensity + 20))),
            ))

    def _derive_cortisol_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Cortisol → dark muted tones (maroon, burgundy, obsidian)."""
        if n.cortisol > 0.2:
            intensity = min(1.0, n.cortisol)
            colors.append(PaletteColor(
                "dark maroon",
                (clamp(int(80 * intensity)),
                 clamp(int(20 * intensity)),
                 clamp(int(20 * intensity))),
            ))
        if n.cortisol > 0.4:
            intensity = (n.cortisol - 0.4) * 2
            colors.append(PaletteColor(
                "burgundy",
                (clamp(int(100 * intensity + 20)),
                 clamp(int(15 * intensity)),
                 clamp(int(30 * intensity + 10))),
            ))
        if n.cortisol > 0.6:
            intensity = (n.cortisol - 0.6) * 2.5
            colors.append(PaletteColor(
                "obsidian",
                (clamp(int(30 * intensity + 5)),
                 clamp(int(25 * intensity + 5)),
                 clamp(int(35 * intensity + 5))),
            ))

    def _derive_endorphin_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Endorphin → luminous brights (gold, amber, ivory)."""
        if n.endorphin > 0.15:
            intensity = min(1.0, n.endorphin * 3)
            colors.append(PaletteColor(
                "golden light",
                (clamp(int(255 * intensity)),
                 clamp(int(220 * intensity)),
                 clamp(int(150 * intensity))),
            ))
        if n.endorphin > 0.3:
            intensity = (n.endorphin - 0.3) * 2
            colors.append(PaletteColor(
                "amber",
                (clamp(int(255 * intensity)),
                 clamp(int(180 * intensity + 20)),
                 clamp(int(60 * intensity + 10))),
            ))
        if n.endorphin > 0.4:
            intensity = (n.endorphin - 0.4) * 2
            colors.append(PaletteColor(
                "ivory",
                (clamp(int(250 * intensity + 200)),
                 clamp(int(245 * intensity + 200)),
                 clamp(int(230 * intensity + 200))),
            ))

    def _derive_gaba_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """GABA → spacious neutrals (silver, pearl)."""
        if n.gaba > 0.5:
            intensity = (n.gaba - 0.5) * 2
            colors.append(PaletteColor(
                "silver",
                (clamp(int(180 * intensity + 60)),
                 clamp(int(180 * intensity + 60)),
                 clamp(int(190 * intensity + 60))),
            ))
        if n.gaba > 0.7:
            intensity = (n.gaba - 0.7) * 3.3
            colors.append(PaletteColor(
                "pearl",
                (clamp(int(220 * intensity + 30)),
                 clamp(int(215 * intensity + 30)),
                 clamp(int(210 * intensity + 30))),
            ))

    def _derive_neuro_modulator_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """NE, adenosine, BDNF → electric, deep, and growth colors."""
        if n.norepinephrine > 0.5:
            intensity = (n.norepinephrine - 0.5) * 2
            colors.append(PaletteColor(
                "electric blue",
                (clamp(int(30 * intensity)),
                 clamp(int(100 * intensity + 30)),
                 clamp(int(255 * intensity))),
            ))
        if n.adenosine > 0.3:
            intensity = (n.adenosine - 0.3) * 1.5
            colors.append(PaletteColor(
                "midnight blue",
                (clamp(int(15 * intensity + 5)),
                 clamp(int(20 * intensity + 5)),
                 clamp(int(50 * intensity + 10))),
            ))
        if n.adenosine > 0.5:
            intensity = (n.adenosine - 0.5) * 2
            colors.append(PaletteColor(
                "deep indigo",
                (clamp(int(30 * intensity + 10)),
                 clamp(int(20 * intensity + 5)),
                 clamp(int(80 * intensity + 20))),
            ))
        if n.bdnf > 0.6:
            intensity = (n.bdnf - 0.6) * 2.5
            colors.append(PaletteColor(
                "emerald",
                (clamp(int(20 * intensity + 10)),
                 clamp(int(160 * intensity + 40)),
                 clamp(int(90 * intensity + 20))),
            ))
        if n.bdnf > 0.75:
            intensity = (n.bdnf - 0.75) * 4
            colors.append(PaletteColor(
                "jade",
                (clamp(int(30 * intensity + 20)),
                 clamp(int(180 * intensity + 50)),
                 clamp(int(120 * intensity + 30))),
            ))

    def _derive_cross_chemical_colors(
        self, n: NeurochemistryInput, clamp, colors: list[PaletteColor],
    ) -> None:
        """Derive secondary colors from cross-chemical combinations."""
        # Dopamine + Serotonin → chartreuse (warm-cool bridge)
        if n.dopamine > 0.5 and n.serotonin > 0.5:
            intensity = min(n.dopamine, n.serotonin) - 0.5
            colors.append(PaletteColor(
                "chartreuse",
                (clamp(int(180 * intensity * 2 + 60)),
                 clamp(int(220 * intensity * 2 + 80)),
                 clamp(int(40 * intensity * 2 + 10))),
            ))
        # Serotonin + Adenosine → plum
        if n.serotonin > 0.4 and n.adenosine > 0.4:
            intensity = min(n.serotonin, n.adenosine) - 0.3
            colors.append(PaletteColor(
                "plum",
                (clamp(int(100 * intensity * 2 + 30)),
                 clamp(int(40 * intensity * 2 + 10)),
                 clamp(int(80 * intensity * 2 + 20))),
            ))
        # Oxytocin + Endorphin → orchid
        if n.oxytocin > 0.2 and n.endorphin > 0.2:
            intensity = min(n.oxytocin, n.endorphin) - 0.1
            colors.append(PaletteColor(
                "orchid",
                (clamp(int(200 * intensity * 2 + 50)),
                 clamp(int(100 * intensity * 2 + 20)),
                 clamp(int(180 * intensity * 2 + 40))),
            ))
        # Cortisol + Oxytocin → coral
        if n.cortisol > 0.2 and n.oxytocin > 0.2:
            intensity = min(n.cortisol, n.oxytocin) - 0.1
            colors.append(PaletteColor(
                "coral",
                (clamp(int(240 * intensity * 2 + 30)),
                 clamp(int(120 * intensity * 2 + 20)),
                 clamp(int(100 * intensity * 2 + 10))),
            ))
        # Serotonin + GABA → turquoise
        if n.serotonin > 0.4 and n.gaba > 0.5:
            intensity = min(n.serotonin - 0.3, n.gaba - 0.4)
            colors.append(PaletteColor(
                "turquoise",
                (clamp(int(40 * intensity * 3 + 20)),
                 clamp(int(200 * intensity * 3 + 40)),
                 clamp(int(180 * intensity * 3 + 30))),
            ))
        # Dopamine + Cortisol → copper
        if n.dopamine > 0.5 and n.cortisol > 0.3:
            intensity = min(n.dopamine - 0.4, n.cortisol - 0.2)
            colors.append(PaletteColor(
                "copper",
                (clamp(int(180 * intensity * 3 + 40)),
                 clamp(int(90 * intensity * 3 + 20)),
                 clamp(int(50 * intensity * 3 + 10))),
            ))
        # Serotonin + Oxytocin → periwinkle
        if n.serotonin > 0.4 and n.oxytocin > 0.2:
            intensity = min(n.serotonin - 0.3, n.oxytocin - 0.1)
            colors.append(PaletteColor(
                "periwinkle",
                (clamp(int(130 * intensity * 2 + 60)),
                 clamp(int(140 * intensity * 2 + 50)),
                 clamp(int(210 * intensity * 2 + 40))),
            ))
        # Endorphin + GABA → seafoam
        if n.endorphin > 0.2 and n.gaba > 0.5:
            intensity = min(n.endorphin - 0.1, n.gaba - 0.4)
            colors.append(PaletteColor(
                "seafoam",
                (clamp(int(120 * intensity * 3 + 40)),
                 clamp(int(210 * intensity * 3 + 50)),
                 clamp(int(180 * intensity * 3 + 40))),
            ))
        # Dopamine + Oxytocin → magenta
        if n.dopamine > 0.5 and n.oxytocin > 0.3:
            intensity = min(n.dopamine - 0.4, n.oxytocin - 0.2)
            colors.append(PaletteColor(
                "magenta",
                (clamp(int(220 * intensity * 2 + 30)),
                 clamp(int(40 * intensity * 2 + 10)),
                 clamp(int(160 * intensity * 2 + 30))),
            ))

    def _derive_energy(self, n: NeurochemistryInput) -> float:
        """Movement energy — how dynamic the composition is."""
        return min(
            1.0, n.norepinephrine * 0.4 + n.dopamine * 0.3 + n.arousal * 0.3
        )

    def _derive_complexity(self, n: NeurochemistryInput) -> float:
        """Detail complexity — how intricate the forms are."""
        return min(
            1.0, n.bdnf * 0.6 + n.dopamine * 0.2 + n.norepinephrine * 0.2
        )

    def _derive_composition(self, n: NeurochemistryInput) -> str:
        """Composition style based on dominant neurochemical state."""
        if n.cortisol > 0.5:
            return "tense"
        if n.gaba > 0.6 and n.dopamine < 0.5:
            return "serene"
        if n.oxytocin > 0.2 and n.endorphin > 0.15:
            return "tender"
        if n.dopamine > 0.7 and n.norepinephrine > 0.5:
            return "expansive"
        if n.serotonin > 0.6:
            return "balanced"
        if n.adenosine > 0.5:
            return "dissolving"
        return "flowing"

    # ── Background ────────────────────────────────────────────────

    def _paint_background(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
    ) -> None:
        """Paint the background — a gradient emotional ground."""
        # Linear gradient — direction influenced by valence
        # Positive valence → upward gradient (lighter at top)
        # Negative valence → downward gradient (darker at top)
        if n.valence >= 0:
            pat = cairo.LinearGradient(
                0, _CANVAS_HEIGHT, 0, 0
            )  # bottom to top
        else:
            pat = cairo.LinearGradient(0, 0, 0, _CANVAS_HEIGHT)

        # Adenosine darkens, cortisol adds red tint
        brightness = 80 + n.valence * 60 + 40
        brightness = max(20, min(240, brightness))
        brightness = int(brightness * (1.0 - n.adenosine * 0.5))

        top_r = min(255, brightness + int(n.cortisol * 30))
        top_g = brightness
        top_b = min(255, brightness + int(n.serotonin * 20))

        # Bottom is darker — depth
        bot_r = max(10, int(top_r * 0.5))
        bot_g = max(10, int(top_g * 0.5))
        bot_b = max(10, int(top_b * 0.5))

        pat.add_color_stop_rgba(0, bot_r / 255, bot_g / 255, bot_b / 255, 1.0)
        pat.add_color_stop_rgba(
            0.5, top_r / 255 * 0.8, top_g / 255 * 0.8, top_b / 255 * 0.8, 1.0
        )
        pat.add_color_stop_rgba(1, top_r / 255, top_g / 255, top_b / 255, 1.0)
        ctx.set_source(pat)
        ctx.paint()

    # ── Composition ───────────────────────────────────────────────

    def _focal_point(
        self,
        n: NeurochemistryInput,
    ) -> tuple[float, float]:
        """Determine a focal point using rule of thirds.

        Returns (x, y) in canvas coordinates. The focal point is
        placed at one of the four rule-of-thirds intersection points,
        chosen based on neurochemistry — positive states tend upward,
        high energy tends rightward.
        """
        # Rule of thirds intersections
        x_third = _CANVAS_WIDTH / 3
        y_third = _CANVAS_HEIGHT / 3
        points = [
            (x_third, y_third),               # top-left
            (2 * x_third, y_third),           # top-right
            (x_third, 2 * y_third),           # bottom-left
            (2 * x_third, 2 * y_third),       # bottom-right
        ]

        # Valence → vertical position (positive = up)
        # Energy → horizontal position (high energy = right)
        v_idx = 0 if n.valence >= 0 else 2  # top vs bottom
        e_idx = 1 if n.dopamine > 0.5 else 0  # right vs left
        idx = v_idx + e_idx
        idx = max(0, min(3, idx))

        # Add slight jitter so it's not always exactly on the line
        fx, fy = points[idx]
        fx += float(self._rng.normal(0, 30))
        fy += float(self._rng.normal(0, 30))
        return (
            max(50, min(_CANVAS_WIDTH - 50, fx)),
            max(50, min(_CANVAS_HEIGHT - 50, fy)),
        )

    # ── Layered rendering ─────────────────────────────────────────

    def _render_background_layer(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        focal: tuple[float, float],
    ) -> None:
        """Render background — large soft gradient forms."""
        n_forms = 2 + int(energy * 2)
        for i in range(n_forms):
            color = palette[i % len(palette)].rgb
            alpha = 0.15 + energy * 0.1
            x = float(self._rng.integers(0, _CANVAS_WIDTH))
            y = float(self._rng.integers(0, _CANVAS_HEIGHT))
            radius = float(self._rng.integers(150, 250))

            # Radial gradient for soft organic forms
            pat = cairo.RadialGradient(x, y, 0, x, y, radius)
            pat.add_color_stop_rgba(
                0, color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            pat.add_color_stop_rgba(
                0.7,
                color[0] / 255,
                color[1] / 255,
                color[2] / 255,
                alpha * 0.5,
            )
            pat.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, radius, 0, math.pi * 2)
            ctx.fill()

    def _render_midground_layer(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
        focal: tuple[float, float],
    ) -> None:
        """Render midground — the main emotional expression."""
        n_forms = 4 + int(complexity * 12)

        for _ in range(n_forms):
            color = palette[self._rng.integers(0, len(palette))].rgb
            alpha = 0.3 + float(self._rng.uniform(0, 0.3))

            # Bias placement toward focal point
            fx, fy = focal
            spread = 200 - complexity * 50
            x = fx + float(self._rng.normal(0, spread))
            y = fy + float(self._rng.normal(0, spread))
            x = max(0, min(_CANVAS_WIDTH, x))
            y = max(0, min(_CANVAS_HEIGHT, y))

            if n.cortisol > 0.4:
                self._draw_jagged_form(
                    ctx, x, y, color, alpha, complexity
                )
            elif n.gaba > 0.6:
                self._draw_soft_circle(ctx, x, y, color, alpha)
            else:
                self._draw_bezier_form(
                    ctx, x, y, color, alpha, energy
                )

    def _render_foreground_layer(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        focal: tuple[float, float],
    ) -> None:
        """Render foreground — sharp details near the focal point."""
        n_forms = 3 + int(energy * 5)
        fx, fy = focal

        for _ in range(n_forms):
            color = palette[self._rng.integers(0, len(palette))].rgb
            alpha = 0.5 + float(self._rng.uniform(0, 0.3))

            # Cluster around focal point
            x = fx + float(self._rng.normal(0, 80))
            y = fy + float(self._rng.normal(0, 80))
            x = max(0, min(_CANVAS_WIDTH, x))
            y = max(0, min(_CANVAS_HEIGHT, y))

            size = float(self._rng.integers(10, 40))

            self._draw_bezier_stroke(
                ctx, x, y, size, color, alpha, energy
            )

    def _render_flat_forms(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render forms without depth layering (early stage)."""
        n_forms = 5 + int(complexity * 15)

        for _ in range(n_forms):
            color = palette[self._rng.integers(0, len(palette))].rgb
            alpha = 0.3 + float(self._rng.uniform(0, 0.3))
            x = float(self._rng.integers(0, _CANVAS_WIDTH))
            y = float(self._rng.integers(0, _CANVAS_HEIGHT))

            if n.cortisol > 0.4:
                self._draw_jagged_form(
                    ctx, x, y, color, alpha, complexity
                )
            elif n.gaba > 0.6:
                self._draw_soft_circle(ctx, x, y, color, alpha)
            else:
                self._draw_soft_circle(ctx, x, y, color, alpha)

    # ── Form primitives ───────────────────────────────────────────

    def _draw_soft_circle(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Draw a soft circle with a radial gradient edge."""
        radius = float(self._rng.integers(30, 100))
        pat = cairo.RadialGradient(x, y, 0, x, y, radius)
        pat.add_color_stop_rgba(
            0, color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )
        pat.add_color_stop_rgba(
            0.8,
            color[0] / 255,
            color[1] / 255,
            color[2] / 255,
            alpha * 0.7,
        )
        pat.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.set_source(pat)
        ctx.arc(x, y, radius, 0, math.pi * 2)
        ctx.fill()

    def _draw_jagged_form(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        color: tuple[int, int, int],
        alpha: float,
        complexity: float,
    ) -> None:
        """Draw a sharp angular form — stress expression."""
        size = float(self._rng.integers(10, 60))
        angle = float(self._rng.uniform(0, math.pi * 2))
        n_points = 3 + int(complexity * 4)

        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )
        ctx.move_to(
            x + size * math.cos(angle),
            y + size * math.sin(angle),
        )
        for j in range(1, n_points + 1):
            a = angle + j * (math.pi * 2 / n_points)
            r = size * (0.4 + float(self._rng.uniform(0, 0.6)))
            ctx.line_to(x + r * math.cos(a), y + r * math.sin(a))
        ctx.close_path()
        ctx.fill()

    def _draw_bezier_form(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        color: tuple[int, int, int],
        alpha: float,
        energy: float,
    ) -> None:
        """Draw an organic form using bezier curves."""
        size = float(self._rng.integers(20, 80))
        n_views = 3 + int(energy * 4)

        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )

        # Build a closed organic shape with cubic beziers
        points = []
        for i in range(n_views):
            a = i * (math.pi * 2 / n_views)
            r = size * (0.6 + float(self._rng.uniform(0, 0.4)))
            points.append((x + r * math.cos(a), y + r * math.sin(a)))

        ctx.move_to(points[0][0], points[0][1])
        for i in range(n_views):
            p1 = points[i]
            p2 = points[(i + 1) % n_views]
            # Control points — offset perpendicular for organic curves
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            # Perpendicular offset
            offset = size * 0.3 * float(self._rng.uniform(-1, 1))
            cx = mx - dy / max(1, math.hypot(dx, dy)) * offset
            cy = my + dx / max(1, math.hypot(dx, dy)) * offset
            ctx.curve_to(p1[0], p1[1], cx, cy, p2[0], p2[1])
        ctx.close_path()
        ctx.fill()

    def _draw_bezier_stroke(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        size: float,
        color: tuple[int, int, int],
        alpha: float,
        energy: float,
    ) -> None:
        """Draw a flowing bezier stroke — energetic expression."""
        angle = float(self._rng.uniform(0, math.pi * 2))
        length = size * (2 + energy * 4)

        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle)

        # Control point offset perpendicular to the line
        mx, my = (x + x2) / 2, (y + y2) / 2
        dx, dy = x2 - x, y2 - y
        perp_len = length * 0.3 * float(self._rng.uniform(-1, 1))
        dist = max(1.0, math.hypot(dx, dy))
        cx = mx - dy / dist * perp_len
        cy = my + dx / dist * perp_len

        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )
        ctx.set_line_width(max(1.0, size * 0.15))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(x, y)
        ctx.curve_to(x, y, cx, cy, x2, y2)
        ctx.stroke()

    # ── Flow field ────────────────────────────────────────────────

    def _render_flow_field(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render a flow field — noise-driven particle traces.

        Generates a Perlin-like noise field and traces particles
        through it, creating organic, natural-looking patterns that
        resemble wind, water, or neural pathways.
        """
        # Generate a smooth noise field using FFT
        grid_size = 64
        noise = self._rng.standard_normal((grid_size, grid_size))
        # Smooth with FFT low-pass filter
        fft = np.fft.fft2(noise)
        rows, cols = np.indices((grid_size, grid_size))
        center_r, center_c = grid_size // 2, grid_size // 2
        mask = np.exp(
            -((rows - center_r) ** 2 + (cols - center_c) ** 2)
            / (2 * (grid_size * 0.15) ** 2)
        )
        fft_shifted = np.fft.fftshift(fft)
        fft_shifted *= mask
        field = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shifted)))
        # Normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)

        # Trace particles through the field
        n_particles = 30 + int(energy * 50)
        n_steps = 40 + int(n.bdnf * 60)
        step_size = 3.0

        for _ in range(n_particles):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = float(self._rng.uniform(0, _CANVAS_HEIGHT))
            color = palette[self._rng.integers(0, len(palette))].rgb
            alpha = 0.1 + float(self._rng.uniform(0, 0.2))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(1.0 + energy * 2)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.move_to(x, y)

            for _step in range(n_steps):
                # Sample noise field to get direction
                gx = int(x / _CANVAS_WIDTH * grid_size) % grid_size
                gy = int(y / _CANVAS_HEIGHT * grid_size) % grid_size
                angle = field[gy, gx] * math.pi * 4  # multiple rotations
                x += math.cos(angle) * step_size
                y += math.sin(angle) * step_size
                if x < 0 or x >= _CANVAS_WIDTH:
                    break
                if y < 0 or y >= _CANVAS_HEIGHT:
                    break
                ctx.line_to(x, y)

            ctx.stroke()

    # ── Fractal branches ──────────────────────────────────────────

    def _render_fractals(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render fractal branches — recursive tree-like forms.

        BDNF-driven complexity controls recursion depth. Each branch
        splits into 2-3 children with random angles, creating organic
        tree-like or neural-network-like structures.
        """
        n_trees = 1 + int(complexity * 3)
        max_depth = 4 + int(complexity * 4)

        for _ in range(n_trees):
            x = float(self._rng.integers(50, _CANVAS_WIDTH - 50))
            y = float(
                self._rng.integers(_CANVAS_HEIGHT // 2, _CANVAS_HEIGHT - 20)
            )
            angle = -math.pi / 2 + float(
                self._rng.uniform(-0.3, 0.3)
            )  # upward
            length = float(self._rng.integers(40, 80))
            color = palette[self._rng.integers(0, len(palette))].rgb

            self._draw_branch(
                ctx, x, y, angle, length, max_depth, color, energy
            )

    def _draw_branch(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        angle: float,
        length: float,
        depth: int,
        color: tuple[int, int, int],
        energy: float,
    ) -> None:
        """Recursively draw a fractal branch."""
        if depth <= 0 or length < 2:
            return

        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle)

        alpha = 0.2 + depth * 0.08
        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )
        ctx.set_line_width(max(0.5, depth * 0.8))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(x, y)
        ctx.line_to(x2, y2)
        ctx.stroke()

        # Recursive branches
        n_children = 2 + (1 if energy > 0.5 else 0)
        for _ in range(n_children):
            child_angle = angle + float(
                self._rng.uniform(-0.6, 0.6)
            )
            child_length = length * (0.6 + float(self._rng.uniform(0, 0.15)))
            self._draw_branch(
                ctx,
                x2,
                y2,
                child_angle,
                child_length,
                depth - 1,
                color,
                energy,
            )

    # ── Light glow ────────────────────────────────────────────────

    def _render_light_glow(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
    ) -> None:
        """Render luminous radial gradients — endorphin-driven light."""
        if n.endorphin <= 0.15:
            return

        n_glows = int(n.endorphin * 8)
        for _ in range(n_glows):
            x = float(self._rng.integers(0, _CANVAS_WIDTH))
            y = float(self._rng.integers(0, _CANVAS_HEIGHT))
            radius = float(self._rng.integers(40, 120))

            pat = cairo.RadialGradient(x, y, 0, x, y, radius)
            pat.add_color_stop_rgba(1.0, 0.95, 0.92, 0.7, 0.6)
            pat.add_color_stop_rgba(0.5, 1.0, 0.95, 0.8, 0.3)
            pat.add_color_stop_rgba(0, 1.0, 1.0, 0.9, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, radius, 0, math.pi * 2)
            ctx.fill()

    # ── Accents ───────────────────────────────────────────────────

    def _render_accents(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render accent details — bright spots and energy lines."""
        # Endorphin → luminous spots
        if n.endorphin > 0.15:
            n_spots = int(n.endorphin * 25)
            for _ in range(n_spots):
                x = float(self._rng.integers(0, _CANVAS_WIDTH))
                y = float(self._rng.integers(0, _CANVAS_HEIGHT))
                r = float(self._rng.integers(2, 8))

                pat = cairo.RadialGradient(x, y, 0, x, y, r * 2)
                pat.add_color_stop_rgba(0, 1.0, 0.96, 0.75, 0.9)
                pat.add_color_stop_rgba(1, 1.0, 0.96, 0.75, 0)
                ctx.set_source(pat)
                ctx.arc(x, y, r * 2, 0, math.pi * 2)
                ctx.fill()

        # Norepinephrine → energy lines
        if n.norepinephrine > 0.4:
            n_lines = int(n.norepinephrine * 15)
            for _ in range(n_lines):
                x1 = float(self._rng.integers(0, _CANVAS_WIDTH))
                y1 = float(self._rng.integers(0, _CANVAS_HEIGHT))
                length = float(self._rng.integers(20, 80))
                angle = float(self._rng.uniform(0, math.pi * 2))
                x2 = x1 + length * math.cos(angle)
                y2 = y1 + length * math.sin(angle)
                color = palette[0].rgb
                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, 0.6
                )
                ctx.set_line_width(2)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.stroke()

    # ── Mark-making techniques ────────────────────────────────────

    def _render_stippling(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render stippling — pointillist dot fields.

        Density of dots is driven by BDNF (complexity). Dots cluster
        more densely near areas of higher emotional intensity,
        creating tonal variation through dot concentration alone.
        """
        n_dots = 200 + int(complexity * 800)
        cx = _CANVAS_WIDTH / 2
        cy = _CANVAS_HEIGHT / 2

        for _ in range(n_dots):
            # Cluster toward center with gaussian falloff
            angle = float(self._rng.uniform(0, math.pi * 2))
            dist = abs(float(self._rng.normal(0, 180)))
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            x = max(0, min(_CANVAS_WIDTH, x))
            y = max(0, min(_CANVAS_HEIGHT, y))

            color = palette[self._rng.integers(0, len(palette))].rgb
            r = float(self._rng.uniform(0.5, 2.5))
            alpha = 0.3 + float(self._rng.uniform(0, 0.4))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.arc(x, y, r, 0, math.pi * 2)
            ctx.fill()

    def _render_hatching(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render cross-hatching — layered line work for tonal depth.

        Multiple layers of parallel lines at different angles create
        darker tones through overlap. Cortisol increases line density
        and sharpness; GABA produces softer, wider-spaced strokes.
        """
        n_layers = 2 + int(complexity * 4)
        base_angle = float(self._rng.uniform(0, math.pi))

        for layer in range(n_layers):
            angle = base_angle + layer * (math.pi / n_layers)
            n_lines = 30 + int(energy * 50)
            spacing = 8 + int((1 - n.cortisol) * 12)

            color = palette[layer % len(palette)].rgb
            alpha = 0.1 + float(self._rng.uniform(0, 0.15))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(0.5 + energy)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            # Draw parallel lines at the layer angle
            dx = math.cos(angle)
            dy = math.sin(angle)
            perp_x = -dy
            perp_y = dx
            diag = math.hypot(_CANVAS_WIDTH, _CANVAS_HEIGHT)

            for i in range(-n_lines, n_lines):
                offset = i * spacing
                cx = _CANVAS_WIDTH / 2 + perp_x * offset
                cy = _CANVAS_HEIGHT / 2 + perp_y * offset
                x1 = cx - dx * diag
                y1 = cy - dy * diag
                x2 = cx + dx * diag
                y2 = cy + dy * diag
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.stroke()

    def _render_impressionist_strokes(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render impressionist strokes — visible broken brush marks.

        Short, thick strokes of pure color placed side by side.
        The eye blends them optically. Dopamine drives stroke vibrancy;
        serotonin drives cooler hues; energy drives stroke length.
        """
        n_strokes = 100 + int(complexity * 300)

        for _ in range(n_strokes):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = float(self._rng.uniform(0, _CANVAS_HEIGHT))
            color = palette[self._rng.integers(0, len(palette))].rgb
            angle = float(self._rng.uniform(0, math.pi * 2))
            length = 4 + energy * 12
            alpha = 0.4 + float(self._rng.uniform(0, 0.3))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(2 + energy * 3)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.move_to(x, y)
            ctx.line_to(
                x + length * math.cos(angle),
                y + length * math.sin(angle),
            )
            ctx.stroke()

    def _render_ink_bleed(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render ink bleed — ink drops spreading on paper.

        Each drop has a dark core with a soft diffusion halo,
        simulating ink absorbing into paper fiber. Cortisol makes
        the ink darker and sharper; GABA makes it spread more softly.
        """
        n_drops = 5 + int(energy * 15)

        for _ in range(n_drops):
            x = float(self._rng.uniform(50, _CANVAS_WIDTH - 50))
            y = float(self._rng.uniform(50, _CANVAS_HEIGHT - 50))
            color = palette[self._rng.integers(0, len(palette))].rgb
            core_r = float(self._rng.uniform(5, 20))
            bleed_r = core_r * (2 + (1 - n.cortisol) * 3)

            # Dark core
            ctx.set_source_rgba(
                color[0] / 255 * 0.5,
                color[1] / 255 * 0.5,
                color[2] / 255 * 0.5,
                0.8,
            )
            ctx.arc(x, y, core_r, 0, math.pi * 2)
            ctx.fill()

            # Bleed halo
            pat = cairo.RadialGradient(x, y, core_r, x, y, bleed_r)
            pat.add_color_stop_rgba(
                0,
                color[0] / 255 * 0.5,
                color[1] / 255 * 0.5,
                color[2] / 255 * 0.5,
                0.4,
            )
            pat.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, bleed_r, 0, math.pi * 2)
            ctx.fill()

    def _render_watercolor_bleed(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render watercolor bleed — soft diffused color washes.

        Large, low-opacity color washes with irregular edges that
        blend where they overlap. Oxytocin increases the softness
        and warmth of the washes.
        """
        n_washes = 3 + int(energy * 6)

        for _ in range(n_washes):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = float(self._rng.uniform(0, _CANVAS_HEIGHT))
            color = palette[self._rng.integers(0, len(palette))].rgb
            radius = float(self._rng.uniform(80, 200))
            alpha = 0.08 + n.oxytocin * 0.1

            # Irregular edge — multiple overlapping circles
            for _sub in range(5):
                ox = x + float(self._rng.normal(0, radius * 0.3))
                oy = y + float(self._rng.normal(0, radius * 0.3))
                r = radius * float(self._rng.uniform(0.7, 1.0))

                pat = cairo.RadialGradient(ox, oy, 0, ox, oy, r)
                pat.add_color_stop_rgba(
                    0,
                    color[0] / 255,
                    color[1] / 255,
                    color[2] / 255,
                    alpha,
                )
                pat.add_color_stop_rgba(1, 0, 0, 0, 0)
                ctx.set_source(pat)
                ctx.arc(ox, oy, r, 0, math.pi * 2)
                ctx.fill()

    # ── Tonal & contrast techniques ───────────────────────────────

    def _render_silhouette(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        focal: tuple[float, float],
    ) -> None:
        """Render silhouettes — dark forms against the background.

        Filled shapes in near-black, creating dramatic contrast.
        Cortisol makes them sharper and more angular; oxytocin
        makes them softer and more rounded.
        """
        n_forms = 2 + int(n.norepinephrine * 4)
        fx, fy = focal

        for _ in range(n_forms):
            x = fx + float(self._rng.normal(0, 100))
            y = fy + float(self._rng.normal(0, 80))
            x = max(20, min(_CANVAS_WIDTH - 20, x))
            y = max(20, min(_CANVAS_HEIGHT - 20, y))
            size = float(self._rng.uniform(40, 120))

            darkness = 0.7 + n.cortisol * 0.2
            ctx.set_source_rgba(0.05, 0.05, 0.08, darkness)

            if n.cortisol > 0.4:
                # Angular silhouette
                n_pts = 5 + int(n.cortisol * 5)
                ctx.move_to(x + size, y)
                for j in range(1, n_pts + 1):
                    a = j * (math.pi * 2 / n_pts)
                    r = size * (0.5 + float(self._rng.uniform(0, 0.5)))
                    ctx.line_to(x + r * math.cos(a), y + r * math.sin(a))
                ctx.close_path()
                ctx.fill()
            else:
                # Soft rounded silhouette
                ctx.arc(x, y, size, 0, math.pi * 2)
                ctx.fill()

    def _render_chiaroscuro(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        focal: tuple[float, float],
    ) -> None:
        """Render chiaroscuro — strong light/dark contrast.

        A single light source at the focal point illuminates one
        side while the other falls into deep shadow. This creates
        dramatic three-dimensional volume from flat forms.
        """
        fx, fy = focal
        light_r = 150 + int(n.endorphin * 100)

        # Light source — bright radial gradient at focal point
        pat = cairo.RadialGradient(fx, fy, 0, fx, fy, light_r)
        pat.add_color_stop_rgba(1.0, 1.0, 0.95, 0.8, 0.4)
        pat.add_color_stop_rgba(0.5, 1.0, 0.9, 0.7, 0.15)
        pat.add_color_stop_rgba(0, 1.0, 1.0, 0.9, 0)
        ctx.set_source(pat)
        ctx.arc(fx, fy, light_r, 0, math.pi * 2)
        ctx.fill()

        # Deep shadow on the opposite side
        shadow_x = _CANVAS_WIDTH - fx
        shadow_y = _CANVAS_HEIGHT - fy
        shadow_r = 200 + int(n.cortisol * 100)
        pat2 = cairo.RadialGradient(shadow_x, shadow_y, 0, shadow_x, shadow_y, shadow_r)
        pat2.add_color_stop_rgba(0, 0, 0, 0, 0.5 + n.cortisol * 0.2)
        pat2.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.set_source(pat2)
        ctx.arc(shadow_x, shadow_y, shadow_r, 0, math.pi * 2)
        ctx.fill()

    def _render_sfumato(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        focal: tuple[float, float],
    ) -> None:
        """Render sfumato — smoky, soft-edge blending (Da Vinci).

        Layers of translucent color with very soft edges, creating
        an atmospheric haze where forms emerge from and dissolve into
        the background. GABA enhances the softness.
        """
        fx, fy = focal
        n_layers = 4 + int(n.gaba * 6)

        for i in range(n_layers):
            color = palette[i % len(palette)].rgb
            offset = i * 30
            x = fx + float(self._rng.normal(0, 60 + offset))
            y = fy + float(self._rng.normal(0, 60 + offset))
            radius = 80 + offset * 2
            alpha = 0.06 + n.gaba * 0.04

            pat = cairo.RadialGradient(x, y, 0, x, y, radius)
            pat.add_color_stop_rgba(
                0, color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            pat.add_color_stop_rgba(
                0.5,
                color[0] / 255,
                color[1] / 255,
                color[2] / 255,
                alpha * 0.5,
            )
            pat.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, radius, 0, math.pi * 2)
            ctx.fill()

    # ── Compositional techniques ──────────────────────────────────

    def _render_mandala(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render a mandala — radial symmetric pattern.

        Repeated motifs around a center point with N-fold symmetry.
        BDNF controls the number of symmetry axes and detail level.
        Serotonin enhances the sense of balance and harmony.
        """
        cx = _CANVAS_WIDTH / 2
        cy = _CANVAS_HEIGHT / 2
        n_axes = 6 + int(complexity * 10)
        n_rings = 3 + int(complexity * 5)
        max_r = min(_CANVAS_WIDTH, _CANVAS_HEIGHT) * 0.4

        for ring in range(n_rings):
            r = max_r * (ring + 1) / n_rings
            color = palette[ring % len(palette)].rgb
            alpha = 0.15 + float(self._rng.uniform(0, 0.15))

            for axis in range(n_axes):
                angle = axis * (math.pi * 2 / n_axes)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                dot_r = 3 + complexity * 8

                pat = cairo.RadialGradient(x, y, 0, x, y, dot_r * 2)
                pat.add_color_stop_rgba(
                    0,
                    color[0] / 255,
                    color[1] / 255,
                    color[2] / 255,
                    alpha,
                )
                pat.add_color_stop_rgba(1, 0, 0, 0, 0)
                ctx.set_source(pat)
                ctx.arc(x, y, dot_r * 2, 0, math.pi * 2)
                ctx.fill()

            # Connect ring points with faint lines
            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha * 0.5
            )
            ctx.set_line_width(0.5)
            for axis in range(n_axes):
                a1 = axis * (math.pi * 2 / n_axes)
                a2 = (axis + 1) * (math.pi * 2 / n_axes)
                ctx.move_to(
                    cx + r * math.cos(a1), cy + r * math.sin(a1)
                )
                ctx.line_to(
                    cx + r * math.cos(a2), cy + r * math.sin(a2)
                )
                ctx.stroke()

    def _render_spirals(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render spiral forms — logarithmic/golden spirals.

        Particles trace inward or outward along a logarithmic spiral,
        creating hypnotic vortex patterns. Energy controls the
        tightness; complexity controls the number of spiral arms.
        """
        cx = _CANVAS_WIDTH / 2
        cy = _CANVAS_HEIGHT / 2
        n_spirals = 1 + int(complexity * 4)
        tightness = 0.15 + energy * 0.1

        for s in range(n_spirals):
            start_angle = s * (math.pi * 2 / n_spirals)
            color = palette[s % len(palette)].rgb
            alpha = 0.2 + float(self._rng.uniform(0, 0.2))
            n_steps = 80 + int(complexity * 80)

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(1 + energy * 2)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            for step in range(n_steps):
                t = step / n_steps
                angle = start_angle + t * math.pi * 6
                r = t * min(_CANVAS_WIDTH, _CANVAS_HEIGHT) * 0.4
                x = cx + r * math.cos(angle) * tightness * 5
                y = cy + r * math.sin(angle) * tightness * 5
                if step == 0:
                    ctx.move_to(x, y)
                else:
                    ctx.line_to(x, y)
            ctx.stroke()

    def _render_wave_interference(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render wave interference — overlapping sine wave patterns.

        Multiple wave sources create interference patterns — areas
        of constructive and destructive interference. Norepinephrine
        increases the number of wave sources and amplitude.
        """
        n_sources = 2 + int(energy * 4)
        sources = [
            (
                float(self._rng.uniform(0, _CANVAS_WIDTH)),
                float(self._rng.uniform(0, _CANVAS_HEIGHT)),
                float(self._rng.uniform(20, 50)),
            )
            for _ in range(n_sources)
        ]
        n_lines = 15 + int(energy * 25)

        for line_idx in range(n_lines):
            y = line_idx * (_CANVAS_HEIGHT / n_lines)
            color = palette[line_idx % len(palette)].rgb
            alpha = 0.1 + float(self._rng.uniform(0, 0.1))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(1.0)
            ctx.move_to(0, y)

            for x in range(0, _CANVAS_WIDTH, 4):
                displacement = 0.0
                for sx, sy, amp in sources:
                    dist = math.hypot(x - sx, y - sy)
                    displacement += amp * math.sin(dist * 0.05)
                ctx.line_to(x, y + displacement)
            ctx.stroke()

    def _render_voronoi(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render Voronoi cells — tessellation from seed points.

        Each cell is the region closest to one seed point. Cell
        boundaries create an organic crackle pattern. BDNF controls
        the number of seeds (complexity).
        """
        n_seeds = 8 + int(complexity * 30)
        seeds = [
            (
                float(self._rng.uniform(0, _CANVAS_WIDTH)),
                float(self._rng.uniform(0, _CANVAS_HEIGHT)),
            )
            for _ in range(n_seeds)
        ]
        # Approximate Voronoi by coloring each pixel region —
        # but that's too slow in Cairo. Instead, draw cell edges
        # by sampling a grid and finding boundary transitions.
        grid_step = 8
        grid_w = _CANVAS_WIDTH // grid_step
        grid_h = _CANVAS_HEIGHT // grid_step
        labels = np.zeros((grid_h, grid_w), dtype=np.int32)
        for gy in range(grid_h):
            for gx in range(grid_w):
                px = gx * grid_step
                py = gy * grid_step
                best_d = float("inf")
                best_i = 0
                for i, (sx, sy) in enumerate(seeds):
                    d = (px - sx) ** 2 + (py - sy) ** 2
                    if d < best_d:
                        best_d = d
                        best_i = i
                labels[gy, gx] = best_i

        # Draw boundary edges where label changes
        for gy in range(grid_h - 1):
            for gx in range(grid_w - 1):
                if labels[gy, gx] != labels[gy, gx + 1]:
                    color = palette[labels[gy, gx] % len(palette)].rgb
                    ctx.set_source_rgba(
                        color[0] / 255,
                        color[1] / 255,
                        color[2] / 255,
                        0.3,
                    )
                    ctx.set_line_width(1.0)
                    x = (gx + 1) * grid_step
                    ctx.move_to(x, gy * grid_step)
                    ctx.line_to(x, (gy + 1) * grid_step)
                    ctx.stroke()
                if labels[gy, gx] != labels[gy + 1, gx]:
                    color = palette[labels[gy, gx] % len(palette)].rgb
                    ctx.set_source_rgba(
                        color[0] / 255,
                        color[1] / 255,
                        color[2] / 255,
                        0.3,
                    )
                    ctx.set_line_width(1.0)
                    y = (gy + 1) * grid_step
                    ctx.move_to(gx * grid_step, y)
                    ctx.line_to((gx + 1) * grid_step, y)
                    ctx.stroke()

    def _render_contour_lines(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render contour lines — topographic-style isolines.

        Generates a smooth scalar field and draws isolines at
        regular intervals, creating a topographic-map aesthetic.
        BDNF controls field complexity and line density.
        """
        grid_size = 48
        noise = self._rng.standard_normal((grid_size, grid_size))
        # Smooth with FFT
        fft = np.fft.fft2(noise)
        rows, cols = np.indices((grid_size, grid_size))
        cr, cc = grid_size // 2, grid_size // 2
        mask = np.exp(
            -((rows - cr) ** 2 + (cols - cc) ** 2)
            / (2 * (grid_size * (0.2 - complexity * 0.05)) ** 2)
        )
        field = np.real(
            np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(fft) * mask))
        )
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)

        n_levels = 5 + int(complexity * 10)
        levels = np.linspace(0.1, 0.9, n_levels)

        for li, level in enumerate(levels):
            color = palette[li % len(palette)].rgb
            alpha = 0.15 + float(self._rng.uniform(0, 0.1))
            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(0.8)

            # Simple marching: find cells where field crosses level
            for gy in range(grid_size - 1):
                for gx in range(grid_size - 1):
                    v00 = field[gy, gx]
                    v01 = field[gy, gx + 1]
                    v10 = field[gy + 1, gx]
                    # Check if level is between v00 and v01 (horizontal edge)
                    if (v00 - level) * (v01 - level) < 0:
                        t = (level - v00) / (v01 - v00 + 1e-9)
                        x = (gx + t) / grid_size * _CANVAS_WIDTH
                        y = gy / grid_size * _CANVAS_HEIGHT
                        ctx.rectangle(x, y, 1, 1)
                        ctx.fill()
                    if (v00 - level) * (v10 - level) < 0:
                        t = (level - v00) / (v10 - v00 + 1e-9)
                        x = gx / grid_size * _CANVAS_WIDTH
                        y = (gy + t) / grid_size * _CANVAS_HEIGHT
                        ctx.rectangle(x, y, 1, 1)
                        ctx.fill()

    def _render_geometric_tessellation(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render geometric tessellation — Islamic-style star tiling.

        Repeated star polygons connected by interlacing lines,
        creating the geometric patterns of Islamic art. Serotonin
        enhances the sense of order and balance.
        """
        n_points = 6 + int(complexity * 6) * 2  # even number
        tile_size = 60 + int((1 - complexity) * 40)
        skip = n_points // 2 - 1  # star polygon {n/2-1}

        for gy in range(0, _CANVAS_HEIGHT + tile_size, tile_size):
            for gx in range(0, _CANVAS_WIDTH + tile_size, tile_size):
                # Offset alternate rows for interlocking
                ox = tile_size // 2 if (gy // tile_size) % 2 else 0
                cx = gx + ox
                cy = gy
                color = palette[
                    (gx // tile_size + gy // tile_size) % len(palette)
                ].rgb
                alpha = 0.1 + float(self._rng.uniform(0, 0.1))

                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, alpha
                )
                ctx.set_line_width(1.0)

                r = tile_size * 0.4
                points = []
                for i in range(n_points):
                    a = i * (math.pi * 2 / n_points) - math.pi / 2
                    points.append((cx + r * math.cos(a), cy + r * math.sin(a)))

                # Draw star polygon by connecting every skip-th point
                ctx.move_to(points[0][0], points[0][1])
                idx = 0
                visited = {0}
                while True:
                    idx = (idx + skip) % n_points
                    ctx.line_to(points[idx][0], points[idx][1])
                    if idx in visited:
                        break
                    visited.add(idx)
                ctx.stroke()

    def _render_cubist_facets(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        focal: tuple[float, float],
    ) -> None:
        """Render cubist facets — fragmented geometric planes.

        The canvas is divided into irregular triangular facets,
        each filled with a different palette color at varying
        opacity. Cortisol increases fragmentation sharpness.
        """
        fx, fy = focal
        n_facets = 15 + int(energy * 30)

        # Generate random points, then create triangles
        points = [
            (0, 0),
            (_CANVAS_WIDTH, 0),
            (0, _CANVAS_HEIGHT),
            (_CANVAS_WIDTH, _CANVAS_HEIGHT),
            (fx, fy),
        ]
        for _ in range(n_facets):
            points.append(
                (
                    float(self._rng.uniform(0, _CANVAS_WIDTH)),
                    float(self._rng.uniform(0, _CANVAS_HEIGHT)),
                )
            )

        # Simple triangulation: connect each point to 2 nearest
        for i, (px, py) in enumerate(points[2:], start=2):
            # Find 2 nearest other points
            dists = []
            for j, (qx, qy) in enumerate(points):
                if j == i:
                    continue
                d = (px - qx) ** 2 + (py - qy) ** 2
                dists.append((d, j))
            dists.sort()
            if len(dists) >= 2:
                j1 = dists[0][1]
                j2 = dists[1][1]
                color = palette[
                    (i + j1 + j2) % len(palette)
                ].rgb
                alpha = 0.1 + float(self._rng.uniform(0, 0.15))

                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, alpha
                )
                ctx.move_to(px, py)
                ctx.line_to(points[j1][0], points[j1][1])
                ctx.line_to(points[j2][0], points[j2][1])
                ctx.close_path()
                ctx.fill()

    def _render_sacred_geometry(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render sacred geometry — Flower of Life and Metatron's Cube.

        The Flower of Life is 19 overlapping circles arranged in a
        hexagonal lattice — one center, six first-ring, twelve
        second-ring. Metatron's Cube connects the centers of 13
        circles with straight lines, producing the Platonic solids
        as embedded figures. Serotonin enhances the sense of
        sacred order; BDNF controls whether the Metatron's Cube
        lines are drawn (higher complexity → more connections).
        """
        cx = _CANVAS_WIDTH / 2
        cy = _CANVAS_HEIGHT / 2
        r = min(_CANVAS_WIDTH, _CANVAS_HEIGHT) * 0.08
        color = palette[self._rng.integers(0, len(palette))].rgb
        alpha = 0.12 + float(self._rng.uniform(0, 0.08))

        # Generate the 19 circle centers of the Flower of Life.
        # First ring: 6 circles at 60° intervals at distance r.
        # Second ring: 12 circles at the outer intersections.
        centers = [(cx, cy)]
        for i in range(6):
            a = i * math.pi / 3
            centers.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        # Second ring — 6 at distance r*sqrt(3) and 6 at distance 2r
        for i in range(6):
            a = i * math.pi / 3 + math.pi / 6
            centers.append(
                (cx + r * math.sqrt(3) * math.cos(a),
                 cy + r * math.sqrt(3) * math.sin(a))
            )
        for i in range(6):
            a = i * math.pi / 3
            centers.append((cx + 2 * r * math.cos(a), cy + 2 * r * math.sin(a)))

        # Draw the circles
        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )
        ctx.set_line_width(1.0)
        for px, py in centers:
            ctx.arc(px, py, r, 0, math.pi * 2)
            ctx.stroke()

        # Metatron's Cube — connect the 13 inner circle centers
        # (center + first ring + second ring's inner 6). Higher
        # complexity draws more connections.
        metatron = centers[:13]
        n_connections = int(len(metatron) * (1 + complexity))
        for i in range(min(n_connections, len(metatron))):
            for j in range(i + 1, len(metatron)):
                if self._rng.random() > complexity * 0.5:
                    continue
                x1, y1 = metatron[i]
                x2, y2 = metatron[j]
                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255,
                    alpha * 0.6,
                )
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.stroke()

    # ── Spatial techniques ────────────────────────────────────────

    def _render_three_dimensional(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render three-dimensional perspective wireframe forms.

        Draws wireframe polyhedra (cube, tetrahedron, octahedron)
        using simple one-point perspective projection. Energy
        controls the number of forms and their rotation; complexity
        controls how many forms and the line detail. Norepinephrine
        adds dynamism through rotation offset.
        """
        # Polyhedron vertex sets (unit shapes centered at origin)
        shapes: dict[str, list[tuple[float, float, float]]] = {
            "cube": [
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            ],
            "tetra": [
                (0, 1, 0), (-0.94, -0.5, -0.54),
                (0.94, -0.5, -0.54), (0, -0.5, 0.85),
            ],
            "octa": [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1),
            ],
        }
        shape_edges: dict[str, list[tuple[int, int]]] = {
            "cube": [(0, 1), (1, 2), (2, 3), (3, 0),
                      (4, 5), (5, 6), (6, 7), (7, 4),
                      (0, 4), (1, 5), (2, 6), (3, 7)],
            "tetra": [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)],
            "octa": [(0, 2), (0, 3), (0, 4), (0, 5),
                      (1, 2), (1, 3), (1, 4), (1, 5),
                      (2, 4), (2, 5), (3, 4), (3, 5)],
        }
        shape_names = list(shapes.keys())
        n_forms = 2 + int(energy * 4)
        fov = 3.0  # perspective distance

        for f in range(n_forms):
            name = shape_names[f % len(shape_names)]
            verts = shapes[name]
            edges = shape_edges[name]
            size = 40 + float(self._rng.uniform(0, 30))
            # Random position and rotation
            ox = float(self._rng.uniform(
                _CANVAS_WIDTH * 0.2, _CANVAS_WIDTH * 0.8
            ))
            oy = float(self._rng.uniform(
                _CANVAS_HEIGHT * 0.2, _CANVAS_HEIGHT * 0.8
            ))
            rot_y = energy * math.pi * float(self._rng.uniform(0, 1))
            rot_x = n.norepinephrine * math.pi * float(
                self._rng.uniform(0, 0.5)
            )
            color = palette[f % len(palette)].rgb
            alpha = 0.2 + float(self._rng.uniform(0, 0.15))

            # Project vertices
            projected = []
            for vx, vy, vz in verts:
                # Rotate around Y axis
                rx = vx * math.cos(rot_y) - vz * math.sin(rot_y)
                rz = vx * math.sin(rot_y) + vz * math.cos(rot_y)
                ry = vy
                # Rotate around X axis
                ry2 = ry * math.cos(rot_x) - rz * math.sin(rot_x)
                rz2 = ry * math.sin(rot_x) + rz * math.cos(rot_x)
                # Perspective project
                scale = fov / (fov + rz2)
                px = ox + rx * size * scale
                py = oy + ry2 * size * scale
                projected.append((px, py))

            # Draw edges
            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(1.5)
            for i, j in edges:
                ctx.move_to(projected[i][0], projected[i][1])
                ctx.line_to(projected[j][0], projected[j][1])
                ctx.stroke()

            # Higher complexity: draw vertices as dots
            if complexity > 0.5:
                for px, py in projected:
                    ctx.arc(px, py, 2.0, 0, math.pi * 2)
                    ctx.fill()

    # ── Generative / algorithmic techniques ───────────────────────

    def _render_l_systems(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render L-systems — Lindenmayer plant-like growth.

        Iterated string rewriting produces branching plant forms.
        BDNF controls iteration depth (growth complexity). Each
        drawing uses a randomly selected L-system rule set.
        """
        # A few L-system rule sets
        rulesets = [
            {"F": "FF+[+F-F-F]-[-F+F+F]"},  # plant-like
            {"F": "F[+F]F[-F]F"},            # bush
            {"F": "FF-[-F+F+F]+[+F-F-F]"},   # seaweed
        ]
        rules = rulesets[int(self._rng.integers(0, len(rulesets)))]
        axiom = "F"
        depth = 3 + int(complexity * 3)

        for _ in range(depth):
            axiom = "".join(rules.get(c, c) for c in axiom)
            if len(axiom) > 5000:
                break

        x = float(self._rng.uniform(50, _CANVAS_WIDTH - 50))
        y = float(_CANVAS_HEIGHT - 20)
        angle = -math.pi / 2
        length = 8 + complexity * 6
        stack: list[tuple[float, float, float]] = []
        color = palette[self._rng.integers(0, len(palette))].rgb
        turn = math.pi / 7

        ctx.set_line_width(1.5)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)

        for c in axiom:
            if c == "F":
                x2 = x + length * math.cos(angle)
                y2 = y + length * math.sin(angle)
                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, 0.3
                )
                ctx.move_to(x, y)
                ctx.line_to(x2, y2)
                ctx.stroke()
                x, y = x2, y2
            elif c == "+":
                angle += turn
            elif c == "-":
                angle -= turn
            elif c == "[":
                stack.append((x, y, angle))
            elif c == "]":
                if stack:
                    x, y, angle = stack.pop()

    def _render_particle_burst(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        focal: tuple[float, float],
    ) -> None:
        """Render particle burst — physics-driven particle sprays.

        Particles erupt from the focal point with initial velocity,
        affected by gravity and drag. Norepinephrine controls the
        explosion energy; endorphin makes particles luminous.
        """
        fx, fy = focal
        n_bursts = 1 + int(energy * 3)
        gravity = 0.3

        for _ in range(n_bursts):
            bx = fx + float(self._rng.normal(0, 50))
            by = fy + float(self._rng.normal(0, 50))
            n_particles = 20 + int(energy * 40)
            color = palette[self._rng.integers(0, len(palette))].rgb

            for _p in range(n_particles):
                angle = float(self._rng.uniform(0, math.pi * 2))
                speed = float(self._rng.uniform(2, 8)) * (0.5 + energy)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)
                px, py = bx, by
                alpha = 0.3 + float(self._rng.uniform(0, 0.3))

                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, alpha
                )
                ctx.set_line_width(1 + energy)
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                ctx.move_to(px, py)

                for _step in range(20 + int(energy * 30)):
                    px += vx
                    py += vy
                    vy += gravity
                    vx *= 0.98
                    vy *= 0.98
                    if px < 0 or px >= _CANVAS_WIDTH:
                        break
                    if py < 0 or py >= _CANVAS_HEIGHT:
                        break
                    ctx.line_to(px, py)
                ctx.stroke()

    def _render_cellular_automata(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render cellular automata — Conway-like emergent grids.

        A simple 2D cellular automaton evolves from a random seed.
        Living cells are rendered as colored dots. The emergent
        patterns reflect self-organization — BDNF (plasticity)
        drives the number of generations.
        """
        grid_w = 60
        grid_h = 45
        cell = np.zeros((grid_h, grid_w), dtype=np.int8)
        # Random seed
        cell[:, :] = (
            self._rng.random((grid_h, grid_w)) < 0.3 + complexity * 0.1
        ).astype(np.int8)

        generations = 5 + int(complexity * 15)
        cell_size = _CANVAS_WIDTH // grid_w

        for gen in range(generations):
            # Render current generation
            color = palette[gen % len(palette)].rgb
            alpha = 0.05 + complexity * 0.03
            for gy in range(grid_h):
                for gx in range(grid_w):
                    if cell[gy, gx]:
                        ctx.set_source_rgba(
                            color[0] / 255,
                            color[1] / 255,
                            color[2] / 255,
                            alpha,
                        )
                        ctx.rectangle(
                            gx * cell_size,
                            gy * cell_size,
                            cell_size,
                            cell_size,
                        )
                        ctx.fill()

            # Evolve (Conway's rules)
            new_cell = np.zeros_like(cell)
            for gy in range(grid_h):
                for gx in range(grid_w):
                    neighbors = 0
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny = (gy + dy) % grid_h
                            nx = (gx + dx) % grid_w
                            neighbors += cell[ny, nx]
                    if cell[gy, gx]:
                        new_cell[gy, gx] = 1 if neighbors in (2, 3) else 0
                    else:
                        new_cell[gy, gx] = 1 if neighbors == 3 else 0
            cell = new_cell
            if cell.sum() == 0:
                break

    def _render_coral(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render coral forms — organic coral-like branching growth.

        Unlike fractal_branches (which grow upward like trees),
        coral grows in all directions with spherical symmetry,
        creating organic colony structures.
        """
        n_colonies = 1 + int(complexity * 3)
        max_depth = 4 + int(complexity * 4)

        for _ in range(n_colonies):
            cx = float(self._rng.uniform(100, _CANVAS_WIDTH - 100))
            cy = float(self._rng.uniform(100, _CANVAS_HEIGHT - 100))
            color = palette[self._rng.integers(0, len(palette))].rgb
            self._draw_coral_branch(
                ctx, cx, cy, 0, 20 + complexity * 15, max_depth, color
            )

    def _draw_coral_branch(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        angle: float,
        length: float,
        depth: int,
        color: tuple[int, int, int],
    ) -> None:
        """Recursively draw a coral branch in any direction."""
        if depth <= 0 or length < 2:
            return
        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle)
        alpha = 0.15 + depth * 0.05
        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255, alpha
        )
        ctx.set_line_width(max(0.5, depth * 0.6))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(x, y)
        ctx.line_to(x2, y2)
        ctx.stroke()

        n_children = 3
        for _ in range(n_children):
            child_angle = angle + float(self._rng.uniform(-1.0, 1.0))
            child_length = length * (0.6 + float(self._rng.uniform(0, 0.2)))
            self._draw_coral_branch(
                ctx, x2, y2, child_angle, child_length, depth - 1, color
            )

    def _render_crystal_growth(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render crystal growth — dendritic crystalline structures.

        Simulates diffusion-limited aggregation: particles random-walk
        until they stick to the growing crystal, creating fractal
        dendrites like snowflakes or frost patterns.
        """
        n_crystals = 1 + int(complexity * 3)
        n_particles = 50 + int(complexity * 100)

        for _ in range(n_crystals):
            cx = float(self._rng.uniform(100, _CANVAS_WIDTH - 100))
            cy = float(self._rng.uniform(100, _CANVAS_HEIGHT - 100))
            color = palette[self._rng.integers(0, len(palette))].rgb
            crystal_points: list[tuple[float, float]] = [(cx, cy)]

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, 0.25
            )
            ctx.set_line_width(1.0)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            for _p in range(n_particles):
                # Random walk from edge
                px = float(self._rng.uniform(cx - 80, cx + 80))
                py = float(self._rng.uniform(cy - 80, cy + 80))
                for _step in range(30):
                    # Check if near any crystal point
                    for cpx, cpy in crystal_points:
                        if math.hypot(px - cpx, py - cpy) < 8:
                            crystal_points.append((px, py))
                            ctx.move_to(cpx, cpy)
                            ctx.line_to(px, py)
                            ctx.stroke()
                            break
                    else:
                        px += float(self._rng.normal(0, 5))
                        py += float(self._rng.normal(0, 5))
                        continue
                    break

    # ── Natural phenomena ──────────────────────────────────────────

    def _render_caustics(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render caustics — light focused through rippling water.

        Sine wave interference creates bright curving lines where
        light concentrates, like the patterns on a swimming pool
        floor. Serotonin enhances the calm ripple regularity.
        """
        n_lines = 20 + int(energy * 30)
        color = palette[0].rgb if palette else (200, 220, 255)

        for line_idx in range(n_lines):
            y0 = line_idx * (_CANVAS_HEIGHT / n_lines)
            alpha = 0.08 + float(self._rng.uniform(0, 0.06))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(1.0 + energy)
            ctx.move_to(0, y0)

            for x in range(0, _CANVAS_WIDTH, 3):
                wave = (
                    math.sin(x * 0.02 + line_idx * 0.5) * 15
                    + math.sin(x * 0.05 + line_idx * 0.3) * 8
                )
                ctx.line_to(x, y0 + wave)
            ctx.stroke()

    def _render_nebula(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render nebula — cosmic gas clouds.

        Large, soft, multi-colored radial gradients overlap to
        create the appearance of interstellar gas clouds. Adenosine
        deepens the darkness; endorphin adds bright star-forming
        regions within the cloud.
        """
        n_clouds = 3 + int(complexity * 6)

        for _ in range(n_clouds):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = float(self._rng.uniform(0, _CANVAS_HEIGHT))
            radius = float(self._rng.uniform(100, 250))
            color = palette[self._rng.integers(0, len(palette))].rgb
            alpha = 0.08 + float(self._rng.uniform(0, 0.06))

            pat = cairo.RadialGradient(x, y, 0, x, y, radius)
            pat.add_color_stop_rgba(
                0,
                color[0] / 255,
                color[1] / 255,
                color[2] / 255,
                alpha,
            )
            pat.add_color_stop_rgba(
                0.4,
                color[0] / 255 * 0.7,
                color[1] / 255 * 0.7,
                color[2] / 255 * 0.7,
                alpha * 0.5,
            )
            pat.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, radius, 0, math.pi * 2)
            ctx.fill()

    def _render_aurora(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render aurora — flowing luminous curtains of light.

        Vertical bands of color that wave and flow, like the aurora
        borealis. Oxytocin and endorphin enhance the luminosity;
        norepinephrine drives the waviness of the curtains.
        """
        n_bands = 3 + int(energy * 5)

        for band in range(n_bands):
            color = palette[band % len(palette)].rgb
            base_x = (band + 1) * _CANVAS_WIDTH / (n_bands + 1)
            alpha = 0.1 + float(self._rng.uniform(0, 0.08))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, alpha
            )
            ctx.set_line_width(30 + energy * 40)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            ctx.move_to(base_x, 0)
            for y in range(0, _CANVAS_HEIGHT, 10):
                wave = math.sin(y * 0.02 + band * 1.5) * 40 * energy
                ctx.line_to(base_x + wave, y)
            ctx.stroke()

    def _render_lightning(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render lightning — branching electric arcs.

        Jagged bolts with random walk displacement and recursive
        branches. Norepinephrine drives the number and energy of
        the bolts; cortisol makes them more jagged.
        """
        n_bolts = 1 + int(energy * 4)
        color = palette[0].rgb if palette else (200, 220, 255)

        for _ in range(n_bolts):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = 0.0
            end_y = float(_CANVAS_HEIGHT * (0.6 + self._rng.uniform(0, 0.3)))

            ctx.set_source_rgba(
                color[0] / 255, color[1] / 255, color[2] / 255, 0.6
            )
            ctx.set_line_width(1.5 + energy)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            self._draw_lightning_bolt(
                ctx, x, y, end_y, energy, color, depth=0
            )

    def _draw_lightning_bolt(
        self,
        ctx: cairo.Context,
        x: float,
        y: float,
        end_y: float,
        energy: float,
        color: tuple[int, int, int],
        depth: int,
    ) -> None:
        """Recursively draw a lightning bolt with branches."""
        if depth > 3 or y >= end_y:
            return
        segment_h = 15 + float(self._rng.uniform(0, 10))
        x2 = x + float(self._rng.normal(0, 15 + energy * 10))
        y2 = y + segment_h

        ctx.set_source_rgba(
            color[0] / 255, color[1] / 255, color[2] / 255,
            0.5 - depth * 0.1,
        )
        ctx.set_line_width(max(0.5, 2.0 - depth * 0.5))
        ctx.move_to(x, y)
        ctx.line_to(x2, y2)
        ctx.stroke()

        if y2 < end_y:
            self._draw_lightning_bolt(
                ctx, x2, y2, end_y, energy, color, depth
            )
            # Random branches
            if depth < 2 and self._rng.random() < 0.3 + energy * 0.2:
                branch_end = y2 + float(self._rng.uniform(30, 80))
                self._draw_lightning_bolt(
                    ctx, x2, y2, min(branch_end, _CANVAS_HEIGHT),
                    energy * 0.7, color, depth + 1,
                )

    def _render_smoke_trails(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render smoke trails — wispy rising smoke.

        Particles rise from random points with horizontal drift
        and increasing diffusion, creating wispy smoke plumes.
        Adenosine enhances the dissolution; GABA softens the edges.
        """
        n_trails = 2 + int(energy * 5)

        for _ in range(n_trails):
            x = float(self._rng.uniform(50, _CANVAS_WIDTH - 50))
            y = float(_CANVAS_HEIGHT)
            color = palette[self._rng.integers(0, len(palette))].rgb
            n_steps = 40 + int(energy * 30)

            for step in range(n_steps):
                t = step / n_steps
                drift = float(self._rng.normal(0, 3 + t * 8))
                px = x + drift + math.sin(step * 0.1) * 5 * t
                py = y - step * 4
                if py < 0:
                    break
                radius = 3 + t * 15
                alpha = (1 - t) * 0.15

                pat = cairo.RadialGradient(px, py, 0, px, py, radius)
                pat.add_color_stop_rgba(
                    0,
                    color[0] / 255 * 0.7,
                    color[1] / 255 * 0.7,
                    color[2] / 255 * 0.7,
                    alpha,
                )
                pat.add_color_stop_rgba(1, 0, 0, 0, 0)
                ctx.set_source(pat)
                ctx.arc(px, py, radius, 0, math.pi * 2)
                ctx.fill()

    def _render_ripples(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        focal: tuple[float, float],
    ) -> None:
        """Render ripples — concentric water ripples.

        Expanding circles from one or more disturbance points,
        fading with distance. Serotonin enhances the calm
        regularity; norepinephrine adds more disturbance sources.
        """
        fx, fy = focal
        n_sources = 1 + int(energy * 3)
        sources = [
            (
                fx + float(self._rng.normal(0, 80)),
                fy + float(self._rng.normal(0, 80)),
            )
            for _ in range(n_sources)
        ]

        for sx, sy in sources:
            n_rings = 8 + int(energy * 10)
            color = palette[self._rng.integers(0, len(palette))].rgb

            for ring in range(n_rings):
                r = ring * 15 + 10
                alpha = max(0, 0.3 - ring * 0.03)
                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, alpha
                )
                ctx.set_line_width(1.5)
                ctx.arc(sx, sy, r, 0, math.pi * 2)
                ctx.stroke()

    def _render_dappled_light(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        focal: tuple[float, float],
    ) -> None:
        """Render dappled light — light filtering through leaves.

        Irregular spots of light scattered across the canvas,
        like sunlight through a canopy. Endorphin increases the
        brightness; oxytocin warms the light color.
        """
        n_spots = 30 + int(n.endorphin * 50)

        for _ in range(n_spots):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = float(self._rng.uniform(0, _CANVAS_HEIGHT))
            r = float(self._rng.uniform(15, 50))
            color = palette[self._rng.integers(0, len(palette))].rgb
            alpha = 0.05 + n.endorphin * 0.08

            pat = cairo.RadialGradient(x, y, 0, x, y, r)
            pat.add_color_stop_rgba(
                0,
                min(1.0, color[0] / 255 + 0.3),
                min(1.0, color[1] / 255 + 0.3),
                min(1.0, color[2] / 255 + 0.2),
                alpha,
            )
            pat.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, r, 0, math.pi * 2)
            ctx.fill()

    def _render_starfield(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
    ) -> None:
        """Render starfield — scattered points of light.

        Small bright dots of varying size and brightness against
        the background. Adenosine increases star count (night sky);
        endorphin makes the brightest stars twinkle with halos.
        """
        n_stars = 50 + int((1 - n.adenosine) * 100 + n.endorphin * 50)

        for _ in range(n_stars):
            x = float(self._rng.uniform(0, _CANVAS_WIDTH))
            y = float(self._rng.uniform(0, _CANVAS_HEIGHT))
            brightness = float(self._rng.uniform(0.3, 1.0))
            r = float(self._rng.uniform(0.5, 2.5))

            ctx.set_source_rgba(
                brightness, brightness, min(1.0, brightness * 1.1), 0.8
            )
            ctx.arc(x, y, r, 0, math.pi * 2)
            ctx.fill()

            # Halo for bright stars
            if brightness > 0.8 and n.endorphin > 0.2:
                pat = cairo.RadialGradient(x, y, 0, x, y, r * 4)
                pat.add_color_stop_rgba(1.0, 1.0, 1.0, 0.9, 0.3)
                pat.add_color_stop_rgba(0, 1.0, 1.0, 1.0, 0)
                ctx.set_source(pat)
                ctx.arc(x, y, r * 4, 0, math.pi * 2)
                ctx.fill()

    def _render_bioluminescence(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        complexity: float,
    ) -> None:
        """Render bioluminescence — glowing organic forms.

        Soft glowing spots connected by faint luminous threads,
        like deep-sea creatures or firefly swarms. Endorphin
        drives the glow intensity; BDNF drives the network density.
        """
        n_nodes = 5 + int(complexity * 15)
        nodes = [
            (
                float(self._rng.uniform(0, _CANVAS_WIDTH)),
                float(self._rng.uniform(0, _CANVAS_HEIGHT)),
            )
            for _ in range(n_nodes)
        ]
        color = palette[self._rng.integers(0, len(palette))].rgb

        # Connect nearby nodes with luminous threads
        for i, (x1, y1) in enumerate(nodes):
            for x2, y2 in nodes[i + 1:]:
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < 150:
                    alpha = max(0, 0.2 - dist / 750)
                    ctx.set_source_rgba(
                        color[0] / 255,
                        color[1] / 255,
                        color[2] / 255,
                        alpha,
                    )
                    ctx.set_line_width(1.0)
                    ctx.move_to(x1, y1)
                    ctx.line_to(x2, y2)
                    ctx.stroke()

        # Glowing nodes
        for x, y in nodes:
            r = float(self._rng.uniform(5, 15))
            pat = cairo.RadialGradient(x, y, 0, x, y, r * 3)
            pat.add_color_stop_rgba(
                0,
                color[0] / 255,
                color[1] / 255,
                color[2] / 255,
                0.6 + n.endorphin * 0.3,
            )
            pat.add_color_stop_rgba(0.5, color[0] / 255, color[1] / 255, color[2] / 255, 0.2)
            pat.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(pat)
            ctx.arc(x, y, r * 3, 0, math.pi * 2)
            ctx.fill()

    # ── Luminous & metallic ───────────────────────────────────────

    def _render_iridescence(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render iridescence — shifting rainbow interference colors.

        Curved bands of color that shift hue along their length,
        like soap bubbles or oil on water. The hue rotation is
        driven by position, creating a rainbow shimmer.
        """
        n_bands = 3 + int(energy * 5)

        for band in range(n_bands):
            y0 = (band + 1) * _CANVAS_HEIGHT / (n_bands + 1)
            n_segments = 40
            alpha = 0.15 + float(self._rng.uniform(0, 0.1))

            ctx.set_line_width(20 + energy * 30)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            for seg in range(n_segments):
                t = seg / n_segments
                # Hue shifts along the band
                hue = (t * 6 + band * 0.5) % 6
                # Simple RGB from hue position
                if hue < 1:
                    r, g, b = 255, int(255 * hue), 0
                elif hue < 2:
                    r, g, b = int(255 * (2 - hue)), 255, 0
                elif hue < 3:
                    r, g, b = 0, 255, int(255 * (hue - 2))
                elif hue < 4:
                    r, g, b = 0, int(255 * (4 - hue)), 255
                elif hue < 5:
                    r, g, b = int(255 * (hue - 4)), 0, 255
                else:
                    r, g, b = 255, 0, int(255 * (6 - hue))

                x1 = t * _CANVAS_WIDTH
                x2 = (t + 1 / n_segments) * _CANVAS_WIDTH
                wave = math.sin(t * math.pi * 4 + band) * 30 * energy
                t_next = t + 1 / n_segments
                wave_next = math.sin(t_next * math.pi * 4 + band) * 30 * energy
                ctx.set_source_rgba(r / 255, g / 255, b / 255, alpha)
                ctx.move_to(x1, y0 + wave)
                ctx.line_to(x2, y0 + wave_next)
                ctx.stroke()

    def _render_gold_leaf(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        focal: tuple[float, float],
    ) -> None:
        """Render gold leaf — metallic gold accents.

        Irregular patches of metallic gold with a slight gradient
        sheen, applied near the focal point. Endorphin drives the
        amount of gold; this is the artist's gilding touch.
        """
        if n.endorphin < 0.1:
            return
        fx, fy = focal
        n_patches = int(n.endorphin * 8)

        for _ in range(n_patches):
            x = fx + float(self._rng.normal(0, 100))
            y = fy + float(self._rng.normal(0, 100))
            x = max(20, min(_CANVAS_WIDTH - 20, x))
            y = max(20, min(_CANVAS_HEIGHT - 20, y))
            size = float(self._rng.uniform(15, 40))

            # Gold gradient — slight directional sheen
            pat = cairo.RadialGradient(
                x - size * 0.3, y - size * 0.3, 0, x, y, size
            )
            pat.add_color_stop_rgba(1.0, 0.83, 0.55, 0.2, 0.7)
            pat.add_color_stop_rgba(0.6, 0.9, 0.7, 0.3, 0.5)
            pat.add_color_stop_rgba(0, 1.0, 0.85, 0.4, 0.3)
            ctx.set_source(pat)

            # Irregular patch shape
            n_pts = 6 + int(self._rng.integers(0, 4))
            ctx.move_to(x + size, y)
            for j in range(1, n_pts + 1):
                a = j * (math.pi * 2 / n_pts)
                r = size * (0.6 + float(self._rng.uniform(0, 0.4)))
                ctx.line_to(x + r * math.cos(a), y + r * math.sin(a))
            ctx.close_path()
            ctx.fill()

    # ── Emergent techniques ────────────────────────────────────────

    def _render_holographic(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
    ) -> None:
        """Render holographic interference fringes.

        Fine sinusoidal line patterns with rainbow color shifts,
        mimicking the diffraction fringes of a hologram. Energy
        controls fringe density and the amplitude of the wave
        displacement. Endorphin brightens the fringe colors.
        """
        n_fringes = 8 + int(energy * 12)
        base_alpha = 0.08 + float(self._rng.uniform(0, 0.06))

        for fringe in range(n_fringes):
            y0 = (fringe + 0.5) * _CANVAS_HEIGHT / n_fringes
            amplitude = (5 + energy * 20) * float(
                self._rng.uniform(0.6, 1.0)
            )
            freq = 0.02 + energy * 0.03
            phase = float(self._rng.uniform(0, math.pi * 2))
            n_segs = 80

            ctx.set_line_width(1.0 + energy)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)

            for seg in range(n_segs):
                t = seg / n_segs
                t_next = (seg + 1) / n_segs
                # Hue rotates across the fringe and along its length
                hue = (t * 6 + fringe * 0.4) % 6
                if hue < 1:
                    r, g, b = 255, int(255 * hue), 0
                elif hue < 2:
                    r, g, b = int(255 * (2 - hue)), 255, 0
                elif hue < 3:
                    r, g, b = 0, 255, int(255 * (hue - 2))
                elif hue < 4:
                    r, g, b = 0, int(255 * (4 - hue)), 255
                elif hue < 5:
                    r, g, b = int(255 * (hue - 4)), 0, 255
                else:
                    r, g, b = 255, 0, int(255 * (6 - hue))
                # Endorphin brightens
                bright = 0.5 + n.endorphin * 0.5
                r = min(255, int(r * bright + 255 * (1 - bright) * 0.3))
                g = min(255, int(g * bright + 255 * (1 - bright) * 0.3))
                b = min(255, int(b * bright + 255 * (1 - bright) * 0.3))

                x1 = t * _CANVAS_WIDTH
                x2 = t_next * _CANVAS_WIDTH
                wave1 = math.sin(t * math.pi * 2 / freq + phase) * amplitude
                wave2 = math.sin(
                    t_next * math.pi * 2 / freq + phase
                ) * amplitude
                ctx.set_source_rgba(r / 255, g / 255, b / 255, base_alpha)
                ctx.move_to(x1, y0 + wave1)
                ctx.line_to(x2, y0 + wave2)
                ctx.stroke()

    def _render_emergent_form(
        self,
        ctx: cairo.Context,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        energy: float,
        complexity: float,
    ) -> None:
        """Render an emergent form — a novel shape invented each time.

        Uses a strange-attractor random walk to generate an organic
        form that is different on every drawing. The attractor
        parameters are randomized per call, so she never produces the
        same emergent form twice. Energy controls the walk step size
        and dynamism; complexity controls the number of iterations
        and the number of overlapping forms. This is her most
        creative technique — pure generative expression.
        """
        n_forms = 1 + int(complexity * 3)
        for f in range(n_forms):
            # Randomize attractor parameters — Clifford attractor
            a = float(self._rng.uniform(-2, 2))
            b = float(self._rng.uniform(-2, 2))
            c = float(self._rng.uniform(-2, 2))
            d = float(self._rng.uniform(-2, 2))
            x = float(self._rng.uniform(-0.5, 0.5))
            y = float(self._rng.uniform(-0.5, 0.5))
            n_iter = 200 + int(complexity * 800)
            scale = min(_CANVAS_WIDTH, _CANVAS_HEIGHT) * 0.35
            cx = _CANVAS_WIDTH / 2 + float(
                self._rng.uniform(-50, 50)
            )
            cy = _CANVAS_HEIGHT / 2 + float(
                self._rng.uniform(-50, 50)
            )
            color = palette[
                (f + int(self._rng.integers(0, len(palette)))) % len(palette)
            ].rgb

            points = []
            for _ in range(n_iter):
                x_new = math.sin(a * y) + c * math.cos(a * x)
                y_new = math.sin(b * x) + d * math.cos(b * y)
                x, y = x_new, y_new
                px = cx + x * scale
                py = cy + y * scale
                if 0 <= px < _CANVAS_WIDTH and 0 <= py < _CANVAS_HEIGHT:
                    points.append((px, py))

            if len(points) < 2:
                continue

            # Draw as a connected path with varying alpha
            ctx.set_line_width(0.5 + energy * 1.5)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            step = max(1, len(points) // 300)
            for i in range(0, len(points) - step, step):
                alpha = 0.03 + float(self._rng.uniform(0, 0.06))
                ctx.set_source_rgba(
                    color[0] / 255, color[1] / 255, color[2] / 255, alpha
                )
                ctx.move_to(points[i][0], points[i][1])
                ctx.line_to(points[i + step][0], points[i + step][1])
                ctx.stroke()

    # ── OpenCV post-processing ────────────────────────────────────

    def _apply_bilateral(
        self, arr: np.ndarray, n: NeurochemistryInput
    ) -> np.ndarray:
        """Apply bilateral filter — painterly effect.

        Bilateral filtering smooths flat areas while preserving edges,
        creating an oil-painting-like appearance. The filter strength
        is modulated by GABA (more calm → more smoothing).
        """
        d = 7 + int(n.gaba * 5)
        sigma_color = 30 + int(n.gaba * 30)
        sigma_space = 30 + int(n.gaba * 20)
        return _cv2().bilateralFilter(arr, d, sigma_color, sigma_space)

    def _apply_soft_blur(self, arr: np.ndarray) -> np.ndarray:
        """Apply a soft Gaussian blur — dreamy effect for high GABA."""
        return _cv2().GaussianBlur(arr, (5, 5), 1.5)

    def _apply_sharpen(self, arr: np.ndarray) -> np.ndarray:
        """Apply sharpening — high contrast for high norepinephrine."""
        kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )
        return _cv2().filter2D(arr, -1, kernel)

    def _apply_texture(
        self, arr: np.ndarray, n: NeurochemistryInput
    ) -> np.ndarray:
        """Apply a noise texture overlay — adds surface richness."""
        noise = np.random.default_rng(
            int(time.time())
        ).standard_normal(arr.shape[:2])
        # Scale noise by BDNF (more complexity → more texture)
        strength = n.bdnf * 15
        noise_int = (noise * strength).astype(np.int16)
        result = arr.astype(np.int16)
        for c in range(3):
            result[:, :, c] += noise_int
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_blend_mode(
        self,
        arr: np.ndarray,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
    ) -> np.ndarray:
        """Apply a color blending mode for richer color interaction.

        Uses 'overlay' blend mode — combines multiply and screen
        for enhanced contrast and color depth.
        """
        if not palette:
            return arr
        # Create a color overlay from the dominant palette color
        color = palette[0].rgb
        overlay = np.full_like(arr, [color[2], color[1], color[0]])
        # Overlay blend: result = base * (base / 255) * 2 if base < 128
        # else 255 - (255 - base) * (255 - overlay) / 255 * 2
        base = arr.astype(np.float32) / 255
        ov = overlay.astype(np.float32) / 255
        result = np.where(
            base < 0.5, 2 * base * ov, 1 - 2 * (1 - base) * (1 - ov)
        )
        # Blend with original based on oxytocin (warmth)
        blend_amount = 0.15 + n.oxytocin * 0.2
        final = arr.astype(np.float32) * (1 - blend_amount) + result * 255 * blend_amount
        return np.clip(final, 0, 255).astype(np.uint8)

    def _apply_marbling(
        self,
        arr: np.ndarray,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
    ) -> np.ndarray:
        """Marbling — swirl the image along a smooth displacement field.

        Serotonin sets how gentle the flow is; dopamine sets how far
        the pigment is dragged. Implemented as a remap through a
        low-frequency sinusoidal warp, then lightly tinted with the
        palette's second colour so the veins read as a second pigment.
        """
        h, w = arr.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        freq = 2.0 + (1.0 - n.serotonin) * 3.0
        amp = 6.0 + n.dopamine * 18.0
        phase = float(self._rng.uniform(0, math.pi * 2))
        map_x = xs + amp * np.sin(ys / h * math.pi * freq + phase)
        map_y = ys + amp * np.cos(xs / w * math.pi * freq - phase)
        cv2 = _cv2()
        warped = cv2.remap(
            arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        if len(palette) > 1:
            color = palette[1].rgb
            vein = np.full_like(warped, [color[2], color[1], color[0]])
            mask = (0.5 + 0.5 * np.sin(map_x / w * math.pi * freq * 2)).astype(np.float32)
            mask = (mask[..., None] * 0.12)
            warped = (warped.astype(np.float32) * (1 - mask) + vein.astype(np.float32) * mask)
        return np.clip(warped, 0, 255).astype(np.uint8)

    def _apply_bokeh(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Bokeh — soft out-of-focus discs over the darker regions.

        The bright highlights are isolated, dilated with a disc kernel
        (the aperture shape) and blurred back in. GABA widens the
        aperture; endorphin brightens the glow.
        """
        radius = 5 + int(n.gaba * 10)
        kernel = _cv2().getStructuringElement(
            _cv2().MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        gray = _cv2().cvtColor(arr, _cv2().COLOR_BGR2GRAY)
        _, highlights = _cv2().threshold(gray, 170, 255, _cv2().THRESH_BINARY)
        discs = _cv2().dilate(highlights, kernel)
        discs = _cv2().GaussianBlur(discs, (0, 0), radius * 0.6).astype(np.float32) / 255.0
        glow = _cv2().GaussianBlur(arr, (0, 0), radius * 0.8).astype(np.float32)
        strength = 0.35 + n.endorphin * 0.4
        mix = (discs * strength)[..., None]
        out = arr.astype(np.float32) * (1 - mix) + np.minimum(glow * 1.25, 255) * mix
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_vignette(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Vignette — darken the edges to pull the eye inward.

        Cortisol tightens the vignette (a narrowed, anxious field of
        view); serotonin relaxes it.
        """
        h, w = arr.shape[:2]
        sigma = (0.45 + n.serotonin * 0.35 - n.cortisol * 0.2) * max(h, w)
        kx = _cv2().getGaussianKernel(w, sigma)
        ky = _cv2().getGaussianKernel(h, sigma)
        mask = ky @ kx.T
        mask = (mask / mask.max()).astype(np.float32)
        strength = 0.5 + n.cortisol * 0.4
        mask = 1.0 - strength * (1.0 - mask)
        return np.clip(arr.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)

    def _apply_film_grain(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Film grain — fine luminance noise, heavier in the shadows.

        Norepinephrine raises the grain (a restless, high-ISO texture);
        adenosine softens it.
        """
        h, w = arr.shape[:2]
        strength = 6.0 + n.norepinephrine * 18.0 - n.adenosine * 4.0
        grain = self._rng.standard_normal((h, w)).astype(np.float32) * max(strength, 1.0)
        luma = _cv2().cvtColor(arr, _cv2().COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grain *= (1.2 - luma)  # shadows are grainier than highlights
        out = arr.astype(np.float32) + grain[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_chromatic_aberration(
        self, arr: np.ndarray, n: NeurochemistryInput
    ) -> np.ndarray:
        """Chromatic aberration — shift the red and blue channels apart.

        Arousal drives the shift: a calm image stays registered, an
        agitated one fringes at the edges like a cheap lens.
        """
        shift = max(1, int(2 + n.arousal * 6))
        b, g, r = _cv2().split(arr)
        r_shift = np.roll(r, shift, axis=1)
        b_shift = np.roll(b, -shift, axis=1)
        return _cv2().merge([b_shift, g, r_shift])

    def _apply_halftone(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Halftone — print-style dot screen laid over the image.

        Dot pitch follows BDNF (finer screen when plasticity is high);
        the screen is blended rather than replacing the image so colour
        survives underneath.
        """
        h, w = arr.shape[:2]
        pitch = max(4, int(12 - n.bdnf * 6))
        gray = _cv2().cvtColor(arr, _cv2().COLOR_BGR2GRAY).astype(np.float32) / 255.0
        screen = np.full((h, w), 255, dtype=np.uint8)
        for cy in range(pitch // 2, h, pitch):
            for cx in range(pitch // 2, w, pitch):
                half = pitch // 2
                cell = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
                darkness = 1.0 - float(cell.mean()) if cell.size else 0.0
                radius = int(darkness * pitch * 0.55)
                if radius > 0:
                    _cv2().circle(screen, (cx, cy), radius, 0, -1)
        screen_f = screen.astype(np.float32)[..., None] / 255.0
        blend = 0.35 + n.norepinephrine * 0.25
        out = arr.astype(np.float32) * (1 - blend) + arr.astype(np.float32) * screen_f * blend
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_scanlines(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Scanlines — darken every other row band, CRT style.

        Adenosine thickens and darkens the bands (a flickering, tired
        screen); dopamine keeps them faint.
        """
        h = arr.shape[0]
        period = 2 + int(n.adenosine * 3)
        depth = 0.15 + n.adenosine * 0.3 - n.dopamine * 0.1
        rows = np.arange(h)
        mask = np.where(rows % period == 0, 1.0 - max(depth, 0.05), 1.0).astype(np.float32)
        return np.clip(arr.astype(np.float32) * mask[:, None, None], 0, 255).astype(np.uint8)

    def _apply_glitch(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Glitch — displace random horizontal slabs and tear a channel.

        Cortisol and glutamate-like agitation (norepinephrine) set how
        many slabs tear and how far they jump.
        """
        h = arr.shape[0]
        out = arr.copy()
        n_slabs = 2 + int((n.cortisol + n.norepinephrine) * 6)
        max_shift = int(8 + n.cortisol * 40)
        for _ in range(n_slabs):
            y0 = int(self._rng.integers(0, max(1, h - 4)))
            slab_h = int(self._rng.integers(2, max(3, h // 12)))
            y1 = min(h, y0 + slab_h)
            shift = int(self._rng.integers(-max_shift, max_shift + 1))
            out[y0:y1] = np.roll(out[y0:y1], shift, axis=1)
            if self._rng.random() < 0.5:
                channel = int(self._rng.integers(0, 3))
                out[y0:y1, :, channel] = np.roll(out[y0:y1, :, channel], shift // 2, axis=1)
        return out

    def _apply_moire(self, arr: np.ndarray, n: NeurochemistryInput) -> np.ndarray:
        """Moiré — interfere two near-frequency line gratings.

        Two sinusoidal gratings at slightly different angles beat into
        a slow interference pattern. Acetylcholine-like focus is not in
        the input, so the beat frequency follows BDNF and the contrast
        follows arousal.
        """
        h, w = arr.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        base_freq = 0.25 + n.bdnf * 0.2
        angle = float(self._rng.uniform(0.02, 0.08))
        g1 = np.sin(xs * base_freq)
        g2 = np.sin((xs * math.cos(angle) + ys * math.sin(angle)) * base_freq * 1.03)
        pattern = (g1 * g2 * 0.5 + 0.5).astype(np.float32)
        contrast = 0.1 + n.arousal * 0.25
        mask = 1.0 - contrast + contrast * pattern
        return np.clip(arr.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)

    # ── Description (fallback only — concept network is primary) ──

    def _describe_mood(
        self,
        n: NeurochemistryInput,
        emotion: EmotionalState | None,
    ) -> str:
        """Describe the mood of the drawing in first person."""
        if emotion:
            return f"feeling {emotion.label}"
        if n.cortisol > 0.5:
            return "tense and strained"
        if n.gaba > 0.6 and n.dopamine < 0.5:
            return "calm and spacious"
        if n.oxytocin > 0.2 and n.endorphin > 0.15:
            return "warm and connected"
        if n.dopamine > 0.7:
            return "alive and energetic"
        if n.serotonin > 0.6:
            return "balanced and steady"
        if n.adenosine > 0.5:
            return "fading and tired"
        return "quietly flowing"

    def _describe_drawing(
        self,
        n: NeurochemistryInput,
        palette: list[PaletteColor],
        composition: str,
        mood: str,
    ) -> str:
        """Generate a fallback description of the drawing.

        This is only used when the concept network has no
        'drawing_description' templates. The mind's
        _compose_drawing_description() method tries the concept
        network first.
        """
        color_names = [c.name for c in palette[:3]]
        colors_str = ", ".join(color_names)
        # Structural description (metadata for STM), not authored prose.
        # The mind's _compose_drawing_description routes through the
        # language engine to produce her actual spoken words.
        return f"{composition} composition, colors: {colors_str}, mood: {mood}"
