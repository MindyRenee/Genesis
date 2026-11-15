"""Perception bundle — core perception, intent, and face recognition.

This package bundles Genesis's perceptual subsystems into a coherent
unit. Consumers import from here rather than individual modules.

Subsystems:
    Perception, Intent, QuestionType — core perception and intent
        classification
    IntegratedPerception — integrated perception pipeline
    perceive, compute_intent_probabilities — perception functions
    FaceRecognizer, DetectedFace, KnownFace — face recognition

The occipital subsystem (V1, V4, VTC, MTL bridge, VisualCortex) has been
extracted to ``occipital_lobe``. The visual cortex
classes are re-exported here for backward compatibility.
"""

from __future__ import annotations

# Backward-compat re-exports of the occipital/temporal visual names
# (see module docstring) — bound before the internal modules so any
# mid-init resolution still finds them.
from temporal_lobe.mtl_bridge import MemoryBridge, TrainingExample
from temporal_lobe.vtc import VTCFeatureSpace

from occipital_lobe import (
    V1Model,
    V4Model,
    VisualContour,
    VisualCortex,
    VisualFeature,
    VisualField,
    VisualPercept,
    decode_image_bytes,
    load_image,
    resize_for_vision,
)

from .core import (
    IntegratedPerception,
    Intent,
    MultisensoryInput,
    MultisensoryIntegrator,
    PatternWeights,
    Perception,
    QuestionType,
    SensoryModality,
    compute_intent_probabilities,
    perceive,
)
from .recognition import DetectedFace, FaceRecognizer, KnownFace

__all__ = [
    "DetectedFace",
    "FaceRecognizer",
    "IntegratedPerception",
    "Intent",
    "KnownFace",
    "MemoryBridge",
    "MultisensoryInput",
    "MultisensoryIntegrator",
    "PatternWeights",
    "Perception",
    "QuestionType",
    "SensoryModality",
    "TrainingExample",
    "V1Model",
    "V4Model",
    "VTCFeatureSpace",
    "VisualContour",
    "VisualCortex",
    "VisualFeature",
    "VisualField",
    "VisualPercept",
    "compute_intent_probabilities",
    "decode_image_bytes",
    "load_image",
    "perceive",
    "resize_for_vision",
]
