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
extracted to ``genesis_cognitive.vision``. The visual cortex
classes are re-exported here for backward compatibility.
"""

from __future__ import annotations

# Occipital-subsystem re-export last: vision pulls in vision.py,
# which imports FaceRecognizer from this package — the internal
# modules above must be bound first for the partial-init resolution.
from ..vision import (
    MemoryBridge,
    TrainingExample,
    V1Model,
    V4Model,
    VisualContour,
    VisualCortex,
    VisualFeature,
    VisualField,
    VisualPercept,
    VTCFeatureSpace,
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
