"""Named API surface — pins for public query/metadata with no internal
consumer. These exist to be read by introspection, tooling, and the
organism's own descriptive surface; the pin keeps them honest."""

from __future__ import annotations

import numpy as np
import pytest
from temporal_lobe.presence import PresenceKind
from worlds.world import OuterWorld

# ─── Chemical spec metadata ─────────────────────────────────────────

_SPEC_CLASSES = [
    "brainstem.norepinephrine.Norepinephrine",
    "brainstem.epinephrine.Epinephrine",
    "brainstem.serotonin.Serotonin",
    "basal_ganglia.dopamine.Dopamine",
    "basal_ganglia.gaba.GABA",
    "basal_forebrain.acetylcholine.Acetylcholine",
    "basal_forebrain.adenosine.Adenosine",
    "cerebellum.endocannabinoid.Endocannabinoid",
    "frontal_lobe.glutamate.Glutamate",
    "hypothalamus.cortisol.Cortisol",
    "hypothalamus.crh.CRH",
    "hypothalamus.histamine.Histamine",
    "hypothalamus.melatonin.Melatonin",
    "hypothalamus.orexin.Orexin",
    "hypothalamus.oxytocin.Oxytocin",
    "hypothalamus.vasopressin.Vasopressin",
    "limbic_system.endorphin.Endorphin",
    "temporal_lobe.bdnf.BDNF",
]


def _load(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


@pytest.mark.parametrize("spec_path", _SPEC_CLASSES)
def test_chemical_spec_metadata_is_readable(spec_path):
    """Every chemical spec carries its descriptive metadata — the
    receptors it signals through and its circadian rule — as named
    fields a spec reader can consume."""
    spec = _load(spec_path)()
    assert isinstance(spec.receptors, tuple) and spec.receptors
    assert all(isinstance(r, str) and r for r in spec.receptors)
    assert isinstance(spec.circadian_modifier, str) and spec.circadian_modifier


# ─── World presence queries ─────────────────────────────────────────


def test_presence_queries():
    """presence/presences/present_presences expose the social world."""
    world = OuterWorld()
    assert world.presence("user") is None
    assert world.presences() == []
    assert world.present_presences() == []

    world.mark_presence("user", PresenceKind.USER)
    p = world.presence("user")
    assert p is not None and p.kind == PresenceKind.USER
    assert [q.presence_id for q in world.presences()] == ["user"]
    assert [q.presence_id for q in world.present_presences()] == ["user"]


# ─── Conversation sleep state ───────────────────────────────────────


def test_is_user_sleeping_reads_both_flags():
    """is_user_sleeping is True only when asleep AND user-initiated."""
    from types import SimpleNamespace

    from worlds.conversation import ConversationMixin

    read = ConversationMixin.is_user_sleeping.fget
    stub = SimpleNamespace(_is_sleeping=True, _user_initiated_sleep=True)
    assert read(stub) is True
    stub._user_initiated_sleep = False
    assert read(stub) is False
    stub._is_sleeping = False
    stub._user_initiated_sleep = True
    assert read(stub) is False


# ─── Embedding similarity ───────────────────────────────────────────


def test_cosine_similarity():
    """cosine_similarity is the embedding store's geometry primitive."""
    from temporal_lobe.concepts.embeddings import EmbeddingStore

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert EmbeddingStore.cosine_similarity(None, a, b) == pytest.approx(1.0)
    assert EmbeddingStore.cosine_similarity(None, a, c) == pytest.approx(0.0)
    assert EmbeddingStore.cosine_similarity(None, a, np.zeros(3)) == 0.0
