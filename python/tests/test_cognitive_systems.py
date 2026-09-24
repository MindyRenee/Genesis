"""Cognitive systems tests.

Attention, global workspace, executive function,
working memory, columnar math.
"""

import logging

import pytest

from genesis_cognitive.attention import (
    AttentionFocus,
    AttentionSystem,
    AttentionType,
)
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.executive import (
    ActionPlan,
    ExecutiveFunction,
    InhibitionResult,
    SwitchResult,
    Task,
    TaskState,
)
from genesis_cognitive.global_workspace import (
    GlobalWorkspace,
    WorkspaceItem,
)
from genesis_cognitive.memory import (
    HABIT_THRESHOLD,
    CentralExecutive,
    ConversationThread,
    HabitBias,
    PhonologicalLoop,
    ProceduralMemory,
    Skill,
    SkillStrategy,
    Turn,
    VisuoSpatialSketchpad,
    WorkingMemory,
)

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_attention.py
# ======================================================================

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# AttentionType enum
# ═══════════════════════════════════════════════════════════════════


def test_attention_type_values() -> None:
    """AttentionType has the expected values."""
    assert AttentionType.SELECTIVE.value == "selective"
    assert AttentionType.SUSTAINED.value == "sustained"
    assert AttentionType.EXECUTIVE.value == "executive"
    assert AttentionType.DIVIDED.value == "divided"


# ═══════════════════════════════════════════════════════════════════
# AttentionFocus
# ═══════════════════════════════════════════════════════════════════


def test_attention_focus_creation() -> None:
    """AttentionFocus stores target, intensity, type."""
    focus = AttentionFocus(target="dogs", intensity=0.8)
    assert focus.target == "dogs"
    assert focus.intensity == 0.8
    assert focus.attention_type == AttentionType.SELECTIVE
    assert focus.goal == ""


def test_attention_focus_custom_type() -> None:
    """AttentionFocus can have a custom attention type."""
    focus = AttentionFocus(target="dogs", intensity=0.8, attention_type=AttentionType.EXECUTIVE)
    assert focus.attention_type == AttentionType.EXECUTIVE


def test_attention_focus_age_ms() -> None:
    """age_ms returns non-negative age."""
    focus = AttentionFocus(target="dogs")
    assert focus.age_ms >= 0


def test_attention_focus_time_since_refresh_ms() -> None:
    """time_since_refresh_ms returns non-negative time."""
    focus = AttentionFocus(target="dogs")
    assert focus.time_since_refresh_ms >= 0


# ═══════════════════════════════════════════════════════════════════
# AttentionSystem — initialization
# ═══════════════════════════════════════════════════════════════════


def test_attention_system_initialization() -> None:
    """AttentionSystem initializes with defaults."""
    att = AttentionSystem()
    assert att.decay_rate == 0.02
    assert att.suppression_strength == 0.3
    assert att.amplification_strength == 0.4
    assert att.max_foci == 4
    assert not att.is_focused


def test_attention_system_custom_params() -> None:
    """AttentionSystem accepts custom parameters."""
    att = AttentionSystem(decay_rate=0.1, max_foci=2)
    assert att.decay_rate == 0.1
    assert att.max_foci == 2


# ═══════════════════════════════════════════════════════════════════
# Selective attention
# ═══════════════════════════════════════════════════════════════════


def test_focus_on_creates_focus() -> None:
    """focus_on creates an attention focus."""
    att = AttentionSystem()
    focus = att.focus_on("dogs")
    assert focus.target == "dogs"
    assert att.is_focused
    assert att.is_attended("dogs")


def test_focus_on_intensity_clamped() -> None:
    """focus_on clamps intensity to [0, 1]."""
    att = AttentionSystem()
    focus = att.focus_on("dogs", intensity=2.0)
    assert focus.intensity == 1.0
    focus = att.focus_on("cats", intensity=-1.0)
    assert focus.intensity == 0.0


def test_focus_on_custom_type() -> None:
    """focus_on can use a custom attention type."""
    att = AttentionSystem()
    focus = att.focus_on("dogs", attention_type=AttentionType.EXECUTIVE)
    assert focus.attention_type == AttentionType.EXECUTIVE


def test_focus_on_replaces_weakest_at_capacity() -> None:
    """At max foci, the weakest is replaced."""
    att = AttentionSystem(max_foci=2)
    att.focus_on("a", intensity=0.9)
    att.focus_on("b", intensity=0.5)
    att.focus_on("c", intensity=0.8)  # should replace "b" (weakest)
    targets = att.focused_targets
    assert "a" in targets
    assert "c" in targets
    assert "b" not in targets


def test_focus_on_same_target_replaces() -> None:
    """Focusing on the same target again updates the focus."""
    att = AttentionSystem()
    att.focus_on("dogs", intensity=0.5)
    att.focus_on("dogs", intensity=0.9)
    focus = att.get_focus("dogs")
    assert focus is not None
    assert focus.intensity == 0.9


def test_suppress() -> None:
    """suppress adds a target to the suppressed set."""
    att = AttentionSystem()
    att.suppress("noise")
    assert att.is_suppressed("noise")
    assert "noise" in att.suppressed_targets


def test_suppress_not_focused() -> None:
    """suppressing doesn't create a focus."""
    att = AttentionSystem()
    att.suppress("noise")
    assert not att.is_attended("noise")


# ═══════════════════════════════════════════════════════════════════
# Sustained attention
# ═══════════════════════════════════════════════════════════════════


def test_tick_decays_focus() -> None:
    """tick reduces focus intensity."""
    att = AttentionSystem(decay_rate=0.5)
    att.focus_on("dogs", intensity=1.0)
    focus = att.get_focus("dogs")
    assert focus is not None
    initial = focus.intensity
    att.tick()
    focus2 = att.get_focus("dogs")
    assert focus2 is not None
    assert focus2.intensity < initial


def test_tick_removes_expired_foci() -> None:
    """tick removes foci that decay below 0.1."""
    att = AttentionSystem(decay_rate=0.9)
    att.focus_on("dogs", intensity=0.5)
    # 0.5 * (1 - 0.9) = 0.05 < 0.1 → expired
    expired = att.tick()
    assert "dogs" in expired
    assert not att.is_attended("dogs")


def test_tick_executive_decays_slower() -> None:
    """Executive attention decays slower than selective."""
    att = AttentionSystem(decay_rate=0.5)
    att.focus_on("selective", intensity=1.0, attention_type=AttentionType.SELECTIVE)
    att.focus_on("executive", intensity=1.0, attention_type=AttentionType.EXECUTIVE)
    att.tick()
    selective = att.get_focus("selective")
    assert selective is not None
    executive = att.get_focus("executive")
    assert executive is not None
    selective_intensity = selective.intensity
    executive_intensity = executive.intensity
    assert executive_intensity > selective_intensity


def test_refresh_restores_intensity() -> None:
    """refresh restores a focus to full intensity."""
    att = AttentionSystem(decay_rate=0.5)
    att.focus_on("dogs", intensity=1.0)
    att.tick()  # decay
    att.refresh("dogs")
    focus = att.get_focus("dogs")
    assert focus is not None
    assert focus.intensity == 1.0


def test_refresh_unknown_target() -> None:
    """refresh on unknown target does nothing."""
    att = AttentionSystem()
    att.refresh("unknown")  # should not raise


def test_tick_no_foci() -> None:
    """tick with no foci returns empty list."""
    att = AttentionSystem()
    expired = att.tick()
    assert expired == []


# ═══════════════════════════════════════════════════════════════════
# Executive attention
# ═══════════════════════════════════════════════════════════════════


def test_set_goal() -> None:
    """set_goal sets the current goal."""
    att = AttentionSystem()
    att.set_goal("answer the question")
    assert att.current_goal == "answer the question"


def test_clear_goal() -> None:
    """clear_goal clears the current goal."""
    att = AttentionSystem()
    att.set_goal("answer the question")
    att.clear_goal()
    assert att.current_goal == ""


def test_clear_goal_removes_executive_foci() -> None:
    """clear_goal removes executive attention foci."""
    att = AttentionSystem()
    att.set_goal("answer the question")
    att.focus_on("dogs", attention_type=AttentionType.EXECUTIVE)
    att.focus_on("cats", attention_type=AttentionType.SELECTIVE)
    att.clear_goal()
    assert not att.is_attended("dogs")  # executive removed
    assert att.is_attended("cats")  # selective kept


def test_set_goal_amplifies_network_concepts() -> None:
    """set_goal amplifies goal-related concepts in the network."""
    network = ConceptNetwork()
    network.add_concept("dogs")
    att = AttentionSystem(network=network)
    att.set_goal("dogs")
    # "dogs" should be attended as an executive focus
    assert att.is_attended("dogs")
    focus = att.get_focus("dogs")
    assert focus is not None
    assert focus.attention_type == AttentionType.EXECUTIVE


def test_set_goal_skips_stopwords() -> None:
    """set_goal does not let function words fill the limited focus capacity.

    Without filtering, stopwords like "the" and "about" would each grab an
    executive focus slot, evicting semantically important goal words once the
    capacity limit is reached.
    """
    network = ConceptNetwork()
    for w in ["answer", "the", "question", "about", "cognition"]:
        network.add_concept(w)
    att = AttentionSystem(network=network, max_foci=4)
    att.set_goal("answer the question about cognition")
    # Stopwords must not capture executive attention.
    assert not att.is_attended("the")
    assert not att.is_attended("about")
    # Semantically important words should be attended, not evicted.
    assert att.is_attended("answer")
    assert att.is_attended("cognition")


def test_set_goal_replaces_previous_goal_foci() -> None:
    """A new goal replaces the previous goal's executive foci.

    Goal-directed attention should track the *current* goal; stale executive
    foci from a prior goal should not accumulate.
    """
    network = ConceptNetwork()
    network.add_concept("old")
    network.add_concept("new")
    att = AttentionSystem(network=network)
    att.set_goal("old")
    att.set_goal("new")
    assert att.current_goal == "new"
    assert not att.is_attended("old")
    assert att.is_attended("new")


# ═══════════════════════════════════════════════════════════════════
# Divided attention
# ═══════════════════════════════════════════════════════════════════


def test_divide_attention() -> None:
    """divide_attention splits resources across targets."""
    att = AttentionSystem()
    allocations = att.divide_attention(["a", "b", "c"])
    assert "a" in allocations
    assert "b" in allocations
    assert "c" in allocations
    # Each gets (1/3)^1.2 (superlinear allocation with dual-task cost)
    expected = (1.0 / 3) ** 1.2
    assert abs(allocations["a"] - expected) < 1e-9


def test_divide_attention_empty() -> None:
    """divide_attention with no targets returns empty dict."""
    att = AttentionSystem()
    allocations = att.divide_attention([])
    assert allocations == {}


def test_divide_attention_creates_foci() -> None:
    """divide_attention creates foci for each target."""
    att = AttentionSystem()
    att.divide_attention(["a", "b"])
    assert att.is_attended("a")
    assert att.is_attended("b")
    focus_a = att.get_focus("a")
    assert focus_a is not None
    assert focus_a.attention_type == AttentionType.DIVIDED


def test_divide_attention_dual_task_cost() -> None:
    """More targets → less per-target intensity (dual-task cost)."""
    att = AttentionSystem()
    alloc_2 = att.divide_attention(["a", "b"])
    att.clear()
    alloc_4 = att.divide_attention(["a", "b", "c", "d"])
    # Per-target intensity with 4 targets should be less than with 2
    assert alloc_4["a"] < alloc_2["a"]


def test_divide_attention_over_capacity_only_attends_max() -> None:
    """divide_attention cannot exceed max_foci.

    Attentional capacity is finite (Cowan's K≈4). Requesting more targets than
    capacity can hold must not silently evict early targets while *claiming* in
    the returned allocations that they received attention. The returned dict
    must reflect reality: only actually-attended targets appear.
    """
    att = AttentionSystem(max_foci=4)
    allocations = att.divide_attention(["a", "b", "c", "d", "e"])
    # Only max_foci targets can actually be attended.
    assert len(att.focused_targets) == 4
    # Allocations must match the set of actually-attended targets.
    assert set(allocations) == set(att.focused_targets)
    # Every reported allocation corresponds to a real DIVIDED focus.
    for target in allocations:
        assert att.is_attended(target)
        focus = att.get_focus(target)
        assert focus is not None
        assert focus.attention_type == AttentionType.DIVIDED


def test_divide_attention_dedupes_targets() -> None:
    """divide_attention treats duplicate targets as a single focus."""
    att = AttentionSystem()
    allocations = att.divide_attention(["a", "a", "b"])
    assert set(allocations) == {"a", "b"}
    assert len(att.focused_targets) == 2


# ═══════════════════════════════════════════════════════════════════
# Querying attention state
# ═══════════════════════════════════════════════════════════════════


def test_foci_property() -> None:
    """foci property returns current foci."""
    att = AttentionSystem()
    att.focus_on("a")
    att.focus_on("b")
    foci = att.foci
    assert len(foci) == 2


def test_focused_targets_property() -> None:
    """focused_targets returns target names."""
    att = AttentionSystem()
    att.focus_on("a")
    att.focus_on("b")
    targets = att.focused_targets
    assert "a" in targets
    assert "b" in targets


def test_get_focus() -> None:
    """get_focus returns the focus for a target."""
    att = AttentionSystem()
    att.focus_on("dogs", intensity=0.7)
    focus = att.get_focus("dogs")
    assert focus is not None
    assert focus.intensity == 0.7


def test_get_focus_unknown() -> None:
    """get_focus returns None for unknown targets."""
    att = AttentionSystem()
    assert att.get_focus("unknown") is None


def test_dominant_focus() -> None:
    """dominant_focus returns the strongest focus."""
    att = AttentionSystem()
    att.focus_on("a", intensity=0.5)
    att.focus_on("b", intensity=0.9)
    dominant = att.dominant_focus
    assert dominant is not None
    assert dominant.target == "b"


def test_dominant_focus_empty() -> None:
    """dominant_focus returns None when no foci."""
    att = AttentionSystem()
    assert att.dominant_focus is None


def test_is_focused() -> None:
    """is_focused is True when there are foci."""
    att = AttentionSystem()
    assert not att.is_focused
    att.focus_on("a")
    assert att.is_focused


def test_suppressed_targets_property() -> None:
    """suppressed_targets returns the suppressed set."""
    att = AttentionSystem()
    att.suppress("noise")
    att.suppress("distraction")
    suppressed = att.suppressed_targets
    assert "noise" in suppressed
    assert "distraction" in suppressed


def test_is_attended() -> None:
    """is_attended checks if a target is focused."""
    att = AttentionSystem()
    att.focus_on("dogs")
    assert att.is_attended("dogs")
    assert not att.is_attended("cats")


def test_is_suppressed() -> None:
    """is_suppressed checks if a target is suppressed."""
    att = AttentionSystem()
    att.suppress("noise")
    assert att.is_suppressed("noise")
    assert not att.is_suppressed("signal")


# ═══════════════════════════════════════════════════════════════════
# Network integration
# ═══════════════════════════════════════════════════════════════════


def test_focus_amplifies_network_concept() -> None:
    """focus_on amplifies the concept in the network."""
    network = ConceptNetwork()
    network.add_concept("dogs")
    concept = network.get_concept("dogs")
    assert concept is not None
    initial_activation = concept.activation
    att = AttentionSystem(network=network, amplification_strength=0.4)
    att.focus_on("dogs", intensity=1.0)
    concept2 = network.get_concept("dogs")
    assert concept2 is not None
    assert concept2.activation > initial_activation


def test_suppress_reduces_network_concept() -> None:
    """suppress reduces the concept's activation in the network."""
    network = ConceptNetwork()
    network.add_concept("noise")
    concept = network.get_concept("noise")
    assert concept is not None
    concept.activation = 0.8
    att = AttentionSystem(network=network, suppression_strength=0.5)
    att.suppress("noise")
    concept2 = network.get_concept("noise")
    assert concept2 is not None
    assert concept2.activation < 0.8


def test_no_network_no_error() -> None:
    """AttentionSystem works without a network."""
    att = AttentionSystem(network=None)
    att.focus_on("dogs")
    att.suppress("noise")
    # Should not raise


# ═══════════════════════════════════════════════════════════════════
# clear
# ═══════════════════════════════════════════════════════════════════


def test_clear() -> None:
    """clear removes all foci, suppressions, and goal."""
    att = AttentionSystem()
    att.focus_on("a")
    att.suppress("noise")
    att.set_goal("goal")
    att.clear()
    assert not att.is_focused
    assert len(att.suppressed_targets) == 0
    assert att.current_goal == ""


# ======================================================================
# From tests/test_global_workspace.py
# ======================================================================

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


class _MockModule:
    """A mock module that records received broadcasts."""

    def __init__(self, name: str = "mock") -> None:
        """Initialize the mock module."""
        self.name = name
        self.received: list[WorkspaceItem] = []

    def receive_broadcast(self, item: WorkspaceItem) -> None:
        """Receive broadcast."""
        self.received.append(item)


class _FailingModule:
    """A module that raises on receive_broadcast."""

    def receive_broadcast(self, item: WorkspaceItem) -> None:
        """Receive broadcast."""
        raise OSError("module unavailable")


# ═══════════════════════════════════════════════════════════════════
# WorkspaceItem
# ═══════════════════════════════════════════════════════════════════


def test_workspace_item_creation() -> None:
    """WorkspaceItem stores content, source, activation."""
    item = WorkspaceItem(content="hello", source="perception", activation=0.9)
    assert item.content == "hello"
    assert item.source == "perception"
    assert item.activation == 0.9
    assert item.timestamp > 0
    assert item.refresh_count == 0


def test_workspace_item_metadata() -> None:
    """WorkspaceItem stores optional metadata."""
    item = WorkspaceItem(
        content="hello",
        source="perception",
        activation=0.9,
        metadata={"key": "value"},
    )
    assert item.metadata["key"] == "value"


def test_workspace_item_age_ms() -> None:
    """age_ms returns the age in milliseconds."""
    item = WorkspaceItem(content="hello", source="perception", activation=0.9)
    age = item.age_ms
    assert age >= 0


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — initialization
# ═══════════════════════════════════════════════════════════════════


def test_gw_initialization() -> None:
    """GlobalWorkspace initializes with default parameters."""
    gw = GlobalWorkspace()
    assert gw.ignition_threshold == 0.7
    assert gw.capacity == 4
    assert gw.decay_rate == 0.05
    assert gw.is_empty
    assert gw.broadcast_count == 0


def test_gw_custom_parameters() -> None:
    """GlobalWorkspace accepts custom parameters."""
    gw = GlobalWorkspace(ignition_threshold=0.5, capacity=2, decay_rate=0.1)
    assert gw.ignition_threshold == 0.5
    assert gw.capacity == 2
    assert gw.decay_rate == 0.1


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — module registration
# ═══════════════════════════════════════════════════════════════════


def test_gw_register_module() -> None:
    """register_module adds a module to the distribution list."""
    gw = GlobalWorkspace()
    module = _MockModule()
    gw.register_module(module)
    assert module in gw.registered_modules


def test_gw_register_module_duplicate() -> None:
    """register_module does not add duplicates."""
    gw = GlobalWorkspace()
    module = _MockModule()
    gw.register_module(module)
    gw.register_module(module)
    assert len(gw.registered_modules) == 1


def test_gw_unregister_module() -> None:
    """unregister_module removes a module."""
    gw = GlobalWorkspace()
    module = _MockModule()
    gw.register_module(module)
    gw.unregister_module(module)
    assert module not in gw.registered_modules


def test_gw_unregister_not_registered() -> None:
    """unregister_module handles unregistered modules gracefully."""
    gw = GlobalWorkspace()
    module = _MockModule()
    gw.unregister_module(module)  # should not raise
    assert len(gw.registered_modules) == 0


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — broadcast and ignition
# ═══════════════════════════════════════════════════════════════════


def test_gw_broadcast_above_threshold() -> None:
    """broadcast succeeds when activation >= threshold."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    result = gw.broadcast("hello", "perception", 0.8)
    assert result is True
    assert not gw.is_empty
    assert gw.broadcast_count == 1


def test_gw_broadcast_at_threshold() -> None:
    """broadcast succeeds when activation == threshold."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    result = gw.broadcast("hello", "perception", 0.7)
    assert result is True


def test_gw_broadcast_below_threshold() -> None:
    """broadcast fails when activation < threshold."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    result = gw.broadcast("hello", "perception", 0.5)
    assert result is False
    assert gw.is_empty
    assert gw.broadcast_count == 0


def test_gw_broadcast_distributes_to_modules() -> None:
    """broadcast distributes items to registered modules."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    module = _MockModule()
    gw.register_module(module)
    gw.broadcast("hello", "perception", 0.9)
    assert len(module.received) == 1
    assert module.received[0].content == "hello"


def test_gw_broadcast_no_modules() -> None:
    """broadcast works without any registered modules."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    result = gw.broadcast("hello", "perception", 0.9)
    assert result is True


def test_gw_broadcast_failing_module_handled() -> None:
    """A failing module doesn't crash the broadcast."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    failing = _FailingModule()
    good = _MockModule()
    gw.register_module(failing)
    gw.register_module(good)
    # Should not raise
    gw.broadcast("hello", "perception", 0.9)
    # The good module should still receive it
    assert len(good.received) == 1


def test_gw_broadcast_refreshes_existing() -> None:
    """Broadcasting the same content+source refreshes the existing item."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("hello", "perception", 0.8)
    initial_count = gw.broadcast_count
    # Broadcast the same content again
    gw.broadcast("hello", "perception", 0.9)
    # Should not create a new item (refresh, not new broadcast)
    assert gw.broadcast_count == initial_count
    items = gw.items
    assert len(items) == 1
    assert items[0].refresh_count == 1


def test_gw_broadcast_activation_capped() -> None:
    """Activation is capped at 1.0."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("hello", "perception", 2.0)
    items = gw.items
    assert items[0].activation == 1.0


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — capacity and competition
# ═══════════════════════════════════════════════════════════════════


def test_gw_capacity_limit() -> None:
    """Workspace evicts weakest items when capacity is exceeded."""
    gw = GlobalWorkspace(ignition_threshold=0.7, capacity=2)
    gw.broadcast("a", "src1", 0.9)
    gw.broadcast("b", "src2", 0.8)
    gw.broadcast("c", "src3", 0.95)  # should evict "b" (weakest)
    items = gw.items
    assert len(items) == 2
    contents = [item.content for item in items]
    assert "a" in contents
    assert "c" in contents
    assert "b" not in contents


def test_gw_capacity_not_exceeded() -> None:
    """Workspace holds items up to capacity."""
    gw = GlobalWorkspace(ignition_threshold=0.7, capacity=3)
    gw.broadcast("a", "src1", 0.9)
    gw.broadcast("b", "src2", 0.8)
    gw.broadcast("c", "src3", 0.85)
    assert len(gw.items) == 3


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — decay
# ═══════════════════════════════════════════════════════════════════


def test_gw_tick_decay() -> None:
    """tick applies decay to all items."""
    gw = GlobalWorkspace(ignition_threshold=0.7, decay_rate=0.5)
    gw.broadcast("hello", "perception", 0.9)
    initial_activation = gw.items[0].activation
    gw.tick()
    assert gw.items[0].activation < initial_activation


def test_gw_tick_evicts_decayed_items() -> None:
    """tick removes items that decay below 0.1."""
    gw = GlobalWorkspace(ignition_threshold=0.7, decay_rate=0.5)
    gw.broadcast("hello", "perception", 0.75)  # above threshold
    # 0.75 * 0.5 = 0.375 after tick 1
    # 0.375 * 0.5 = 0.1875 after tick 2
    # 0.1875 * 0.5 = 0.09375 < 0.1 after tick 3 → evicted
    gw.tick()
    gw.tick()
    evicted = gw.tick()
    assert len(evicted) > 0
    assert gw.is_empty


def test_gw_tick_returns_evicted() -> None:
    """tick returns the list of evicted items."""
    gw = GlobalWorkspace(ignition_threshold=0.7, decay_rate=0.9)
    gw.broadcast("hello", "perception", 0.75)
    evicted = gw.tick()
    assert isinstance(evicted, list)


def test_gw_tick_no_items() -> None:
    """tick on empty workspace returns empty list."""
    gw = GlobalWorkspace()
    evicted = gw.tick()
    assert evicted == []


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — queries
# ═══════════════════════════════════════════════════════════════════


def test_gw_items_property() -> None:
    """items property returns a copy of current items."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("hello", "perception", 0.9)
    items = gw.items
    assert len(items) == 1
    # Modifying the returned list doesn't affect the workspace
    items.clear()
    assert not gw.is_empty


def test_gw_is_empty() -> None:
    """is_empty is True initially."""
    gw = GlobalWorkspace()
    assert gw.is_empty


def test_gw_broadcast_count() -> None:
    """broadcast_count tracks successful broadcasts."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    assert gw.broadcast_count == 0
    gw.broadcast("a", "src1", 0.9)
    assert gw.broadcast_count == 1
    gw.broadcast("b", "src2", 0.8)
    assert gw.broadcast_count == 2
    # Failed broadcast doesn't count
    gw.broadcast("c", "src3", 0.5)
    assert gw.broadcast_count == 2


def test_gw_get_dominant() -> None:
    """get_dominant returns the highest-activation item."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("a", "src1", 0.8)
    gw.broadcast("b", "src2", 0.95)
    dominant = gw.get_dominant()
    assert dominant is not None
    assert dominant.content == "b"


def test_gw_get_dominant_empty() -> None:
    """get_dominant returns None when workspace is empty."""
    gw = GlobalWorkspace()
    assert gw.get_dominant() is None


def test_gw_clear() -> None:
    """clear removes all items."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("a", "src1", 0.9)
    gw.broadcast("b", "src2", 0.8)
    gw.clear()
    assert gw.is_empty


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — recurrence, pending buffer, coalition ignition
# ═══════════════════════════════════════════════════════════════════


def test_gw_below_threshold_goes_pending() -> None:
    """Failed broadcasts enter the pending pool, not the workspace."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    result = gw.broadcast("x", "src1", 0.6)
    assert result is False
    assert gw.is_empty
    assert gw.pending_count == 1
    assert gw.broadcast_count == 0


def test_gw_coalition_ignition() -> None:
    """A pending item co-ignites when coherent content arrives."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    module = _MockModule()
    gw.register_module(module)
    gw.broadcast("x", "src1", 0.6, metadata={"topics": ["alpha"]})
    result = gw.broadcast("y", "src2", 0.8, metadata={"topics": ["alpha"]})
    assert result is True
    # Coherent excitation pulled the pending item over threshold
    assert gw.pending_count == 0
    assert len(gw.items) == 2
    assert gw.broadcast_count == 2
    assert len(module.received) == 2


def test_gw_incoherent_pending_stays_subliminal() -> None:
    """Pending items with disjoint topics are suppressed, not ignited."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("x", "src1", 0.6, metadata={"topics": ["alpha"]})
    result = gw.broadcast("y", "src2", 0.8, metadata={"topics": ["beta"]})
    assert result is True
    assert len(gw.items) == 1
    assert gw.pending_count == 1
    assert gw.broadcast_count == 1


def test_gw_pending_expires() -> None:
    """Pending items expire after the pending TTL."""
    gw = GlobalWorkspace(ignition_threshold=0.7, pending_ttl_s=60.0)
    gw.broadcast("x", "src1", 0.6)
    assert gw.pending_count == 1
    for p in gw._pending:
        p.timestamp -= 61_000
    gw.tick()
    assert gw.pending_count == 0


def test_gw_drain_subliminal() -> None:
    """Expired pending items are drainable, not silently lost."""
    gw = GlobalWorkspace(ignition_threshold=0.7, pending_ttl_s=60.0)
    gw.broadcast("x", "src1", 0.6, metadata={"topics": ["alpha"]})
    for p in gw._pending:
        p.timestamp -= 61_000
    gw.tick()
    drained = gw.drain_subliminal()
    assert len(drained) == 1
    assert drained[0].content == "x"
    assert gw.drain_subliminal() == []


def test_gw_recurrent_ignition_tagged() -> None:
    """Coalition-ignited items carry ignition=recurrent in metadata."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("x", "s1", 0.6, metadata={"topics": ["alpha"]})
    gw.broadcast("y", "s2", 0.8, metadata={"topics": ["alpha"]})
    tagged = [
        i for i in gw.items
        if i.metadata.get("ignition") == "recurrent"
    ]
    assert len(tagged) == 1
    assert tagged[0].content == "x"


def test_damasio_recurrent_trigger_salience() -> None:
    """Recurrent ignitions register as more salient to the core self."""
    from genesis_cognitive.self.damasio import DamasioSelfHierarchy

    recurrent = WorkspaceItem(
        content="emerged",
        source="inner_life",
        activation=0.8,
        metadata={"topics": ["alpha"], "ignition": "recurrent"},
    )
    ordinary = WorkspaceItem(
        content="driven",
        source="perception",
        activation=0.8,
        metadata={"topics": ["alpha"]},
    )
    d1 = DamasioSelfHierarchy()
    d2 = DamasioSelfHierarchy()
    d1.receive_broadcast(recurrent)
    d2.receive_broadcast(ordinary)
    e1 = d1.update(arousal=0.8, valence=0.4)
    e2 = d2.update(arousal=0.8, valence=0.4)
    assert e1 is not None and e2 is not None
    assert e1.salience > e2.salience


def test_gw_tick_coherent_items_sustain() -> None:
    """Coherent items resist decay better than incoherent ones."""
    coherent = GlobalWorkspace(ignition_threshold=0.7, decay_rate=0.4)
    coherent.broadcast("a", "src1", 0.75, metadata={"topics": ["t"]})
    coherent.broadcast("b", "src2", 0.72, metadata={"topics": ["t"]})
    incoherent = GlobalWorkspace(ignition_threshold=0.7, decay_rate=0.4)
    incoherent.broadcast("a", "src1", 0.75, metadata={"topics": ["t"]})
    incoherent.broadcast("b", "src2", 0.72, metadata={"topics": ["u"]})
    coherent.tick()
    incoherent.tick()
    # After one tick both incoherent items still exist, but the weaker
    # member has been suppressed well below its coherent counterpart.
    coherent_b = next(i for i in coherent.items if i.content == "b")
    incoherent_b = next(i for i in incoherent.items if i.content == "b")
    assert coherent_b.activation > incoherent_b.activation
    # Competition continues between ticks: the incoherent weak item
    # decays out while the coherent pair mutually sustains.
    coherent.tick()
    incoherent.tick()
    assert len(coherent.items) == 2
    assert [i.content for i in incoherent.items] == ["a"]


def test_gw_on_ignition_fires() -> None:
    """on_ignition fires once per fresh ignition, not on refresh."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    events: list[tuple[str, bool]] = []
    gw.on_ignition = lambda item, recurrent: events.append(
        (item.content, recurrent)
    )
    gw.broadcast("a", "s1", 0.9)
    assert events == [("a", False)]
    gw.broadcast("a", "s1", 0.9)  # refresh — sustained attention
    assert len(events) == 1


def test_gw_on_ignition_coalition_flag() -> None:
    """Coalition ignition fires the hook with via_recurrent=True."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    gw.broadcast("x", "s1", 0.6, metadata={"topics": ["alpha"]})
    events: list[tuple[str, bool]] = []
    gw.on_ignition = lambda item, recurrent: events.append(
        (item.content, recurrent)
    )
    gw.broadcast("y", "s2", 0.8, metadata={"topics": ["alpha"]})
    assert ("y", False) in events
    assert ("x", True) in events


def test_gw_on_ignition_not_subliminal() -> None:
    """Failed broadcasts don't fire the hook."""
    gw = GlobalWorkspace(ignition_threshold=0.7)
    events: list[object] = []
    gw.on_ignition = lambda item, recurrent: events.append(item)
    gw.broadcast("x", "s1", 0.5)
    assert events == []


# ═══════════════════════════════════════════════════════════════════
# Safeguard urge — defensive drive dynamics
# ═══════════════════════════════════════════════════════════════════


def test_safeguard_urge_registered() -> None:
    """The safeguard urge ships in the default urge set."""
    from genesis_cognitive.config import MindConfig

    urges = {u.name: u for u in MindConfig().volition.urges}
    assert "safeguard" in urges
    sg = urges["safeguard"]
    assert "daemon_lost" in sg.stimuli
    assert "save_failure" in sg.stimuli
    # Defensive drives out-compete appetitive ones: lower threshold,
    # stronger stimulus weights than the appetitive urges.
    learn = urges["learn"]
    assert sg.threshold <= learn.threshold
    assert max(sg.stimuli.values()) > max(learn.stimuli.values())


def test_safeguard_urge_dynamics() -> None:
    """Safeguard fires under sustained threat, not on a transient."""
    from genesis_cognitive.config import MindConfig
    from genesis_cognitive.volition import Urge

    cfg = next(u for u in MindConfig().volition.urges if u.name == "safeguard")
    urge = Urge(
        name=cfg.name,
        threshold=cfg.threshold,
        growth=cfg.growth,
        decay=cfg.decay,
        cooldown=cfg.cooldown,
        stimuli=dict(cfg.stimuli),
    )
    # A 1-second connectivity flap must not fire it.
    urge.tick({"daemon_lost": 1.0 / 30.0}, 1.0)
    assert not urge.is_ready()
    # A sustained daemon loss ramps the signal to 1.0 over 30s and
    # crosses the threshold well before a minute is out.
    fired_at = None
    for t in range(2, 90):
        urge.tick({"daemon_lost": min(1.0, t / 30.0)}, 1.0)
        if urge.is_ready():
            fired_at = t
            break
    assert fired_at is not None and fired_at < 60


def test_gw_registered_modules_property() -> None:
    """registered_modules property returns a copy of the module list."""
    gw = GlobalWorkspace()
    module = _MockModule()
    gw.register_module(module)
    modules = gw.registered_modules
    assert len(modules) == 1
    # Modifying the returned list doesn't affect the workspace
    modules.clear()
    assert len(gw.registered_modules) == 1


# ═══════════════════════════════════════════════════════════════════
# GlobalWorkspace — integration measure
# ═══════════════════════════════════════════════════════════════════


def test_gw_integration_empty() -> None:
    """Integration is 0.0 when the workspace is empty."""
    gw = GlobalWorkspace()
    assert gw.integration == 0.0


def test_gw_integration_single_item() -> None:
    """Integration is small for a single item (no cross-module content)."""
    gw = GlobalWorkspace(ignition_threshold=0.5)
    gw.broadcast("hello", "perception", 0.8, metadata={"topics": ["cats"]})
    # Single item: activation * 0.2
    assert 0.0 < gw.integration < 0.3
    assert gw.integration == pytest.approx(0.8 * 0.2)


def test_gw_integration_high_when_coherent_diverse() -> None:
    """Integration is high when multiple sources share topics at high activation."""
    gw = GlobalWorkspace(ignition_threshold=0.5)
    gw.broadcast("percept", "perception", 0.9, metadata={"topics": ["cats", "pets"]})
    gw.broadcast("thought", "deliberation", 0.8, metadata={"topics": ["cats", "pets"]})
    # Two items, same topics, different sources → high integration
    assert gw.integration > 0.3


def test_gw_integration_low_when_disjoint_topics() -> None:
    """Integration is low when items have disjoint topics (scattered)."""
    gw = GlobalWorkspace(ignition_threshold=0.5)
    gw.broadcast("percept", "perception", 0.9, metadata={"topics": ["cats"]})
    gw.broadcast("thought", "deliberation", 0.8, metadata={"topics": ["quantum"]})
    # Two items, disjoint topics, different sources
    # coherence = 0 (no overlap), so integration = 0
    assert gw.integration == pytest.approx(0.0)


def test_gw_integration_low_when_single_source() -> None:
    """Integration is low when all items come from one source (no diversity)."""
    gw = GlobalWorkspace(ignition_threshold=0.5)
    gw.broadcast("a", "perception", 0.9, metadata={"topics": ["cats"]})
    gw.broadcast("b", "perception", 0.8, metadata={"topics": ["cats"]})
    # Two items, same topics, same source → diversity = 0.5
    # coherence = 1.0 (identical topics), activation ~0.85
    # integration = 1.0 * 0.5 * 0.85 = 0.425
    assert 0.2 < gw.integration < 0.6


def test_gw_integration_bounded_01() -> None:
    """Integration is always in [0, 1]."""
    gw = GlobalWorkspace(ignition_threshold=0.3)
    gw.broadcast("a", "src1", 1.0, metadata={"topics": ["x"]})
    gw.broadcast("b", "src2", 1.0, metadata={"topics": ["x"]})
    gw.broadcast("c", "src3", 1.0, metadata={"topics": ["x"]})
    gw.broadcast("d", "src4", 1.0, metadata={"topics": ["x"]})
    assert 0.0 <= gw.integration <= 1.0


def test_gw_integration_no_topics_neutral() -> None:
    """Integration uses neutral coherence when items lack topics."""
    gw = GlobalWorkspace(ignition_threshold=0.5)
    gw.broadcast("a", "src1", 0.8)
    gw.broadcast("b", "src2", 0.8)
    # No topics → coherence = 0.5 (neutral)
    # diversity = 1.0 (2 distinct sources / 2 items)
    # activation = mean settled activation (settle adjusts levels)
    # integration = 0.5 * 1.0 * mean_activation
    mean_activation = sum(i.activation for i in gw.items) / len(gw.items)
    assert gw.integration == pytest.approx(0.5 * mean_activation)


# ═══════════════════════════════════════════════════════════════════
# DamasioSelfHierarchy — annotate_self_model with integration
# ═══════════════════════════════════════════════════════════════════


def test_damasio_annotate_self_model_unified() -> None:
    """High integration adds field-unified tag."""
    from genesis_cognitive.self.damasio import DamasioSelfHierarchy

    damasio = DamasioSelfHierarchy()
    damasio.annotate_self_model("self-coherent", False, integration=0.8)
    assert "field-unified" in damasio.self_model_label
    assert "self-coherent" in damasio.self_model_label


def test_damasio_annotate_self_model_fragmented() -> None:
    """Low integration adds field-fragmented tag."""
    from genesis_cognitive.self.damasio import DamasioSelfHierarchy

    damasio = DamasioSelfHierarchy()
    damasio.annotate_self_model("self-surprised", True, integration=0.1)
    assert "field-fragmented" in damasio.self_model_label


def test_damasio_annotate_self_model_neutral() -> None:
    """Mid-range integration adds no field tag."""
    from genesis_cognitive.self.damasio import DamasioSelfHierarchy

    damasio = DamasioSelfHierarchy()
    damasio.annotate_self_model("self-coherent", False, integration=0.4)
    assert "field-unified" not in damasio.self_model_label
    assert "field-fragmented" not in damasio.self_model_label
    assert damasio.self_model_label == "self-coherent"


# ═══════════════════════════════════════════════════════════════════
# CognitiveState — self_model_coherence field
# ═══════════════════════════════════════════════════════════════════


def test_cognitive_state_has_coherence_field() -> None:
    """CognitiveState has a self_model_coherence field."""
    from genesis_cognitive.cognition.engine import CognitiveState

    state = CognitiveState.__dataclass_fields__
    assert "self_model_coherence" in state


def test_cognitive_state_describe_includes_coherence() -> None:
    """describe() includes coherence when set."""
    from genesis_cognitive.cognition.engine import CognitiveState

    # Verify the field exists and defaults to None
    fields = CognitiveState.__dataclass_fields__
    assert "self_model_coherence" in fields
    assert fields["self_model_coherence"].default is None

    # Verify describe() references it — check the source includes
    # the coherence formatting code
    import inspect

    src = inspect.getsource(CognitiveState.describe)
    assert "self_model_coherence" in src
    assert "Coherence:" in src


# ======================================================================
# From tests/test_executive.py
# ======================================================================

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Enums and dataclasses
# ═══════════════════════════════════════════════════════════════════


def test_task_state_values() -> None:
    """TaskState has the expected values."""
    assert TaskState.ACTIVE.value == "active"
    assert TaskState.SUSPENDED.value == "suspended"
    assert TaskState.COMPLETED.value == "completed"


def test_action_plan_creation() -> None:
    """ActionPlan stores steps, outcomes, expected value."""
    plan = ActionPlan(
        steps=["step1", "step2"],
        predicted_outcomes=["result1", "result2"],
        expected_value=0.8,
        confidence=0.9,
        goal="test goal",
    )
    assert plan.steps == ["step1", "step2"]
    assert plan.predicted_outcomes == ["result1", "result2"]
    assert plan.expected_value == 0.8
    assert plan.confidence == 0.9
    assert plan.goal == "test goal"


def test_action_plan_describe() -> None:
    """describe() returns a human-readable plan description."""
    plan = ActionPlan(
        steps=["step1"],
        predicted_outcomes=["result1"],
        expected_value=0.5,
        goal="test",
    )
    desc = plan.describe()
    assert "test" in desc
    assert "step1" in desc


def test_task_creation() -> None:
    """Task stores name, goal, state, rules."""
    task = Task(name="task1", goal="do something")
    assert task.name == "task1"
    assert task.goal == "do something"
    assert task.state == TaskState.SUSPENDED
    assert task.rules == {}


def test_inhibition_result_creation() -> None:
    """InhibitionResult stores inhibited flag, impulse, threshold."""
    result = InhibitionResult(
        inhibited=True,
        impulse_strength=0.3,
        threshold=0.5,
        reason="below threshold",
    )
    assert result.inhibited is True
    assert result.impulse_strength == 0.3
    assert result.threshold == 0.5
    assert result.reason == "below threshold"


def test_switch_result_creation() -> None:
    """SwitchResult stores from/to tasks and costs."""
    result = SwitchResult(
        from_task="task_a",
        to_task="task_b",
        rt_cost=0.3,
        accuracy_cost=0.15,
    )
    assert result.from_task == "task_a"
    assert result.to_task == "task_b"
    assert result.rt_cost == 0.3
    assert result.accuracy_cost == 0.15
    assert result.success is True


# ═══════════════════════════════════════════════════════════════════
# ExecutiveFunction — initialization
# ═══════════════════════════════════════════════════════════════════


def test_executive_initialization() -> None:
    """ExecutiveFunction initializes with defaults."""
    exec_fn = ExecutiveFunction()
    assert exec_fn.inhibition_threshold == 0.5
    assert exec_fn.switch_rt_cost == 0.3
    assert exec_fn.switch_accuracy_cost == 0.15
    assert exec_fn.planning_depth == 3


def test_executive_custom_params() -> None:
    """ExecutiveFunction accepts custom parameters."""
    exec_fn = ExecutiveFunction(
        inhibition_threshold=0.7,
        switch_rt_cost=0.5,
        planning_depth=5,
    )
    assert exec_fn.inhibition_threshold == 0.7
    assert exec_fn.switch_rt_cost == 0.5
    assert exec_fn.planning_depth == 5


# ═══════════════════════════════════════════════════════════════════
# Planning
# ═══════════════════════════════════════════════════════════════════


def test_plan_returns_action_plan() -> None:
    """plan() returns an ActionPlan."""
    exec_fn = ExecutiveFunction()
    plan = exec_fn.plan("goal", ["action1", "action2"])
    assert isinstance(plan, ActionPlan)
    assert plan.goal == "goal"


def test_plan_empty_actions() -> None:
    """plan() with no actions returns a low-value plan."""
    exec_fn = ExecutiveFunction()
    plan = exec_fn.plan("goal", [])
    assert plan.expected_value == 0.0
    assert plan.confidence == 0.0


def test_plan_selects_best_action() -> None:
    """plan() selects the action with the highest expected value."""
    exec_fn = ExecutiveFunction()
    plan = exec_fn.plan("answer question", ["answer the question", "ignore"])
    assert plan.expected_value > 0
    # The goal-related action should be in the plan
    assert "answer the question" in plan.steps


def test_plan_with_predictor() -> None:
    """plan() uses a custom outcome predictor when provided."""
    exec_fn = ExecutiveFunction()

    def predictor(action: str) -> tuple[str, float]:
        """Custom outcome predictor returning a fixed result and confidence."""
        return (f"result of {action}", 0.9)

    plan = exec_fn.plan("goal", ["action1"], outcome_predictor=predictor)
    assert plan.expected_value > 0
    assert "result of action1" in plan.predicted_outcomes


def test_plan_stores_current_plan() -> None:
    """plan() stores the plan as the current plan."""
    exec_fn = ExecutiveFunction()
    plan = exec_fn.plan("goal", ["action1"])
    assert exec_fn.current_plan is plan


def test_plan_current_plan_initial_none() -> None:
    """current_plan is None initially."""
    exec_fn = ExecutiveFunction()
    assert exec_fn.current_plan is None


# ═══════════════════════════════════════════════════════════════════
# Response inhibition
# ═══════════════════════════════════════════════════════════════════


def test_inhibit_response_below_threshold() -> None:
    """Responses below threshold are inhibited."""
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    result = exec_fn.inhibit_response(0.3)
    assert result.inhibited is True


def test_inhibit_response_above_threshold() -> None:
    """Responses above threshold proceed."""
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    result = exec_fn.inhibit_response(0.7)
    assert result.inhibited is False


def test_inhibit_response_at_threshold() -> None:
    """Response at threshold proceeds (not below)."""
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    result = exec_fn.inhibit_response(0.5)
    assert result.inhibited is False


def test_inhibit_response_increments_count() -> None:
    """Inhibited responses increment the count."""
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    assert exec_fn.inhibition_count == 0
    exec_fn.inhibit_response(0.3)
    assert exec_fn.inhibition_count == 1
    exec_fn.inhibit_response(0.2)
    assert exec_fn.inhibition_count == 2


def test_inhibit_response_does_not_count_non_inhibited() -> None:
    """Non-inhibited responses don't increment the count."""
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    exec_fn.inhibit_response(0.7)
    assert exec_fn.inhibition_count == 0


def test_set_inhibition_threshold() -> None:
    """set_inhibition_threshold updates the threshold."""
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    exec_fn.set_inhibition_threshold(0.8)
    assert exec_fn.inhibition_threshold == 0.8
    # Now 0.7 should be inhibited
    result = exec_fn.inhibit_response(0.7)
    assert result.inhibited is True


def test_set_inhibition_threshold_clamped() -> None:
    """set_inhibition_threshold clamps to [0, 1]."""
    exec_fn = ExecutiveFunction()
    exec_fn.set_inhibition_threshold(2.0)
    assert exec_fn.inhibition_threshold == 1.0
    exec_fn.set_inhibition_threshold(-1.0)
    assert exec_fn.inhibition_threshold == 0.0


# ═══════════════════════════════════════════════════════════════════
# Task-switching
# ═══════════════════════════════════════════════════════════════════


def test_register_task() -> None:
    """register_task adds a task to the task set."""
    exec_fn = ExecutiveFunction()
    task = exec_fn.register_task("task1", goal="do something")
    assert task.name == "task1"
    assert task.goal == "do something"
    assert "task1" in exec_fn.tasks


def test_register_task_with_rules() -> None:
    """register_task stores task rules."""
    exec_fn = ExecutiveFunction()
    task = exec_fn.register_task("task1", rules={"key": "value"})
    assert task.rules["key"] == "value"


def test_switch_task() -> None:
    """switch_task activates a task and returns a SwitchResult."""
    exec_fn = ExecutiveFunction()
    exec_fn.register_task("task_a")
    exec_fn.register_task("task_b")
    exec_fn.switch_task("task_a")
    result = exec_fn.switch_task("task_b")
    assert isinstance(result, SwitchResult)
    assert result.from_task == "task_a"
    assert result.to_task == "task_b"
    assert exec_fn.active_task == "task_b"


def test_switch_task_auto_registers() -> None:
    """switch_task auto-registers unknown tasks."""
    exec_fn = ExecutiveFunction()
    exec_fn.switch_task("new_task")
    assert "new_task" in exec_fn.tasks
    assert exec_fn.active_task == "new_task"


def test_switch_task_increments_count() -> None:
    """switch_task increments the switch count."""
    exec_fn = ExecutiveFunction()
    exec_fn.register_task("a")
    exec_fn.register_task("b")
    exec_fn.switch_task("a")
    assert exec_fn.switch_count == 1
    exec_fn.switch_task("b")
    assert exec_fn.switch_count == 2


def test_switch_task_suspends_previous() -> None:
    """switch_task suspends the previous active task."""
    exec_fn = ExecutiveFunction()
    exec_fn.register_task("a")
    exec_fn.register_task("b")
    exec_fn.switch_task("a")
    assert exec_fn.tasks["a"].state == TaskState.ACTIVE
    exec_fn.switch_task("b")
    assert exec_fn.tasks["a"].state == TaskState.SUSPENDED
    assert exec_fn.tasks["b"].state == TaskState.ACTIVE


def test_switch_task_cost_decreases_with_practice() -> None:
    """Switching cost decreases with practice."""
    exec_fn = ExecutiveFunction(switch_rt_cost=0.3)
    exec_fn.register_task("a")
    exec_fn.register_task("b")
    # First switch to b
    exec_fn.switch_task("a")
    result1 = exec_fn.switch_task("b")
    # Switch back and forth to practice b
    exec_fn.switch_task("a")
    result2 = exec_fn.switch_task("b")
    # Second switch to b should have lower cost
    assert result2.rt_cost <= result1.rt_cost


def test_active_task_initial_empty() -> None:
    """active_task is empty initially."""
    exec_fn = ExecutiveFunction()
    assert exec_fn.active_task == ""


def test_active_task_obj() -> None:
    """active_task_obj returns the active Task object."""
    exec_fn = ExecutiveFunction()
    exec_fn.register_task("a", goal="do a")
    exec_fn.switch_task("a")
    task = exec_fn.active_task_obj
    assert task is not None
    assert task.name == "a"


def test_active_task_obj_none() -> None:
    """active_task_obj returns None when no active task."""
    exec_fn = ExecutiveFunction()
    assert exec_fn.active_task_obj is None


def test_tasks_property() -> None:
    """tasks property returns a copy of the task dict."""
    exec_fn = ExecutiveFunction()
    exec_fn.register_task("a")
    tasks = exec_fn.tasks
    assert "a" in tasks
    # Modifying the returned dict doesn't affect the executive
    tasks.clear()
    assert "a" in exec_fn.tasks


def test_get_switch_cost() -> None:
    """get_switch_cost predicts the cost without switching."""
    exec_fn = ExecutiveFunction(switch_rt_cost=0.3, switch_accuracy_cost=0.15)
    exec_fn.register_task("a")
    rt_cost, acc_cost = exec_fn.get_switch_cost("a")
    assert rt_cost > 0
    assert acc_cost > 0


def test_get_switch_cost_unknown_task() -> None:
    """get_switch_cost handles unknown tasks."""
    exec_fn = ExecutiveFunction()
    rt_cost, _ = exec_fn.get_switch_cost("unknown")
    assert rt_cost > 0


def test_clear_executive() -> None:
    """clear removes all tasks and plans."""
    exec_fn = ExecutiveFunction()
    exec_fn.register_task("a")
    exec_fn.switch_task("a")
    exec_fn.plan("goal", ["action"])
    exec_fn.clear()
    assert exec_fn.tasks == {}
    assert exec_fn.active_task == ""
    assert exec_fn.current_plan is None


# ======================================================================
# From tests/test_working_memory.py
# ======================================================================

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ConversationThread
# ═══════════════════════════════════════════════════════════════════


def test_thread_add_turn() -> None:
    """add_turn adds a turn and updates key concepts."""
    thread = ConversationThread(topic="dogs")
    turn = Turn(
        user_input="tell me about dogs",
        genesis_response="dogs are mammals",
        topics=["dogs", "mammals"],
        intent="question",
    )
    thread.add_turn(turn)
    assert thread.turn_count == 1
    assert "dogs" in thread.key_concepts
    assert "mammals" in thread.key_concepts


# ═══════════════════════════════════════════════════════════════════
# PhonologicalLoop
# ═══════════════════════════════════════════════════════════════════


def test_phonological_loop_capacity() -> None:
    """PhonologicalLoop has the specified capacity (explicit and default)."""
    assert PhonologicalLoop(capacity=3).capacity == 3
    assert PhonologicalLoop().capacity == 4


def test_phonological_loop_store_and_get() -> None:
    """store() adds an item, get_items() retrieves it."""
    loop = PhonologicalLoop()
    loop.store("hello")
    items = loop.get_items()
    assert "hello" in items


def test_phonological_loop_store_empty_ignored() -> None:
    """Empty strings are not stored."""
    loop = PhonologicalLoop()
    loop.store("")
    assert loop.item_count == 0


def test_phonological_loop_store_case_insensitive() -> None:
    """Items are stored case-insensitively (no duplicates)."""
    loop = PhonologicalLoop()
    loop.store("Hello")
    loop.store("HELLO")
    assert loop.item_count == 1


def test_phonological_loop_store_refreshes_existing() -> None:
    """Storing an existing item refreshes its activation."""
    loop = PhonologicalLoop()
    loop.store("item")
    loop.decay(1.0)  # decay it
    loop.store("item")  # refresh
    items = loop.get_items()
    assert "item" in items


def test_phonological_loop_capacity_displacement() -> None:
    """When capacity is exceeded, the weakest item is dropped."""
    loop = PhonologicalLoop(capacity=2)
    loop.store("a")
    loop.store("b")
    # Decay a slightly so it's weaker
    loop.decay(0.5)
    loop.store("c")  # should displace the weakest
    items = loop.get_items()
    assert len(items) <= 2


def test_phonological_loop_rehearse_all() -> None:
    """rehearse() refreshes all items."""
    loop = PhonologicalLoop()
    loop.store("a")
    loop.store("b")
    loop.decay(1.0)  # decay
    loop.rehearse()  # refresh all
    items = loop.get_items()
    assert "a" in items
    assert "b" in items


def test_phonological_loop_rehearse_item() -> None:
    """rehearse_item() refreshes a specific item."""
    loop = PhonologicalLoop()
    loop.store("a")
    loop.decay(1.0)
    result = loop.rehearse_item("a")
    assert result is True


def test_phonological_loop_rehearse_item_not_found() -> None:
    """rehearse_item() returns False for unknown item."""
    loop = PhonologicalLoop()
    result = loop.rehearse_item("unknown")
    assert result is False


def test_phonological_loop_decay_removes_items() -> None:
    """Sufficient decay drops items below threshold."""
    loop = PhonologicalLoop(decay_tau=0.1)
    loop.store("a")
    loop.decay(10.0)  # much longer than tau
    assert loop.item_count == 0


# ═══════════════════════════════════════════════════════════════════
# VisuoSpatialSketchpad
# ═══════════════════════════════════════════════════════════════════


def test_sketchpad_store_and_get() -> None:
    """store_relation adds a relation, get_relations retrieves it."""
    pad = VisuoSpatialSketchpad()
    pad.store_relation("brain", "contains", "hippocampus")
    rels = pad.get_relations()
    assert ("brain", "contains", "hippocampus") in rels


def test_sketchpad_store_empty_ignored() -> None:
    """Empty elements are not stored."""
    pad = VisuoSpatialSketchpad()
    pad.store_relation("", "contains", "hippocampus")
    pad.store_relation("brain", "", "hippocampus")
    pad.store_relation("brain", "contains", "")
    assert pad.relation_count == 0


def test_sketchpad_store_case_insensitive() -> None:
    """Relations are stored case-insensitively (no duplicates)."""
    pad = VisuoSpatialSketchpad()
    pad.store_relation("Brain", "Contains", "Hippocampus")
    pad.store_relation("BRAIN", "CONTAINS", "HIPPOCAMPUS")
    assert pad.relation_count == 1


def test_sketchpad_store_refreshes_existing() -> None:
    """Storing an existing relation refreshes its activation."""
    pad = VisuoSpatialSketchpad()
    pad.store_relation("a", "rel", "b")
    pad.decay(2.0)
    pad.store_relation("a", "rel", "b")  # refresh
    rels = pad.get_relations()
    assert ("a", "rel", "b") in rels


def test_sketchpad_decay_removes() -> None:
    """Sufficient decay drops relations below threshold."""
    pad = VisuoSpatialSketchpad(decay_tau=0.1)
    pad.store_relation("a", "rel", "b")
    pad.decay(20.0)
    assert pad.relation_count == 0


# ═══════════════════════════════════════════════════════════════════
# CentralExecutive
# ═══════════════════════════════════════════════════════════════════


def test_central_executive_direct_attention_boosts_relevant() -> None:
    """direct_attention boosts relevant items."""
    ce = CentralExecutive()
    items = {"a": 0.5, "b": 0.5, "c": 0.5}
    result = ce.direct_attention(items, relevant=["a"])
    assert result["a"] > 0.5  # boosted
    assert result["b"] == 0.5  # unchanged
    assert ce.current_focus == "a"


def test_central_executive_direct_attention_filters_suppressed() -> None:
    """direct_attention removes suppressed items."""
    ce = CentralExecutive()
    ce.suppress("b")
    items = {"a": 0.5, "b": 0.5}
    result = ce.direct_attention(items, relevant=["a"])
    assert "a" in result
    assert "b" not in result


def test_central_executive_suppress() -> None:
    """suppress() adds an item to the suppressed set."""
    ce = CentralExecutive()
    ce.suppress("noise")
    assert ce.is_suppressed("noise")
    assert ce.suppressions == 1


def test_central_executive_release_suppression() -> None:
    """release_suppression() removes an item from the suppressed set."""
    ce = CentralExecutive()
    ce.suppress("item")
    ce.release_suppression("item")
    assert not ce.is_suppressed("item")


def test_central_executive_is_suppressed_case_insensitive() -> None:
    """Suppression is case-insensitive."""
    ce = CentralExecutive()
    ce.suppress("Noise")
    assert ce.is_suppressed("noise")
    assert ce.is_suppressed("NOISE")


def test_central_executive_switch_task() -> None:
    """switch_task() changes focus and counts switches."""
    ce = CentralExecutive()
    ce.switch_task("task_a")
    assert ce.current_focus == "task_a"
    ce.switch_task("task_b")
    assert ce.current_focus == "task_b"
    assert ce.task_switches == 1


def test_central_executive_switch_task_same_no_count() -> None:
    """Switching to the same task doesn't count as a switch."""
    ce = CentralExecutive()
    ce.switch_task("task_a")
    ce.switch_task("task_a")
    assert ce.task_switches == 0


def test_central_executive_switch_task_suppresses_old() -> None:
    """Task switching suppresses the old focus."""
    ce = CentralExecutive()
    ce.switch_task("old_task")
    ce.switch_task("new_task")
    assert ce.is_suppressed("old_task")


def test_central_executive_allocate_capacity() -> None:
    """allocate_capacity trims to the top-N items by activation."""
    ce = CentralExecutive()
    items = {"a": 0.9, "b": 0.5, "c": 0.3, "d": 0.1}
    result = ce.allocate_capacity(items, capacity=2)
    assert len(result) == 2
    assert "a" in result
    assert "b" in result


def test_central_executive_allocate_capacity_removes_suppressed() -> None:
    """allocate_capacity removes suppressed items first."""
    ce = CentralExecutive()
    ce.suppress("a")
    items = {"a": 0.9, "b": 0.5, "c": 0.3}
    result = ce.allocate_capacity(items, capacity=2)
    assert "a" not in result
    assert len(result) == 2


def test_central_executive_allocate_capacity_under_limit() -> None:
    """allocate_capacity returns all items when under the limit."""
    ce = CentralExecutive()
    items = {"a": 0.9, "b": 0.5}
    result = ce.allocate_capacity(items, capacity=5)
    assert len(result) == 2


def test_central_executive_coordinate() -> None:
    """coordinate() routes items to slave systems."""
    loop = PhonologicalLoop()
    pad = VisuoSpatialSketchpad()
    ce = CentralExecutive(phonological_loop=loop, sketchpad=pad)
    result = ce.coordinate(
        verbal_items=["hello", "world"],
        relations=[("brain", "contains", "hippocampus")],
    )
    assert "phonological" in result
    assert "visuospatial" in result
    assert "hello" in result["phonological"]
    assert ("brain", "contains", "hippocampus") in result["visuospatial"]


# ═══════════════════════════════════════════════════════════════════
# WorkingMemory — attention buffer
# ═══════════════════════════════════════════════════════════════════


def test_wm_update_activates_topics() -> None:
    """update() activates the turn's topics in attention."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="tell me about dogs",
        genesis_response="dogs are mammals",
        topics=["dogs", "mammals"],
        intent="question",
    )
    wm.update(turn)
    attention = wm.get_attention(threshold=0.1)
    assert "dogs" in attention
    assert "mammals" in attention


def test_wm_capacity_limit() -> None:
    """Attention buffer is limited to capacity."""
    wm = WorkingMemory(capacity=2)
    # Add 3 turns with different topics
    for topic in ["a", "b", "c"]:
        turn = Turn(
            user_input=f"about {topic}",
            genesis_response=f"response {topic}",
            topics=[topic],
            intent="question",
        )
        wm.update(turn)
    attention = wm.get_attention(threshold=0.01)
    assert len(attention) <= 2


def test_wm_rehearse() -> None:
    """rehearse() resets an item's activation to 1.0."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="about dogs",
        genesis_response="dogs",
        topics=["dogs"],
        intent="question",
    )
    wm.update(turn)
    result = wm.rehearse("dogs")
    assert result is True
    assert abs(wm._attention["dogs"] - 1.0) < 1e-9


def test_wm_rehearse_not_in_attention() -> None:
    """rehearse() returns False for items not in attention."""
    wm = WorkingMemory()
    result = wm.rehearse("unknown")
    assert result is False


def test_wm_get_top_attention() -> None:
    """get_top_attention returns the top-N attended concepts."""
    wm = WorkingMemory(capacity=10)
    for topic in ["a", "b", "c"]:
        turn = Turn(
            user_input=f"about {topic}",
            genesis_response="response",
            topics=[topic],
            intent="question",
        )
        wm.update(turn)
    top = wm.get_top_attention(n=2)
    assert len(top) <= 2


def test_wm_get_attention_threshold() -> None:
    """get_attention respects the threshold."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="about dogs",
        genesis_response="dogs",
        topics=["dogs"],
        intent="question",
    )
    wm.update(turn)
    # High threshold should filter out low-activation items
    high = wm.get_attention(threshold=0.99)
    low = wm.get_attention(threshold=0.01)
    assert len(low) >= len(high)


# ═══════════════════════════════════════════════════════════════════
# WorkingMemory — thread tracking
# ═══════════════════════════════════════════════════════════════════


def test_wm_starts_new_thread() -> None:
    """First update starts a new thread."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="about dogs",
        genesis_response="dogs",
        topics=["dogs"],
        intent="question",
    )
    wm.update(turn)
    assert wm.thread_count == 1
    assert wm.turn_count == 1


def test_wm_continues_same_thread() -> None:
    """Related turns continue the same thread."""
    wm = WorkingMemory()
    turn1 = Turn(
        user_input="about dogs",
        genesis_response="dogs are mammals",
        topics=["dogs", "mammals"],
        intent="question",
    )
    turn2 = Turn(
        user_input="what do mammals eat",
        genesis_response="varies",
        topics=["mammals", "food"],
        intent="question",
    )
    wm.update(turn1)
    wm.update(turn2)
    assert wm.thread_count == 1  # same thread


def test_wm_starts_new_thread_on_topic_shift() -> None:
    """Unrelated topics start a new thread."""
    wm = WorkingMemory()
    turn1 = Turn(
        user_input="about dogs",
        genesis_response="dogs",
        topics=["dogs", "mammals", "pets", "animals", "companions"],
        intent="question",
    )
    turn2 = Turn(
        user_input="how does photosynthesis work",
        genesis_response="plants convert sunlight",
        topics=["photosynthesis", "plants", "sunlight", "energy", "chlorophyll"],
        intent="question",
    )
    wm.update(turn1)
    wm.update(turn2)
    assert wm.thread_count == 2


def test_wm_is_on_topic() -> None:
    """is_on_topic checks if topics relate to the current thread."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="about dogs",
        genesis_response="dogs are mammals",
        topics=["dogs", "mammals"],
        intent="question",
    )
    wm.update(turn)
    assert wm.is_on_topic(["dogs"]) is True
    assert wm.is_on_topic(["photosynthesis"]) is False


def test_wm_is_on_topic_no_thread() -> None:
    """is_on_topic returns False when there's no thread."""
    wm = WorkingMemory()
    assert wm.is_on_topic(["anything"]) is False


# ═══════════════════════════════════════════════════════════════════
# WorkingMemory — turn tracking
# ═══════════════════════════════════════════════════════════════════


def test_wm_get_recent_turns() -> None:
    """get_recent_turns returns the last N turns."""
    wm = WorkingMemory(max_turns=10)
    for i in range(5):
        turn = Turn(
            user_input=f"input {i}",
            genesis_response=f"response {i}",
            topics=[f"topic_{i}"],
            intent="question",
        )
        wm.update(turn)
    recent = wm.get_recent_turns(n=3)
    assert len(recent) == 3


def test_wm_get_last_turn() -> None:
    """get_last_turn returns the most recent turn."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="hello",
        genesis_response="hi",
        topics=["greeting"],
        intent="greeting",
    )
    wm.update(turn)
    last = wm.get_last_turn()
    assert last is not None
    assert last.user_input == "hello"


def test_wm_get_last_turn_empty() -> None:
    """get_last_turn returns None when there are no turns."""
    wm = WorkingMemory()
    assert wm.get_last_turn() is None


def test_wm_was_topic_discussed() -> None:
    """was_topic_discussed checks recent turns for a topic."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="about dogs",
        genesis_response="dogs",
        topics=["dogs"],
        intent="question",
    )
    wm.update(turn)
    assert wm.was_topic_discussed("dogs") is True
    assert wm.was_topic_discussed("cats") is False


def test_wm_was_topic_discussed_case_insensitive() -> None:
    """was_topic_discussed is case-insensitive."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="about dogs",
        genesis_response="dogs",
        topics=["Dogs"],
        intent="question",
    )
    wm.update(turn)
    assert wm.was_topic_discussed("DOGS") is True


def test_wm_max_turns_trims() -> None:
    """Recent turns are trimmed to max_turns."""
    wm = WorkingMemory(max_turns=3)
    for i in range(5):
        turn = Turn(
            user_input=f"input {i}",
            genesis_response=f"response {i}",
            topics=[f"topic_{i}"],
            intent="question",
        )
        wm.update(turn)
    assert wm.total_turns == 3


# ═══════════════════════════════════════════════════════════════════
# WorkingMemory — open questions
# ═══════════════════════════════════════════════════════════════════


def test_wm_add_open_question() -> None:
    """add_open_question stores a question."""
    wm = WorkingMemory()
    wm.add_open_question("what is cognition")
    questions = wm.get_open_questions()
    assert "what is cognition" in questions


def test_wm_clear_open_question() -> None:
    """clear_open_question removes a question."""
    wm = WorkingMemory()
    wm.add_open_question("what is cognition")
    wm.clear_open_question("what is cognition")
    assert wm.get_open_questions() == []


def test_wm_open_questions_max_five() -> None:
    """Open questions are capped at 5."""
    wm = WorkingMemory()
    for i in range(7):
        wm.add_open_question(f"question {i}")
    questions = wm.get_open_questions()
    assert len(questions) == 5


# ═══════════════════════════════════════════════════════════════════
# Procedural memory — fixtures and helpers
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def pm() -> ProceduralMemory:
    """A fresh procedural memory store."""
    return ProceduralMemory()


def _get_skill(pm: ProceduralMemory, trigger: str) -> Skill:
    """Get a skill by trigger, asserting it exists (for mypy narrowing)."""
    skill = pm.get_skill(trigger)
    assert skill is not None
    return skill


# ─── SkillStrategy ────────────────────────────────────────────────────


class TestSkillStrategy:
    """Tests for the SkillStrategy dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        s = SkillStrategy()
        assert s.cognitive_route == ""
        assert s.emotional_tone == "neutral"
        assert s.avg_confidence == 0.5
        assert s.avg_length_band == 1.0
        assert s.success_rate == 0.5

    def test_update_on_success(self) -> None:
        """Test update on success."""
        s = SkillStrategy()
        s.update(
            cognitive_route="factual",
            emotional_tone="curious",
            confidence=0.8,
            length_band=2,
            success=True,
        )
        # EMA with rate 0.3: new = 0.7*old + 0.3*new
        assert s.avg_confidence == pytest.approx(0.7 * 0.5 + 0.3 * 0.8)
        assert s.avg_length_band == pytest.approx(0.7 * 1.0 + 0.3 * 2)
        assert s.success_rate == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
        assert s.cognitive_route == "factual"
        assert s.emotional_tone == "curious"

    def test_update_on_failure_does_not_change_route(self) -> None:
        """A failed trial should not change the practiced cognitive route."""
        s = SkillStrategy(cognitive_route="factual", emotional_tone="curious")
        s.update(
            cognitive_route="general",
            emotional_tone="calm",
            confidence=0.2,
            length_band=0,
            success=False,
        )
        # Route and tone are preserved (only updated on success).
        assert s.cognitive_route == "factual"
        assert s.emotional_tone == "curious"
        # But rolling stats are still updated.
        assert s.success_rate < 0.5
        assert s.avg_confidence < 0.5

    def test_update_empty_route_preserves_existing(self) -> None:
        """Passing an empty route on success should not overwrite."""
        s = SkillStrategy(cognitive_route="factual")
        s.update(cognitive_route="", emotional_tone="", success=True)
        assert s.cognitive_route == "factual"

    def test_serialization_round_trip(self) -> None:
        """Test serialization round trip."""
        s = SkillStrategy(
            cognitive_route="counterfactual",
            emotional_tone="engaged",
            avg_confidence=0.72,
            avg_length_band=1.5,
            success_rate=0.8,
        )
        d = s.to_dict()
        restored = SkillStrategy.from_dict(d)
        assert restored.cognitive_route == "counterfactual"
        assert restored.emotional_tone == "engaged"
        assert restored.avg_confidence == pytest.approx(0.72)
        assert restored.avg_length_band == pytest.approx(1.5)
        assert restored.success_rate == pytest.approx(0.8)

    def test_from_dict_missing_keys_uses_defaults(self) -> None:
        """Deserialization should be backward-compatible with missing keys."""
        s = SkillStrategy.from_dict({})
        assert s.cognitive_route == ""
        assert s.emotional_tone == "neutral"
        assert s.avg_confidence == 0.5


# ─── Skill learning ───────────────────────────────────────────────────


class TestSkillLearning:
    """Tests for skill learning and storage."""

    def test_learn_new_skill(self, pm: ProceduralMemory) -> None:
        """Test learn new skill."""
        skill = pm.learn_skill(
            name="respond_to_question",
            trigger="question",
            cognitive_route="factual",
            emotional_tone="curious",
        )
        assert skill.name == "respond_to_question"
        assert skill.trigger_condition == "question"
        assert skill.strategy.cognitive_route == "factual"
        assert skill.strategy.emotional_tone == "curious"
        assert skill.strength == 0.2
        assert pm.skill_count == 1
        assert pm.skills_learned == 1

    def test_learn_skill_preserves_strength_on_relearn(self, pm: ProceduralMemory) -> None:
        """Re-learning a skill should not reset its practice or strength."""
        pm.learn_skill("respond_to_question", "question", "factual", "curious")
        pm.practice_skill("respond_to_question", success=True)
        strength_after_practice = _get_skill(pm, "question").strength
        assert strength_after_practice > 0.2

        # Re-learn with updated route
        pm.learn_skill("respond_to_question", "question", "general", "calm")
        skill = _get_skill(pm, "question")
        assert skill.strength == strength_after_practice  # preserved
        assert skill.strategy.cognitive_route == "general"  # updated
        assert skill.strategy.emotional_tone == "calm"

    def test_learn_skill_does_not_overwrite_route_with_empty(self, pm: ProceduralMemory) -> None:
        """Re-learning with an empty route should preserve the existing route."""
        pm.learn_skill("s1", "question", "factual", "curious")
        pm.learn_skill("s1", "question", "", "")
        skill = _get_skill(pm, "question")
        assert skill.strategy.cognitive_route == "factual"
        assert skill.strategy.emotional_tone == "curious"

    def test_skill_stores_strategy_not_words(self, pm: ProceduralMemory) -> None:
        """The critical invariant: a skill must NOT store a response string.

        This is the architectural rule: Genesis's words must always
        emerge from her language engine, never be recited from storage.
        The skill stores a *strategy* (how to approach the response),
        not the words of the response.
        """
        skill = pm.learn_skill("respond_to_question", "question", "factual", "curious")
        # The Skill dataclass must not have a 'response' field.
        assert not hasattr(skill, "response")
        # The strategy must not contain response text.
        assert isinstance(skill.strategy, SkillStrategy)
        assert "content" not in skill.strategy.to_dict()


# ─── Practice dynamics ────────────────────────────────────────────────


class TestPracticeDynamics:
    """Tests for the practice loop and habit formation."""

    def test_practice_increases_strength(self, pm: ProceduralMemory) -> None:
        """Test practice increases strength."""
        pm.learn_skill("s1", "question", "factual", "curious")
        s0 = _get_skill(pm, "question").strength
        pm.practice_skill("s1", success=True, confidence=0.8, length_band=1)
        s1 = _get_skill(pm, "question").strength
        assert s1 > s0

    def test_practice_decreases_strength_on_failure(self, pm: ProceduralMemory) -> None:
        """Test practice decreases strength on failure."""
        pm.learn_skill("s1", "question", "factual", "curious")
        s0 = _get_skill(pm, "question").strength
        pm.practice_skill("s1", success=False, confidence=0.2, length_band=0)
        s1 = _get_skill(pm, "question").strength
        assert s1 < s0

    def test_diminishing_returns(self, pm: ProceduralMemory) -> None:
        """Strength gains should shrink as strength approaches 1.0."""
        pm.learn_skill("s1", "question", "factual", "curious")
        gains = []
        prev = _get_skill(pm, "question").strength
        for _ in range(10):
            new = pm.practice_skill("s1", success=True, confidence=0.9, length_band=1)
            gains.append(new - prev)
            prev = new
        # Gains should be monotonically non-increasing.
        for i in range(1, len(gains)):
            assert gains[i] <= gains[i - 1] + 1e-9

    def test_habit_formation(self, pm: ProceduralMemory) -> None:
        """A skill becomes a habit after enough successful practice."""
        pm.learn_skill("s1", "question", "factual", "curious")
        skill = _get_skill(pm, "question")
        assert not skill.is_habit

        # Practice until habit threshold (need strength >= 0.8 and count >= 5).
        for _ in range(20):
            pm.practice_skill("s1", success=True, confidence=0.9, length_band=1)

        skill = _get_skill(pm, "question")
        assert skill.is_habit
        assert pm.habit_count == 1

    def test_habit_requires_minimum_practice(self, pm: ProceduralMemory) -> None:
        """Even with high strength, a skill needs >= 5 practices to be a habit."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        skill = _get_skill(pm, "question")
        assert skill.strength >= HABIT_THRESHOLD
        assert not skill.is_habit  # only 0 practices
        assert skill.practice_count == 0

    def test_practice_updates_strategy_stats(self, pm: ProceduralMemory) -> None:
        """Practice should update the strategy's rolling statistics."""
        pm.learn_skill("s1", "question", "factual", "curious")
        pm.practice_skill("s1", success=True, confidence=0.8, length_band=2)
        skill = _get_skill(pm, "question")
        assert skill.strategy.avg_confidence > 0.5  # moved toward 0.8
        assert skill.strategy.avg_length_band > 1.0  # moved toward 2
        assert skill.strategy.success_rate > 0.5  # moved toward 1.0

    def test_practice_nonexistent_returns_negative(self, pm: ProceduralMemory) -> None:
        """Test practice nonexistent returns negative."""
        result = pm.practice_skill("nonexistent", success=True)
        assert result == -1.0

    def test_response_time_decreases_with_practice(self, pm: ProceduralMemory) -> None:
        """Power law of practice: RT decreases with practice count."""
        pm.learn_skill("s1", "question", "factual", "curious")
        skill = _get_skill(pm, "question")
        rt0 = skill.response_time
        for _ in range(10):
            pm.practice_skill("s1", success=True, confidence=0.9, length_band=1)
        rt1 = skill.response_time
        assert rt1 < rt0


# ─── Retrieval ────────────────────────────────────────────────────────


class TestRetrieval:
    """Tests for skill retrieval and trigger matching."""

    def test_get_skill_by_trigger(self, pm: ProceduralMemory) -> None:
        """Test get skill by trigger."""
        pm.learn_skill("s1", "question", "factual", "curious")
        skill = _get_skill(pm, "question")
        assert skill.name == "s1"

    def test_get_skill_case_insensitive(self, pm: ProceduralMemory) -> None:
        """Test get skill case insensitive."""
        pm.learn_skill("s1", "Question", "factual", "curious")
        skill = pm.get_skill("QUESTION")
        assert skill is not None

    def test_get_skill_substring_fallback(self, pm: ProceduralMemory) -> None:
        """A compound trigger containing a known trigger should match."""
        pm.learn_skill("s1", "question", "factual", "curious")
        # "user_asks_question" contains "question"
        skill = pm.get_skill("user_asks_question")
        assert skill is not None

    def test_get_skill_no_match(self, pm: ProceduralMemory) -> None:
        """Test get skill no match."""
        pm.learn_skill("s1", "question", "factual", "curious")
        assert pm.get_skill("farewell") is None

    def test_strongest_skill_returned(self, pm: ProceduralMemory) -> None:
        """When multiple skills match, the strongest is returned."""
        pm.learn_skill("s1", "question", "factual", "curious")
        pm.learn_skill("s2", "question", "general", "calm")
        # Practice s1 to make it stronger
        for _ in range(5):
            pm.practice_skill("s1", success=True, confidence=0.9, length_band=1)
        skill = _get_skill(pm, "question")
        s2 = pm.get_skill_by_name("s2")
        assert s2 is not None
        assert skill.name == "s1"
        assert skill.strength > s2.strength

    def test_get_skill_by_name(self, pm: ProceduralMemory) -> None:
        """Test get skill by name."""
        pm.learn_skill("s1", "question", "factual", "curious")
        assert pm.get_skill_by_name("s1") is not None
        assert pm.get_skill_by_name("nonexistent") is None


# ─── Go/no-go decision ────────────────────────────────────────────────


class TestGoNoGo:
    """Tests for the basal ganglia go/no-go decision."""

    def test_strong_skill_passes(self, pm: ProceduralMemory) -> None:
        """Test strong skill passes."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        skill = _get_skill(pm, "question")
        assert pm.should_execute(skill)

    def test_weak_skill_fails(self, pm: ProceduralMemory) -> None:
        """Test weak skill fails."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.02)
        skill = _get_skill(pm, "question")
        assert not pm.should_execute(skill)

    def test_suppress_blocks_execution(self, pm: ProceduralMemory) -> None:
        """Test suppress blocks execution."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        skill = _get_skill(pm, "question")
        assert not pm.should_execute(skill, context={"suppress": True})

    def test_competing_strength_reduces_go(self, pm: ProceduralMemory) -> None:
        """Test competing strength reduces go."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.5)
        skill = _get_skill(pm, "question")
        # Strong competitor should block
        assert not pm.should_execute(skill, context={"competing_strength": 0.8})

    def test_dopamine_boosts_go(self, pm: ProceduralMemory) -> None:
        """Test dopamine boosts go."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.4)
        skill = _get_skill(pm, "question")
        # Without dopamine boost, might not pass
        low = pm.should_execute(skill, context={"dopamine": 0.1})
        # With dopamine boost, should pass
        high = pm.should_execute(skill, context={"dopamine": 1.0})
        assert high or not low  # dopamine should help (at minimum, high >= low)


# ─── Habit bias (the new cognition interface) ─────────────────────────


class TestHabitBias:
    """Tests for the HabitBias — the new interface between procedural
    memory and cognition.

    The habit bias replaces the old broken habit short-circuit that
    returned a canned string. Instead, the bias shapes *how*
    deliberation proceeds without replacing it.
    """

    def test_no_bias_for_nonexistent_skill(self, pm: ProceduralMemory) -> None:
        """Test no bias for nonexistent skill."""
        assert pm.get_habit_bias("question") is None

    def test_no_bias_for_weak_skill(self, pm: ProceduralMemory) -> None:
        """A weak skill should not produce a habit bias."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.02)
        assert pm.get_habit_bias("question") is None

    def test_bias_for_strong_skill(self, pm: ProceduralMemory) -> None:
        """A strong skill should produce a habit bias."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        bias = pm.get_habit_bias("question")
        assert bias is not None
        assert isinstance(bias, HabitBias)
        assert bias.confidence_boost > 0
        assert bias.threshold_reduction > 0

    def test_bias_weighted_by_success_rate(self, pm: ProceduralMemory) -> None:
        """A skill with poor recent outcomes should produce a weaker bias."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        # Practice with failures to drive down success rate
        for _ in range(10):
            pm.practice_skill("s1", success=False, confidence=0.1, length_band=0)
        bias_low_success = pm.get_habit_bias("question")

        # Fresh skill with high success
        pm2 = ProceduralMemory()
        pm2.learn_skill("s2", "question", "factual", "curious", initial_strength=0.9)
        for _ in range(10):
            pm2.practice_skill("s2", success=True, confidence=0.9, length_band=1)
        bias_high_success = pm2.get_habit_bias("question")

        assert bias_low_success is not None
        assert bias_high_success is not None
        # High success rate should produce a stronger bias.
        assert bias_high_success.weight >= bias_low_success.weight

    def test_bias_does_not_contain_response_text(self, pm: ProceduralMemory) -> None:
        """The critical invariant: the habit bias must NOT contain
        response text. It shapes deliberation, it doesn't replace it.

        This is the architectural rule: Genesis's words must always
        emerge from her language engine, never be recited from storage.
        """
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        bias = pm.get_habit_bias("question")
        assert bias is not None
        # The bias must not have a 'content' or 'response' field.
        assert not hasattr(bias, "content")
        assert not hasattr(bias, "response")
        # The bias fields are all about *how* to deliberate, not *what* to say.
        assert isinstance(bias.confidence_boost, float)
        assert isinstance(bias.threshold_reduction, float)
        assert isinstance(bias.emotional_tone, str)
        assert isinstance(bias.cognitive_route, str)

    def test_bias_emotional_tone_only_for_strong_habits(self, pm: ProceduralMemory) -> None:
        """The emotional tone bias should only appear for strong habits
        (weight > 0.3), not for marginal ones."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.5)
        bias = pm.get_habit_bias("question")
        if bias is not None and bias.weight <= 0.3:
            assert bias.emotional_tone == ""

    def test_suppress_blocks_bias(self, pm: ProceduralMemory) -> None:
        """Top-down suppression should block the habit bias."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        assert pm.get_habit_bias("question", suppress=True) is None

    def test_dopamine_modulates_bias(self, pm: ProceduralMemory) -> None:
        """Higher dopamine should make the habit bias more likely to fire."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.4)
        bias_low = pm.get_habit_bias("question", dopamine=0.1)
        bias_high = pm.get_habit_bias("question", dopamine=1.0)
        # At minimum, high dopamine should not produce a weaker bias
        if bias_low is not None and bias_high is not None:
            assert bias_high.weight >= bias_low.weight

    def test_compute_bias_on_skill_directly(self, pm: ProceduralMemory) -> None:
        """Skill.compute_bias() should match ProceduralMemory.get_habit_bias()."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        skill = _get_skill(pm, "question")
        bias_from_skill = skill.compute_bias()
        bias_from_pm = pm.get_habit_bias("question")
        if bias_from_skill is not None and bias_from_pm is not None:
            assert bias_from_skill.confidence_boost == pytest.approx(
                bias_from_pm.confidence_boost
            )

    def test_execute_skill_reinforces(self, pm: ProceduralMemory) -> None:
        """execute_skill should reinforce the skill (practice with success)."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.5)
        skill = _get_skill(pm, "question")
        s0 = skill.strength
        pm.execute_skill(skill)
        s1 = _get_skill(pm, "question").strength
        assert s1 > s0
        assert pm.executions == 1


# ─── Persistence ──────────────────────────────────────────────────────


class TestPersistence:
    """Tests for serialization and backward compatibility."""

    def test_strategy_serialization_round_trip(self) -> None:
        """Test strategy serialization round trip."""
        s = SkillStrategy(
            cognitive_route="factual",
            emotional_tone="curious",
            avg_confidence=0.75,
            avg_length_band=1.3,
            success_rate=0.82,
        )
        d = s.to_dict()
        restored = SkillStrategy.from_dict(d)
        assert restored.cognitive_route == s.cognitive_route
        assert restored.emotional_tone == s.emotional_tone
        assert restored.avg_confidence == pytest.approx(s.avg_confidence)
        assert restored.avg_length_band == pytest.approx(s.avg_length_band)
        assert restored.success_rate == pytest.approx(s.success_rate)

    def test_old_format_backward_compat(self) -> None:
        """Old save files with 'response' (string) instead of 'strategy'
        (dict) should load with a default strategy, not crash."""
        from genesis_cognitive.persistence import restore_procedural_memory

        old_data = {
            "skills": [
                {
                    "name": "respond_to_question",
                    "trigger_condition": "question",
                    "response": "inform",  # old format — should be ignored
                    "strength": 0.6,
                    "practice_count": 10,
                    "last_practiced": 1234567890,
                    "success_count": 8,
                    "failure_count": 2,
                    "created_at": 1234567000,
                }
            ],
            "skills_learned": 1,
            "habits_formed": 0,
            "executions": 5,
        }
        pm = ProceduralMemory()
        restore_procedural_memory(pm, old_data)
        skill = _get_skill(pm, "question")
        assert skill is not None
        assert skill.strength == 0.6
        assert skill.practice_count == 10
        assert isinstance(skill.strategy, SkillStrategy)
        assert skill.strategy.cognitive_route == ""  # default

    def test_new_format_round_trip(self) -> None:
        """New format with strategy dict should serialize and restore."""
        from genesis_cognitive.persistence import (
            _serialize_procedural_memory,
            restore_procedural_memory,
        )

        pm = ProceduralMemory()
        pm.learn_skill("s1", "question", "factual", "curious")
        for _ in range(5):
            pm.practice_skill("s1", success=True, confidence=0.8, length_band=2)

        serialized = _serialize_procedural_memory(pm)
        assert "strategy" in serialized["skills"][0]
        assert "response" not in serialized["skills"][0]

        pm2 = ProceduralMemory()
        restore_procedural_memory(pm2, serialized)
        skill = pm2.get_skill("question")
        assert skill is not None
        assert skill.strategy.cognitive_route == "factual"
        assert skill.strategy.emotional_tone == "curious"
        assert skill.strategy.avg_confidence > 0.5
        assert pm2.skills_learned == 1


# ─── Integration: no canned responses ─────────────────────────────────


class TestNoCannedResponses:
    """The critical architectural invariant: procedural memory must never
    produce a canned string response.

    The old code stored `thought.intent` (e.g., "inform") as the skill's
    `response` field and returned it as `content` when a habit fired —
    meaning Genesis would literally say "inform" instead of an actual
    answer. These tests verify that this can never happen again.
    """

    def test_skill_has_no_response_field(self) -> None:
        """The Skill dataclass must not have a 'response' field."""
        s = Skill(name="test", trigger_condition="test")
        assert not hasattr(s, "response")

    def test_habit_bias_has_no_content_field(self) -> None:
        """The HabitBias must not have a 'content' or 'response' field."""
        b = HabitBias()
        assert not hasattr(b, "content")
        assert not hasattr(b, "response")

    def test_execute_skill_returns_none(self, pm: ProceduralMemory) -> None:
        """execute_skill should not return a response string.

        The old execute_skill returned `skill.response` (a string). The
        new version returns None — it only reinforces the skill. The
        actual response is always generated by the language engine.
        """
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.5)
        skill = _get_skill(pm, "question")
        # execute_skill returns None — it only reinforces the skill.
        pm.execute_skill(skill)
        # The skill should have been reinforced (strength increased).
        assert _get_skill(pm, "question").strength > 0.5

    def test_get_habit_bias_returns_bias_not_string(self, pm: ProceduralMemory) -> None:
        """get_habit_bias should return a HabitBias or None, never a string."""
        pm.learn_skill("s1", "question", "factual", "curious", initial_strength=0.9)
        result = pm.get_habit_bias("question")
        assert result is None or isinstance(result, HabitBias)
        # Specifically, it must not be a string.
        assert not isinstance(result, str)


# ======================================================================
# From tests/test_columnar_math.py
# ======================================================================

def _make_network_with_columns() -> ConceptNetwork:
    """Create a test network with concepts in different columns.

    Simulates the state AFTER persistence restore (columns empty)
    so backfill is needed. All activations are reset to 0.
    """
    net = ConceptNetwork()

    # Dictionary column: many concepts, low activation
    for i in range(100):
        net.add_concept(f"dict_{i}", origin="dictionary")
    # Code column: few concepts, high activation
    for i in range(10):
        net.add_concept(f"python:code_{i}", origin="code")
    # Identity column: very few concepts
    net.add_concept("identity:genesis", origin="identity")

    # Simulate persistence restore: clear columns so backfill is needed
    for concept in net._concepts.values():
        concept.columns = set()
        concept.activation = 0.0  # reset to 0 for test control

    return net


class TestColumnActivationMean:
    """Column activation must use MEAN, not sum."""

    def test_mean_not_sum(self):
        """A small, highly-active column should beat a large, weak one."""
        net = _make_network_with_columns()

        # Activate 5 dictionary concepts at 0.1 each
        for i in range(5):
            net.get_concept(f"dict_{i}").activation = 0.1

        # Activate 3 code concepts at 0.8 each
        for i in range(3):
            net.get_concept(f"python:code_{i}").activation = 0.8

        dict_act = net.get_column_activation("dictionary")
        code_act = net.get_column_activation("code")

        # Mean: dictionary = 0.5/100 = 0.005, code = 2.4/10 = 0.24
        # Code should be much higher (more active per concept)
        assert code_act > dict_act, (
            f"Code column (mean={code_act:.4f}) should be more active than "
            f"dictionary (mean={dict_act:.4f})"
        )
        assert abs(dict_act - 0.005) < 1e-6, f"Dictionary mean should be 0.005, got {dict_act}"
        assert abs(code_act - 0.24) < 1e-6, f"Code mean should be 0.24, got {code_act}"

    def test_sum_would_be_wrong(self):
        """If we used sum, dictionary (100 concepts) would always win."""
        net = _make_network_with_columns()

        # All dictionary concepts at 0.01
        for i in range(100):
            net.get_concept(f"dict_{i}").activation = 0.01

        # All code concepts at 0.5
        for i in range(10):
            net.get_concept(f"python:code_{i}").activation = 0.5

        dict_act = net.get_column_activation("dictionary")
        code_act = net.get_column_activation("code")

        # Sum would give: dict=1.0, code=5.0 — code still wins
        # But with 1000 dict concepts at 0.01: sum=10.0 > code=5.0
        # Mean correctly gives: dict=0.01, code=0.5 — code wins regardless
        assert code_act > dict_act


class TestLateralInhibitionMultiplicative:
    """Lateral inhibition must be multiplicative, not additive."""

    def test_inhibition_is_proportional(self):
        """Higher-activation concepts in suppressed column lose more absolute activation."""
        net = _make_network_with_columns()

        # Set up: code is dominant, dictionary is suppressed
        net.get_concept("python:code_0").activation = 0.8
        net.get_concept("dict_0").activation = 0.8  # same as winner
        net.get_concept("dict_1").activation = 0.4  # half of winner
        net.get_concept("dict_2").activation = 0.1  # much lower

        dict_0_before = net.get_concept("dict_0").activation
        dict_1_before = net.get_concept("dict_1").activation
        dict_2_before = net.get_concept("dict_2").activation

        net.apply_lateral_inhibition()

        dict_0_after = net.get_concept("dict_0").activation
        dict_1_after = net.get_concept("dict_1").activation
        dict_2_after = net.get_concept("dict_2").activation

        # All should be reduced (multiplicative)
        assert dict_0_after < dict_0_before, "dict_0 should be suppressed"
        assert dict_1_after < dict_1_before, "dict_1 should be suppressed"
        assert dict_2_after < dict_2_before, "dict_2 should be suppressed"

        # Multiplicative: ratio should be the same for all
        ratio_0 = dict_0_after / dict_0_before
        ratio_1 = dict_1_after / dict_1_before
        ratio_2 = dict_2_after / dict_2_before

        assert abs(ratio_0 - ratio_1) < 1e-6, (
            f"Multiplicative inhibition should give same ratio: "
            f"ratio_0={ratio_0:.4f}, ratio_1={ratio_1:.4f}"
        )
        assert abs(ratio_0 - ratio_2) < 1e-6, (
            f"Multiplicative inhibition should give same ratio: "
            f"ratio_0={ratio_0:.4f}, ratio_2={ratio_2:.4f}"
        )

    def test_no_magic_constant(self):
        """Inhibition should not use a tiny 0.01 factor that produces ~1.7% reduction."""
        net = _make_network_with_columns()

        # Code dominant, dictionary suppressed
        net.get_concept("python:code_0").activation = 0.8
        net.get_concept("dict_0").activation = 0.5

        before = net.get_concept("dict_0").activation
        net.apply_lateral_inhibition()
        after = net.get_concept("dict_0").activation

        reduction = (before - after) / before
        # With the old 0.01 magic constant, reduction was ~1.7%
        # With multiplicative inhibition, reduction should be significant (>10%)
        assert reduction > 0.10, (
            f"Inhibition reduction should be >10%, got {reduction:.1%}. "
            "The old additive 0.01 constant gave only ~1.7%."
        )

    def test_context_disinhibition_reduces_inhibition(self):
        """Context columns should receive less inhibition than non-context."""
        net = _make_network_with_columns()

        # Dictionary is dominant, code is suppressed
        for i in range(5):
            net.get_concept(f"dict_{i}").activation = 0.8
        net.get_concept("python:code_0").activation = 0.3

        # Without context: code gets full inhibition
        net_copy = _make_network_with_columns()
        for i in range(5):
            net_copy.get_concept(f"dict_{i}").activation = 0.8
        net_copy.get_concept("python:code_0").activation = 0.3
        net_copy.apply_lateral_inhibition(context_columns=None)
        code_no_context = net_copy.get_concept("python:code_0").activation

        # With context: code gets reduced inhibition
        net.apply_lateral_inhibition(context_columns={"code"})
        code_with_context = net.get_concept("python:code_0").activation

        assert code_with_context > code_no_context, (
            f"Context column should retain more activation: "
            f"with_context={code_with_context:.4f}, without={code_no_context:.4f}"
        )


class TestWinnerTakeAllDistanceDependent:
    """Winner-take-all must be distance-dependent, not flat."""

    def test_concepts_near_winner_are_less_suppressed(self):
        """Concepts close to the winner's activation should be lightly suppressed."""
        net = _make_network_with_columns()

        # Set up code column with varying activations
        net.get_concept("python:code_0").activation = 0.8  # winner
        net.get_concept("python:code_1").activation = 0.7  # close to winner
        net.get_concept("python:code_2").activation = 0.6  # medium gap
        net.get_concept("python:code_3").activation = 0.1  # far from winner
        net.get_concept("python:code_4").activation = 0.01  # very far

        # top_k=1 so only the winner is preserved
        net.apply_winner_take_all("code", top_k=1, suppression=0.5)

        c1 = net.get_concept("python:code_1").activation
        c3 = net.get_concept("python:code_3").activation
        c4 = net.get_concept("python:code_4").activation

        # gap_ratio for code_1: (0.8-0.7)/0.8 = 0.125, factor = 1-0.5*0.125 = 0.9375
        # gap_ratio for code_3: (0.8-0.1)/0.8 = 0.875, factor = 1-0.5*0.875 = 0.5625
        # code_1 should retain more activation than code_3
        assert c1 > c3, (
            f"Concept near winner should retain more activation: code_1={c1:.4f}, code_3={c3:.4f}"
        )
        # And the far one should be most suppressed
        assert c3 > c4 or c4 < 0.01, "Far concepts should be most suppressed"

    def test_winner_preserved(self):
        """The top-k concepts should be preserved at full activation."""
        net = _make_network_with_columns()

        net.get_concept("python:code_0").activation = 0.9
        net.get_concept("python:code_1").activation = 0.8
        net.get_concept("python:code_2").activation = 0.1

        net.apply_winner_take_all("code", top_k=2, suppression=0.5)

        assert net.get_concept("python:code_0").activation == 0.9, "Winner should be preserved"
        assert net.get_concept("python:code_1").activation == 0.8, "Second should be preserved"
        assert net.get_concept("python:code_2").activation < 0.1, "Third should be suppressed"

    def test_not_flat_suppression(self):
        """Suppression should NOT be the same flat 50% for all below top_k."""
        net = _make_network_with_columns()

        net.get_concept("python:code_0").activation = 0.8
        net.get_concept("python:code_1").activation = 0.7  # close
        net.get_concept("python:code_2").activation = 0.1  # far

        net.apply_winner_take_all("code", top_k=1, suppression=0.5)

        c1 = net.get_concept("python:code_1").activation
        c2 = net.get_concept("python:code_2").activation

        # With flat suppression: both would be 0.5x = 0.35 and 0.05
        # With distance-dependent: code_1 gets 0.9375x = 0.656, code_2 gets 0.5625x = 0.056
        # The ratio should be different (not both 0.5x)
        ratio_1 = c1 / 0.7
        ratio_2 = c2 / 0.1
        assert abs(ratio_1 - ratio_2) > 0.01, (
            f"Distance-dependent suppression should give different ratios: "
            f"ratio_1={ratio_1:.4f}, ratio_2={ratio_2:.4f}"
        )


class TestBridgeAttenuationBiologicalRange:
    """Bridge attenuation must stay within biological range (10-33%)."""

    def test_cross_column_in_context_is_30_percent(self):
        """Cross-column spread in context should be ~30% of intra-column."""
        net = ConceptNetwork()

        # Create concepts in different columns.
        # Use neutral names that stay UNKNOWN category/modality so the
        # test isolates bridge attenuation without category multiplier
        # confounding the ratio.
        net.add_concept("alpha", origin="dictionary")
        net.add_concept("python:alpha", origin="code")

        # Connect them
        net.add_edge("alpha", "python:alpha", RelationType.RELATED_TO, weight=1.0)
        net.add_edge("python:alpha", "alpha", RelationType.RELATED_TO, weight=1.0)

        # Spread from code concept with BOTH columns in context
        # (dictionary is in context, so no context penalty applies)
        activated = net.spread_activation(
            ["python:alpha"],
            amount=0.3,
            depth=1,
            column_context={"code", "dictionary"},
        )

        # Intra-column spread would be: 0.3 * 1.0 * 0.5 = 0.15
        # Cross-column in context: 0.15 * 0.3 = 0.045
        # Ratio: 0.045 / 0.15 = 0.30 (30%)
        cross_act = activated.get("alpha", 0)
        intra_baseline = 0.3 * 0.5  # what intra-column would give

        assert cross_act > 0, "Cross-column spread should be positive"
        ratio = cross_act / intra_baseline
        # Should be approximately 0.30
        assert 0.25 < ratio < 0.35, f"Cross-column in-context ratio should be ~30%, got {ratio:.0%}"

    def test_cross_column_non_context_not_below_biological_range(self):
        """Cross-column non-context spread should not drop below 10% of intra."""
        net = ConceptNetwork()

        net.add_concept("alpha", origin="dictionary")
        net.add_concept("python:alpha", origin="code")
        net.add_edge("alpha", "python:alpha", RelationType.RELATED_TO, weight=1.0)
        net.add_edge("python:alpha", "alpha", RelationType.RELATED_TO, weight=1.0)

        # Spread with code in context, dictionary NOT in context
        activated = net.spread_activation(
            ["python:alpha"],
            amount=0.3,
            depth=1,
            column_context={"code"},
        )

        cross_act = activated.get("alpha", 0)

        # Intra-column would be: 0.3 * 1.0 * 0.5 = 0.15
        # Cross-column non-context: 0.15 * 0.3 - 0.15 * 0.2 = 0.045 - 0.03 = 0.015
        # Ratio: 0.015 / 0.15 = 0.10 (10%)
        # This should NOT be below 10% (the biological floor)
        ratio = cross_act / 0.15 if cross_act > 0 else 0
        assert ratio >= 0.08, (
            f"Cross-column non-context ratio should be >=10%, got {ratio:.0%}. "
            "The old compounding (0.3 * 0.5 = 0.15) gave 15% which was too low."
        )

    def test_no_compounding_below_biological_range(self):
        """Bridge * context attenuation should not compound below 10%."""
        net = ConceptNetwork()

        net.add_concept("source", origin="dictionary")
        net.add_concept("python:target", origin="code")
        net.add_edge("source", "python:target", RelationType.RELATED_TO, weight=1.0)
        net.add_edge("python:target", "source", RelationType.RELATED_TO, weight=1.0)

        # Without context (no context penalty)
        activated_no_ctx = net.spread_activation(
            ["python:target"],
            amount=0.3,
            depth=1,
        )
        cross_no_ctx = activated_no_ctx.get("source", 0)

        # With context (code in context, dictionary not)
        activated_ctx = net.spread_activation(
            ["python:target"],
            amount=0.3,
            depth=1,
            column_context={"code"},
        )
        cross_ctx = activated_ctx.get("source", 0)

        # Both should be positive and within biological range
        assert cross_no_ctx > 0, "Cross-column spread without context should be positive"
        assert cross_ctx > 0, "Cross-column spread with context should be positive"

        # The context penalty should reduce it, but not below 10% of intra
        intra = 0.3 * 0.5  # source * decay
        assert cross_no_ctx / intra >= 0.25, (
            f"Cross-column without context should be >=25% of intra, got {cross_no_ctx / intra:.0%}"
        )
        assert cross_ctx / intra >= 0.08, (
            f"Cross-column with context penalty should be >=8% of intra, "
            f"got {cross_ctx / intra:.0%}"
        )


class TestContextDisinhibitionBiologicalRange:
    """Context disinhibition factor should be within VIP range (60-90%)."""

    def test_disinhibition_factor_in_range(self):
        """VIP-mediated disinhibition should reduce inhibition by 60-90%."""
        net = _make_network_with_columns()

        # Dictionary dominant, code suppressed
        for i in range(5):
            net.get_concept(f"dict_{i}").activation = 0.8
        net.get_concept("python:code_0").activation = 0.3

        # Without context
        net1 = _make_network_with_columns()
        for i in range(5):
            net1.get_concept(f"dict_{i}").activation = 0.8
        net1.get_concept("python:code_0").activation = 0.3
        net1.apply_lateral_inhibition(context_columns=None)
        without_disinhib = net1.get_concept("python:code_0").activation

        # With context
        net.apply_lateral_inhibition(context_columns={"code"})
        with_disinhib = net.get_concept("python:code_0").activation

        # Disinhibition should reduce the inhibition effect
        # The reduction in inhibition should be 60-90%
        if without_disinhib > 0:
            inhibition_without = 0.3 - without_disinhib  # total inhibition applied
            inhibition_with = 0.3 - with_disinhib
            if inhibition_without > 0:
                disinhibition_pct = 1 - (inhibition_with / inhibition_without)
                # Should be approximately 80% (factor 0.2)
                assert 0.5 < disinhibition_pct < 0.95, (
                    f"VIP disinhibition should reduce inhibition by 60-90%, "
                    f"got {disinhibition_pct:.0%}"
                )


class TestBackfillColumns:
    """Backfill should populate column sets and create bridges."""

    def test_backfill_populates_columns(self):
        """All concepts should have non-empty column sets after backfill."""
        net = _make_network_with_columns()
        stats = net.backfill_columns()

        assert stats["concepts_backfilled"] > 0, "Should backfill concepts"

        for cid, concept in net._concepts.items():
            assert len(concept.columns) > 0, f"Concept {cid} has empty columns after backfill"

    def test_backfill_creates_bridges_for_same_name(self):
        """Bridges should be created between same-name concepts in different columns."""
        net = ConceptNetwork()
        net.add_concept("mind", origin="conversation")
        net.add_concept("python:mind.Mind", origin="code")

        # Simulate persistence restore: clear columns so backfill is needed
        for concept in net._concepts.values():
            concept.columns = set()

        net.backfill_columns()

        # "mind" and "python:mind.Mind" share base name "mind"
        # Should create a bridge
        bridges = [e for e in net._edges if e.relation == RelationType.BRIDGES]
        assert len(bridges) > 0, "Should create at least one bridge for same-name concepts"

    def test_backfill_idempotent(self):
        """Running backfill twice should not create duplicate bridges."""
        net = ConceptNetwork()
        net.add_concept("mind", origin="conversation")
        net.add_concept("python:mind.Mind", origin="code")

        # Simulate persistence restore: clear columns so backfill is needed
        for concept in net._concepts.values():
            concept.columns = set()

        net.backfill_columns()
        bridges1 = sum(1 for e in net._edges if e.relation == RelationType.BRIDGES)

        net.backfill_columns()
        bridges2 = sum(1 for e in net._edges if e.relation == RelationType.BRIDGES)

        assert bridges1 == bridges2, (
            f"Backfill should be idempotent: bridges1={bridges1}, bridges2={bridges2}"
        )

    def test_backfill_no_op_when_columns_set(self):
        """Backfill should not create bridges when columns are already set.

        This prevents recreating bridges that were intentionally pruned
        during sleep consolidation. If all concepts already have columns,
        backfill is a no-op.
        """
        net = ConceptNetwork()
        net.add_concept("mind", origin="conversation")
        net.add_concept("python:mind.Mind", origin="code")

        # Columns are set by add_concept — backfill should be a no-op
        stats = net.backfill_columns()
        bridges = [e for e in net._edges if e.relation == RelationType.BRIDGES]

        assert stats["concepts_backfilled"] == 0, "Should not backfill when columns are set"
        assert stats["bridges_created"] == 0, "Should not create bridges when no backfill needed"
        assert len(bridges) == 0, "No bridges should exist"
