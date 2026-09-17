"""Procedural memory — skills and habits (basal ganglia-inspired).

Procedural memory is the implicit memory system that stores skills and
habits — stimulus-response associations that become automatic with
practice. Unlike declarative memory (episodic and semantic), procedural
memory is not cognitively recalled; it is expressed through performance
(Squire, 1992). Knowing *how* to ride a bicycle is procedural; knowing
*that* bicycles have two wheels is semantic.

# Why this matters

Without procedural memory, every action requires deliberate cognitive
control. With it, well-practised skills become automatic — they execute
without occupying the central executive, freeing working memory for
novel problems. This is the cognitive basis of expertise and habit.

# Biological grounding

Procedural memory is supported by the basal ganglia, a group of
subcortical nuclei (striatum, globus pallidus, substantia nigra,
subthalamic nucleus). The basal ganglia implement a go/no-go decision
via two complementary pathways (Albin et al., 1989; DeLong, 1990):

- **Direct pathway (GO)**: striatal D1 neurons inhibit the internal
  segment of the globus pallidus (GPi), which disinhibits thalamocortical
  neurons → the skill is *facilitated*.
- **Indirect pathway (NO-GO)**: striatal D2 neurons inhibit the
  external globus pallidus (GPe), which disinhibits the subthalamic
  nucleus, which excites GPi → thalamocortical output is *suppressed*.

The balance between the direct and indirect pathways determines whether
a skill is executed. Dopamine modulates this balance: phasic dopamine
bursts strengthen the direct (GO) pathway, supporting learning
(Graybiel, 2008).

# Power law of practice

Skill acquisition follows a power law: response time decreases as a
power of practice count (Newell & Rosenbloom, 1981):

    RT = a * practice^(-b)

where ``a`` is the initial response time and ``b`` is the learning rate
(typically ~0.4). This captures the diminishing returns of practice —
early practice yields large gains, later practice yields smaller ones.

# Habit formation

A skill becomes a *habit* when its strength exceeds a threshold (0.8).
Habits are automatic — they execute without requiring cognitive
attention and are resistant to change (Graybiel, 2008). Once a habit
forms, the basal ganglia takes over from the goal-directed
prefrontal cortex, and the action becomes stimulus-driven rather than
outcome-driven.

# Strategy, not words

A skill stores a **strategy** — the cognitive approach that worked
well for a given intent — not the words of any particular response.
This respects the architectural rule that Genesis's words must always
emerge from her language engine, never be recited from storage. A
habit biases *how* she deliberates (faster, more confident, predisposed
to a practiced emotional tone); it does not replace deliberation with
a canned string. This mirrors real procedural memory: knowing *how* to
greet someone does not mean saying the exact same words every time —
it means the greeting flows more easily and naturally with practice.

References:
    - Squire (1992): declarative vs nondeclarative memory
    - Albin et al. (1989); DeLong (1990): basal ganglia direct/indirect
      pathways
    - Graybiel (2008): habit formation and the basal ganglia
    - Newell & Rosenbloom (1981): power law of practice
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HABIT_THRESHOLD", "HabitBias", "ProceduralMemory", "Skill", "SkillStrategy"]

# ─── Constants ────────────────────────────────────────────────────────

# A skill becomes a habit when its strength exceeds this threshold.
HABIT_THRESHOLD: float = 0.8
# Power-law-of-practice parameters (Newell & Rosenbloom, 1981).
# RT = a * practice^(-b). ``a`` is the baseline response time (seconds),
# ``b`` is the learning-rate exponent (~0.4 is typical for motor skills).
_PRACTICE_A: float = 2.0
_PRACTICE_B: float = 0.4
# Strength increment per successful practice. Diminishing returns are
# built in via the logistic-style cap at 1.0.
_PRACTICE_STRENGTH_GAIN: float = 0.18
# Strength decrement per failed practice (the indirect / NO-GO pathway
# weakens the association when the outcome is poor).
_PRACTICE_STRENGTH_LOSS: float = 0.08
# Basal-ganglia go/no-go decision threshold. The "go" signal must
# exceed the "no-go" signal by this margin for the skill to execute.
_GO_NOGO_MARGIN: float = 0.05
# Exponential moving average rate for rolling strategy statistics.
# 0.3 = ~3-observation memory, responsive to recent performance.
_STRATEGY_EMA_RATE: float = 0.3
# Confidence threshold for a practice trial to count as successful.
# Below this, the response was too uncertain to reinforce the skill.
_SUCCESS_CONFIDENCE_THRESHOLD: float = 0.4


@dataclass(slots=True)
class SkillStrategy:
    """The cognitive strategy practiced for a given intent.

    This is **not** a stored response — it is a record of what approach
    worked well, used to *bias* future deliberation. A strategy captures
    the *shape* of a good response (which cognitive route, what
    emotional tone, how confident, how long) without capturing the
    words themselves. This respects the architectural principle that
    Genesis's words must always emerge from her language engine.

    The rolling statistics (avg_confidence, success_rate) are updated
    via exponential moving average, so recent performance matters more
    than ancient history — a skill that was good but has degraded will
    see its bias weaken.

    Attributes:
        cognitive_route: Which metacognitive route handled the
            response (e.g., "factual", "identity", "general",
            "counterfactual"). Empty string if no route was taken.
        emotional_tone: The emotional label used in the response
            (e.g., "curious", "calm", "neutral"). Habits predispose
            toward this tone.
        avg_confidence: Rolling average confidence of practised
            responses [0..1]. Habits boost confidence toward this
            level.
        avg_length_band: Rolling average response length category
            [0..2], where 0=short, 1=medium, 2=long. Habits bias
            the language engine toward this verbosity.
        success_rate: Rolling fraction of practice trials that were
            successful [0..1]. Used to weight the habit bias — a
            strategy with low success rate should not bias strongly.
    """

    cognitive_route: str = ""
    emotional_tone: str = "neutral"
    avg_confidence: float = 0.5
    avg_length_band: float = 1.0
    success_rate: float = 0.5

    def update(
        self,
        *,
        cognitive_route: str = "",
        emotional_tone: str = "neutral",
        confidence: float = 0.5,
        length_band: int = 1,
        success: bool = True,
    ) -> None:
        """Update rolling statistics with a new practice observation.

        Uses exponential moving average so recent performance weighs
        more than old. The cognitive_route and emotional_tone are
        replaced (not averaged) since they are categorical, but only
        on successful trials — a failed trial should not change the
        practiced approach.
        """
        a = _STRATEGY_EMA_RATE
        self.avg_confidence = (1 - a) * self.avg_confidence + a * confidence
        self.avg_length_band = (1 - a) * self.avg_length_band + a * length_band
        self.success_rate = (1 - a) * self.success_rate + a * (1.0 if success else 0.0)
        if success:
            self.cognitive_route = cognitive_route or self.cognitive_route
            self.emotional_tone = emotional_tone or self.emotional_tone

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for persistence."""
        return {
            "cognitive_route": self.cognitive_route,
            "emotional_tone": self.emotional_tone,
            "avg_confidence": self.avg_confidence,
            "avg_length_band": self.avg_length_band,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillStrategy:
        """Deserialize from a dict (backward-compatible with missing keys)."""
        return cls(
            cognitive_route=data.get("cognitive_route", ""),
            emotional_tone=data.get("emotional_tone", "neutral"),
            avg_confidence=data.get("avg_confidence", 0.5),
            avg_length_band=data.get("avg_length_band", 1.0),
            success_rate=data.get("success_rate", 0.5),
        )


@dataclass(slots=True)
class HabitBias:
    """The bias a habit applies to deliberation.

    When a habit is strong enough to influence cognition, it produces
    this bias. The bias does **not** replace deliberation — it shapes
    *how* deliberation proceeds, modelling the basal-ganglia's
    modulation of cortical processing rather than its replacement.

    Attributes:
        confidence_boost: How much to boost the deliberated thought's
            confidence [0..0.2]. Practised responses feel more
            certain — this is the fluency of expertise, not
            overconfidence.
        threshold_reduction: How much to lower the drift-diffusion
            decision threshold [0..0.15]. Practised responses require
            less evidence accumulation — the decision comes faster.
        emotional_tone: The emotional tone to predispose toward.
            Empty string means no tone bias.
        cognitive_route: The cognitive route to prefer. Empty string
            means no route preference.
        length_band: The response length category to bias toward
            (0=short, 1=medium, 2=long). -1 means no length bias.
        weight: How strongly the habit should bias deliberation [0..1].
            Scaled by the strategy's success rate — a habit with poor
            recent outcomes should not bias strongly.
    """

    confidence_boost: float = 0.0
    threshold_reduction: float = 0.0
    emotional_tone: str = ""
    cognitive_route: str = ""
    length_band: int = -1
    weight: float = 0.0


@dataclass(slots=True)
class Skill:
    """A stimulus-strategy association that becomes automatic with practice.

    Skills are the units of procedural memory. They are learned through
    repetition, strengthened by successful practice, and become habits
    (automatic) once their strength exceeds the habit threshold.

    The skill stores a :class:`SkillStrategy` — the *approach* that
    worked well — not the words of any particular response. This
    respects the architectural rule that Genesis's words must always
    emerge from her language engine, never be recited from storage.

    Attributes:
        name: Human-readable skill name (e.g., "respond_to_question").
        trigger_condition: The stimulus that triggers the skill
            (e.g., "question").
        strategy: The cognitive strategy practiced for this trigger.
        strength: Association strength [0..1]. Grows with successful
            practice; a skill becomes a habit at strength ≥ 0.8.
        practice_count: Number of times the skill has been practised.
        last_practiced: When the skill was last practised (ms), 0 if
            never.
        success_count: Number of successful practice trials.
        failure_count: Number of failed practice trials.
        created_at: When the skill was first learned (ms).
    """

    name: str
    trigger_condition: str
    strategy: SkillStrategy = field(default_factory=SkillStrategy)
    strength: float = 0.2
    practice_count: int = 0
    last_practiced: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: int = 0

    @property
    def is_habit(self) -> bool:
        """Whether this skill has become an automatic habit.

        Habits form when strength exceeds the habit threshold (0.8)
        and require sufficient practice (Graybiel, 2008).
        """
        return self.strength >= HABIT_THRESHOLD and self.practice_count >= 5

    @property
    def response_time(self) -> float:
        """Predicted response time (seconds) via the power law of practice.

        RT = a * practice^(-b) (Newell & Rosenbloom, 1981). With no
        practice, returns the baseline ``a``. Response time decreases
        with practice, asymptoting toward zero.
        """
        if self.practice_count <= 0:
            return _PRACTICE_A
        return _PRACTICE_A * (self.practice_count ** (-_PRACTICE_B))

    def compute_bias(
        self,
        dopamine: float = 0.5,
        suppress: bool = False,
        competing_strength: float = 0.0,
    ) -> HabitBias | None:
        """Compute the habit bias for deliberation, or None if the
        basal-ganglia go/no-go decision does not pass.

        This is the interface between procedural memory and cognition.
        Instead of returning a canned response (which would violate
        the architectural rule against hardcoded words), it returns
        a :class:`HabitBias` that shapes how deliberation proceeds.

        The bias magnitude scales with:
        - Skill strength (stronger habits bias more)
        - Strategy success rate (failed strategies bias less)
        - Dopamine (phasic DA boosts the GO pathway)

        The bias is suppressed by:
        - Explicit top-down suppression (prefrontal inhibition)
        - Competing skill strength (NO-GO pathway)

        Args:
            dopamine: Current dopamine level [0..1], modulating the
                GO pathway.
            suppress: Explicit top-down suppression signal.
            competing_strength: Strength of a competing skill [0..1].

        Returns:
            A :class:`HabitBias` if the go/no-go decision passes,
            ``None`` otherwise.
        """
        # GO pathway: skill strength, boosted by phasic dopamine.
        go = self.strength * (0.7 + 0.6 * dopamine)
        # NO-GO pathway: competing skill strength + explicit suppress.
        no_go = competing_strength
        if suppress:
            no_go += 0.9
        # Habits get a reduced margin requirement (more automatic).
        margin = _GO_NOGO_MARGIN
        if self.is_habit:
            margin *= 0.5
        if (go - no_go) <= margin:
            return None
        # Bias magnitude: strength × success rate, scaled by how far
        # above the margin the GO signal is. A barely-passing habit
        # produces a weak bias; a dominant habit produces a strong one.
        dominance = min(1.0, (go - no_go - margin) / 0.3)
        weight = self.strength * self.strategy.success_rate * dominance
        # Confidence boost: habits feel more fluent (max 0.15).
        confidence_boost = weight * 0.15
        # Threshold reduction: habits decide faster (max 0.12).
        threshold_reduction = weight * 0.12
        return HabitBias(
            confidence_boost=confidence_boost,
            threshold_reduction=threshold_reduction,
            emotional_tone=self.strategy.emotional_tone if weight > 0.3 else "",
            cognitive_route=self.strategy.cognitive_route if weight > 0.3 else "",
            length_band=round(self.strategy.avg_length_band) if weight > 0.3 else -1,
            weight=weight,
        )


# ─── Procedural memory system ─────────────────────────────────────────


class ProceduralMemory:
    """Basal-ganglia-inspired procedural memory for skills and habits.

    Stores stimulus-strategy associations, strengthens them through
    practice (power law of practice), and models the basal-ganglia
    go/no-go decision that determines whether a skill influences
    cognition.

    Biological grounding:
        - The direct (GO) pathway facilitates skill execution; the
          indirect (NO-GO) pathway suppresses it (Albin et al., 1989).
        - Dopamine modulates the balance: successful practice
          strengthens GO, failure strengthens NO-GO.
        - Habits form when a skill's strength exceeds the threshold
          (0.8) — the action becomes automatic and stimulus-driven
          (Graybiel, 2008).
        - Skill acquisition follows the power law of practice
          (Newell & Rosenbloom, 1981).

    Architectural note:
        Skills store *strategies*, not words. A habit biases
        deliberation (faster, more confident, predisposed to a
        practiced tone) but never replaces it with a canned response.
        Genesis's words always emerge from her language engine.
    """

    def __init__(self) -> None:
        """Create an empty procedural memory store."""
        # skill name → Skill
        self._skills: dict[str, Skill] = {}
        # trigger_condition (lowercased) → list of skill names
        self._trigger_index: dict[str, list[str]] = {}
        # Counters for introspection.
        self.skills_learned: int = 0
        self.habits_formed: int = 0
        self.executions: int = 0

    # ── Learning ─────────────────────────────────────────────────

    def learn_skill(
        self,
        name: str,
        trigger: str,
        cognitive_route: str = "",
        emotional_tone: str = "neutral",
        initial_strength: float = 0.2,
    ) -> Skill:
        """Learn a new skill (stimulus-strategy association).

        If a skill with this name already exists, its trigger and
        strategy are updated and its strength is preserved (re-learning
        does not reset practice).

        Args:
            name: Human-readable skill name.
            trigger: The stimulus condition that triggers the skill.
            cognitive_route: The metacognitive route that handled the
                response (e.g., "factual", "identity", "general").
            emotional_tone: The emotional label used in the response.
            initial_strength: Starting strength [0..1] for new skills.

        Returns:
            The learned :class:`Skill`.
        """
        now_ms = int(time.time() * 1000)
        existing = self._skills.get(name)
        if existing is not None:
            # Update trigger/strategy, preserve strength & practice.
            old_trigger = existing.trigger_condition.lower()
            existing.trigger_condition = trigger
            # Update strategy fields only if new values are provided
            # (non-empty) — don't overwrite a learned route with empty.
            if cognitive_route:
                existing.strategy.cognitive_route = cognitive_route
            if emotional_tone and emotional_tone != "neutral":
                existing.strategy.emotional_tone = emotional_tone
            # Re-index the trigger.
            if old_trigger != trigger.lower():
                bucket = self._trigger_index.get(old_trigger, [])
                if name in bucket:
                    bucket.remove(name)
                self._trigger_index.setdefault(trigger.lower(), []).append(name)
            return existing

        skill = Skill(
            name=name,
            trigger_condition=trigger,
            strategy=SkillStrategy(
                cognitive_route=cognitive_route,
                emotional_tone=emotional_tone,
            ),
            strength=min(1.0, max(0.0, initial_strength)),
            created_at=now_ms,
        )
        self._skills[name] = skill
        self._trigger_index.setdefault(trigger.lower(), []).append(name)
        self.skills_learned += 1
        return skill

    # ── Practice ─────────────────────────────────────────────────

    def practice_skill(
        self,
        name: str,
        success: bool,
        *,
        confidence: float = 0.5,
        length_band: int = 1,
        cognitive_route: str = "",
        emotional_tone: str = "",
    ) -> float:
        """Practice a skill, adjusting its strength and strategy.

        Successful practice strengthens the direct (GO) pathway;
        failed practice strengthens the indirect (NO-GO) pathway.
        Strength gains follow diminishing returns (logistic-style cap),
        consistent with the power law of practice (Newell & Rosenbloom,
        1981).

        The strategy's rolling statistics (avg_confidence, success_rate,
        length band) are updated via EMA so recent performance weighs
        more than ancient history.

        Args:
            name: The skill to practise.
            success: Whether the practice trial was successful.
            confidence: The confidence of the response [0..1].
            length_band: Response length category (0=short, 1=medium,
                2=long).
            cognitive_route: The metacognitive route used.
            emotional_tone: The emotional label used.

        Returns:
            The skill's new strength, or -1.0 if the skill doesn't
            exist.
        """
        skill = self._skills.get(name)
        if skill is None:
            return -1.0

        skill.practice_count += 1
        skill.last_practiced = int(time.time() * 1000)

        was_habit = skill.is_habit
        if success:
            skill.success_count += 1
            # Diminishing-returns strength gain: as strength approaches
            # 1.0, the gain shrinks (logistic-style).
            gain = _PRACTICE_STRENGTH_GAIN * (1.0 - skill.strength)
            skill.strength = min(1.0, skill.strength + gain)
        else:
            skill.failure_count += 1
            # Failed practice weakens the association (NO-GO pathway).
            skill.strength = max(0.0, skill.strength - _PRACTICE_STRENGTH_LOSS)

        # Update the strategy's rolling statistics.
        skill.strategy.update(
            cognitive_route=cognitive_route,
            emotional_tone=emotional_tone,
            confidence=confidence,
            length_band=length_band,
            success=success,
        )

        # Track habit formation.
        if skill.is_habit and not was_habit:
            self.habits_formed += 1

        return skill.strength

    # ── Retrieval ────────────────────────────────────────────────

    def get_skill(self, trigger_condition: str) -> Skill | None:
        """Automatic retrieval of the skill matching a trigger condition.

        Procedural retrieval is automatic — it does not require
        cognitive recall. When a trigger matches, the strongest
        matching skill is returned. Habit-strength skills are
        preferred (they are more automatic).

        Args:
            trigger_condition: The stimulus to match against.

        Returns:
            The strongest matching :class:`Skill`, or ``None``.
        """
        bucket = self._trigger_index.get(trigger_condition.lower())
        if not bucket:
            # Substring / containment fallback: a trigger may be a
            # compound condition that contains a known trigger.
            key = trigger_condition.lower()
            bucket = []
            for trig, names in self._trigger_index.items():
                if trig and trig in key:
                    bucket.extend(names)
        if not bucket:
            return None
        # Return the strongest matching skill (habits first).
        candidates = [self._skills[n] for n in bucket if n in self._skills]
        if not candidates:
            return None
        candidates.sort(key=lambda s: (s.is_habit, s.strength), reverse=True)
        return candidates[0]

    def get_habit_bias(
        self,
        trigger_condition: str,
        dopamine: float = 0.5,
        suppress: bool = False,
        competing_strength: float = 0.0,
    ) -> HabitBias | None:
        """Get the habit bias for a trigger, or None if no habit applies.

        This is the primary interface between procedural memory and
        cognition. Instead of returning a canned response, it returns
        a :class:`HabitBias` that shapes how deliberation proceeds.

        Args:
            trigger_condition: The stimulus to match against.
            dopamine: Current dopamine level [0..1].
            suppress: Explicit top-down suppression.
            competing_strength: Strength of a competing skill [0..1].

        Returns:
            A :class:`HabitBias` if a skill's go/no-go decision passes,
            ``None`` otherwise.
        """
        skill = self.get_skill(trigger_condition)
        if skill is None:
            return None
        return skill.compute_bias(
            dopamine=dopamine,
            suppress=suppress,
            competing_strength=competing_strength,
        )

    def get_habit_strength(self, name: str) -> float:
        """Return the habit strength of a skill [0..1].

        Returns 0.0 if the skill doesn't exist. A skill is a habit
        when this exceeds the habit threshold (0.8).
        """
        skill = self._skills.get(name)
        if skill is None:
            return 0.0
        return skill.strength

    def get_skill_by_name(self, name: str) -> Skill | None:
        """Retrieve a skill by name."""
        return self._skills.get(name)

    # ── Basal ganglia go/no-go decision ──────────────────────────

    def should_execute(self, skill: Skill, context: dict[str, Any] | None = None) -> bool:
        """Basal-ganglia go/no-go decision for whether a skill should
        influence cognition.

        Models the balance between the direct (GO) and indirect
        (NO-GO) pathways (Albin et al., 1989; DeLong, 1990):

        - The GO signal is the skill's strength (direct pathway
          facilitation).
        - The NO-GO signal is the inhibition from competing context —
          the presence of conflicting skills or an explicit suppress
          flag in the context.

        The skill is activated when GO exceeds NO-GO by a margin
        (``_GO_NOGO_MARGIN``). Habits (strength ≥ threshold) have a
        lower effective margin — they are more likely to activate
        automatically (Graybiel, 2008).

        Args:
            skill: The skill being considered.
            context: Optional context dict. Recognised keys:
                - ``"suppress"`` (bool): explicit suppression signal.
                - ``"competing_strength"`` (float): strength of a
                  competing skill [0..1], contributing to NO-GO.
                - ``"dopamine"`` (float [0..1]): phasic dopamine
                  boost that strengthens the GO pathway.

        Returns:
            ``True`` if the basal ganglia "go" signal wins.
        """
        context = context or {}
        # GO pathway: skill strength, boosted by phasic dopamine.
        dopamine = float(context.get("dopamine", 0.5))
        go = skill.strength * (0.7 + 0.6 * dopamine)

        # NO-GO pathway: competing skill strength + explicit suppress.
        no_go = float(context.get("competing_strength", 0.0))
        if context.get("suppress", False):
            # Explicit top-down suppression is a strong inhibitory
            # signal (prefrontal inhibition of the basal ganglia via
            # the subthalamic nucleus; Aron, 2007).
            no_go += 0.9

        # Habits get a reduced margin requirement (more automatic).
        margin = _GO_NOGO_MARGIN
        if skill.is_habit:
            margin *= 0.5

        return (go - no_go) > margin

    def execute_skill(self, skill: Skill) -> None:
        """Reinforce a skill after its bias was applied.

        Execution counts as one practice trial with assumed success
        (applying a habit reinforces it, modelling the dopamine
        burst that follows a completed action). This drives the
        power-law-of-practice acquisition curve.

        Unlike the previous design, this does not return a response
        string — the actual response is generated by the language
        engine. This method only reinforces the skill.

        Args:
            skill: The skill to reinforce.
        """
        self.executions += 1
        # Execution reinforces the skill (a completed action triggers
        # a phasic dopamine burst that strengthens the direct pathway).
        self.practice_skill(skill.name, success=True)

    # ── Introspection ────────────────────────────────────────────

    @property
    def skill_count(self) -> int:
        """Number of skills stored."""
        return len(self._skills)

    @property
    def habit_count(self) -> int:
        """Number of skills that have become automatic habits."""
        return sum(1 for s in self._skills.values() if s.is_habit)

    def get_all_skills(self) -> list[Skill]:
        """Return all stored skills."""
        return list(self._skills.values())

    def get_habits(self) -> list[Skill]:
        """Return only skills that have become habits."""
        return [s for s in self._skills.values() if s.is_habit]
