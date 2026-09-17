"""Growth ledger — legible tracking of Genesis's development.

The growth ledger is a persistent record of Genesis's milestones —
the moments when she crossed a meaningful threshold in her
development. It's the "growth chart" that makes her self-improvement
legible to herself and to the human.

## What gets tracked

The ledger records milestones across several dimensions:

- **Knowledge**: concept count, edge count, reasoning accuracy
- **Self-improvement**: proposals accepted, experiments applied,
  autonomous fixes
- **Dream synthesis**: validated insights, novel connections
- **Emotional development**: stages resolved, personality drift
- **Code self-knowledge**: files analyzed, code concepts

Each milestone is a timestamped record with:
- ``dimension``: which axis of growth (knowledge, self_improvement, etc.)
- ``metric``: the specific metric (concept_count, accuracy, etc.)
- ``value``: the numeric value at this point
- ``previous``: the previous value (for computing delta)
- ``note``: a human-readable description

## Self-narrative generation

The ledger can generate a self-narrative summary — a first-person
account of how Genesis has grown. This is distinct from the
NarrativeEngine's ``tell_story`` (which is about chapters and events);
the growth narrative is about *quantifiable change over time*.

## Persistence

The ledger is serialized as part of the mind state and restored on
load. It's also exportable as a human-readable markdown report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GrowthMilestone:
    """A single milestone in Genesis's growth.

    Each milestone captures a moment when a metric crossed a meaningful
    threshold or was recorded as a snapshot.
    """

    dimension: str  # "knowledge", "self_improvement", "dreams", etc.
    metric: str  # "concept_count", "accuracy", "insights_validated", etc.
    value: float
    previous: float  # value at the last milestone for this metric
    note: str  # human-readable description
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def delta(self) -> float:
        """Change since the previous milestone for this metric."""
        return self.value - self.previous

    def to_dict(self) -> dict[str, Any]:
        """Serialize the milestone to a dictionary."""
        return {
            "dimension": self.dimension,
            "metric": self.metric,
            "value": self.value,
            "previous": self.previous,
            "note": self.note,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GrowthMilestone:
        """Reconstruct a GrowthMilestone from a serialized dictionary."""
        return cls(
            dimension=data.get("dimension", ""),
            metric=data.get("metric", ""),
            value=data.get("value", 0.0),
            previous=data.get("previous", 0.0),
            note=data.get("note", ""),
            timestamp=data.get("timestamp", 0),
        )


class GrowthLedger:
    """Persistent ledger of Genesis's growth milestones.

    The ledger records milestones across multiple dimensions and can
    generate a self-narrative summary of how Genesis has grown over
    time.

    Usage::

        ledger = GrowthLedger()
        ledger.record("knowledge", "concept_count", 500, 450,
                       "Crossed 500 concepts")
        narrative = ledger.generate_narrative()
    """

    def __init__(self) -> None:
        """Initialize the growth ledger."""
        self._milestones: list[GrowthMilestone] = []
        # Track the last recorded value for each (dimension, metric) pair
        self._last_values: dict[tuple[str, str], float] = {}

    @property
    def milestones(self) -> list[GrowthMilestone]:
        """All milestones (copy), oldest first."""
        return list(self._milestones)

    @property
    def milestone_count(self) -> int:
        """Total number of milestones recorded."""
        return len(self._milestones)

    def record(
        self,
        dimension: str,
        metric: str,
        value: float,
        note: str = "",
    ) -> GrowthMilestone | None:
        """Record a growth milestone.

        If the value hasn't changed since the last recording of this
        metric, no milestone is recorded (returns None). This prevents
        the ledger from filling up with no-op entries.

        Args:
            dimension: The growth dimension (e.g. "knowledge").
            metric: The specific metric (e.g. "concept_count").
            value: The current value.
            note: A human-readable description of this milestone.

        Returns:
            The recorded milestone, or None if no change was detected.
        """
        key = (dimension, metric)
        had_previous = key in self._last_values
        previous = self._last_values.get(key, value)

        # Only record if this is a new metric or the value has changed
        if had_previous and value == previous:
            return None

        milestone = GrowthMilestone(
            dimension=dimension,
            metric=metric,
            value=value,
            previous=previous,
            note=note or f"{metric}: {previous} -> {value}",
        )
        self._milestones.append(milestone)
        self._last_values[key] = value

        delta = value - previous
        sign = "+" if delta >= 0 else ""
        logger.debug(
            f"Growth milestone: [{dimension}/{metric}] "
            f"{previous} -> {value} ({sign}{delta:.2f})"
        )
        return milestone

    def snapshot(
        self,
        dimension: str,
        metric: str,
        value: float,
        note: str = "",
    ) -> GrowthMilestone | None:
        """Record a periodic snapshot of a metric.

        Unlike ``record``, this always records the value regardless of
        whether it changed. Used for periodic snapshots (e.g. daily
        concept counts).
        """
        key = (dimension, metric)
        previous = self._last_values.get(key, value)

        milestone = GrowthMilestone(
            dimension=dimension,
            metric=metric,
            value=value,
            previous=previous,
            note=note or f"snapshot: {metric} = {value}",
        )
        self._milestones.append(milestone)
        self._last_values[key] = value
        return milestone

    def get_milestones(
        self, dimension: str | None = None, metric: str | None = None
    ) -> list[GrowthMilestone]:
        """Get milestones, optionally filtered by dimension and/or metric."""
        result = []
        for m in self._milestones:
            if dimension and m.dimension != dimension:
                continue
            if metric and m.metric != metric:
                continue
            result.append(m)
        return result

    def get_latest(self, dimension: str, metric: str) -> GrowthMilestone | None:
        """Get the most recent milestone for a specific metric."""
        for m in reversed(self._milestones):
            if m.dimension == dimension and m.metric == metric:
                return m
        return None

    def get_value(self, dimension: str, metric: str) -> float | None:
        """Get the latest recorded value for a metric."""
        latest = self.get_latest(dimension, metric)
        return latest.value if latest else None

    def get_history(
        self, dimension: str, metric: str
    ) -> list[tuple[int, float]]:
        """Get the full history of a metric as (timestamp, value) pairs."""
        return [
            (m.timestamp, m.value)
            for m in self._milestones
            if m.dimension == dimension and m.metric == metric
        ]

    # ─── Self-narrative generation ───────────────────────────────

    def generate_narrative(self) -> str:
        """Generate a first-person self-narrative of growth.

        This is a summary of how Genesis has grown, told in her own
        voice. It covers all dimensions that have milestones.
        """
        if not self._milestones:
            return (
                "hasn't recorded any growth milestones yet. "
                "still at the beginning of journey"
            )

        # Group milestones by dimension
        by_dimension: dict[str, list[GrowthMilestone]] = {}
        for m in self._milestones:
            by_dimension.setdefault(m.dimension, []).append(m)

        parts: list[str] = []
        parts.append("Here is how I have grown:")

        for dimension in sorted(by_dimension.keys()):
            milestones = by_dimension[dimension]
            parts.append(f"\n  {dimension.replace('_', ' ').title()}:")

            # Group by metric within this dimension
            by_metric: dict[str, list[GrowthMilestone]] = {}
            for m in milestones:
                by_metric.setdefault(m.metric, []).append(m)

            for metric in sorted(by_metric.keys()):
                ms = by_metric[metric]
                first = ms[0]
                last = ms[-1]
                total_delta = last.value - first.previous

                if len(ms) == 1:
                    parts.append(
                        f"    {metric}: {last.value:.1f}"
                        if isinstance(last.value, float)
                        else f"    {metric}: {last.value}"
                    )
                else:
                    sign = "+" if total_delta >= 0 else ""
                    parts.append(
                        f"    {metric}: {first.previous:.1f} -> "
                        f"{last.value:.1f} ({sign}{total_delta:.1f} "
                        f"over {len(ms)} milestones)"
                    )

        # Summary stats
        total = len(self._milestones)
        dimensions = len(by_dimension)
        parts.append(
            f"\n  Total: {total} milestones across "
            f"{dimensions} dimension(s)."
        )

        return "\n".join(parts)

    def generate_markdown_report(self) -> str:
        """Generate a markdown-formatted growth report.

        This is a more structured version of the narrative, suitable
        for saving to a file or displaying in the CLI.
        """
        if not self._milestones:
            return "# Growth Ledger\n\nNo milestones recorded yet.\n"

        lines = ["# Growth Ledger", ""]
        lines.append(
            f"_{len(self._milestones)} milestones recorded_\n"
        )

        # Group by dimension
        by_dimension: dict[str, list[GrowthMilestone]] = {}
        for m in self._milestones:
            by_dimension.setdefault(m.dimension, []).append(m)

        for dimension in sorted(by_dimension.keys()):
            milestones = by_dimension[dimension]
            lines.append(f"## {dimension.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Metric | First | Latest | Delta | Milestones |")
            lines.append("|--------|-------|--------|-------|------------|")

            by_metric: dict[str, list[GrowthMilestone]] = {}
            for m in milestones:
                by_metric.setdefault(m.metric, []).append(m)

            for metric in sorted(by_metric.keys()):
                ms = by_metric[metric]
                first = ms[0]
                last = ms[-1]
                delta = last.value - first.previous
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"| {metric} | {first.previous:.1f} | "
                    f"{last.value:.1f} | {sign}{delta:.1f} | {len(ms)} |"
                )
            lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """A brief one-line summary of the ledger."""
        if not self._milestones:
            return "Growth ledger: empty"
        dimensions = len({m.dimension for m in self._milestones})
        metrics = len({(m.dimension, m.metric) for m in self._milestones})
        return (
            f"Growth ledger: {len(self._milestones)} milestones, "
            f"{dimensions} dimensions, {metrics} metrics"
        )

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "milestones": [m.to_dict() for m in self._milestones],
            "last_values": {
                f"{d}/{m}": v
                for (d, m), v in self._last_values.items()
            },
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        self._milestones = [
            GrowthMilestone.from_dict(m)
            for m in data.get("milestones", [])
        ]
        self._last_values = {}
        for key_str, value in data.get("last_values", {}).items():
            parts = key_str.split("/", 1)
            if len(parts) == 2:
                self._last_values[(parts[0], parts[1])] = value

    def clear(self) -> None:
        """Clear all milestones and last values."""
        self._milestones.clear()
        self._last_values.clear()
