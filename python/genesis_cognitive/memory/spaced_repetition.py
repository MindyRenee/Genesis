"""Spaced repetition scheduler — when to review what.

Spaced repetition exploits the *spacing effect* (Ebbinghaus, 1885):
information is retained better when reviews are spread out over
expanding intervals rather than massed together. The brain's memory
trace decays following a forgetting curve, and each successful
review *stabilizes* the memory, lengthening the time until the next
review is needed.

# The forgetting curve

Ebbinghaus's forgetting curve models memory retention as exponential
decay:

    R(t) = exp(−t / S)

where R is the retention probability, t is the time since last
review, and S is the *memory stability* (how slowly the trace
decays). Each successful review increases S, pushing the curve
flatter and the next review further out. A failed review resets S
to a low value — the memory needs to be rebuilt.

# Expanding intervals

After each successful review, the next interval grows:

    S_{n+1} = S_n × ease_factor

where the ease factor reflects how easy the concept was to recall
(success → larger intervals, struggle → smaller). This produces the
expanding review schedule used by systems like Anki and SuperMemo
(Wozniak, 1995).

# Priority

Not all concepts are equally worth reviewing. Priority combines:

- **Urgency**: how soon until the concept is forgotten
  (1 − R(t)). A concept about to be forgotten is urgent.
- **Importance**: how central the concept is in the network
  (degree centrality — hub concepts matter more).

    priority = urgency × importance

This integrates with the concept network to identify which concepts
need review and how important they are.

References:
    - Ebbinghaus (1885): Memory: A Contribution to Experimental
      Psychology.
    - Wozniak (1995): Optimization of learning: SuperMemo.
    - Cepeda et al. (2006): Distributed practice in verbal recall.
      Psychological Bulletin, 132(3), 354-380.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..concepts import ConceptNetwork

__all__ = ["ReviewRecord", "SpacedRepetitionScheduler"]


@dataclass(slots=True)
class ReviewRecord:
    """Per-concept spaced repetition state.

    Attributes:
        concept: The concept being tracked.
        stability: Memory stability S (seconds). Higher = decays
            slower. Starts at a default and grows with successful
            reviews.
        review_count: Total number of reviews.
        success_count: Number of successful (recalled) reviews.
        last_review: Timestamp (seconds) of the last review, or 0
            if never reviewed.
        ease_factor: Multiplier for the next interval (≥1.3 per
            SuperMemo convention).
    """

    concept: str
    stability: float
    review_count: int = 0
    success_count: int = 0
    last_review: float = 0.0
    ease_factor: float = 2.5


class SpacedRepetitionScheduler:
    """Schedules concept reviews using spaced repetition.

    Tracks last access time and memory stability for each concept,
    computes retention via the Ebbinghaus forgetting curve, and
    prioritizes reviews by urgency × importance.

    Args:
        network: The concept network (used for importance via degree
            centrality).
        default_stability: Initial memory stability for a new concept
            (seconds). Default 1 day (86400 s).
        min_stability: Minimum stability after a failed review.
        min_ease: Minimum ease factor (prevents intervals from
            shrinking too fast).
        retention_threshold: The retention level below which a
            concept is considered due for review (default 0.7).
    """

    DEFAULT_STABILITY: float = 86400.0  # 1 day
    MIN_STABILITY: float = 300.0  # 5 minutes
    MIN_EASE: float = 1.3  # SuperMemo minimum ease factor
    RETENTION_THRESHOLD: float = 0.7  # review when retention drops below this
    EASE_BUMP: float = 0.1  # ease increase on success
    EASE_PENALTY: float = 0.2  # ease decrease on failure

    def __init__(
        self,
        network: ConceptNetwork,
        *,
        default_stability: float = DEFAULT_STABILITY,
        min_stability: float = MIN_STABILITY,
        min_ease: float = MIN_EASE,
        retention_threshold: float = RETENTION_THRESHOLD,
    ) -> None:
        """Initialize the spaced repetition scheduler."""
        self.network = network
        self.default_stability = default_stability
        self.min_stability = min_stability
        self.min_ease = min_ease
        # Clamp retention_threshold to (0, 1) — math.log(threshold) in
        # next_interval raises ValueError on ≤ 0 and produces nonsensical
        # negative intervals on > 1.
        self.retention_threshold = min(max(retention_threshold, 1e-6), 0.999999)

        # Per-concept review records
        self._records: dict[str, ReviewRecord] = {}

    # ─── Public API ─────────────────────────────────────────────

    def record_review(self, concept: str, success: bool) -> None:
        """Record a review outcome for a concept.

        On success: stability grows by the ease factor (expanding
        interval), and the ease factor increases slightly. On
        failure: stability resets to the minimum, and the ease
        factor decreases.

        Args:
            concept: The concept reviewed.
            success: Whether the review was successful (recalled).
        """
        cid = self.network._resolve(concept) or concept
        record = self._records.get(cid)
        now = time.time()

        if record is None:
            record = ReviewRecord(
                concept=cid,
                stability=self.default_stability,
            )
            self._records[cid] = record

        record.review_count += 1
        record.last_review = now

        if success:
            record.success_count += 1
            # Expand the interval: stability grows by ease factor.
            record.stability *= record.ease_factor
            # Reward: increase ease factor (cap at a reasonable max).
            record.ease_factor = min(3.5, record.ease_factor + self.EASE_BUMP)
        else:
            # Reset stability — the memory trace is weak.
            record.stability = self.min_stability
            # Penalize ease factor (floor at minimum).
            record.ease_factor = max(self.min_ease, record.ease_factor - self.EASE_PENALTY)

    def retention(self, concept: str, now: float | None = None) -> float:
        """Compute current retention for a concept.

        Uses the Ebbinghaus forgetting curve: R(t) = exp(−t / S),
        where t is the time since last review and S is the memory
        stability. A concept never reviewed has retention 0.0
        (completely unknown).

        Args:
            concept: The concept to query.
            now: Optional current timestamp (seconds). Defaults to
                time.time().

        Returns:
            Retention probability in [0, 1].
        """
        cid = self.network._resolve(concept) or concept
        record = self._records.get(cid)
        if record is None or record.last_review == 0.0:
            return 0.0
        t = (now or time.time()) - record.last_review
        if record.stability <= 0:
            return 0.0
        return math.exp(-t / record.stability)

    def urgency(self, concept: str, now: float | None = None) -> float:
        """How urgent is reviewing this concept?

        Urgency = 1 − retention. A concept about to be forgotten
        (low retention) is urgent; a freshly reviewed concept is not.

        Args:
            concept: The concept to query.
            now: Optional current timestamp (seconds).

        Returns:
            Urgency in [0, 1].
        """
        return 1.0 - self.retention(concept, now)

    def importance(self, concept: str) -> float:
        """How important is a concept (network centrality).

        Uses degree centrality: the fraction of the network that
        this concept connects to. Hub concepts (many edges) are more
        important to keep fresh because forgetting them disrupts more
        of the knowledge graph.

        Args:
            concept: The concept to query.

        Returns:
            Importance in [0, 1].
        """
        cid = self.network._resolve(concept)
        if not cid:
            return 0.0
        n = self.network.size
        if n <= 1:
            return 1.0
        degree = len(self.network.get_edges(cid, direction="both"))
        # Normalize by max possible degree (n-1)
        return min(1.0, degree / (n - 1)) if n > 1 else 1.0

    def priority(self, concept: str, now: float | None = None) -> float:
        """Review priority = urgency × importance.

        Args:
            concept: The concept to query.
            now: Optional current timestamp (seconds).

        Returns:
            Priority score in [0, 1].
        """
        return self.urgency(concept, now) * self.importance(concept)

    def get_review_schedule(
        self, limit: int = 20, now: float | None = None
    ) -> list[tuple[str, float]]:
        """Get the concepts most due for review, sorted by priority.

        Only concepts whose retention has dropped below the
        retention threshold are included (they are "due").

        Args:
            limit: Maximum number of concepts to return.
            now: Optional current timestamp (seconds).

        Returns:
            A list of (concept, urgency) tuples, sorted by priority
            (urgency × importance) descending.
        """
        current = now or time.time()
        due: list[tuple[str, float, float]] = []

        for cid, record in self._records.items():
            t = current - record.last_review if record.last_review else float("inf")
            if record.stability <= 0:
                ret = 0.0
            else:
                ret = math.exp(-t / record.stability) if t < float("inf") else 0.0
            if ret < self.retention_threshold:
                urg = 1.0 - ret
                imp = self.importance(cid)
                due.append((cid, urg, urg * imp))

        due.sort(key=lambda x: -x[2])
        return [(cid, urg) for cid, urg, _ in due[:limit]]

    def get_next_review(self, now: float | None = None) -> str | None:
        """Get the single most urgent concept to review next.

        Args:
            now: Optional current timestamp (seconds).

        Returns:
            The concept ID most due for review, or None if nothing
            is due.
        """
        schedule = self.get_review_schedule(limit=1, now=now)
        return schedule[0][0] if schedule else None

    def next_interval(self, concept: str) -> float:
        """The scheduled time until the next review (seconds).

        Based on current stability and the retention threshold:
        the interval at which retention drops to the threshold.

            t = −S × ln(threshold)

        Args:
            concept: The concept to query.

        Returns:
            Seconds until the next review is due (from now). Returns
            0 if already due.
        """
        cid = self.network._resolve(concept) or concept
        record = self._records.get(cid)
        if record is None:
            return 0.0
        # Time from last review until retention hits the threshold
        threshold_time = -record.stability * math.log(self.retention_threshold)
        elapsed = time.time() - record.last_review if record.last_review else float("inf")
        remaining = threshold_time - elapsed
        return max(0.0, remaining)

    def get_statistics(self) -> dict[str, int | float]:
        """Return statistics about the scheduler."""
        total = len(self._records)
        if total == 0:
            return {
                "tracked_concepts": 0,
                "total_reviews": 0,
                "avg_stability": 0.0,
                "avg_ease": 0.0,
            }
        return {
            "tracked_concepts": total,
            "total_reviews": sum(r.review_count for r in self._records.values()),
            "avg_stability": sum(r.stability for r in self._records.values()) / total,
            "avg_ease": sum(r.ease_factor for r in self._records.values()) / total,
        }
