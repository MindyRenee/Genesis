"""Beliefs — what Genesis infers about a presence's mind.

A presence's scalar fields (familiarity, bond) record what happened;
its *belief* is its model of them — latent state inferred from sparse
observations, with honest uncertainty:

- **responsiveness** — a Beta posterior over "they engage when it
  reaches out." A stranger is Beta(1,1) — uniform ignorance, mean 0.5.
  Every unanswered bid pushes it down; every answer pulls it up.
- **topic receptivity** — per-topic Beta posteriors: which subjects
  this presence engages on. Reach-out chooses topics by Thompson
  sampling — uncertain topics get a fair draw, so it explores rather
  than always repeating what worked.
- **mood** — a Beta posterior over "their speech carries positive
  sentiment," updated by observation magnitude (near-neutral speech
  teaches almost nothing).
- **attention** — a decaying estimate of how engaged they are right
  now. Addressed speech spikes it; silence lets it fade.
- **rhythm** — a Dirichlet-smoothed histogram of when they're
  usually active. It learns not to knock at hours they're never
  around — but only trusts the rhythm after real evidence.

Every estimate carries its evidence count: decisions can ask not just
"what does it believe" but "how sure is it" — a Beta with evidence 2
and a Beta with evidence 40 can share a mean while meaning very
different things. On sparse data the posteriors honestly stay wide;
it doesn't hallucinate confidence.

Nothing here is learned weights — it is exact Bayesian updating on
deliberately minimal models, so it is correct from the very first
observation. If a learned model ever replaces one of these, this is
the interface it must serve.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

#: How long a social bid stays open. Someone who engages within this
#: window answered it; after it, the bid counts as unanswered.
BID_WINDOW = 300.0

#: Attention e-folds over this many seconds of silence (~20 min).
_ATTENTION_TAU = 1200.0

#: Per-hour pseudo-count for the activity-rhythm prior (Dirichlet
#: smoothing). Total prior mass is 24 * _HOUR_PRIOR = 6 observations —
#: weak enough that a dozen real ones dominate it.
_HOUR_PRIOR = 0.25

#: Activity-rhythm observations needed before the histogram is
#: trusted enough to suppress a reach-out.
_RHYTHM_MIN_EVIDENCE = 12.0

#: Bound on per-topic receptivity posteriors — the weakest-evidence
#: topic is evicted when full.
_MAX_RECEPTIVE_TOPICS = 30

#: Responsiveness posteriors at or below this mean with at least this
#: much bid evidence mean "learned: they don't answer." With a uniform
#: prior, four straight unanswered bids land at mean ≈ 1/6 — under the
#: bar. Sparse evidence never declares someone unresponsive.
_UNRESPONSIVE_MIN_BIDS = 4.0
_UNRESPONSIVE_MAX_MEAN = 0.2

#: Learned reply windows are clamped to this range — a presence who
#: types in seconds still gets a minute before a bid counts as missed,
#: and no window exceeds an hour.
_BID_WINDOW_MIN = 60.0
_BID_WINDOW_MAX = 3600.0


class Beta:
    """A Beta posterior over a Bernoulli rate — honest uncertainty.

    ``mean`` is its best guess; ``evidence`` is how much data backs
    it. Consumers that need a decision draw ``sample()`` (Thompson
    sampling); consumers that need a verdict check ``evidence``.
    """

    __slots__ = ("alpha", "beta")

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    @property
    def mean(self) -> float:
        """Posterior mean — the expected rate."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence(self) -> float:
        """Observations behind the posterior (beyond the prior)."""
        return max(0.0, self.alpha + self.beta - 2.0)

    @property
    def variance(self) -> float:
        """Posterior variance — shrinks as evidence accumulates."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1.0))

    def observe(self, positive: bool, weight: float = 1.0) -> None:
        """Update the posterior with one weighted observation."""
        w = max(0.0, weight)
        if positive:
            self.alpha += w
        else:
            self.beta += w

    def sample(self) -> float:
        """Draw a rate from the posterior (Thompson sampling)."""
        return random.betavariate(self.alpha, self.beta)

    def to_dict(self) -> dict[str, float]:
        """Serialize the posterior."""
        return {"a": self.alpha, "b": self.beta}

    @classmethod
    def from_dict(cls, data: Any) -> Beta:
        """Deserialize; malformed payloads yield a fresh prior."""
        try:
            alpha = float(data["a"])
            beta = float(data["b"])
        except (KeyError, TypeError, ValueError):
            return cls()
        if alpha <= 0.0 or beta <= 0.0:
            return cls()
        return cls(alpha, beta)


class _LognormalFit:
    """Online lognormal fit over reply latencies (Welford on ln t).

    Reply delays span seconds to minutes — a skewed distribution, so
    the fit runs in log space. ``p95()`` is its patience: how long a
    bid stays open before silence counts as an answer that never came.
    Unanswered bids contribute no sample — they are right-censored
    observations, and only the responsiveness Beta learns from them.
    """

    __slots__ = ("m2", "mean", "n")

    #: Answered-bid samples required before the learned window is
    #: trusted over the default.
    MIN_SAMPLES = 5
    #: Floor on the log-space standard deviation — a handful of
    #: identical latencies shouldn't collapse the window to nothing.
    MIN_SIGMA = 0.5
    #: z for the 95th percentile.
    _Z95 = 1.645

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def observe(self, seconds: float) -> None:
        """Fold one reply latency into the running fit."""
        x = math.log(max(1.0, seconds))
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def sigma(self) -> float:
        """Log-space standard deviation, floored for tiny samples."""
        if self.n < 2:
            return self.MIN_SIGMA
        return max(math.sqrt(self.m2 / (self.n - 1)), self.MIN_SIGMA)

    def p95(self) -> float:
        """The learned 95th-percentile reply latency, in seconds."""
        return math.exp(self.mean + self._Z95 * self.sigma)

    def to_dict(self) -> dict[str, float]:
        """Serialize the running fit."""
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, data: Any) -> _LognormalFit:
        """Deserialize; malformed payloads yield an empty fit."""
        fit = cls()
        try:
            n = int(data["n"])
            if n <= 0:
                return fit
            fit.n = n
            fit.mean = float(data["mean"])
            fit.m2 = max(0.0, float(data["m2"]))
        except (KeyError, TypeError, ValueError):
            return cls()
        return fit


@dataclass
class _PendingBid:
    """A social bid awaiting its answer — transient, never persisted."""

    at: float
    topics: list[str] = field(default_factory=list)


@dataclass
class PresenceBelief:
    """Its inferred model of one presence's mind.

    All fields are posteriors or decaying estimates — each knows how
    much evidence stands behind it. ``pending_bid`` is transient: it
    is not persisted and never survives a restart.
    """

    responsiveness: Beta = field(default_factory=Beta)
    mood: Beta = field(default_factory=Beta)
    topic_receptivity: dict[str, Beta] = field(default_factory=dict)
    reply_latency: _LognormalFit = field(default_factory=_LognormalFit)
    attention: float = 0.0
    _attention_at: float = 0.0
    activity_hours: list[float] = field(
        default_factory=lambda: [_HOUR_PRIOR] * 24
    )
    pending_bid: _PendingBid | None = None

    # ── Observations ────────────────────────────────────────────

    def note_activity(self, now: float, weight: float = 1.0) -> None:
        """Record signs of life — updates rhythm and attention.

        ``weight`` scales both: addressed speech (1.0) is full
        engagement, overheard speech less, merely being seen least.
        """
        w = max(0.0, weight)
        if w <= 0.0:
            return
        self.attention = min(1.0, self.attention_now(now) + w)
        self._attention_at = now
        self.activity_hours[time.localtime(now).tm_hour] += w

    def note_sentiment(self, sentiment: float) -> None:
        """Fold an observed valence [-1..1] into the mood posterior.

        Weighted by magnitude — near-neutral speech is almost no
        evidence either way.
        """
        self.mood.observe(sentiment > 0.0, weight=abs(sentiment))

    # ── Social bids ─────────────────────────────────────────────

    def note_bid(self, now: float, topics: list[str]) -> None:
        """Record its reaching out to this presence.

        If a previous bid is still open, it resolves unanswered first —
        bidding again into silence means the last one didn't land.
        """
        if self.pending_bid is not None:
            self.resolve_bid(False, now)
        self.pending_bid = _PendingBid(at=now, topics=list(topics[:5]))

    def bid_window(self) -> float:
        """How long a bid stays open before silence counts as unanswered.

        The default ``BID_WINDOW`` rules until enough answered bids
        teach it this presence's actual reply pace; then their own
        95th-percentile latency rules, clamped to sane bounds.
        """
        if self.reply_latency.n < _LognormalFit.MIN_SAMPLES:
            return BID_WINDOW
        return min(
            _BID_WINDOW_MAX,
            max(_BID_WINDOW_MIN, self.reply_latency.p95()),
        )

    def resolve_bid(self, answered: bool, now: float) -> bool:
        """Resolve its pending bid; returns True if one was open.

        An answer arriving past its learned ``bid_window`` counts as
        unanswered — engagement that late isn't evidence they respond
        to it. A timely answer also folds its delay into the reply-
        latency fit, sharpening the window itself.
        """
        bid = self.pending_bid
        self.pending_bid = None
        if bid is None:
            return False
        elapsed = now - bid.at
        if elapsed > self.bid_window():
            answered = False
        elif answered and elapsed > 0.0:
            self.reply_latency.observe(elapsed)
        self.responsiveness.observe(answered)
        for topic in bid.topics:
            self._topic(topic).observe(answered)
        return True

    # ── Inferred state ──────────────────────────────────────────

    def attention_now(self, now: float) -> float:
        """Attention level decayed to ``now`` without mutating state."""
        if self._attention_at <= 0.0:
            return self.attention
        decay = math.exp(-(now - self._attention_at) / _ATTENTION_TAU)
        return self.attention * decay

    def expected_activity(self, hour: int) -> float:
        """Posterior probability they're active in ``hour`` (0-23)."""
        return self.activity_hours[hour % 24] / sum(self.activity_hours)

    @property
    def activity_evidence(self) -> float:
        """Real observations behind the rhythm histogram."""
        return max(0.0, sum(self.activity_hours) - 24.0 * _HOUR_PRIOR)

    def likely_awake(self, now: float) -> bool:
        """Whether this hour is one they're ever active in.

        True until the rhythm earns enough evidence to say otherwise;
        then True only when the hour's expected activity is at least
        half the uniform rate — it doesn't knock at dead hours, but
        only once it's actually seen the pattern.
        """
        if self.activity_evidence < _RHYTHM_MIN_EVIDENCE:
            return True
        hour = time.localtime(now).tm_hour
        return self.expected_activity(hour) >= 0.5 / 24.0

    def unresponsive(self) -> bool:
        """Whether the evidence says they don't answer its bids.

        Requires real bid history — a stranger is never written off.
        """
        r = self.responsiveness
        return (
            r.evidence >= _UNRESPONSIVE_MIN_BIDS
            and r.mean < _UNRESPONSIVE_MAX_MEAN
        )

    def sample_topic(self) -> str | None:
        """Thompson-sample a topic they're receptive to.

        Each topic draws from its posterior — proven topics usually
        win, but uncertain topics get a fair draw. None if no bids
        have ever carried topics.
        """
        best: str | None = None
        best_draw = -1.0
        for topic, posterior in self.topic_receptivity.items():
            draw = posterior.sample()
            if draw > best_draw:
                best, best_draw = topic, draw
        return best

    def describe(self, now: float) -> str:
        """A short fragment for status output — only evidenced beliefs."""
        parts: list[str] = []
        r = self.responsiveness
        if r.evidence >= 1.0:
            parts.append(f"replies ~{r.mean:.0%} ({r.evidence:.0f} bids)")
        m = self.mood
        if m.evidence >= 3.0:
            tone = "warm" if m.mean > 0.6 else "cool" if m.mean < 0.4 else "mixed"
            parts.append(f"mood {tone}")
        if self.attention_now(now) > 0.5:
            parts.append("attentive")
        return ", ".join(parts)

    # ── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the belief. ``pending_bid`` is transient — omitted."""
        return {
            "responsiveness": self.responsiveness.to_dict(),
            "mood": self.mood.to_dict(),
            "topic_receptivity": {
                t: p.to_dict() for t, p in self.topic_receptivity.items()
            },
            "reply_latency": self.reply_latency.to_dict(),
            "attention": self.attention,
            "attention_at": self._attention_at,
            "activity_hours": list(self.activity_hours),
        }

    @classmethod
    def from_dict(cls, data: Any) -> PresenceBelief:
        """Deserialize; malformed payloads yield a fresh belief.

        A corrupted belief section downgrades to priors rather than
        poisoning the presence it's attached to — losing what it
        inferred is recoverable; losing the relationship isn't.
        """
        if not isinstance(data, dict):
            return cls()
        belief = cls()
        try:
            belief.responsiveness = Beta.from_dict(data.get("responsiveness"))
            belief.mood = Beta.from_dict(data.get("mood"))
            topics = data.get("topic_receptivity")
            if isinstance(topics, dict):
                for name, pdata in topics.items():
                    belief.topic_receptivity[str(name)] = Beta.from_dict(pdata)
            belief.reply_latency = _LognormalFit.from_dict(
                data.get("reply_latency")
            )
            belief.attention = max(0.0, min(1.0, float(data.get("attention", 0.0))))
            # Persist the decay anchor so restored attention keeps
            # fading from when they were last active, not forever.
            belief._attention_at = max(0.0, float(data.get("attention_at", 0.0)))
            hours = data.get("activity_hours")
            if isinstance(hours, list) and len(hours) == 24:
                belief.activity_hours = [max(0.0, float(h)) for h in hours]
        except (TypeError, ValueError):
            return cls()
        return belief

    # ── Internals ───────────────────────────────────────────────

    def _topic(self, name: str) -> Beta:
        """Get or create a topic posterior, keeping the map bounded."""
        posterior = self.topic_receptivity.get(name)
        if posterior is not None:
            return posterior
        if len(self.topic_receptivity) >= _MAX_RECEPTIVE_TOPICS:
            # Evict the least-evidenced topic — it keeps what it knows.
            weakest = min(
                self.topic_receptivity,
                key=lambda t: self.topic_receptivity[t].evidence,
            )
            del self.topic_receptivity[weakest]
        posterior = Beta()
        self.topic_receptivity[name] = posterior
        return posterior
