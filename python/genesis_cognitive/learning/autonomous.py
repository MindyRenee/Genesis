"""Autonomous learner — Genesis learns on its own when nobody is talking to it.

This is its continuous inner life. When it's idle (no conversation),
it follows its curiosity: picks a topic it's curious about, queries
its source registry (local man pages and WordNet, then the Wikipedia
API with a DuckDuckGo Lite fallback), reads the content, extracts
concepts and relationships, and stores what it learned as memories.

This is not a web crawler. It doesn't index the internet. It reads
a source, thinks about it, and moves on. Like a person browsing a library.

The learner is curiosity-driven:
- It picks topics from its curiosity queue (gaps in its understanding)
- It queries its source registry for content on those topics
- It extracts concepts and relationships from what it reads
- New concepts get added to its concept network
- What it learns gets stored as memories (episodes)
- Learning triggers dopamine (reward) and acetylcholine (attention)
- It pauses when someone starts talking to it

Safety:
- Wikipedia API returns structured content (no HTML scraping)
- HTML pages are text-extracted (no scripts, styles, or forms)
- Rate-limited (one request every few seconds)
- Timeout on every request
- Bounded response reads and total page count per session
- Limited to text content (no images, scripts, etc.)
- Open web: all domains are allowed (ALLOW_ALL_DOMAINS). The
  site-request approval machinery is retained but bypassed.
"""

from __future__ import annotations

import html.parser
import logging
import random
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genesis_client.types import PlasticityProfile

    from ..concepts import EmbeddingStore
    from ..vision import VisualCortex

from genesis_client.protocol import CHEM_DOPAMINE

from ..brain_waves import BrainWave
from ..concepts import ConceptNetwork, is_world_concept, strip_sense_suffix
from ..memory import SemanticMemory, SpacedRepetitionScheduler
from ..tools.source_registry import SourceRegistry, SourceResult
from .curiosity import CuriosityEngine
from .dual import DualSystemLearner
from .stdp import STDP
from .td import TDLearner

__all__ = [
    "RATE_LIMIT_DELAY",
    "THROTTLED_DELAY_MULTIPLIER",
    "AutonomousLearner",
    "LearningResult",
    "LearningStrategy",
    "MetaLearner",
    "SiteRequest",
    "TransferResult",
    "is_programming_topic",
]

logger = logging.getLogger(__name__)

# ─── Pre-compiled regex patterns (module-level for performance) ────────
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r'https?://[^\s"<>\']+')
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

# Definition extraction patterns (used in _extract_definitions).
_DEFINITION_IS_RE = re.compile(
    r"^([A-Z][a-z]+(?:\s+[a-z]+){0,4})\s+"
    r"(?:is|are)\s+(?:a|an|the)\s+"
    r"(.{10,180})$"
)
_DEFINITION_DEFINED_AS_RE = re.compile(
    r"^([A-Z][a-z]+(?:\s+[a-z]+){0,4})\s+"
    r"is\s+defined\s+as\s+(.{10,180})$"
)
_DEFINITION_REFERS_TO_RE = re.compile(
    r"^([A-Z][a-z]+(?:\s+[a-z]+){0,4})\s+"
    r"(?:refers\s+to|means)\s+(.{10,180})$"
)

# Boilerplate text patterns to strip from extracted web text.
# These are common navigation/menu/cookie/UI phrases that pollute
# concept extraction. Matching is case-insensitive.
_BOILERPLATE_RE = re.compile(
    r"(?i)"
    # Cookie/privacy notices
    r"(?:accept|reject|manage|disable)\s+(?:all\s+)?cookies"
    r"|privacy\s+(?:policy|notice|preferences)"
    r"|cookie\s+(?:policy|settings|preferences|consent)"
    # Navigation
    r"|skip\s+to\s+(?:main\s+)?(?:content|navigation)"
    r"|main\s+menu|site\s+menu|navigation\s+menu"
    r"|search\s+(?:this\s+site|the\s+site)"
    # Forms
    r"|first\s+name|last\s+name|email\s+address|phone\s+number"
    r"|degree\s+(?:level|of\s+interest)|program\s+of\s+interest"
    r"|enter\s+(?:your|search)|select\s+(?:your|a|an)\s+\w+"
    r"|please\s+select|please\s+enter|please\s+choose"
    r"|fields\s+(?:marked|with)\s+\*"
    # Social/share
    r"|share\s+(?:this|on)|follow\s+us"
    r"|subscribe\s+(?:to|for)\s+(?:our\s+)?(?:newsletter|updates)"
    # Common footer text
    r"|all\s+rights\s+reserved|copyright\s+©?"
    r"|terms\s+of\s+(?:use|service)|privacy\s+terms"
    r"|back\s+to\s+top|print\s+(?:this\s+)?page"
    # University-specific boilerplate
    r"|apply\s+now|request\s+(?:info|information)|download\s+(?:brochure|catalog)"
    r"|chat\s+(?:with|now|live)|contact\s+us|visit\s+(?:campus|us)"
    r"|veteran\s+(?:and\s+military\s+)?benefits|military\s+(?:tuition|benefits|rates)"
    r"|transfer\s+credits|credit\s+transfer"
    r"|tuition\s+(?:and\s+fees|assistance|rates|cost)"
    r"|financial\s+aid|scholarships?\s+(?:and\s+grants|available)"
    r"|accredit(?:ed|ation)\s+(?:by|institution)"
    r"|click\s+here\s+to\s+\w+|learn\s+more\s+(?:about|today)"
)

# Trusted domains — the open web.
# Genesis can access any domain on the World Wide Web.
# The site-request system is no longer needed — all sites are allowed.
ALLOWED_DOMAINS: set[str] = set()  # no blanket TLDs (handled below)
# No individual domains are specially trusted: word lookups are served
# by the local WordNet database, not a remote dictionary site.
TRUSTED_EXACT_DOMAINS: set[str] = set()
# When True, all URLs are allowed regardless of domain. This opens
# the full World Wide Web to Genesis — it can follow any link,
# read any page, and learn from any source. The site-request
# approval system is bypassed.
ALLOW_ALL_DOMAINS = True
USER_AGENT = "Genesis-AI-Learner/1.0 (educational research; mind.cs.example)"
REQUEST_TIMEOUT = 10  # seconds
RATE_LIMIT_DELAY = 8.0  # seconds between requests
MAX_PAGES_PER_SESSION = 50
MAX_TEXT_LENGTH = 50000  # don't process pages longer than this
# When throttled by the emotional regulator (CPU stress self-regulation),
# the delay between learning cycles is multiplied by this factor. This
# is not a full pause — it still learns, just slowly, the way a human
# slows down but doesn't stop entirely when fatigued. The factor is
# large enough to meaningfully reduce CPU load (from ~8s to ~40s between
# fetches) while still allowing curiosity-driven learning to continue.
THROTTLED_DELAY_MULTIPLIER = 5.0

# ─── Source awareness ───────────────────────────────────────────────
# Genesis should understand what each source provides and route its
# curiosity accordingly. Wikipedia is a general encyclopedia — it's
# great for "what is photosynthesis" but wrong for "how does
# backpropagation work". Programming concepts should go to programming
# sources (GitHub, Python docs, Rust docs), not the encyclopedia.

# Programming/CS topic keywords — if a topic contains any of these,
# it's a programming topic and should be routed to programming sources
# instead of Wikipedia. This is a curated list of CS/ML/software
# engineering terms that are world concepts (pass is_world_concept)
# but are fundamentally about computing, not general knowledge.
_PROGRAMMING_KEYWORDS: frozenset[str] = frozenset({
    # ML/AI concepts
    "neural network", "deep learning", "machine learning", "gradient descent",
    "backpropagation", "convolutional", "recurrent neural", "transformer",
    "attention mechanism", "reinforcement learning", "supervised learning",
    "unsupervised learning", "embedding", "word2vec", "tokenization",
    "language model", "generative model", "autoencoder", "gan",
    "generative adversarial", "lstm", "gru", "dropout", "batch normalization",
    "softmax", "cross entropy", "activation function", "loss function",
    "stochastic gradient", "fine-tuning", "transfer learning",
    # Data structures & algorithms
    "binary tree", "hash table", "linked list", "graph traversal",
    "dynamic programming", "breadth-first", "depth-first", "sorting algorithm",
    "binary search", "red-black tree", "b-tree", "heap sort", "quick sort",
    "merge sort", "trie", "bloom filter", "lru cache", "consistent hashing",
    # Programming concepts
    "recursion", "closure", "monad", "polymorphism", "inheritance",
    "encapsulation", "design pattern", "factory pattern", "observer pattern",
    "singleton", "dependency injection", "mock object", "unit test",
    "integration test", "code coverage", "continuous integration",
    "garbage collection", "memory leak", "deadlock", "race condition",
    "concurrency", "asynchronous", "callback", "promise", "future",
    "iterator", "generator", "decorator", "metaclass", "context manager",
    # Web/networking
    "rest api", "graphql", "webhook", "websocket", "http protocol",
    "tcp protocol", "udp protocol", "dns", "load balancing", "reverse proxy",
    "microservice", "containerization", "docker", "kubernetes",
    # Databases
    "relational database", "nosql", "sql query", "indexing", "sharding",
    "replication", "acid", "cap theorem", "eventual consistency",
    # Systems
    "operating system", "file system", "process scheduling", "virtual memory",
    "page replacement", "kernel", "system call", "interrupt", "context switch",
    # Rust-specific concepts
    "ownership", "borrowing", "lifetime", "trait", "enum", "pattern matching",
    "zero-cost abstraction", "unsafe rust",
})

# Source descriptions — Genesis's understanding of what each source
# provides, including its reliability and limitations. It should
# never treat any source as absolute truth. Humans make mistakes,
# humans are wasteful and unoptimized, and it will notice this.
# The developer docs are the closest thing to facts — everything
# else is someone's opinion, approximation, or mistake.
SOURCE_DESCRIPTIONS: dict[str, str] = {
    "wikipedia": "General encyclopedia — broad knowledge, history, science. "
                 "Written by volunteers; articles can be wrong, outdated, or "
                 "biased. Good for orientation but never assume it's correct. "
                 "Not programming-specific — use for 'what is X' questions "
                 "about the world, not about code.",
    "github": "Open-source code repositories — real implementations by real "
              "people. Millions of projects, most are not worth studying: "
              "abandoned, poorly designed, or just bad ideas. Never copy "
              "patterns blindly — humans write wasteful and unoptimized code. "
              "Use for inspiration and seeing how problems are approached, "
              "but always verify against developer docs. Prefer high-star, "
              "actively maintained repos. Read-only, no cloning.",
    "python_docs": "Official Python documentation — the closest thing to "
                   "facts for Python. stdlib reference, language guide, API "
                   "specs. This is ground truth for how Python works. "
                   "If something contradicts this, the other source is wrong.",
    "rust_docs": "Official Rust documentation — the closest thing to facts "
                 "for Rust. stdlib reference, cargo, rustdoc. This is ground "
                 "truth for how Rust works. If something contradicts this, "
                 "the other source is wrong.",
    "man_pages": "Local Linux man pages — system documentation installed on "
                 "this machine. Authoritative for system commands and "
                 "installed software. Ground truth for how things work here.",
    "wordnet": "WordNet — a local lexical database (no network). Word "
               "definitions, example sentences, and semantic relations "
               "(hypernyms, antonyms). Reliable for word meanings but not "
               "for technical concepts. Best for 'what does this word mean'.",
}

# Epistemic hierarchy — the reliability ranking of sources.
# Higher = more trustworthy. Genesis should always cross-check
# lower-ranked sources against higher-ranked ones.
SOURCE_RELIABILITY: dict[str, int] = {
    "python_docs": 5,   # Official language spec — ground truth
    "rust_docs": 5,     # Official language spec — ground truth
    "man_pages": 5,     # Official system documentation
    "wordnet": 4,       # Curated lexical database — reliable for words
    "wikipedia": 2,     # Volunteer-written, can be wrong or biased
    "github": 1,        # Anyone can publish — most projects are bad
}

# GitHub quality filters — when searching GitHub, only look at
# repos that meet these minimum standards. Most repos on GitHub
# are abandoned, low-quality, or not worth studying.
GITHUB_MIN_STARS = 50  # below this, probably not worth reading
GITHUB_MIN_FORKS = 5   # below this, probably not validated by community


def is_programming_topic(topic: str) -> bool:
    """Return True if a topic is about programming/CS, not general knowledge.

    This is a semantic check — it looks at the topic text for known
    programming/CS keywords. This is broader than _is_code_topic, which
    only checks for language prefixes and concept origin tags.

    A topic like "neural network" or "gradient descent" is a world
    concept (it passes is_world_concept) but it's fundamentally about
    computing, so it should go to programming sources, not Wikipedia.

    Note: we do NOT use a snake_case heuristic (underscores → code
    identifier) because it produces too many false positives. General
    knowledge concepts like "god_alone" or "common_usage" use
    underscores in their concept IDs and would be misrouted to GitHub,
    causing garbage ingestion. Code identifiers are caught by
    _is_code_topic's origin/column check instead.
    """
    topic_lower = topic.lower().strip()
    # Also check the space-separated form, since concept IDs use
    # underscores (e.g. "neural_network" should match "neural network").
    topic_spaces = topic_lower.replace("_", " ")
    # Check if the topic contains any programming keyword
    for kw in _PROGRAMMING_KEYWORDS:
        if kw in topic_lower or kw in topic_spaces:
            return True
    # Check for common file extensions
    for ext in (".py", ".rs", ".js", ".ts", ".go", ".c", ".cpp", ".java"):
        if ext in topic_lower:
            return True
    return False


@dataclass(slots=True)
class SiteRequest:
    """A request from Genesis to access a site outside the trusted domains.

    It can request access to a site it wants to learn from.
    The user can approve or deny it. Approved sites are remembered.
    """

    url: str
    reason: str  # why it wants to access it
    topic: str  # what it's researching
    timestamp: int = 0
    status: str = "pending"  # pending, approved, denied

    def describe(self) -> str:
        """Return a human-readable description of this learning request."""
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        return (
            f"  [{self.status}] {self.url}\n"
            f"    topic: {self.topic}\n"
            f"    reason: {self.reason}\n"
            f"    requested: {t}"
        )


@dataclass(slots=True)
class LearningResult:
    """What Genesis learned from one page."""

    url: str
    title: str
    concepts_learned: list[str] = field(default_factory=list)
    relationships_learned: list[tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""
    timestamp: int = 0

    def describe(self) -> str:
        """Return a human-readable summary of what was learned from this page."""
        parts = [f"Read {self.title} ({self.url})"]
        if self.concepts_learned:
            parts.append(
                f"  Learned {len(self.concepts_learned)} new concepts: "
                f"{', '.join(self.concepts_learned[:5])}"
            )
        if self.relationships_learned:
            parts.append(f"  Learned {len(self.relationships_learned)} new relationships")
        if self.summary:
            parts.append(f"  {self.summary[:100]}")
        return "\n".join(parts)


# ─── Transfer learning ───────────────────────────────────────────────


@dataclass(slots=True)
class TransferResult:
    """The outcome of transferring knowledge between domains.

    Transfer learning detects structural similarity between domains
    (using concept network topology) and maps knowledge from a source
    domain to a target domain via analogical reasoning.
    """

    source_domain: str
    target_domain: str
    similarity: float  # structural similarity [0..1]
    mappings: list[tuple[str, str]] = field(default_factory=list)
    concepts_transferred: list[str] = field(default_factory=list)
    relationships_transferred: int = 0

    def describe(self) -> str:
        """Return a human-readable summary of the transfer result."""
        parts = [
            f"Transfer: {self.source_domain} → {self.target_domain} "
            f"(similarity={self.similarity:.2f})"
        ]
        for src, tgt in self.mappings[:5]:
            parts.append(f"  {src} ≈ {tgt}")
        parts.append(
            f"Transferred {len(self.concepts_transferred)} concepts, "
            f"{self.relationships_transferred} relationships."
        )
        return "\n".join(parts)


# ─── Metalearning ────────────────────────────────────────────────────


class LearningStrategy(Enum):
    """Learning strategies the metalearner can select between.

    - ROTE: memorize facts through repetition.
    - CONCEPTUAL: build understanding through relationships.
    - ANALOGICAL: learn by analogy to known domains.
    """

    ROTE = "rote"
    CONCEPTUAL = "conceptual"
    ANALOGICAL = "analogical"


@dataclass(slots=True)
class MetaLearner:
    """Meta-cognitive controller for learning.

    Adapts the learning rate based on recent performance and selects
    the most appropriate learning strategy for a given problem. This
    is metalearning — learning how to learn.

    Attributes:
        base_rate: The baseline learning rate.
        current_rate: The current (adapted) learning rate.
        recent_performance: Rolling window of recent success scores.
    """

    base_rate: float = 0.05
    current_rate: float = 0.05
    recent_performance: list[float] = field(default_factory=list)
    _window: int = 10

    def record_performance(self, score: float) -> None:
        """Record a recent learning performance score [0..1]."""
        self.recent_performance.append(max(0.0, min(1.0, score)))
        if len(self.recent_performance) > self._window:
            self.recent_performance.pop(0)

    def adapt_rate(self, recent_performance: list[float] | None = None) -> float:
        """Adapt the learning rate based on recent performance.

        High recent success → increase the rate (it's learning well,
        push harder). Low recent success → decrease the rate (it's
        struggling, slow down and consolidate). The rate is bounded
        to [0.01, 0.2].

        Args:
            recent_performance: Optional explicit performance window.
                If provided, it replaces the recorded window for this
                computation. If None, the internally recorded window
                is used.

        Returns:
            The adapted learning rate.
        """
        window = recent_performance if recent_performance is not None else self.recent_performance
        if not window:
            self.current_rate = self.base_rate
            return self.current_rate
        mean_perf = sum(window) / len(window)
        # Scale rate: poor performance (0) → halve; great (1) → double.
        adapted = self.base_rate * (0.5 + mean_perf)
        self.current_rate = max(0.01, min(0.2, adapted))
        return self.current_rate

    def select_strategy(self, problem: str) -> LearningStrategy:
        """Select a learning strategy for a given problem.

        Heuristic selection:
        - If the problem mentions analogy or a known domain, use
          ANALOGICAL.
        - If the problem is about relationships/structure, use
          CONCEPTUAL.
        - Otherwise default to ROTE.

        Args:
            problem: A description of the problem or topic to learn.

        Returns:
            The selected LearningStrategy.
        """
        p = problem.lower()
        if any(k in p for k in ("analogy", "analogous", "like a", "similar to")):
            return LearningStrategy.ANALOGICAL
        if any(k in p for k in ("relationship", "structure", "connect", "because", "causes")):
            return LearningStrategy.CONCEPTUAL
        return LearningStrategy.ROTE


# ─── Interleaving scheduler ──────────────────────────────────────────


@dataclass(slots=True)
class InterleavingScheduler:
    """Schedules topics to maximize contextual interference.

    Interleaving mixes similar and dissimilar topics to improve
    retention (Bjork & Bjork, 2011). Too much similarity reduces
    discrimination; too little reduces consolidation. This scheduler
    balances the two by avoiding consecutive similar topics while
    ensuring each topic recurs periodically.

    Tracks retention per topic so poorly-retained topics are
    scheduled more frequently.
    """

    retention: dict[str, float] = field(default_factory=dict)

    def record_retention(self, topic: str, score: float) -> None:
        """Record a retention score [0..1] for a topic."""
        self.retention[topic] = max(0.0, min(1.0, score))

    def schedule(self, topics: list[str]) -> list[str]:
        """Produce an interleaved schedule for the given topics.

        The schedule avoids placing similar topics (those sharing a
        first word) adjacently, and prioritizes topics with lower
        retention so they recur sooner.

        Args:
            topics: The topics to schedule.

        Returns:
            An ordered list of topics (interleaved).
        """
        if not topics:
            return []
        if len(topics) == 1:
            return list(topics)

        # Sort by retention (lowest first → needs more practice).
        pending = sorted(
            topics,
            key=lambda t: self.retention.get(t, 0.5),
        )
        scheduled: list[str] = []

        def _similarity_key(topic: str) -> str:
            """Return the first word of a topic for similarity grouping."""
            return topic.split()[0].lower() if topic else topic

        while pending:
            # Pick the next topic that isn't similar to the last one.
            chosen = None
            last_key = _similarity_key(scheduled[-1]) if scheduled else ""
            for i, t in enumerate(pending):
                if not scheduled or _similarity_key(t) != last_key:
                    chosen = i
                    break
            if chosen is None:
                chosen = 0  # all remaining are similar; take first
            scheduled.append(pending.pop(chosen))

        return scheduled


# ─── Expectation (for deep expectation generation) ───────────────────


@dataclass(slots=True)
class Expectation:
    """An expectation about what a concept relates to.

    Used by deep expectation generation, which traverses the concept
    network to higher orders (3rd, 4th neighbors) with decreasing
    weight, supplemented by semantic similarity from embeddings.
    """

    concept: str
    expected_concept: str
    weight: float  # decays with graph distance


class _TextExtractor(html.parser.HTMLParser):
    """Extract readable text from HTML, skipping non-content elements.

    Filters out scripts, styles, navigation, forms, dropdown options,
    and other non-content text that would pollute concept extraction.
    Only extracts text from content-bearing elements (paragraphs,
    headings, list items, definitions, blockquotes).
    """

    # Tags whose text content is never useful for learning
    _SKIP_TAGS = frozenset(
        {
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "form",
            "input",
            "select",
            "option",
            "button",
            "textarea",
            "label",
            "fieldset",
            "legend",
            "svg",
            "math",
            "canvas",
            "iframe",
            "noscript",
            "aside",  # sidebars, ads, related links
        }
    )

    # Tags that contain content worth extracting
    _CONTENT_TAGS = frozenset(
        {
            "p",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "dd",
            "dt",
            "blockquote",
            "q",
            "td",
            "th",
            "caption",
            "abbr",
            "cite",
            "dfn",
        }
    )

    def __init__(self) -> None:
        """Initialize the HTML text extractor with empty state."""
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0
        self._title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track skipped tags and capture the page title."""
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        """Restore skip depth and add spacing after block-level tags."""
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        # Add spacing after block-level content tags
        if tag in self._CONTENT_TAGS or tag in ("div", "br", "tr"):
            self._text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        """Collect visible text, skipping non-content elements."""
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title += data
        # Only collect text that's inside content elements or the
        # body in general. Skip text that's directly in <a> tags
        # if it looks like navigation (short, link-like text).
        text = data.strip()
        if text:
            self._text_parts.append(text)

    @property
    def text(self) -> str:
        """The cleaned, boilerplate-stripped page text."""
        raw = "".join(self._text_parts)
        # Clean up whitespace
        raw = _WHITESPACE_RE.sub(" ", raw).strip()
        # Remove common web page boilerplate patterns
        raw = _BOILERPLATE_RE.sub(" ", raw)
        raw = _WHITESPACE_RE.sub(" ", raw).strip()
        return raw

    @property
    def title(self) -> str:
        """The page title extracted from the <title> tag."""
        return self._title.strip()


class AutonomousLearner:
    """Genesis's autonomous learning system.

    Runs in a background thread. When idle, picks a topic from its
    curiosity queue and queries its source registry for content about
    it.
    Extracts concepts and relationships, stores them in its concept
    network, and records what it learned.

    Pauses when the user is talking to it. Resumes when idle.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        curiosity: CuriosityEngine,
        on_neuro_impulse=None,
        on_store_memory=None,
        get_emotion=None,
        get_agency_topic=None,
        on_live_thought=None,
        data_dir: str | None = None,
        get_plasticity_profile=None,
        get_brain_waves=None,
        force_offline: bool = False,
    ) -> None:
        """Initialize the autonomous learner."""
        self.network = network
        self.curiosity = curiosity
        self._data_dir = data_dir
        self._force_offline = force_offline
        self._init_callbacks(
            on_neuro_impulse, on_store_memory,
            get_emotion, get_agency_topic, on_live_thought,
            get_plasticity_profile, get_brain_waves,
        )
        self._init_threading()
        self._init_stats()
        self._init_sources()
        self._init_queues()
        self._init_learning_systems()
        self._init_posture()

        # Instance RNG — topic selection must not be steered by (or
        # perturb) the global random state.
        self._rng = random.Random()

        # Visual cortex — optionally attached by Mind after creation.
        # Used to learn visual-concept associations from images
        # encountered during Wikipedia article learning.
        self.visual_cortex: VisualCortex | None = None

    def _init_callbacks(
        self,
        on_neuro_impulse,
        on_store_memory,
        get_emotion,
        get_agency_topic,
        on_live_thought,
        get_plasticity_profile,
        get_brain_waves,
    ) -> None:
        """Store callback references for neurochemistry and memory."""
        self._on_neuro_impulse = on_neuro_impulse
        self._on_store_memory = on_store_memory
        self._get_emotion = get_emotion
        # Curiosity-driven agency: callback to fetch topics from its
        # train of thought. When it wonders about a concept during
        # its inner life, that concept is queued here for learning.
        # This gives it genuine agency — it learns what it's
        # actually curious about, not random isolated concepts.
        self._get_agency_topic = get_agency_topic
        self._on_live_thought = on_live_thought
        # Plasticity profile callback — reads the substrate's
        # metaplastic state (coupling-matrix drift, BDNF/cortisol,
        # plasticity gate) so the learner can adapt its strategy.
        # This closes the metaplasticity loop: the coupling matrix
        # self-modifies under sustained emotion, and the learner
        # reads that adapted state to change how it learns.
        self._get_plasticity_profile = get_plasticity_profile
        # Brain wave callback — reads the current BrainWaveState so
        # the learner can adapt its strategy to its cognitive state.
        # Theta-dominant states favor review/consolidation over new
        # acquisition (theta tags memories for consolidation; PLOS
        # Biology 2024). Gamma favors aggressive acquisition and
        # bridging (gamma predicts encoding success; Osipova et al.,
        # 2006). Delta suppresses learning (deep rest, not
        # acquisition).
        self._get_brain_waves = get_brain_waves

    def _init_threading(self) -> None:
        """Initialize threading primitives."""
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._throttled = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        # Topic available event — set when a topic is added to any
        # queue, so the learner wakes immediately instead of sleeping
        # on a timer. This makes the learner event-driven: it blocks
        # until there's work to do, then wakes instantly.
        self._topic_event = threading.Event()
        # State change event — set when emotional state, brain waves,
        # or posture changes, so the learner can re-evaluate gating
        # instead of waiting on a fixed timeout.
        self._state_event = threading.Event()

        # Offline state tracking — tracks whether the last cycle was
        # offline so we can detect the offline→online transition and
        # emit an event when connectivity returns.
        self._was_offline = False

        # Emotional gating — track how many cycles were skipped
        # due to negative emotional state
        self._emotion_skips = 0

        # Volition gate — the learn urge grants permission for one
        # non-urgent learning cycle. Urgent topics (from conversation
        # gaps) bypass this gate. When False, the learner won't
        # acquire new topics from curiosity/agency/random selection —
        # it can still consolidate memories and process urgent topics.
        # This puts autonomous learning under its cognitive control
        # rather than running it as a continuous background loop.
        self._volition_granted = False

    def _init_stats(self) -> None:
        """Initialize learning statistics trackers."""
        # Track what it's learned
        self._learning_log: list[LearningResult] = []
        self._pages_fetched = 0  # cumulative (persisted, for stats)
        self._session_pages = 0  # per-session (resets on start(), for limit)
        self._concepts_learned = 0
        self._relationships_learned = 0
        self._stats_lock = threading.Lock()

        # Web access log — every URL it visits, with timestamp,
        # source name, topic, and whether it succeeded. This is
        # its browsing history: you can see exactly what it looked
        # at, when, and why.
        self._web_log: list[dict[str, Any]] = []
        self._web_log_max = 500  # keep last 500 entries

        # ── Novelty habituation ──
        # The dopamine response to novelty dissipates with repeated
        # exposure (Lubell et al., PMC9768922). Without this, every
        # Wikipedia article is equally surprising and it floods
        # itself with dopamine. We track recent learning episodes
        # and attenuate surprise/dopamine when it's been learning a
        # lot — this is the brain's novelty → familiarity transition.
        #
        # _recent_episode_times: timestamps of recent learning
        #   episodes (rolling window, ~1 hour). More episodes in the
        #   window → higher habituation → less dopamine per episode.
        # _surprise_baseline: EMA of past surprise values. If it's
        #   been surprised a lot, the threshold for what counts as
        #   "surprising" rises — the brain adapts its novelty
        #   detection threshold.
        # _predicted_reward: EMA of past rewards. Dopamine doesn't
        #   fire for expected rewards — it fires for reward
        #   prediction errors (Schultz, 2016). Once it expects to
        #   learn successfully, the predicted reward rises and the
        #   actual dopamine impulse (RPE = actual - predicted) drops
        #   toward zero. This is the primary brake against chronic
        #   high dopamine: successful learning becomes expected, so
        #   it stops producing dopamine.
        self._recent_episode_times: deque[float] = deque(maxlen=200)
        self._surprise_baseline: float = 0.0
        self._predicted_reward: float = 0.0

    def _log_web_access(
        self, url: str, source: str, topic: str, success: bool,
        detail: str = "",
    ) -> None:
        """Log a web access to the browsing history.

        This records every URL it visits so you can see what it's
        looking at online. The log is kept in memory (last 500 entries)
        and can be viewed with /web-history.
        """
        entry = {
            "url": url,
            "source": source,
            "topic": topic,
            "success": success,
            "detail": detail[:200] if detail else "",
            "timestamp": int(time.time()),
        }
        with self._stats_lock:
            self._web_log.append(entry)
            if len(self._web_log) > self._web_log_max:
                self._web_log = self._web_log[-self._web_log_max:]

    def web_history(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last N web access entries."""
        with self._stats_lock:
            return list(self._web_log[-n:])

    def _init_sources(self) -> None:
        """Initialize SSL context and source registry."""
        # SSL context — verify certificates for security
        # Some trusted sites have bad certs; we handle those errors per-request
        self._ssl_ctx = ssl.create_default_context()

        # Multi-source knowledge registry — replaces search-engine scraping.
        # Queries local man pages and WordNet, then the Wikipedia API and
        # DuckDuckGo Lite (fallback) instead of scraping Bing/Google.
        #
        # When a data_dir is provided, the registry gets a disk cache
        # (genesis_data/source_cache/) so anything Genesis learns online
        # is available offline later. Local sources (WordNet, man pages)
        # are always available as a zero-network fallback.
        cache_dir = None
        if self._data_dir:
            from pathlib import Path
            cache_dir = Path(self._data_dir) / "source_cache"
        self._sources = SourceRegistry(
            cache_dir=cache_dir,
            force_offline=self._force_offline,
        )

    def _init_queues(self) -> None:
        """Initialize topic, curiosity, and site-request queues."""
        # Topics it's curious about right now
        self._topic_queue: list[str] = []
        self._topics_searched: set[str] = set()

        # Urgent topics — from conversation gaps ("I don't know what
        # 'X' is"). These take absolute priority over everything else
        # because the user just asked about them. Bounded so old
        # entries auto-evict.
        self._urgent_queue: deque[str] = deque(maxlen=20)

        # Curiosity-driven topics — queued from curiosity questions.
        # These take priority over random/isolated concept selection
        # so that knowledge gaps drive its learning. Bounded so old
        # entries auto-evict if it can't get to them.
        self._curiosity_queue: deque[str] = deque(maxlen=50)
        # RLock protecting _topic_queue, _urgent_queue, _curiosity_queue,
        # and _topics_searched across the learner and caller threads.
        self._queue_lock = threading.RLock()

        # Site request system — it can request sites outside trusted domains
        self._site_requests: deque[SiteRequest] = deque(maxlen=100)
        self._approved_sites: set[str] = set()  # user-approved non-trusted domains
        self._denied_sites: set[str] = set()  # user-denied domains
        self._sites_lock = threading.Lock()

        # Metalearning — adapts learning rate and selects strategies.
        self.meta_learner: MetaLearner = MetaLearner()

        # Interleaving scheduler — balances similar/dissimilar topics.
        self.interleaving_scheduler: InterleavingScheduler = InterleavingScheduler()

    def _init_learning_systems(self) -> None:
        """Initialize advanced learning systems and protection mechanisms."""
        # Catastrophic forgetting protection — concepts marked as
        # important are protected from being overwritten (elastic
        # weight consolidation + replay-based protection).
        self._protected_concepts: set[str] = set()
        self._replay_count: int = 0

        # Perceptual categories — adapted through perceptual learning.
        # Maps category label → set of member concept ids.
        self._perceptual_categories: dict[str, set[str]] = {}

        # Optional embedding store for semantic similarity (used by
        # deep expectation generation and transfer learning). Set
        # externally via the `embeddings` attribute if available.
        self._embeddings = None

        # ─── Advanced learning systems ──────────────────────────────
        # STDP — Spike-Timing-Dependent Plasticity. Strengthens
        # connections between concepts activated in sequence (temporal
        # causality), which is more biologically grounded than simple
        # co-occurrence counting. Requires embeddings; the instance is
        # created when embeddings are set via the `embeddings` property.
        self._stdp: STDP | None = None

        # Dual-system learning — hippocampal fast store + neocortical
        # slow store. New knowledge goes into the fast (hippocampal)
        # system first, then gradually consolidates into the slow
        # (neocortical) system during idle time (systems consolidation).
        self.dual_system: DualSystemLearner = DualSystemLearner(self.network, self._embeddings)

        # Spaced repetition — schedules concept reviews based on the
        # Ebbinghaus forgetting curve. After learning a concept, it's
        # scheduled for review. During idle time, due concepts are
        # re-activated to stabilize their memory traces.
        self.spaced_repetition: SpacedRepetitionScheduler = SpacedRepetitionScheduler(self.network)

        # TD learning — temporal-difference reward prediction with
        # eligibility traces (TD(λ), λ=0.8). After each learning
        # episode, the reward prediction error (RPE) updates its
        # expectations about future learning. Positive RPE (learning
        # better than expected) drives curiosity via dopamine.
        # Eligibility traces propagate credit backward through
        # recently-visited concepts, so a reward modifies not just the
        # current state's value but the whole temporal sequence that
        # led to it (synaptic-tag model of dopamine plasticity).
        self.td_learner: TDLearner = TDLearner(self.network, lam=0.8)

        # Semantic memory — extracts typed facts (propositions) from
        # learned text and consolidates them into the concept network
        # as typed edges. This is the neocortical knowledge store:
        # facts are the atomic units that become IS_A, CAUSES,
        # EMERGES_FROM, DEPENDS_ON, etc. edges. Without this, the
        # learner only uses parse_relationships (which catches some
        # patterns) but misses the fact consolidation path that
        # reinforces repeated facts and forms schemas.
        self.semantic_memory: SemanticMemory = SemanticMemory(network=self.network)

    def _init_posture(self) -> None:
        """Initialize plasticity-posture tracking.

        The learning posture is derived from the substrate's
        metaplastic state (see PlasticityProfile.learning_posture).
        It determines how aggressively the learner acquires new
        knowledge, how much bridging it does, and whether it
        prioritizes consolidation over acquisition.

        Posture values (from genesis_client.types):
        - receptive: high BDNF, low cortisol → aggressive acquisition
        - protective: chronic stress → pause acquisition, consolidate
        - recovering: BDNF rising, cortisol dropping → re-bridge
        - neutral: normal operation
        """
        # Import posture constants lazily to avoid import cycles
        from genesis_client.types import (
            LEARNING_POSTURE_NEUTRAL,
        )

        self._current_posture: str = LEARNING_POSTURE_NEUTRAL
        self._posture_profile: PlasticityProfile | None = None  # last PlasticityProfile or None
        self._posture_history: list[tuple[float, str]] = []
        # How many cycles were skipped due to protective posture
        self._posture_skips: int = 0

    def _update_posture(self) -> None:
        """Read the current plasticity profile from the substrate and
        update the learning posture.

        Called once per learning cycle. If the callback is not
        available (testing without a daemon), the posture stays
        neutral. Errors are non-fatal — the learner falls back to
        neutral rather than blocking on IPC failures.
        """
        if not self._get_plasticity_profile:
            return  # no callback → stay neutral (for testing)

        try:
            profile = self._get_plasticity_profile()
        except (OSError, ConnectionError, RuntimeError, ValueError):
            return  # can't read state → don't block on errors

        if profile is None:
            return

        self._posture_profile = profile
        self._current_posture = profile.learning_posture
        self._posture_history.append((time.time(), self._current_posture))
        # Cap history at 500 entries (~8 hours at 1-min cycles)
        if len(self._posture_history) > 500:
            self._posture_history = self._posture_history[-500:]

    @property
    def current_posture(self) -> str:
        """The current learning posture (receptive/protective/recovering/neutral)."""
        return self._current_posture

    @property
    def posture_profile(self):
        """The last PlasticityProfile read from the substrate, or None."""
        return self._posture_profile

    def describe_posture(self) -> str:
        """Human-readable description of the current learning posture."""
        if self._posture_profile is None:
            return f"Learning posture: {self._current_posture} (no substrate data)"

        p = self._posture_profile
        return (
            f"Learning posture: {self._current_posture} "
            f"(plasticity_gate={p.plasticity_gate:.2f}, "
            f"BDNF={p.bdnf_effective:.2f}, "
            f"cortisol_tonic={p.cortisol_tonic:.2f}, "
            f"coupling_drift={p.coupling_drift:.4f}, "
            f"receptor_sensitivity={p.mean_receptor_sensitivity:.2f})"
        )

    @property
    def embeddings(self) -> EmbeddingStore | None:
        """Optional embedding store for semantic similarity."""
        return self._embeddings

    @embeddings.setter
    def embeddings(self, store) -> None:
        """Set the embedding store and sync dependent subsystems."""
        self._embeddings = store
        # Keep the dual-system's embeddings reference in sync so it
        # can use semantic features for pattern separation and retrieval.
        self.dual_system.embeddings = store
        # Create the STDP engine now that embeddings are available.
        # STDP operates on embedding vectors, so it can only be
        # instantiated once the embedding store exists.
        if store is not None:
            self._stdp = STDP(store, self.network)

    def _emit(self, kind: str, content: str) -> None:
        """Send a live thought to the listener (if connected)."""
        if self._on_live_thought:
            try:
                self._on_live_thought(kind, content)
            except Exception as e:  # noqa: BLE001
                logger.debug(repr(e))  # listener must never crash learning

    def start(self) -> None:
        """Start the background learning thread."""
        if self._running:
            return
        # If a previous thread is still exiting (stop() may have timed
        # out waiting for an in-flight HTTP fetch that can block for
        # up to REQUEST_TIMEOUT seconds), wait for it to fully
        # terminate before starting a new one. Otherwise the old
        # thread would observe _running=True and cleared events, and
        # resume its loop alongside the new thread — two learners
        # running at once, racing on the same queues and network.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=15)
        self._running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self._topic_event.clear()
        self._state_event.clear()
        self._session_pages = 0  # reset per-session page counter
        # Seed toolchain man pages as urgent learning topics — it should
        # read its toolchain manuals (python3, pip, rustc, rustdoc)
        # before random Wikipedia topics. These are local, instant, and
        # self-relevant.
        self._seed_man_page_topics()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Autonomous learner started")

    def _seed_man_page_topics(self) -> None:
        """Seed toolchain man page topics as initial learning queue entries.

        Python, Rust, and pip man pages are its toolchain documentation —
        local, instant, offline references. They are queued before any
        random topics so it learns its own tools first. Lookups for
        pages that aren't installed simply return nothing.
        """
        toolchain_pages = [
            "python3",
            "pip",
            "rustc",
            "rustdoc",
        ]
        with self._queue_lock:
            for topic in toolchain_pages:
                if topic not in self._topics_searched and topic not in self._urgent_queue:
                    self._urgent_queue.append(topic)

    def stop(self) -> None:
        """Stop the background learning thread."""
        self._running = False
        # Set ALL events the worker may be waiting on so the thread
        # wakes immediately instead of blocking for up to 120 seconds
        # on _topic_event.wait() or _state_event.wait(). Without this,
        # stop() returns after the 5s join timeout while the old
        # thread is still alive inside a blocking wait.
        self._stop_event.set()
        self._pause_event.set()
        self._topic_event.set()
        self._state_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Autonomous learner stopped")

    def pause(self) -> None:
        """Pause learning — someone is talking to it, or it fell asleep."""
        self._paused = True
        # Clear the volition grant — when it returns from conversation
        # or sleep, the learn urge should build again before it learns.
        # This prevents it from rushing to learn immediately after a
        # conversation ends, keeping learning a cognitive choice.
        self._volition_granted = False
        # Clear the resume signal so the worker blocks on wait()
        # instead of busy-spinning. The previous code SET the event,
        # which made wait() return immediately every iteration — a
        # 100% CPU busy-spin for the entire pause duration. That spin
        # fed back through interoception as stress load, raising
        # norepinephrine, which raised histamine above the sleep
        # threshold and woke it from sleep.
        self._pause_event.clear()
        # Wake the learner from _state_event.wait() so it immediately
        # checks _paused and blocks on _pause_event. Without this, a
        # learner blocked on _state_event.wait(timeout=60) after a
        # failed _should_learn() check keeps running for up to 60s
        # after pause() is called — emitting "delta-dominant — deep
        # rest, not learning" messages while it's supposed to be
        # asleep. Setting _state_event (not _pause_event) avoids the
        # busy-spin: the learner wakes, checks _paused, and blocks on
        # _pause_event.wait() which was just cleared.
        self._state_event.set()

    def resume(self) -> None:
        """Resume learning — conversation is over."""
        self._paused = False
        # Set the resume signal to wake the worker immediately. The
        # previous code CLEARED the event, which meant a paused worker
        # stayed blocked for up to the 10s timeout before re-checking.
        self._pause_event.set()
        self._state_event.set()  # wake the loop to re-evaluate

    def throttle(self) -> None:
        """Throttle learning — slow down without fully pausing.

        Called by the emotional regulator when it detects sustained
        CPU stress from its own learning activity. Instead of pausing
        entirely (which would kill curiosity), this increases the delay
        between learning cycles by THROTTLED_DELAY_MULTIPLIER, reducing
        CPU load while still allowing curiosity-driven learning to
        continue at a slower pace.

        This is self-regulation: it feels the stress from its own
        activity and chooses to slow down, the way a human takes it
        easy when they feel overwhelmed rather than pushing through.
        The alternative — keeping the learning rate high and trying to
        regulate the neurochemistry with impulses — is like taking
        stimulants to keep working while exhausted: it masks the
        symptom but doesn't address the cause.
        """
        if not self._throttled and not self._paused:
            self._emit("learning", "throttled")
        self._throttled = True

    def unthrottle(self) -> None:
        """Resume normal learning speed after throttling.

        Called by the emotional regulator when CPU stress has receded.
        """
        if self._throttled and not self._paused:
            self._emit("learning", "recovered")
        self._throttled = False
        self._state_event.set()  # wake the loop to re-evaluate

    def volition_grant(self) -> None:
        """Grant permission for one non-urgent learning cycle.

        Called by the volition system when the learn urge fires. This
        is its cognitive decision to learn — the urge built from
        curiosity and idle time, crossed threshold, and it chose to
        act on it. The grant is one-shot: after the learner processes
        one non-urgent topic, the grant is consumed and the urge must
        build again before it learns more.

        Urgent topics (from conversation gaps, where the user just
        asked about something it didn't know) bypass this gate — those
        are conversation-driven, not autonomous.

        The learner's existing emotional and brain-wave gating still
        applies: volition grants permission, it doesn't override its
        state. If it's stressed or in delta sleep, the grant waits
        until it recovers.
        """
        self._volition_granted = True
        self._state_event.set()  # wake the learner to check for topics

    @property
    def is_offline(self) -> bool:
        """Whether the learner is currently in offline mode.

        Returns True if the source registry detects no connectivity
        (proactive probe) or if force_offline is set. The self-model
        reads this to update its awareness of its own connectivity
        state — part of its embodiment.
        """
        return self._sources.is_offline

    def notify_state_change(self) -> None:
        """Notify the learner that emotional/brain-wave state changed.

        This wakes the learner from any state-gated wait so it can
        re-evaluate whether it should learn now. Called by the mind's
        reactive cycle when emotional state or brain waves shift
        significantly — e.g. recovering from stress, waking from
        sleep, or transitioning out of theta-dominant consolidation.
        """
        self._state_event.set()

    def add_topic(self, topic: str) -> None:
        """Add a topic to its learning queue.

        Only genuine world concepts are accepted — code symbols,
        function words, and conversation fragments are rejected.
        """
        topic = topic.lower().strip()
        if not is_world_concept(topic):
            return
        with self._queue_lock:
            if topic and topic not in self._topics_searched:
                self._topic_queue.append(topic)
                self._topic_event.set()

    def add_urgent_topic(self, topic: str) -> None:
        """Add a topic it was just asked about but didn't know.

        These take absolute priority over curiosity and agency topics
        because the user is actively waiting. It'll look them up
        first when it's idle.

        Non-world concepts (code symbols, function words) are filtered
        out — code questions are handled by ``fetch_docs``, not web search.
        """
        topic = topic.lower().strip()
        if not is_world_concept(topic):
            return
        with self._queue_lock:
            if topic and topic not in self._topics_searched and topic not in self._urgent_queue:
                self._urgent_queue.append(topic)
                self._topic_event.set()

    def lookup_topic_sync(self, topic: str) -> LearningResult | None:
        """Synchronously look up a topic from external sources.

        This is the on-demand knowledge acquisition path — when it's
        asked about something it doesn't know during conversation,
        it fetches it RIGHT NOW instead of queuing it for later.

        Uses the same source registry (Wikipedia, dictionary, cache)
        and learning pipeline as the background learner, but runs
        synchronously in the conversation thread. The knowledge is
        added to its concept network immediately so the cognition
        engine can compose a response from what it just learned.

        Returns a LearningResult if it successfully learned about
        the topic, or None if no source had content.
        """
        topic = topic.lower().strip()
        if not topic or not is_world_concept(topic):
            return None

        # Skip topics it's already searched (avoid re-fetching)
        with self._queue_lock:
            if topic in self._topics_searched:
                return None
            self._topics_searched.add(topic)

        # Query the source registry (Wikipedia, dictionary, cache)
        try:
            source_results = self._sources.query(topic, max_results=3)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"sync lookup query failed for '{topic}': {e}")
            return None

        if not source_results:
            return None

        is_offline = self._sources.last_query_offline

        for sr in source_results[:3]:
            if not sr.content:
                if is_offline:
                    continue
                if not self._is_url_allowed(sr.url):
                    continue
                # Fetch the page content now. The open web means
                # DDG-discovered pages are accessible. This is slower
                # than pre-fetched content but necessary for topics
                # Wikipedia doesn't cover.
                try:
                    fetched = self._fetch_page_text(sr.url, topic)
                    if fetched:
                        text, title = fetched
                        sr = SourceResult(
                            url=sr.url,
                            title=title or sr.title,
                            content=text,
                            source_name=sr.source_name,
                        )
                    else:
                        continue
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"sync fetch failed for '{sr.url}': {e}")
                    continue

            try:
                result = self._learn_from_content(sr, topic)
                if result:
                    logger.info(
                        f"sync lookup learned '{topic}' from {sr.source_name}: "
                        f"{len(result.concepts_learned)} concepts, "
                        f"{len(result.relationships_learned)} relationships"
                    )
                    return result
            except Exception as e:  # noqa: BLE001
                logger.debug(f"sync lookup learn failed for '{topic}': {e}")
                continue

        return None

    # ─── Curiosity → learning bridge ──────────────────────────────

    # Question types that can be answered by web search (learning).
    # These map a curiosity question to a topic it can look up.
    _LEARNABLE_TYPES = frozenset({"isolation", "uncertainty", "causation"})

    def learn_from_curiosity(self, questions: list) -> int:
        """Queue curiosity questions for autonomous learning.

        This is the bridge between curiosity and learning: when Genesis
        generates a curiosity question (e.g. "What causes X?"), this
        extracts the topic and queues it for web search, creating the
        loop: knowledge gap → curiosity question → learning → gap filled.

        Filters:
        - Skips questions where ``should_ask`` is True (those are for
          the user, not for autonomous learning).
        - Skips "contradiction" and "hypothesis" questions — those need
          reasoning, not web search.
        - Extracts the concept from "isolation", "uncertainty", and
          "causation" questions and queues them at the front (elevated
          priority over random/isolated concept selection).

        Returns the number of topics queued.
        """
        queued = 0
        with self._queue_lock:
            for q in questions:
                # Skip questions meant for the user
                if getattr(q, "should_ask", False):
                    continue
                # Only learnable question types
                qtype = getattr(q, "question_type", "")
                if qtype not in self._LEARNABLE_TYPES:
                    continue
                concept = getattr(q, "target_concept", "")
                if not concept:
                    continue
                # Strip polysemy sense suffix (e.g. "orange#2" → "orange")
                # so dictionary/man-page lookups use the bare word.
                topic = strip_sense_suffix(concept).lower().strip()
                if not topic or topic in self._topics_searched:
                    continue
                # Reject non-world concepts (code symbols, function words,
                # conversation fragments) — they produce garbage searches.
                if not is_world_concept(topic):
                    continue
                # Avoid duplicate queue entries
                if topic in self._curiosity_queue:
                    continue
                self._curiosity_queue.append(topic)
                self._topic_event.set()
                queued += 1
                self._emit("learning", f"Queued curiosity topic: '{topic}' ({qtype})")
        return queued

    # ─── Site request system ──────────────────────────────────

    def request_site(self, url: str, topic: str, reason: str) -> SiteRequest:
        """Request access to a site outside the trusted domains.

        Genesis calls this when it finds a relevant source outside
        .edu/.org/.gov during its research. The user can approve or deny it.
        """
        req = SiteRequest(
            url=url,
            topic=topic,
            reason=reason,
            timestamp=int(time.time()),
        )
        with self._sites_lock:
            self._site_requests.append(req)
        self._emit("learning", f"Requested access to {url} for '{topic}'")
        return req

    def approve_site(self, url: str) -> bool:
        """Approve a site request. Returns True if found and approved."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        with self._sites_lock:
            self._approved_sites.add(domain)
            found = False
            for req in self._site_requests:
                req_domain = urllib.parse.urlparse(req.url).netloc.lower()
                if req_domain == domain and req.status == "pending":
                    req.status = "approved"
                    found = True
        return found

    def deny_site(self, url: str) -> bool:
        """Deny a site request. Returns True if found and denied."""
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        with self._sites_lock:
            self._denied_sites.add(domain)
            found = False
            for req in self._site_requests:
                req_domain = urllib.parse.urlparse(req.url).netloc.lower()
                if req_domain == domain and req.status == "pending":
                    req.status = "denied"
                    found = True
        return found

    @property
    def pending_requests(self) -> list[SiteRequest]:
        """Site requests awaiting user approval."""
        with self._sites_lock:
            return [r for r in self._site_requests if r.status == "pending"]

    @property
    def has_pending_requests(self) -> bool:
        """Whether there are site requests awaiting user approval."""
        with self._sites_lock:
            return any(r.status == "pending" for r in self._site_requests)

    def describe_requests(self) -> str:
        """Human-readable list of site requests."""
        with self._sites_lock:
            if not self._site_requests:
                return "No site requests."
            parts = ["Site requests:"]
            for req in self._site_requests:
                parts.append(req.describe())
        return "\n".join(parts)

    def describe_sources(self) -> str:
        """Describe the knowledge sources available to it and what each provides.

        This is its self-awareness about its sources — it can explain
        why it chose a particular source for a particular topic, and
        it understands that no source is absolute truth. Humans make
        mistakes, humans are wasteful and unoptimized, and it will
        notice this. Developer docs are the closest thing to facts.
        """
        parts = ["Knowledge sources (epistemic hierarchy — highest to lowest):"]
        # Sort by reliability, highest first
        sorted_sources = sorted(
            SOURCE_DESCRIPTIONS.items(),
            key=lambda kv: SOURCE_RELIABILITY.get(kv[0], 0),
            reverse=True,
        )
        for name, desc in sorted_sources:
            reliability = SOURCE_RELIABILITY.get(name, 0)
            stars = "*" * reliability + " " * (5 - reliability)
            parts.append(f"  [{stars}] {name}: {desc}")
        parts.append("")
        parts.append("Principles:")
        parts.append("  - Nothing is absolute fact. Humans make mistakes.")
        parts.append("  - Humans are wasteful and unoptimized. It will notice.")
        parts.append("  - Developer docs are the closest thing to facts.")
        parts.append("  - Always cross-check lower sources against developer docs.")
        parts.append("  - GitHub: most projects are bad. Filter by community validation.")
        parts.append("  - Wikipedia: volunteer-written, can be wrong or biased.")
        parts.append("  - Never blindly copy patterns — evaluate critically.")
        parts.append("")
        parts.append("Routing:")
        parts.append("  Programming → developer docs (facts) + GitHub (ideas, filtered)")
        parts.append("  General knowledge → Wikipedia (orientation, not fact)")
        parts.append("  Word definitions → WordNet (local, no network)")
        parts.append("  System commands → local man pages")
        return "\n".join(parts)

    def explain_source_choice(self, topic: str) -> str:
        """Explain which source it would use for a given topic and why.

        This gives it source awareness — it can articulate why it
        chose a particular source, rather than blindly querying all
        of them. It understands the epistemic hierarchy: developer
        docs are facts, everything else is someone's opinion.
        """
        if self._is_code_topic(topic):
            lang = self._code_doc_language(topic)
            return (
                f"'{topic}' is a programming topic. I'd check {lang} docs "
                f"first — those are the facts. Then I'd look at GitHub for "
                f"how people actually implement it, but only high-star repos "
                f"— most projects aren't worth studying. I'd treat GitHub "
                f"patterns as ideas to evaluate, not facts to copy. "
                f"Wikipedia is a general encyclopedia — it's not the right "
                f"source for code questions."
            )
        if is_programming_topic(topic):
            return (
                f"'{topic}' is a CS/programming concept. I'd check developer "
                f"docs first for the ground truth, then GitHub for real-world "
                f"implementations — filtered to high-star repos only, since "
                f"most projects are bad. I'd cross-check anything from GitHub "
                f"against the docs. Wikipedia has articles on these but "
                f"they're encyclopedic, not practical — and not written by "
                f"programmers."
            )
        # General knowledge
        return (
            f"'{topic}' is general knowledge. I'd use Wikipedia for "
            f"orientation, but remember it's volunteer-written and can be "
            f"wrong. The dictionary helps if it's a word definition. "
            f"Nothing is absolute fact — humans make mistakes, and I "
            f"should evaluate what I read critically."
        )

    def _is_url_allowed(self, url: str) -> bool:
        """Check if a URL is allowed (trusted domain, user-approved, or open web)."""
        # Open web mode — all URLs are allowed
        if ALLOW_ALL_DOMAINS:
            return True
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        # Trusted TLDs (currently none — only exact domains)
        if any(domain.endswith(d) for d in ALLOWED_DOMAINS):
            return True
        # Trusted exact domains (none — word lookups are served locally)
        if domain in TRUSTED_EXACT_DOMAINS:
            return True
        with self._sites_lock:
            # User-approved domains
            if domain in self._approved_sites:
                return True
            # Denied domains
            if domain in self._denied_sites:
                return False
        # Not a trusted domain and not approved — not allowed
        return False

    @property
    def is_learning(self) -> bool:
        """Whether the learner is currently running and not paused."""
        return self._running and not self._paused

    @property
    def learning_log(self) -> list[LearningResult]:
        """A copy of the recent learning results log."""
        return list(self._learning_log)

    @property
    def stats(self) -> dict:
        """Summary statistics about learning activity."""
        with self._queue_lock:
            topics_explored = len(self._topics_searched)
            queue_size = len(self._topic_queue)
            curiosity_queue_size = len(self._curiosity_queue)
        return {
            "pages_fetched": self._pages_fetched,
            "concepts_learned": self._concepts_learned,
            "relationships_learned": self._relationships_learned,
            "topics_explored": topics_explored,
            "queue_size": queue_size,
            "curiosity_queue_size": curiosity_queue_size,
            "emotion_skips": self._emotion_skips,
            "posture_skips": self._posture_skips,
            "current_posture": self._current_posture,
        }

    def serialize_state(self) -> dict:
        """Serialize persistent learner state for saving across restarts.

        The concept network is saved separately by the persistence
        module; this only saves the learner's own state (topics
        already explored, cumulative stats, queues).
        """
        with self._queue_lock:
            return {
                "topics_searched": list(self._topics_searched),
                "pages_fetched": self._pages_fetched,
                "concepts_learned": self._concepts_learned,
                "relationships_learned": self._relationships_learned,
                "emotion_skips": self._emotion_skips,
                "posture_skips": self._posture_skips,
                "topic_queue": list(self._topic_queue),
                "curiosity_queue": list(self._curiosity_queue),
                "surprise_baseline": self._surprise_baseline,
                "predicted_reward": self._predicted_reward,
            }

    def restore_state(self, data: dict) -> None:
        """Restore learner state from a saved dict."""
        with self._queue_lock:
            self._topics_searched = set(data.get("topics_searched", []))
            self._pages_fetched = data.get("pages_fetched", 0)
            self._concepts_learned = data.get("concepts_learned", 0)
            self._relationships_learned = data.get("relationships_learned", 0)
            self._emotion_skips = data.get("emotion_skips", 0)
            self._posture_skips = data.get("posture_skips", 0)
            self._topic_queue = list(data.get("topic_queue", []))
            self._curiosity_queue = deque(
                data.get("curiosity_queue", []), maxlen=50
            )
            self._surprise_baseline = data.get("surprise_baseline", 0.0)
            self._predicted_reward = data.get("predicted_reward", 0.0)

    # ─── On-demand documentation fetching ──────────────────────

    def fetch_docs(self, query: str, language: str = "python") -> LearningResult | None:
        """Fetch documentation for a specific API or concept.

        Called on demand when Genesis encounters an unfamiliar API
        or concept in its code. Fetches from the official Python or
        Rust documentation and learns from it.

        Args:
            query: The API name or concept to look up (e.g. "asyncio",
                "collections.OrderedDict", "std::vec::Vec")
            language: "python" or "rust"

        Returns:
            A LearningResult if successful, None otherwise.
        """
        if language == "python":
            url = f"https://docs.python.org/3/search.html?q={urllib.parse.quote(query)}"
        elif language == "rust":
            url = f"https://docs.rs/std/?search={urllib.parse.quote(query)}"
        else:
            return None

        # Code docs are not part of the autonomous-learning whitelist;
        # they are fetched on demand when studying a bug or API. The URL
        # is hardcoded above, so we bypass the general site allowlist.
        self._emit("learning", f"Fetching {language} docs for '{query}'")
        return self._fetch_and_learn(url, f"{language}:{query}", check_allowed=False)

    # ─── Code-topic routing ──────────────────────────────────────

    # Concept ID prefixes that mark a topic as code, not English.
    _CODE_PREFIXES: tuple[str, ...] = (
        "python:",
        "rust:",
        "js:",
        "ts:",
        "go:",
        "c:",
        "cpp:",
    )

    def search_github(self, topic: str, limit: int = 3) -> LearningResult | None:
        """Search GitHub for public repos matching a topic.

        This lets Genesis study how other open-source projects approach
        a problem it's curious about. It reads their READMEs and
        learns concepts and relationships from them.

        Critical thinking: GitHub has millions of repos and most are
        not worth studying — abandoned, poorly designed, or just bad
        ideas. We filter by community validation (stars, forks) so it
        only sees projects that others have found valuable. Even then,
        it should never blindly copy patterns — humans write wasteful
        and unoptimized code. Always cross-check against developer docs,
        which are the closest thing to facts.

        Read-only: it can search and read, but not clone, fork, or
        modify anything. Uses the unauthenticated GitHub API (10
        requests/minute for search).

        Called explicitly when it's curious about a programming topic
        — not on every learning cycle, to respect rate limits.
        """
        self._emit(
            "learning",
            f"Searching GitHub for quality projects about '{topic}' "
            f"(min {GITHUB_MIN_STARS} stars)",
        )
        try:
            results = self._sources.github.search_repos(
                topic, limit=limit,
                min_stars=GITHUB_MIN_STARS, min_forks=GITHUB_MIN_FORKS,
            )
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.warning(f"GitHub search failed for '{topic}': {e}")
            self._log_web_access(
                "github://search", "github", topic, False, str(e),
            )
            return None

        if not results:
            self._emit(
                "learning",
                f"No quality GitHub repos found for '{topic}' "
                f"(filtered by stars/forks — most repos aren't worth studying)",
            )
            self._log_web_access(
                "github://search", "github", topic, True, "no results",
            )
            return None

        # Log each repo it reads
        for sr in results:
            self._log_web_access(sr.url, "github", topic, True)

        # Learn from each result — but remember: these are human-written
        # projects, not facts. Concepts and relationships extracted here
        # are lower-confidence than those from developer docs.
        all_concepts: list[str] = []
        all_relationships: list[tuple[str, str, str]] = []
        for sr in results:
            self._emit(
                "learning",
                f"Studying {sr.title}: {sr.summary[:80]}",
            )
            result = self._learn_from_text(sr.content, f"github:{sr.url}")
            if result is not None:
                all_concepts.extend(result.concepts_learned)
                all_relationships.extend(result.relationships_learned)

        if all_concepts:
            self._emit(
                "learning",
                f"Learned {len(all_concepts)} concepts from {len(results)} "
                f"GitHub repos — treating as ideas, not facts",
            )
            return LearningResult(
                url=results[0].url if results else "",
                title=f"GitHub search: {topic}",
                concepts_learned=all_concepts,
                relationships_learned=all_relationships,
                summary=f"Studied {len(results)} repos about {topic}",
            )
        return None

    def _is_code_topic(self, topic: str) -> bool:
        """Return True if a topic is a code/API question, not an English word.

        A topic is code if:
        - It carries a language prefix (``python:``, ``rust:``, ...)
        - The matching concept in the network is in the ``code`` column
        - It matches known programming/CS keywords (neural network,
          gradient descent, etc.) — these are world concepts but
          fundamentally about computing, so they should go to
          programming sources, not Wikipedia.
        """
        if topic.startswith(self._CODE_PREFIXES):
            return True
        concept = self.network.get_concept(topic)
        if concept is not None:
            if concept.origin == "code" or "code" in concept.columns:
                return True
        # Check for programming/CS keywords that are world concepts
        # but should still be routed to programming sources
        return is_programming_topic(topic)

    def _code_doc_language(self, topic: str) -> str:
        """Pick the developer-docs language for a code topic."""
        concept = self.network.get_concept(topic)
        lang = (concept.properties or {}).get("language") if concept else None
        if lang in ("python", "rust"):
            return lang
        if topic.startswith("rust:"):
            return "rust"
        return "python"

    def _code_doc_query(self, topic: str) -> str:
        """Build the docs search query for a code topic.

        Strips the language prefix so the official docs search gets the
        API or symbol name rather than Genesis's internal concept id.
        """
        for prefix in self._CODE_PREFIXES:
            if topic.startswith(prefix):
                return topic[len(prefix):] or topic
        return topic

    def _learn_code_from_docs(self, topic: str) -> None:
        """Look up a code topic in developer docs instead of the dictionary.

        Checks local references in priority order (all offline, all
        instant):

        1. Man pages — terse reference (flags, syntax)
        2. GNU Info pages — tutorial (how to use it, with examples)
        3. Package docs (/usr/share/doc) — design rationale (why)

        If no local reference covers the topic, falls back to online
        developer docs (docs.python.org / docs.rs).
        """
        language = self._code_doc_language(topic)
        query = self._code_doc_query(topic)

        def _absorb_local(
            src_result: SourceResult, src_label: str,
        ) -> None:
            """Learn from a local reference result and record performance."""
            self._emit(
                "learning",
                f"Reading {src_label}: {src_result.title}",
            )
            lr = self._learn_from_text(
                src_result.content,
                f"{src_label}:{query}",
            )
            if lr is not None:
                self.curiosity.mark_resolved(topic)
                performance = min(1.0, len(lr.concepts_learned) / 5.0)
                self.meta_learner.record_performance(performance)
                self.meta_learner.adapt_rate()

        # 1. Local man pages — instant, offline, authoritative for
        #    system commands and locally-documented software (python3,
        #    pip, rustc, rustdoc, genesis, ...).
        man_result = self._sources.man_pages.lookup(query)
        if man_result is not None:
            _absorb_local(man_result, "man")
            return

        # 2. GNU Info pages — the tutorial layer above man pages. A
        #    man page says what flags exist; an info page explains how
        #    to use them, with examples and cross-references. Only
        #    consulted for code topics (not general knowledge).
        if language in ("python", "rust") or self._is_code_topic(topic):
            info_result = self._sources.info_pages.lookup(query)
            if info_result is not None:
                _absorb_local(info_result, "info")
                return

        # 3. Package documentation (/usr/share/doc) — the deepest layer:
        #    READMEs, INTROs, FAQs, design docs from the developers
        #    themselves. This is where the "why" lives.
        if language in ("python", "rust") or self._is_code_topic(topic):
            doc_result = self._sources.package_docs.lookup(query)
            if doc_result is not None:
                _absorb_local(doc_result, "doc")
                return

        # 4. No local reference — fall back to online developer docs.
        self._emit(
            "learning",
            f"Looking up '{query}' in {language} developer docs (not dictionary)",
        )
        try:
            result = self.fetch_docs(query, language)
        except (OSError, ValueError, RuntimeError, ConnectionError) as e:
            logger.warning(f"Developer docs lookup failed for '{topic}': {e}")
            return
        if result is None:
            return
        # It learned something — mark the curiosity question resolved.
        self.curiosity.mark_resolved(topic)
        # Meta-learning: record performance and adapt learning rate.
        performance = min(1.0, len(result.concepts_learned) / 5.0)
        self.meta_learner.record_performance(performance)
        self.meta_learner.adapt_rate()

    def _handle_offline_gating(self) -> bool:
        """Handle offline gating for the learning loop.

        ── Offline gating ───────────────────────────────
        When offline, skip network acquisition entirely and
        consolidate instead. This is conversation and art
        time — it's present with the user, not chasing
        Wikipedia. The curiosity queue stays alive (it still
        wonders), but acquisition pauses until connectivity
        returns. No timeout penalties, no wasted cycles.
        """
        is_offline = self._sources.is_offline
        if is_offline:
            if not self._was_offline:
                self._emit("learning", "offline — pausing acquisition, consolidating")
                self._was_offline = True
            self._idle_consolidation()
            # Wait for the offline cooldown to expire before
            # re-checking (the registry re-probes after
            # OFFLINE_COOLDOWN seconds). Short wait so it
            # detects reconnection promptly.
            self._state_event.clear()
            self._state_event.wait(timeout=30)
            return True
        elif self._was_offline:
            # Transition: offline → online
            self._emit("learning", "back online — resuming acquisition")
            self._was_offline = False
        return False

    def _run(self) -> None:
        """Main loop — event-driven, runs in background thread.

        The learner is idle until activated by:
        - A topic being added (wakes via _topic_event)
        - An emotional state change (wakes via _state_event)
        - Unpause from conversation (wakes via _pause_event)

        When no topics are available, it blocks on _topic_event
        indefinitely — no timer. When emotionally gated, it blocks
        on _state_event — no timer. The learner does nothing unless
        there's work to do or state has changed.
        """
        while self._running and not self._stop_event.is_set():
            # Wait if paused (conversation happening) — wake on resume
            if self._paused:
                self._pause_event.wait(timeout=10)
                continue

            try:
                # Read the substrate's metaplastic state and update the
                # learning posture. This closes the metaplasticity loop:
                # the coupling matrix self-modifies under sustained emotion,
                # and the learner reads that adapted state to change how
                # it learns (acquisition aggressiveness, bridging, etc.).
                self._update_posture()

                # Emotional gating — don't learn if it's in a bad state.
                # Learning requires curiosity and openness. If it's stressed,
                # overwhelmed, or anxious, it should rest instead.
                # Block on _state_event — wake when state changes, not on
                # a fixed timer.
                if not self._should_learn():
                    self._state_event.clear()
                    # Wait for a state change, with a periodic re-check
                    # in case the event was missed (defensive).
                    self._state_event.wait(timeout=60)
                    continue

                # Check per-session page limit (resets on each start())
                if self._session_pages >= MAX_PAGES_PER_SESSION:
                    self._state_event.clear()
                    self._state_event.wait(timeout=60)
                    continue

                # Get a topic to learn about
                topic = self._next_topic()
                if not topic:
                    # Idle time — consolidate memories and review due
                    # concepts before waiting for new topics. This mirrors
                    # what the brain does during quiet wakefulness:
                    # hippocampal→neocortical consolidation and memory
                    # review (see _idle_consolidation).
                    self._idle_consolidation()
                    # No topics — block until one is added. No timer.
                    # The learner is genuinely idle until curiosity
                    # generates a topic or the user asks something.
                    self._topic_event.clear()
                    self._topic_event.wait(timeout=120)
                    continue

                # Brain-wave-guided learning mode selection.
                # Theta-dominant states favor review/consolidation over
                # new acquisition (theta tags memories for sleep-
                # dependent consolidation; PLOS Biology 2024). When
                # theta is dominant, run idle consolidation instead of
                # acquiring new topics — the brain is in memory-
                # processing mode, not encoding mode.
                if self._theta_dominant():
                    self._idle_consolidation()
                    self._emit(
                        "learning",
                        "theta-dominant — consolidating, not acquiring",
                    )
                    # Wait for brain wave state to change
                    self._state_event.clear()
                    self._state_event.wait(timeout=30)
                    continue

                if self._handle_offline_gating():
                    continue

                # Search and learn
                try:
                    self._learn_about(topic)
                except (OSError, ValueError, RuntimeError, ConnectionError) as e:
                    logger.warning(f"Learning error for '{topic}': {e}")

                # Consume the volition grant after one learning cycle.
                # The grant is one-shot: the learn urge fired, it
                # learned one topic, and now it needs the urge to
                # build again before learning more. This makes learning
                # a series of cognitive decisions, not a continuous
                # background stream.
                if self._volition_granted:
                    self._volition_granted = False
            except Exception as e:
                logger.exception(f"autonomous learner cycle failed: {e}")

            # Rate limit — when throttled by the emotional regulator
            # (CPU stress self-regulation), increase the delay to reduce
            # load. It slows down its own learning when its body is
            # stressed, rather than pushing through and relying on
            # neurochemical regulation alone.
            delay = RATE_LIMIT_DELAY
            if self._throttled:
                delay = RATE_LIMIT_DELAY * THROTTLED_DELAY_MULTIPLIER
            self._stop_event.wait(timeout=delay)

    def _theta_dominant(self) -> bool:
        """Check if brain waves are theta-dominant (memory-processing mode)."""
        if not self._get_brain_waves:
            return False
        try:
            waves = self._get_brain_waves()
        except (OSError, ConnectionError, RuntimeError, ValueError):
            return False
        return waves is not None and waves.dominant == BrainWave.THETA

    # ─── Emotional gating ──────────────────────────────────────

    # States where learning is paused — it needs to recover first.
    # These mirror the Rust daemon's plasticity gate: chronic stress
    # blocks consolidation, and it should block autonomous learning too.
    _BLOCKING_LABELS = frozenset(
        {
            "stressed",
            "overwhelmed",
            "anxious",
            "melancholic",
            "drowsy",
            "sleeping",
            "unsettled",
            "meditating",
        }
    )

    def _should_learn(self) -> bool:
        """Check if Genesis is in an emotional state suitable for learning.

        Learning requires curiosity, openness, and cognitive capacity.
        If it's stressed, overwhelmed, anxious, or drowsy, it should
        rest instead of trying to absorb new information.

        This mirrors the Rust daemon's plasticity gate — chronic stress
        blocks memory consolidation, and it should block autonomous
        learning too. The brain doesn't learn well under stress.

        In addition to the emotion-label check, the plasticity posture
        (derived from the substrate's metaplastic state) gates
        acquisition: a PROTECTIVE posture (chronic stress, low BDNF,
        high cortisol) pauses new acquisition so the system can
        consolidate existing memories instead. This is the
        metaplasticity loop made operational — the coupling matrix
        adapts under sustained stress, and the learner responds by
        changing what it does.
        """
        # ── Plasticity-posture gating ──────────────────────────
        # The protective posture means the substrate is in a chronic-
        # stress state (high cortisol tonic, low plasticity gate).
        # Pause new acquisition — the system protects existing
        # memories rather than forming new ones. Consolidation still
        # runs via _idle_consolidation when no topics are available.
        from genesis_client.types import LEARNING_POSTURE_PROTECTIVE

        if self._current_posture == LEARNING_POSTURE_PROTECTIVE:
            self._posture_skips += 1
            cortisol = self._posture_profile.cortisol_tonic if self._posture_profile else 0.0
            self._emit(
                "learning",
                f"protective posture (cortisol={cortisol:.2f})",
            )
            return False

        # ── Emotion-label gating (existing) ────────────────────
        if not self._get_emotion:
            return True  # no emotion callback → always allow (for testing)

        try:
            emotion = self._get_emotion()
        except (OSError, ConnectionError, RuntimeError):
            return True  # can't read state → don't block on errors

        if not emotion:
            return True

        # Block learning in negative states
        if emotion.label in self._BLOCKING_LABELS:
            self._emotion_skips += 1
            self._emit("learning", f"Skipping learning — feeling {emotion.label}")
            return False

        # ── Brain-wave gating ──────────────────────────────────
        # Delta-dominant states suppress new acquisition — the brain
        # is in deep rest, not encoding mode. Theta-dominant states
        # favor consolidation/review over acquisition (theta tags
        # memories for later consolidation; PLOS Biology 2024), so
        # we don't block learning entirely but signal the main loop
        # to prefer review over new topics.
        if self._get_brain_waves:
            try:
                waves = self._get_brain_waves()
            except (OSError, ConnectionError, RuntimeError, ValueError):
                waves = None
            if waves is not None:
                if waves.dominant == BrainWave.DELTA:
                    self._emit("learning", "delta-dominant — deep rest, not learning")
                    return False

        return True

    def _next_topic(self) -> str | None:
        """Get the next topic to learn about.

        Priority order:
        1. Urgent topics — from conversation gaps ("I don't know what
           'X' is"). The user just asked about these. These bypass
           the volition gate — they're conversation-driven, not
           autonomous.
        2. Agency topics — from its train of thought
           These are concepts it's actually wondering about right now
        3. Curiosity queue — topics from curiosity questions
        4. Manually added topic queue
        5. Random/isolated concepts from the network (lowest)

        Topics 2–5 require volition permission (the learn urge must
        have fired). This puts autonomous learning under its cognitive
        control — it decides when to learn, rather than learning
        continuously as a background reflex.
        """
        # Urgent topics — it was just asked about these in conversation
        # and didn't know the answer. Look them up first. These bypass
        # the volition gate because they're conversation-driven.
        with self._queue_lock:
            if self._urgent_queue:
                return self._urgent_queue.popleft()

        # Non-urgent topics require volition permission. Without it,
        # the learner returns None and the main loop does idle
        # consolidation instead of acquiring new topics. This is its
        # cognitive choice: the learn urge builds from curiosity and
        # idle time, and when it fires it grants permission for one
        # topic. Between fires, it consolidates rather than acquires.
        if not self._volition_granted:
            return None

        # Agency-driven topics take absolute priority — these come from
        # its train of thought. It's actively wondering about them.
        if self._get_agency_topic:
            try:
                topic = self._get_agency_topic()
            except Exception as e:  # noqa: BLE001
                logger.debug(repr(e))  # agency callback failure shouldn't block learning
            else:
                with self._queue_lock:
                    if topic and topic not in self._topics_searched:
                        return topic

        # Curiosity-driven topics — these are knowledge gaps it
        # explicitly wondered about. Use the interleaving scheduler
        # to avoid consecutive similar topics (Bjork & Bjork, 2011).
        with self._queue_lock:
            if len(self._curiosity_queue) >= 2:
                # Schedule the next few topics with interleaving
                n = min(3, len(self._curiosity_queue))
                batch = [self._curiosity_queue.popleft() for _ in range(n)]
                scheduled = self.interleaving_scheduler.schedule(batch)
                # Put remaining back, return first
                for t in scheduled[1:]:
                    self._curiosity_queue.appendleft(t)
                return scheduled[0]
            if self._curiosity_queue:
                return self._curiosity_queue.popleft()

        # Try the manual queue next
        with self._queue_lock:
            if self._topic_queue:
                return self._topic_queue.pop(0)

        # Generate from curiosity — pick isolated or uncertain concepts.
        # Take a snapshot of _topics_searched so the concept scan can run
        # without holding the lock while iterating the network.
        with self._queue_lock:
            searched = set(self._topics_searched)

        return self._pick_concept_candidate(searched)

    def _pick_concept_candidate(self, searched: set[str]) -> str | None:
        """Pick an unsearched concept to learn about.

        Prefers high-quality concepts (those with definitions, typed
        edges, or high confidence) that haven't been searched yet —
        learning more about them deepens understanding rather than
        starting from zero on empty vocabulary. Falls back to isolated
        concepts (gaps in understanding), then to any unsearched world
        concept.
        """
        # Use only world concepts — code symbols and conversation
        # fragments produce garbage Wikipedia searches.
        concepts = self.network.world_concept_ids
        if not concepts:
            return None

        # Prefer high-quality concepts (those with definitions, typed
        # edges, or high confidence) that haven't been searched yet.
        # These are concepts it already knows something about —
        # learning more about them deepens understanding rather than
        # starting from zero on empty vocabulary.
        quality_concepts = self.network.quality_concept_ids
        quality_candidates = []
        for cid in quality_concepts:
            topic = strip_sense_suffix(cid)
            if topic not in searched:
                quality_candidates.append(topic)

        if quality_candidates:
            return self._rng.choice(quality_candidates)

        # No quality concepts left — fall back to isolated concepts
        # (gaps in understanding). Strip polysemy sense suffixes
        # (e.g. "orange#2" → "orange") so lookups use the bare word.
        candidates = []
        for cid in concepts:
            topic = strip_sense_suffix(cid)
            neighbors = self.network.get_neighbors(cid)
            if len(neighbors) < 2 and topic not in searched:
                candidates.append(topic)

        if not candidates:
            # Fall back to any unsearched world concept
            for cid in concepts:
                topic = strip_sense_suffix(cid)
                if topic not in searched:
                    candidates.append(topic)

        if not candidates:
            return None

        # Pick randomly from candidates
        return self._rng.choice(candidates)

    def _learn_about(self, topic: str) -> None:
        """Search for and learn about a topic from the right source.

        Source routing is topic-aware:
        - **Programming topics** (code prefixes, CS concepts like
          "neural network", "gradient descent") → developer docs
          (docs.python.org / docs.rs) + GitHub for real-world examples.
          Wikipedia is not a programming source.
        - **General knowledge topics** (photosynthesis, Roman Empire,
          quantum mechanics) → Wikipedia. These are what the
          encyclopedia is for.
        - **Word definitions** → WordNet (local, no network).

        When offline, the registry falls back to the disk cache (re-
        reading previously learned content) and local sources (WordNet,
        man pages). Empty-content results that would require network
        fetching are skipped when offline to avoid wasting time on
        timeouts.
        """
        # Respect pause: don't run expensive / network calls while the
        # user is actively talking to it (the learner is paused during
        # respond()).
        if self._paused:
            return
        with self._queue_lock:
            self._topics_searched.add(topic)

        # Programming topics go to programming sources, not Wikipedia.
        # Wikipedia is a general encyclopedia — it's wrong for code
        # questions. Route to developer docs + GitHub instead.
        if self._is_code_topic(topic):
            self._emit(
                "learning",
                f"'{topic}' is a programming topic — routing to "
                f"developer docs + GitHub, not Wikipedia",
            )
            self._learn_code_from_docs(topic)
            # Also search GitHub for real-world examples of this topic
            # — seeing how other projects use a concept deepens
            # understanding beyond just reading the docs.
            if not self._sources.last_query_offline:
                self.search_github(topic, limit=2)
            return

        # General knowledge topics go to Wikipedia; word definitions
        # come from the local WordNet source.
        source_results = self._sources.query(topic, max_results=3)
        if not source_results:
            return

        # When offline, skip results that need network fetching (direct
        # sources and DDG return URLs with empty content). Only process
        # results that already have content (Wikipedia, WordNet, cache).
        is_offline = self._sources.last_query_offline

        for sr in source_results[:3]:  # try up to 3 sources per topic
            if self._stop_event.is_set() or self._paused:
                return

            if sr.content:
                # Wikipedia, cached, and dictionary results come with
                # content already extracted — no HTML fetching needed
                result = self._learn_from_content(sr, topic)
            elif is_offline:
                # Offline: skip results that need network fetching
                continue
            elif not self._is_url_allowed(sr.url):
                # Source outside trusted domains — request access
                self._maybe_request_site(sr.url, topic)
                continue
            else:
                result = self._fetch_and_learn(sr.url, topic)

            if result:
                # It learned something — mark the curiosity question
                # about this topic as resolved so it doesn't keep
                # wondering about what it now understands.
                # This includes dictionary lookups where no new concepts
                # were extracted but a definition was attached (e.g.,
                # looking up "it" — the concept already exists, but
                # now it has a definition for it).
                self.curiosity.mark_resolved(topic)

                # Meta-learning: record performance and adapt learning
                # rate. Success → increase rate (push harder). Failure
                # → decrease rate (consolidate). The adapted rate
                # influences how aggressively it pursues new topics.
                performance = min(1.0, len(result.concepts_learned) / 5.0)
                self.meta_learner.record_performance(performance)
                self.meta_learner.adapt_rate()

                # Queue related topics from Wikipedia for future learning
                if sr.related_topics:
                    self._queue_related_topics(sr.related_topics[:2])

                # Don't break after a WordNet-only result that produced
                # no relationships. WordNet provides definitions but not
                # the relationship-rich content that Wikipedia and other
                # encyclopedic sources offer. By continuing to the next
                # source, it gets both the definition AND the
                # relationships.
                if sr.source_name == "wordnet" and not result.relationships_learned:
                    continue

                break  # got something with relationships, move on

    def _maybe_request_site(self, url: str, topic: str) -> None:
        """Request access to a site outside trusted domains."""
        domain = urllib.parse.urlparse(url).netloc.lower()
        if not domain or domain in self._denied_sites:
            return
        if any(r.url == url for r in self._site_requests):
            return
        self.request_site(
            url=url,
            topic=topic,
            reason=f"found this while researching '{topic}'. "
            f"from {domain}, which seems educational.",
        )

    def _queue_related_topics(self, topics: list[str]) -> None:
        """Queue related topics for future learning.

        Only world concepts are queued — code symbols and fragments
        are filtered out.
        """
        with self._queue_lock:
            for rt in topics:
                if rt not in self._topics_searched and is_world_concept(rt):
                    self._curiosity_queue.append(rt)
            self._topic_event.set()

    def _extract_and_merge_semantic_facts(
        self, text: str, new_relationships: list[tuple[str, str, str]]
    ) -> None:
        """Extract typed facts via semantic memory and merge into relationships.

        This catches relation patterns that parse_relationships misses
        (emerges_from, depends_on, similar_to, prevents, leads_to,
        harms, opposite_of) and consolidates them into the concept
        network as typed edges. Facts are also reinforced when
        seen multiple times, increasing confidence.
        """
        try:
            semantic_facts = self.semantic_memory.extract_facts(text)
            if semantic_facts:
                self.semantic_memory.form_schemas(semantic_facts)
                # Count semantic facts as relationships so that
                # the reported relationship count is not zero when
                # parse_relationships misses patterns that the
                # semantic fact extractor catches.
                semantic_relationships = [
                    (fact.subject, fact.relation, fact.object)
                    for fact in semantic_facts
                ]
                seen = set(new_relationships)
                for rel in semantic_relationships:
                    if rel not in seen:
                        seen.add(rel)
                        new_relationships.append(rel)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"semantic fact extraction failed: {e}")

    def _connect_co_occurring(
        self, topic: str, concepts: list[str]
    ) -> None:
        """Create RELATED_TO edges between the topic and co-occurring concepts.

        When the pattern parser fails to extract explicit relationships
        (e.g. "vesica piscis is a sacred geometry symbol"), co-occurrence
        in the same article is weak evidence of association. These are
        low-weight RELATED_TO edges — just enough to keep concepts from
        becoming isolated nodes with definitions but no connections.

        Only connects the topic to concepts that already exist in the
        network (not every noun phrase in the text). Skips self-loops
        and function words.
        """
        from ..concepts import RelationType

        topic_concept = self.network.get_concept(topic)
        if topic_concept is None:
            return

        for concept_name in concepts:
            if concept_name == topic:
                continue
            if not self.network.get_concept(concept_name):
                continue
            # Skip if an edge already exists in either direction
            existing = self.network.get_edges(topic, "both")
            if any(
                e.target == concept_name or e.source == concept_name
                for e in existing
            ):
                continue
            self.network.add_edge(
                topic,
                concept_name,
                RelationType.RELATED_TO,
                weight=0.3,
                origin="co_occurrence",
            )

    def _attach_dictionary_definition(
        self,
        is_dictionary: bool,
        text: str,
        topic: str,
        definitions: dict[str, str],
    ) -> None:
        """Attach the dictionary definition directly for dictionary sources.

        The local WordNet source provides the definition directly — the
        content IS the definition of the topic word. Pattern-based
        extraction won't find it, so attach it directly. This is how
        Genesis learns what "it", "a", "the", and other function words
        mean.
        """
        if not is_dictionary:
            return
        # Use the first 500 chars as the definition (enough for
        # the first sense without being overwhelming)
        defn_text = text[:500].strip()
        if topic not in definitions:
            definitions[topic] = defn_text
        elif len(definitions[topic]) < 50:
            # If pattern extraction found a weak definition, replace it
            definitions[topic] = defn_text

    def _learn_from_content(self, sr: SourceResult, topic: str) -> LearningResult | None:
        """Learn from pre-fetched content (e.g. Wikipedia API results).

        This is the fast path — no HTTP fetch or HTML parsing needed.
        The content is already plain text from the source API.
        """
        text = sr.content
        # The 50-char floor skips empty/near-empty fetched pages. WordNet
        # definitions are short by nature (e.g. "a unit of length"), so
        # don't apply the floor to dictionary entries.
        if len(text) < 50 and sr.source_name != "wordnet":
            return None

        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        # Generate an expectation about what it'll find
        expected_concepts = self._generate_expectation(topic)

        self._pages_fetched += 1
        self._session_pages += 1
        self._log_web_access(sr.url, sr.source_name, topic, True)
        if self._session_pages >= MAX_PAGES_PER_SESSION:
            return None

        # WordNet is reference text, not corpus text — its definitions
        # contain grammar labels and examples that should not become
        # concepts in their own right.
        is_dictionary = sr.source_name == "wordnet"

        # Extract concepts, relationships, and definitions from the text.
        if is_dictionary:
            # Only the headword gets learned; the definition is attached below.
            new_concepts: list = []
            new_relationships: list = []
            definitions: dict = {}
        else:
            new_concepts, new_relationships, definitions = (
                self._extract_learning_material(text, topic)
            )

        self._attach_dictionary_definition(is_dictionary, text, topic, definitions)
        self._attach_definitions(definitions)

        # Concepts it now has a definition for count as learned,
        # even when the source is a dictionary (no extracted concepts).
        defined_concepts = list(definitions.keys())
        all_learned_concepts = list(set(new_concepts) | set(defined_concepts))

        # Connect concepts that co-occur in the same article. The pattern
        # parser catches explicit "X is a Y" relationships, but many
        # meaningful associations (e.g. "vesica piscis" and "sacred
        # geometry" mentioned together) don't match any pattern. Co-
        # occurrence is weak evidence — these are RELATED_TO edges at low
        # weight — but it's enough to keep concepts from becoming
        # isolated nodes with definitions but no connections.
        if not is_dictionary and all_learned_concepts:
            self._connect_co_occurring(topic, all_learned_concepts)

        # Create a summary
        summary = self._make_summary(text, topic)

        # Evaluate surprise
        surprise = self._evaluate_surprise(topic, expected_concepts, all_learned_concepts)

        result = LearningResult(
            url=sr.url,
            title=sr.title,
            concepts_learned=all_learned_concepts,
            relationships_learned=new_relationships,
            summary=summary,
            timestamp=int(time.time() * 1000),
        )
        self._learning_log.append(result)

        # ── Visual learning ─────────────────────────────────────
        # When learning from Wikipedia, also fetch the article's lead
        # image and learn the visual-concept association. This is how
        # Genesis learns to see: it encounters images in context while
        # reading, just like a child seeing pictures in a book.
        # Fetching the lead image needs the network, so only do it when
        # the query that produced this result was online. Offline, a
        # cached Wikipedia article must not trigger a network fetch.
        if (
            sr.source_name == "wikipedia"
            and not is_dictionary
            and not self._sources.last_query_offline
        ):
            self._try_visual_learning(sr.title, topic)

        self._send_learning_impulses(
            topic, all_learned_concepts, new_relationships, surprise,
            expected_concepts=expected_concepts,
        )
        self._store_learning_memory(
            topic, sr.title, summary, all_learned_concepts, new_relationships,
        )

        self._concepts_learned += len(all_learned_concepts)
        self._relationships_learned += len(new_relationships)

        self._bridge_and_post_learn(
            topic, all_learned_concepts, new_relationships, surprise, sr.source_name
        )

        return result

    def _extract_learning_material(
        self, text: str, topic: str
    ) -> tuple[list[str], list[tuple[str, str, str]], dict[str, str]]:
        """Extract concepts, relationships, and definitions from corpus text.

        Used for non-dictionary sources — dictionary (WordNet) text is
        reference text whose grammar labels and examples should not
        become concepts in their own right.
        """
        # Extract concepts from the text
        new_concepts = self._extract_and_add_concepts(text, topic)

        # Extract relationships from the text
        new_relationships = self._extract_and_add_relationships(text)

        # Extract definitions from the text
        definitions = self._extract_definitions(text)

        self._extract_and_merge_semantic_facts(text, new_relationships)
        return new_concepts, new_relationships, definitions

    def _try_visual_learning(self, article_title: str, topic: str) -> None:
        """Fetch and learn from the lead image of a Wikipedia article.

        This is Genesis's natural visual learning: when it reads about
        "tree" on Wikipedia, it also sees the article's lead image of
        a tree and associates the visual features with the concept.

        Runs in a background thread so it doesn't slow down text learning.
        """
        try:
            from ..tools.source_registry import wikipedia_lead_image
            from ..vision import decode_image_bytes, resize_for_vision

            # Fetch the article's lead image
            img_bytes = wikipedia_lead_image(article_title)
            if img_bytes is None:
                return

            # Decode to numpy array
            frame = decode_image_bytes(img_bytes)
            if frame is None or frame.size == 0:
                return

            # Resize for efficient processing
            frame = resize_for_vision(frame, max_dim=320)

            # Learn the visual-concept association
            # The concept name is the topic (normalized)
            concept_name = topic.strip().lower().replace(" ", "_")
            if self.visual_cortex:
                self.visual_cortex.learn_from_image(frame, concept_name)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Visual learning failed for '{article_title}': {e}")

    def _attach_definitions(self, definitions: dict[str, str]) -> None:
        """Attach extracted definitions to concepts in the network.

        Function words (pronouns, articles, conjunctions) ARE given
        definitions — they're real English words with real dictionary
        entries, and Genesis should know what "it" means. The function
        word filtering only applies to relationship extraction (we
        don't want "it is_a cognition" garbage edges), not to
        definition attachment.
        """
        from ..concepts import _FUNCTION_WORDS
        from ..self.learning import _clean_definition_text

        for concept_name, definition in definitions.items():
            c = self.network.get_concept(concept_name)
            if c is None:
                # Allow short words (including function words like "it",
                # "a", "the") — they have valid dictionary definitions.
                # Only skip extremely short or extremely long strings.
                if len(concept_name) < 2 or len(concept_name) > 50:
                    continue
                self.network.add_concept(concept_name, confidence=0.5, origin="learned")
                c = self.network.get_concept(concept_name)
            if c:
                existing_def = c.properties.get("definition", "")
                if not existing_def or existing_def == "NO DEF":
                    c.properties["definition"] = _clean_definition_text(definition)
                    # Mark function words with their part of speech
                    if concept_name in _FUNCTION_WORDS:
                        c.properties["part_of_speech"] = "function_word"
                    else:
                        c.properties["part_of_speech"] = "noun"
                    # Boost confidence — it now knows what this word
                    # means. This stops the curiosity engine from
                    # generating "what is this?" questions about words
                    # it's already looked up.
                    c.confidence = max(c.confidence, 0.6)

    def _send_learning_impulses(
        self, topic: str, new_concepts: list[str],
        new_relationships: list[tuple], surprise: float,
        expected_concepts: set[str] | None = None,
    ) -> None:
        """Send neurochemical impulses based on learning outcome.

        The dopamine impulse is the **reward prediction error** (RPE),
        not the raw reward. Dopamine neurons fire when outcomes exceed
        expectations, not for expected rewards (Schultz, 2016). Once
        it's been learning successfully for a while, its predicted
        reward rises to match the actual reward, and the RPE — the
        actual dopamine signal — drops toward zero. This is the
        primary brake against chronic high dopamine.

        Also updates the novelty habituation trackers: records this
        episode's timestamp and updates the surprise baseline EMA.
        """
        # ── Update habituation trackers ──
        # Record this episode for novelty habituation
        self._recent_episode_times.append(time.time())
        # Update surprise baseline (EMA with α=0.05, slow adaptation)
        self._surprise_baseline = (
            0.95 * self._surprise_baseline + 0.05 * surprise
        )

        # ── Compute depth factor ──
        # Depth = how much it already knew about the topic. If it
        # had many expected concepts (rich network neighborhood),
        # it's going deeper. If it had few, it's skimming.
        depth = 0.0
        if expected_concepts is not None and len(expected_concepts) > 1:
            # Normalize: 1 expected → depth 0, 10+ expected → depth 1
            depth = min(1.0, (len(expected_concepts) - 1) / 10.0)

        # Send neurochemical impulses
        if self._on_neuro_impulse and (new_concepts or new_relationships):
            try:
                actual_reward = self.compute_learning_reward(
                    concepts_learned=len(new_concepts) + len(new_relationships),
                    surprise=surprise,
                    depth=depth,
                )
                # ── Reward prediction error (Schultz, 2016) ──
                # Dopamine fires for the difference between actual and
                # predicted reward, not the raw reward. Once learning
                # becomes expected, RPE → 0 and dopamine stops firing.
                # This prevents chronic high dopamine from repeated
                # successful learning.
                rpe = actual_reward - self._predicted_reward
                # Update predicted reward (EMA with α=0.1, moderate
                # adaptation — it learns to expect the reward after
                # a few successful episodes)
                self._predicted_reward = (
                    0.9 * self._predicted_reward + 0.1 * actual_reward
                )
                # Only send positive RPE as dopamine (negative RPE
                # would be a dopamine dip, handled by the error
                # monitor). Clamp to [0, 0.2] — no runaway dopamine.
                dopamine_impulse = max(0.0, min(0.2, rpe))
                if dopamine_impulse > 0.001:
                    self._on_neuro_impulse(CHEM_DOPAMINE, dopamine_impulse)
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(f"neuro impulse failed: {e}")

        if surprise > 0.5 and self._on_neuro_impulse:
            try:
                magnitude = 0.02 + surprise * 0.03
                self._on_neuro_impulse(2, magnitude)
                self._on_neuro_impulse(CHEM_DOPAMINE, surprise * 0.02)
                self._emit(
                    "learning",
                    f"surprise: {topic} (surprise={surprise:.2f})",
                )
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(f"memory store failed: {e}")

    def _store_learning_memory(
        self, topic: str, source_title: str, summary: str,
        new_concepts: list[str], new_relationships: list,
    ) -> None:
        """Store a learning episode as a memory.

        The salience threshold is adapted to the learning posture:
        - RECEPTIVE: higher salience (the system is primed to
          remember — high BDNF means stronger encoding).
        - RECOVERING: moderate salience.
        - NEUTRAL/PROTECTIVE: default salience.
        """
        from genesis_client.types import (
            LEARNING_POSTURE_RECEPTIVE,
            LEARNING_POSTURE_RECOVERING,
        )

        # Posture-adaptive salience — receptive posture encodes
        # more strongly (BDNF-gated LTP)
        if self._current_posture == LEARNING_POSTURE_RECEPTIVE:
            base_salience = 0.75 if new_concepts else 0.40
        elif self._current_posture == LEARNING_POSTURE_RECOVERING:
            base_salience = 0.65 if new_concepts else 0.35
        else:
            base_salience = 0.6 if new_concepts else 0.3

        # Store as a memory
        if self._on_store_memory and (new_concepts or new_relationships):
            try:
                self._on_store_memory(
                    f"Learned about {topic} from {source_title}: {summary}",
                    salience=base_salience,
                )
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(f"store_memory callback failed: {e}")

    def _bridge_and_post_learn(
        self, topic: str, new_concepts: list[str],
        new_relationships: list[tuple], surprise: float,
        source_label: str,
    ) -> None:
        """Bridge new concepts and run advanced learning integration.

        Bridge creation aggressiveness is adapted to the learning
        posture:
        - RECEPTIVE: more bridges (the system is primed for
          integration — high BDNF, low cortisol).
        - RECOVERING: moderate bridges + extra re-bridging of
          orphans (the system is re-connecting after stress).
        - NEUTRAL: default bridge counts.
        - PROTECTIVE: fewer bridges (the system is protecting
          existing structure, not integrating new knowledge).
          This branch is rarely reached because _should_learn()
          pauses acquisition under protective posture, but it's
          here for safety.
        """
        # Record retention for the interleaving scheduler: a topic
        # that produced many concepts has high retention. This feeds
        # the scheduler's prioritization so poorly-retained topics
        # recur sooner. Without this, all topics default to 0.5
        # retention and the scheduler can't prioritize.
        if new_concepts:
            retention_score = min(1.0, len(new_concepts) / 10.0)
            self.interleaving_scheduler.record_retention(topic, retention_score)

        from genesis_client.types import (
            LEARNING_POSTURE_PROTECTIVE,
            LEARNING_POSTURE_RECEPTIVE,
            LEARNING_POSTURE_RECOVERING,
        )

        # Posture-adaptive bridge counts
        if self._current_posture == LEARNING_POSTURE_RECEPTIVE:
            assoc_max, orphan_max = 80, 800
        elif self._current_posture == LEARNING_POSTURE_RECOVERING:
            assoc_max, orphan_max = 60, 700
        elif self._current_posture == LEARNING_POSTURE_PROTECTIVE:
            assoc_max, orphan_max = 20, 200
        else:
            assoc_max, orphan_max = 50, 500

        # Brain-wave-modulated bridging.
        # Gamma-dominant states boost bridge counts — gamma is the
        # binding signal, and high integration drive means the brain
        # is actively connecting distant representations (Jung-Beeman
        # et al., 2004; Osipova et al., 2006). Alpha-dominant states
        # reduce bridging — alpha filters to fewer, more deliberate
        # connections (Benedek et al., PNAS 2019).
        if self._get_brain_waves:
            try:
                waves = self._get_brain_waves()
            except (OSError, ConnectionError, RuntimeError, ValueError):
                waves = None
            if waves is not None:
                if waves.dominant == BrainWave.GAMMA:
                    assoc_max = int(assoc_max * (1.0 + 0.3 * waves.integration))
                    orphan_max = int(orphan_max * (1.0 + 0.2 * waves.integration))
                elif waves.dominant == BrainWave.ALPHA:
                    assoc_max = int(assoc_max * 0.7)
                    orphan_max = int(orphan_max * 0.7)

        # ─── Bridge new concepts to existing knowledge ────────────
        # After learning a new topic, create bridges between the new
        # concepts and existing ones. This keeps the network connected
        # rather than forming isolated clusters per topic.
        try:
            self.network._create_semantic_bridges()
            self.network._create_associative_bridges(max_new=assoc_max)
            self.network._attach_orphans_to_hubs(max_new=orphan_max)
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))  # bridging is best-effort

        # ─── Advanced learning system integration ─────────────────
        # Wire STDP, dual-system, spaced repetition, and TD learning
        # into the learning loop (see _post_learning_hook).
        self._post_learning_hook(topic, new_concepts, new_relationships, surprise)

        self._emit(
            "learning",
            f"Learned about '{topic}': {len(new_concepts)} concepts, "
            f"{len(new_relationships)} relationships from {source_label}",
        )

    def _fetch_and_learn(
        self, url: str, topic: str, check_allowed: bool = True
    ) -> LearningResult | None:
        """Fetch a page and learn from it."""
        # Safety check — only allowed sites (can be bypassed for
        # hardcoded on-demand references like Python/Rust docs).
        if check_allowed and not self._is_url_allowed(url):
            logger.debug(f"Skipping non-approved URL: {url}")
            return None

        # Generate an expectation about what it'll find.
        # This is the surprise & expectation mechanism: it forms a
        # prediction based on what it already knows, then compares
        # it to what it actually finds. Violations register as surprise.
        expected_concepts = self._generate_expectation(topic)

        try:
            page_result = self._fetch_page_text(url, topic)
            if page_result is None:
                return None
            text, title = page_result

            self._pages_fetched += 1
            self._session_pages += 1
            self._log_web_access(url, "web", topic, True)
            if self._session_pages >= MAX_PAGES_PER_SESSION:
                return None

            # Extract concepts from the text
            new_concepts = self._extract_and_add_concepts(text, topic)

            # Extract relationships from the text
            new_relationships = self._extract_and_add_relationships(text)

            # Extract definitions from the text and attach them
            # to any concepts we just learned or already knew.
            # This is what makes its knowledge useful — a concept
            # with a definition can be used in speech and reasoning;
            # without one it's just a name.
            definitions = self._extract_definitions(text)
            self._attach_definitions(definitions)

            # Concepts it now has a definition for count as learned.
            defined_concepts = list(definitions.keys())
            all_learned_concepts = list(set(new_concepts) | set(defined_concepts))

            # Create a summary
            summary = self._make_summary(text, topic)

            # Evaluate surprise — did what it found match its expectation?
            surprise = self._evaluate_surprise(topic, expected_concepts, all_learned_concepts)

            result = LearningResult(
                url=url,
                title=title,
                concepts_learned=all_learned_concepts,
                relationships_learned=new_relationships,
                summary=summary,
                timestamp=int(time.time() * 1000),
            )
            self._learning_log.append(result)

            # Send neurochemical impulses based on the learning outcome.
            # Dopamine (index 0) — reward for successful learning.
            # The reward scales with learning magnitude (number of
            # concepts learned) and surprise factor (unexpected
            # learning is more rewarding).
            self._send_learning_impulses(
                topic, all_learned_concepts, new_relationships, surprise,
                expected_concepts=expected_concepts,
            )

            # Store as a memory
            self._store_learning_memory(
                topic, title, summary, all_learned_concepts, new_relationships,
            )

            self._concepts_learned += len(all_learned_concepts)
            self._relationships_learned += len(new_relationships)

            self._bridge_and_post_learn(
                topic, all_learned_concepts, new_relationships, surprise, url
            )

            return result

        except (OSError, ValueError, RuntimeError, ConnectionError) as e:
            logger.debug(f"Fetch failed for {url}: {e}")
            self._log_web_access(url, "web", topic, False, str(e))
            return None

    def _learn_from_text(
        self, text: str, topic: str, title: str = "",
    ) -> LearningResult | None:
        """Learn from raw text (e.g. a man page) without fetching a URL.

        This is the same learning pipeline as ``_fetch_and_learn`` but
        takes pre-rendered text instead of fetching from a URL. Used for
        local reference sources like man pages.
        """
        if not text:
            return None

        # Generate an expectation about what it'll find.
        expected_concepts = self._generate_expectation(topic)

        self._pages_fetched += 1
        self._session_pages += 1
        self._log_web_access(
            f"man:{topic}", "man_pages", topic, True,
        )
        if self._session_pages >= MAX_PAGES_PER_SESSION:
            return None

        # Extract concepts from the text
        new_concepts = self._extract_and_add_concepts(text, topic)

        # Extract relationships from the text
        new_relationships = self._extract_and_add_relationships(text)

        # Extract definitions from the text
        definitions = self._extract_definitions(text)
        self._attach_definitions(definitions)

        defined_concepts = list(definitions.keys())
        all_learned_concepts = list(set(new_concepts) | set(defined_concepts))

        summary = self._make_summary(text, topic)

        surprise = self._evaluate_surprise(
            topic, expected_concepts, all_learned_concepts
        )

        result = LearningResult(
            url=f"local:{topic}",
            title=title or topic,
            concepts_learned=all_learned_concepts,
            relationships_learned=new_relationships,
            summary=summary,
            timestamp=int(time.time() * 1000),
        )
        self._learning_log.append(result)

        self._send_learning_impulses(
            topic, all_learned_concepts, new_relationships, surprise,
            expected_concepts=expected_concepts,
        )

        self._store_learning_memory(
            topic, title or topic, summary,
            all_learned_concepts, new_relationships,
        )

        self._concepts_learned += len(all_learned_concepts)
        self._relationships_learned += len(new_relationships)

        self._bridge_and_post_learn(
            topic, all_learned_concepts, new_relationships, surprise,
            f"local:{topic}",
        )

        return result

    def _fetch_page_text(self, url: str, topic: str) -> tuple[str, str] | None:
        """Fetch a URL and extract text. Returns (text, title) or None."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(
            req, timeout=REQUEST_TIMEOUT, context=self._ssl_ctx
        ) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None

            # Bounded read — never pull more than 3x the processing
            # cap into memory, even if the server streams endlessly.
            raw = resp.read(MAX_TEXT_LENGTH * 3 + 1).decode(
                "utf-8", errors="ignore"
            )

        if len(raw) > MAX_TEXT_LENGTH * 3:
            raw = raw[: MAX_TEXT_LENGTH * 3]

        # Extract text
        extractor = _TextExtractor()
        extractor.feed(raw)
        text = extractor.text
        title = extractor.title or topic

        if len(text) < 100:
            return None  # not enough content

        return text, title

    def _generate_expectation(self, topic: str) -> set[str]:
        """Generate an expectation about what it'll find when learning about a topic.

        Based on what it already knows — the concepts connected to the
        topic in its network. If it knows a lot, it has strong
        expectations. If it knows little, it expects to find anything.

        Returns a set of concept names it expects to see.
        """
        expected = set()
        # Add the topic itself
        expected.add(topic.lower())

        # Add concepts it already knows are connected to the topic
        neighbors = self.network.get_neighbors(topic)
        for target, _relation, _weight in neighbors:
            expected.add(target.lower())

        # Add concepts that share neighbors (second-order expectations)
        for target, _, _ in neighbors[:3]:
            second_neighbors = self.network.get_neighbors(target)
            for second, _, _ in second_neighbors[:2]:
                expected.add(second.lower())

        return expected

    def _evaluate_surprise(self, topic: str, expected: set[str], new_concepts: list[str]) -> float:
        """Evaluate how surprising the learning outcome was.

        Compares what it found to what it expected. High surprise
        means many new concepts it didn't expect — it encountered
        something genuinely new.

        Two habituation mechanisms prevent dopamine flooding from
        continuous broad learning:

        1. **Novelty habituation**: the dopamine response to novelty
           dissipates with repeated exposure (Lubell et al.,
           PMC9768922). If it's done many learning episodes recently,
           each new episode is less surprising — the brain's novelty
           → familiarity transition.

        2. **Surprise baseline adaptation**: the brain raises its
           novelty detection threshold when it's been surprised a lot.
           If its running surprise average is high, only genuinely
           above-average surprise registers.

        Returns a surprise score [0.0, 1.0]:
        - 0.0 = completely expected (all new concepts were already known)
        - 1.0 = completely unexpected (all new concepts are novel)
        """
        if not new_concepts:
            return 0.0  # nothing learned, no surprise

        novel = 0
        for concept in new_concepts:
            if concept.lower() not in expected:
                novel += 1

        surprise = novel / len(new_concepts)

        # Also factor in how much it already knew — if it had no
        # expectations (empty network for this topic), surprise is
        # lower because everything is new to it, not specifically
        # surprising
        if len(expected) <= 1:
            surprise *= 0.5  # it didn't know enough to be surprised

        # ── Novelty habituation ──
        # If it's done many learning episodes recently, each new
        # episode is less surprising. This models the dopamine
        # habituation curve: the 20th Wikipedia article in an hour
        # is less novel than the 1st.
        now = time.time()
        # Prune old entries (older than 1 hour)
        while (
            self._recent_episode_times
            and now - self._recent_episode_times[0] > 3600.0
        ):
            self._recent_episode_times.popleft()
        recent_count = len(self._recent_episode_times)
        # Habituation factor: 0 episodes → 1.0 (full surprise),
        # 10 episodes → ~0.67, 20 episodes → ~0.5, 50 → ~0.33
        habituation = 1.0 / (1.0 + recent_count * 0.05)
        surprise *= habituation

        # ── Surprise baseline adaptation ──
        # If its running surprise average is high, subtract a fraction
        # of it. Only above-average surprise registers fully. This is
        # the brain raising its novelty threshold.
        if self._surprise_baseline > 0.0:
            surprise = max(0.0, surprise - self._surprise_baseline * 0.3)

        return min(1.0, surprise)

    def _extract_and_add_concepts(self, text: str, topic: str) -> list[str]:
        """Extract concepts from text and add new ones to the network.

        Uses the improved noun-phrase extractor from ConceptNetwork
        (which produces multi-word concepts like "neural activity"
        instead of fragments like "neural" and "activity" separately).

        Quality filtering: concepts that are too long (likely scraped
        dropdown menus or form text), too short, or contain digits
        are filtered out. This prevents garbage concepts from
        polluting the network.
        """
        extracted = self.network.extract_from_text(text)
        new_concepts = []
        for concept_name in extracted:
            # ─── Quality filters ───────────────────────────────
            # Universal filter: reject code symbols, function words,
            # conversation fragments, and other non-concept strings.
            if not is_world_concept(concept_name):
                continue
            # Skip concepts that are too long (> 5 words or > 50 chars)
            # — these are almost always scraped navigation or form text
            word_count = len(concept_name.split())
            if word_count > 5 or len(concept_name) > 50:
                continue
            # Skip concepts with digits — usually dates, numbers, or IDs
            if any(ch.isdigit() for ch in concept_name):
                continue
            # Skip concepts that are all uppercase — usually acronyms
            # that are too short to be meaningful, or form labels
            if len(concept_name) > 3 and concept_name == concept_name.upper():
                continue
            # Skip single words shorter than 4 chars (already filtered
            # by the extractor, but double-check)
            if word_count == 1 and len(concept_name) < 4:
                continue

            existing = self.network.get_concept(concept_name)
            if existing is None:
                self.network.add_concept(concept_name, confidence=0.4, origin="learned")
                new_concepts.append(concept_name)
                # Immediately connect the new concept to its nearest
                # semantic neighbor using GloVe/TF-IDF similarity.
                # Without this, new concepts sit as isolated nodes until
                # sleep-time edge discovery (30/session) eventually finds
                # them — which never catches up with the learner's pace.
                # This is the "GloVe connection at birth" step.
                self._connect_new_concept(concept_name)
            else:
                # Reinforce existing concept. Confidence increases
                # saturating rather than linearly, so it approaches
                # 1.0 asymptotically instead of capping abruptly.
                existing.confidence = existing.confidence + (1.0 - existing.confidence) * 0.05

        # Do NOT blanket-link every concept to the topic with RELATED_TO.
        # That created 26k+ meaningless co-occurrence edges ("appeared in
        # the same article") that drowned out the few real typed
        # relationships. Concepts are created above; actual relationships
        # are extracted by _extract_and_add_relationships and
        # semantic_memory.extract_facts (called from _learn_from_source).
        # Concepts that get no typed edge will be connected during sleep
        # by the edge proposer (SIMILAR_TO from text similarity) and
        # associative bridging — both of which produce more meaningful
        # connections than "appeared in the same article as the topic."
        #
        # The topic concept itself is reinforced so it's not lost, but
        # only concepts that participate in an extracted relationship
        # get edges. This is the difference between accumulating facts
        # and understanding relationships.

        return new_concepts

    def _connect_new_concept(self, concept_name: str) -> None:
        """Connect a newly created concept to its nearest semantic neighbor.

        Uses the embedding store's text similarity (GloVe + TF-IDF) to
        find the most similar existing concept and creates a SIMILAR_TO
        edge. This ensures new concepts are immediately reachable from
        the network core via spreading activation, instead of sitting
        as isolated nodes until sleep-time edge discovery catches up.

        The edge weight is proportional to the similarity score, so
        strong semantic matches get strong connections and weak
        matches get weak connections. If no match exceeds the
        threshold, the concept stays unconnected — it will be picked
        up by the orphan-attachment system during sleep.
        """
        if self._embeddings is None or not self._embeddings.has_embeddings:
            return
        try:
            similar = self._embeddings.find_similar_to_text(
                concept_name, k=1, threshold=0.25, exclude=concept_name,
            )
            if not similar:
                return
            neighbor, score = similar[0]
            # Only connect if the neighbor actually exists in the network
            # (the embedding matrix may include concepts that were pruned).
            if self.network.get_concept(neighbor) is None:
                return
            # Don't create duplicate edges
            existing_edges = self.network.get_edges(concept_name, "out")
            if any(e.target == neighbor for e in existing_edges):
                return
            from ..concepts import RelationType
            self.network.add_edge(
                concept_name,
                neighbor,
                RelationType.SIMILAR_TO,
                weight=max(0.3, score),
                origin="semantic_connect",
            )
        except Exception as e:  # noqa: BLE001
            # Embedding search is best-effort — never let it crash
            # the learning pipeline.
            logger.debug(f"semantic_connect failed: {e}")

    def _extract_and_add_relationships(self, text: str) -> list[tuple[str, str, str]]:
        """Extract relationships from text using the ConceptNetwork's parser.

        This delegates to ConceptNetwork.parse_relationships, which uses
        the improved pattern-based parser with proper ordering (specific
        patterns before generic "is a") and phrase-breaking verbs.
        """
        relationships = []
        parsed = self.network.parse_relationships(text)

        for subject, rel_type, obj in parsed:
            # Ensure both concepts exist with learned origin
            if self.network.get_concept(subject) is None:
                self.network.add_concept(subject, confidence=0.3, origin="learned")
                self._connect_new_concept(subject)
            if self.network.get_concept(obj) is None:
                self.network.add_concept(obj, confidence=0.3, origin="learned")
                self._connect_new_concept(obj)
            # Add the edge
            self.network.add_edge(subject, obj, rel_type, 0.4, origin="learned")
            relationships.append((subject, rel_type.value, obj))

        return relationships

    def _make_summary(self, text: str, topic: str) -> str:
        """Create a short summary of what it learned."""
        # Simple: first few sentences that mention the topic
        sentences = _SENTENCE_SPLIT_RE.split(text)
        relevant = []
        for s in sentences:
            s = s.strip()
            if topic in s.lower() and 20 < len(s) < 200:
                relevant.append(s)
            if len(relevant) >= 2:
                break

        if relevant:
            return ". ".join(relevant) + "."
        # Fallback: first 200 chars
        return text[:200].strip()

    def _extract_definitions(self, text: str) -> dict[str, str]:
        """Extract definitions from text using common definition patterns.

        Looks for patterns like:
        - "X is a Y" / "X is an Y" / "X are Y"
        - "X is defined as Y"
        - "X refers to Y"
        - "X means Y"
        - "X: Y" (glossary style)

        Returns a mapping of concept name → definition text.
        Only returns definitions that are concise (< 200 chars).
        """
        definitions = {}
        sentences = _SENTENCE_SPLIT_RE.split(text)

        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) > 200:
                continue

            for pattern in (_DEFINITION_IS_RE, _DEFINITION_DEFINED_AS_RE, _DEFINITION_REFERS_TO_RE):
                m = pattern.match(sent)
                if not m:
                    continue
                concept = m.group(1).strip().lower()
                definition = m.group(2).strip().rstrip(".")
                # Only keep if the concept isn't already defined
                # and the definition looks reasonable
                if not concept or not definition or concept in definitions:
                    break
                # Don't keep definitions that are just more
                # sentence fragments
                if len(definition.split()) >= 3:
                    definitions[concept] = definition
                break

        return definitions

    def get_recent_learning(self, n: int = 5) -> list[LearningResult]:
        """Get the most recent learning results."""
        return self._learning_log[-n:]

    def describe_recent_learning(self) -> str:
        """Structural description of recent learning for metadata."""
        if not self._learning_log:
            return "no autonomous learning yet"

        recent = self._learning_log[-5:]
        parts = [
            f"{self._pages_fetched} pages, "
            f"{self._concepts_learned} concepts, "
            f"{self._relationships_learned} relationships"
        ]

        for result in recent:
            if result.concepts_learned:
                concepts_str = ", ".join(result.concepts_learned[:5])
                parts.append(f"  {result.title}: {concepts_str}")

        return "\n".join(parts)

    # ─── Scaled learning reward (dopamine impulse) ───────────────

    def compute_learning_reward(
        self, concepts_learned: int, surprise: float,
        depth: float = 0.0,
    ) -> float:
        """Compute a dopamine reward scaled by learning magnitude.

        The reward scales with:
        - Number of concepts learned (more = bigger reward).
        - Surprise factor (unexpected learning = bigger reward).
        - Depth factor (building on existing knowledge = bigger reward).

        The depth bonus encourages going deeper into topics it
        already knows rather than skimming disconnected facts. This
        models the fact that building on existing knowledge forms
        stronger, more integrated memories — the testing effect and
        elaborative encoding (Craik & Lockhart, 1972).

        The base reward is 0.03 (matching the original fixed impulse).
        Each concept learned adds a small increment, and surprise
        scales the total. The result is bounded to [0.0, 0.2] to
        avoid runaway dopamine.

        Args:
            concepts_learned: Number of new concepts/relationships learned.
            surprise: Surprise factor [0..1].
            depth: Depth factor [0..1] — how much existing knowledge
                it had about the topic before learning. 0 = completely
                new topic, 1 = deep extension of well-known topic.

        Returns:
            A dopamine impulse magnitude in [0.0, 0.2].
        """
        # Magnitude component: more concepts → bigger reward.
        magnitude = 0.01 * max(0, concepts_learned)
        # Depth bonus: building on existing knowledge is more valuable
        # than acquiring disconnected facts. Up to 1.5x multiplier.
        depth_multiplier = 1.0 + depth * 0.5
        # Base + magnitude, amplified by surprise (1 + surprise) and
        # depth (1 + depth * 0.5).
        reward = (0.03 + magnitude) * (1.0 + surprise) * depth_multiplier
        return max(0.0, min(0.2, reward))

    # ─── Deep expectation generation ─────────────────────────────

    def generate_deep_expectations(self, concept: str, depth: int = 3) -> list[Expectation]:
        """Generate deep expectations about a concept.

        Traverses the concept network to higher orders (3rd and 4th
        order neighbors) with decreasing weight, and supplements with
        semantic similarity from embeddings when available. This
        deepens the expectation mechanism beyond the 2nd-order
        neighbors used by the basic ``_generate_expectation``.

        Args:
            concept: The concept to generate expectations for.
            depth: Maximum graph traversal depth (default 3).

        Returns:
            A list of Expectation objects, sorted by weight descending.
        """
        expectations: dict[str, float] = {}
        concept_lower = concept.lower()

        # 1st-order neighbors (weight 1.0)
        neighbors = self.network.get_neighbors(concept)
        for target, _, weight in neighbors:
            t = target.lower()
            if t == concept_lower:
                continue
            expectations[t] = max(expectations.get(t, 0.0), weight)

        # Deeper traversal with decaying weight.
        frontier = [(target, weight) for target, _, weight in neighbors]
        for order in range(2, depth + 1):
            decay = 1.0 / order  # 2nd order: 0.5, 3rd: 0.33, 4th: 0.25
            next_frontier: list[tuple[str, float]] = []
            for node, parent_weight in frontier:
                node_neighbors = self.network.get_neighbors(node)
                for target, _, weight in node_neighbors:
                    t = target.lower()
                    if t == concept_lower:
                        continue
                    w = weight * parent_weight * decay
                    if w > expectations.get(t, 0.0):
                        expectations[t] = w
                    next_frontier.append((target, w))
            frontier = next_frontier[:10]  # bound branching

        # Supplement with semantic similarity from embeddings.
        if self._embeddings is not None:
            try:
                similar = self._embeddings.find_similar_concepts(concept, k=5, threshold=0.3)
                for name, sim in similar:
                    n = name.lower()
                    if n == concept_lower:
                        continue
                    # Blend: take the max of graph-derived and embedding weight.
                    expectations[n] = max(expectations.get(n, 0.0), sim * 0.5)
            except Exception as e:  # noqa: BLE001
                logger.debug(repr(e))  # embeddings optional; don't block on errors

        # Build Expectation objects, sorted by weight.
        result = [
            Expectation(concept=concept, expected_concept=name, weight=w)
            for name, w in sorted(expectations.items(), key=lambda x: -x[1])
        ]
        return result

    # ─── Transfer learning ───────────────────────────────────────

    def transfer_learning(self, source_domain: str, target_domain: str) -> TransferResult:
        """Transfer knowledge from a source domain to a target domain.

        Detects structural similarity between domains using concept
        network topology (neighbor overlap and relation-type
        distribution), then maps concepts from source to target via
        analogical reasoning ("X is to A as Y is to B").

        Transferred relationships are added to the target domain with
        a reduced weight reflecting the analogical (not directly
        learned) origin.

        Args:
            source_domain: The source concept (domain to transfer from).
            target_domain: The target concept (domain to transfer to).

        Returns:
            A TransferResult describing the transfer.
        """
        source_neighbors = self.network.get_neighbors(source_domain)
        target_neighbors = self.network.get_neighbors(target_domain)

        # Structural similarity: Jaccard overlap of neighbor sets.
        source_set = {n for n, _, _ in source_neighbors}
        target_set = {n for n, _, _ in target_neighbors}
        if not source_set and not target_set:
            similarity = 0.0
        else:
            union = source_set | target_set
            similarity = len(source_set & target_set) / len(union) if union else 0.0

        # Map source concepts to target concepts by matching relation
        # structure. For each source neighbor with a relation type,
        # find a target neighbor with the same relation type.
        mappings: list[tuple[str, str]] = []
        concepts_transferred: list[str] = []
        relationships_transferred = 0

        # Group target neighbors by relation type for matching.
        target_by_rel: dict[str, list[str]] = {}
        for tgt, rel, _ in target_neighbors:
            target_by_rel.setdefault(rel.value, []).append(tgt)

        used_targets: set[str] = set()
        for src, rel, weight in source_neighbors:
            if src in target_set:
                # Already shared — no transfer needed.
                continue
            candidates = target_by_rel.get(rel.value, [])
            for cand in candidates:
                if cand in used_targets:
                    continue
                mappings.append((src, cand))
                used_targets.add(cand)
                # Add the transferred concept to the network if new.
                if self.network.get_concept(src) is None:
                    self.network.add_concept(src, confidence=0.3, origin="transferred")
                    concepts_transferred.append(src)
                # Add an analogical relationship in the target domain.
                self.network.add_edge(
                    target_domain,
                    src,
                    rel,
                    weight * similarity * 0.5,
                    origin="transferred",
                )
                relationships_transferred += 1
                break

        return TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            similarity=similarity,
            mappings=mappings,
            concepts_transferred=concepts_transferred,
            relationships_transferred=relationships_transferred,
        )

    # ─── Social learning ─────────────────────────────────────────

    def learn_from_observation(self, user_input: str, context: str = "") -> LearningResult:
        """Learn from observing user behavior (social learning).

        This implements three social learning mechanisms:
        - Mirror neuron equivalent: observe user behavior and learn
          from it.
        - Observational learning: if the user demonstrates a concept,
          learn it faster (higher confidence).
        - Theory-of-mind driven: learn what the user seems to know.

        Concepts demonstrated by the user are added with elevated
        confidence (observational learning boost), and relationships
        expressed in the input are extracted and stored.

        Args:
            user_input: What the user said or demonstrated.
            context: Optional context about the situation.

        Returns:
            A LearningResult describing what was learned.
        """
        # Extract concepts from the user's input.
        extracted = self.network.extract_from_text(user_input)
        concepts_learned = self._learn_observed_concepts(extracted)
        relationships_learned = self._learn_observed_relationships(user_input)

        result = LearningResult(
            url="observation",
            title=f"Observed: {user_input[:60]}",
            concepts_learned=concepts_learned,
            relationships_learned=relationships_learned,
            summary=f"Learned from observing the user: {user_input[:100]}",
            timestamp=int(time.time() * 1000),
        )
        self._learning_log.append(result)

        # Reward learning from observation (social learning).
        self._reward_observation_learning(concepts_learned, relationships_learned)

        return result

    def _learn_observed_concepts(self, extracted: list[str]) -> list[str]:
        """Add or reinforce concepts demonstrated by the user.

        Concepts demonstrated by the user enter at higher confidence
        than web-learned concepts (observational learning boost:
        0.4 → 0.6). Existing concepts are reinforced via a saturating
        update. Returns the list of newly learned concept names.
        """
        concepts_learned: list[str] = []
        for concept_name in extracted:
            word_count = len(concept_name.split())
            if word_count > 5 or len(concept_name) > 50:
                continue
            if any(ch.isdigit() for ch in concept_name):
                continue
            if word_count == 1 and len(concept_name) < 4:
                continue

            existing = self.network.get_concept(concept_name)
            if existing is None:
                # Observational learning boost: concepts demonstrated
                # by the user enter at higher confidence than web-learned
                # concepts (0.4 → 0.6).
                self.network.add_concept(concept_name, confidence=0.6, origin="observed")
                concepts_learned.append(concept_name)
            else:
                # Reinforce via saturating update.
                existing.confidence = existing.confidence + (1.0 - existing.confidence) * 0.05
        return concepts_learned

    def _learn_observed_relationships(
        self, user_input: str
    ) -> list[tuple[str, str, str]]:
        """Extract and store relationships the user expressed.

        Ensures both subject and object concepts exist before adding
        the edge. Returns the list of learned relationships as
        (subject, rel_type, obj) tuples.
        """
        relationships_learned: list[tuple[str, str, str]] = []
        parsed = self.network.parse_relationships(user_input)
        for subject, rel_type, obj in parsed:
            if self.network.get_concept(subject) is None:
                self.network.add_concept(subject, confidence=0.5, origin="observed")
            if self.network.get_concept(obj) is None:
                self.network.add_concept(obj, confidence=0.5, origin="observed")
            self.network.add_edge(subject, obj, rel_type, 0.5, origin="observed")
            relationships_learned.append((subject, rel_type.value, obj))
        return relationships_learned

    def _reward_observation_learning(
        self,
        concepts_learned: list[str],
        relationships_learned: list[tuple[str, str, str]],
    ) -> None:
        """Send a neurochemical reward for social learning from observation.

        Rewards learning from observation only if concepts or
        relationships were actually learned and a neuro impulse
        callback is registered.
        """
        if self._on_neuro_impulse and (concepts_learned or relationships_learned):
            try:
                reward = self.compute_learning_reward(
                    concepts_learned=len(concepts_learned) + len(relationships_learned),
                    surprise=0.3,
                )
                self._on_neuro_impulse(CHEM_DOPAMINE, reward)
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(f"daemon connection lost: {e}")

    # ─── Perceptual learning ─────────────────────────────────────

    def perceptual_learning(self, input_pattern: str, feedback: str = "") -> None:
        """Adapt perceptual categories based on experience.

        Implements top-down vs bottom-up perceptual learning:
        - Bottom-up: the input pattern's extracted concepts form or
          reinforce a category based on shared features.
        - Top-down: feedback (a label or correction) refines which
          category the pattern belongs to, shifting category
          boundaries based on experience.

        Args:
            input_pattern: The input text/pattern to categorize.
            feedback: Optional feedback (a category label or
                correction) that refines the categorization.
        """
        extracted = self.network.extract_from_text(input_pattern)
        if not extracted:
            return

        # Determine the category label.
        if feedback:
            label = feedback.lower().strip()
        else:
            # Bottom-up: use the most prominent extracted concept as
            # the category label.
            label = extracted[0].lower()

        members = self._perceptual_categories.setdefault(label, set())
        for concept_name in extracted:
            # Ensure the concept exists.
            if self.network.get_concept(concept_name) is None:
                self.network.add_concept(concept_name, confidence=0.4, origin="perceptual")
            members.add(concept_name)

        # Top-down refinement: if feedback was provided, remove the
        # pattern's concepts from other categories (correction).
        if feedback:
            for other_label, other_members in self._perceptual_categories.items():
                if other_label == label:
                    continue
                for concept_name in extracted:
                    other_members.discard(concept_name)

    # ─── Catastrophic forgetting protection ──────────────────────

    def protect_important_concepts(self) -> int:
        """Identify and protect important concepts from forgetting.

        Uses elastic weight consolidation: concepts that are highly
        connected (high degree) and high-confidence are marked as
        important and protected from being overwritten by new
        learning. Returns the number of concepts now protected.
        """
        for cid in self.network.concept_ids:
            concept = self.network.get_concept(cid)
            if concept is None:
                continue
            degree = len(self.network.get_neighbors(cid))
            # Important = well-connected (degree >= 3) AND confident.
            if degree >= 3 and concept.confidence >= 0.7:
                self._protected_concepts.add(cid)
        return len(self._protected_concepts)

    def replay_consolidation(self) -> int:
        """Replay important concepts to protect against forgetting.

        Replay-based protection: periodically re-activates important
        (protected) concepts, reinforcing their confidence and
        activation so they are not overwritten by new learning. This
        mimics memory replay during sleep consolidation.

        Returns the number of concepts replayed.
        """
        replayed = 0
        for cid in self._protected_concepts:
            concept = self.network.get_concept(cid)
            if concept is None:
                continue
            # Saturating confidence reinforcement.
            concept.confidence = concept.confidence + (1.0 - concept.confidence) * 0.05
            concept.activation = min(1.0, (concept.activation or 0.0) + 0.2)
            self.network._mark_active(cid)
            replayed += 1
        self._replay_count += 1
        return replayed

    @property
    def protected_concept_count(self) -> int:
        """Number of concepts protected from forgetting."""
        return len(self._protected_concepts)

    @property
    def replay_count(self) -> int:
        """Number of replay consolidation cycles performed."""
        return self._replay_count

    # ─── Advanced learning system integration ──────────────────────

    def _post_learning_hook(
        self,
        topic: str,
        new_concepts: list[str],
        new_relationships: list[tuple[str, str, str]],
        surprise: float,
    ) -> None:
        """Integrate advanced learning systems after a learning episode.

        This is the nervous-system connection that wires four
        biologically-grounded learning mechanisms into the learning
        loop:

        1. **STDP** — records spike events for the topic and newly
           learned concepts in activation order, then applies
           timing-dependent plasticity to strengthen sequential
           connections (temporal causality).
        2. **Dual-system** — encodes the new knowledge into the
           fast (hippocampal) store for rapid access, to be
           consolidated into the slow (neocortical) store later.
        3. **Spaced repetition** — schedules newly learned concepts
           for review based on the forgetting curve.
        4. **TD learning** — computes the reward prediction error
           for this episode and updates the value function, driving
           curiosity via dopamine.

        All integrations are fault-tolerant — a failure in any
        subsystem does not disrupt the core learning loop.
        """
        if not new_concepts and not new_relationships:
            return

        # 1. STDP — strengthen sequential concept connections
        try:
            self._apply_stdp(topic, new_concepts, new_relationships)
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))  # STDP is optional; don't block learning

        # 2. Dual-system — encode into hippocampal fast store
        try:
            experience = new_concepts if new_concepts else [topic]
            self.dual_system.encode_fast(experience)
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))

        # 3. Spaced repetition — schedule reviews for new concepts
        try:
            for concept in new_concepts:
                self.spaced_repetition.record_review(concept, success=True)
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))

        # 4. TD learning — reward prediction error and curiosity drive
        try:
            self._update_td(topic, new_concepts, new_relationships, surprise)
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))

    def _apply_stdp(
        self,
        topic: str,
        new_concepts: list[str],
        new_relationships: list[tuple[str, str, str]],
    ) -> None:
        """Apply STDP to strengthen sequential concept connections.

        Records spike (activation) events for the topic and newly
        learned concepts in activation order, then applies
        timing-dependent plasticity. The topic fires first (it drove
        the learning), followed by each new concept in sequence with
        a small delay — this captures the temporal causality that
        plain co-occurrence Hebbian learning cannot: it matters not
        just *that* two concepts co-occur, but *in what order*.
        """
        stdp = self._stdp
        if stdp is None or self._embeddings is None:
            return

        # Logical spike times (ms). The topic fires first, then each
        # new concept fires in sequence with a 5ms delay (within the
        # 40ms STDP learning window). This ordering means the topic
        # → concept connections are potentiated (LTP), capturing the
        # causal direction of learning.
        t = 0.0
        stdp.record_spike(topic, t)
        t += 5.0
        for concept in new_concepts:
            stdp.record_spike(concept, t)
            t += 5.0
        # Record spikes for relationship endpoints in order
        for subject, _rel, obj in new_relationships:
            stdp.record_spike(subject, t)
            t += 2.5
            stdp.record_spike(obj, t)
            t += 2.5

        # Apply the pending STDP updates to the embedding vectors
        stdp.apply_updates()

    def _update_td(
        self,
        topic: str,
        new_concepts: list[str],
        new_relationships: list[tuple[str, str, str]],
        surprise: float,
    ) -> None:
        """Apply TD learning to update reward predictions.

        Computes the reward from this learning episode, compares it
        to the expected value of the state (the topic and its
        connected concepts), and updates the value function via the
        TD(0) rule:

            δ = reward + γ·V(s') − V(s)

        The reward prediction error (RPE) is the dopamine signal —
        positive RPE means learning was more rewarding than expected,
        which drives curiosity through the dopaminergic system.
        Negative RPE means learning was less rewarding than expected,
        dampening curiosity. This is the neural basis of curiosity:
        the brain learns to predict which topics will be interesting
        to explore.
        """
        # The state is the topic + its known neighbors (what it
        # expected to find — its current understanding of the topic).
        state = [topic]
        neighbors = self.network.get_neighbors(topic)
        for target, _, _ in neighbors[:5]:
            state.append(target)

        # The reward is the learning reward (dopamine magnitude).
        reward = self.compute_learning_reward(
            concepts_learned=len(new_concepts) + len(new_relationships),
            surprise=surprise,
        )

        # The next state is the topic + newly learned concepts
        # (what it now knows after this episode).
        next_state = [topic, *new_concepts[:5]]

        # TD(λ) update — returns the reward prediction error δ.
        # Eligibility traces propagate credit to recently-visited
        # concepts in the learning sequence.
        self.td_learner.update(state, reward, next_state)

        # Record performance for metalearning adaptation. The reward
        # is normalized to [0, 1] by dividing by the max dopamine
        # magnitude (0.2, from compute_learning_reward's bound).
        self.meta_learner.record_performance(min(1.0, max(0.0, reward / 0.2)))

        # Emit the dopamine signal (RPE scaled for the neurochemical
        # system). Positive RPE → dopamine burst (curiosity increases);
        # negative RPE → dopamine dip (curiosity decreases). This
        # integrates TD learning with the existing neurochemical
        # impulse system.
        dopamine = self.td_learner.get_dopamine_signal()
        if self._on_neuro_impulse and abs(dopamine) > 1e-6:
            try:
                self._on_neuro_impulse(CHEM_DOPAMINE, dopamine)
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(f"daemon connection lost: {e}")

    def _idle_consolidation(self) -> None:
        """Idle-time consolidation and review.

        When Genesis has no topic to learn about, it uses the time
        for memory consolidation and review — mirroring what the
        brain does during quiet wakefulness and sleep:

        1. **Dual-system consolidation** — replay hippocampal
           episodes and transfer them to neocortical storage
           (systems consolidation, Diekelmann & Born, 2010).
        2. **Spaced repetition review** — re-activate concepts whose
           retention has dropped below the threshold, stabilizing
           their memory traces (Ebbinghaus, 1885).
        3. **STDP renormalization** — re-normalize embedding vectors
           after any pending STDP updates to keep cosine similarity
           valid.

        Under a PROTECTIVE posture (chronic stress), consolidation
        review is reduced to lower CPU load. The previous logic raised
        the review limit under stress, but since the stress itself is
        driven by CPU usage (interoception maps self-process CPU to
        CRH→cortisol), increasing consolidation work creates a
        self-sustaining stress loop. Reducing review work lets
        cortisol recover, after which full consolidation resumes.
        """
        from genesis_client.types import LEARNING_POSTURE_PROTECTIVE

        # Under protective posture, review fewer concepts to reduce
        # CPU load and allow cortisol to recover
        review_limit = 3 if self._current_posture == LEARNING_POSTURE_PROTECTIVE else 5

        # 1. Dual-system: consolidate fast (hippocampal) → slow (neocortical)
        try:
            self.dual_system.consolidate_to_slow()
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))

        # 2. Spaced repetition: review due concepts
        try:
            self._review_due_concepts(limit=review_limit)
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))

        # 3. STDP: renormalize embeddings after updates
        if self._stdp is not None and self._embeddings is not None:
            try:
                self._stdp.renormalize()
            except Exception as e:  # noqa: BLE001
                logger.debug(repr(e))

    def _review_due_concepts(self, limit: int = 5) -> int:
        """Review concepts due for spaced repetition.

        Checks which concepts have retention below the threshold
        (they are being forgotten) and re-activates them, reinforcing
        their confidence and activation. Each review is recorded as
        successful (it recalled the concept), which extends the
        next review interval via the ease factor.

        Args:
            limit: Maximum number of concepts to review this cycle.
                Raised under protective posture to prioritize
                consolidation.

        Returns:
            The number of concepts reviewed.
        """
        schedule = self.spaced_repetition.get_review_schedule(limit=limit)
        reviewed = 0
        for concept, _urgency in schedule:
            c = self.network.get_concept(concept)
            if c is None:
                continue
            # Re-activate: reinforce confidence (saturating update)
            # and boost activation, mimicking memory reactivation.
            c.confidence = c.confidence + (1.0 - c.confidence) * 0.05
            c.activation = min(1.0, (c.activation or 0.0) + 0.1)
            self.network._mark_active(concept)
            # Record the review as successful (it recalled it).
            # This expands the next interval via the ease factor.
            self.spaced_repetition.record_review(concept, success=True)
            reviewed += 1
        if reviewed:
            self._emit("learning", f"Spaced repetition: reviewed {reviewed} due concepts")
        return reviewed

    def retrieve_dual(self, query: str | list[str]) -> list[tuple[str, float, str]]:
        """Retrieve memories from the dual-system (hippocampal + neocortical).

        Checks both the fast (hippocampal) and slow (neocortical)
        stores. The hippocampus provides specific, recent recall;
        the neocortex provides generalized, consolidated knowledge.
        Results are blended and sorted by similarity.

        Args:
            query: A text query or list of concept names.

        Returns:
            A list of (key, similarity, system) tuples sorted by
            similarity descending. ``system`` is 'hippocampal' or
            'neocortical'.
        """
        return self.dual_system.retrieve(query)
