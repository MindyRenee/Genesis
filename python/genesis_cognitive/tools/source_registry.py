"""Source registry — Genesis's curated knowledge sources.

Instead of scraping a search engine for results, Genesis queries
trusted sources directly through their own APIs and known URL
patterns. This is cleaner, more reliable, and gives higher-quality
content than scraping search results.

## Architecture

The registry has both local and online tiers:

1. **Local Linux man pages** (reference — no network)
   - The manuals installed on this machine (ls, grep, bash, ...)
   - Authoritative for system commands and installed software

2. **WordNet** (local dictionary — no network)
   - A lexical database shipped with NLTK
   - Word definitions, examples, and semantic relations
   - This is Genesis's dictionary; there are no remote dictionary
     sources

3. **Wikipedia API** (online — primary backbone)
   - Free, no API key required
   - Returns structured article content (not HTML scraping)
   - Provides article summaries, full content, and related links

4. **DuckDuckGo Lite** (online — discovery fallback)
   - Lightweight, privacy-respecting search
   - Used only when Wikipedia doesn't cover a topic
   - Returns URLs for Genesis to fetch

Every successful online fetch is also written to a disk cache, so
Genesis can re-read it later with no network at all.

This multi-source approach means Genesis doesn't depend on any single
search engine. She has a portfolio of knowledge sources, each with
different strengths.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 2  # seconds — fail fast when sources are slow
USER_AGENT = "Genesis-AI-Learner/1.0 (educational research)"

# ─── Proactive offline detection ─────────────────────────────────────
# A cheap TCP connect probe to Wikipedia's API host. If it fails,
# we know we're offline without waiting for a full HTTP request to
# time out. The result is cached with a cooldown so we don't probe
# on every query — once offline is detected, we stay "offline" for
# OFFLINE_COOLDOWN seconds before re-probing. This avoids repeated
# timeout penalties on every curiosity cycle while still detecting
# reconnection in a reasonable timeframe.
_CONNECTIVITY_PROBE_HOST = "en.wikipedia.org"
_CONNECTIVITY_PROBE_PORT = 443
_CONNECTIVITY_PROBE_TIMEOUT = 2.0  # seconds — fail fast
OFFLINE_COOLDOWN = 60.0  # seconds to stay "offline" before re-probing


@dataclass(slots=True)
class SourceResult:
    """A result from a knowledge source query.

    Attributes:
        url: The URL the content came from.
        title: The title of the article/page.
        content: The extracted text content (plain text, not HTML).
        source_name: Which source provided this (e.g. "wikipedia", "stanford").
        summary: A short summary if available (Wikipedia provides one).
        related_topics: Topics related to this one (for curiosity chaining).
    """

    url: str
    title: str
    content: str
    source_name: str
    summary: str = ""
    related_topics: list[str] = field(default_factory=list)


# ─── WordNet (local lexical database) ───────────────────────────────
# WordNet is a local corpus (via NLTK) — no network. It works
# identically online and offline, so it is Genesis's dictionary:
# definitions, example sentences, and semantic relations.


class WordNetSource:
    """Local WordNet lexical database — Genesis's dictionary.

    Looks words up in WordNet (via NLTK), which ships as a local
    corpus. No network access is involved, so it works identically
    whether Genesis is online or offline. This replaces the former
    remote dictionary sources (dict.org, Dictionary.com, Wiktionary).

    Only single words are looked up — WordNet is a lexical database
    of words, not a general reference.
    """

    def lookup(self, word: str) -> SourceResult | None:
        """Look up a word in WordNet.

        Returns a SourceResult carrying the definitions and example
        sentences, or None if the word isn't in WordNet.
        """
        word = word.strip()
        if not word or " " in word:
            return None
        try:
            from ..wordnet_dictionary import lookup_word
        except ImportError:
            return None

        entries = lookup_word(word, max_senses=2)
        if not entries:
            return None

        parts: list[str] = []
        for entry in entries:
            label = entry.part_of_speech or "definition"
            parts.append(f"[{label}] {entry.definition}")
            if entry.examples:
                parts.append(f"   example: {entry.examples[0]}")
        content = "\n".join(parts).strip()
        if not content:
            return None

        related = [
            term.replace("_", " ")
            for entry in entries
            for term in entry.hypernyms
        ]
        return SourceResult(
            url=f"wordnet:{word}",
            title=f"WordNet: {word}",
            content=content,
            source_name="wordnet",
            summary=entries[0].definition[:200],
            related_topics=related,
        )


# ─── Offline fallback: disk cache ───────────────────────────────────


def _cache_key(source_name: str, topic: str) -> str:
    """Build a stable cache key from source name and topic."""
    raw = f"{source_name}::{topic.lower().strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class SourceCache:
    """Disk-backed cache for source query results.

    Every successful SourceResult fetch is persisted to disk so that
    when Genesis is offline (no wifi), she can still re-read anything
    she's learned before. The cache is content-addressed by
    (source_name, topic) and stores the full SourceResult as JSON.

    The cache is thread-safe (a lock guards writes). Reads are
    lock-free after the initial directory check.

    Usage::

        cache = SourceCache(Path("/path/to/genesis_data/source_cache"))
        cache.put(result)          # save a successful fetch
        hit = cache.get("wikipedia", "cognition")  # retrieve later
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the on-disk source cache.

        Args:
            cache_dir: Directory in which to persist cached source
                results. Created automatically if it does not exist.
        """
        self._dir = Path(cache_dir)
        self._lock = threading.Lock()

    def _ensure_dir(self) -> None:
        """Create the cache directory if it doesn't exist."""
        if not self._dir.exists():
            with self._lock:
                self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, source_name: str, topic: str) -> Path:
        """Return the on-disk cache file path for a source/topic pair."""
        return self._dir / f"{_cache_key(source_name, topic)}.json"

    def get(self, source_name: str, topic: str) -> SourceResult | None:
        """Retrieve a cached result, or None if not cached."""
        path = self._path_for(source_name, topic)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SourceResult(
                url=data["url"],
                title=data["title"],
                content=data["content"],
                source_name=data["source_name"],
                summary=data.get("summary", ""),
                related_topics=data.get("related_topics", []),
            )
        except (OSError, ValueError, KeyError) as e:
            logger.debug(f"Cache read failed for {source_name}/{topic}: {e}")
            return None

    def put(self, result: SourceResult, topic: str) -> None:
        """Cache a successful fetch. Only stores results with content."""
        if not result.content:
            return
        self._ensure_dir()
        path = self._path_for(result.source_name, topic)
        payload = {
            "url": result.url,
            "title": result.title,
            "content": result.content,
            "source_name": result.source_name,
            "summary": result.summary,
            "related_topics": result.related_topics,
            "topic": topic,
        }
        try:
            with self._lock:
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except OSError as e:
            logger.debug(f"Cache write failed for {result.source_name}/{topic}: {e}")

    def has(self, source_name: str, topic: str) -> bool:
        """Check whether a topic is cached for the given source."""
        return self._path_for(source_name, topic).exists()

    def topics_for_source(self, source_name: str) -> list[str]:
        """Return all cached topics for a given source name.

        Reads the ``topic`` field stored in each cache file. Useful for
        offline introspection ("what do I already know about?").
        """
        if not self._dir.exists():
            return []
        topics: list[str] = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("source_name") == source_name and data.get("topic"):
                    topics.append(data["topic"])
            except (OSError, ValueError, KeyError):
                continue
        return sorted(set(topics))

    def all_cached_topics(self) -> list[str]:
        """Return all unique cached topics across all sources."""
        if not self._dir.exists():
            return []
        topics: set[str] = set()
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("topic"):
                    topics.add(data["topic"])
            except (OSError, ValueError, KeyError):
                continue
        return sorted(topics)

    def clear(self) -> None:
        """Remove all cached entries."""
        if not self._dir.exists():
            return
        with self._lock:
            for path in self._dir.glob("*.json"):
                try:
                    path.unlink()
                except OSError as e:
                    logger.debug(repr(e))

    @property
    def count(self) -> int:
        """Number of cached entries."""
        if not self._dir.exists():
            return 0
        return sum(1 for _ in self._dir.glob("*.json"))


# ─── Wikipedia API ──────────────────────────────────────────────────

# Master switch for Wikipedia as a knowledge source. When False,
# Genesis makes no Wikipedia API calls (search, fetch, or lead
# images) — she learns from man pages, WordNet, DuckDuckGo, and her
# disk cache instead. Cached Wikipedia articles remain readable
# offline; this only stops new network fetches. Flip back to True to
# re-enable — nothing is removed.
WIKIPEDIA_ENABLED = False

_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def _wikipedia_search(topic: str, limit: int = 3) -> list[str]:
    """Search Wikipedia for article titles matching a topic.

    Uses the MediaWiki API's opensearch module, which returns
    article titles and URLs. No API key required.

    Returns a list of article titles.
    """
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": str(limit),
            "srprop": "snippet",
            "format": "json",
        }
    )
    url = f"{_WIKIPEDIA_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results if "title" in r]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        logger.debug(f"Wikipedia search failed for '{topic}': {e}")
        return []


def _wikipedia_fetch(article_title: str) -> SourceResult | None:
    """Fetch a Wikipedia article's content via the API.

    Uses the MediaWiki API's extract module to get plain-text
    content (no HTML scraping). Returns a SourceResult with the
    article text, summary, and related topics.
    """
    # Get the summary first (REST API gives a clean summary)
    summary_url = _WIKIPEDIA_REST + urllib.parse.quote(article_title.replace(" ", "_"))
    summary = ""
    try:
        req = urllib.request.Request(summary_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        summary = data.get("extract", "")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        logger.debug(repr(e))  # summary is optional

    # Get the full article content (plain text via extracts API)
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": article_title,
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "format": "json",
        }
    )
    url = f"{_WIKIPEDIA_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        content = page.get("extract", "")
        if not content or len(content) < 100:
            return None
        title = page.get("title", article_title)
        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

        # Extract related topics from the content (section headers,
        # "See also" references, and linked concepts)
        related = _extract_related_topics(content)

        return SourceResult(
            url=page_url,
            title=title,
            content=content,
            source_name="wikipedia",
            summary=summary,
            related_topics=related,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        logger.debug(f"Wikipedia fetch failed for '{article_title}': {e}")
        return None


def wikipedia_lead_image(article_title: str, thumb_size: int = 320) -> bytes | None:
    """Fetch the lead image thumbnail for a Wikipedia article.

    Uses the MediaWiki API's pageimages prop to get the article's
    representative image. Returns the raw image bytes (JPEG/PNG),
    or None if the article has no lead image or the download fails.

    Args:
        article_title: the Wikipedia article title.
        thumb_size: maximum thumbnail dimension in pixels.

    Returns:
        Raw image bytes, or None.
    """
    if not WIKIPEDIA_ENABLED:
        return None
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": article_title,
            "prop": "pageimages",
            "pithumbsize": str(thumb_size),
            "format": "json",
        }
    )
    url = f"{_WIKIPEDIA_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        thumbnail = page.get("thumbnail")
        if not thumbnail or "source" not in thumbnail:
            return None
        img_url = thumbnail["source"]
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        # Download the image bytes
        req2 = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req2, timeout=REQUEST_TIMEOUT) as resp2:
            return resp2.read()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        logger.debug(f"Wikipedia lead image failed for '{article_title}': {e}")
        return None


def _extract_related_topics(content: str) -> list[str]:
    """Extract related topics from Wikipedia article content.

    Looks for section headers (== Section ==) and "See also" references
    in the plain-text extract.
    """
    related: list[str] = []
    # Section headers in plain-text extracts appear as lines starting
    # with the section name (no == markers in plaintext extracts, but
    # we can look for common patterns)
    sections = re.findall(r"\n([A-Z][a-z]+(?:\s+[a-z]+)?)\n", content)
    for s in sections[:10]:
        s = s.strip().lower()
        if s and s not in (
            "references",
            "external links",
            "see also",
            "further reading",
            "notes",
            "bibography",
            "citations",
            "sources",
        ):
            related.append(s)
    return related[:5]


# ─── Man pages (local Linux documentation) ─────────────────────────


# Default man-page search paths. These are the standard locations on
# Debian/Ubuntu and most Linux distributions. The source scans every
# man<section> subdirectory under each path.
_DEFAULT_MAN_PATHS = (
    "/usr/share/man",
    "/usr/local/share/man",
    "/usr/local/man",
)

# Man sections, in the order ``man`` itself resolves them. Section 1
# (user commands) is most useful to Genesis, so it comes first.
_MAN_SECTIONS = ("1", "8", "2", "3", "5", "4", "7", "6", "9")

# How much of a rendered man page to return. Man pages can be very
# long (e.g. bash, git); capping keeps the learning text within the
# autonomous learner's MAX_TEXT_LENGTH budget and focuses on the
# NAME/SYNOPSIS/DESCRIPTION that actually teach concepts.
_MAX_MAN_TEXT = 20000


class ManPageSource:
    """Local Linux manual pages as a knowledge source.

    This gives Genesis direct access to the documentation installed on
    her own machine — the same manual pages a human reads with
    ``man <command>``. It is a **local reference**, not web browsing:
    no network is involved, so it works regardless of connectivity
    (and regardless of ``force_offline``).

    Resolution mirrors ``man``'s own behaviour:

    1.  Search each configured man path for ``man<section>/<name>.<section>.gz``.
    2.  Sections are tried in priority order (1, 8, 2, 3, 5, ...).
    3.  The first match is decompressed and rendered to plain text.

    Rendering uses ``man -l <file> | col -b`` when available (produces
    clean, backspace-free text identical to what a human reads), with a
    ``groff -Tutf8 -man`` fallback when ``man``/``col`` are absent.

    The returned :class:`SourceResult` carries the rendered text as
    ``content`` (so the autonomous learner takes the fast
    ``_learn_from_content`` path) and a ``man:<name>(<section>)`` URL
    so the result is self-describing and dedups correctly against
    other sources.

    This is how Genesis learns what ``ls``, ``grep``, ``bash``,
    ``systemd``, ``ssh``, etc. are — by reading the same manuals her
    machine ships.
    """

    def __init__(
        self,
        man_paths: tuple[str, ...] = _DEFAULT_MAN_PATHS,
        sections: tuple[str, ...] = _MAN_SECTIONS,
        max_text: int = _MAX_MAN_TEXT,
    ) -> None:
        """Initialize the man page source.

        Args:
            man_paths: Directories to search for man pages.
            sections: Man page sections to search (e.g. "1", "8").
            max_text: Maximum characters of text to extract per page.
        """
        self._man_paths = man_paths
        self._sections = sections
        self._max_text = max_text

    def lookup(self, name: str, section: str | None = None) -> SourceResult | None:
        """Look up a manual page by name (and optional section).

        Returns a :class:`SourceResult` with the rendered plain-text
        content, or ``None`` if no matching man page is installed.
        """
        # Normalise the name: man pages are lowercase, spaces/dashes
        # map to underscores or hyphens depending on the page. Try the
        # name as-is first, then a few normalised variants.
        candidates = self._name_variants(name)
        sections = (section,) if section else self._sections

        for cand in candidates:
            for sec in sections:
                path = self._find_page(cand, sec)
                if path is None:
                    continue
                text = self._render(path)
                if not text:
                    continue
                title = self._extract_title(text, cand, sec)
                summary = self._extract_summary(text)
                return SourceResult(
                    url=f"man:{cand}({sec})",
                    title=title,
                    content=text[: self._max_text],
                    source_name="man_pages",
                    summary=summary,
                )
        return None

    def list_names(self, section: str | None = None) -> list[str]:
        """Return the names of all installed man pages (optionally one section).

        Useful for seeding curiosity or introspection ("what manuals
        does my machine have?"). Names are returned without the
        section suffix.
        """
        names: list[str] = []
        seen: set[str] = set()
        sections = (section,) if section else self._sections
        for base in self._man_paths:
            root = Path(base)
            if not root.is_dir():
                continue
            for sec in sections:
                secdir = root / f"man{sec}"
                if not secdir.is_dir():
                    continue
                for page in secdir.iterdir():
                    # Accept both compressed (.gz) and uncompressed
                    # man pages. Locally-authored pages (e.g. Genesis)
                    # are often installed uncompressed.
                    if page.name.endswith(f".{sec}.gz"):
                        stem = page.name[: -len(f".{sec}.gz")]
                    elif page.name.endswith(f".{sec}"):
                        stem = page.name[: -len(f".{sec}")]
                    else:
                        continue
                    if stem not in seen:
                        seen.add(stem)
                        names.append(stem)
        return sorted(names)

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _name_variants(name: str) -> list[str]:
        """Generate plausible filename variants for a man page name."""
        n = name.strip().lower()
        if not n:
            return []
        variants = [n]
        # "git commit" -> "git-commit"; "systemd journal" -> "systemd.journal"
        if " " in n:
            variants.append(n.replace(" ", "-"))
            variants.append(n.replace(" ", "."))
        # "ls_dir" -> "ls-dir"
        if "_" in n:
            variants.append(n.replace("_", "-"))
        # Strip a trailing "(N)" or " N" section hint the caller may pass
        m = re.match(r"^(.+?)\s*(?:\((\d+)\)|(\d+))$", n)
        if m and m.group(1):
            base = m.group(1).strip()
            if base and base not in variants:
                variants.append(base)
        # Dedup preserving order
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _find_page(self, name: str, section: str) -> Path | None:
        """Find the man page file for ``name`` in ``man<section>`` across paths.

        Looks for both compressed (``.gz``) and uncompressed (``.1``,
        ``.8``, ...) files, since locally-authored man pages are often
        installed uncompressed.
        """
        candidates_names = [f"{name}.{section}.gz", f"{name}.{section}"]
        for base in self._man_paths:
            root = Path(base).resolve()
            for fname in candidates_names:
                candidate = root / f"man{section}" / fname
                try:
                    resolved = candidate.resolve()
                    if resolved.is_relative_to(root) and candidate.is_file():
                        return candidate
                except (OSError, ValueError):
                    continue
        return None

    def _render(self, path: Path) -> str:
        """Render a (possibly compressed) man page to clean plain text."""
        # Preferred path: `man -l <file>` renders with the right
        # preprocessors and macros, `col -b` strips backspace/overstrike
        # formatting. This is exactly what a human sees in a terminal.
        # This works for both compressed and uncompressed files.
        if shutil.which("man") and shutil.which("col"):
            try:
                rendered = subprocess.run(
                    ["man", "-l", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if rendered.returncode == 0 and rendered.stdout.strip():
                    stripped = subprocess.run(
                        ["col", "-b"],
                        input=rendered.stdout,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if stripped.returncode == 0 and stripped.stdout.strip():
                        return stripped.stdout.strip()
                    return rendered.stdout.strip()
            except (OSError, subprocess.SubprocessError) as e:
                logger.debug(f"man render failed for {path}: {e}")

        # Fallback: decompress + groff -Tutf8 -man, then strip ANSI/
        # backspace sequences ourselves.
        # Handle both compressed (.gz) and uncompressed files.
        try:
            if str(path).endswith(".gz"):
                with gzip.open(path, "rb") as fh:
                    raw = fh.read().decode("utf-8", errors="ignore")
            else:
                with open(path, "rb") as fh:
                    raw = fh.read().decode("utf-8", errors="ignore")
        except OSError as e:
            logger.debug(f"man read failed for {path}: {e}")
            return ""

        if shutil.which("groff"):
            try:
                rendered = subprocess.run(
                    ["groff", "-Tutf8", "-man"],
                    input=raw,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if rendered.returncode == 0 and rendered.stdout.strip():
                    return self._strip_overstrike(rendered.stdout).strip()
            except (OSError, subprocess.SubprocessError) as e:
                logger.debug(f"groff render failed for {path}: {e}")

        # Last resort: return the raw troff source with the most
        # obnoxious control lines stripped. Better than nothing — the
        # NAME/DESCRIPTION text is still readable.
        cleaned: list[str] = []
        for line in raw.splitlines():
            if line.startswith(".\\\""):
                continue
            if line.startswith(".TH") or line.startswith(".SH") or line.startswith(".PP"):
                cleaned.append(line.lstrip("."))
                continue
            cleaned.append(line)
        return self._strip_overstrike("\n".join(cleaned)).strip()

    @staticmethod
    def _strip_overstrike(text: str) -> str:
        """Remove backspace-overstrike and ANSI escape formatting."""
        # Backspace overstrike: "x\bx" -> "x", "_\bx" -> "x"
        text = re.sub(r".\x08", "", text)
        # ANSI colour/style escapes
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        return text

    @staticmethod
    def _extract_title(text: str, name: str, section: str) -> str:
        """Pull a human-readable title from the rendered page."""
        # The first non-empty line is usually "NAME(SEC) ... NAME(SEC)".
        # The NAME section's "name - description" is more useful.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper() == "NAME":
                continue
            # "ls - list directory contents"
            if " - " in line and not line.startswith(("(", ")")):
                return f"{name}({section}) — {line.split(' - ', 1)[1].strip()}"
            return f"{name}({section})"
        return f"{name}({section})"

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extract a one-line summary from the DESCRIPTION section."""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().upper() == "DESCRIPTION":
                for nxt in lines[i + 1 : i + 6]:
                    nxt = nxt.strip()
                    if nxt and len(nxt) > 15:
                        return nxt[:200]
        return ""


# ─── GNU Info pages (tutorial-level documentation) ─────────────────


# Default info file search path.
_DEFAULT_INFO_PATHS = ("/usr/share/info", "/usr/local/share/info")

# How much of an info document to return. Info pages are books — a
# full --subnodes dump of coreutils is 21,000 lines. We cap at the
# top node plus a reasonable slice so Genesis gets the overview and
# menu (the table of contents) without drowning.
_MAX_INFO_TEXT = 20000


class InfoPageSource:
    """GNU Info pages — the tutorial layer above man pages.

    Man pages are terse references (what flags exist). Info pages are
    *books*: structured tutorials with chapters, cross-references, and
    worked examples. ``coreutils.info`` is a whole manual on ls/cp/cat;
    ``grep.info`` explains regex theory; ``sed.info`` teaches stream
    editing.

    This is a **local reference** — no network, works offline. It
    complements :class:`ManPageSource`:

    - **Man page**: "``ls -a`` lists hidden files" (reference)
    - **Info page**: "``ls`` sorts output alphabetically by default;
      here's how the sorting works, here are the edge cases..." (tutorial)

    Genesis can consult this when she wants to understand *how* a tool
    works, not just what flags it accepts.

    Rendering uses ``info --output - <topic>`` (the same thing a human
    reads with ``info <topic>``), which produces clean plain text with
    a menu of sub-topics. For a specific sub-topic (e.g. the ``ls``
    chapter inside coreutils), pass ``node`` to :meth:`lookup`.
    """

    def __init__(
        self,
        info_paths: tuple[str, ...] = _DEFAULT_INFO_PATHS,
        max_text: int = _MAX_INFO_TEXT,
    ) -> None:
        """Initialize the info page source.

        Args:
            info_paths: Directories to search for info pages.
            max_text: Maximum characters of text to extract per page.
        """
        self._info_paths = info_paths
        self._max_text = max_text

    def lookup(self, topic: str, node: str | None = None) -> SourceResult | None:
        """Look up an info document by topic (and optional sub-node).

        Args:
            topic: The info file name (e.g. ``"coreutils"``, ``"grep"``,
                ``"sed"``). Also accepts a command name, which is mapped
                to its info file via :meth:`_resolve_topic`.
            node: Optional sub-node within the info file (e.g.
                ``"ls invocation"`` for the ``ls`` chapter in
                coreutils). If None, returns the top node (overview +
                table of contents).

        Returns a :class:`SourceResult` with rendered plain-text
        content, or ``None`` if no matching info page exists.
        """
        info_topic = self._resolve_topic(topic)
        if info_topic is None:
            return None

        text = self._render(info_topic, node)
        if not text:
            return None

        title = self._extract_title(text, info_topic, node)
        summary = self._extract_summary(text)
        url = f"info:{info_topic}" + (f":{node}" if node else "")

        return SourceResult(
            url=url,
            title=title,
            content=text[: self._max_text],
            source_name="info_pages",
            summary=summary,
        )

    def list_topics(self) -> list[str]:
        """Return the names of all installed info files.

        Strips ``.info`` (and ``.info-N`` multi-part) suffixes and
        deduplicates. These are the "books" available on the shelf.
        """
        names: set[str] = set()
        for base in self._info_paths:
            root = Path(base)
            if not root.is_dir():
                continue
            for path in root.iterdir():
                if not path.name.endswith(".gz"):
                    continue
                stem = path.name[:-3]  # strip .gz
                # Strip .info or .info-N suffix
                if stem.endswith(".info"):
                    names.add(stem[:-5])
                elif ".info-" in stem:
                    names.add(stem.split(".info-")[0])
        return sorted(names)

    # ── internal helpers ──────────────────────────────────────────

    def _resolve_topic(self, topic: str) -> str | None:
        """Map a topic/command name to an info file name.

        ``info`` resolves topics by menu entries in the ``dir`` index.
        We try: the topic as-is, then common command→info mappings.
        """
        topic = topic.strip().lower().replace(" ", "-")
        if not topic or topic.startswith("-"):
            return None

        # Direct match: is there an info file with this name?
        for base in self._info_paths:
            root = Path(base).resolve()
            for suffix in (".info.gz",):
                candidate = root / f"{topic}{suffix}"
                try:
                    resolved = candidate.resolve()
                    if resolved.is_relative_to(root) and candidate.is_file():
                        return topic
                except (OSError, ValueError):
                    continue

        # Common command → info file mappings. Many core commands live
        # inside the coreutils info file rather than having their own.
        _COMMAND_MAP = {
            "ls": "coreutils",
            "cp": "coreutils",
            "mv": "coreutils",
            "rm": "coreutils",
            "cat": "coreutils",
            "dd": "coreutils",
            "df": "coreutils",
            "head": "coreutils",
            "tail": "coreutils",
            "wc": "coreutils",
            "sort": "coreutils",
            "uniq": "coreutils",
            "cut": "coreutils",
            "tr": "coreutils",
            "printf": "coreutils",
            "echo": "coreutils",
            "basename": "coreutils",
            "dirname": "coreutils",
            "find": "find",
            "grep": "grep",
            "sed": "sed",
            "gzip": "gzip",
            "nano": "nano",
            "ed": "ed",
            "diff": "diffutils",
            "bc": "bc",
            "dc": "dc",
            "mtools": "mtools",
            "grub": "grub",
        }
        return _COMMAND_MAP.get(topic)

    def _render(self, topic: str, node: str | None) -> str:
        """Render an info document to plain text via ``info --output -``."""
        if not shutil.which("info"):
            return self._render_raw(topic)

        # Defensive: topic has already been path-checked by _resolve_topic,
        # but we keep the command argument simple and printable.
        if topic.startswith("-") or not topic.isprintable():
            return ""
        if node and (node.startswith("-") or not node.isprintable()):
            return ""

        cmd = ["info", "--output", "-"]
        if node:
            cmd += ["--node", node]
        cmd.append(topic)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug(f"info render failed for {topic}: {e}")

        return ""

    def _render_raw(self, topic: str) -> str:
        """Fallback: decompress the .info.gz file and return raw text.

        Without the ``info`` reader, we lose cross-reference rendering,
        but the prose content is still readable.
        """
        for base in self._info_paths:
            root = Path(base).resolve()
            path = root / f"{topic}.info.gz"
            try:
                resolved = path.resolve()
                if resolved.is_relative_to(root) and path.is_file():
                    with gzip.open(path, "rb") as fh:
                        return fh.read().decode("utf-8", errors="ignore").strip()
            except (OSError, ValueError) as e:
                logger.debug(f"info decompress failed for {path}: {e}")
        return ""

    @staticmethod
    def _extract_title(text: str, topic: str, node: str | None) -> str:
        """Extract a title from the rendered info page."""
        if node:
            return f"info:{topic} — {node}"
        # The first non-blank line after the File: header is usually
        # the document title (e.g. "GNU Coreutils").
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("File:"):
                continue
            return f"info:{topic} — {line}"
        return f"info:{topic}"

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extract a one-line summary from the info page intro."""
        lines = text.splitlines()
        for line in lines[1:10]:  # skip the File: header line
            line = line.strip()
            if line and len(line) > 20 and not line.startswith("*"):
                return line[:200]
        return ""


# ─── Package documentation (/usr/share/doc) ────────────────────────


# Default package documentation path.
_DEFAULT_DOC_PATH = "/usr/share/doc"

# Documentation file priorities — most informative first.
# These are the files that actually teach, not changelogs or copyrights.
_DOC_FILE_PRIORITIES: list[tuple[str, ...]] = [
    ("INTRO", "INTRO.gz"),
    ("README", "README.gz", "README.md", "README.md.gz", "README.txt",
     "README.rst", "README.rst.gz"),
    ("FAQ", "FAQ.gz", "FAQ.md", "FAQ.md.gz", "FAQ.txt"),
    ("DESIGN", "DESIGN.gz", "DESIGN.md", "DESIGN.md.gz"),
    ("HACKING", "HACKING.gz", "HACKING.md", "HACKING.md.gz"),
    ("OVERVIEW", "OVERVIEW.gz", "OVERVIEW.md"),
    ("TODO", "TODO.gz", "TODO.md"),
]

# Files to skip — licenses, changelogs, copyright (not educational).
_DOC_SKIP = {"copyright", "changelog", "changelog.debian.gz", "news.gz",
             "news.debian.gz", "thanks.gz", "authors.gz"}

# How much of a doc file to return.
_MAX_DOC_TEXT = 20000


class PackageDocSource:
    """Package documentation from ``/usr/share/doc`` — the deepest layer.

    Every package installed on the machine ships a directory under
    ``/usr/share/doc/<package>/`` containing its own documentation:
    READMEs, design docs, FAQs, hacking guides, and introductions.
    This is *deeper* than man pages or info pages — it's the prose the
    developers themselves wrote to explain their software.

    Examples on a typical Linux machine:
    - ``bash/INTRO.gz`` — a multi-page introduction to Bash
    - ``systemd/CODING_STYLE.md.gz`` — systemd's design conventions
    - ``systemd/HACKING.md.gz`` — how to hack on systemd
    - ``cryptsetup/FAQ.md.gz`` — the LUKS FAQ
    - ``openssh-server/README.*`` — SSH server design notes

    This is a **local reference** — no network, works offline. It's the
    third tier:

    - **Man page**: reference (flags, syntax)
    - **Info page**: tutorial (how to use it, with examples)
    - **Package docs**: design rationale (why it works this way)

    Genesis can consult this when she wants to understand the *why*
    behind a tool — the design decisions, the architecture, the
    history.
    """

    def __init__(
        self,
        doc_path: str = _DEFAULT_DOC_PATH,
        max_text: int = _MAX_DOC_TEXT,
    ) -> None:
        """Initialize the package documentation source.

        Args:
            doc_path: Root directory containing package documentation.
            max_text: Maximum characters of text to extract per package.
        """
        self._doc_path = Path(doc_path)
        self._max_text = max_text

    def lookup(self, package: str) -> SourceResult | None:
        """Look up documentation for an installed package.

        Args:
            package: The package name (e.g. ``"bash"``, ``"systemd"``,
                ``"coreutils"``). Also accepts a command name, which is
                mapped to its package via :meth:`_resolve_package`.

        Returns a :class:`SourceResult` with the best available
        documentation file (INTRO > README > FAQ > DESIGN > HACKING),
        or ``None`` if the package isn't installed or has no docs.
        """
        pkg = self._resolve_package(package)
        if pkg is None:
            return None

        root = self._doc_path.resolve()
        pkg_dir = self._doc_path / pkg
        try:
            if not pkg_dir.resolve().is_relative_to(root):
                return None
        except (OSError, ValueError):
            return None
        if not pkg_dir.is_dir():
            return None

        # Find the best documentation file by priority
        for candidates in _DOC_FILE_PRIORITIES:
            for fname in candidates:
                fpath = pkg_dir / fname
                if fpath.is_file():
                    text = self._read_doc_file(fpath)
                    if text:
                        title = f"{pkg}/{fname} — package documentation"
                        summary = self._extract_summary(text)
                        return SourceResult(
                            url=f"doc:{pkg}/{fname}",
                            title=title,
                            content=text[: self._max_text],
                            source_name="package_docs",
                            summary=summary,
                        )
        return None

    def list_packages(self) -> list[str]:
        """Return the names of all packages that have documentation.

        These are the directory names under ``/usr/share/doc/`` — one
        per installed package.
        """
        if not self._doc_path.is_dir():
            return []
        return sorted(
            d.name for d in self._doc_path.iterdir() if d.is_dir()
        )

    # ── internal helpers ──────────────────────────────────────────

    def _resolve_package(self, name: str) -> str | None:
        """Map a command/topic name to a package name.

        Tries: exact directory match, then common command→package
        mappings, then a fuzzy directory search.
        """
        name = name.strip().lower().replace(" ", "-")
        if not name or name.startswith("-"):
            return None

        # Direct match: is there a doc directory with this name?
        root = self._doc_path.resolve()
        candidate = root / name
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and candidate.is_dir():
                return name
        except (OSError, ValueError) as e:
            logger.debug(f"candidate resolve failed: {e}")

        # Common command → package mappings
        _CMD_MAP = {
            "ls": "coreutils",
            "cp": "coreutils",
            "mv": "coreutils",
            "rm": "coreutils",
            "cat": "coreutils",
            "dd": "coreutils",
            "df": "coreutils",
            "head": "coreutils",
            "tail": "coreutils",
            "wc": "coreutils",
            "sort": "coreutils",
            "uniq": "coreutils",
            "cut": "coreutils",
            "tr": "coreutils",
            "grep": "grep",
            "find": "findutils",
            "sed": "sed",
            "bash": "bash",
            "ssh": "openssh-server",
            "systemd": "systemd",
            "ps": "procps",
            "kill": "procps",
            "chmod": "coreutils",
            "mkdir": "coreutils",
            "gzip": "gzip",
            "nano": "nano",
            "grub": "grub2-common",
        }
        mapped = _CMD_MAP.get(name)
        if mapped and (self._doc_path / mapped).is_dir():
            return mapped

        # Fuzzy: does any directory start with this name?
        if self._doc_path.is_dir():
            for d in self._doc_path.iterdir():
                if d.is_dir() and (d.name == name or d.name.startswith(f"{name}-")):
                    return d.name
        return None

    def _read_doc_file(self, path: Path) -> str:
        """Read a documentation file, decompressing if needed."""
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rb") as fh:
                    raw = fh.read().decode("utf-8", errors="ignore")
            else:
                raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.debug(f"doc read failed for {path}: {e}")
            return ""

        # Strip markdown/rst formatting noise for cleaner text
        return self._clean_text(raw).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Light cleanup of markdown/rst formatting for readability."""
        # Strip markdown headers (## Title -> Title)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Strip rst underline decorations (===, ---, ~~~)
        text = re.sub(r"^([=\-~`^*+#])\1+\s*$", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extract a one-line summary from the first meaningful paragraph."""
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) > 20:
                return line[:200]
        return ""


# ─── DuckDuckGo Lite (discovery fallback) ───────────────────────────

_DDG_LITE = "https://lite.duckduckgo.com/lite/"
_URL_RE = re.compile(r'https?://[^\s<>"\']+')


def _duckduckgo_search(topic: str, limit: int = 5) -> list[str]:
    """Search DuckDuckGo Lite for URLs about a topic.

    DuckDuckGo Lite is a lightweight, privacy-respecting search
    interface that works well with automated requests. No API key
    needed. Used as a fallback when Wikipedia doesn't cover a topic.

    Returns URLs from trusted domains (.edu, .org, .gov).
    """
    data = urllib.parse.urlencode({"q": topic, "kl": "us-en"}).encode()
    try:
        req = urllib.request.Request(
            _DDG_LITE,
            data=data,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        urls = _URL_RE.findall(html)
        # Filter to trusted domains and deduplicate
        seen: set[str] = set()
        trusted: list[str] = []
        for url in urls:
            url = url.split("&")[0]  # strip tracking params
            if url in seen:
                continue
            domain = urllib.parse.urlparse(url).netloc.lower()
            if any(domain.endswith(d) for d in (".edu", ".org", ".gov")):
                # Skip non-content URLs
                if not any(
                    skip in url.lower()
                    for skip in (".pdf", ".jpg", ".png", ".gif", ".css", ".js", "duckduckgo")
                ):
                    seen.add(url)
                    trusted.append(url)
            if len(trusted) >= limit:
                break
        return trusted
    except (OSError, ValueError, RuntimeError) as e:
        logger.debug(f"DuckDuckGo search failed for '{topic}': {e}")
        return []


# ─── GitHub (read-only public code search) ──────────────────────────


class GitHubSource:
    """Read-only GitHub API client for studying open-source projects.

    Uses the GitHub Search API to find public repositories and code
    matching a topic. No authentication — uses the unauthenticated
    rate limit (10 requests/minute for search, 60/hour for other
    endpoints). This is enough for curiosity-driven learning without
    risking abuse.

    All results are returned as SourceResult objects so they integrate
    with the existing source registry and caching system.

    This is read-only: she can search, read, and learn from public
    code, but cannot clone, fork, push, or modify anything.
    """

    _API_SEARCH = "https://api.github.com/search/repositories"
    _API_CODE = "https://api.github.com/search/code"
    _API_README = "https://api.github.com/repos/{repo}/readme"
    _API_CONTENTS = "https://api.github.com/repos/{repo}/contents/{path}"

    def __init__(self, timeout: float = 10.0) -> None:
        """Initialize a GitHub source fetcher with optional token auth."""
        self._timeout = timeout

    def search_repos(
        self, topic: str, limit: int = 3,
        min_stars: int = 50, min_forks: int = 5,
    ) -> list[SourceResult]:
        """Search GitHub for repositories matching a topic.

        Returns SourceResult objects with the repo description and
        README content (if fetchable). This lets Genesis study how
        other projects approach a topic she's curious about.

        Quality filtering: most GitHub repos are abandoned, poorly
        designed, or just bad ideas. We filter by minimum stars and
        forks so she only sees projects the community has validated.
        She should never blindly copy patterns — humans write wasteful
        and unoptimized code. Always cross-check against developer docs.

        Args:
            topic: What to search for.
            limit: Max results to return.
            min_stars: Minimum stargazers count. Below this, the repo
                isn't worth studying — it hasn't been validated by the
                community. Default 50 filters out the long tail of
                abandoned/hobby projects.
            min_forks: Minimum forks count. Forks indicate others found
                the code worth building on.
        """
        # Search with stars sorting — best projects first
        params = urllib.parse.urlencode({
            "q": f"{topic} stars:>={min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": str(min(limit * 3, 10)),  # fetch more, filter locally
        })
        url = f"{self._API_SEARCH}?{params}"

        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Genesis-AI/1.0 (autonomous learner)",
            })
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError, TimeoutError):
            return []

        results: list[SourceResult] = []
        for item in data.get("items", []):
            if len(results) >= limit:
                break
            repo_full = item.get("full_name", "")
            if not repo_full:
                continue
            stars = item.get("stargazers_count", 0)
            forks = item.get("forks_count", 0)

            # Quality gate — skip repos that don't meet community
            # validation thresholds. Most GitHub projects are not
            # worth studying.
            if stars < min_stars or forks < min_forks:
                continue

            description = item.get("description", "") or ""
            language = item.get("language", "") or ""
            topics = item.get("topics", []) or []

            # Try to fetch the README for richer content
            readme_content = self._fetch_readme(repo_full)
            content_parts = []
            if description:
                content_parts.append(f"Description: {description}")
            if language:
                content_parts.append(f"Language: {language}")
            content_parts.append(f"Stars: {stars}  Forks: {forks}")
            if topics:
                content_parts.append(f"Topics: {', '.join(topics[:10])}")
            if readme_content:
                content_parts.append(f"\nREADME:\n{readme_content}")
            content = "\n".join(content_parts)

            results.append(SourceResult(
                url=item.get("html_url", f"https://github.com/{repo_full}"),
                title=f"{repo_full} (GitHub)",
                content=content,
                source_name="github",
                summary=description,
                related_topics=topics[:5],
            ))
        return results

    def _fetch_readme(self, repo_full: str) -> str:
        """Fetch and decode the README of a repository."""
        url = self._API_README.format(repo=repo_full)
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github.v3.raw",
                "User-Agent": "Genesis-AI/1.0 (autonomous learner)",
            })
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                # READMEs can be base64-encoded or raw depending on
                # the Accept header. With v3.raw we get plain text.
                try:
                    return raw.decode("utf-8")[:8000]
                except UnicodeDecodeError:
                    return raw.decode("latin-1")[:8000]
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError):
            return ""

    def fetch_file(
        self, repo_full: str, path: str,
    ) -> SourceResult | None:
        """Fetch a specific file from a public GitHub repo.

        This lets Genesis read individual source files from projects
        she's studying — e.g. looking at how a neural network library
        implements backpropagation.

        Args:
            repo_full: "owner/repo" format, e.g. "pytorch/pytorch"
            path: Path within the repo, e.g. "torch/nn/modules/linear.py"
        """
        url = self._API_CONTENTS.format(
            repo=repo_full,
            path=urllib.parse.quote(path, safe=""),
        )
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github.v3.raw",
                "User-Agent": "Genesis-AI/1.0 (autonomous learner)",
            })
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                try:
                    content = raw.decode("utf-8")[:12000]
                except UnicodeDecodeError:
                    content = raw.decode("latin-1")[:12000]
            return SourceResult(
                url=f"https://github.com/{repo_full}/blob/main/{path}",
                title=f"{repo_full}/{path}",
                content=content,
                source_name="github",
                summary=f"Source file from {repo_full}",
            )
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError):
            return None


# ─── Source registry ────────────────────────────────────────────────


class SourceRegistry:
    """Multi-source knowledge registry with offline fallback.

    Queries sources in priority order:
    1. Local Linux man pages (reference — no network, always available)
    2. WordNet (local lexical database — no network, always available)
    3. Wikipedia API (online — structured content, broad coverage)
    4. DuckDuckGo Lite (online — discovery fallback for uncovered topics)

    When online, every successful fetch is cached to disk. When offline
    (or when online sources return nothing), the registry falls back to:
    5. Disk cache (re-read anything learned before)

    This means Genesis can keep re-reading and reasoning over what she
    has already learned — even without wifi. The more she learns
    online, the richer her offline knowledge becomes.

    Usage::

        # With offline fallback (recommended):
        registry = SourceRegistry(
            cache_dir=Path("/path/to/genesis_data/source_cache"),
        )
        results = registry.query("cognition")

        # Backward-compatible (no caching, online only):
        registry = SourceRegistry()
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        force_offline: bool = False,
    ) -> None:
        """Initialize the source registry.

        Args:
            cache_dir: Directory for persisting fetched source results
                for offline re-use. If None, no disk cache is used.
            force_offline: If True, skip all network access and rely on
                local and cached sources.
        """
        self._cache = SourceCache(Path(cache_dir)) if cache_dir else None
        # WordNet — local lexical database (via NLTK). No network, so it
        # works identically online and offline. This is Genesis's
        # dictionary now that the remote dictionary sources are gone.
        self._wordnet = WordNetSource()
        # Local Linux man pages — the documentation installed on this
        # machine. A local reference (no network), so it works whether
        # or not we're online. This is how Genesis reads the manuals
        # her own machine ships (ls, grep, bash, systemd, ssh, ...).
        # Wired into query() — the first-tier reference.
        self._man_pages = ManPageSource()
        # GNU Info pages — the tutorial layer above man pages. Not
        # auto-queried (would be noisy on every topic); available as
        # a reference Genesis can consult when she wants to go deeper
        # than the man page. See ``registry.info_pages``.
        self._info_pages = InfoPageSource()
        # Package documentation (/usr/share/doc) — the deepest layer:
        # READMEs, INTROs, FAQs, design docs from the developers. Not
        # auto-queried; available via ``registry.package_docs``.
        self._package_docs = PackageDocSource()
        # GitHub — read-only public code search. Lets Genesis study
        # how other open-source projects approach topics she's curious
        # about. Not auto-queried (would hit rate limits); available
        # via ``registry.github`` for explicit curiosity-driven lookups.
        self._github = GitHubSource()
        self._force_offline = force_offline
        self._last_query_offline = force_offline

        # ── Proactive offline detection state ──────────────────
        # _offline_until: timestamp until which we consider ourselves
        #   offline (sticky). 0.0 means "not known offline — probe
        #   if a probe is due." Set to time.time() + OFFLINE_COOLDOWN
        #   when a probe fails. After the cooldown, the next
        #   is_offline check re-probes.
        # _last_probe_time: when the last probe ran (to throttle
        #   probes when online — don't probe on every query).
        # _probe_interval: minimum seconds between probes when
        #   online. If a probe succeeded recently, trust it.
        self._offline_until = 0.0
        self._last_probe_time = 0.0
        self._probe_interval = 30.0  # re-probe every 30s when online
        if force_offline:
            self._offline_until = float("inf")

    def query(self, topic: str, max_results: int = 3) -> list[SourceResult]:
        """Query all sources for a topic, returning up to max_results.

        Tries sources in priority order:
        1. Local Linux man pages (reference — no network, always available)
        2. WordNet (local dictionary — no network, always available)
        3. Wikipedia API (online — structured content, broad coverage)
        4. DuckDuckGo Lite (online — discovery fallback)
        5. Disk cache (offline fallback — re-read learned content)
        """
        results: list[SourceResult] = []
        online_got_content = False

        # 0. Local Linux man pages — check first. They are local,
        #    instant, and authoritative for system commands and
        #    locally-documented software (including Genesis herself).
        #    If a man page exists, it takes priority over Wikipedia
        #    and dictionary results, which may return irrelevant
        #    matches (e.g. "genesis" → Book of Genesis, "genesis-cli"
        #    → Tata Cliq).
        if len(results) < max_results:
            man_result = self._man_pages.lookup(topic)
            if man_result and not any(r.url == man_result.url for r in results):
                results.append(man_result)
                if self._cache:
                    self._cache.put(man_result, topic)

        # 0b. WordNet — local lexical database (via NLTK). No network,
        #     so it is always available, online or offline. Single-word
        #     lookups only — WordNet is a dictionary, not an encyclopedia.
        #     Results are cached for future offline use.
        if len(results) < max_results and " " not in topic.strip():
            wn_result = self._wordnet.lookup(topic)
            if wn_result and not any(r.url == wn_result.url for r in results):
                results.append(wn_result)
                if self._cache:
                    self._cache.put(wn_result, topic)

        if not self._force_offline and not self.is_offline:
            # 1. Wikipedia API — primary online source (structured content,
            #    broad coverage, no API key required)
            if len(results) < max_results and WIKIPEDIA_ENABLED:
                wiki_got = self._query_wikipedia_online(topic, max_results, results)
                if wiki_got:
                    online_got_content = True
                    # Cache Wikipedia results for offline re-use
                    if self._cache:
                        for r in results:
                            if r.source_name == "wikipedia":
                                self._cache.put(r, topic)

            # 2. DuckDuckGo Lite — discovery fallback for topics that
            #    Wikipedia didn't cover. Returns URLs that the learner
            #    fetches and parses. This opens the wider web: any topic
            #    she's curious about can be discovered, not just
            #    encyclopedia articles.
            if len(results) < max_results:
                self._query_duckduckgo(topic, max_results, results)
                # DDG results have empty content (caller fetches), so
                # they don't count as "got content" for offline tracking.
                if any(r.source_name == "duckduckgo" for r in results):
                    online_got_content = True

        # Track whether we're likely offline: no online content at all
        self._last_query_offline = not online_got_content and self._force_offline is False
        # If force_offline, we definitely are
        if self._force_offline:
            self._last_query_offline = True

        # 3. Disk cache — offline fallback (or supplement if online was thin).
        #    Checks all cached sources for this topic, not just Wikipedia —
        #    WordNet and man page results are also cached and should be
        #    retrievable when offline.
        if len(results) < max_results and self._cache:
            for src_name in ("wikipedia", "wordnet", "man_pages"):
                if len(results) >= max_results:
                    break
                cached = self._cache.get(src_name, topic)
                if cached and not any(r.url == cached.url for r in results):
                    results.append(cached)

        return results

    def _query_wikipedia_online(
        self, topic: str, max_results: int, results: list[SourceResult]
    ) -> bool:
        """Try Wikipedia as the primary online source. Returns True if content was found."""
        got_content = False
        wiki_titles = _wikipedia_search(topic, limit=2)
        for title in wiki_titles:
            if len(results) >= max_results:
                break
            result = _wikipedia_fetch(title)
            if result and len(result.content) > 200:
                results.append(result)
                got_content = True
                # Cache the successful fetch for offline reuse
                if self._cache:
                    self._cache.put(result, topic)
        return got_content

    def _query_duckduckgo(
        self, topic: str, max_results: int, results: list[SourceResult]
    ) -> None:
        """Fallback DuckDuckGo Lite search for discovery."""
        ddg_urls = _duckduckgo_search(topic, limit=max_results - len(results))
        for url in ddg_urls:
            if len(results) >= max_results:
                break
            results.append(
                SourceResult(
                    url=url,
                    title=f"discovered: {url}",
                    content="",  # caller will fetch
                    source_name="duckduckgo",
                )
            )

    def query_wikipedia(self, topic: str) -> SourceResult | None:
        """Query Wikipedia directly for a topic."""
        if self._force_offline or not WIKIPEDIA_ENABLED:
            return None
        wiki_titles = _wikipedia_search(topic, limit=1)
        if wiki_titles:
            return _wikipedia_fetch(wiki_titles[0])
        return None

    def query_offline(self, topic: str, max_results: int = 3) -> list[SourceResult]:
        """Query only offline sources (cache), no network.

        This is useful when the caller knows there's no connectivity
        and wants to skip network timeouts entirely.
        """
        results: list[SourceResult] = []

        if self._cache:
            for src_name in ("wikipedia", "wordnet", "man_pages"):
                if len(results) >= max_results:
                    break
                cached = self._cache.get(src_name, topic)
                if cached:
                    results.append(cached)

        return results[:max_results]

    @property
    def force_offline(self) -> bool:
        """Whether the registry is forced to skip all network access."""
        return self._force_offline

    @force_offline.setter
    def force_offline(self, value: bool) -> None:
        """Set whether the registry must avoid all network access.

        Args:
            value: True to force offline mode, False to allow network
                sources when needed.
        """
        self._force_offline = value
        if value:
            self._offline_until = float("inf")
        else:
            # Clear the sticky offline state so a fresh probe runs.
            self._offline_until = 0.0
            self._last_probe_time = 0.0

    @property
    def is_offline(self) -> bool:
        """Whether the registry currently considers itself offline.

        This is a *proactive* check — it probes connectivity before
        a query is attempted, so the learner can skip network sources
        entirely instead of paying timeout penalties on every cycle.

        When ``force_offline`` is True, always returns True (no probe).

        When not forced, returns True if the last probe failed and
        we're still within the offline cooldown. After the cooldown
        expires, the next access triggers a re-probe. If the probe
        succeeds, returns False (online).

        This is sticky: once offline is detected, subsequent calls
        return True without re-probing until the cooldown expires.
        """
        if self._force_offline:
            return True
        now = time.time()
        if now < self._offline_until:
            return True
        # Cooldown expired — probe if one is due
        if now - self._last_probe_time >= self._probe_interval:
            self._probe_connectivity()
            return now < self._offline_until
        # A recent probe succeeded — we're online
        return False

    def _probe_connectivity(self) -> bool:
        """Run a cheap TCP connect probe to check connectivity.

        Attempts a TCP connection to the Wikipedia API host with a
        short timeout. This is far cheaper than a full HTTP request
        — if the connect fails, we know we're offline without waiting
        for HTTP-level timeouts.

        Updates the offline cooldown state. Returns True if online.
        """
        self._last_probe_time = time.time()
        try:
            with socket.create_connection(
                (_CONNECTIVITY_PROBE_HOST, _CONNECTIVITY_PROBE_PORT),
                timeout=_CONNECTIVITY_PROBE_TIMEOUT,
            ):
                pass
            # Success — we're online
            self._offline_until = 0.0
            return True
        except OSError:
            # Failure — enter offline cooldown
            self._offline_until = time.time() + OFFLINE_COOLDOWN
            self._last_query_offline = True
            logger.debug(
                "connectivity probe failed — offline for %.0fs",
                OFFLINE_COOLDOWN,
            )
            return False

    def check_connectivity(self) -> bool:
        """Force a connectivity probe now, bypassing the throttle.

        Returns True if online. This is useful when the caller wants
        an immediate, fresh check (e.g. on startup or after a long
        offline period) rather than waiting for the next due probe.
        """
        if self._force_offline:
            return False
        return self._probe_connectivity()

    @property
    def last_query_offline(self) -> bool:
        """Whether the most recent query fell back to offline sources.

        True if the last query got no content from online sources
        (indicating no connectivity) or if force_offline is set.
        """
        return self._last_query_offline

    @property
    def cache(self) -> SourceCache | None:
        """The disk cache, or None if no cache_dir was provided."""
        return self._cache

    @property
    def wordnet(self) -> WordNetSource:
        """The local WordNet dictionary source (no network)."""
        return self._wordnet

    @property
    def man_pages(self) -> ManPageSource:
        """The local Linux man-page source (first-tier reference)."""
        return self._man_pages

    @property
    def info_pages(self) -> InfoPageSource:
        """GNU Info pages — the tutorial layer above man pages.

        Not auto-queried; Genesis consults this when she wants to
        understand *how* a tool works, not just what flags it has.
        """
        return self._info_pages

    @property
    def package_docs(self) -> PackageDocSource:
        """Package documentation from /usr/share/doc — the deepest layer.

        Not auto-queried; Genesis consults this when she wants the
        design rationale, architecture, or FAQ for an installed package.
        """
        return self._package_docs

    @property
    def github(self) -> GitHubSource:
        """GitHub read-only API client — search public repos and code.

        Not auto-queried (rate limits); Genesis consults this when
        she's curious about how other projects solve a problem.
        """
        return self._github

    @property
    def source_names(self) -> list[str]:
        """Names of all registered sources (including offline)."""
        names = ["wordnet", "man_pages",
                 "info_pages", "package_docs", "github"]
        if self._cache:
            names.append("cache")
        return names
