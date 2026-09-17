"""Web search — Genesis's on-demand gateway to the World Wide Web.

This module gives Genesis the ability to search the web whenever she
wants — in conversation, for fun, to learn. It is a read-only capability:
she can search, fetch pages, and read content, but cannot submit forms,
post data, or modify anything online.

Search uses DuckDuckGo Lite (no API key, privacy-respecting, free).
Page fetching uses plain GET requests with a descriptive User-Agent.

Content filtering:
    A blocklist of adult/pornographic, malware, and tracking domains
    is enforced. URLs on these domains are filtered from search
    results and refused by the fetcher. Search queries containing
    explicit adult terms are refused outright. This is a blocklist
    (deny known-bad), not a whitelist (allow only known-good) — the
    open web is available by default, with known-bad content blocked.

This is a building block (a tool), not a response generator. The
text returned by ``fetch`` is raw content that her cognition learns
from and composes with — she never recites web text verbatim.
"""

from __future__ import annotations

import html.parser
import logging
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Network constants ─────────────────────────────────────────────

USER_AGENT = "Genesis-AI/1.0 (cognitive mind)"
REQUEST_TIMEOUT = 10  # seconds
MAX_TEXT_LENGTH = 50000  # don't process pages longer than this

# DuckDuckGo Lite — lightweight, privacy-respecting, no API key.
_DDG_LITE = "https://lite.duckduckgo.com/lite/"
_URL_RE = re.compile(r'https?://[^\s<>"\']+')


# ─── Content filter ────────────────────────────────────────────────
#
# A blocklist of domains known to host adult/pornographic content,
# malware, phishing, or aggressive tracking. This is NOT a whitelist
# — the open web is available by default. These domains are blocked
# because they are known-bad: porn, malware, trackers, link farms.
#
# The list is intentionally curated and conservative. It blocks the
# most common adult/malware domains. It is not exhaustive — no
# blocklist is — but it catches the obvious cases. The query filter
# below catches adult-themed searches regardless of domain.

# Adult/pornographic domains — the most common ones.
_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    # Pornographic tube/aggregate sites
    "pornhub.com", "www.pornhub.com",
    "xvideos.com", "www.xvideos.com",
    "xnxx.com", "www.xnxx.com",
    "xhamster.com", "www.xhamster.com",
    "redtube.com", "www.redtube.com",
    "youporn.com", "www.youporn.com",
    "tube8.com", "www.tube8.com",
    "spankbang.com", "www.spankbang.com",
    "txxx.com", "www.txxx.com",
    "porn300.com", "www.porn300.com",
    "eporner.com", "www.eporner.com",
    "beeg.com", "www.beeg.com",
    "drtuber.com", "www.drtuber.com",
    "nuvid.com", "www.nuvid.com",
    "pornone.com", "www.pornone.com",
    "hclips.com", "www.hclips.com",
    "upornia.com", "www.upornia.com",
    "tubepornstars.com", "www.tubepornstars.com",
    "porntrex.com", "www.porntrex.com",
    "camwhores.tv", "www.camwhores.tv",
    "chaturbate.com", "www.chaturbate.com",
    "bongacams.com", "www.bongacams.com",
    "livejasmin.com", "www.livejasmin.com",
    "cam4.com", "www.cam4.com",
    "stripchat.com", "www.stripchat.com",
    # Adult image boards / aggregators
    "imagefap.com", "www.imagefap.com",
    "motherless.com", "www.motherless.com",
    "efukt.com", "www.efukt.com",
    # Adult content networks
    "onlyfans.com", "www.onlyfans.com",
    "fansly.com", "www.fansly.com",
    # Pornographic search/aggregators
    "porndig.com", "www.porndig.com",
    "pornmd.com", "www.pornmd.com",
    "findtubes.com", "www.findtubes.com",
    "porntube.com", "www.porntube.com",
    "keezmovies.com", "www.keezmovies.com",
    "extremetube.com", "www.extremetube.com",
    # Malware / phishing / scam domains
    "doubleclick.net", "www.doubleclick.net",
    "googletagmanager.com", "www.googletagmanager.com",
    "googletagservices.com", "www.googletagservices.com",
    "google-analytics.com", "www.google-analytics.com",
    "scorecardresearch.com", "www.scorecardresearch.com",
    "quantserve.com", "www.quantserve.com",
    "adnxs.com", "www.adnxs.com",
    "criteo.com", "www.criteo.com",
    "taboola.com", "www.taboola.com",
    "outbrain.com", "www.outbrain.com",
})

# Search query terms that indicate an adult search. If any of these
# appear in the query, the search is refused. This catches adult
# content regardless of which domain it lives on.
_ADULT_QUERY_TERMS: frozenset[str] = frozenset({
    "porn", "pornography", "xxx", "hentai", "rule34", "rule 34",
    "nsfw", "camgirl", "cam girl", "strip cam", "sex cam",
    "escort", "hookup", "fuck", "dick", "cock", "pussy",
    "boob", "tits", "milf", "anal sex", "blowjob", "handjob",
    "creampie", "gangbang", "bondage", "bdsm",
    "onlyfans leak", "nude", "nudes", "naked",
})

# File extensions that are never useful content for learning.
_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
    ".css", ".js", ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".dmg",
    ".iso", ".bin", ".dat", ".db", ".sqlite",
)


def _domain_of(url: str) -> str:
    """Extract the lowercase netloc from a URL, stripping leading www."""
    netloc = urllib.parse.urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def is_url_safe(url: str) -> bool:
    """Return True if a URL is not on the adult/malware blocklist.

    This is a blocklist check, not a whitelist. The open web is
    allowed by default; only known-bad domains are refused.
    """
    domain = _domain_of(url)
    if not domain:
        return False
    if domain in _BLOCKED_DOMAINS:
        return False
    # Check suffix matches (e.g. doubleclick.net matches ad.doubleclick.net)
    for blocked in _BLOCKED_DOMAINS:
        if domain.endswith("." + blocked):
            return False
    # Skip non-content file types
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(_SKIP_EXTENSIONS):
        return False
    return True


def is_query_safe(query: str) -> bool:
    """Return True if a search query does not contain adult terms."""
    lower = query.lower()
    for term in _ADULT_QUERY_TERMS:
        if term in lower:
            return False
    return True


# ─── HTML text extraction ───────────────────────────────────────────


class _TextExtractor(html.parser.HTMLParser):
    """Extract readable text from HTML, skipping non-content elements."""

    _SKIP_TAGS = frozenset({
        "script", "style", "nav", "footer", "header", "form",
        "input", "select", "option", "button", "textarea",
        "label", "fieldset", "legend", "svg", "math", "canvas",
        "iframe", "noscript", "aside",
    })

    # HTML void elements — these never have closing tags, so they
    # must not increment the skip depth (otherwise the depth grows
    # unbounded and swallows all subsequent content).
    _VOID_TAGS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth = 0
        self._title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS and tag not in self._VOID_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing tags (e.g. <input/>). These don't affect depth."""
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and tag not in self._VOID_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li",
                    "dd", "dt", "blockquote", "div", "br", "tr", "td"):
            self._text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title += data
        else:
            self._text_parts.append(data)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._text_parts)).strip()

    @property
    def title(self) -> str:
        return self._title.strip()


# ─── Search result type ─────────────────────────────────────────────


@dataclass(slots=True)
class WebSearchResult:
    """A single web search result.

    Attributes:
        url: The result URL.
        title: The page title (may be empty until fetched).
        snippet: A short text snippet from the search result.
    """

    url: str
    title: str = ""
    snippet: str = ""


@dataclass(slots=True)
class WebFetchResult:
    """The content of a fetched web page.

    Attributes:
        url: The URL that was fetched.
        title: The page title extracted from HTML.
        content: The readable text content (plain text, not HTML).
    """

    url: str
    title: str
    content: str
    related_links: list[str] = field(default_factory=list)


# ─── SSL context (shared, cached) ───────────────────────────────────

_ssl_ctx: ssl.SSLContext | None = None


def _get_ssl_context() -> ssl.SSLContext:
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


# ─── Public API ─────────────────────────────────────────────────────


def search(query: str, limit: int = 5) -> list[WebSearchResult]:
    """Search the web (DuckDuckGo Lite) for a query.

    Returns up to ``limit`` results, filtered through the content
    blocklist. Adult/malware/tracking domains are excluded. If the
    query itself contains adult terms, returns an empty list.

    Args:
        query: The search query.
        limit: Maximum number of results to return.

    Returns:
        A list of WebSearchResult objects (url, title, snippet).
    """
    query = query.strip()
    if not query:
        return []
    if not is_query_safe(query):
        logger.debug(f"web search refused (adult query): {query!r}")
        return []

    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
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
            html_text = resp.read().decode("utf-8", errors="ignore")
    except (OSError, ValueError, RuntimeError) as e:
        logger.debug(f"web search failed for {query!r}: {e}")
        return []

    urls = _URL_RE.findall(html_text)
    seen: set[str] = set()
    results: list[WebSearchResult] = []
    for url in urls:
        # Strip tracking params
        url = url.split("&")[0]
        if url in seen:
            continue
        # Skip DuckDuckGo's own navigation links
        if "duckduckgo" in url.lower():
            continue
        if not is_url_safe(url):
            continue
        seen.add(url)
        results.append(WebSearchResult(url=url))
        if len(results) >= limit:
            break
    return results


def fetch(url: str) -> WebFetchResult | None:
    """Fetch a web page and extract its readable text.

    Read-only GET request. Refuses URLs on the adult/malware
    blocklist. Only processes text/html and text/plain responses.

    Args:
        url: The URL to fetch.

    Returns:
        A WebFetchResult with the page title and text content,
        or None if the fetch failed, the URL is blocked, or the
        content is too short.
    """
    if not is_url_safe(url):
        logger.debug(f"web fetch refused (blocked URL): {url}")
        return None

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(
            req, timeout=REQUEST_TIMEOUT, context=_get_ssl_context()
        ) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None
            raw = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
        logger.debug(f"web fetch failed for {url}: {e}")
        return None

    if len(raw) > MAX_TEXT_LENGTH * 10:
        raw = raw[: MAX_TEXT_LENGTH * 10]

    extractor = _TextExtractor()
    extractor.feed(raw)
    text = extractor.text
    title = extractor.title or ""

    if len(text) < 100:
        return None

    # Extract outbound links for curiosity chaining (follow links
    # from articles). Only keep safe, non-self links.
    related: list[str] = []
    for m in _URL_RE.findall(raw):
        link = m.split("&")[0]
        if link == url:
            continue
        if "duckduckgo" in link.lower():
            continue
        if is_url_safe(link) and link not in related:
            related.append(link)
        if len(related) >= 5:
            break

    return WebFetchResult(
        url=url,
        title=title,
        content=text[:MAX_TEXT_LENGTH],
        related_links=related,
    )


def search_and_fetch(
    query: str, limit: int = 3
) -> WebFetchResult | None:
    """Search the web and fetch the top safe result.

    Convenience function: searches for the query, then fetches the
    first result that passes the content filter. Returns the page
    content, or None if no safe results or all fetches failed.

    Args:
        query: The search query.
        limit: Max search results to try fetching.

    Returns:
        A WebFetchResult for the first successfully fetched page.
    """
    results = search(query, limit=limit)
    for result in results:
        page = fetch(result.url)
        if page is not None:
            return page
    return None
