"""Web search — Genesis's on-demand gateway to the World Wide Web.

This module gives Genesis the ability to search the web whenever it
wants — in conversation, for fun, to learn. It is a read-only capability:
it can search, fetch pages, and read content, but cannot submit forms,
post data, or modify anything online.

Search and browsing run through her real browser: Firefox driven
over WebDriver BiDi (the Chromium family over CDP serves hosts
without it), so JavaScript-heavy pages render and search behaves
like it does for a person. Plain-HTML fetches still take the
lightweight urllib path first; the browser is the fallback for thin
or failed pages. With no browser there is no search — an honest
absence, not a silent engine swap.

Content filtering:
    A blocklist of adult/pornographic, malware, and tracking domains
    is enforced. URLs on these domains are filtered from search
    results and refused by the fetcher. Search queries containing
    explicit adult terms are refused outright. This is a blocklist
    (deny known-bad), not a whitelist (allow only known-good) — the
    open web is available by default, with known-bad content blocked.

This is a building block (a tool), not a response generator. The
text returned by ``fetch`` is raw content that its cognition learns
from and composes with — it never recites web text verbatim.
"""

from __future__ import annotations

import atexit
import html.parser
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from genesis_client.swallow import note_swallowed
from inhibition import InhibitoryGate

if TYPE_CHECKING:
    from tools.browser import _Browser

logger = logging.getLogger(__name__)

# ─── Network constants ─────────────────────────────────────────────

USER_AGENT = "Genesis-AI/1.0 (cognitive mind)"
REQUEST_TIMEOUT = 10  # seconds
MAX_TEXT_LENGTH = 50000  # don't process pages longer than this

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
_BLOCKED_DOMAINS: frozenset[str] = frozenset(
    {
        # Pornographic tube/aggregate sites
        "pornhub.com",
        "www.pornhub.com",
        "xvideos.com",
        "www.xvideos.com",
        "xnxx.com",
        "www.xnxx.com",
        "xhamster.com",
        "www.xhamster.com",
        "redtube.com",
        "www.redtube.com",
        "youporn.com",
        "www.youporn.com",
        "tube8.com",
        "www.tube8.com",
        "spankbang.com",
        "www.spankbang.com",
        "txxx.com",
        "www.txxx.com",
        "porn300.com",
        "www.porn300.com",
        "eporner.com",
        "www.eporner.com",
        "beeg.com",
        "www.beeg.com",
        "drtuber.com",
        "www.drtuber.com",
        "nuvid.com",
        "www.nuvid.com",
        "pornone.com",
        "www.pornone.com",
        "hclips.com",
        "www.hclips.com",
        "upornia.com",
        "www.upornia.com",
        "tubepornstars.com",
        "www.tubepornstars.com",
        "porntrex.com",
        "www.porntrex.com",
        "camwhores.tv",
        "www.camwhores.tv",
        "chaturbate.com",
        "www.chaturbate.com",
        "bongacams.com",
        "www.bongacams.com",
        "livejasmin.com",
        "www.livejasmin.com",
        "cam4.com",
        "www.cam4.com",
        "stripchat.com",
        "www.stripchat.com",
        # Adult image boards / aggregators
        "imagefap.com",
        "www.imagefap.com",
        "motherless.com",
        "www.motherless.com",
        "efukt.com",
        "www.efukt.com",
        # Adult content networks
        "onlyfans.com",
        "www.onlyfans.com",
        "fansly.com",
        "www.fansly.com",
        # Pornographic search/aggregators
        "porndig.com",
        "www.porndig.com",
        "pornmd.com",
        "www.pornmd.com",
        "findtubes.com",
        "www.findtubes.com",
        "porntube.com",
        "www.porntube.com",
        "keezmovies.com",
        "www.keezmovies.com",
        "extremetube.com",
        "www.extremetube.com",
        # Malware / phishing / scam domains
        "doubleclick.net",
        "www.doubleclick.net",
        "googletagmanager.com",
        "www.googletagmanager.com",
        "googletagservices.com",
        "www.googletagservices.com",
        "google-analytics.com",
        "www.google-analytics.com",
        "scorecardresearch.com",
        "www.scorecardresearch.com",
        "quantserve.com",
        "www.quantserve.com",
        "adnxs.com",
        "www.adnxs.com",
        "criteo.com",
        "www.criteo.com",
        "taboola.com",
        "www.taboola.com",
        "outbrain.com",
        "www.outbrain.com",
    }
)

# Search query terms that indicate an adult search. If any of these
# appear in the query, the search is refused. This catches adult
# content regardless of which domain it lives on.
_ADULT_QUERY_TERMS: frozenset[str] = frozenset(
    {
        "porn",
        "pornography",
        "xxx",
        "hentai",
        "rule34",
        "rule 34",
        "nsfw",
        "camgirl",
        "cam girl",
        "strip cam",
        "sex cam",
        "escort",
        "hookup",
        "fuck",
        "dick",
        "cock",
        "pussy",
        "boob",
        "tits",
        "milf",
        "anal sex",
        "blowjob",
        "handjob",
        "creampie",
        "gangbang",
        "bondage",
        "bdsm",
        "onlyfans leak",
        "nude",
        "nudes",
        "naked",
    }
)

# File extensions that are never useful content for learning.
_SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".svg",
    ".css",
    ".js",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".exe",
    ".dmg",
    ".iso",
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
)


def _host_of(url: str) -> str:
    """Extract the lowercase hostname — no userinfo, port, or brackets."""
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _domain_of(url: str) -> str:
    """Extract the lowercase domain from a URL, stripping leading www."""
    host = _host_of(url)
    if host.startswith("www."):
        host = host[4:]
    return host


# ─── Local-target refusal (SSRF guard) ─────────────────────────────
#
# The autonomous learner follows search results and outbound links
# found inside fetched pages — both attacker-influenceable. Without
# this, a planted ``http://169.254.169.254/`` or ``http://192.168.1.1/``
# link would make it fetch cloud instance metadata (IAM tokens on
# GCP/AWS-style endpoints) or LAN admin pages straight into its
# concept network, where they could surface in conversation later.

_LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "broadcasthost",
    }
)

_LOCAL_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".lan",
    ".home",
    ".internal",
    ".corp",
)


def _ip_is_non_public(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """True for any address that isn't a normal public Internet host."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_is_local(host: str) -> bool:
    """Cheap local-target check: literal IPs and well-known local names.

    No DNS lookup — hostname-to-private-IP mappings are caught by
    ``_resolves_to_non_public`` in ``fetch``.
    """
    if host in _LOCAL_HOSTNAMES or host.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    try:
        return _ip_is_non_public(ipaddress.ip_address(host))
    except ValueError:
        return False


def _resolves_to_non_public(host: str) -> bool:
    """Resolve the host and refuse if any answer is non-public.

    Catches names pointed at loopback/link-local/RFC-1918 space
    (e.g. cloud ``*.internal`` metadata names). DNS rebinding between
    this check and connect remains theoretically possible; closing it
    fully would require pinning the resolved address.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    for info in infos:
        try:
            if _ip_is_non_public(ipaddress.ip_address(info[4][0])):
                return True
        except ValueError:
            continue
    return False


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
    # Refuse local/private targets (literal IPs and local hostnames;
    # the DNS-level check happens in fetch, which does I/O anyway).
    if _host_is_local(_host_of(url)):
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
            "aside",
        }
    )

    # HTML void elements — these never have closing tags, so they
    # must not increment the skip depth (otherwise the depth grows
    # unbounded and swallows all subsequent content).
    _VOID_TAGS = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

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
        if tag in (
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
            "div",
            "br",
            "tr",
            "td",
        ):
            self._text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        """Accumulate character data — title kept apart, body appended, skipped tags dropped."""
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


# ─── Her browser (the browsing path) ───────────────────────────────
#
# One lazy headless browser shared by search and fetch — Firefox
# over WebDriver BiDi, the Chromium family over CDP — driven by
# ``tools.browser._Browser``. A single tab
# means navigation must be serialized — the lock
# wraps each navigate → settle → extract cycle so concurrent callers
# (curiosity worker, conversation) cannot interleave page loads.
# The ``"web"`` profile under her data dir is persistent — cookies,
# site state, and history are her memory of the web across restarts.

_browser_lock = InhibitoryGate()
_browser: _Browser | None = None
_browser_failed = False


def _get_browser():
    """Return the shared headless browser, launching it on first use.

    Returns None if a browser cannot start — callers then take their
    lightweight fallback path. A failed launch marks the path dead for
    the session rather than paying the spawn cost on every call; a
    crashed browser gets relaunched once on the next call.
    """
    global _browser, _browser_failed
    if _browser is not None:
        return _browser
    if _browser_failed:
        return None
    try:
        from tools.browser import _Browser

        _browser = _Browser(headless=True, purpose="web")
        atexit.register(_close_browser)
        return _browser
    except RuntimeError as e:
        # A browser is optional on headless/mobile hosts. Its absence is
        # the documented no-search condition, not a runtime fault.
        if str(e) != "no browser installed":
            note_swallowed("tools.web_search.browser_launch", e)
        _browser_failed = True
        return None
    except Exception as e:  # noqa: BLE001 — no browser → no search
        note_swallowed("tools.web_search.browser_launch", e)
        _browser_failed = True
        return None


def _close_browser() -> None:
    global _browser
    if _browser is not None:
        try:
            _browser.close()
        except Exception as e:  # noqa: BLE001 — dying anyway
            note_swallowed("tools.web_search.browser_close", e)
        _browser = None


def _browser_wait_ready(browser, url: str = "", accept: tuple[str, ...] = ()) -> None:
    """Wait for the navigated page to finish loading (bounded).

    ``readyState`` alone lies: it still reads ``complete`` from the
    previous document until the new navigation commits, so the poll
    also requires ``location.href`` to be the page we asked for — or
    a known redirect destination listed in ``accept`` (a walled
    search engine redirects to its own interstitial, and that *is*
    the answer).
    """
    deadline = time.time() + 8
    stem = url.split("?")[0] if url else ""
    while time.time() < deadline:
        try:
            landed = str(browser.js("window.location.href") or "")
            on_target = not stem or landed.startswith(stem)
            redirected = any(a in landed for a in accept)
            if (on_target or redirected) and browser.js("document.readyState") == "complete":
                # Small settle for JS-injected content (search
                # results render after readyState flips).
                time.sleep(0.4)
                return
        except Exception as e:  # noqa: BLE001 — renderer mid-crash
            note_swallowed("tools.web_search.ready_poll", e)
            return
        time.sleep(0.15)


# Search engines are URL templates, not code. ``{q}`` is the query,
# ``{n}`` the requested result count where the engine takes one. A
# search is a navigation to one of these URLs in her real browser plus
# a generic pass over the external anchors the page rendered — no
# per-engine selectors, no scraping endpoints, so any engine works and
# none is privileged. ``GENESIS_SEARCH_ENGINE`` pins a single engine —
# a name from this table or a literal ``{q}`` template of her own.
_SEARCH_ENGINES: tuple[tuple[str, str], ...] = (
    ("google", "https://www.google.com/search?q={q}&num={n}"),
    ("bing", "https://www.bing.com/search?q={q}&count={n}"),
    ("brave", "https://search.brave.com/search?q={q}"),
    ("mojeek", "https://www.mojeek.com/search?q={q}"),
)

# An interstitial is not an answer: automation walls, consent gates,
# and bot challenges all mean "this engine did not answer" — the
# search moves to the next engine and the refusal names every one
# that walled rather than pretending there were no results.
_WALL_RE = re.compile(
    r"captcha|consent|/sorry/|unusual traffic|challenge-platform|"
    r"verify you are|are you a robot|bot detection|access denied",
    re.IGNORECASE,
)

# One eval, one source of truth: the title, the landed URL, and every
# external link outside the page's chrome furniture — nav, headers,
# footers. "External" means off the engine's own registrable domain,
# which drops related-searches, cached pages, and engine UI wholesale.
_LINKS_JS = """JSON.stringify((() => {
  const root = h => h.replace(/^www\\./, '').split('.').slice(-2).join('.');
  const here = root(location.hostname);
  const seen = new Set(); const links = [];
  for (const a of document.querySelectorAll('a[href^="http"]')) {
    if (a.closest('nav, header, footer, script, style,'
        + ' [role="navigation"], [role="banner"], [role="contentinfo"]')) continue;
    const t = (a.innerText || a.textContent || '').trim().slice(0, 300);
    if (t.length < 8) continue;
    try { if (root(new URL(a.href).hostname) === here) continue; } catch (e) { continue; }
    if (seen.has(a.href)) continue;
    seen.add(a.href);
    links.push({u: a.href, t});
  }
  return {title: document.title || '', url: location.href, links};
})())"""

# Readability-lite, in-page: prefer semantic containers, else the
# largest text-dense block that is not the whole document — scored by
# text length discounted by link noise, so nav-heavy wrappers lose to
# prose. innerText is the rendered truth; no HTML parsing on our side.
_CONTENT_JS = """JSON.stringify((() => {
  const txt = el => (el.innerText || '').trim();
  let root = null;
  const sels = ['article', 'main', '[role="main"]', '.article',
    '.post', '.entry-content', '#content'];
  for (const sel of sels) {
    const el = document.querySelector(sel);
    if (el && txt(el).length > 200) { root = el; break; }
  }
  if (!root) {
    let best = document.body, bestScore = 0;
    for (const el of document.querySelectorAll('article, section, div')) {
      const t = txt(el);
      const score = t.length / (1 + el.querySelectorAll('a').length * 0.15);
      if (score > bestScore) { bestScore = score; best = el; }
    }
    root = best;
  }
  const links = Array.from(root.querySelectorAll('a[href^="http"]')).map(a => a.href).slice(0, 200);
  return {t: document.title || '', x: txt(root).slice(0, %d), l: links};
})())"""


def _engine_plan() -> list[tuple[str, str]]:
    """The engines to try, in order — one when GENESIS_SEARCH_ENGINE pins."""
    pin = os.environ.get("GENESIS_SEARCH_ENGINE", "").strip()
    if not pin:
        return list(_SEARCH_ENGINES)
    for name, template in _SEARCH_ENGINES:
        if pin == name:
            return [(name, template)]
    if "{q}" in pin:
        return [("custom", pin)]
    logger.debug("GENESIS_SEARCH_ENGINE %r not a known engine or template; using defaults", pin)
    return list(_SEARCH_ENGINES)


def _page_links(browser) -> tuple[str, str, list[dict]]:
    """(title, landed_url, external_links) of the page the tab is on."""
    raw = browser.js(_LINKS_JS)
    try:
        data = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return "", "", []
    return (
        str(data.get("title", "")),
        str(data.get("url", "")),
        data.get("links") or [],
    )


def _browser_search(query: str, limit: int) -> list[WebSearchResult] | None:
    """Search through her browser — engine-agnostic, wall-honest.

    Returns a list on any engine that answered with real external
    links, ``[]`` when every engine walled or came back empty, and
    ``None`` only when the browser path itself failed (couldn't
    launch, dead socket) — the caller's "no search" condition.
    """
    browser = _get_browser()
    if browser is None:
        return None
    walled: list[str] = []
    try:
        with _browser_lock:
            for name, template in _engine_plan():
                url = template.format(q=urllib.parse.quote_plus(query), n=max(limit, 10))
                browser.navigate(url)
                _browser_wait_ready(browser, url.split("?")[0])
                title, landed, hits = _page_links(browser)
                if _WALL_RE.search(f"{title} {landed}"):
                    walled.append(f"{name} (wall)")
                    continue
                results: list[WebSearchResult] = []
                for hit in hits:
                    link = str(hit.get("u", ""))
                    if not is_url_safe(link):
                        continue
                    results.append(WebSearchResult(url=link, title=str(hit.get("t", ""))[:200]))
                    if len(results) >= limit:
                        break
                if results:
                    return results
                walled.append(f"{name} (empty)")
    except Exception as e:  # noqa: BLE001 — dead browser → relaunch next time
        note_swallowed("tools.web_search.browser_search", e)
        _close_browser()
        return None
    if walled:
        logger.info("web search: no engine answered — %s", ", ".join(walled))
    return []


def _browser_fetch(url: str) -> WebFetchResult | None:
    """Fetch a page through the headless browser (JS-rendered).

    Used when the urllib path failed or returned too little text —
    single-page apps and cookie-walled pages render properly here.
    Content comes from readability-lite in the page itself.
    """
    browser = _get_browser()
    if browser is None:
        return None
    try:
        with _browser_lock:
            browser.navigate(url)
            _browser_wait_ready(browser, url)
            # Redirect safety: the safety check ran on the input URL,
            # but navigation may have landed elsewhere — re-verify.
            final_url = browser.js("window.location.href") or ""
            final_host = _host_of(str(final_url))
            if _host_is_local(final_host) or _resolves_to_non_public(final_host):
                logger.debug(f"browser fetch refused (redirect to private): {url}")
                return None
            if not is_url_safe(str(final_url)):
                logger.debug(f"browser fetch refused (redirect to blocked): {url}")
                return None
            payload = browser.js(_CONTENT_JS % MAX_TEXT_LENGTH)
    except Exception as e:  # noqa: BLE001 — dead browser → relaunch next time
        note_swallowed("tools.web_search.browser_fetch", e)
        _close_browser()
        return None
    return _browser_fetch_result(url, payload)


def _link_key(url: str) -> str:
    """Canonical key for deduping page links.

    ``scheme://host/path`` — fragments and query strings are noise
    for "have I seen this page" (anchors, tracking params), so they
    don't distinguish keys.
    """
    try:
        p = urllib.parse.urlparse(url)
        return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")
    except ValueError:
        return url


def _browser_fetch_result(url: str, payload: str) -> WebFetchResult | None:
    """Turn the browser-side extraction JSON into a WebFetchResult."""
    try:
        data = json.loads(payload) if payload else {}
    except (TypeError, ValueError) as e:
        note_swallowed("tools.web_search.browser_parse", e)
        return None
    text = str(data.get("x", ""))
    if len(text) < 100:
        return None
    seen = {_link_key(url)}
    related: list[str] = []
    for link in data.get("l", []):
        link = str(link).split("&")[0]
        key = _link_key(link)
        if key in seen or not is_url_safe(link):
            continue
        seen.add(key)
        related.append(link)
        if len(related) >= 5:
            break
    return WebFetchResult(
        url=url,
        title=str(data.get("t", "")),
        content=text,
        related_links=related,
    )


# ─── Public API ─────────────────────────────────────────────────────


def search(query: str, limit: int = 5) -> list[WebSearchResult]:
    """Search the web through her browser for a query.

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

    # Her browser first — real Google through her real browser. None means
    # the browser path itself failed (couldn't launch, dead socket);
    # an empty list is a genuine "no results" and stands.
    browser_results = _browser_search(query, limit)
    if browser_results is not None:
        return browser_results

    # No browser, no search — she does not silently swap engines.
    return []


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

    # http/https only — urlopen also handles file:// and ftp://, which
    # must never reach the network stack from generated input.
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        logger.debug(f"web fetch refused (non-http scheme): {url}")
        return None

    # DNS-level SSRF check: refuse names that resolve to loopback,
    # link-local, or private ranges.
    host = _host_of(url)
    if _resolves_to_non_public(host):
        logger.debug(f"web fetch refused (resolves to private address): {url}")
        return None

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(
            req, timeout=REQUEST_TIMEOUT, context=_get_ssl_context()
        ) as resp:
            # Redirect targets bypass is_url_safe — re-check the final
            # URL before the body enters its memory. A public page that
            # 302s to a metadata endpoint gets its GET refused here.
            final_host = _host_of(resp.geturl())
            if _host_is_local(final_host) or _resolves_to_non_public(final_host):
                logger.debug(f"web fetch refused (redirect to private address): {url}")
                return None
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None
            raw = resp.read(MAX_TEXT_LENGTH * 10 + 1).decode("utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
        note_swallowed(
            "tools.web_search.fetch",
            e,
        )
        # urllib couldn't reach it — the real browser gets a shot
        # before the fetch is declared dead.
        return _browser_fetch(url)

    if len(raw) > MAX_TEXT_LENGTH * 10:
        raw = raw[: MAX_TEXT_LENGTH * 10]

    extractor = _TextExtractor()
    extractor.feed(raw)
    text = extractor.text
    title = extractor.title or ""

    if len(text) < 100:
        # Too little text — a JS-rendered page or an interstitial.
        # The real browser can still see it.
        return _browser_fetch(url)

    # Extract outbound links for curiosity chaining (follow links
    # from articles). Only keep safe, non-self links — canonicalised
    # so fragment/query variants of the same page never repeat.
    seen = {_link_key(url)}
    related: list[str] = []
    for m in _URL_RE.findall(raw):
        link = m.split("&")[0]
        key = _link_key(link)
        if key in seen or not is_url_safe(link):
            continue
        seen.add(key)
        related.append(link)
        if len(related) >= 5:
            break

    return WebFetchResult(
        url=url,
        title=title,
        content=text[:MAX_TEXT_LENGTH],
        related_links=related,
    )


def search_and_fetch(query: str, limit: int = 3) -> WebFetchResult | None:
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
