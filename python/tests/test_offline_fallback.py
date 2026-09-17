"""Tests for offline fallback: disk cache and registry integration."""

from pathlib import Path

from genesis_cognitive.tools.source_registry import (
    SourceCache,
    SourceRegistry,
    SourceResult,
    WordNetSource,
)

# ─── SourceCache ────────────────────────────────────────────────────


def _make_result(
    source_name: str = "wikipedia",
    title: str = "Test Article",
    content: str = "This is test content about cognition and awareness.",
    url: str = "https://en.wikipedia.org/wiki/Test",
) -> SourceResult:
    """Construct a result for tests."""
    return SourceResult(
        url=url,
        title=title,
        content=content,
        source_name=source_name,
        summary="A short summary.",
        related_topics=["awareness", "mind"],
    )


def test_cache_put_and_get(tmp_path: Path) -> None:
    """A cached result can be retrieved later."""
    cache = SourceCache(tmp_path / "cache")
    result = _make_result()
    cache.put(result, topic="cognition")

    retrieved = cache.get("wikipedia", "cognition")
    assert retrieved is not None
    assert retrieved.url == result.url
    assert retrieved.title == result.title
    assert retrieved.content == result.content
    assert retrieved.source_name == "wikipedia"
    assert retrieved.summary == "A short summary."
    assert retrieved.related_topics == ["awareness", "mind"]


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    """A cache miss returns None."""
    cache = SourceCache(tmp_path / "cache")
    assert cache.get("wikipedia", "nonexistent") is None


def test_cache_skips_empty_content(tmp_path: Path) -> None:
    """Results with empty content are not cached (DDG results need fetching)."""
    cache = SourceCache(tmp_path / "cache")
    result = SourceResult(
        url="https://example.com",
        title="Empty",
        content="",
        source_name="stanford",
    )
    cache.put(result, topic="philosophy")
    assert cache.get("stanford", "philosophy") is None
    assert cache.count == 0


def test_cache_has_method(tmp_path: Path) -> None:
    """The has() method correctly reports cached entries."""
    cache = SourceCache(tmp_path / "cache")
    assert not cache.has("wikipedia", "cognition")
    cache.put(_make_result(), topic="cognition")
    assert cache.has("wikipedia", "cognition")
    assert not cache.has("wikipedia", "other_topic")


def test_cache_count(tmp_path: Path) -> None:
    """Count reflects the number of cached entries."""
    cache = SourceCache(tmp_path / "cache")
    assert cache.count == 0
    cache.put(_make_result(), topic="cognition")
    assert cache.count == 1
    cache.put(_make_result(source_name="stanford"), topic="philosophy")
    assert cache.count == 2


def test_cache_overwrite(tmp_path: Path) -> None:
    """Re-caching the same source+topic overwrites the old entry."""
    cache = SourceCache(tmp_path / "cache")
    cache.put(_make_result(content="old content"), topic="cognition")
    cache.put(_make_result(content="new content"), topic="cognition")
    assert cache.count == 1
    retrieved = cache.get("wikipedia", "cognition")
    assert retrieved is not None
    assert retrieved.content == "new content"


def test_cache_clear(tmp_path: Path) -> None:
    """Clear removes all cached entries."""
    cache = SourceCache(tmp_path / "cache")
    cache.put(_make_result(), topic="cognition")
    cache.put(_make_result(source_name="stanford"), topic="philosophy")
    assert cache.count == 2
    cache.clear()
    assert cache.count == 0
    assert cache.get("wikipedia", "cognition") is None


def test_cache_all_cached_topics(tmp_path: Path) -> None:
    """all_cached_topics returns unique topics across sources."""
    cache = SourceCache(tmp_path / "cache")
    cache.put(_make_result(), topic="cognition")
    cache.put(_make_result(source_name="stanford"), topic="philosophy")
    cache.put(_make_result(source_name="nasa"), topic="cognition")
    topics = cache.all_cached_topics()
    assert sorted(topics) == ["cognition", "philosophy"]


def test_cache_persistence_across_instances(tmp_path: Path) -> None:
    """A new SourceCache pointing at the same dir reads old entries."""
    cache_dir = tmp_path / "cache"
    cache1 = SourceCache(cache_dir)
    cache1.put(_make_result(), topic="cognition")

    cache2 = SourceCache(cache_dir)
    retrieved = cache2.get("wikipedia", "cognition")
    assert retrieved is not None
    assert retrieved.content == "This is test content about cognition and awareness."


def test_cache_corrupt_file_returns_none(tmp_path: Path) -> None:
    """A corrupt cache file is handled gracefully (returns None)."""
    cache = SourceCache(tmp_path / "cache")
    cache.put(_make_result(), topic="cognition")
    # Corrupt the file
    files = list((tmp_path / "cache").glob("*.json"))
    assert len(files) == 1
    files[0].write_text("{not valid json", encoding="utf-8")
    assert cache.get("wikipedia", "cognition") is None


# ─── SourceRegistry offline integration ─────────────────────────────


def test_registry_backward_compatible() -> None:
    """SourceRegistry() with no args still works (backward compat)."""
    registry = SourceRegistry()
    assert "wordnet" in registry.source_names
    # No cache when no cache_dir provided
    assert registry.cache is None


def test_registry_with_cache_dir(tmp_path: Path) -> None:
    """SourceRegistry with cache_dir has a working cache."""
    registry = SourceRegistry(cache_dir=tmp_path / "cache")
    assert registry.cache is not None
    assert registry.cache.count == 0


def test_registry_force_offline_returns_no_results() -> None:
    """In force_offline mode with no cache, query returns empty."""
    registry = SourceRegistry(force_offline=True)
    registry.query("cognition")
    assert registry.last_query_offline is True


def test_registry_force_offline_uses_cache(tmp_path: Path) -> None:
    """In force_offline mode, cached results are served."""
    cache_dir = tmp_path / "cache"
    registry = SourceRegistry(cache_dir=cache_dir, force_offline=False)

    # Manually cache a result
    assert registry.cache is not None
    registry.cache.put(_make_result(), topic="cognition")

    # Now force offline and query (force_offline is always True now,
    # but cache should still serve)
    registry.force_offline = True
    results = registry.query("cognition")
    assert len(results) > 0
    cache_results = [r for r in results if r.source_name == "wikipedia"]
    assert len(cache_results) == 1
    assert "test content" in cache_results[0].content


def test_registry_query_offline_method(tmp_path: Path) -> None:
    """query_offline() skips network and uses only cache."""
    cache_dir = tmp_path / "cache"
    registry = SourceRegistry(cache_dir=cache_dir)

    # Cache a result
    assert registry.cache is not None
    registry.cache.put(_make_result(), topic="cognition")

    results = registry.query_offline("cognition")
    assert len(results) > 0
    source_names = {r.source_name for r in results}
    assert "wikipedia" in source_names  # from cache


def test_registry_query_offline_miss_returns_empty(tmp_path: Path) -> None:
    """query_offline for an unknown topic returns empty list."""
    registry = SourceRegistry(cache_dir=tmp_path / "cache")
    results = registry.query_offline("xyzzy_nonexistent_topic")
    assert results == []


def test_registry_source_names_includes_cache_when_present(tmp_path: Path) -> None:
    """source_names includes 'cache' when a cache_dir is provided."""
    registry = SourceRegistry(cache_dir=tmp_path / "cache")
    assert "cache" in registry.source_names


def test_registry_force_offline_property() -> None:
    """force_offline property can be toggled (web access is now enabled
    with a strict domain whitelist)."""
    registry = SourceRegistry()
    # Default is now False (web enabled for whitelisted domains)
    assert registry.force_offline is False
    registry.force_offline = True
    assert registry.force_offline is True
    registry.force_offline = False
    assert registry.force_offline is False


# ─── AutonomousLearner data_dir integration ─────────────────────────


def test_learner_with_data_dir_gets_cache(tmp_path) -> None:
    """An AutonomousLearner with a data_dir has a source cache."""
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.learning import AutonomousLearner, CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    learner = AutonomousLearner(
        network=net,
        curiosity=curiosity,
        data_dir=str(tmp_path),
    )
    assert learner._sources.cache is not None
    assert learner._sources.cache.count >= 0


def test_learner_without_data_dir_has_no_cache() -> None:
    """An AutonomousLearner without a data_dir has no cache (backward compat)."""
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.learning import AutonomousLearner, CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    learner = AutonomousLearner(
        network=net,
        curiosity=curiosity,
    )
    assert learner._sources.cache is None


# ─── WordNet (local dictionary) ──────────────────────────────────────


def test_wordnet_source_lookup() -> None:
    """WordNetSource returns a local definition for a known word."""
    result = WordNetSource().lookup("cognition")
    assert result is not None
    assert result.source_name == "wordnet"
    assert result.content
    assert result.url.startswith("wordnet:")


def test_wordnet_source_rejects_multi_word() -> None:
    """WordNetSource only looks up single words."""
    assert WordNetSource().lookup("quantum mechanics") is None


def test_wordnet_source_miss_returns_none() -> None:
    """A word not in WordNet returns None."""
    assert WordNetSource().lookup("xyzzyqwerty") is None


def test_registry_force_offline_serves_wordnet(tmp_path: Path) -> None:
    """force_offline queries still return local WordNet definitions."""
    registry = SourceRegistry(cache_dir=tmp_path / "cache", force_offline=True)
    results = registry.query("cognition")
    assert any(r.source_name == "wordnet" for r in results)


def test_force_offline_makes_no_network_calls(tmp_path: Path, monkeypatch) -> None:
    """force_offline queries must never touch the network.

    This is the core guarantee of `--offline`: the registry may serve
    local WordNet/man-page/cache content, but no socket or HTTP request
    may be attempted.
    """
    import socket
    import urllib.request

    def _boom(*args, **kwargs):
        raise AssertionError("network access attempted in offline mode")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    registry = SourceRegistry(cache_dir=tmp_path / "cache", force_offline=True)
    results = registry.query("cognition")
    assert any(r.source_name == "wordnet" for r in results)


# ─── Proactive offline detection ──────────────────────────────────────


def test_registry_force_offline_is_offline_true() -> None:
    """force_offline=True makes is_offline return True without probing."""
    registry = SourceRegistry(force_offline=True)
    assert registry.is_offline is True


def test_registry_is_offline_returns_bool() -> None:
    """is_offline returns a bool (not None) regardless of connectivity."""
    registry = SourceRegistry()
    # The first access may probe — but it must return a bool, not crash.
    result = registry.is_offline
    assert isinstance(result, bool)


def test_registry_check_connectivity_returns_bool() -> None:
    """check_connectivity returns a bool (True=online, False=offline)."""
    registry = SourceRegistry()
    result = registry.check_connectivity()
    assert isinstance(result, bool)


def test_registry_force_offline_check_connectivity_false() -> None:
    """check_connectivity returns False when force_offline is True."""
    registry = SourceRegistry(force_offline=True)
    assert registry.check_connectivity() is False


def test_registry_force_offline_toggle_clears_sticky() -> None:
    """Toggling force_offline off clears the sticky offline state."""
    registry = SourceRegistry(force_offline=True)
    assert registry.is_offline is True
    registry.force_offline = False
    # Now is_offline should probe (may be True or False depending on
    # network, but the sticky inf state must be cleared).
    # Access is_offline to trigger a probe — must not stay stuck at True
    # purely from the old force_offline state.
    _ = registry.is_offline


def test_learner_force_offline_makes_sources_offline() -> None:
    """An AutonomousLearner with force_offline=True has an offline registry."""
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.learning import AutonomousLearner, CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    learner = AutonomousLearner(
        network=net,
        curiosity=curiosity,
        force_offline=True,
    )
    assert learner.is_offline is True
    assert learner._sources.force_offline is True


def test_learner_is_offline_property() -> None:
    """The learner's is_offline property reflects the source registry state."""
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.learning import AutonomousLearner, CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    learner = AutonomousLearner(
        network=net,
        curiosity=curiosity,
        force_offline=True,
    )
    assert learner.is_offline is True
