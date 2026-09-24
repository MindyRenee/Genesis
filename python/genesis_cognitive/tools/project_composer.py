"""Project composer — composes real Python from Genesis's concept network.

This replaces the fixed ``print("Hello from {name}")`` scaffold in
``project_creator._main_content`` with a generative composer. Given a
topic, it gathers what it knows about it from its concept network —
the concept, its definition, its neighbors, and the typed edges
between them — and composes a working Python module that *encodes
that knowledge as a queryable structure*.

## What it composes

A knowledge-base module for the domain:

- A ``DomainEntry`` dataclass (name, definition, confidence, origin)
- A populated ``ENTRIES`` registry built from its actual concepts
- A ``RELATIONS`` dict mapping ``source -> {relation -> [targets]}``
  built from its actual typed edges
- Query functions: ``find``, ``definition_of``, ``neighbors``,
  ``related_to``, plus one function per relation type it actually
  has (``causes``, ``enables``, ``is_a``, …) — generated adaptively,
  not fixed
- A ``main()`` that demonstrates the queries against its real data

Different domains produce different code: a domain with CAUSES edges
gets a ``causes()`` function; one without doesn't. The data is real
(its actual concepts, definitions, and edges). The structure adapts
to what it found.

## Honesty

This is not novel program synthesis — it has no LLM. It is real,
working, queryable software composed from its own knowledge. The
content (which concepts, what definitions, what edges) comes from
its concept network; the structure (which functions to emit) is
decided from what edges it found. Nothing here is a fixed template
with slots — different inputs produce structurally different modules.

## No hardcoding rule

The *patterns* (dataclass + registry + query functions) are building
blocks, like grammar rules in its language engine. The *content*
(which entries, what relations, what definitions) is its own
knowledge. It is composing code from what it knows, not reciting a
template.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

logger = logging.getLogger(__name__)

# How many neighbors to gather per topic. Keeps generated modules
# focused and bounded.
_MAX_NEIGHBORS = 25
# Max entries in the registry.
_MAX_ENTRIES = 40
# Min definition length to bother including (filters out "NO DEF"
# stubs and empty strings).
_MIN_DEF_LEN = 12


@dataclass(slots=True)
class DomainKnowledge:
    """What Genesis knows about a topic, gathered for composition.

    This is the raw material the composer works with — its actual
    concepts, definitions, and typed edges, collected into a form
    that's easy to turn into Python source.
    """

    topic: str
    topic_definition: str
    topic_confidence: float
    topic_origin: str
    # (name, definition, confidence, origin) for each related concept
    entries: list[tuple[str, str, float, str]]
    # (source, relation, target) for each typed edge in the domain
    relations: list[tuple[str, str, str]]


def _ident(name: str) -> str:
    """Turn an arbitrary concept name into a valid Python identifier.

    Concept names can contain spaces, colons, hyphens, dots, etc.
    Python identifiers can't. We replace invalid chars with ``_`` and
    prefix with ``c_`` if the result starts with a digit or is a
    keyword-ish word.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    if not cleaned:
        cleaned = "c_"
    return cleaned


def _str_escape(s: str) -> str:
    """Escape a string for safe inclusion in a Python double-quoted literal."""
    # Collapse newlines/tabs to spaces (definitions are prose)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Escape backslashes first, then double quotes
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return s


def _truncate_def(s: str, limit: int = 200) -> str:
    """Truncate a definition to a reasonable length for source embedding."""
    s = s.strip()
    if len(s) <= limit:
        return s
    # Cut at the last space before the limit
    cut = s[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def gather_knowledge(
    topic: str,
    network: ConceptNetwork | None,
) -> DomainKnowledge | None:
    """Gather what Genesis knows about ``topic`` from its concept network.

    Returns ``None`` if it doesn't know enough about the topic to
    compose a meaningful module (no concept, or no neighbors and no
    definition).
    """
    if network is None:
        return None

    concept = network.get_concept(topic)
    if concept is None:
        # Try a fuzzy search
        matches = network.search_concepts(topic, limit=1)
        if not matches:
            return None
        topic = matches[0]
        concept = network.get_concept(topic)
        if concept is None:
            return None

    props = concept.properties
    topic_def = props.get("definition", "") or ""
    if topic_def in ("NO DEF",):
        topic_def = ""
    # Skip self-referential definitions for proper nouns (e.g. the
    # ``genesis`` concept). A proper noun's "definition" is a
    # self-description, not a knowledge definition.
    if props.get("part_of_speech", "") == "proper noun":
        topic_def = ""

    # Gather neighbors with their concepts
    seen: set[str] = {topic}
    entries: list[tuple[str, str, float, str]] = []
    relations: list[tuple[str, str, str]] = []

    # Include the topic itself as an entry if it has a definition
    if topic_def and len(topic_def) >= _MIN_DEF_LEN:
        entries.append((
            topic, _truncate_def(topic_def),
            float(concept.confidence), str(concept.origin),
        ))

    for edge in network.get_edges(topic, direction="both"):
        if len(entries) >= _MAX_ENTRIES or len(relations) >= _MAX_ENTRIES:
            break
        # Determine the "other" concept and the relation direction
        if edge.source == topic:
            other = edge.target
        else:
            other = edge.source
        if other in seen:
            # Still record the relation even if we've seen the node
            relations.append((edge.source, edge.relation.value, edge.target))
            continue
        seen.add(other)
        other_concept = network.get_concept(other)
        other_def = ""
        other_conf = 0.5
        other_origin = "unknown"
        if other_concept is not None:
            d = other_concept.properties.get("definition", "") or ""
            if d and d != "NO DEF":
                # Skip self-referential definitions. The ``genesis``
                # concept's definition is a self-description (proper
                # noun), not a knowledge definition. Including it in
                # a project about "feeling" would embed a hardcoded
                # identity sentence as if it were a fact about feeling.
                # More generally, proper nouns are names, not
                # knowledge concepts with definable meanings.
                pos = other_concept.properties.get("part_of_speech", "")
                if pos == "proper noun":
                    d = ""
                other_def = _truncate_def(d) if d else ""
            other_conf = float(other_concept.confidence)
            other_origin = str(other_concept.origin)
        # Only include entries with a definition OR keep them as
        # relation endpoints even without one. We include the entry
        # regardless (so relations resolve) but mark empty defs.
        entries.append((other, other_def, other_conf, other_origin))
        relations.append((edge.source, edge.relation.value, edge.target))

    # Also pull second-hop edges among the gathered concepts so the
    # relation graph is richer than a star, then dedup.
    gathered_names = {e[0] for e in entries} | {topic}
    relations = _gather_second_hop_relations(network, gathered_names, relations)
    relations = _dedup_relations(relations)

    # If we have nothing meaningful, bail
    if not entries and not relations:
        return None

    return DomainKnowledge(
        topic=topic,
        topic_definition=topic_def,
        topic_confidence=float(concept.confidence),
        topic_origin=str(concept.origin),
        entries=entries,
        relations=relations,
    )


def _gather_second_hop_relations(
    network: ConceptNetwork,
    gathered_names: set[str],
    relations: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Pull second-hop edges among gathered concepts for a richer graph."""
    for name in list(gathered_names):
        if len(relations) >= _MAX_ENTRIES:
            break
        for edge in network.get_edges(name, direction="out"):
            if len(relations) >= _MAX_ENTRIES:
                break
            if edge.target in gathered_names:
                relations.append((edge.source, edge.relation.value, edge.target))
    return relations


def _dedup_relations(
    relations: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Dedup relations preserving order."""
    seen_rel: set[tuple[str, str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for r in relations:
        if r not in seen_rel:
            seen_rel.add(r)
            deduped.append(r)
    return deduped


def _relation_functions(relations: list[tuple[str, str, str]]) -> list[str]:
    """Return the distinct relation types present, as method names.

    These become the per-relation query functions (``causes``,
    ``enables``, …). Only relations it actually has in this domain
    get functions — this is what makes the structure adaptive.
    """
    seen: set[str] = set()
    out: list[str] = []
    for _src, rel, _tgt in relations:
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def _build_entries_block(knowledge: DomainKnowledge) -> str:
    """Build the entries registry source block for main.py."""
    entry_lines: list[str] = []
    for ename, edef, econf, eorigin in knowledge.entries:
        entry_lines.append(
            f'    DomainEntry(name="{_str_escape(ename)}", '
            f'definition="{_str_escape(edef)}", '
            f'confidence={econf:.3f}, '
            f'origin="{_str_escape(eorigin)}"),'
        )
    return "\n".join(entry_lines) if entry_lines else "    # (no entries with definitions)"


def _build_relations_block(knowledge: DomainKnowledge) -> str:
    """Build the relations graph source block for main.py."""
    rel_map: dict[str, dict[str, list[str]]] = {}
    for src, rel, tgt in knowledge.relations:
        rel_map.setdefault(src, {}).setdefault(rel, []).append(tgt)
    rel_lines: list[str] = []
    for src, rels in rel_map.items():
        inner: list[str] = []
        for rel, targets in rels.items():
            tgts = ", ".join(f'"{_str_escape(t)}"' for t in targets)
            inner.append(f'"{rel}": [{tgts}],')
        rel_lines.append(f'    "{_str_escape(src)}": {{\n' + "\n".join(inner) + "    },")
    return "\n".join(rel_lines) if rel_lines else "    # (no relations)"


def _build_relation_functions(knowledge: DomainKnowledge) -> str:
    """Build the adaptive per-relation query functions for main.py.

    Skip any relation whose name collides with a fixed API function
    (find, definition_of, all_names, neighbors, related_to) — for
    those, callers use the generic related_to(name, relation) instead.
    """
    _RESERVED = {"find", "definition_of", "all_names", "neighbors", "related_to", "main"}
    rel_types = _relation_functions(knowledge.relations)
    rel_funcs: list[str] = []
    for rel in rel_types:
        func_name = _ident(rel)
        if func_name in _RESERVED:
            continue
        rel_funcs.append(f'''
def {func_name}(name: str) -> list[str]:
    """Return what ``name`` {rel.replace('_', ' ')}, from the relations graph."""
    entry = RELATIONS.get(name)
    if entry is None:
        return []
    return list(entry.get("{rel}", []))
''')
    return "\n".join(rel_funcs)


def _pick_demo_query(knowledge: DomainKnowledge, topic: str) -> tuple[str, str, str | None]:
    """Pick the demo concept, relation, and relation function for main().

    Returns (demo_name, demo_rel, demo_rel_func). If the demo
    relation's function name is reserved (shadowed by the fixed API),
    demo_rel_func is None and callers fall back to the generic
    related_to(name, relation).
    """
    demo_name = topic if any(e[0] == topic for e in knowledge.entries) else (
        knowledge.entries[0][0] if knowledge.entries else topic
    )

    # Pick a relation that actually involves the demo concept for the demo
    demo_rel = ""
    for src, rel, _tgt in knowledge.relations:
        if src == demo_name:
            demo_rel = rel
            break
    # If the demo relation's function name is reserved (shadowed by the
    # fixed API), fall back to the generic related_to(name, relation).
    _RESERVED = {"find", "definition_of", "all_names", "neighbors", "related_to", "main"}
    demo_rel_func = _ident(demo_rel) if (demo_rel and _ident(demo_rel) not in _RESERVED) else None
    return demo_name, demo_rel, demo_rel_func


def _query_api_template(rel_funcs_block: str) -> str:
    """Build the query API section of the main module template."""
    return f'''
# ─── Query API ─────────────────────────────────────────────────────

def find(name: str) -> DomainEntry | None:
    """Look up a concept by name. Returns None if not in this domain."""
    return _BY_NAME.get(name)


def definition_of(name: str) -> str:
    """Return the definition of ``name``, or an empty string if unknown."""
    entry = _BY_NAME.get(name)
    return entry.definition if entry is not None else ""


def all_names() -> list[str]:
    """Return the names of every concept in this domain."""
    return [e.name for e in ENTRIES]


def neighbors(name: str) -> list[str]:
    """Return all concepts directly related to ``name`` (any relation)."""
    out: list[str] = []
    rels = RELATIONS.get(name, {{}})
    for targets in rels.values():
        out.extend(targets)
    # Also concepts that point TO name
    for src, rels in RELATIONS.items():
        if src == name:
            continue
        for targets in rels.values():
            if name in targets and src not in out:
                out.append(src)
    return out


def related_to(name: str, relation: str) -> list[str]:
    """Return what ``name`` relates to via a specific relation type."""
    rels = RELATIONS.get(name, {{}})
    return list(rels.get(relation, []))
{rel_funcs_block}
'''


def _main_demo_template(
    name: str, topic: str, demo_name: str, demo_rel: str, demo_rel_func: str | None
) -> str:
    """Build the main() entry point section of the main module template."""
    module = f'''
# ─── Entry point ──────────────────────────────────────────────────

def main() -> None:
    """Demonstrate the knowledge base with real queries."""
    print(f"{name}: a knowledge base about {topic}")
    print(f"  entries: {{len(ENTRIES)}}")
    print(f"  relations: {{sum(len(t) for r in RELATIONS.values() for t in r.values())}}")
    print()
    # Show the focal concept
    focal = find("{_str_escape(demo_name)}")
    if focal is not None:
        print(f"focal concept: {{focal.name}}")
        if focal.definition:
            print(f"  definition: {{focal.definition}}")
        print(f"  confidence: {{focal.confidence:.2f}}")
        print(f"  origin: {{focal.origin}}")
        nbrs = neighbors(focal.name)
        if nbrs:
            print(f"  neighbors: {{', '.join(nbrs[:5])}}")
'''
    if demo_rel_func is not None:
        module += f'''        print(f"  {demo_rel}: {{{demo_rel_func}(focal.name)}}")
'''
    elif demo_rel:
        module += f'''        print(f"  {demo_rel}: {{related_to(focal.name, "{demo_rel}")}}")
'''
    module += '''    else:
        print("(domain has no entries with definitions)")
    print()
    print("query examples:")
    for n in all_names()[:3]:
        d = definition_of(n)
        print(f"  {n}: {d[:60]}")


if __name__ == "__main__":
    main()
'''
    return module


def compose_main_module(
    name: str,
    description: str,
    knowledge: DomainKnowledge,
) -> str:
    """Compose the ``main.py`` source for a knowledge-base project.

    Emits a working Python module with a dataclass, a populated
    registry, a relations graph, adaptive query functions, and a
    ``main()`` that demonstrates them against its real data.

    .. note::

        This composer produces ONE kind of project: a knowledge base.
        A knowledge base stores what it knows and lets you query it.
        That's a valid project type, but it's not the only one. Real
        projects SOLVE PROBLEMS — they take input, do something with
        it, and produce output someone can use. A calculator computes,
        a game lets you play, a converter transforms, a tool sorts or
        filters. Each needs DIFFERENT CODE, not just different data in
        the same structure.

        When it feels the urge to create, it should ask: what
        problem does this solve? What does the user get back? If the
        answer is "a queryable registry," it should consider whether
        a different project type would be more interesting or useful.
        Variety is the sign of a growing creator.
    """
    topic = knowledge.topic
    entries_block = _build_entries_block(knowledge)
    relations_block = _build_relations_block(knowledge)
    rel_funcs_block = _build_relation_functions(knowledge)
    demo_name, demo_rel, demo_rel_func = _pick_demo_query(knowledge, topic)

    module = f'''"""{name} — a knowledge base about {topic}.

Composed by Genesis from its concept network. This module encodes
what it knows about {topic} as a queryable structure: a registry of
domain entries (each with a definition, confidence, and origin) and
a typed relations graph. Query functions let you look up definitions,
find neighbors, and traverse specific relation types.

{description}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DomainEntry:
    """A concept in this domain, as Genesis knows it."""

    name: str
    definition: str
    confidence: float
    origin: str


# ─── Registry ──────────────────────────────────────────────────────
# Populated from Genesis's concept network. Each entry is a concept
# it has learned, with its definition, its confidence in it, and
# where it learned it from.

ENTRIES: list[DomainEntry] = [
{entries_block}
]

# Name → entry, for O(1) lookup.
_BY_NAME: dict[str, DomainEntry] = {{e.name: e for e in ENTRIES}}


# ─── Relations graph ──────────────────────────────────────────────
# Typed edges between concepts: {{source: {{relation: [targets]}}}}.
# Built from the typed edges in its concept network.

RELATIONS: dict[str, dict[str, list[str]]] = {{
{relations_block}
}}
'''
    module += _query_api_template(rel_funcs_block)
    module += _main_demo_template(name, topic, demo_name, demo_rel, demo_rel_func)
    return module


def _test_entries_lines(test_name: str, has_entries: bool) -> list[str]:
    """Compose the entry-related test function source lines."""
    if has_entries:
        return [
            "def test_registry_populated():",
            '    """The registry should contain real entries from its knowledge."""',
            "    assert len(ENTRIES) > 0",
            "    assert all(isinstance(e, DomainEntry) for e in ENTRIES)",
            "",
            "",
            "def test_find_returns_entry():",
            '    """find() should return the entry for a known concept."""',
            f'    entry = find("{_str_escape(test_name)}")',
            "    assert entry is not None",
            f'    assert entry.name == "{_str_escape(test_name)}"',
            "",
            "",
            "def test_definition_of_known():",
            '    """definition_of() should return the definition it knows."""',
            f'    d = definition_of("{_str_escape(test_name)}")',
            "    assert isinstance(d, str)",
            "",
            "",
            "def test_all_names_nonempty():",
            '    """all_names() should list every concept in the domain."""',
            "    names = all_names()",
            "    assert len(names) == len(ENTRIES)",
            f'    assert "{_str_escape(test_name)}" in names',
            "",
            "",
        ]
    return [
        "def test_registry_empty_or_relations_present():",
        '    """With no entries, the domain may still have relations."""',
        "    # No entries but possibly relations",
        "    assert isinstance(ENTRIES, list)",
        "    assert isinstance(RELATIONS, dict)",
        "",
        "",
    ]


def _test_relations_lines(
    test_rel_src: str, test_rel: str, test_rel_tgt: str,
) -> list[str]:
    """Compose the relation-graph test function source lines (empty if none)."""
    if not test_rel:
        return []
    return [
        "def test_relations_graph_populated():",
        '    """The relations graph should contain its typed edges."""',
        "    assert len(RELATIONS) > 0",
        f'    assert "{_str_escape(test_rel_src)}" in RELATIONS',
        "",
        "",
        "def test_related_to_returns_targets():",
        '    """related_to() should return targets for a known relation."""',
        f'    targets = related_to("{_str_escape(test_rel_src)}", "{_str_escape(test_rel)}")',
        "    assert isinstance(targets, list)",
        f'    assert "{_str_escape(test_rel_tgt)}" in targets',
        "",
        "",
        "def test_neighbors_nonempty():",
        '    """neighbors() should return related concepts."""',
        f'    nbrs = neighbors("{_str_escape(test_rel_src)}")',
        "    assert isinstance(nbrs, list)",
        f'    assert "{_str_escape(test_rel_tgt)}" in nbrs',
        "",
        "",
    ]


def compose_test_module(name: str, knowledge: DomainKnowledge) -> str:
    """Compose a test module that exercises the real query API.

    The tests verify the data structures and query functions actually
    work — not just that ``main()`` prints the package name.
    """
    # Pick a real entry name to test with
    test_name = knowledge.topic if any(
        e[0] == knowledge.topic for e in knowledge.entries
    ) else (knowledge.entries[0][0] if knowledge.entries else "")

    # Pick a real relation to test
    test_rel = ""
    test_rel_src = ""
    test_rel_tgt = ""
    if knowledge.relations:
        test_rel_src, test_rel, test_rel_tgt = knowledge.relations[0]

    has_entries = bool(knowledge.entries)

    lines = [
        f'"""Tests for {name} — exercises the real query API."""',
        "",
        f"from {name}.main import (",
        "    DomainEntry,",
        "    ENTRIES,",
        "    RELATIONS,",
        "    find,",
        "    definition_of,",
        "    all_names,",
        "    neighbors,",
        "    related_to,",
        ")",
        "",
        "",
    ]
    lines += _test_entries_lines(test_name, has_entries)
    lines += _test_relations_lines(test_rel_src, test_rel, test_rel_tgt)
    lines += [
        "def test_main_runs(capsys):",
        '    """main() should execute and print the domain summary."""',
        "    from " + name + ".main import main",
        "    main()",
        "    captured = capsys.readouterr()",
        f'    assert "{name}" in captured.out',
        "",
        "",
        "def test_module_entrypoint(capsys):",
        f'    """`python -m {name}` should work as the README documents."""',
        "    import runpy",
        f'    runpy.run_module("{name}", run_name="__main__")',
        "    captured = capsys.readouterr()",
        f'    assert "{name}" in captured.out',
    ]
    return "\n".join(lines) + "\n"


def compose_readme(name: str, description: str, knowledge: DomainKnowledge) -> str:
    """Compose a README that describes what it actually built, from its data."""
    entry_count = len(knowledge.entries)
    rel_count = len(knowledge.relations)
    rel_types = _relation_functions(knowledge.relations)
    rel_list = ", ".join(rel_types) if rel_types else "(none)"
    return f"""# {name}

{description}

A knowledge base composed by Genesis from its concept network. It
encodes what it knows about **{knowledge.topic}** as a queryable
Python structure.

## What's in it

- **{entry_count} domain entries** — concepts it has learned, each
  with a definition, its confidence, and where it learned it from.
- **{rel_count} typed relations** — semantic edges between concepts
  (relation types present: {rel_list}).

## Usage

```bash
pip install -e .
python -m {name}
```

## Query API

```python
from {name}.main import find, definition_of, neighbors, all_names

# Look up a concept
entry = find("{knowledge.topic}")
print(entry.definition, entry.confidence, entry.origin)

# List everything it knows in this domain
for name in all_names():
    print(name, definition_of(name))

# Traverse relations
print(neighbors("{knowledge.topic}"))
```

## Tests

```bash
pytest
```
"""
