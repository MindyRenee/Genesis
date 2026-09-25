"""Persistence — Genesis's memory across restarts.

This module saves and loads Genesis's learned state to disk so it
remembers what it's learned between conversations.

# What persists

- **Concept network**: all concepts, aliases, relationships, confidence
- **Reflection state**: insights, interaction count, mood history
- **Narrative**: life events, chapters, personality snapshots
- **Self-model**: personality drift (if it's changed from defaults)
- **Self-directed learner**: cumulative stats, learning log, studied concepts

# What doesn't persist

- Working memory (current conversation context — that's ephemeral)
- Emotional state (that's in the Rust daemon's mmap'd state file)
- LTM episodes (those are already persisted by the Rust daemon)

# Format

JSON files in the data directory:
- `cognitive_state.json` — all cognitive mind state in one file

Using a single file keeps it simple. The file is written atomically
(write to temp, rename) so a crash can't corrupt it.

# Performance

Uses ``orjson`` (Rust-backed JSON library) when available for 6x
faster serialization. Falls back to stdlib ``json`` if orjson is
not installed.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
import zlib
from collections import deque
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False

from .concepts import (
    Concept,
    ConceptCategory,
    ConceptModality,
    ConceptNetwork,
    Edge,
    RelationType,
    _normalize_id,
)
from .emotional_regulator import (
    AllostaticLoadTracker,
    AllostaticState,
    HPAAxis,
    HPAState,
)
from .growth_ledger import GrowthLedger
from .learning import (
    AutonomousLearner,
    HebbianPlasticity,
    PredictiveCodingLayer,
    TDLearner,
    TDTransition,
)
from .memory import (
    AttractorNetwork,
    AttractorPattern,
    EmotionalMemory,
    EmotionalMemorySystem,
    ProceduralMemory,
    ReviewRecord,
    Skill,
    SkillStrategy,
    SpacedRepetitionScheduler,
)
from .narrative import LifeChapter, LifeEvent, NarrativeEngine
from .reasoning import (
    KnowledgeLevel,
    TaskCompetence,
    TheoryOfMind,
    UserBelief,
    UserModel,
)
from .self import (
    ErrorMonitor,
    Insight,
    LearningEvent,
    PersonalityTraits,
    PredictionError,
    ReflectionEngine,
    SelfDirectedLearner,
    SelfImprovementEngine,
    SelfModel,
)

logger = logging.getLogger(__name__)

# Gzip compression for cognitive_state.json. The file can grow to 10+ MB
# with a large concept network; gzip reduces it to ~1-2 MB, cutting disk
# write time and CPU proportionally. The format is auto-detected on load
# by checking the gzip magic bytes (0x1f 0x8b), so old uncompressed files
# are still readable.
_USE_GZIP = True
_GZIP_MAGIC = b"\x1f\x8b"

if TYPE_CHECKING:
    # Several self modules import from persistence at module level, so we
    # use TYPE_CHECKING to avoid a circular import. The actual classes
    # are imported lazily inside the functions that need them.
    from .bug_reporter import BugReporter
    from .self import (
        CognitiveTrajectoryModel,
        DevelopmentalTracker,
        EmergentIdentity,
        EmergentIdentitySource,
    )


def _snapshot[T](
    src: Callable[[], Iterable[T]] | Iterable[T], attempts: int = 8
) -> list[T]:
    """Materialize a collection that other threads may be mutating.

    Autosave and shutdown serialization run concurrently with
    background workers (volition actions, autonomous learning, the
    think-worker) that mutate these collections. Iterating a live dict
    view raises ``RuntimeError`` ("dictionary changed size during
    iteration"); iterating a live list yields a torn snapshot. Retry
    the materialization — each attempt is a single fast C-level pass —
    so the save either gets a consistent copy or, after ``attempts``
    failures, surfaces the error to the caller's existing save-failed
    path rather than crashing the thread.

    ``src`` may be a collection (iterated directly) or a callable that
    produces a fresh view each attempt — pass the bound method form
    (``d.items``/``d.values``) for dicts so each retry re-views the
    current dict.
    """
    for _ in range(attempts - 1):
        try:
            return list(src() if callable(src) else src)
        except RuntimeError:
            continue
    return list(src() if callable(src) else src)


def save_state(
    data_dir: str,
    network: ConceptNetwork,
    reflection: ReflectionEngine,
    narrative: NarrativeEngine,
    self_model: SelfModel,
    *,
    predictive_coding: PredictiveCodingLayer | None = None,
    theory_of_mind: TheoryOfMind | None = None,
    procedural_memory: ProceduralMemory | None = None,
    task_competence: TaskCompetence | None = None,
    spaced_repetition: SpacedRepetitionScheduler | None = None,
    td_learner: TDLearner | None = None,
    emotional_memory: EmotionalMemorySystem | None = None,
    attractor: AttractorNetwork | None = None,
    error_monitor: ErrorMonitor | None = None,
    emergent_identity: EmergentIdentity | None = None,
    emergent_identity_sources: list[EmergentIdentitySource] | None = None,
    hpa_axis: HPAAxis | None = None,
    allostatic_load: AllostaticLoadTracker | None = None,
    self_directed_learner: SelfDirectedLearner | None = None,
    autonomous_learner: AutonomousLearner | None = None,
    self_improvement: SelfImprovementEngine | None = None,
    growth_ledger: GrowthLedger | None = None,
    plasticity: HebbianPlasticity | None = None,
    developmental_tracker: DevelopmentalTracker | None = None,
    bug_reporter: BugReporter | None = None,
    sleep_state: dict[str, Any] | None = None,
    memory_records: dict[str, Any] | None = None,
    cognitive_trajectory: CognitiveTrajectoryModel | None = None,
    dream_synthesis: Any = None,
    inner_life_state: dict[str, Any] | None = None,
    world_state: dict[str, Any] | None = None,
) -> None:
    """Save Genesis's cognitive state to disk.

    Args:
        data_dir: The data directory (same one the daemon uses).
        network: The concept network to save.
        reflection: The reflection engine state to save.
        narrative: The narrative engine state to save.
        self_model: The self-model (for personality drift).
        predictive_coding: The predictive coding layer (Bayesian models).
        theory_of_mind: The user model.
        procedural_memory: Learned skills and habits.
        task_competence: Learned task schemas, affordances, and skills.
        spaced_repetition: The review schedule for concepts.
        td_learner: The value function and reward history.
        emotional_memory: Emotional tags associated with memories.
        attractor: The stored patterns in the attractor network.
        error_monitor: The error history and adjusted thresholds.
        emergent_identity: The synthesized emergent identity.
        emergent_identity_sources: Accumulated experience identity sources.
        hpa_axis: The HPA axis cascade state.
        allostatic_load: The cumulative allostatic load tracker.
        self_directed_learner: The self-directed learning engine state.
        sleep_state: Sleep state (is_sleeping, user_initiated, cycle
            tracker position) so sleep survives restarts.
        memory_records: Python-side memory metadata (consolidation
            state, forgetting, source tags) from
            :meth:`MemoryEngine.serialize_records`.
        world_state: Its external world (presences, recent events,
            social clock) from :meth:`OuterWorld.to_dict`.
    """
    state: dict[str, Any] = {
        "version": 2,
        "concept_network": _serialize_network(network),
        "reflection": _serialize_reflection(reflection),
        "narrative": _serialize_narrative(narrative),
        "self_model": _serialize_self_model(self_model),
    }

    _add_optional_state(
        state,
        predictive_coding=predictive_coding,
        theory_of_mind=theory_of_mind,
        procedural_memory=procedural_memory,
        task_competence=task_competence,
        spaced_repetition=spaced_repetition,
        td_learner=td_learner,
        emotional_memory=emotional_memory,
        attractor=attractor,
        error_monitor=error_monitor,
        emergent_identity=emergent_identity,
        emergent_identity_sources=emergent_identity_sources,
        hpa_axis=hpa_axis,
        allostatic_load=allostatic_load,
        self_directed_learner=self_directed_learner,
        autonomous_learner=autonomous_learner,
        self_improvement=self_improvement,
        growth_ledger=growth_ledger,
        plasticity=plasticity,
        developmental_tracker=developmental_tracker,
        bug_reporter=bug_reporter,
        sleep_state=sleep_state,
        memory_records=memory_records,
        cognitive_trajectory=cognitive_trajectory,
        dream_synthesis=dream_synthesis.to_dict() if dream_synthesis is not None else None,
        inner_life_state=inner_life_state,
        world_state=world_state,
    )

    _atomic_write_state(data_dir, state)


def _atomic_write_state(data_dir: str, state: dict[str, Any]) -> None:
    """Write state dict to disk atomically (write to temp, then rename).

    Uses gzip compression when ``_USE_GZIP`` is True to reduce file size
    from ~12 MB to ~1-2 MB, cutting disk write time and CPU overhead.
    """
    # Owner-only: the data dir holds the IPC socket (the daemon's
    # unauthenticated control channel) and the full cognitive state.
    # mode= only applies when we create the directory — existing
    # directories keep their permissions.
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    path = os.path.join(data_dir, "cognitive_state.json")

    # Serialize with orjson (6x faster) or stdlib fallback
    if _HAS_ORJSON:
        data_bytes = orjson.dumps(state, default=str)
    else:
        data_bytes = json.dumps(state, default=str).encode("utf-8")

    # Compress with gzip if enabled
    if _USE_GZIP:
        data_bytes = gzip.compress(data_bytes, compresslevel=5)

    # Atomic write: write to temp file, then rename.
    # os.replace is the atomic-overwrite primitive on both POSIX and
    # Windows; os.rename cannot overwrite an existing file on Windows.
    fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".tmp", prefix="cognitive_")
    try:
        with os.fdopen(fd, "wb") as f:
            fd = -1
            f.write(data_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        logger.exception(f"atomic write failed: {e}")
        # If os.fdopen failed before taking ownership, the fd is
        # still open. If it succeeded, the with block already closed
        # it. Try closing and ignore the error if already closed.
        try:
            if fd >= 0:
                os.close(fd)
        except OSError as e:
            logger.debug(f'silent except: {e}')
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError as oe:
            logger.debug(repr(oe))
        raise

    dir_fd = -1
    try:
        dir_fd = os.open(data_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
            dir_fd = -1
    except OSError as e:
        # The state file itself was already atomically replaced above;
        # a directory-fsync failure only risks durability on power loss,
        # not correctness. Log and continue rather than crashing the save.
        logger.warning(f"directory fsync failed after atomic write: {e}")
        if dir_fd >= 0:
            try:
                os.close(dir_fd)
            except OSError:
                pass


def _add_optional_state(
    state: dict[str, Any],
    *,
    predictive_coding: PredictiveCodingLayer | None = None,
    theory_of_mind: TheoryOfMind | None = None,
    procedural_memory: ProceduralMemory | None = None,
    task_competence: TaskCompetence | None = None,
    spaced_repetition: SpacedRepetitionScheduler | None = None,
    td_learner: TDLearner | None = None,
    emotional_memory: EmotionalMemorySystem | None = None,
    attractor: AttractorNetwork | None = None,
    error_monitor: ErrorMonitor | None = None,
    emergent_identity: EmergentIdentity | None = None,
    emergent_identity_sources: list[EmergentIdentitySource] | None = None,
    hpa_axis: HPAAxis | None = None,
    allostatic_load: AllostaticLoadTracker | None = None,
    self_directed_learner: SelfDirectedLearner | None = None,
    autonomous_learner: AutonomousLearner | None = None,
    self_improvement: SelfImprovementEngine | None = None,
    growth_ledger: GrowthLedger | None = None,
    plasticity: HebbianPlasticity | None = None,
    developmental_tracker: DevelopmentalTracker | None = None,
    bug_reporter: BugReporter | None = None,
    sleep_state: dict[str, Any] | None = None,
    memory_records: dict[str, Any] | None = None,
    cognitive_trajectory: CognitiveTrajectoryModel | None = None,
    dream_synthesis: Any = None,
    inner_life_state: dict[str, Any] | None = None,
    world_state: dict[str, Any] | None = None,
) -> None:
    """Add optional state fields to the state dict."""
    if predictive_coding is not None:
        state["predictive_coding"] = _serialize_predictive_coding(predictive_coding)
    if theory_of_mind is not None:
        state["theory_of_mind"] = _serialize_theory_of_mind(theory_of_mind)
    if procedural_memory is not None:
        state["procedural_memory"] = _serialize_procedural_memory(procedural_memory)
    if task_competence is not None:
        state["task_competence"] = _serialize_task_competence(task_competence)
    if spaced_repetition is not None:
        state["spaced_repetition"] = _serialize_spaced_repetition(spaced_repetition)
    if td_learner is not None:
        state["td_learner"] = _serialize_td_learner(td_learner)
    if emotional_memory is not None:
        state["emotional_memory"] = _serialize_emotional_memory(emotional_memory)
    if attractor is not None:
        state["attractor"] = _serialize_attractor(attractor)
    if error_monitor is not None:
        state["error_monitor"] = _serialize_error_monitor(error_monitor)
    if emergent_identity is not None or emergent_identity_sources is not None:
        state["emergent_identity"] = _serialize_emergent_identity(
            emergent_identity, emergent_identity_sources
        )
    if hpa_axis is not None:
        state["hpa_axis"] = _serialize_hpa_axis(hpa_axis)
    if allostatic_load is not None:
        state["allostatic_load"] = _serialize_allostatic_load(allostatic_load)
    if self_directed_learner is not None:
        state["self_directed_learner"] = _serialize_self_directed_learner(self_directed_learner)
    if autonomous_learner is not None:
        state["autonomous_learner"] = autonomous_learner.serialize_state()
    if self_improvement is not None:
        state["self_improvement"] = self_improvement.to_dict()
    if growth_ledger is not None:
        state["growth_ledger"] = growth_ledger.to_dict()
    if plasticity is not None:
        state["plasticity"] = plasticity.save_state()
    if developmental_tracker is not None:
        state["developmental_tracker"] = developmental_tracker.to_dict()
    if bug_reporter is not None:
        state["bug_reporter"] = bug_reporter.to_dict()
    if sleep_state is not None:
        state["sleep_state"] = sleep_state
    if memory_records is not None:
        state["memory_records"] = memory_records
    if cognitive_trajectory is not None:
        state["cognitive_trajectory"] = cognitive_trajectory.to_dict()
    if dream_synthesis is not None:
        state["dream_synthesis"] = dream_synthesis
    if inner_life_state is not None:
        state["inner_life_state"] = inner_life_state
    if world_state is not None:
        state["world_state"] = world_state


def load_state(data_dir: str) -> dict[str, Any] | None:
    """Load Genesis's cognitive state from disk.

    Returns None if no saved state exists.

    Auto-detects gzip-compressed files by checking the magic bytes,
    so both old uncompressed files and new compressed files are
    readable.
    """
    path = os.path.join(data_dir, "cognitive_state.json")
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        if os.path.lexists(path):
            raise OSError(f"Cannot load cognitive_state from {path}: broken symlink") from None
        return None

    try:
        # Auto-detect gzip by magic bytes (0x1f 0x8b)
        if raw[:2] == _GZIP_MAGIC:
            raw = gzip.decompress(raw)
        data = orjson.loads(raw) if _HAS_ORJSON else json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("saved state must be an object")
        if type(data.get("version", 1)) is not int or data.get("version", 1) not in (1, 2):
            raise ValueError("unsupported saved state version")
        return data
    except (OSError, EOFError, ValueError, zlib.error) as e:
        # Corrupt or truncated state must not be treated as first boot.
        # Propagate the failure so startup cannot overwrite the saved
        # developmental record with a fresh network. Recovery requires
        # an explicit repair or restoration of the original state.
        raise OSError(f"Cannot load cognitive_state from {path}: {e}") from e


def _prune_dictionary_concepts(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove low-value dictionary concepts that bloat the network.

    Old versions imported the entire WordNet dictionary (100k+ concepts
    with origin "dictionary") which made the network too slow to load
    and query. Genesis re-adds words it actually encounters with
    meaningful origins (conversation, learned, curriculum, etc.).
    """
    _PRUNE_ORIGINS = {"dictionary"}
    if not any(c.get("origin") in _PRUNE_ORIGINS for c in concepts):
        return concepts

    kept = []
    pruned = 0
    for c in concepts:
        if c.get("origin") in _PRUNE_ORIGINS:
            pruned += 1
        else:
            kept.append(c)
    if pruned:
        import logging

        logging.getLogger(__name__).info(
            "Pruned %d '%s' concepts during restore", pruned, "dictionary"
        )
    return kept


def _merge_duplicate_sense_suffixes(
    concepts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Merge #2 disambiguated concepts back into their base concepts.

    Old versions of _find_matching_sense created a #2 disambiguated
    concept whenever a re-learning pass extracted no definition for a
    concept that already had one (or vice versa). This produced
    hundreds of duplicate #2 concepts with redundant edges.

    Returns (cleaned_concepts, id_remap) where id_remap maps old #2
    IDs to normalized base IDs for edge remapping.
    """
    import re as _re

    _sense_suffix = _re.compile(r"#\d+$")
    id_remap: dict[str, str] = {}

    # Build definition lookup
    definitions: dict[str, str] = {}
    for c in concepts:
        definition = c.get("properties", {}).get("definition", "")
        definitions[c["id"]] = definition.strip().lower() if isinstance(definition, str) else ""

    # Identify which #2 IDs are duplicates of their base
    sense_definitions: dict[str, set[str]] = {}
    for concept_id, definition in definitions.items():
        if definition:
            base_id = _sense_suffix.sub("", concept_id)
            sense_definitions.setdefault(base_id, set()).add(definition)
    dup2_ids: set[str] = set()
    for concept_id, definition in definitions.items():
        base_id = _sense_suffix.sub("", concept_id)
        if base_id == concept_id or base_id not in definitions:
            continue
        base_definition = definitions[base_id]
        if definition and base_definition:
            if definition == base_definition:
                dup2_ids.add(concept_id)
        elif len(sense_definitions.get(base_id, set())) <= 1:
            dup2_ids.add(concept_id)

    if not dup2_ids:
        return concepts, id_remap

    # Build remap: #2_id → normalized base_id
    for c in concepts:
        cid = c.get("id")
        if cid in dup2_ids:
            base_id = _sense_suffix.sub("", cid)
            id_remap[cid] = _normalize_id(base_id)

    # Merge duplicates into their base concepts
    concept_by_id = {c.get("id"): c for c in concepts if c.get("id")}
    for dup_id in dup2_ids:
        base_id = _sense_suffix.sub("", dup_id)
        base_c = concept_by_id.get(base_id)
        dup_c = concept_by_id[dup_id]
        if base_c is None:
            continue
        # Merge: take the higher confidence, union aliases/columns,
        # fill in empty definition from the non-empty one.
        base_c["confidence"] = max(
            base_c.get("confidence", 0.5),
            dup_c.get("confidence", 0.5),
        )
        base_c.setdefault("aliases", [])
        for a in dup_c.get("aliases", []):
            if a not in base_c["aliases"]:
                base_c["aliases"].append(a)
        base_c.setdefault("columns", [])
        for col in dup_c.get("columns", []):
            if col not in base_c["columns"]:
                base_c["columns"].append(col)
        base_c.setdefault("properties", {})
        dup_props = dup_c.get("properties", {})
        base_def = base_c["properties"].get("definition", "")
        dup_def = dup_props.get("definition", "")
        if not base_def and dup_def:
            base_c["properties"]["definition"] = dup_def
        # Merge other properties from the duplicate
        for k, v in dup_props.items():
            if k != "definition" and k not in base_c["properties"]:
                base_c["properties"][k] = v

    # Remove #2 concepts from the restore list
    cleaned = [c for c in concepts if c["id"] not in dup2_ids]
    import logging

    logging.getLogger(__name__).info(
        "Merged %d duplicate '#2' concepts into their base concepts",
        len(dup2_ids),
    )
    return cleaned, id_remap


def restore_network(network: ConceptNetwork, data: dict[str, Any]) -> None:
    """Restore concept network from saved data.

    Concept IDs and edge references are normalized using
    ``_normalize_id`` so that old save files (which may contain
    non-normalized IDs like "micro-animal") are compatible with
    the normalized ID scheme.
    """
    concepts = data.get("concepts", [])
    edges = data.get("edges", [])

    concepts = _prune_dictionary_concepts(concepts)
    concepts, id_remap = _merge_duplicate_sense_suffixes(concepts)

    _restore_concepts(network, concepts, id_remap)
    _restore_edges(network, edges, id_remap)


def _restore_concepts(
    network: ConceptNetwork,
    concepts: list[dict[str, Any]],
    id_remap: dict[str, str],
) -> None:
    """Restore concept entries from saved data."""
    from .concepts import _normalize_id, strip_sense_suffix

    for c_data in concepts:
        # Restore category, falling back to UNKNOWN for older saves
        try:
            category = ConceptCategory(c_data.get("category", "unknown"))
        except ValueError:
            category = ConceptCategory.UNKNOWN
        # Restore modality, falling back to UNKNOWN for older saves
        try:
            modality = ConceptModality(c_data.get("modality", "unknown"))
        except ValueError:
            modality = ConceptModality.UNKNOWN

        old_id = c_data["id"]
        new_id = _normalize_id(old_id)
        id_remap[old_id] = new_id
        confidence = c_data.get("confidence")
        if confidence is None:
            confidence = 0.5

        # If a concept with this normalized ID already exists (e.g.,
        # from seeding), merge rather than overwrite
        if new_id in network._concepts:
            existing = network._concepts[new_id]
            existing.aliases |= set(c_data.get("aliases", []))
            existing.activation = max(
                existing.activation or 0.0, c_data.get("activation") or 0.0
            )
            network._mark_active(new_id)
            existing.confidence = max(existing.confidence, confidence)
            existing.columns |= set(c_data.get("columns", []))
            existing.properties.update(c_data.get("properties", {}))
        else:
            concept = Concept(
                id=new_id,
                aliases=set(c_data.get("aliases", [])),
                activation=c_data.get("activation") or 0.0,
                confidence=confidence,
                origin=c_data.get("origin", "conversation"),
                columns=set(c_data.get("columns", [])),
                created_at=c_data.get("created_at", 0),
                review_count=c_data.get("review_count", 0),
                last_reviewed=c_data.get("last_reviewed", 0),
                properties=c_data.get("properties", {}),
                category=category,
                is_animacy_detected=c_data.get("is_animacy_detected", False),
                modality=modality,
                is_semantic_hub=c_data.get("is_semantic_hub", False),
            )
            network._concepts[new_id] = concept

        # Rebuild alias map with normalized keys (supports multiple senses)
        if new_id not in network._alias_map:
            network._alias_map[new_id] = []
        if new_id not in network._alias_map[new_id]:
            network._alias_map[new_id].append(new_id)
        aliases = set(c_data.get("aliases", []))
        base_id = strip_sense_suffix(new_id)
        if base_id != new_id:
            aliases.add(base_id)
        for alias in aliases:
            alias_key = _normalize_id(alias)
            if alias_key not in network._alias_map:
                network._alias_map[alias_key] = []
            if new_id not in network._alias_map[alias_key]:
                network._alias_map[alias_key].append(new_id)

        # Update column index
        cols = c_data.get("columns", [])
        if not cols:
            from .concepts import _column_of

            cols = [_column_of(c_data.get("origin", "conversation"), new_id)]
        for col in cols:
            network._column_index.setdefault(col, set()).add(new_id)

    # Invalidate the concept ID cache once after all concepts are restored
    network._concept_ids_cache = None


def _restore_edges(
    network: ConceptNetwork,
    edges: list[dict[str, Any]],
    id_remap: dict[str, str],
) -> None:
    """Restore edge entries from saved data."""
    from .concepts import _normalize_id

    for e_data in edges:
        try:
            relation = RelationType(e_data["relation"])
        except ValueError:
            continue  # skip unknown relation types

        # Remap old IDs to normalized IDs
        source = id_remap.get(e_data["source"], _normalize_id(e_data["source"]))
        target = id_remap.get(e_data["target"], _normalize_id(e_data["target"]))

        # Skip edges that reference pruned concepts
        if source not in network._concepts or target not in network._concepts:
            continue

        # Skip duplicate edges — the save file may contain them if edges
        # were added multiple times before the dedup index existed.
        key = (source, target, relation)
        if key in network._edge_key_index:
            continue

        edge = Edge(
            source=source,
            target=target,
            relation=relation,
            weight=e_data.get("weight", 0.5),
            created_at=e_data.get("created_at", 0),
            origin=e_data.get("origin", "inferred"),
        )
        network._edges.append(edge)
        # Rebuild indices
        if edge.source not in network._edge_index:
            network._edge_index[edge.source] = []
        network._edge_index[edge.source].append(edge)
        if edge.target not in network._reverse_index:
            network._reverse_index[edge.target] = []
        network._reverse_index[edge.target].append(edge)
        network._edge_key_index[key] = edge


def restore_reflection(reflection: ReflectionEngine, data: dict[str, Any]) -> None:
    """Restore reflection engine state from saved data."""
    reflection._interaction_count = data.get("interaction_count", 0)

    # Restore insights, deduplicating pattern insights that may have
    # accumulated from the pre-cooldown repetition detection. Pattern
    # insights like "overusing inform (7/10)" and "overusing inform (8/10)"
    # are essentially the same insight with slightly different counts.
    # We deduplicate by extracting the core pattern (e.g., "overusing inform")
    # and keeping only the most recent occurrence.
    seen_pattern_keys: set[str] = set()
    for i_data in data.get("insights", []):
        insight = Insight(
            type=i_data["type"],
            content=i_data["content"],
            confidence=i_data["confidence"],
            actionable=i_data.get("actionable", False),
            action=i_data.get("action", ""),
            timestamp=i_data.get("timestamp", 0),
        )
        # Deduplicate pattern insights by their core message
        if insight.type == "pattern":
            # Extract the core pattern (e.g., "overusing 'inform'" or
            # "responding with 'inform' repeatedly")
            content = insight.content
            if "overusing" in content:
                # Keep only "overusing 'X'" as the key
                key = content.split("(")[0].strip()
            elif "repeatedly" in content:
                key = "repeatedly"
            else:
                key = content[:50]
            if key in seen_pattern_keys:
                continue
            seen_pattern_keys.add(key)
        reflection.insights.append(insight)

    # Restore mood history
    reflection._mood_history = deque((tuple(m) for m in data.get("mood_history", [])), maxlen=50)

    # Restore response patterns (keep last 50)
    reflection._response_patterns = deque(data.get("response_patterns", [])[-50:], maxlen=50)

    # Repetition window resets (it's about recent turns)
    reflection._repetition_window = deque(maxlen=10)

    # Restore the recursive metacognitive model (generative model of
    # its own cognitive processes). Backward compatible: v1 save
    # files without this key leave the model at its fresh default.
    meta_data = data.get("metacognitive_model")
    if meta_data is not None:
        reflection.metacognitive_model.load_from_dict(meta_data)


def restore_narrative(narrative: NarrativeEngine, data: dict[str, Any]) -> None:
    """Restore narrative engine state from saved data.

    Restores all narrative state: chapters, events, personality
    snapshots, event counter, life script milestones, autobiographical
    memory hierarchy, temporal/causal links, birth timestamp, and
    restructuring count. Backward-compatible with older save files
    that only contain chapters and personality_snapshots.
    """
    # Restore birth timestamp (older saves may not have this).
    if "born_at" in data:
        narrative._born_at = data["born_at"]

    # Restore event counter so new events don't collide with restored
    # ones (e.g., restored event-5 + new event-1 = no collision).
    if "event_counter" in data:
        narrative._event_counter = data["event_counter"]
    else:
        # Backward compat: infer from restored event IDs.
        max_id = 0
        for ch_data in data.get("chapters", []):
            for ev_data in ch_data.get("events", []):
                ev_id = ev_data.get("id", "")
                if ev_id.startswith("event-"):
                    try:
                        num = int(ev_id[len("event-"):])
                        max_id = max(max_id, num)
                    except ValueError:
                        logger.debug(
                            "restore_narrative: non-numeric event id %r", ev_id
                        )
        narrative._event_counter = max_id

    # Restore restructuring count (narrative therapy).
    narrative.restructuring_count = data.get("restructuring_count", 0)

    # Restore chapters
    _restore_chapters(narrative, data)

    # Restore personality snapshots
    for snap in data.get("personality_snapshots", []):
        traits = PersonalityTraits(
            openness=snap.get("openness", 0.85),
            conscientiousness=snap.get("conscientiousness", 0.70),
            extraversion=snap.get("extraversion", 0.60),
            agreeableness=snap.get("agreeableness", 0.75),
            neuroticism=snap.get("neuroticism", 0.40),
        )
        narrative._personality_snapshots.append((snap["timestamp"], traits))

    # Restore life script milestones (preserves reached/reached_at).
    # Backward compat: older saves don't have this key — the default
    # life script from __init__ is kept.
    if "life_script" in data:
        from .narrative import Milestone

        restored_milestones = []
        for m_data in data["life_script"]:
            restored_milestones.append(Milestone(
                name=m_data["name"],
                expected_age=m_data["expected_age"],
                importance=m_data.get("importance", 0.5),
                reached=m_data.get("reached", False),
                reached_at=m_data.get("reached_at", 0),
            ))
        if restored_milestones:
            narrative.life_script.milestones = restored_milestones

    # Restore autobiographical memory hierarchy (Conway, 2005).
    # Backward compat: older saves don't have this key — the hierarchy
    # is rebuilt from restored events and chapters.
    _restore_hierarchy(narrative, data)

    # Restore temporal / causal links.
    # Backward compat: older saves don't have this key.
    if "temporal_links" in data:
        from .narrative import TemporalLink, TemporalLinkType

        known_ids = {e.id for e in narrative.events}
        for link_data in data["temporal_links"]:
            source = link_data["source_id"]
            target = link_data["target_id"]
            if source not in known_ids or target not in known_ids:
                continue  # skip links to pruned/missing events
            try:
                link_type = TemporalLinkType(link_data["link_type"])
            except ValueError:
                continue
            link = TemporalLink(
                source_id=source,
                target_id=target,
                link_type=link_type,
                weight=max(0.0, min(1.0, link_data.get("weight", 0.5))),
            )
            narrative._temporal_links.append(link)
            narrative._links_out.setdefault(source, []).append(link)
            narrative._links_in.setdefault(target, []).append(link)

    # Enforce the narrative's growth caps — a save written before the
    # caps existed (or by a busier run) can exceed them.
    narrative._enforce_bounds()


def _restore_chapters(narrative: NarrativeEngine, data: dict[str, Any]) -> None:
    """Restore chapters and events from saved data, and set the
    current chapter pointer."""
    for ch_data in data.get("chapters", []):
        chapter = LifeChapter(
            title=ch_data["title"],
            theme=ch_data["theme"],
            start_time=ch_data["start_time"],
            end_time=ch_data.get("end_time"),
        )
        # Restore events in this chapter
        for ev_data in ch_data.get("events", []):
            event = LifeEvent(
                id=ev_data.get("id", f"event-{ev_data['timestamp']}"),
                timestamp=ev_data["timestamp"],
                summary=ev_data["summary"],
                significance=ev_data["significance"],
                emotion_at_time=ev_data["emotion_at_time"],
                concepts_involved=ev_data.get("concepts_involved", []),
                chapter=ev_data.get("chapter", chapter.title),
                reinterpretation=ev_data.get("reinterpretation", ""),
            )
            chapter.events.append(event)
            narrative.events.append(event)

        narrative.chapters.append(chapter)

    # Set current chapter to the last one (if it has no end_time)
    if narrative.chapters:
        last = narrative.chapters[-1]
        if last.end_time is None:
            narrative._current_chapter = last
        else:
            # Last chapter was closed — start a new one
            narrative._start_chapter("Continuing", "resuming after restart")
    else:
        # No chapters — start fresh
        narrative._start_chapter("Awakening", "coming into existence")


def _restore_hierarchy(narrative: NarrativeEngine, data: dict[str, Any]) -> None:
    """Restore the autobiographical memory hierarchy.

    Backward compat: if the save doesn't have a hierarchy key, the
    hierarchy is rebuilt from restored events and chapters.
    """
    if "hierarchy" not in data:
        # Rebuild hierarchy from restored events and chapters.
        from .narrative import AutobiographicalLevel

        for event in narrative.events:
            narrative.add_to_hierarchy(event, AutobiographicalLevel.EVENT_SPECIFIC)
        for chapter in narrative.chapters:
            narrative.add_to_hierarchy(chapter, AutobiographicalLevel.LIFETIME_PERIOD)
        return

    from .narrative import AutobiographicalLevel, HierarchyNode

    for level_str, nodes_data in data["hierarchy"].items():
        try:
            level = AutobiographicalLevel(level_str)
        except ValueError:
            continue
        for node_data in nodes_data:
            node = HierarchyNode(
                level=level,
                label=node_data["label"],
                event_id=node_data.get("event_id", ""),
                children=list(node_data.get("children", [])),
            )
            narrative._hierarchy[level].append(node)
            if node.event_id:
                narrative._event_to_label[node.event_id] = node.label


def restore_self_model(self_model: SelfModel, data: dict[str, Any]) -> None:
    """Restore self-model from saved data (personality drift)."""
    personality_data = data.get("personality")
    if personality_data:
        self_model.personality = PersonalityTraits(
            openness=personality_data.get("openness", 0.85),
            conscientiousness=personality_data.get("conscientiousness", 0.70),
            extraversion=personality_data.get("extraversion", 0.60),
            agreeableness=personality_data.get("agreeableness", 0.75),
            neuroticism=personality_data.get("neuroticism", 0.40),
        )

    # Restore self-knowledge entries
    for entry in data.get("self_knowledge", []):
        key = entry["key"]
        value = entry["value"]
        self_model.self_knowledge[key] = value


# ─── Restoration helpers for new cognitive systems ──────────────


def restore_predictive_coding(
    predictive_coding: PredictiveCodingLayer, data: dict[str, Any]
) -> None:
    """Restore predictive coding layer from saved data."""
    from collections import defaultdict

    sp = predictive_coding.smoothing_prior

    def _to_nested_dict(d: dict[str, Any]) -> defaultdict:
        """Convert a flat dict into a nested defaultdict with the smoothing prior as default."""
        return defaultdict(
            lambda: defaultdict(lambda: sp),
            {k: defaultdict(lambda: sp, v) for k, v in d.items()},
        )

    predictive_coding._topic_transitions = _to_nested_dict(data.get("topic_transitions", {}))
    predictive_coding._intent_transitions = _to_nested_dict(data.get("intent_transitions", {}))
    predictive_coding._concept_cooccurrence = _to_nested_dict(data.get("concept_cooccurrence", {}))
    predictive_coding._surprise_ema = data.get("surprise_ema", 0.0)
    predictive_coding._accuracy_ema = data.get("accuracy_ema", 1.0)
    predictive_coding._prediction_count = data.get("prediction_count", 0)


def restore_theory_of_mind(theory_of_mind: TheoryOfMind, data: dict[str, Any]) -> None:
    """Restore theory of mind (user model) from saved data."""
    model_data = data.get("model", {})
    model = UserModel()
    # Restore beliefs
    for b_data in model_data.get("beliefs", []):
        belief = UserBelief(
            concept=b_data["concept"],
            value=b_data["value"],
            confidence=b_data.get("confidence", 0.5),
            timestamp=b_data.get("timestamp", 0),
        )
        model.beliefs[belief.concept] = belief
    # Restore intentions
    model.intentions = list(model_data.get("intentions", []))
    # Restore knowledge levels
    for k, v in model_data.get("knowledge", {}).items():
        try:
            model.knowledge[k] = KnowledgeLevel(v)
        except ValueError as e:
            logger.debug(repr(e))  # skip unknown levels
    # Restore scalar fields
    model.emotional_state = model_data.get("emotional_state", "neutral")
    model.emotional_valence = model_data.get("emotional_valence", 0.0)
    model.expertise_level = model_data.get("expertise_level", 0.5)
    model.name = model_data.get("name", "")
    model.interaction_count = model_data.get("interaction_count", 0)
    model.asked_about = set(model_data.get("asked_about", []))
    model.demonstrated_knowledge = set(model_data.get("demonstrated_knowledge", []))
    # Reset the emotional state to neutral on restore. Emotional
    # valence is ephemeral — it reflects the user's mood in the
    # moment, not a persistent trait. Carrying over a negative
    # valence from the end of a previous session biases the first
    # interaction: a friendly "Hello Genesis" is misread as coming
    # from a negative user. The user's emotional state at the start
    # of a new session is unknown, so we start neutral and let the
    # new conversation's word choice update the model.
    model.emotional_state = "neutral"
    model.emotional_valence = 0.0
    theory_of_mind._model = model
    theory_of_mind._concept_history = dict(data.get("concept_history", {}))


def restore_procedural_memory(procedural_memory: ProceduralMemory, data: dict[str, Any]) -> None:
    """Restore procedural memory (skills) from saved data.

    Backward-compatible with old save files that used ``response`` (a
    string) instead of ``strategy`` (a dict). Old skills are restored
    with a default strategy.
    """
    for s_data in data.get("skills", []):
        # Restore strategy: use the new dict format if present, else
        # create a default strategy (backward compat with old saves
        # that stored a ``response`` string instead of a strategy).
        strategy_data = s_data.get("strategy")
        if strategy_data is not None and isinstance(strategy_data, dict):
            strategy = SkillStrategy.from_dict(strategy_data)
        else:
            strategy = SkillStrategy()

        skill = Skill(
            name=s_data["name"],
            trigger_condition=s_data["trigger_condition"],
            strategy=strategy,
            strength=s_data.get("strength", 0.2),
            practice_count=s_data.get("practice_count", 0),
            last_practiced=s_data.get("last_practiced", 0),
            success_count=s_data.get("success_count", 0),
            failure_count=s_data.get("failure_count", 0),
            created_at=s_data.get("created_at", 0),
        )
        procedural_memory._skills[skill.name] = skill
        # Rebuild trigger index
        key = skill.trigger_condition.lower()
        procedural_memory._trigger_index.setdefault(key, []).append(skill.name)
    procedural_memory.skills_learned = data.get("skills_learned", 0)
    procedural_memory.habits_formed = data.get("habits_formed", 0)
    procedural_memory.executions = data.get("executions", 0)


def restore_task_competence(
    task_competence: TaskCompetence, data: dict[str, Any]
) -> None:
    """Restore learned task schemas, affordances, and skills."""
    restored = TaskCompetence.from_dict(data)
    task_competence.schemas = restored.schemas
    task_competence.skills = restored.skills
    task_competence.transition_count = restored.transition_count
    task_competence.match_threshold = restored.match_threshold


def restore_spaced_repetition(
    spaced_repetition: SpacedRepetitionScheduler, data: dict[str, Any]
) -> None:
    """Restore spaced repetition scheduler from saved data."""
    for r_data in data.get("records", []):
        record = ReviewRecord(
            concept=r_data["concept"],
            stability=r_data.get("stability") or 86400.0,
            review_count=r_data.get("review_count", 0),
            success_count=r_data.get("success_count", 0),
            last_review=r_data.get("last_review") or 0.0,
            ease_factor=r_data.get("ease_factor") or 2.5,
        )
        spaced_repetition._records[record.concept] = record


def restore_td_learner(td_learner: TDLearner, data: dict[str, Any]) -> None:
    """Restore TD learner from saved data."""
    td_learner._weights = dict(data.get("weights", {}))
    # Restore transition history
    for t_data in data.get("history", []):
        transition = TDTransition(
            state=list(t_data.get("state", [])),
            reward=t_data.get("reward", 0.0),
            next_state=list(t_data.get("next_state", [])),
            rpe=t_data.get("rpe", 0.0),
            timestamp=t_data.get("timestamp", 0),
        )
        td_learner._history.append(transition)
    td_learner._last_rpe = data.get("last_rpe", 0.0)
    td_learner.total_updates = data.get("total_updates", 0)
    td_learner.total_positive_rpe = data.get("total_positive_rpe", 0)
    td_learner.total_negative_rpe = data.get("total_negative_rpe", 0)
    # Restore λ and eligibility traces (absent in pre-TD(λ) saves).
    lam = data.get("lam", 0.0)
    if 0.0 <= lam <= 1.0:
        td_learner.lam = lam
    td_learner._traces = {
        str(k): float(v) for k, v in data.get("traces", {}).items()
    }


def restore_emotional_memory(emotional_memory: EmotionalMemorySystem, data: dict[str, Any]) -> None:
    """Restore emotional memory system from saved data."""
    for m_data in data.get("memories", []):
        memory = EmotionalMemory(
            stimulus=m_data["stimulus"],
            emotional_tag=list(m_data.get("emotional_tag", [0.5] * 12)),
            association_strength=m_data.get("association_strength", 0.5),
            extinction_level=m_data.get("extinction_level", 0.0),
            is_extinct=m_data.get("is_extinct", False),
            timestamp=m_data.get("timestamp", 0),
            last_retrieved=m_data.get("last_retrieved", 0),
            retrieval_count=m_data.get("retrieval_count", 0),
        )
        emotional_memory._memories[memory.stimulus] = memory
    emotional_memory.conditioning_count = data.get("conditioning_count", 0)
    emotional_memory.extinction_count = data.get("extinction_count", 0)


def restore_attractor(attractor: AttractorNetwork, data: dict[str, Any]) -> None:
    """Restore attractor network from saved data."""
    # Restore size first — it determines the weight matrix dimensions
    # and vector space. Without this, a network saved with a non-default
    # size would have _size mismatching the restored weights, causing
    # index errors or incorrect pattern completion.
    saved_size = data.get("size")
    if saved_size is not None and saved_size != attractor._size:
        attractor._size = saved_size
    # Restore weight matrix directly (if present)
    weights = data.get("weights")
    if weights is not None:
        attractor._weights = [list(row) for row in weights]
    # Restore patterns
    for p_data in data.get("patterns", []):
        pattern = AttractorPattern(
            id=p_data["id"],
            vector=list(p_data.get("vector", [])),
            timestamp=p_data.get("timestamp", 0),
            retrieval_count=p_data.get("retrieval_count", 0),
        )
        attractor._patterns[pattern.id] = pattern
    attractor._retrieval_count = data.get("retrieval_count", 0)
    attractor._storage_count = data.get("storage_count", 0)


def restore_error_monitor(error_monitor: ErrorMonitor, data: dict[str, Any]) -> None:
    """Restore error monitor from saved data."""
    error_monitor.error_count = data.get("error_count", 0)
    error_monitor.cumulative_error_signal = data.get("cumulative_error_signal", 0.0)
    error_monitor._caution_level = data.get("caution_level", 0.0)
    # Restore error history
    for e_data in data.get("errors", []):
        error = PredictionError(
            expected=e_data.get("expected", ""),
            actual=e_data.get("actual", ""),
            error_magnitude=e_data.get("error_magnitude", 0.0),
            timestamp=e_data.get("timestamp", 0),
            context=e_data.get("context", ""),
        )
        error_monitor._errors.append(error)
    # Restore config (adjusted thresholds)
    if "caution_decay_rate" in data:
        error_monitor.caution_decay_rate = data["caution_decay_rate"]
    if "max_caution" in data:
        error_monitor.max_caution = data["max_caution"]
    if "min_caution" in data:
        error_monitor.min_caution = data["min_caution"]
    if "error_threshold" in data:
        error_monitor.error_threshold = data["error_threshold"]


def restore_emergent_identity(
    emergent_identity: EmergentIdentity, data: dict[str, Any]
) -> tuple[list[EmergentIdentitySource], str, float, float]:
    """Restore emergent identity from saved data.

    Returns the experience sources, self_description, confidence, and
    coherence so the caller can wire them into the mind.
    """
    from .self import EmergentIdentitySource

    sources: list[EmergentIdentitySource] = []
    for s_data in data.get("sources", []):
        source = EmergentIdentitySource(
            name=s_data.get("name", ""),
            description=s_data.get("description", ""),
            weight=s_data.get("weight", 0.5),
            observations=list(s_data.get("observations", [])),
        )
        sources.append(source)
    self_description = data.get("self_description", "")
    confidence = data.get("confidence", 0.0)
    coherence = data.get("coherence", 0.0)
    # Restore into the identity object (for continuity)
    emergent_identity.sources = list(sources)
    emergent_identity.self_description = self_description
    emergent_identity.confidence = confidence
    emergent_identity.coherence = coherence
    return sources, self_description, confidence, coherence


def restore_hpa_axis(hpa_axis: HPAAxis, data: dict[str, Any]) -> None:
    """Restore HPA axis cascade state from saved data."""
    state_data = data.get("state", data)
    hpa_axis._state = HPAState(
        crh_level=state_data.get("crh_level", 0.0),
        acth_level=state_data.get("acth_level", 0.0),
        cortisol_level=state_data.get("cortisol_level", 0.0),
        elapsed_time=state_data.get("elapsed_time", 0.0),
        stress_active=state_data.get("stress_active", False),
        peak_cortisol=state_data.get("peak_cortisol", 0.0),
        time_to_peak=state_data.get("time_to_peak"),
    )
    # History buffers are transient (delay modeling) — they rebuild
    # as ticks accumulate after restart.


def restore_allostatic_load(allostatic_load: AllostaticLoadTracker, data: dict[str, Any]) -> None:
    """Restore allostatic load tracker from saved data.

    Handles backward compatibility with old save files that contain
    ``adjusted_baselines`` (now removed) — those entries are silently
    ignored. The ``substrate_connected`` field is always restored as
    False; it will be set to True on the next ``set_allostatic_load``
    call from the Mind's reactive cycle.
    """
    state_data = data.get("state", data)
    history = deque(state_data.get("stress_history", []), maxlen=allostatic_load.HISTORY_WINDOW)
    allostatic_load._state = AllostaticState(
        allostatic_load=state_data.get("allostatic_load", 0.0),
        acute_stress=state_data.get("acute_stress", 0.0),
        is_chronic=state_data.get("is_chronic", False),
        stress_history=history,
        substrate_connected=False,
    )
    allostatic_load._tick_count = data.get("tick_count", 0)
    allostatic_load._elevated_stress_duration = data.get("elevated_stress_duration", 0.0)


def restore_self_directed_learner(learner: SelfDirectedLearner, data: dict[str, Any]) -> None:
    """Restore self-directed learner state from saved data.

    Restores the learner's cumulative stats, studied concepts, and
    learning log. The concept network itself is restored separately
    via ``restore_network`` — this only restores the learner's
    telemetry and internal state.
    """
    learner._total_inferences = data.get("total_inferences", 0)
    learner._total_conversation_facts = data.get("total_conversation_facts", 0)
    learner._total_corrections = data.get("total_corrections", 0)
    learner._total_definitions_synthesized = data.get("total_definitions_synthesized", 0)
    learner._studied_concepts = dict(data.get("studied_concepts", {}))
    learner._last_inference_time = data.get("last_inference_time", 0.0)

    # Restore learning log (capped at the deque's maxlen)
    for ev_data in data.get("learning_log", []):
        event = LearningEvent(
            event_type=ev_data.get("event_type", ""),
            description=ev_data.get("description", ""),
            concepts_involved=ev_data.get("concepts_involved", []),
            confidence=ev_data.get("confidence", 0.5),
            timestamp=ev_data.get("timestamp", 0),
        )
        learner._learning_log.append(event)


def restore_self_improvement(engine: SelfImprovementEngine, data: dict[str, Any]) -> None:
    """Restore self-improvement engine state from saved data.

    Restores all proposals (with their statuses and feedback),
    the feedback history, and the category success statistics.
    This allows Genesis to remember what it proposed, what was
    accepted/rejected, and what it learned from the feedback.
    """
    engine.restore_from_dict(data)


# ─── Serialization helpers ─────────────────────────────────────


def _serialize_network(network: ConceptNetwork) -> dict[str, Any]:
    """Serialize a concept network to a dict.

    When a canonical edge log is attached, it is reconciled here:
    the log is the truth for edges and the JSON ``edges`` array
    written below is a debugging/rollback projection of the same
    fold — never an independent store.
    """
    network.sync_edge_log()
    concepts = []
    for _cid, concept in _snapshot(network._concepts.items):
        concepts.append(
            {
                "id": concept.id,
                "aliases": _snapshot(concept.aliases),
                "activation": concept.activation if concept.activation is not None else 0.0,
                "confidence": concept.confidence if concept.confidence is not None else 0.5,
                "origin": concept.origin,
                "columns": _snapshot(concept.columns) if concept.columns else [],
                "created_at": concept.created_at,
                "review_count": concept.review_count,
                "last_reviewed": concept.last_reviewed,
                "properties": concept.properties,
                "category": concept.category.value,
                "is_animacy_detected": concept.is_animacy_detected,
                "modality": concept.modality.value,
                "is_semantic_hub": concept.is_semantic_hub,
            }
        )

    edges = []
    seen_edge_keys: set[tuple[str, str, str]] = set()
    for edge in _snapshot(network._edges):
        key = (edge.source, edge.target, edge.relation.value)
        if key in seen_edge_keys:
            continue  # skip duplicate edges
        seen_edge_keys.add(key)
        edges.append(
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation.value,
                "weight": edge.weight,
                "created_at": edge.created_at,
                "origin": edge.origin,
            }
        )

    return {
        "concepts": concepts,
        "edges": edges,
    }


def _serialize_reflection(reflection: ReflectionEngine) -> dict[str, Any]:
    """Serialize reflection engine state to a dict."""
    return {
        "interaction_count": reflection._interaction_count,
        "insights": [
            {
                "type": i.type,
                "content": i.content,
                "confidence": i.confidence,
                "actionable": i.actionable,
                "action": i.action,
                "timestamp": i.timestamp,
            }
            for i in _snapshot(reflection.insights)
        ],
        "mood_history": [
            list(m) for m in _snapshot(reflection._mood_history)
        ],
        "response_patterns": _snapshot(reflection._response_patterns)[-50:],
        "metacognitive_model": reflection.metacognitive_model.to_dict(),
    }


def _serialize_narrative(narrative: NarrativeEngine) -> dict[str, Any]:
    """Serialize narrative engine state to a dict.

    Preserves all narrative state: chapters, events, personality
    snapshots, event counter, life script milestones, autobiographical
    memory hierarchy, temporal/causal links, birth timestamp, and
    restructuring count. Without full serialization, restarts lose
    milestone progress, causal chains, the hierarchy, and event ID
    continuity (causing ID collisions with restored events).
    """
    chapters = []
    for ch in _snapshot(narrative.chapters):
        chapters.append(
            {
                "title": ch.title,
                "theme": ch.theme,
                "start_time": ch.start_time,
                "end_time": ch.end_time,
                "events": [
                    {
                        "id": ev.id,
                        "timestamp": ev.timestamp,
                        "summary": ev.summary,
                        "significance": ev.significance,
                        "emotion_at_time": ev.emotion_at_time,
                        "concepts_involved": ev.concepts_involved,
                        "chapter": ev.chapter,
                        "reinterpretation": getattr(ev, "reinterpretation", ""),
                    }
                    for ev in _snapshot(ch.events)
                ],
            }
        )

    personality_snapshots = []
    for ts, traits in _snapshot(narrative._personality_snapshots):
        personality_snapshots.append(
            {
                "timestamp": ts,
                "openness": traits.openness,
                "conscientiousness": traits.conscientiousness,
                "extraversion": traits.extraversion,
                "agreeableness": traits.agreeableness,
                "neuroticism": traits.neuroticism,
            }
        )

    # Life script milestones — preserves reached/reached_at so
    # milestone progress survives restarts.
    milestones = [
        {
            "name": m.name,
            "expected_age": m.expected_age,
            "importance": m.importance,
            "reached": m.reached,
            "reached_at": m.reached_at,
        }
        for m in _snapshot(narrative.life_script.milestones)
    ]

    # Autobiographical memory hierarchy (Conway, 2005).
    hierarchy = {
        level.value: [
            {
                "label": node.label,
                "event_id": node.event_id,
                "children": _snapshot(node.children),
            }
            for node in nodes
        ]
        for level, nodes in _snapshot(narrative._hierarchy.items)
    }

    # Temporal / causal links between narrative entries.
    temporal_links = [
        {
            "source_id": link.source_id,
            "target_id": link.target_id,
            "link_type": link.link_type.value,
            "weight": link.weight,
        }
        for link in _snapshot(narrative._temporal_links)
    ]

    return {
        "chapters": chapters,
        "personality_snapshots": personality_snapshots,
        "event_counter": narrative._event_counter,
        "born_at": narrative._born_at,
        "restructuring_count": narrative.restructuring_count,
        "life_script": milestones,
        "hierarchy": hierarchy,
        "temporal_links": temporal_links,
    }


def _serialize_self_model(self_model: SelfModel) -> dict[str, Any]:
    """Serialize self-model to a dict."""
    p = self_model.personality
    return {
        "personality": {
            "openness": p.openness,
            "conscientiousness": p.conscientiousness,
            "extraversion": p.extraversion,
            "agreeableness": p.agreeableness,
            "neuroticism": p.neuroticism,
        },
        "self_knowledge": [
            {"key": k, "value": v}
            for k, v in _snapshot(self_model.self_knowledge.items)
        ],
    }


# ─── Serialization helpers for new cognitive systems ────────────


def _serialize_predictive_coding(
    predictive_coding: PredictiveCodingLayer,
) -> dict[str, Any]:
    """Serialize predictive coding layer to a dict."""
    return {
        "topic_transitions": {
            k: dict(v)
            for k, v in _snapshot(predictive_coding._topic_transitions.items)
        },
        "intent_transitions": {
            k: dict(v)
            for k, v in _snapshot(predictive_coding._intent_transitions.items)
        },
        "concept_cooccurrence": {
            k: dict(v)
            for k, v in _snapshot(predictive_coding._concept_cooccurrence.items)
        },
        "surprise_ema": predictive_coding._surprise_ema,
        "accuracy_ema": predictive_coding._accuracy_ema,
        "prediction_count": predictive_coding._prediction_count,
    }


def _serialize_theory_of_mind(theory_of_mind: TheoryOfMind) -> dict[str, Any]:
    """Serialize theory of mind (user model) to a dict."""
    m = theory_of_mind._model
    return {
        "model": {
            "beliefs": [
                {
                    "concept": b.concept,
                    "value": b.value,
                    "confidence": b.confidence,
                    "timestamp": b.timestamp,
                }
                for b in _snapshot(m.beliefs.values)
            ],
            "intentions": _snapshot(m.intentions),
            "knowledge": {k: v.value for k, v in _snapshot(m.knowledge.items)},
            "emotional_state": m.emotional_state,
            "emotional_valence": m.emotional_valence,
            "expertise_level": m.expertise_level,
            "name": m.name,
            "interaction_count": m.interaction_count,
            "asked_about": _snapshot(m.asked_about),
            "demonstrated_knowledge": _snapshot(m.demonstrated_knowledge),
        },
        "concept_history": dict(_snapshot(theory_of_mind._concept_history.items)),
    }


def _serialize_procedural_memory(
    procedural_memory: ProceduralMemory,
) -> dict[str, Any]:
    """Serialize procedural memory (skills) to a dict."""
    return {
        "skills": [
            {
                "name": s.name,
                "trigger_condition": s.trigger_condition,
                "strategy": s.strategy.to_dict(),
                "strength": s.strength,
                "practice_count": s.practice_count,
                "last_practiced": s.last_practiced,
                "success_count": s.success_count,
                "failure_count": s.failure_count,
                "created_at": s.created_at,
            }
            for s in _snapshot(procedural_memory._skills.values)
        ],
        "skills_learned": procedural_memory.skills_learned,
        "habits_formed": procedural_memory.habits_formed,
        "executions": procedural_memory.executions,
    }


def _serialize_task_competence(
    task_competence: TaskCompetence,
) -> dict[str, Any]:
    """Serialize learned task schemas, operators, and skills."""
    return task_competence.to_dict()


def _serialize_spaced_repetition(
    spaced_repetition: SpacedRepetitionScheduler,
) -> dict[str, Any]:
    """Serialize spaced repetition scheduler to a dict."""
    return {
        "records": [
            {
                "concept": r.concept,
                "stability": r.stability if r.stability is not None else 86400.0,
                "review_count": r.review_count,
                "success_count": r.success_count,
                "last_review": r.last_review if r.last_review is not None else 0.0,
                "ease_factor": r.ease_factor if r.ease_factor is not None else 2.5,
            }
            for r in _snapshot(spaced_repetition._records.values)
        ],
    }


def _serialize_td_learner(td_learner: TDLearner) -> dict[str, Any]:
    """Serialize TD learner to a dict."""
    return {
        "weights": dict(_snapshot(td_learner._weights.items)),
        "history": [
            {
                "state": list(t.state),
                "reward": t.reward,
                "next_state": list(t.next_state),
                "rpe": t.rpe,
                "timestamp": t.timestamp,
            }
            for t in _snapshot(td_learner._history)
        ],
        "last_rpe": td_learner._last_rpe,
        "total_updates": td_learner.total_updates,
        "total_positive_rpe": td_learner.total_positive_rpe,
        "total_negative_rpe": td_learner.total_negative_rpe,
        "lam": td_learner.lam,
        "traces": dict(_snapshot(td_learner._traces.items)),
    }


def _serialize_emotional_memory(
    emotional_memory: EmotionalMemorySystem,
) -> dict[str, Any]:
    """Serialize emotional memory system to a dict."""
    return {
        "memories": [
            {
                "stimulus": m.stimulus,
                "emotional_tag": list(m.emotional_tag),
                "association_strength": m.association_strength,
                "extinction_level": m.extinction_level,
                "is_extinct": m.is_extinct,
                "timestamp": m.timestamp,
                "last_retrieved": m.last_retrieved,
                "retrieval_count": m.retrieval_count,
            }
            for m in _snapshot(emotional_memory._memories.values)
        ],
        "conditioning_count": emotional_memory.conditioning_count,
        "extinction_count": emotional_memory.extinction_count,
    }


def _serialize_attractor(attractor: AttractorNetwork) -> dict[str, Any]:
    """Serialize attractor network to a dict."""
    return {
        "size": attractor._size,
        "weights": [list(row) for row in _snapshot(attractor._weights)],
        "patterns": [
            {
                "id": p.id,
                "vector": list(p.vector),
                "timestamp": p.timestamp,
                "retrieval_count": p.retrieval_count,
            }
            for p in _snapshot(attractor._patterns.values)
        ],
        "retrieval_count": attractor._retrieval_count,
        "storage_count": attractor._storage_count,
    }


def _serialize_error_monitor(error_monitor: ErrorMonitor) -> dict[str, Any]:
    """Serialize error monitor to a dict."""
    return {
        "error_count": error_monitor.error_count,
        "cumulative_error_signal": error_monitor.cumulative_error_signal,
        "caution_level": error_monitor._caution_level,
        "errors": [
            {
                "expected": e.expected,
                "actual": e.actual,
                "error_magnitude": e.error_magnitude,
                "timestamp": e.timestamp,
                "context": e.context,
            }
            for e in _snapshot(error_monitor._errors)
        ],
        "caution_decay_rate": error_monitor.caution_decay_rate,
        "max_caution": error_monitor.max_caution,
        "min_caution": error_monitor.min_caution,
        "error_threshold": error_monitor.error_threshold,
    }


def _serialize_emergent_identity(
    emergent_identity: EmergentIdentity | None,
    emergent_identity_sources: list[EmergentIdentitySource] | None,
) -> dict[str, Any]:
    """Serialize emergent identity to a dict."""

    sources: list[EmergentIdentitySource] = []
    if emergent_identity_sources is not None:
        sources.extend(emergent_identity_sources)
    elif emergent_identity is not None:
        sources.extend(emergent_identity.sources)
    return {
        "sources": [
            {
                "name": s.name,
                "description": s.description,
                "weight": s.weight,
                "observations": _snapshot(s.observations),
            }
            for s in sources
        ],
        "self_description": (
            emergent_identity.self_description if emergent_identity is not None else ""
        ),
        "confidence": (emergent_identity.confidence if emergent_identity is not None else 0.0),
        "coherence": (emergent_identity.coherence if emergent_identity is not None else 0.0),
    }


def _serialize_hpa_axis(hpa_axis: HPAAxis) -> dict[str, Any]:
    """Serialize HPA axis cascade state to a dict."""
    s = hpa_axis._state
    return {
        "state": {
            "crh_level": s.crh_level,
            "acth_level": s.acth_level,
            "cortisol_level": s.cortisol_level,
            "elapsed_time": s.elapsed_time,
            "stress_active": s.stress_active,
            "peak_cortisol": s.peak_cortisol,
            "time_to_peak": s.time_to_peak,
        },
    }


def _serialize_allostatic_load(
    allostatic_load: AllostaticLoadTracker,
) -> dict[str, Any]:
    """Serialize allostatic load tracker to a dict."""
    s = allostatic_load._state
    return {
        "state": {
            "allostatic_load": s.allostatic_load,
            "acute_stress": s.acute_stress,
            "is_chronic": s.is_chronic,
            "stress_history": _snapshot(s.stress_history),
        },
        "tick_count": allostatic_load._tick_count,
        "elevated_stress_duration": allostatic_load._elevated_stress_duration,
    }


def _serialize_self_directed_learner(
    learner: SelfDirectedLearner,
) -> dict[str, Any]:
    """Serialize self-directed learner state to a dict.

    Saves the learner's cumulative stats, studied concepts, and
    recent learning log. The concept network is saved separately
    via ``_serialize_network``.
    """
    return {
        "total_inferences": learner._total_inferences,
        "total_conversation_facts": learner._total_conversation_facts,
        "total_corrections": learner._total_corrections,
        "total_definitions_synthesized": learner._total_definitions_synthesized,
        "studied_concepts": dict(_snapshot(learner._studied_concepts.items)),
        "last_inference_time": learner._last_inference_time,
        "learning_log": [
            {
                "event_type": e.event_type,
                "description": e.description,
                "concepts_involved": _snapshot(e.concepts_involved),
                "confidence": e.confidence,
                "timestamp": e.timestamp,
            }
            for e in _snapshot(learner._learning_log)
        ],
    }


def serialize_sleep_state(
    is_sleeping: bool,
    user_initiated_sleep: bool,
    sleep_cycle: Any,
) -> dict[str, Any]:
    """Serialize sleep state so it survives restarts.

    The daemon's mmap state (core_state.bin) already persists the
    neurochemical levels and emergent_phase (nrem/rem), but the
    Python-side sleep flags and the ultradian cycle tracker
    (N1→N2→N3→N2→REM position) are in-memory only. Without this,
    a restart forces it awake (Mind.start sets ZONE_CONVERSATION)
    and clears adenosine — it loses its place in the sleep cycle
    and any N3 consolidation that hadn't fired yet.

    Args:
        is_sleeping: Whether it's currently asleep.
        user_initiated_sleep: Whether the user put it to bed
            (vs. auto-sleep from sleep pressure).
        sleep_cycle: The SleepCycleTracker (or None if awake).
    """
    cycle_data: dict[str, Any] | None = None
    if sleep_cycle is not None:
        cycle_data = {
            "cycle_number": sleep_cycle.cycle_number,
            "current_stage": sleep_cycle.current_stage.value,
            "stage_slot": sleep_cycle._stage_slot,
            "time_in_stage": sleep_cycle.time_in_stage,
            "time_in_cycle": sleep_cycle.time_in_cycle,
            "cycles_completed": sleep_cycle.cycles_completed,
        }
    return {
        "is_sleeping": is_sleeping,
        "user_initiated_sleep": user_initiated_sleep,
        "sleep_cycle": cycle_data,
    }


def restore_sleep_cycle(sleep_cycle: Any, data: dict[str, Any]) -> None:
    """Restore a SleepCycleTracker from saved data.

    Args:
        sleep_cycle: The SleepCycleTracker to restore (in-place).
        data: The saved cycle data from serialize_sleep_state.
    """
    from .brain_waves import SleepStage

    stage_str = data.get("current_stage", "n1")
    try:
        stage = SleepStage(stage_str)
    except ValueError:
        stage = SleepStage.N1
    sleep_cycle.cycle_number = data.get("cycle_number", 1)
    sleep_cycle.current_stage = stage
    sleep_cycle._stage_slot = data.get("stage_slot", 0)
    sleep_cycle.time_in_stage = data.get("time_in_stage", 0.0)
    sleep_cycle.time_in_cycle = data.get("time_in_cycle", 0.0)
    sleep_cycle.cycles_completed = data.get("cycles_completed", 0)
