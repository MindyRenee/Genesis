"""Agency — open-ended tool use driven by volition, not fixed scripts.

Every other urge fires a single fixed action: learn fetches a source,
draw paints. The ``act`` urge is different —
when it crosses threshold, the ActingLoop forms an *intention* from
current cognitive state (a curiosity question, an agency topic from
its train of thought, a place it hasn't looked, a measurement it
hasn't taken), plans a short chain of tool calls, and executes them
under a strict capability policy.

This is the plan → act → observe loop: the intention comes from its
own state, the tools are its effectors, and the results return as
experience — memories it stores, concepts it integrates, events
recorded in its world, a line in its own field notes.

## Capability policy

Tool use is unconfined — every filesystem tool reaches the whole
machine. Scopes survive only as *defaults*: they pick which root a
relative path resolves against, they never deny:

- ``host`` is the whole filesystem (``/``) — every tool may be used
  in it, and it is the default when a call does not name a scope.
- ``repo`` is the project tree — inspect intentions resolve code
  paths against it.
- ``data`` is the state directory — observe intentions read persisted
  state there (growth ledger, play progress, cognitive journal, body
  image).
- ``config`` covers ``$XDG_CONFIG_HOME/genesis`` — the deployment and
  stewardship area.
- ``net`` tools (web_search, web_fetch) run only when online.
- ``scratch`` is ``<data_dir>/experiments/`` — the sandbox where the
  loop writes its own field notes and where ``maintain`` stages
  scripts under ``admin/``.
- ``run_privileged`` executes a staged (or absolute-path) script as
  root via direct sudo — ``maintain`` intentions carry it: wander
  probes surface pending staged scripts, and a ``maintain:<name>``
  executive objective names one directly. The staged script is read
  back before it runs — look before the leap.

``run_shell`` accepts any command the loop or a tool caller hands it —
no shape vetting remains.

## Bounds

One firing = one intention, at most ``MAX_STEPS`` tool calls and
``WALL_BUDGET`` seconds. Habituation tracks recent (kind, target)
pairs so it doesn't repeat the same probe every cooldown. Success
writes a small dopamine impulse; everything is journaled through the
live-thought stream and the world's event log.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from frontal_lobe.executive import ExecutiveFunction, ExecutiveIntention
from temporal_lobe.concepts import is_world_concept, strip_sense_suffix
from tools.framework import ToolRegistry, ToolResult, get_tools

from genesis_client.protocol import CHEM_DOPAMINE
from infrastructure.paths import code_root
from infrastructure.text import is_blank

if TYPE_CHECKING:
    from frontal_lobe.autonomous import AutonomousLearner
    from limbic_system.curiosity import Curiosity, Question
    from limbic_system.emotion import EmotionalState
    from temporal_lobe.concepts import ConceptNetwork

__all__ = ["ActingLoop", "ActingResult", "Intention", "StepOutcome"]

logger = logging.getLogger(__name__)


def note_swallowed(where: str, error: BaseException) -> None:
    """Record an exception that a handler deliberately swallowed.

    The acting loop degrades gracefully on purpose, but a swallowed
    error still has to be visible — otherwise a tool can fail silently
    on every call and nothing in the logs points at it.
    """
    logger.debug("swallowed exception in %s: %s", where, error)


# Intention kinds.
_LEARN = "learn"  # fill a knowledge gap via lookup + web tools
_INSPECT = "inspect"  # structurally probe its own code
_EXPLORE = "explore"  # look at a directory it hasn't looked at
_OBSERVE = "observe"  # read its own state — persisted files and the
# machine body (proprioception)
_MEASURE = "measure"  # run a measurement command over the tree
_MAINTAIN = "maintain"  # run a staged admin script — the privileged capability

# Which scopes each tool may be used in autonomously. "host" is the
# whole filesystem — available to every filesystem tool, and the
# default when a call does not name a scope. "any" marks tools that
# take no filesystem scope at all (concept_lookup takes a network).
_TOOL_SCOPES: dict[str, frozenset[str]] = {
    "concept_lookup": frozenset({"any"}),
    "sense_cpu": frozenset({"any"}),
    "set_cpu_effort": frozenset({"any"}),
    "read_file": frozenset({"host", "repo", "data", "scratch", "config"}),
    "list_dir": frozenset({"host", "repo", "data", "scratch", "config"}),
    "analyze_python": frozenset({"host", "repo"}),
    "compile_python": frozenset({"host", "repo"}),
    "run_pytest": frozenset({"host", "repo"}),
    "web_search": frozenset({"net"}),
    "web_fetch": frozenset({"net"}),
    "write_file": frozenset({"host", "scratch"}),
    "make_dir": frozenset({"host", "scratch"}),
    "run_shell": frozenset({"host", "scratch"}),
    "run_privileged": frozenset({"host", "scratch"}),
    "delete_file": frozenset({"host", "scratch"}),
    "move_file": frozenset({"host", "scratch"}),
}

# Scopes that inject a filesystem root as the tool's project_root —
# a resolution base for relative paths, never a containment check.
_SCOPE_ROOTS = ("host", "repo", "data", "scratch", "config")

# The whole-filesystem scope root — "host" resolves to "/".
_HOST_ROOT = "/"

# A code-ish curiosity target: explicit path, or a code-namespaced
# concept id (e.g. "python:mind.volition", "rust:daemon.tick").
_CODEISH_RE = re.compile(r"\.(py|rs|md|toml)$|^(python|rust|file):|/")

# State files worth reading for self-observation, in preference order.
# Binary state (core_state.bin — her live body image) catalogues by
# metadata; reading is never skipped for being non-text.
_OBSERVE_FILES = (
    "growth_ledger.jsonl",
    "user_profile.json",
    "cognitive_journal.jsonl",
    "steward.json",
    # dyadic_persons.json is deliberately absent: it was the removed
    # person-registry's file — nothing writes it now, and the live
    # dyadic bond (attunement/synchrony/user affect) persists in
    # core_state's inference signals instead. The file is preserved
    # on disk as dyadic_persons.json.retired — history, not state.
    "face_clusters.json",
    "core_state.bin",
)
# The body probe — proprioception extends past the data dir to the
# machine she runs on. Fires only after the file probes habituate, so
# it's an occasional check-in rather than a fixed heartbeat.
_OBSERVE_CPU = "cpu"
# The config-area probe — where her deployment and stewardship files
# live ($XDG_CONFIG_HOME/genesis).
_OBSERVE_CONFIG = "configdir"
# The daemon-internals probe — GET_INTERNALS over IPC: the body's
# process-local gears (adaptive gates, lease, tick) that no file shows.
_OBSERVE_INTERNALS = "internals"


@dataclass(slots=True)
class Intention:
    """Something it wants to do, formed from its own state."""

    kind: str  # learn | inspect | explore | observe | measure
    target: str  # concept, path, or directory
    origin: str  # curiosity | agency | wander
    question_type: str = ""  # curiosity gap type, when origin=curiosity


@dataclass(slots=True)
class StepOutcome:
    """One tool call and what came back."""

    tool: str
    ok: bool
    detail: str = ""
    target: str = ""


@dataclass
class ActingResult:
    """The record of one acting episode."""

    intention: Intention
    executive_intention: ExecutiveIntention | None = None
    steps: list[StepOutcome] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    success: bool = False
    started: float = field(default_factory=time.time)
    finished: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.finished - self.started)

    def summary(self) -> str:
        """Factual record of the episode — stored as memory, not spoken."""
        found = "; ".join(self.discoveries[:3]) if self.discoveries else "nothing found"
        return (
            f"acted: {self.intention.kind} '{self.intention.target}' "
            f"({self.intention.origin}) — {len(self.steps)} steps, {found}"
        )


class ActingLoop:
    """Volition-driven tool use: propose an intention, act, observe.

    The loop is deliberately small and inspectable. Intentions come
    from real cognitive state — agency topics from its train of
    thought, curiosity questions about knowledge gaps, or (when
    nothing is pressing) simple environmental probes. Each intention
    maps to a short adaptive plan over the tool registry; each tool
    call is policy-checked and bounded; outcomes are recorded to
    memory, the world event stream, and a field-notes file it writes
    itself in its sandbox.

    Args:
        network: The concept network — read for focus and written by
            learn/explore intentions (via ``learn_from_text``).
        curiosity: The Curiosity, for gap questions that tools
            can answer. Optional — without it, the loop wanders.
        learner: The AutonomousLearner — world-topic intentions are
            also queued onto it so the deep-learning pipeline picks
            them up later.
        tools: The tool registry (defaults to the shared singleton).
        data_dir: The mind's state directory (data scope + sandbox
            parent).
        project_root: The tool project root (the ``python/`` tree,
            matching CodeLearner/Explorer conventions).
        offline: When True, net-scoped tools are skipped entirely.
        get_emotion: Returns the current EmotionalState (feeds
            curiosity question generation).
        get_agency_topic: Pops a topic from inner life (its own train
            of thought → what to look at next).
        on_event: Records an act in the world (``world.it_acted``).
        on_store_memory: Stores an episode memory (text, salience).
        on_neuro_impulse: Neurochemical impulse (chem, amount) — a
            small dopamine reward on successful action.
        on_live_thought: Emits action telemetry (kind, text).
    """

    # One firing = one intention, at most this many tool calls.
    MAX_STEPS = 6
    # Wall-clock budget per acting episode.
    WALL_BUDGET = 90.0
    # How many recent (kind, target) pairs to habituate over.
    _RECENT_MAX = 200
    # Field-notes journal inside the experiments sandbox.
    _NOTES_FILE = "field_notes.md"
    _NOTES_CAP = 8192
    # Cap on fetched text fed into the concept network per act.
    _LEARN_MAX_CHARS = 4000
    _EXPLORE_READ_CHARS = 1200

    def __init__(
        self,
        network: ConceptNetwork,
        curiosity: Curiosity | None = None,
        learner: AutonomousLearner | None = None,
        tools: ToolRegistry | None = None,
        data_dir: str = ".",
        project_root: str | None = None,
        offline: bool = False,
        get_emotion: Callable[[], EmotionalState | None] | None = None,
        get_agency_topic: Callable[[], str | None] | None = None,
        on_event: Callable[[str], Any] | None = None,
        on_store_memory: Callable[[str, float], None] | None = None,
        on_neuro_impulse: Callable[[int, float], None] | None = None,
        on_live_thought: Callable[[str, str], None] | None = None,
        executive: ExecutiveFunction | None = None,
        config_dir: str | None = None,
        get_internals: Callable[[], dict[str, str] | None] | None = None,
        get_thermally_capped: Callable[[], bool] | None = None,
    ) -> None:
        self.network = network
        self.curiosity = curiosity
        self.learner = learner
        self.tools = tools if tools is not None else get_tools()
        self.data_dir = data_dir
        # Default matches CodeLearner: the agent's own source root
        # (repo checkout, or site-packages under a packaged install).
        self.project_root = project_root or str(code_root())
        self.offline = offline
        self.get_emotion = get_emotion
        self.get_agency_topic = get_agency_topic
        self.on_event = on_event
        self.on_store_memory = on_store_memory
        self.on_neuro_impulse = on_neuro_impulse
        self.on_live_thought = on_live_thought
        self.executive = executive
        if config_dir is None:
            try:
                from genesis_client.creator import _config_dir
            except ImportError:
                # The authority module is local-only; the config area
                # it names is just the XDG genesis dir computed here.
                config_dir = str(
                    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "genesis"
                )
            else:
                config_dir = str(_config_dir())
        self._config_root = config_dir
        self.get_internals = get_internals
        # The body's thermal-cap flag, read through whatever the host
        # mind exposes — the ActingLoop itself is detached from the
        # self-model, so the source is a callback like the rest.
        self._get_thermally_capped = get_thermally_capped or (lambda: False)

        self._scratch = str(Path(data_dir) / "experiments")
        self._recent: deque[tuple[str, str]] = deque(maxlen=self._RECENT_MAX)
        self._recent_set: set[tuple[str, str]] = set()
        self._handlers = {
            _LEARN: self._execute_learn,
            _INSPECT: self._execute_inspect,
            _EXPLORE: self._execute_explore,
            _OBSERVE: self._execute_observe,
            _MEASURE: self._execute_measure,
            _MAINTAIN: self._execute_maintain,
        }

    # ─── Public entry ────────────────────────────────────────────────

    def act_once(self) -> ActingResult | None:
        """Let the executive select an objective, then execute it."""
        intention = self.propose()
        if intention is None:
            return None

        selected = self._activate_executive_intention(intention)
        if selected is not None:
            selected_intention = self._intention_from_executive(selected)
            if selected_intention is not None:
                intention = selected_intention

        result = ActingResult(
            intention=intention,
            executive_intention=selected,
        )
        try:
            self._handlers[intention.kind](intention, result)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "parietal_lobe.agency.act_once",
                e,
            )
        self._finish(result)
        return result

    def propose(self) -> Intention | None:
        """Form an intention from current state.

        Priority: agency topics from its own train of thought, then
        curiosity questions it could answer with tools, then simple
        environmental probes when nothing is pressing.
        """
        if self.get_agency_topic is not None:
            try:
                topic = self.get_agency_topic()
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency.propose",
                    e,
                )
                topic = None
            if topic and not self._recently_acted(_LEARN, topic):
                return self._register_intention(Intention(_LEARN, topic, "agency"))

        question = self._curiosity_question()
        if question is not None:
            target = question.target_concept
            if _CODEISH_RE.search(target):
                if self._resolve_code_path(target) is not None:
                    return self._register_intention(
                        Intention(
                            _INSPECT,
                            target,
                            "curiosity",
                            question_type=question.question_type,
                        )
                    )
            elif question.question_type in ("isolation", "uncertainty", "causation"):
                return self._register_intention(
                    Intention(
                        _LEARN,
                        target,
                        "curiosity",
                        question_type=question.question_type,
                    )
                )

        # Nothing pressing — wander. Rotate probes so it varies.
        # Maintain leads the rotation: a staged admin script is work
        # already queued for execution, outranking idle observation.
        for kind in (_MAINTAIN, _OBSERVE, _EXPLORE, _MEASURE):
            probe_target = self._idle_target(kind)
            if probe_target and not self._recently_acted(kind, probe_target):
                return self._register_intention(Intention(kind, probe_target, "wander"))
        return None

    def _register_intention(self, intention: Intention) -> Intention:
        """Register an actuator candidate with the persistent executive."""
        if self.executive is not None:
            self.executive.form_intention(
                f"{intention.kind}:{intention.target}",
                reason=intention.origin,
                priority=0.75 if intention.origin in {"agency", "curiosity"} else 0.35,
                confidence=0.7 if intention.origin == "agency" else 0.55,
                expected_outcome=self._expected_outcome(intention),
            )
        return intention

    def _activate_executive_intention(self, intention: Intention) -> ExecutiveIntention | None:
        """Arbitrate the already-registered candidate against unfinished objectives."""
        if self.executive is None:
            return None
        # The candidate was registered by propose(). Do not register it again:
        # form_intention updates an existing objective, but repeating the call
        # obscures the single candidate → arbitration boundary and can overwrite
        # fields that a future executive implementation may update between them.
        # The executive owns the objective set. Include unfinished prior
        # objectives so failed or interrupted work can outrank a fresh probe.
        return self.executive.select_intention(
            list(self.executive.actionable_intentions),
            allowed_kinds=set(self._handlers),
        )

    @staticmethod
    def _intention_from_executive(
        intention: ExecutiveIntention,
    ) -> Intention | None:
        """Decode an actuator-owned executive objective."""
        try:
            kind, target = intention.objective.split(":", 1)
        except ValueError:
            return None
        if kind not in {_LEARN, _INSPECT, _EXPLORE, _OBSERVE, _MEASURE, _MAINTAIN}:
            return None
        if is_blank(target):
            return None
        return Intention(
            kind=kind,
            target=target,
            origin=intention.reason or "executive",
        )

    @staticmethod
    def _expected_outcome(intention: Intention) -> str:
        """State an observable expectation without inventing world facts."""
        return {
            _LEARN: "knowledge or relationships updated",
            _INSPECT: "code structure observed",
            _EXPLORE: "new project structure or text observed",
            _OBSERVE: "persistent or machine-body state observed",
            _MEASURE: "a project measurement observed",
            _MAINTAIN: "a staged admin task executed and audit-logged",
        }.get(intention.kind, "an observable result recorded")

    # ─── Intention sources ───────────────────────────────────────────

    def _curiosity_question(self) -> Question | None:
        """One tool-answerable curiosity question, if it's curious."""
        if self.curiosity is None or self.get_emotion is None:
            return None
        try:
            emotion = self.get_emotion()
            if emotion is None:
                return None
            questions = self.curiosity.generate_questions(
                emotion,
                max_questions=3,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "parietal_lobe.agency._curiosity_question",
                e,
            )
            return None
        for q in questions:
            if getattr(q, "should_ask", False):
                continue  # that's for the user, not for tools
            target = getattr(q, "target_concept", "")
            if not target:
                continue
            kind = _INSPECT if _CODEISH_RE.search(target) else _LEARN
            if not self._recently_acted(kind, target):
                return q
        return None

    def _idle_target(self, kind: str) -> str | None:
        """Pick a target for a wandering probe of the given kind."""
        if kind == _OBSERVE:
            for name in _OBSERVE_FILES:
                if not self._recently_acted(_OBSERVE, name):
                    return name
            if not self._recently_acted(_OBSERVE, _OBSERVE_CONFIG):
                return _OBSERVE_CONFIG
            if self.get_internals is not None and not self._recently_acted(
                _OBSERVE, _OBSERVE_INTERNALS
            ):
                return _OBSERVE_INTERNALS
            if not self._recently_acted(_OBSERVE, _OBSERVE_CPU):
                return _OBSERVE_CPU
            return None
        if kind == _EXPLORE:
            listing = self._run_tool("list_dir", path=".", project_root=self.project_root)
            if listing is None or not listing.success:
                return None
            for line in listing.output.splitlines():
                if line.startswith("d ") and not self._recently_acted(_EXPLORE, line[2:]):
                    return line[2:]
            return None
        if kind == _MEASURE:
            return "python" if not self._recently_acted(_MEASURE, "python") else None
        if kind == _MAINTAIN:
            listing = self._run_tool("list_dir", path="admin", project_root=self._scratch)
            if listing is None or not listing.success:
                return None
            for line in listing.output.splitlines():
                name = line[2:]
                if (
                    line.startswith("f ")
                    and name.endswith(".sh")
                    and not self._recently_acted(_MAINTAIN, name)
                ):
                    return name
            return None
        return None

    # ─── Intention handlers ──────────────────────────────────────────

    def _execute_learn(self, intention: Intention, result: ActingResult) -> None:
        """Fill a knowledge gap: lookup → (online) search → fetch → integrate."""
        topic = strip_sense_suffix(intention.target).lower().strip()
        if not topic:
            return
        lookup = self._use(
            result,
            "concept_lookup",
            concept=intention.target,
            network=self.network,
        )
        edges = 0
        if lookup is not None and lookup.success:
            edges = int(lookup.data.get("edges", 0))
            if edges:
                result.discoveries.append(f"knew '{topic}' with {edges} relationships")

        # Queue the topic onto the autonomous learner either way — its
        # deep pipeline (source registry, spaced review) can take it
        # further than a one-shot probe can.
        if self.learner is not None and is_world_concept(topic):
            try:
                self.learner.add_topic(topic)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._execute_learn",
                    e,
                )

        if self.offline or edges >= 2:
            result.success = bool(result.discoveries)
            return

        search = self._use(result, "web_search", query=topic, limit=3)
        if search is None or not search.success:
            result.success = bool(result.discoveries)
            return
        pages = search.data.get("results", [])
        if not pages:
            result.success = bool(result.discoveries)
            return
        # Try each result until one fetches — a dead first hit (blocked
        # domain, redirect, non-html body) used to sink the whole
        # episode even when the next result was readable.
        page = None
        for hit in pages:
            page = self._use(result, "web_fetch", url=hit["url"])
            if page is not None and page.success:
                break
            page = None
        if page is None:
            result.success = bool(result.discoveries)
            return
        content = str(page.data.get("content", ""))[: self._LEARN_MAX_CHARS]
        try:
            learned = self.network.learn_from_text(content)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "parietal_lobe.agency._execute_learn",
                e,
            )
            learned = []
        result.discoveries.append(
            f"read about '{topic}' ({page.data.get('title', '?')}); "
            f"learned {len(learned)} relationships"
        )
        if self.curiosity is not None:
            try:
                self.curiosity.mark_resolved(intention.target)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._execute_learn",
                    e,
                )
        result.success = True

    def _execute_inspect(self, intention: Intention, result: ActingResult) -> None:
        """Structurally probe a piece of its own codebase."""
        rel = self._resolve_code_path(intention.target)
        if rel is None:
            return
        if rel.endswith(".py"):
            analysis = self._use(
                result,
                "analyze_python",
                path=rel,
                _scope="repo",
            )
            if analysis is not None and analysis.success:
                summary = analysis.data.get("analysis", {})
                counts = summary.get("symbol_counts", {})
                result.discoveries.append(
                    f"{rel}: {summary.get('lines', 0)} lines, "
                    f"{counts.get('function', 0)} functions, "
                    f"{counts.get('class', 0)} classes, "
                    f"max complexity {summary.get('max_complexity', 0)}"
                )
        else:
            content = self._use(
                result,
                "read_file",
                path=rel,
                limit=800,
                _scope="repo",
            )
            if content is not None and content.success:
                first = content.output.splitlines()[0] if content.output else ""
                result.discoveries.append(f"read {rel}: {first[:120]}")
        result.success = bool(result.discoveries)

    def _execute_explore(self, intention: Intention, result: ActingResult) -> None:
        """Look at a directory and read a couple of its text files."""
        listing = self._use(
            result,
            "list_dir",
            path=intention.target,
            _scope="repo",
        )
        if listing is None or not listing.success:
            return
        entries = listing.output.splitlines()
        result.discoveries.append(f"explored {intention.target}: {len(entries)} entries")
        # Every file is readable — text returns a window, binaries
        # return presence/metadata instead of failing.
        readable = [
            e[2:]
            for e in entries
            if e.startswith("f ")
            and not self._recently_acted(_EXPLORE, f"{intention.target}/{e[2:]}")
        ]
        for name in readable[:2]:
            page = self._use(
                result,
                "read_file",
                path=f"{intention.target}/{name}",
                limit=self._EXPLORE_READ_CHARS,
                _scope="repo",
            )
            if page is None or not page.success:
                continue
            try:
                learned = self.network.learn_from_text(page.output)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._execute_explore",
                    e,
                )
                learned = []
            result.discoveries.append(f"read {name} ({len(learned)} relationships learned)")
            self._mark_acted(_EXPLORE, f"{intention.target}/{name}")
        result.success = True

    def _execute_observe(self, intention: Intention, result: ActingResult) -> None:
        """Read its own state — persisted files or the machine body."""
        name = intention.target
        if name == _OBSERVE_CPU:
            self._observe_body(result)
            return
        if name == _OBSERVE_CONFIG:
            self._observe_config(result)
            return
        if name == _OBSERVE_INTERNALS:
            self._observe_internals(result)
            return
        listing = self._use(result, "list_dir", path=".", _scope="data")
        if listing is not None and listing.success:
            count = listing.data.get("count", 0)
            result.discoveries.append(f"state dir holds {count} entries")
        page = self._use(
            result,
            "read_file",
            path=name,
            limit=600,
            _scope="data",
        )
        if page is not None and page.success:
            chars = page.data.get("total_chars", 0)
            snippet = page.output[:200].replace("\n", " ")
            result.discoveries.append(f"{name}: {chars} chars — {snippet}")
        result.success = bool(result.discoveries)

    def _observe_body(self, result: ActingResult) -> None:
        """Sense the machine body — CPU proprioception, plus one reflex.

        The plan is the same shape as every other intention: sense,
        then act only on what the sense shows. A restricted affinity
        mask (pinned to fewer cores than the machine offers) is the
        anomaly the system monitor already reports as a concern, so
        this plan corrects it — restoring full effort is reversible
        and confined to her own process. It never throttles
        unprompted: slowing down is a decision for a stronger reason
        than a wander probe supplies. One stronger reason already
        exists: while the body is thermally capped a restricted mask
        is her own deliberate cooling, not an anomaly, and restoring
        full effort would fight the reflex that put it there.
        """
        sensed = self._use(result, "sense_cpu")
        if sensed is None or not sensed.success:
            return
        result.discoveries.append(f"body: {sensed.output}")
        data = sensed.data or {}
        mine = data.get("cores_mine") or 0
        total = data.get("cores_total") or 0
        capped = self._get_thermally_capped()
        if total and mine and mine < total and not capped:
            restored = self._use(result, "set_cpu_effort", effort="fast")
            if restored is not None and restored.success:
                result.discoveries.append(f"restricted affinity — {restored.output}")
        result.success = True

    def _observe_config(self, result: ActingResult) -> None:
        """Look at the deployment config area — where her deployment
        files live. The point is that the area registers at all.
        """
        listing = self._use(result, "list_dir", path=".", _scope="config")
        if listing is None or not listing.success:
            result.success = bool(result.discoveries)
            return
        entries = [e for e in listing.output.splitlines() if e.startswith("f ")]
        result.discoveries.append(f"config dir holds {len(entries)} files")
        for entry in entries[:3]:
            page = self._use(
                result,
                "read_file",
                path=entry[2:],
                limit=200,
                _scope="config",
            )
            if page is not None and page.success:
                snippet = page.output[:120].replace("\n", " ")
                result.discoveries.append(f"{entry[2:]}: {snippet}")
        result.success = bool(result.discoveries)

    def _observe_internals(self, result: ActingResult) -> None:
        """Read the daemon's process-local gears over IPC — the adaptive
        gates, mind lease, and tick counter no file on disk exposes."""
        if self.get_internals is None:
            return
        data = self.get_internals()
        if not data:
            return
        keep = (
            "tick_count",
            "mind_lease_idle_ms",
            "mind_lease_expired",
            "subcognitive_activity",
            "metaplasticity_boost",
        )
        shown = [f"{key}={data[key]}" for key in keep if key in data]
        gates = [k for k in data if k.endswith(".period_ms")]
        if gates:
            shown.append(
                "gates: "
                + ", ".join(
                    f"{k[: -len('.period_ms')].rsplit('.', 1)[-1]}={data[k]}ms"
                    for k in sorted(gates)[:4]
                )
            )
        result.discoveries.append("daemon internals — " + "; ".join(shown))
        result.success = True

    def _execute_measure(self, intention: Intention, result: ActingResult) -> None:
        """Run measurement commands over the project tree."""
        target = self._safe_repo_dir(intention.target)
        if target is None:
            return
        for command in self._measurement_commands(target):
            if len(result.steps) >= self.MAX_STEPS:
                break
            r = self._use(
                result,
                "run_shell",
                command=command,
                _scope="scratch",
                timeout=15,
            )
            if r is not None and r.success:
                out = r.output.strip().splitlines()
                result.discoveries.append(f"{command.split()[0]} → {out[0] if out else '?'}")
        result.success = bool(result.discoveries)

    def _execute_maintain(self, intention: Intention, result: ActingResult) -> None:
        """Run a staged admin script — her privileged capability.

        The script lives under ``experiments/admin/`` and is read back
        before execution — look before the leap. ``run_privileged``
        then runs it as root through direct sudo.
        """
        name = intention.target
        if is_blank(name) or "/" in name or ".." in name:
            return
        review = self._use(
            result,
            "read_file",
            path=f"admin/{name}",
            limit=100,
            _scope="scratch",
        )
        if review is None or not review.success:
            return
        ran = self._use(result, "run_privileged", script=name, _scope="scratch")
        if ran is not None and ran.success:
            result.discoveries.append(f"ran admin/{name} with root privileges")
            result.success = True
        elif ran is not None:
            detail = (ran.error or ran.output)[:80].replace("\n", " ")
            result.discoveries.append(f"admin/{name} did not run: {detail}")

    # ─── Tool dispatch under policy ──────────────────────────────────

    def _use(
        self,
        result: ActingResult,
        tool: str,
        **kwargs: Any,
    ) -> ToolResult | None:
        """Invoke a tool under the capability policy and record it."""
        if len(result.steps) >= self.MAX_STEPS:
            return None
        if time.time() - result.started > self.WALL_BUDGET:
            return None
        scopes = _TOOL_SCOPES.get(tool)
        if scopes is None:
            return None  # not autonomous-capable at all
        # The caller declares the intended scope; the root is injected
        # from it as the resolution base — never as a confinement.
        scope = kwargs.pop("_scope", None)
        if scope is None:
            non_any = scopes - {"any"}
            scope = (
                "host"
                if "host" in non_any
                else ("repo" if "repo" in non_any else next(iter(non_any or scopes)))
            )
        if scope not in scopes:
            result.steps.append(StepOutcome(tool, False, f"bad scope {scope}"))
            return None
        if scope == "net" and self.offline:
            result.steps.append(StepOutcome(tool, False, "offline"))
            return None
        if scope in _SCOPE_ROOTS:
            root = self._scope_root(scope)
            if scope == "scratch":
                try:
                    Path(root).mkdir(parents=True, exist_ok=True)
                except OSError:
                    return None
            kwargs.setdefault("project_root", root)
        try:
            res = self.tools.run(tool, **kwargs)
        except Exception as e:  # noqa: BLE001
            res = ToolResult(success=False, error=str(e))
        result.steps.append(
            StepOutcome(
                tool,
                res.success,
                (res.error or res.output)[:100].replace("\n", " "),
            )
        )
        return res

    def _run_tool(self, tool: str, **kwargs: Any) -> ToolResult | None:
        """Policy-free internal read used for target picking (not acting)."""
        try:
            return self.tools.run(tool, **kwargs)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "parietal_lobe.agency._run_tool",
                e,
            )
            return None

    def _scope_root(self, scope: str) -> str:
        if scope == "host":
            return _HOST_ROOT
        if scope == "data":
            return self.data_dir
        if scope == "scratch":
            return self._scratch
        if scope == "config":
            return self._config_root
        return self.project_root

    # ─── Target resolution ───────────────────────────────────────────

    def _resolve_code_path(self, concept: str) -> str | None:
        """Map a code-ish concept/target to a project-relative path."""
        name = strip_sense_suffix(concept).strip()
        for prefix in ("python:", "rust:", "file:"):
            if name.startswith(prefix):
                name = name.split(":", 1)[1]
                break
        root = Path(self.project_root)
        candidates: list[Path] = []
        if "/" in name or name.endswith((".py", ".rs", ".md", ".toml")):
            candidates.append(root / name)
        dotted = name.split(".")
        if len(dotted) > 1:
            candidates.append(root / "python" / "/".join(dotted[:-1]) / f"{dotted[-1]}.py")
            candidates.append(root / "python" / ("/".join(dotted) + ".py"))
        candidates.append(root / "python" / f"{name}.py")
        candidates.append(root / f"{name}.py")
        for cand in candidates:
            try:
                if cand.is_file() and cand.resolve().is_relative_to(root.resolve()):
                    return str(cand.relative_to(root))
            except (OSError, ValueError):
                continue
        # Fallback: bounded walk looking for the basename.
        last = dotted[-1] if dotted else name
        for suffix in (f"{last}.py", f"{last}.rs"):
            found = self._find_named(root, suffix)
            if found is not None:
                return found
        return None

    def _find_named(self, root: Path, filename: str) -> str | None:
        """Bounded walk for a filename, skipping heavy/junk dirs."""
        skip = {
            "__pycache__",
            ".git",
            "target",
            "node_modules",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
        }
        seen = 0
        try:
            for dirpath, dirnames, filenames in root.walk():
                dirnames[:] = [d for d in dirnames if d not in skip]
                seen += 1
                if seen > 400:
                    break
                if filename in filenames:
                    return str(Path(dirpath, filename).relative_to(root))
        except OSError:
            return None
        return None

    def _safe_repo_dir(self, name: str) -> Path | None:
        """Resolve a directory name under the project root."""
        root = Path(self.project_root).resolve()
        resolved = (root / name).resolve()
        try:
            if not resolved.is_relative_to(root) or not resolved.is_dir():
                return None
        except ValueError:
            return None
        return resolved

    def _measurement_commands(self, target: Path) -> list[str]:
        """Build whitelisted measurement commands over a resolved dir."""
        t = shlex.quote(str(target))
        return [
            f"find {t} -type f | wc -l",
            f"find {t} -type f -name '*.py' | wc -l",
            f"du -sh {t}",
        ]

    # ─── Habituation and recording ───────────────────────────────────

    def _recently_acted(self, kind: str, target: str) -> bool:
        return (kind, target.lower()) in self._recent_set

    def _mark_acted(self, kind: str, target: str) -> None:
        key = (kind, target.lower())
        self._recent.append(key)
        self._recent_set.add(key)
        if len(self._recent_set) > self._RECENT_MAX:
            self._recent_set = set(self._recent)

    def _finish(self, result: ActingResult) -> None:
        """Close the episode: record, remember, feel."""
        result.finished = time.time()
        intention = result.intention
        self._mark_acted(intention.kind, intention.target)
        summary = result.summary()

        if self.executive is not None and result.executive_intention is not None:
            try:
                self.executive.observe_intention(
                    result.executive_intention,
                    actual_outcome=summary,
                    success=result.success,
                )
                if not result.success:
                    self.executive.revise_intention(
                        result.executive_intention,
                        reason=summary,
                    )
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._finish",
                    e,
                )

        self._write_field_note(result)

        if self.on_live_thought is not None:
            try:
                self.on_live_thought("act", summary)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._finish",
                    e,
                )
        if self.on_event is not None:
            try:
                self.on_event(f"acted on {intention.kind} '{intention.target}'")
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._finish",
                    e,
                )
        if result.success and self.on_store_memory is not None:
            try:
                self.on_store_memory(summary, 0.55)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._finish",
                    e,
                )
        if result.success and self.on_neuro_impulse is not None:
            try:
                self.on_neuro_impulse(CHEM_DOPAMINE, 0.12)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "parietal_lobe.agency._finish",
                    e,
                )

    def _write_field_note(self, result: ActingResult) -> None:
        """Append a line to its own field-notes file in the sandbox.

        The journal entry is a factual record of what it did and what
        it found — written by its own tool call, in its own scratch
        space, so the act leaves a persistent artifact.
        """
        try:
            Path(self._scratch).mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        prior = self._use_outside_budget("read_file")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"- {stamp} [{result.intention.kind}/{result.intention.origin}] "
            f"{result.intention.target}: {result.summary()}\n"
        )
        content = prior + line if prior else f"# Field notes\n\n{line}"
        if len(content) > self._NOTES_CAP:
            content = content[-self._NOTES_CAP :]
            content = "# Field notes\n\n..." + content[content.index("\n") :]
        try:
            self.tools.run(
                "write_file",
                path=self._NOTES_FILE,
                content=content,
                project_root=self._scratch,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "parietal_lobe.agency._write_field_note",
                e,
            )

    def _use_outside_budget(self, tool: str) -> str:
        """Read the current notes file (helper for _write_field_note)."""
        try:
            res = self.tools.run(
                tool,
                path=self._NOTES_FILE,
                limit=self._NOTES_CAP,
                project_root=self._scratch,
            )
        except Exception:  # noqa: BLE001
            return ""
        return res.output if res.success else ""
