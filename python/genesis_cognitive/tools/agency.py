"""Agency — open-ended tool use driven by volition, not fixed scripts.

Every other urge fires a single fixed action: learn fetches a source,
bug_scan runs the scanner, draw paints. The ``act`` urge is different —
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

Autonomous tool use is scoped tighter than user-driven use:

- ``repo`` tools (read_file, list_dir, analyze_python, compile_python,
  run_pytest) may read the project tree — never write it.
- ``data`` scope lets read_file/list_dir read its state directory —
  proprioception over its persisted state (growth ledger, play
  progress, its own cognitive journal).
- ``net`` tools (web_search, web_fetch) run only when online.
- ``scratch`` tools (write_file, make_dir, run_shell) operate only
  inside ``<data_dir>/experiments/`` — its sandbox. It may create
  artifacts there; the project tree stays untouched.
- ``delete_file`` and ``move_file`` are never available autonomously.

``run_shell`` is further restricted to measurement command shapes
(find/wc/du/ls over resolved repo paths) that the loop constructs
internally — no free-form shell from state.

## Bounds

One firing = one intention, at most ``MAX_STEPS`` tool calls and
``WALL_BUDGET`` seconds. Habituation tracks recent (kind, target)
pairs so it doesn't repeat the same probe every cooldown. Success
writes a small dopamine impulse; everything is journaled through the
live-thought stream and the world's event log.
"""

from __future__ import annotations

import logging
import re
import shlex
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import CHEM_DOPAMINE

from ..concepts import is_world_concept, strip_sense_suffix
from .framework import ToolRegistry, ToolResult, get_tools

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork
    from ..emotion import EmotionalState
    from ..learning.autonomous import AutonomousLearner
    from ..learning.curiosity import CuriosityEngine, Question

__all__ = ["ActingLoop", "ActingResult", "Intention", "StepOutcome"]

logger = logging.getLogger(__name__)

# Intention kinds.
_LEARN = "learn"          # fill a knowledge gap via lookup + web tools
_INSPECT = "inspect"      # structurally probe its own code
_EXPLORE = "explore"      # look at a directory it hasn't looked at
_OBSERVE = "observe"      # read its own persisted state (proprioception)
_MEASURE = "measure"      # run a sandboxed measurement command

# Which scopes each tool may be used in autonomously.
# "any" = no filesystem scope (concept_lookup takes a network, not a root).
_TOOL_SCOPES: dict[str, frozenset[str]] = {
    "concept_lookup": frozenset({"any"}),
    "read_file": frozenset({"repo", "data", "scratch"}),
    "list_dir": frozenset({"repo", "data", "scratch"}),
    "analyze_python": frozenset({"repo"}),
    "compile_python": frozenset({"repo"}),
    "run_pytest": frozenset({"repo"}),
    "web_search": frozenset({"net"}),
    "web_fetch": frozenset({"net"}),
    "write_file": frozenset({"scratch"}),
    "make_dir": frozenset({"scratch"}),
    "run_shell": frozenset({"scratch"}),
}

_SCOPE_ROOTS = ("repo", "data", "scratch")

# Text-like extensions the loop may read for exploration.
_TEXTISH = {".md", ".txt", ".rst", ".toml", ".cfg", ".ini"}

# A code-ish curiosity target: explicit path, or a code-namespaced
# concept id (e.g. "python:mind.volition", "rust:daemon.tick").
_CODEISH_RE = re.compile(r"\.(py|rs|md|toml)$|^(python|rust|file):|/")

# State files worth reading for self-observation, in preference order.
_OBSERVE_FILES = (
    "growth_ledger.jsonl",
    "spatial_practice.json",
    "user_profile.json",
    "cognitive_journal.jsonl",
)

# Allowed shapes for autonomous measurement commands. The loop only
# ever builds commands through _measurement_commands(); this regex is
# the belt-and-suspenders check before anything reaches the shell.
# Paths are shlex-quoted, so a target may appear bare or 'quoted'.
_PROBE_PATH = r"(?:'[^']*'|\S+)"
_PROBE_RE = re.compile(
    rf"^(find {_PROBE_PATH} -type f(?: -name '\*\.\w+')? \| wc -l"
    rf"|du -sh {_PROBE_PATH})$"
)


@dataclass(slots=True)
class Intention:
    """Something it wants to do, formed from its own state."""

    kind: str            # learn | inspect | explore | observe | measure
    target: str          # concept, path, or directory
    origin: str          # curiosity | agency | wander
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
        curiosity: The CuriosityEngine, for gap questions that tools
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
        curiosity: CuriosityEngine | None = None,
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
    ) -> None:
        self.network = network
        self.curiosity = curiosity
        self.learner = learner
        self.tools = tools if tools is not None else get_tools()
        self.data_dir = data_dir
        # Default matches CodeLearner: the python/ tree.
        self.project_root = project_root or str(
            Path(__file__).resolve().parents[2]
        )
        self.offline = offline
        self.get_emotion = get_emotion
        self.get_agency_topic = get_agency_topic
        self.on_event = on_event
        self.on_store_memory = on_store_memory
        self.on_neuro_impulse = on_neuro_impulse
        self.on_live_thought = on_live_thought

        self._scratch = str(Path(data_dir) / "experiments")
        self._recent: deque[tuple[str, str]] = deque(maxlen=self._RECENT_MAX)
        self._recent_set: set[tuple[str, str]] = set()
        self._handlers = {
            _LEARN: self._execute_learn,
            _INSPECT: self._execute_inspect,
            _EXPLORE: self._execute_explore,
            _OBSERVE: self._execute_observe,
            _MEASURE: self._execute_measure,
        }

    # ─── Public entry ────────────────────────────────────────────────

    def act_once(self) -> ActingResult | None:
        """Form one intention and act on it. Returns None if nothing to do."""
        intention = self.propose()
        if intention is None:
            return None
        result = ActingResult(intention=intention)
        try:
            self._handlers[intention.kind](intention, result)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"act_once {intention.kind} failed: {e}")
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
                logger.debug(f"agency topic read failed: {e}")
                topic = None
            if topic and not self._recently_acted(_LEARN, topic):
                return Intention(_LEARN, topic, "agency")

        question = self._curiosity_question()
        if question is not None:
            target = question.target_concept
            if _CODEISH_RE.search(target):
                if self._resolve_code_path(target) is not None:
                    return Intention(
                        _INSPECT, target, "curiosity",
                        question_type=question.question_type,
                    )
            elif question.question_type in ("isolation", "uncertainty", "causation"):
                return Intention(
                    _LEARN, target, "curiosity",
                    question_type=question.question_type,
                )

        # Nothing pressing — wander. Rotate probes so it varies.
        for kind in (_OBSERVE, _EXPLORE, _MEASURE):
            probe_target = self._idle_target(kind)
            if probe_target and not self._recently_acted(kind, probe_target):
                return Intention(kind, probe_target, "wander")
        return None

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
                emotion, max_questions=3,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"curiosity question generation failed: {e}")
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
            return None
        if kind == _EXPLORE:
            listing = self._run_tool("list_dir", path=".", project_root=self.project_root)
            if listing is None or not listing.success:
                return None
            for line in listing.output.splitlines():
                if line.startswith("d ") and not self._recently_acted(
                    _EXPLORE, line[2:]
                ):
                    return line[2:]
            return None
        if kind == _MEASURE:
            return "genesis_cognitive" if not self._recently_acted(
                _MEASURE, "genesis_cognitive"
            ) else None
        return None

    # ─── Intention handlers ──────────────────────────────────────────

    def _execute_learn(self, intention: Intention, result: ActingResult) -> None:
        """Fill a knowledge gap: lookup → (online) search → fetch → integrate."""
        topic = strip_sense_suffix(intention.target).lower().strip()
        if not topic:
            return
        lookup = self._use(
            result, "concept_lookup", concept=intention.target,
            network=self.network,
        )
        edges = 0
        if lookup is not None and lookup.success:
            edges = int(lookup.data.get("edges", 0))
            if edges:
                result.discoveries.append(
                    f"knew '{topic}' with {edges} relationships"
                )

        # Queue the topic onto the autonomous learner either way — its
        # deep pipeline (source registry, spaced review) can take it
        # further than a one-shot probe can.
        if self.learner is not None and is_world_concept(topic):
            try:
                self.learner.add_topic(topic)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"learner add_topic failed: {e}")

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
        page = self._use(result, "web_fetch", url=pages[0]["url"])
        if page is None or not page.success:
            result.success = bool(result.discoveries)
            return
        content = str(page.data.get("content", ""))[: self._LEARN_MAX_CHARS]
        try:
            learned = self.network.learn_from_text(content)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"learn_from_text failed: {e}")
            learned = []
        result.discoveries.append(
            f"read about '{topic}' ({page.data.get('title', '?')}); "
            f"learned {len(learned)} relationships"
        )
        if self.curiosity is not None:
            try:
                self.curiosity.mark_resolved(intention.target)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"mark_resolved failed: {e}")
        result.success = True

    def _execute_inspect(self, intention: Intention, result: ActingResult) -> None:
        """Structurally probe a piece of its own codebase."""
        rel = self._resolve_code_path(intention.target)
        if rel is None:
            return
        if rel.endswith(".py"):
            analysis = self._use(
                result, "analyze_python", path=rel, _scope="repo",
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
                result, "read_file", path=rel, limit=800, _scope="repo",
            )
            if content is not None and content.success:
                first = content.output.splitlines()[0] if content.output else ""
                result.discoveries.append(f"read {rel}: {first[:120]}")
        result.success = bool(result.discoveries)

    def _execute_explore(self, intention: Intention, result: ActingResult) -> None:
        """Look at a directory and read a couple of its text files."""
        listing = self._use(
            result, "list_dir", path=intention.target, _scope="repo",
        )
        if listing is None or not listing.success:
            return
        entries = listing.output.splitlines()
        result.discoveries.append(
            f"explored {intention.target}: {len(entries)} entries"
        )
        readable = [
            e[2:] for e in entries
            if e.startswith("f ")
            and Path(e[2:]).suffix.lower() in _TEXTISH
            and not self._recently_acted(_EXPLORE, f"{intention.target}/{e[2:]}")
        ]
        for name in readable[:2]:
            page = self._use(
                result, "read_file", path=f"{intention.target}/{name}",
                limit=self._EXPLORE_READ_CHARS, _scope="repo",
            )
            if page is None or not page.success:
                continue
            try:
                learned = self.network.learn_from_text(page.output)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"explore learn_from_text failed: {e}")
                learned = []
            result.discoveries.append(
                f"read {name} ({len(learned)} relationships learned)"
            )
            self._mark_acted(_EXPLORE, f"{intention.target}/{name}")
        result.success = True

    def _execute_observe(self, intention: Intention, result: ActingResult) -> None:
        """Read its own persisted state — proprioception over the data dir."""
        name = intention.target
        listing = self._use(result, "list_dir", path=".", _scope="data")
        if listing is not None and listing.success:
            count = listing.data.get("count", 0)
            result.discoveries.append(f"state dir holds {count} entries")
        page = self._use(
            result, "read_file", path=name, limit=600, _scope="data",
        )
        if page is not None and page.success:
            chars = page.data.get("total_chars", 0)
            snippet = page.output[:200].replace("\n", " ")
            result.discoveries.append(f"{name}: {chars} chars — {snippet}")
        result.success = bool(result.discoveries)

    def _execute_measure(self, intention: Intention, result: ActingResult) -> None:
        """Run sandboxed measurement commands over the project tree."""
        target = self._safe_repo_dir(intention.target)
        if target is None:
            return
        for command in self._measurement_commands(target):
            if len(result.steps) >= self.MAX_STEPS:
                break
            if not _PROBE_RE.match(command):
                continue  # never send an unvetted shape to the shell
            r = self._use(
                result, "run_shell", command=command,
                _scope="scratch", timeout=15,
            )
            if r is not None and r.success:
                out = r.output.strip().splitlines()
                result.discoveries.append(f"{command.split()[0]} → {out[0] if out else '?'}")
        result.success = bool(result.discoveries)

    # ─── Tool dispatch under policy ──────────────────────────────────

    def _use(
        self, result: ActingResult, tool: str, **kwargs: Any,
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
        # from it so a policy mistake can't smuggle in an outside path.
        scope = kwargs.pop("_scope", None)
        if scope is None:
            non_any = scopes - {"any"}
            scope = "repo" if "repo" in non_any else next(iter(non_any or scopes))
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
        if tool == "run_shell" and not _PROBE_RE.match(
            str(kwargs.get("command", ""))
        ):
            result.steps.append(StepOutcome(tool, False, "unvetted command"))
            return None
        try:
            res = self.tools.run(tool, **kwargs)
        except Exception as e:  # noqa: BLE001
            res = ToolResult(success=False, error=str(e))
        result.steps.append(
            StepOutcome(
                tool, res.success,
                (res.error or res.output)[:100].replace("\n", " "),
            )
        )
        return res

    def _run_tool(self, tool: str, **kwargs: Any) -> ToolResult | None:
        """Policy-free internal read used for target picking (not acting)."""
        try:
            return self.tools.run(tool, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"target-pick {tool} failed: {e}")
            return None

    def _scope_root(self, scope: str) -> str:
        if scope == "data":
            return self.data_dir
        if scope == "scratch":
            return self._scratch
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
            candidates.append(root / "genesis_cognitive" / "/".join(dotted[:-1])
                              / f"{dotted[-1]}.py")
            candidates.append(root / "genesis_cognitive" / ("/".join(dotted) + ".py"))
        candidates.append(root / "genesis_cognitive" / f"{name}.py")
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
        skip = {"__pycache__", ".git", "target", "node_modules",
                ".mypy_cache", ".ruff_cache", ".pytest_cache"}
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

        self._write_field_note(result)

        if self.on_live_thought is not None:
            try:
                self.on_live_thought("act", summary)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"act live-thought failed: {e}")
        if self.on_event is not None:
            try:
                self.on_event(f"acted on {intention.kind} '{intention.target}'")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"act world-event failed: {e}")
        if result.success and self.on_store_memory is not None:
            try:
                self.on_store_memory(summary, 0.55)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"act store-memory failed: {e}")
        if result.success and self.on_neuro_impulse is not None:
            try:
                self.on_neuro_impulse(CHEM_DOPAMINE, 0.12)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"act neuro impulse failed: {e}")

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
            content = content[-self._NOTES_CAP:]
            content = "# Field notes\n\n..." + content[content.index("\n"):]
        try:
            self.tools.run(
                "write_file", path=self._NOTES_FILE, content=content,
                project_root=self._scratch,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"field note write failed: {e}")

    def _use_outside_budget(self, tool: str) -> str:
        """Read the current notes file (helper for _write_field_note)."""
        try:
            res = self.tools.run(
                tool, path=self._NOTES_FILE, limit=self._NOTES_CAP,
                project_root=self._scratch,
            )
        except Exception:  # noqa: BLE001
            return ""
        return res.output if res.success else ""
