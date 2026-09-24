"""Code tool handling — dispatch code tools and compose code discussions.

Extracted from CognitionEngine as a focused subsystem. When the user
mentions a .py file with an action word (compile, test, read, check),
the handler dispatches the appropriate tool and composes a response
from the tool output. For general code discussions, it composes from
concept-network knowledge about code-related topics.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for code concept lookup and neighbors
    - language: LanguageEngine for rendering file-not-found thoughts
    - composer: ThoughtComposer for concept-network composition fallbacks
    - tools: ToolRegistry for run_pytest, compile_python, read_file tools
    - resolve_topics: callable from TopicResolver
    - build_meta_emotion: callable for emotional state in error reports
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought
from ..perception import Perception

if TYPE_CHECKING:
    from ..cognition.thought_composer import ThoughtComposer
    from ..concepts import ConceptNetwork
    from ..tools.framework import ToolRegistry

__all__ = ["CodeToolHandler"]


class CodeToolHandler:
    """Dispatch code tools and compose code-discussion responses.

    Tool output is reported directly (it's factual tool output, not
    Genesis's self-expression). Code discussion fallbacks compose from
    concept-network knowledge, never from hardcoded template strings.
    """

    # Action words that trigger tool dispatch. Substring matching is
    # intentional — "test" matches "test", "tests", and "pytest".
    _ACTION_WORDS: tuple[str, ...] = (
        "compile", "test", "read", "show", "check", "analyze", "impact",
    )

    def __init__(
        self,
        network: ConceptNetwork,
        language: LanguageEngine,
        composer: ThoughtComposer,
        tools: ToolRegistry,
        resolve_topics: Callable[[list[str], str], list[str]],
        build_meta_emotion: Callable[[], EmotionalState],
    ) -> None:
        """Wire the tool dispatcher to its dependencies."""
        self._network = network
        self._language = language
        self._composer = composer
        self._tools = tools
        self._resolve_topics = resolve_topics
        self._build_meta_emotion = build_meta_emotion

    # ─── Code tool dispatch ──────────────────────────────────────

    def deliberate_code_tools(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Top-priority code-tool dispatch: if the user asks about a .py file, act."""
        py_path = self._extract_py_path(perception.raw_text)
        if not py_path:
            return None
        lower = perception.raw_text.lower()
        if not any(word in lower for word in self._ACTION_WORDS):
            return None
        result_text = self.run_code_tools(py_path, lower)
        return Thought(
            content=result_text,
            intent="discuss_code",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.75,
            metadata={"tool_path": py_path},
        )

    def handle_code_discussion(
        self,
        perception: Perception,
        emotion: EmotionalState,
    ) -> Thought:
        """Handle code-related discussion.

        Uses the thought composer to generate responses from its
        actual knowledge about code, not hardcoded strings.
        """
        # If the user mentions a .py file with an action word, dispatch
        # a concrete tool. Without an action word, fall through to the
        # concept-network composition path — the user is *discussing*
        # the file, not asking to run a tool on it.
        py_path = self._extract_py_path(perception.raw_text)
        if py_path:
            lower = perception.raw_text.lower()
            if any(word in lower for word in self._ACTION_WORDS):
                result_text = self.run_code_tools(py_path, lower)
                return Thought(
                    content=result_text,
                    intent="discuss_code",
                    emotion=emotion.label,
                    topics=perception.topics,
                    confidence=0.75,
                    metadata={"tool_path": py_path},
                )

        # Resolve topics to filter out question structure words
        topics = self._resolve_topics(perception.topics, perception.raw_text)

        # Try to compose from what it knows about the topic
        for topic in topics:
            thought = self._composer.compose_about(topic, emotion)
            if thought and thought.confidence > 0.3:
                return Thought(
                    content=thought.content,
                    intent="discuss_code",
                    emotion=emotion.label,
                    topics=topics,
                    confidence=thought.confidence,
                )

        # Fallback: compose a reflection about code in general
        # using its concept network knowledge. Pass the real edges
        # as knowledge metadata so the generative engine composes
        # the actual words — no hardcoded poetic statements.
        code_concept = self._network.get_concept("code")
        if code_concept:
            neighbors = self._network.get_neighbors("code")
            if neighbors:
                knowledge = [
                    (relation.value.replace("_", " "), target, weight)
                    for target, relation, weight in neighbors[:2]
                ]
                return Thought(
                    content="code",
                    intent="discuss_code",
                    emotion=emotion.label,
                    topics=topics,
                    confidence=0.6,
                    metadata={"knowledge": knowledge, "topic": "code"},
                )

        # Last resort: honest acknowledgment — let the engine compose
        return Thought(
            content="code",
            intent="discuss_code",
            emotion=emotion.label,
            topics=topics,
            confidence=0.4,
            metadata={"topic": "code"},
        )

    # ─── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_py_path(text: str) -> str | None:
        """Find a .py file path in the user's message."""
        match = re.search(r"\b([\w./-]+\.py)\b", text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_symbol(text: str, path: str) -> str | None:
        """Find an optional symbol name in an analyze/impact request.

        Matches "impact of X", "analyze X in file.py", or a bare
        CamelCase / dotted identifier that isn't the file path itself.
        """
        match = re.search(
            r"(?:impact\s+(?:of|on)|analyze)\s+([A-Za-z_][\w.]*)\b",
            text,
        )
        if match and not match.group(1).endswith(".py"):
            return match.group(1)
        stem = os.path.basename(path).removesuffix(".py")
        for token in re.findall(r"\b[A-Za-z_][\w.]*\b", text):
            if token.endswith(".py") or token == stem:
                continue
            if "." in token or token[:1].isupper():
                return token
        return None

    def _resolve_project_path(self, path: str) -> str | None:
        """Resolve a file path against the project root.

        If the path exists as-is (relative to CWD or absolute), return it.
        Otherwise, search the project tree for a matching file name.
        """
        if os.path.exists(path):
            return path
        # Search the Python package tree. The root is three levels
        # above this file: .../genesis/python/genesis_cognitive/cognition/
        # → .../genesis/python/
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        target_name = os.path.basename(path)
        _skip = {".git", "target", "node_modules", "__pycache__", ".venv", "venv"}
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in _skip]
            if target_name in files:
                full = os.path.join(root, target_name)
                rel = os.path.relpath(full, os.getcwd())
                return rel
        return None

    def run_code_tools(self, path: str, action: str) -> str:
        """Run the requested code tools and compose a response."""
        outputs = []
        if "test" in action:
            resolved = self._resolve_project_path(path)
            target = resolved if resolved is not None else path
            result = self._tools.run(
                "run_pytest", target=target, project_root=".",
            )
            if result.success:
                tail = result.output.strip().split("\n")[-1]
                outputs.append(f"The tests for {path} passed. {tail}")
            else:
                outputs.append(
                    f"The tests for {path} did not pass. {result.error or result.output}"
                )
        if "analyze" in action or "impact" in action:
            resolved = self._resolve_project_path(path)
            if resolved is not None:
                result = self._tools.run(
                    "analyze_python",
                    path=resolved,
                    symbol=self._extract_symbol(action, path),
                )
                outputs.append(result.output or result.error)
            else:
                meta_emotion = self._build_meta_emotion()
                thought = Thought(
                    content=f"cannot find {path}",
                    intent="self_report",
                    emotion=meta_emotion.label,
                    confidence=0.3,
                    metadata={"file_not_found": path},
                )
                outputs.append(self._language.render(thought, meta_emotion))
        if "compile" in action or "check" in action or "read" in action or "show" in action:
            resolved = self._resolve_project_path(path)
            if resolved is not None:
                if "compile" in action or "check" in action:
                    result = self._tools.run("compile_python", path=resolved)
                    outputs.append(result.output or result.error)
                wants_read = "read" in action or "show" in action
                bare_check = "check" in action and not (
                    "compile" in action or "test" in action
                )
                if wants_read or bare_check:
                    result = self._tools.run("read_file", path=resolved, limit=300)
                    outputs.append(result.output or result.error)
            else:
                meta_emotion = self._build_meta_emotion()
                thought = Thought(
                    content=f"cannot find {path}",
                    intent="self_report",
                    emotion=meta_emotion.label,
                    confidence=0.3,
                    metadata={"file_not_found": path},
                )
                outputs.append(self._language.render(thought, meta_emotion))
        if not outputs:
            resolved = self._resolve_project_path(path)
            fallback_path = resolved if resolved is not None else path
            result = self._tools.run("read_file", path=fallback_path, limit=300)
            outputs.append(result.output or result.error)
        return " ".join(outputs)
