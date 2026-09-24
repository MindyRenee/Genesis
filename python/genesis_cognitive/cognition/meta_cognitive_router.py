"""Metacognitive router — decides which subsystem should handle input.

Before Genesis runs the full cognition pipeline, the router does a quick
first pass over the user's input and classifies the kind of speech act.
If it can confidently identify the route, the main pipeline can hand the
question to a specialist (identity, user model, code, etc.) instead of
letting the big monolithic think() produce a noisy answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(slots=True)
class Route:
    """A routing decision for a user input."""

    route: str  # e.g. "identity", "user", "code", "command", "general"
    confidence: float  # 0..1
    sub_kind: str = ""  # fine-grained tag, e.g. "name", "preference"


class MetaCognitiveRouter:
    """Fast, rule-based metacognitive routing for user input."""

    # Maps the verb captured from "what do i <verb>" to the preference
    # category used by UserProfile.preferences.
    _PREFERENCE_VERB_MAP: ClassVar[dict[str, str]] = {
        "like": "likes",
        "love": "likes",
        "enjoy": "likes",
        "dislike": "dislikes",
        "hate": "dislikes",
        "want": "wants",
        "need": "wants",
        "believe": "believes",
        "think": "believes",
    }

    _PATTERNS: ClassVar[list[tuple[str, str, re.Pattern]]] = [
        # Identity / self-inquiry
        ("identity", "name", re.compile(r"\bwhat('?s| is) your name\b")),
        ("identity", "name", re.compile(r"\byour name\b")),
        ("identity", "who", re.compile(r"\bwho are you\b")),
        ("identity", "self", re.compile(r"\b(tell me about yourself|describe yourself)\b")),
        ("identity", "self", re.compile(r"\bwhat are you\b(?:[.?!]|$)")),
        # Self-reflective questions — these are about HER state, not
        # factual lookups. Must come before the factual route.
        ("identity", "curious", re.compile(r"\bwhat are you curious about\b")),
        ("identity", "curious", re.compile(
            r"\bwhat (?:questions?|topics?) (?:are|do) you "
            r"(?:feel )?(?:most )?(?:curious about|wonder about)\b"
        )),
        ("identity", "curious", re.compile(r"\bwhat (?:do|are) you (?:wonder|curious)\b")),
        ("identity", "want_learn", re.compile(
            r"\bwhat do you want to (?:learn|know|understand)\b"
        )),
        ("identity", "want_learn", re.compile(
            r"\bwhat (?:are you|do you) (?:trying to|want to) learn\b"
        )),
        ("identity", "learned", re.compile(r"\bwhat have you (?:been )?learn(?:ed|ing)\b")),
        ("identity", "learned", re.compile(r"\bwhat did you learn\b")),
        ("identity", "learned", re.compile(
            r"\bwhat (?:are you|have you been) "
            r"(?:learning|studying|reading|exploring)(?: about)?\b"
        )),
        # Personal preference questions — "what's your favorite X?",
        # "do you like X?", "what do you enjoy?". These are about HER
        # preferences, not factual lookups about a concept. Must come
        # before the factual route, otherwise "what's your favorite
        # color" matches the factual pattern and tries to look up
        # "your favorite color" as a concept.
        ("preference", "favorite", re.compile(r"\bwhat('?s| is) your favorite (.+?)(?:[.?!]|$)")),
        ("preference", "hobby", re.compile(
            r"\bwhat do you (?:like to do|do for fun|enjoy doing)\b"
        )),
        ("preference", "like", re.compile(r"\bdo you (?:like|love|enjoy|prefer) (.+?)(?:[.?!]|$)")),
        # User model (questions only — statements are learned by the
        # theory-of-mind / user profile pipeline)
        ("user", "name", re.compile(r"\bwhat('?s| is) my name\b")),
        ("user", "name", re.compile(r"\bdo you know my name\b")),
        ("user", "name", re.compile(r"\bdo you remember my name\b")),
        ("user", "name", re.compile(r"\bwho am i\b")),
        (
            "user",
            "preference",
            re.compile(r"\bwhat do i (like|love|enjoy|dislike|hate|want|need|believe|think)\b"),
        ),
        (
            "user",
            "preference",
            re.compile(r"\bwhat am i interested in\b"),
        ),
        ("user", "summary", re.compile(r"\bwhat do you know about me\b")),
        ("user", "summary", re.compile(r"\btell me about myself\b")),
        # Code / self-knowledge
        ("code", "bugs", re.compile(r"\b(bugs|issues|problems) in your code\b")),
        ("code", "bugs", re.compile(r"\bdo you have (bugs|issues|problems)\b")),
        ("code", "code", re.compile(r"\b(your code|source code|how are you implemented)\b")),
        ("code", "improve", re.compile(r"\bimprove yourself\b")),
        # Self-improvement proposals / experiments / growth.
        # These must come before the factual route, otherwise
        # "what are your proposals" matches the factual pattern
        # and tries to look up "your proposals" as a concept.
        ("self_improvement", "proposals", re.compile(r"\b(your|any) proposals\b")),
        ("self_improvement", "proposals", re.compile(r"\bwhat (?:have you|did you) propose\b")),
        ("self_improvement", "proposals", re.compile(r"\bshow me your proposals\b")),
        ("self_improvement", "experiments", re.compile(r"\b(your|any) experiments\b")),
        (
            "self_improvement",
            "experiments",
            re.compile(r"\bwhat (?:have you|did you) experiment(?:ed|ing)\b"),
        ),
        (
            "self_improvement",
            "growth",
            re.compile(r"\b(your|how you) (?:have )?grown\b"),
        ),
        ("self_improvement", "growth", re.compile(r"\bhow (?:have|did) you grow\b")),
        ("self_improvement", "growth", re.compile(r"\btell me about your growth\b")),
        ("self_improvement", "growth", re.compile(r"\byour growth (?:narrative|journey|story)\b")),
        # Emotional / well-being
        ("emotion", "feel", re.compile(r"\bhow (are|do) you feel\b")),
        # Corrections / feedback on its claims.
        (
            "correction",
            "negated_cause",
            re.compile(
                r"\b(.+?)\s+(?:does not|doesn't|do not|don't|did not|didn't)\s+"
                r"(?:cause|lead to|create|enable|make)\s+(.+?)(?:[.?!]|$)"
            ),
        ),
        (
            "correction",
            "prevents",
            re.compile(r"\b(.+?)\s+(?:prevents|stops|blocks)\s+(.+?)(?:[.?!]|$)"),
        ),
        (
            "correction",
            "negated_is_a",
            re.compile(r"\b(.+?)\s+is not\s+(?:a|an)\s+(.+?)\b"),
        ),
        (
            "correction",
            "unrelated",
            re.compile(
                r"\b(.+?)\s+and\s+(.+?)\s+(?:are not|aren't|not related|unrelated|not connected)\b"
            ),
        ),
        (
            "correction",
            "no",
            re.compile(
                r"^(?:no|nope|wrong|not quite|not really|not exactly|incorrect)[.!?]*$"
            ),
        ),
        (
            "correction",
            "yes",
            re.compile(r"^(?:yes|yeah|yep|right|correct|exactly)[.!?]*$"),
        ),
        # Memory / what have we discussed.
        (
            "memory",
            "what_told",
            re.compile(r"\bwhat (?:did|have) (?:i|you) (?:tell|taught|say|say about)\b"),
        ),
        (
            "memory",
            "what_talked",
            re.compile(r"\bwhat (?:have we|did we) (?:talk|discuss) (?:about)?\b"),
        ),
        (
            "memory",
            "what_learned",
            re.compile(r"\bwhat (?:did you|have you) (?:been )?learn(?:ed|ing)\b"),
        ),
        ("memory", "what_remember", re.compile(r"\bwhat do you remember\b")),
        (
            "memory",
            "what_wrong",
            re.compile(r"\bwhat (?:was|were) (?:i|you) wrong about\b"),
        ),
        # Mission / purpose.
        ("mission", "mission", re.compile(r"\bwhat is your (?:mission|purpose)\b")),
        ("mission", "main_goal", re.compile(r"\bwhat is your main goal\b")),
        # Prospective / planning questions. The capture group is the goal.
        (
            "planning",
            "how",
            re.compile(
                r"\bhow can i (?:make|get|cause|increase|decrease) "
                r"(.+?)(?:[.?!]|$)"
            ),
        ),
        ("planning", "do", re.compile(r"\bwhat should i do to (.+?)(?:[.?!]|$)")),
        ("planning", "do", re.compile(r"\bhow (?:do|to) i (.+?)(?:[.?!]|$)")),
        ("planning", "do", re.compile(r"\bhow to (.+?)(?:[.?!]|$)")),
        ("planning", "cause", re.compile(r"\bwhat would cause (.+?)(?:[.?!]|$)")),
        # Self-model / goal questions.
        (
            "goal",
            "goals",
            re.compile(
                r"\bwhat are you (?:trying to learn|thinking about|working on)\b"
            ),
        ),
        ("goal", "goal", re.compile(r"\bwhat is your (?:goal|current goal)\b")),
        ("goal", "why_ask", re.compile(r"\bwhy (?:are you|do you) ask(?:ing)?\b")),
        # Explanatory / "why" questions. The capture group is the query.
        ("explanation", "why_cause", re.compile(r"\bwhy does (.+?) cause (.+?)(?:[.?!]|$)")),
        ("explanation", "why", re.compile(r"\bwhy(?: is| does)?\s+(.+?)(?:[.?!]|$)")),
        ("explanation", "why", re.compile(r"\bhow come (.+?)(?:[.?!]|$)")),
        ("explanation", "why", re.compile(r"\bwhy (?:is|are) (.+?)(?:[.?!]|$)")),
        # Comparison questions — "difference between X and Y", "compare X and Y".
        # sub_kind encodes both concepts as "X|||Y" for the handler to split.
        (
            "comparison",
            "difference",
            re.compile(
                r"\bwhat(?:'s| is) (?:the )?difference between (.+?) and (.+?)(?:[.?!]|$)"
            ),
        ),
        (
            "comparison",
            "difference",
            re.compile(r"\bdifference between (.+?) and (.+?)(?:[.?!]|$)"),
        ),
        (
            "comparison",
            "compare",
            re.compile(r"\bcompare (.+?) (?:with|and|to) (.+?)(?:[.?!]|$)"),
        ),
        (
            "comparison",
            "compare",
            re.compile(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[.?!]|$)"),
        ),
        # List / parts questions — "what are the X of Y", "what X are in Y",
        # "what X does Y have". sub_kind encodes the container concept; the
        # handler finds all parts/types/instances of it.
        # Only match plural "are" + quantity words to avoid catching
        # "what is the best interest of the child" (singular = factual).
        (
            "parts",
            "of",
            re.compile(
                r"\bwhat are (?:the )?"
                r"(?:parts|branches|types|kinds|members|components|sections|"
                r"categories|divisions|elements|aspects|features|classes|"
                r"subdivisions|departments|units|courts|levels|steps|stages)"
                r" of (.+?)(?:[.?!]|$)"
            ),
        ),
        (
            "parts",
            "in",
            re.compile(r"\bwhat (.+?) (?:are|is) in (.+?)(?:[.?!]|$)"),
        ),
        (
            "parts",
            "have",
            re.compile(r"\bwhat (.+?) does (.+?) have(?:[.?!]|$)"),
        ),
        (
            "parts",
            "include",
            re.compile(r"\bwhat (.+?) (?:does|do) (.+?) include(?:[.?!]|$)"),
        ),
        # Consumes questions — "what do glips eat?" "what does a human
        # drink?" sub_kind encodes the consumer concept; the handler finds
        # what it consumes.
        (
            "consumes",
            "eat",
            re.compile(r"\bwhat (?:do|does) (.+?) (?:eat|drink|consume)(?:[.?!]|$)"),
        ),
        # Needs/depends questions — "what does X need?" "what does X
        # depend on?" These are different from wants/desires because they
        # query dependencies, not goals.
        (
            "consumes",
            "need",
            re.compile(r"\bwhat (?:do|does) (.+?) (?:need|require)(?:[.?!]|$)"),
        ),
        # Factual / definition questions. The capture group is the target concept.
        ("factual", "definition", re.compile(r"\bwhat(?:'s| is) (.+?)(?:[.?!]|$)")),
        ("factual", "definition", re.compile(r"\bwhat are (.+?)(?:[.?!]|$)")),
        ("factual", "definition", re.compile(r"\btell me about (.+?)(?:[.?!]|$)")),
        ("factual", "definition", re.compile(r"\bwhat do you know about (.+?)(?:[.?!]|$)")),
        ("factual", "definition", re.compile(r"\bdescribe (.+?)(?:[.?!]|$)")),
        # Counterfactual / "what if" questions. The capture group is the
        # target plus a change (e.g., "entropy increases").
        ("counterfactual", "what_if", re.compile(r"\bwhat if (.+?)(?:[.?!]|$)")),
        ("counterfactual", "what_if", re.compile(r"\bwhat happens if (.+?)(?:[.?!]|$)")),
        ("counterfactual", "what_if", re.compile(r"\bwhat would happen if (.+?)(?:[.?!]|$)")),
        ("counterfactual", "what_if", re.compile(r"\bwhat happens when (.+?)(?:[.?!]|$)")),
        # Commands (slash commands handled separately, but catch natural ones)
        ("command", "sleep", re.compile(r"\bgo to sleep\b")),
        ("command", "wake", re.compile(r"\bwake up\b")),
    ]

    # Imperative verbs that indicate the user is asking Genesis to do
    # something with code, even if the sentence ends with a period.
    _CODE_REQUEST_STARTERS: ClassVar[frozenset[str]] = frozenset(
        {
            "tell", "show", "give", "explain", "describe", "help",
            "improve", "fix", "update", "change", "check", "review",
        }
    )

    _FAREWELL_RE: ClassVar[re.Pattern] = re.compile(
        r"\b(goodbye|bye|see you|until next time|farewell|"
        r"talk to you later|i'?m leaving|good night)\b",
        re.IGNORECASE,
    )

    def _is_code_request(self, user_input: str) -> bool:
        """A code route should only fire for questions or commands.

        Mere statements that happen to mention "your code" (e.g. "we are
        fixing your code") should go through the normal pipeline so Genesis
        can learn from them instead of always offering a code menu.
        """
        lower = user_input.lower().strip()
        if lower.endswith("?"):
            return True
        first_word = lower.split()[0] if lower.split() else ""
        return first_word in self._CODE_REQUEST_STARTERS

    def decide(self, user_input: str, perception) -> Route:
        """Return the most confident route for the input."""
        lower = user_input.lower().strip()

        # Code tool requests (e.g. "check utils.rs") should
        # go to the code tools layer in deliberation, not be intercepted by
        # the metacognitive router. If the input contains a file path and
        # an action word, bypass routing entirely.
        if re.search(r"\b[\w./-]+\.(py|rs)\b", user_input) and any(
            w in lower for w in ("read", "show", "check", "compile", "test", "pytest")
        ):
            return Route(route="general", confidence=0.0, sub_kind="")

        # Farewells should never be hijacked by topic routers.
        intent = getattr(perception, "intent", None)
        intent_value: str = (
            intent.value if intent is not None and hasattr(intent, "value") else str(intent)
        )
        if intent_value == "farewell" or self._FAREWELL_RE.search(lower):
            return Route(route="farewell", confidence=0.9, sub_kind="")

        direct_request = lower.startswith(
            ("please", "could you", "would you", "can you")
        )
        if intent_value == "request" and not direct_request:
            return Route(route="indirect_request", confidence=0.9, sub_kind="")

        named_entities = re.findall(r"(?<![.!?]\s)\b[A-Z][a-z]+\b", user_input)
        has_reference = re.search(r"\b(?:he|she|they|him|her|them|it)\b", lower)
        has_followup_question = re.search(r"[.!?]\s+(?:who|which|what)\b", lower)
        if len(set(named_entities)) >= 2 and has_reference and has_followup_question:
            return Route(route="reference_ambiguity", confidence=0.9, sub_kind="")

        for route, sub_kind, pattern in self._PATTERNS:
            match = pattern.search(lower)
            if match:
                if route == "code" and not self._is_code_request(user_input):
                    # Mentioning code in a statement is not a request for
                    # the code specialist; let normal learning/responding
                    # handle it.
                    continue
                if route == "comparison" and match.groups() and match.group(2):
                    a = match.group(1).strip().strip("?").strip(".").strip("!")
                    b = match.group(2).strip().strip("?").strip(".").strip("!")
                    if a and b:
                        return Route(route=route, confidence=0.9, sub_kind=f"{a}|||{b}")
                if route == "parts" and match.groups():
                    # For "of" patterns, group(1) is the container.
                    # For "in/have/include" patterns, group(2) is the container.
                    if sub_kind == "of":
                        container = match.group(1).strip().strip("?").strip(".").strip("!")
                    else:
                        container = match.group(2).strip().strip("?").strip(".").strip("!")
                    if container:
                        return Route(route=route, confidence=0.9, sub_kind=container)
                if route == "consumes" and match.groups() and match.group(1):
                    target = match.group(1).strip().strip("?").strip(".").strip("!")
                    if target:
                        return Route(route=route, confidence=0.9, sub_kind=target)
                # Personal preference questions: extract the topic the
                # user is asking about (e.g., "color" from "what's your
                # favorite color"). The sub_kind encodes both the
                # question type and the topic as "type:topic".
                if route == "preference" and match.groups():
                    if sub_kind == "favorite":
                        topic = match.group(2).strip().strip("?").strip(".").strip("!")
                    else:
                        topic = match.group(1).strip().strip("?").strip(".").strip("!")
                    if topic:
                        return Route(
                            route=route, confidence=0.9,
                            sub_kind=f"{sub_kind}:{topic}",
                        )
                    return Route(route=route, confidence=0.9, sub_kind=sub_kind)
                if (
                    route in ("factual", "counterfactual", "explanation", "planning")
                    and match.groups()
                ):
                    target = match.group(1).strip().strip("?").strip(".").strip("!")
                    if target:
                        return Route(route=route, confidence=0.9, sub_kind=target)
                # User preference questions: map the captured verb to a
                # preference category so _route_user can extract only the
                # relevant category instead of dumping the entire profile.
                if route == "user" and sub_kind == "preference" and match.groups():
                    verb = match.group(1).strip().lower()
                    pref_category = self._PREFERENCE_VERB_MAP.get(verb, "likes")
                    return Route(
                        route=route, confidence=0.9, sub_kind=f"preference:{pref_category}"
                    )
                return Route(route=route, confidence=0.9, sub_kind=sub_kind)
        # Fallback: use perception intent if it is already clearly one of the
        # self-inquiry / user intents we have wired in perception.py.
        if intent is not None:
            if intent_value == "self_inquiry":
                return Route(route="identity", confidence=0.7, sub_kind="")
            if intent_value == "question":
                return Route(route="general", confidence=0.5, sub_kind="")
        return Route(route="general", confidence=0.5, sub_kind="")
