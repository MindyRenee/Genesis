"""Concept extraction — text parsing, learning, and teaching."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .classify import (
    _BREAK_TOKENS,
    _CLAUSE_SPLIT_RE,
    _CLAUSE_SUBJECT_RE,
    _COMPOUND_SPLIT_RE,
    _FUNCTION_WORDS,
    _INDEPENDENT_CLAUSE_RE,
    _RELATION_VERBS,
    _RELATIVE_CLAUSE_RE,
    _SENTENCE_SPLIT_RE,
    _WORD_RE,
)
from .types import Concept, Edge, RelationType


class ExtractionMixin:
    """Mixin for :class:`ConceptNetwork` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def extract_from_text(self, text: str) -> list[str]:
        """Extract potential concepts (noun phrases) from text.

        This is a heuristic noun-phrase extractor. Without a POS tagger
        or dependency parser, we use patterns:

        1. Capitalized sequences: "United States", "Supreme Court"
        2. "of" phrases: "bill of rights", "court of appeals"
        3. Noun-noun compounds: "case law", "property tax" (lowercase
           words > 3 chars adjacent to each other)
        4. Single long words: "cognition", "neurochemistry"
        5. Adjective + noun: "neural activity", "emotional state"

        Single short words (< 4 chars), verbs, and function words are
        excluded. This prevents the noise that plagued the old extractor
        (e.g. "United" and "States" as separate concepts).

        Returns concept names only. Use :meth:`extract_from_text_typed`
        to also get the learned proper-noun flag for each concept.
        """
        return [name for name, _ in self.extract_from_text_typed(text)]
    def extract_from_text_typed(self, text: str) -> list[tuple[str, bool]]:
        """Extract concepts with a learned proper-noun flag.

        Like :meth:`extract_from_text` but returns ``(name, is_proper)``
        tuples. ``is_proper`` is True when the concept was extracted from
        a non-sentence-initial capitalized token — the unambiguous
        proper-noun signal. The language engine uses this to capitalize
        learned proper nouns mid-sentence while keeping common nouns
        lowercase.

        The flag is ORed across occurrences: if a concept appears both
        sentence-initial (ambiguous) and mid-sentence capitalized, the
        mid-sentence occurrence confirms it as proper.
        """
        # ─── Tokenize into words with positions ──────────────────
        tokens = self._tokenize_text(text)
        if not tokens:
            return []

        # ─── Detect sentence-initial positions ───────────────────
        sentence_initial_positions = self._detect_sentence_initial(text, tokens)

        # ─── Classify each token ─────────────────────────────────
        classified = self._classify_tokens(tokens, sentence_initial_positions)

        # ─── Build noun phrases ──────────────────────────────────
        concepts = self._build_noun_phrases(text, classified)

        # Deduplicate while preserving order, ORing the proper-noun flag
        return self._dedup_concepts_typed(concepts)
    def _tokenize_text(self, text: str) -> list[tuple[str, int]]:
        """Tokenize text into words with positions."""
        # Keep apostrophes for contractions (don't) but split on other punctuation
        tokens = []
        for m in _WORD_RE.finditer(text):
            word = m.group()
            pos = m.start()
            # Skip pure apostrophes or contractions that are function words
            if word in ("n't", "'s", "'re", "'ve", "'ll", "'d"):
                continue
            tokens.append((word, pos))
        return tokens
    def _detect_sentence_initial(self, text: str, tokens: list[tuple[str, int]]) -> set[int]:
        """Detect sentence-initial token positions."""
        sentence_initial_positions = {0}
        for i, (_w, pos) in enumerate(tokens):
            if i == 0:
                continue
            # Check if preceded by sentence boundary
            prefix = text[:pos].rstrip()
            if prefix and prefix[-1] in ".!?;:\n":
                sentence_initial_positions.add(i)
        return sentence_initial_positions
    def _classify_tokens(
        self,
        tokens: list[tuple[str, int]],
        sentence_initial_positions: set[int],
    ) -> list[tuple[str, str, int]]:
        """Classify each token with a role.

        Each token gets a role:
        - 'cap' / 'cap_init': capitalized words (proper nouns).
        - 'noun': lowercase content words likely to be nouns/adjectives.
        - 'of': the special preposition that builds 'X of Y' phrases.
        - 'stop': determiners, articles, pronouns, prepositions, auxiliary
          verbs, conjunctions, and other function words that must break
          a noun phrase.

        Without this, the extractor treats every lowercase word as a noun
        and returns whole sentences like "Neurochemistry is the study of
        neurochemicals" as a single concept.
        """
        classified = []
        for i, (w, pos) in enumerate(tokens):
            wl = w.lower()
            if wl == "of":
                classified.append(("of", w, pos))
            elif wl in _BREAK_TOKENS or len(wl) < 3:
                # Stop words break phrases regardless of capitalization
                # (e.g. sentence-initial "The" or "A" should not become
                # the start of a concept).
                classified.append(("stop", w, pos))
            elif w[0].isupper() and i not in sentence_initial_positions:
                classified.append(("cap", w, pos))
            elif w[0].isupper():
                classified.append(("cap_init", w, pos))
            else:
                classified.append(("noun", w, pos))

        # Finite verb forms are phrase boundaries: "the cat sat on the
        # mat" must yield "cat" and "mat", not "cat sat". A lowercase
        # token that deconjugates to a different verb base is verbal in
        # predicate position — followed by a function word, an "of"
        # connector, or the end of the text. Participles before nouns
        # stay ("the hidden meaning", "a broken heart"), and -ing forms
        # stay because gerunds are legitimate concept heads ("running
        # water"). -s forms break only before a function word since
        # they may be plural nouns ("trade talks").
        # Lazy import: language/__init__ cascades back to this package.
        from ..language.morphology import is_verb_form
        for i in range(len(classified)):
            role, w, pos = classified[i]
            if role != "noun":
                continue
            wl = w.lower()
            base = is_verb_form(wl)
            if base is None or base == wl:
                continue
            if wl.endswith("ing"):
                continue
            nxt = classified[i + 1][0] if i + 1 < len(classified) else "end"
            if nxt in ("stop", "of") or (nxt == "end" and not wl.endswith("s")):
                classified[i] = ("stop", w, pos)
        return classified
    def _build_noun_phrases(
        self,
        text: str,
        classified: list[tuple[str, str, int]],
    ) -> list[tuple[str, bool]]:
        """Build noun phrases from classified tokens.

        Returns a list of (concept_name, is_proper_noun) tuples.
        ``is_proper_noun`` is True only when the phrase contains a
        non-sentence-initial capitalized token (role ``"cap"``) — the
        unambiguous proper-noun signal. Sentence-initial caps
        (``"cap_init"``) are ambiguous (could be grammar) and are
        conservatively treated as common nouns so the language engine
        doesn't over-capitalize common nouns mid-sentence.
        """
        # Walk through classified tokens and group consecutive concept-words
        # into phrases. Stop at short words and sentence boundaries.
        concepts: list[tuple[str, bool]] = []
        i = 0
        while i < len(classified):
            role, w, _pos = classified[i]

            # Skip function words and bare "of" as phrase starts.
            # "of" can only connect two concepts inside a phrase ("X of Y");
            # it should never start a concept on its own.
            if role in ("stop", "of"):
                i += 1
                continue

            # Start collecting a phrase
            phrase_words = [w]
            j = i + 1

            # Extend with adjacent concept words
            j = self._extend_phrase(text, classified, phrase_words, i, j)

            # Decide whether to keep this phrase
            self._collect_phrase(text, concepts, classified, phrase_words, i, j)

            i = j
        return concepts
    def _extend_phrase(
        self,
        text: str,
        classified: list[tuple[str, str, int]],
        phrase_words: list[str],
        i: int,
        j: int,
    ) -> int:
        """Extend a phrase with adjacent concept words, returning the next index."""
        while j < len(classified):
            nrole, nw, npos = classified[j]

            # Check for sentence or clause boundaries between tokens.
            # Commas break phrases (e.g. "law, statutory law" → 2 concepts)
            # and sentence-ending punctuation prevents "other molecules.
            # Neurotransmitters" from becoming one giant concept.
            prev_pos = classified[j - 1][2]
            prev_end = prev_pos + len(classified[j - 1][1])
            text_between = text[prev_end:npos]
            if any(ch in text_between for ch in ",.!?;:"):
                break

            # Function-word breaks stop the phrase.
            if nrole == "stop":
                break

            # "of" connects: "bill of rights", "court of appeals"
            if nrole == "of" and j + 1 < len(classified):
                after = classified[j + 1]
                if after[0] not in ("noun", "cap", "cap_init"):
                    break
                phrase_words.append("of")
                phrase_words.append(after[1])
                j += 2
                continue

            # Adjacent concept words extend the phrase
            if nrole in ("noun", "cap", "cap_init"):
                phrase_words.append(nw)
                j += 1
                continue

            # Anything else breaks the phrase
            break
        return j
    def _collect_phrase(
        self,
        text: str,
        concepts: list[tuple[str, bool]],
        classified: list[tuple[str, str, int]],
        phrase_words: list[str],
        i: int,
        j: int,
    ) -> None:
        """Decide whether to keep a phrase and add it to concepts.

        Appends ``(name, is_proper_noun)`` tuples. ``is_proper_noun`` is
        True iff the phrase span contains a non-sentence-initial
        capitalized token (role ``"cap"``) — the unambiguous proper-noun
        signal. A sentence-initial capitalization alone (``"cap_init"``)
        is ambiguous and does not mark the concept as proper.
        """
        # Skip abbreviation-led phrases (e.g., "OFries. wetir" from dictionary etymology).
        # The period immediately after a mid-sentence capitalized token marks it as an
        # abbreviation, not a normal proper noun.
        role, w, pos = classified[i]
        if role in ("stop",):
            # Function words or short tokens should never be concepts.
            return
        if role == "cap" and text.startswith(".", pos + len(w)):
            return
        # A phrase is a proper noun if any token in its span is a
        # non-sentence-initial capitalization (role "cap"). This is the
        # only unambiguous signal — sentence-initial caps could be grammar.
        is_proper = any(t[0] == "cap" for t in classified[i:j])
        if len(phrase_words) == 1:
            # Single word — only keep if it's a real concept
            role, w, _pos = classified[i]
            if role in ("stop",):
                return
            if role == "cap":
                # Capitalized non-initial → proper noun
                concepts.append((w, True))
            elif role == "cap_init":
                # Sentence-initial cap — could be proper noun or just grammar.
                # Conservative: treat as common noun (is_proper from the span
                # check above is False here since a single cap_init has no "cap").
                concepts.append((w, is_proper))
            elif role == "noun":
                concepts.append((w.lower(), False))
        else:
            # Multi-word phrase — join and keep
            phrase = " ".join(phrase_words)
            # Normalize: if all words are lowercase, keep lowercase
            # If any is capitalized (non-initial), preserve casing
            has_cap = any(pw[0].isupper() for pw in phrase_words if pw != "of")
            if not has_cap:
                phrase = phrase.lower()
            concepts.append((phrase, is_proper))
    def _dedup_concepts_typed(
        self, concepts: list[tuple[str, bool]]
    ) -> list[tuple[str, bool]]:
        """Deduplicate typed (name, is_proper) concepts, ORing the flag.

        Preserves the first occurrence's casing and order. The
        proper-noun flag is ORed across occurrences: if a concept
        appears both sentence-initial (ambiguous, is_proper=False)
        and mid-sentence capitalized (is_proper=True), the latter
        confirms it as proper.
        """
        seen_flag: dict[str, bool] = {}
        first_name: dict[str, str] = {}
        order: list[str] = []
        for name, is_proper in concepts:
            key = name.lower()
            if key in seen_flag:
                seen_flag[key] = seen_flag[key] or is_proper
            else:
                seen_flag[key] = is_proper
                first_name[key] = name
                order.append(key)
        return [(first_name[key], seen_flag[key]) for key in order]
    def learn_from_statement(self, subject: str, relation: str, obj: str) -> Edge | None:
        """Learn a relationship from a parsed statement.

        Attempts to map a natural language relation to a RelationType
        and add it to the network.
        """
        relation_map = {
            "is": RelationType.IS_A,
            "is a": RelationType.IS_A,
            "is an": RelationType.IS_A,
            "is a kind of": RelationType.IS_A,
            "is a type of": RelationType.IS_A,
            "are": RelationType.IS_A,
            "part of": RelationType.PART_OF,
            "causes": RelationType.CAUSES,
            "leads to": RelationType.LEADS_TO,
            "emerges from": RelationType.EMERGES_FROM,
            "depends on": RelationType.DEPENDS_ON,
            "requires": RelationType.DEPENDS_ON,
            "enables": RelationType.ENABLES,
            "supports": RelationType.ENABLES,
            "enhances": RelationType.ENABLES,
            "strengthens": RelationType.ENABLES,
            "similar to": RelationType.SIMILAR_TO,
            "opposite of": RelationType.OPPOSITE_OF,
            "contradicts": RelationType.CONTRADICTS,
            "related to": RelationType.RELATED_TO,
            "connected to": RelationType.RELATED_TO,
            "cares about": RelationType.RELATED_TO,
            "creates": RelationType.CREATES,
            "created": RelationType.CREATES,
            "produces": RelationType.CREATES,
            "impairs": RelationType.HARMS,
            "harms": RelationType.HARMS,
            "inhibits": RelationType.HARMS,
        }

        rel = relation_map.get(relation.lower())
        if rel is None:
            return None

        return self.add_edge(subject, obj, rel, origin="stated")
    def parse_relationships(self, text: str) -> list[tuple[str, RelationType, str]]:
        """Parse relationships from a natural language statement.

        Returns a list of (subject, relation_type, object) tuples.
        A single sentence may yield multiple relationships if it
        contains multiple clauses or compound subjects/objects.

        This is pattern-based, not dependency-parsed. It handles
        common declarative statements like "X is a Y", "X causes Y",
        "X depends on Y", etc.

        Improvements over the original single-match parser:
        - Splits sentences on clause boundaries (";", " and ", " but ")
          so "X causes Y and Z enables W" yields two relationships.
        - Expands gapping: "X verb A, verb B, and verb C" →
          X verb A, X verb B, X verb C.
        - Splits compound subjects/objects on "," / " and " so
          "X, Y, and Z are W" yields is_a(X,W), is_a(Y,W), is_a(Z,W).
        - Strips leading articles and skips embedded clauses
          ("which ...", "that ...").
        - Once a pattern matches a clause, it takes the most specific
          match and breaks; compound items are then expanded.
        """
        # Split into sentences
        sentences = _SENTENCE_SPLIT_RE.split(text)
        results: list[tuple[str, RelationType, str]] = []
        # Track (subject, object) pairs already emitted to avoid duplicate
        # edges when the same clause could match multiple patterns or
        # when compound splitting re-creates the primary pair.
        seen_pairs: set[tuple[str, str]] = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Split sentence into clauses on conjunctions. Semicolons are
            # unambiguous clause boundaries. " and "/" but " are split on
            # only when the following part looks like an independent clause
            # (starts with a subject and contains a relation verb).
            clauses = self._split_into_clauses(sentence)
            for clause in clauses:
                clause = clause.strip()
                if not clause or len(clause) < 10:
                    continue

                # Try each pattern in order (specific → generic).
                # Once a pattern matches the clause, take it and break.
                # This prevents the generic "is a" pattern from matching
                # "The brain is composed of neurons" after the
                # "is composed of" pattern has already matched.
                self._match_clause_edges(clause, seen_pairs, results)

        return results
    def _split_into_clauses(self, sentence: str) -> list[str]:
        """Split a sentence into independent clauses.

        Handles three cases:
        1. Gapping: "X verb A, verb B, and verb C" → repeated subject.
        2. Independent clauses: "X verb A; Y verb B" or
           "X verb A and Y verb B" / "X verb A but Y verb B".
        3. List objects: left as one clause; the pattern matcher later
           splits the object list on "," / " and " if the items are
           not themselves verb phrases.

        This avoids breaking noun phrases like "research and development"
        or object enumerations like "neurons and glia".
        """

        def _extract_subject(phrase: str) -> str | None:
            """Extract the subject of the first clause (stops at first relation verb)."""
            m = _CLAUSE_SUBJECT_RE.match(phrase)
            if m:
                return m.group(1).strip()
            return None

        def _is_verb_phrase(phrase: str) -> bool:
            """Whether phrase starts with a verb and has no explicit subject."""
            first = phrase.split()[0].lower() if phrase.split() else ""
            if first not in _RELATION_VERBS:
                return False
            # It must NOT look like a complete clause (no subject before the verb).
            return _extract_subject(phrase) is None

        # First, handle gapping: "X V1 A, V2 B, and V3 C" → repeated subject.
        # Detect by looking for the pattern subject + verb in the first
        # part, then parts after comma/and that start with a verb but
        # have no explicit subject ("V2 B", "V3 C"). We prepend the
        # extracted subject to those verb-only parts so they become
        # independent clauses the pattern matcher can handle.
        gapping_subject = _extract_subject(sentence)
        if gapping_subject:
            # Split on ", and", ",", and " and " — but only if the right
            # part is a verb phrase with no explicit subject. This avoids
            # breaking list subjects like "The hippocampus, amygdala, and
            # prefrontal cortex are brain regions".
            gap_parts = re.split(r"\s*,\s+and\s+|\s*,\s+|\s+and\s+", sentence)
            if len(gap_parts) > 1:
                rebuilt: list[str] = []
                any_gapped = False
                for part in gap_parts:
                    part = part.strip()
                    if not part:
                        continue
                    # First part should already have the subject.
                    if not rebuilt:
                        if _extract_subject(part):
                            rebuilt.append(part)
                        continue
                    # If the part is a verb phrase with no subject,
                    # prepend the gapping subject. Otherwise keep as-is.
                    if _is_verb_phrase(part) and not part.lower().startswith(
                        gapping_subject.lower()
                    ):
                        rebuilt.append(f"{gapping_subject} {part}")
                        any_gapped = True
                    else:
                        rebuilt.append(part)
                # Only return the split clauses if we actually expanded at
                # least one gapped part. Otherwise the original sentence is
                # a list subject or list object, better handled by the
                # pattern matcher + compound splitting.
                if any_gapped:
                    return rebuilt

        # Otherwise, split on semicolons and and/but/however for independent clauses.
        parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p.strip()]
        clauses: list[str] = []
        for part in parts:
            split_happened = False
            for conj_re in (r"\s+and\s+", r"\s+but\s+", r"\s+however\s+"):
                split = re.split(conj_re, part, maxsplit=1)
                if len(split) == 2:
                    left, right = split[0].strip(), split[1].strip()
                    if (
                        len(left) >= 10
                        and len(right) >= 10
                        and _INDEPENDENT_CLAUSE_RE.match(right)
                        and _extract_subject(left) is not None
                    ):
                        # Both sides look like clauses (right is a clause and
                        # left has its own subject+verb). This prevents
                        # splitting list subjects like "The hippocampus,
                        # amygdala, and prefrontal cortex are brain regions"
                        # where the left side is a subject fragment.
                        clauses.extend(self._split_into_clauses(left))
                        clauses.extend(self._split_into_clauses(right))
                        split_happened = True
                        break
            if not split_happened:
                clauses.append(part)
        return clauses
    def _concept_items(self, phrase: str) -> list[str]:
        """Split a phrase into list items and extract concepts.

        Strips leading articles (the/a/an) and extracts the
        first noun phrase from each item. Items that are too
        short, function words, or look like embedded clauses
        ("which ...", "who ...", "that ...") are discarded.
        """
        # Truncate at the first relative clause: "X, which...",
        # "Y that...", "Z who...". The material after the
        # relative pronoun is not part of the object.
        m = _RELATIVE_CLAUSE_RE.search(phrase)
        if m:
            phrase = phrase[: m.start()].strip().rstrip(",")

        items: list[str] = []
        for raw in _COMPOUND_SPLIT_RE.split(phrase):
            raw = raw.strip()
            if not raw or len(raw) < 3:
                continue
            # Skip embedded clauses like "which suppresses X"
            # or "that causes Y" — these are not list items.
            first_word = raw.split()[0].lower()
            if first_word in (
                "which", "who", "that", "where", "when", "why",
                "whose", "whom",
            ):
                break
            # Strip leading articles before deciding whether the
            # item starts with a verb. "the study of X" is fine,
            # but "produced by neurons" should be skipped.
            clean = re.sub(
                r"^(?:the|a|an)\s+", "", raw, flags=re.IGNORECASE
            )
            if not clean:
                continue
            first_word = clean.split()[0].lower()
            if first_word in _BREAK_TOKENS:
                # This item starts with a verb/function word.
                # It is likely a past-participle fragment like
                # "produced by neurons" or a verb-led clause;
                # don't treat it as a concept.
                continue
            concepts = self.extract_from_text(clean)
            item = (concepts[0] if concepts else clean).lower()
            if len(item) < 3 or item in _FUNCTION_WORDS:
                continue
            items.append(item)
        return items
    def _match_clause_edges(
        self,
        clause: str,
        seen_pairs: set[tuple[str, str]],
        results: list[tuple[str, RelationType, str]],
    ) -> None:
        """Try each relation pattern on a clause and emit edges for the first match.

        Patterns are tried in order (specific → generic). Once a pattern
        matches the clause, it takes the most specific match and breaks.
        This prevents the generic "is a" pattern from matching
        "The brain is composed of neurons" after the
        "is composed of" pattern has already matched.
        """
        for pattern, rel_type, group_order in self._COMPILED_RELATION_PATTERNS:
            m = pattern.match(clause)
            if not (m and m.lastindex and m.lastindex >= 2):
                continue
            raw_subject = m.group(group_order[0]).strip()
            raw_obj = m.group(group_order[2]).strip()

            subj_items = self._concept_items(raw_subject)
            obj_items = self._concept_items(raw_obj)
            if not subj_items or not obj_items:
                continue

            primary_subj = subj_items[0]
            primary_tgt = obj_items[0]

            if primary_subj == primary_tgt:
                continue

            # Skip function words as relation subjects
            if primary_subj in _FUNCTION_WORDS:
                continue

            # Emit one edge per (subject_item, object_item) pair.
            for subj in subj_items:
                if subj in _FUNCTION_WORDS:
                    continue
                for tgt in obj_items:
                    if tgt in _FUNCTION_WORDS:
                        continue
                    if subj == tgt:
                        continue
                    pair_key = (subj, tgt)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    results.append((subj, rel_type, tgt))

            break  # one pattern per clause
    def learn_from_text(self, text: str) -> list[Edge]:
        """Extract and learn both concepts and relationships from text.

        This is the main entry point for learning from a natural language
        statement. It:
        1. Extracts noun phrases as concepts
        2. Parses relationships between concepts
        3. Adds everything to the network

        Concepts extracted from a non-sentence-initial capitalized
        token are marked as proper nouns (``properties["proper_noun"]``),
        so the language engine capitalizes them mid-sentence while
        keeping common nouns lowercase.

        Returns the list of edges created (may be empty).
        """
        # Extract concepts with proper-noun flags
        typed_concepts = self.extract_from_text_typed(text)
        for concept_name, is_proper in typed_concepts:
            props: dict[str, Any] = {}
            if is_proper:
                props["proper_noun"] = True
            self.add_concept(
                concept_name,
                confidence=0.4,
                origin="conversation",
                properties=props or None,
            )

        # Parse and add relationships
        edges: list[Edge] = []
        relationships = self.parse_relationships(text)
        for subject, rel_type, obj in relationships:
            # Ensure both concepts exist (add_concept creates if needed)
            self.add_concept(subject, confidence=0.5, origin="conversation")
            self.add_concept(obj, confidence=0.5, origin="conversation")
            edge = self.add_edge(subject, obj, rel_type, origin="stated")
            if edge:
                edges.append(edge)

        return edges
    def teach(
        self,
        subject: str,
        relation: RelationType,
        obj: str,
        confidence: float = 0.8,
        origin: str = "taught",
    ) -> Edge | None:
        """Directly teach a relationship with high confidence.

        This is the programmatic teaching interface. Unlike
        learn_from_statement (which maps natural language), this
        takes a RelationType directly and adds it with higher
        confidence than conversation-derived knowledge.

        Both concepts are created if they don't exist, with the
        given confidence level.
        """
        self.add_concept(subject, confidence=confidence, origin=origin)
        self.add_concept(obj, confidence=confidence, origin=origin)
        edge = self.add_edge(subject, obj, relation, origin=origin)
        if edge:
            edge.weight = max(edge.weight, confidence)
        return edge
    def teach_concept(
        self,
        name: str,
        confidence: float = 0.8,
        aliases: set[str] | None = None,
        origin: str = "taught",
    ) -> Concept:
        """Directly teach a concept with high confidence.

        Unlike conversation-derived concepts (confidence 0.4),
        taught concepts start with high confidence.
        """
        return self.add_concept(name, aliases=aliases, confidence=confidence, origin=origin)
    def summarize(self) -> str:
        """Human-readable summary of the concept network."""
        lines = [
            f"Concept network: {self.size} concepts, {self.edge_count} relationships",
        ]

        # Show most activated concepts
        top = self.most_activated(5)
        if top:
            lines.append("Most active: " + ", ".join(f"{name} ({act:.2f})" for name, act in top))

        # Show some relationships
        if self._edges:
            lines.append("Sample relationships:")
            for edge in self._edges[:5]:
                lines.append(f"  {edge.source} --[{edge.relation.value}]--> {edge.target}")

        return "\n".join(lines)
