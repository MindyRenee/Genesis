"""Induction — abstraction from paired examples.

════════════════════════════════════════════════════════════════════════
WHAT THIS IS
════════════════════════════════════════════════════════════════════════

Given *examples* — pairs of ``(before, after)`` patterns — infer the
*rule* that produced each consequent from its antecedent, so the rule
can be applied to a novel input. Six demonstrations of ``glyph →
glyph`` become "translate by substitution"; two demonstrations of
``board → rotated board`` become "rotate 90°". The capacity this adds:
not "what does this action do" (the competence layer's effect models)
but "what single operation do all these exemplars share".

════════════════════════════════════════════════════════════════════════
ANATOMY
════════════════════════════════════════════════════════════════════════

This module is the frontopolar end of the prefrontal abstraction
gradient — the machinery lesion studies and fMRI localize to
rostrolateral PFC (lateral area 10) and mid-dorsolateral PFC
(areas 9/46):

    - **Relational integration (RLPFC)**. Christoff et al. (2001)
      showed RLPFC is recruited specifically when *two or more
      relations must be considered simultaneously* — 2-relational
      Raven's items, not 1-relational ones. An induced rule is
      exactly that: a second-order relation over the set of
      pairwise (antecedent → consequent) relations. ``induce()``
      accepts a candidate only when it covers *every* exemplar at
      once — integration, not one-pair pattern-matching.

    - **Rule search (mid-DLPFC)**. Reverberi et al. (2011) found
      mid-DLPFC active during rule *search and discovery* but not
      during rule *following*. The candidate space is searched in
      the order Badre & D'Esposito's rostrocaudal gradient implies:
      response-level rules first (identity, constants), then
      feature-level (rigid transforms, recolorings), then
      relational-level (substitution-by-exemplar — a rule that
      *is* a set of relations). Concrete before abstract.

    - **Rule-change signal (FPC)**. Strange et al. (2001): fronto-
      polar cortex fires selectively after a rule change and its
      response *declines as the rule is acquired*. ``observe()``
      encodes that: a new exemplar contradicting the standing rule
      marks ``needs_search`` — the engagement signal — and a rule
      that keeps predicting quiets it.

    - **Two induction routes** (Carpenter et al.; exemplar- vs
      rule-based transfer literatures). When an operator covers
      all pairs the answer is a rule — the mPFC + anterior
      hippocampus *schematic* route. When none does, ``apply``
      falls back to the nearest stored exemplar under rigid
      transforms — the intraparietal-sulcus + posterior hippocampus
      *exemplar-similarity* route. Both are kept because the brain
      keeps both.

    - **Integrative encoding (aHPC/CA1)**. Pairs are stored as
      observed and recombined at induction time — generalization
      emerging from linked episodes rather than a blended trace.

    - **Inhibition of irrelevant features** is itself a candidate:
      whether antecedent-matching must be exact or may be
      orientation-invariant is *discovered* from the exemplars, not
      assumed — a demonstrator may show tilt-invariant glyphs while
      presenting query glyphs already rotated, so the match
      transform must be learned rather than presumed.

The parietal contribution (the bound relational representations the
search operates on — entities, panels, patterns) lives upstream in
``parietal_lobe`` scene parsing; the acquired rule, once it keeps
predicting, is what ``competence.py`` consolidates as a skill —
rule *following* is temporal/motor, not prefrontal.

════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from parietal_lobe.grid import Grid
from parietal_lobe.strips import join_run, orbit, segment_ex, tile_seq, unit_shape

__all__ = [
    "AbstractLevel",
    "ExampleInductor",
    "PairStore",
    "RuleStatus",
    "TransformRule",
]

logger = logging.getLogger(__name__)

#: Like HypothesisLedger: support counts consistent trials; violations
#: count contradictions. A rule needs this much support to stand as
#: verified — low, because induction gates attention, not action.
_VERIFY_SUPPORT = 2


class AbstractLevel(Enum):
    """Candidate rules ordered by the rostrocaudal abstraction axis.

    Rostral-most last: each level's rules are tried only after every
    more-concrete level fails to cover the exemplars — the search
    prefers the least abstract rule that explains all the evidence.
    """

    RESPONSE = 0  # caudal-most: identity / constant output
    FEATURE = 1  # rigid transforms, uniform recolor
    DIMENSION = 2  # reserved: conditional maps (future)
    RELATIONAL = 3  # rostral-most: substitution-by-exemplar


class RuleStatus(Enum):
    """Same epistemic bookkeeping as hypotheses.py."""

    PROPOSED = "proposed"
    VERIFIED = "verified"
    REFUTED = "refuted"


@dataclass
class _Pair:
    antecedent: Grid
    consequent: Grid


class PairStore:
    """The exemplar memory — aHPC's integrative-encoding half.

    Pairs are stored as encoded, never blended; generalization
    recombines them at induction time (retrieval-based accounts of
    hippocampal generalization). Capped so a flood of near-identical
    demonstrations doesn't bury informative ones: an incoming pair
    identical to a stored one just reinforces it via ``count``.
    """

    CAP = 64

    def __init__(self) -> None:
        self.pairs: list[_Pair] = []
        self.counts: list[int] = []

    def add(self, antecedent: Grid, consequent: Grid) -> bool:
        """Encode one (antecedent → consequent) demonstration.

        Returns True if a genuinely new pair was stored — i.e. the
        evidence base changed and re-induction may be warranted.
        """
        for i, p in enumerate(self.pairs):
            if p.antecedent == antecedent and p.consequent == consequent:
                self.counts[i] += 1
                return False
        if len(self.pairs) >= self.CAP:
            # Evict the weakest (least-reinforced) — same ledger
            # discipline HYPOTHESIS_CAP imposes on hypotheses.py.
            i = self.counts.index(min(self.counts))
            del self.pairs[i]
            del self.counts[i]
        self.pairs.append(_Pair(antecedent, consequent))
        self.counts.append(1)
        return True

    def __len__(self) -> int:
        return len(self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [
                {
                    "in": p.antecedent.to_lists(),
                    "out": p.consequent.to_lists(),
                    "n": c,
                }
                for p, c in zip(self.pairs, self.counts, strict=True)
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairStore:
        """Rebuild the exemplar ledger — pairs with their stored
        reinforcement counts, exactly as ``to_dict`` wrote them."""
        store = cls()
        for item in data.get("pairs", []):
            store.pairs.append(
                _Pair(
                    Grid.from_lists(item["in"]),
                    Grid.from_lists(item["out"]),
                )
            )
            store.counts.append(int(item.get("n", 1)))
        return store


# ── Candidate rules ─────────────────────────────────────────────
#
# Every rule answers one question — ``predict(antecedent)`` — and is
# fitted from *all* stored pairs at once. A rule is a candidate for
# verification only when it covers every exemplar: relational
# integration is the acceptance test, not a similarity threshold.


class TransformRule:
    """A transformation rule induced from exemplars."""

    kind: str

    def predict(self, query: Grid) -> Grid | None:  # pragma: no cover
        raise NotImplementedError

    @property
    def level(self) -> AbstractLevel:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover
        return self.kind


class _Identity(TransformRule):
    kind = "identity"

    @property
    def level(self) -> AbstractLevel:
        return AbstractLevel.RESPONSE

    def predict(self, query: Grid) -> Grid | None:
        return query


class _Constant(TransformRule):
    """Every antecedent produced the same consequent → emit it."""

    kind = "constant"

    def __init__(self, output: Grid) -> None:
        self.output = output

    @property
    def level(self) -> AbstractLevel:
        return AbstractLevel.RESPONSE

    def predict(self, query: Grid) -> Grid | None:
        return self.output


class _Rigid(TransformRule):
    """One dihedral transform (rotations × reflections) or scale."""

    kind = "rigid"

    def __init__(self, op: str, param: int = 0) -> None:
        self.op = op
        self.param = param

    @property
    def level(self) -> AbstractLevel:
        return AbstractLevel.FEATURE

    def predict(self, query: Grid) -> Grid | None:
        if self.op == "rot":
            return query.rotate90(self.param)
        if self.op == "reflect_h":
            return query.reflect_h()
        if self.op == "reflect_v":
            return query.reflect_v()
        if self.op == "reflect_diag":
            return query.reflect_diag()
        if self.op == "scale":
            return query.scale(self.param)
        return None

    def describe(self) -> str:
        return f"{self.op}({self.param})" if self.op == "rot" else self.op


class _Recolor(TransformRule):
    """A uniform color permutation — built cell-wise across pairs."""

    kind = "recolor"

    def __init__(self, mapping: dict[int, int]) -> None:
        self.mapping = mapping

    @property
    def level(self) -> AbstractLevel:
        return AbstractLevel.FEATURE

    def predict(self, query: Grid) -> Grid | None:
        return query.recolor(dict(self.mapping))

    def describe(self) -> str:
        return f"recolor({self.mapping})"


class _Substitution(TransformRule):
    """The relational-level rule: a dictionary of exemplars.

    Second-order — it doesn't transform its input, it *consults the
    stored relations*: segment the query into antecedent-sized
    blocks, match each block to an exemplar antecedent (under the
    discovered matching invariance), emit the exemplar's consequent.
    A wall of demonstrations is this object: the rule is the set of
    relations.

    ``match_dihedral`` is the learned inhibition-of-features bit:
    when the exemplars themselves demonstrate that matching survives
    rigid transforms (a rotated antecedent still names its glyph),
    matching is dihedral-invariant; otherwise exact.
    """

    kind = "substitution"

    def __init__(
        self,
        pairs: list[_Pair],
        match_dihedral: bool,
        glyph_probe: Callable[[Grid], str | None] | None = None,
    ) -> None:
        self.pairs = pairs
        self.match_dihedral = match_dihedral
        #: Wall color discovered at parse time — separators are a
        #: property of the *strip*, not of the tiles.
        self.separator: int | None = None
        #: Optional ventral-stream lookup: tile → its learned shape
        #: concept, or None. Consulted only where exact matching
        #: misses — a recognition assist, never a replacement for
        #: evidence. Runtime wiring; nothing here serializes it.
        self.glyph_probe = glyph_probe

    @property
    def level(self) -> AbstractLevel:
        return AbstractLevel.RELATIONAL

    def _variants(self, g: Grid) -> list[Grid]:
        """All dihedral views of a pattern (the glyph under tilt)."""
        return orbit(g)

    def _unit(self) -> tuple[int, int] | None:
        """The tile unit every run in this dictionary shares —
        antecedent runs and consequent runs alike."""
        return unit_shape([p.antecedent for p in self.pairs] + [p.consequent for p in self.pairs])

    def _match_pair(self, block: Grid) -> _Pair | None:
        """The first exemplar whose antecedent this block equals —
        first-rule-wins, dictionary order."""
        if not self.match_dihedral:
            for p in self.pairs:
                if p.antecedent == block:
                    return p
            return None
        for p in self.pairs:
            if any(v == block for v in self._variants(p.antecedent)):
                return p
        return None

    def _match_tile(self, ante: Grid, block: Grid) -> bool:
        """One tile of a run against one query tile — the pair's
        matching invariance applies tile-by-tile. Exact and dihedral
        evidence decide first; when both miss, a learned visual
        identity may still bridge — but only if the probe recognizes
        *both* tiles as the same earned concept."""
        exact = any(v == block for v in orbit(ante)) if self.match_dihedral else ante == block
        if exact:
            return True
        probe = self.glyph_probe
        if probe is None:
            return False
        ia, ib = probe(ante), probe(block)
        return ia is not None and ia == ib

    def _match_seq(
        self, blocks: list[Grid], pos: int
    ) -> tuple[list[Grid], int | None, int, int] | None:
        """The first dictionary entry whose antecedent *run* covers
        ``blocks[pos:pos+m]`` tile-for-tile — first-rule-wins, same
        ordered matching the demonstrator's checker uses. Returns
        ``(consequent tiles, wall, gap, tiles consumed)``."""
        unit = self._unit()
        for p in self.pairs:
            seq = tile_seq(p.antecedent, unit) if unit else None
            antes = seq[0] if seq else [p.antecedent]
            m = len(antes)
            if pos + m > len(blocks) or m < 1:
                continue
            if all(self._match_tile(antes[j], blocks[pos + j]) for j in range(m)):
                tiles, w, g = self._tiles_of(p)
                return tiles, w, g, m
        return None

    def match(self, block: Grid) -> Grid | None:
        """The stored consequent of the first matching exemplar."""
        pair = self._match_pair(block)
        return pair.consequent if pair is not None else None

    def _tiles_of(self, pair: _Pair) -> tuple[list[Grid], int | None, int]:
        """The matched consequent as a tile sequence.

        A consequent may be a *run* — one antecedent mapping to
        several consequent tiles laid out side by side at a fixed
        pitch behind shared walls. Split it on the dictionary's
        shared tile unit; a single tile stays atomic.
        """
        unit = self._unit()
        seq = tile_seq(pair.consequent, unit) if unit else None
        if seq is None:
            seq = tile_seq(pair.consequent, pair.antecedent.shape)
        if seq is None:
            return [pair.consequent], None, 0
        return seq

    def _translate(self, query: Grid) -> tuple[list[Grid], int | None, int] | None:
        """Query strip → flat consequent-tile sequence, with the
        wall color and pitch the dictionary's right-hand side uses.
        None when the query isn't a covered phrase."""
        unit = self._unit()
        parsed = segment_ex(
            query,
            self.separator,
            exemplar_shape=unit,
        )
        blocks: list[Grid] | None = None
        qgap = 0
        if parsed is not None:
            blocks, sep = parsed
            self.separator = sep
            # The query's own pitch is the only demonstrated separator
            # width when consequents are single tiles — carry it so a
            # joined answer is still a legible strip.
            n = len(blocks)
            if n > 1:
                if all(b.height == query.height for b in blocks):
                    qgap = max(
                        0,
                        (query.width - sum(b.width for b in blocks)) // (n - 1),
                    )
                elif all(b.width == query.width for b in blocks):
                    qgap = max(
                        0,
                        (query.height - sum(b.height for b in blocks)) // (n - 1),
                    )
        elif unit is not None:
            seq = tile_seq(query, unit)
            if seq is not None:
                blocks, _qw, qgap = seq
        if not blocks:
            return None
        emitted: list[Grid] = []
        wall: int | None = self.separator
        gap = qgap
        pos = 0
        while pos < len(blocks):
            hit = self._match_seq(blocks, pos)
            if hit is None:
                return None
            tiles, w, g, m = hit
            emitted.extend(tiles)
            if w is not None:
                # The answer is written in the consequents' own
                # separator — the dictionary's right-hand side
                # shares the answer strip's wall color.
                wall, gap = w, g
            pos += m
        return emitted, wall, gap

    def predict_seq(self, query: Grid) -> list[Grid] | None:
        """The answer as a tile sequence — what a slot-writer needs,
        before any reassembly layout is imposed."""
        parts = self._translate(query)
        if parts is not None:
            return parts[0]
        pair = self._match_pair(query)
        if pair is None:
            return None
        return self._tiles_of(pair)[0]

    def predict(self, query: Grid) -> Grid | None:
        parts = self._translate(query)
        if parts is None:
            # Try one whole block — the query may itself be one cell
            # of the phrase rather than a strip.
            hit = self.match(query)
            return hit
        tiles, wall, gap = parts
        # Re-emit in the consequents' own run layout: wall columns
        # between tiles, none at the edges — so predicting a stored
        # antecedent reproduces its stored consequent exactly.
        return join_run(tiles, wall if wall is not None else self.separator, gap)

    def describe(self) -> str:
        mode = "dihedral" if self.match_dihedral else "exact"
        return f"substitute[{mode}]({len(self.pairs)} rules)"


# ── The inductor ────────────────────────────────────────────────


class ExampleInductor:
    """Rostrolateral-PFC machinery: integrate exemplar pairs into rules.

    Lifecycle mirrors the neural dissociation:

      - ``observe(a, b)`` — encode a demonstration; a pair that
        contradicts the standing rule flags ``needs_search`` (the
        FPC rule-change signal).
      - ``induce()`` — the mid-DLPFC search: candidates tried in
        rostrocaudal order, each required to cover ALL pairs
        (relational integration). Returns the verified rule or None.
      - ``apply(query)`` — rule *following*: the verified rule's
        prediction, else the exemplar-similarity fallback.
    """

    def __init__(self, *, match_invariance: bool = True) -> None:
        self.store = PairStore()
        self.match_invariance = match_invariance
        self.rule: TransformRule | None = None
        self.status = RuleStatus.PROPOSED
        self.support = 0
        self.violations = 0
        #: True when stored evidence outran the current rule — the
        #: frontopolar "rule changed" engagement flag.
        self.needs_search = False
        #: Demonstration epoch — advanced once per observed scene.
        #: Pairs committed in the same epoch are simultaneous
        #: dictionary entries (first entry wins); a pair from an
        #: earlier epoch contradicted by a later demonstration is a
        #: *changed dictionary* and gets replaced.
        self._gen = 0
        self._pair_gens: list[int] = []
        #: Set to the antecedent when a pair is replaced — the
        #: session uses it to re-open the seen-gate so displaced
        #: entries can re-earn their place.
        self.last_replaced: Grid | None = None
        #: Optional ventral-stream tile→concept lookup, stamped into
        #: every ``_Substitution`` this inductor produces. Runtime
        #: only — installed by whoever owns a visual hierarchy;
        #: serialization ignores it and exact matching doesn't need
        #: it.
        self.glyph_probe: Callable[[Grid], str | None] | None = None

    def set_glyph_probe(self, probe: Callable[[Grid], str | None] | None) -> None:
        """Install (or remove) the visual-identity lookup — also
        stamped onto a standing substitution rule, since a rule
        acquired before the probe arrived should not stay blind."""
        self.glyph_probe = probe
        if isinstance(self.rule, _Substitution):
            self.rule.glyph_probe = probe

    def begin_frame(self) -> None:
        """Advance the demonstration epoch — call once per observed
        frame/scene so evidence freshness is measurable."""
        self._gen += 1

    # ── Encoding ──────────────────────────────────────────────

    def observe(self, antecedent: Grid, consequent: Grid) -> bool:
        """Encode one demonstration pair.

        Returns True when this observation changed the evidence base
        (new pair, or a contradiction of the standing rule).

        A re-demonstrated antecedent carrying a *different*
        consequent in a *later* epoch is a changed dictionary, not
        accumulated evidence: the stale entry is replaced in place —
        dictionary position keeps first-match ordering honest —
        rather than left to shadow the live rule. Duplicates
        committed in the *same* epoch are simultaneous entries and
        keep first-wins order. (Same-antecedent-same-consequent
        re-demonstrations still reinforce via the store.)
        """
        for i, p in enumerate(self.store.pairs):
            if p.antecedent == antecedent and p.consequent == consequent:
                # Identical demonstration — reinforce and mark it
                # freshly seen; a contradiction must arrive in a
                # later epoch to displace it.
                self.store.counts[i] += 1
                self._pair_gens[i] = self._gen
                new = False
                break
            same = p.antecedent == antecedent or (
                self.match_invariance
                and p.antecedent.shape == antecedent.shape
                and any(v == antecedent for v in orbit(p.antecedent))
            )
            if same and self._pair_gens[i] < self._gen:
                self.store.pairs[i] = _Pair(antecedent, consequent)
                self.store.counts[i] = 1
                self._pair_gens[i] = self._gen
                self.needs_search = True
                self.last_replaced = antecedent
                return True
        else:
            if len(self.store.pairs) >= self.store.CAP:
                i = self.store.counts.index(min(self.store.counts))
                del self.store.pairs[i]
                del self.store.counts[i]
                del self._pair_gens[i]
            self.store.pairs.append(_Pair(antecedent, consequent))
            self.store.counts.append(1)
            self._pair_gens.append(self._gen)
            new = True
        if self.rule is not None:
            pred = self.rule.predict(antecedent)
            if pred is not None and pred == consequent:
                self.support += 1
            else:
                self.violations += 1
                self.needs_search = True
        return new

    # ── Search ────────────────────────────────────────────────

    def _covers(self, rule: TransformRule, pairs: list[_Pair]) -> bool:
        """Relational integration: the rule must hold under every
        exemplar relation simultaneously, not most of them."""
        return all(rule.predict(p.antecedent) == p.consequent for p in pairs)

    def _fit_recolor(self, pairs: list[_Pair]) -> _Recolor | None:
        mapping: dict[int, int] = {}
        for p in pairs:
            if p.antecedent.shape != p.consequent.shape:
                return None
            for r, c, v in p.antecedent.iter_cells():
                out = p.consequent.at(r, c)
                if v in mapping and mapping[v] != out:
                    return None
                mapping[v] = out
        if not mapping or all(k == v for k, v in mapping.items()):
            return None
        return _Recolor(mapping)

    def _match_is_dihedral(self, pairs: list[_Pair]) -> bool:
        """Do exemplars demonstrate orientation-invariant matching?

        True when some pair's antecedent and consequent are equal
        only under a rigid transform — the evidence itself says
        identity survives rotation (or the pair structure implies
        the query side will arrive tilted).
        """
        probe = _Substitution(pairs, match_dihedral=True)
        for p in pairs:
            if p.antecedent != p.consequent:
                for v in probe._variants(p.antecedent):
                    if v == p.consequent:
                        return True
        return self.match_invariance

    def induce(self, *, force: bool = False) -> TransformRule | None:
        """Search the candidate space, concrete → abstract.

        Runs only when evidence outran the rule (or forced): the FPC
        declines its engagement as the rule is acquired — a covering
        rule that keeps predicting is not re-searched.
        """
        pairs = self.store.pairs
        if len(pairs) < 1:
            return None
        if self.rule is not None and not self.needs_search and not force:
            if self._covers(self.rule, pairs):
                self.status = (
                    RuleStatus.VERIFIED if self.support >= _VERIFY_SUPPORT else RuleStatus.PROPOSED
                )
                return self.rule

        candidates: list[TransformRule] = []
        # RESPONSE level.
        candidates.append(_Identity())
        outs = {p.consequent for p in pairs}
        if len(outs) == 1:
            candidates.append(_Constant(outs.pop()))
        # FEATURE level.
        for k in range(1, 4):
            candidates.append(_Rigid("rot", k))
        for op in ("reflect_h", "reflect_v", "reflect_diag"):
            candidates.append(_Rigid(op))
        for s in (2, 3):
            candidates.append(_Rigid("scale", s))
        if recolor := self._fit_recolor(pairs):
            candidates.append(recolor)
        # RELATIONAL level — the exemplar set as a dictionary.
        # Matching is per-block, so the pairs needn't share one
        # shape: a heterogeneous store is still a lookup table —
        # a mixed-size dictionary is a dictionary.
        if len(pairs) >= 2:
            candidates.append(
                _Substitution(
                    pairs,
                    match_dihedral=self._match_is_dihedral(pairs),
                    glyph_probe=self.glyph_probe,
                )
            )

        for rule in sorted(candidates, key=lambda r: r.level.value):
            try:
                if self._covers(rule, pairs):
                    self.rule = rule
                    self.support = sum(
                        1 for p in pairs if rule.predict(p.antecedent) == p.consequent
                    )
                    self.violations = 0
                    self.needs_search = False
                    self.status = (
                        RuleStatus.VERIFIED
                        if self.support >= _VERIFY_SUPPORT
                        else RuleStatus.PROPOSED
                    )
                    return rule
            except Exception as e:  # noqa: BLE001
                note = f"induction candidate {rule.describe()} failed"
                logger.debug("%s: %s", note, e)
        self.status = RuleStatus.REFUTED if self.violations else RuleStatus.PROPOSED
        return None

    # ── Following ─────────────────────────────────────────────

    def apply(self, query: Grid) -> Grid | None:
        """Rule following — verified rule first, exemplar fallback.

        The fallback is the brain's second induction route: when no
        operator covers the exemplars, the nearest stored exemplar
        (exact or rigid-transform match) still answers — similarity-
        based transfer instead of rule transfer.
        """
        if self.rule is None:
            self.induce()
        if self.rule is not None:
            out = self.rule.predict(query)
            if out is not None:
                return out
        # Exemplar route: nearest antecedent wins its consequent.
        sub = _Substitution(
            self.store.pairs,
            match_dihedral=self.match_invariance,
            glyph_probe=self.glyph_probe,
        )
        return sub.match(query)

    def apply_seq(self, query: Grid) -> list[Grid] | None:
        """The answer as a tile sequence — the substitution rule's
        flat consequent list, before any reassembly layout. For a
        slot-writer this is the target; for everything else
        ``apply``'s joined grid is equivalent."""
        if self.rule is None:
            self.induce()
        if isinstance(self.rule, _Substitution):
            return self.rule.predict_seq(query)
        out = self.apply(query)
        return [out] if out is not None else None

    def describe(self) -> str:
        """One-line telemetry — what the induced rule currently is."""
        if self.rule is None:
            return f"no rule ({len(self.store)} pairs, {self.status.value})"
        return (
            f"{self.rule.describe()} [{self.status.value}] "
            f"support={self.support} violations={self.violations}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store.to_dict(),
            "match_invariance": self.match_invariance,
            "support": self.support,
            "violations": self.violations,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleInductor:
        """Restore an inductor from ``to_dict`` — the pair store,
        the evidence ledger, and the status the rule was left in.
        The rule itself is *not* carried: it re-induces from the
        pairs on next contact."""
        ind = cls(match_invariance=bool(data.get("match_invariance", True)))
        ind.store = PairStore.from_dict(data.get("store", {}))
        # Imported evidence is *past* evidence — every stored pair
        # sits at epoch 0, displaceable by live demonstrations.
        ind._pair_gens = [0] * len(ind.store.pairs)
        ind.support = int(data.get("support", 0))
        ind.violations = int(data.get("violations", 0))
        ind.status = RuleStatus(data.get("status", RuleStatus.PROPOSED.value))
        return ind
