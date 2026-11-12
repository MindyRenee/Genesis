"""Formal logic — the model-theoretic substrate of reasoning.

The ``Reasoner`` walks the concept network associatively: chains,
confidence, revision-on-evidence. That is the right-hemisphere style
of inference — fast, defeasible, semantic. This module is the other
organ: *mechanical* logic, where conclusions are forced rather than
plausible.

The mathematics implemented is the canonical set for each human
logic function:

- **Deduction — model theory** (Johnson-Laird's mental models, not
  proof search): enumerate the valuations consistent with the
  premises; a conclusion is valid iff it holds in *every* such model.
  An invalid inference returns the countermodel that refutes it —
  counterexample search is what makes humans rational, not rules.

- **Categorical / syllogistic — region semantics**: three terms
  partition a world into eight Venn regions; a model is the pattern
  of inhabited regions. "All A are B" empties A∧¬B; "Some A are B"
  inhabits A∧B. Validity is checked over all 256 patterns.

- **Belief revision — AGM**: expansion ``K + p``, contraction
  ``K ÷ p``, revision ``K * p = (K ÷ ¬p) + p`` (the Levi identity).
  Contraction surrenders the *least entrenched* sentences first —
  Quine's maxim of minimal mutilation.

- **Conditionals & counterfactuals — the Ramsey test**:
  "if p then q" is acceptable iff ``K * p ⊨ q``. Counterfactual
  evaluation is revision semantics applied to a false antecedent —
  the same operator, no second machinery.

- **Abduction — minimal explanans**: the smallest consistent subset
  of candidate explanations that entails the observation.

Everything here is a set-of-sentences engine: callers ground atoms
however they like (concept ids, world facts, propositions). The
engine decides *what must be true*; the concept network decides
what is worth asking.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product

__all__ = [
    "MAX_ATOMS",
    "Atom",
    "Categorical",
    "Deduction",
    "Formula",
    "Quantifier",
    "all_of",
    "and_",
    "any_of",
    "atom",
    "contract",
    "entails",
    "expand",
    "explain",
    "iff",
    "implies",
    "models",
    "not_",
    "or_",
    "parse",
    "ramsey",
    "revise",
    "satisfiable",
    "syllogism",
    "verify",
]

MAX_ATOMS = 14
"""Bound on distinct atoms per problem — 2^14 valuations is instant
and still generous: human mental models hold ~3 possibilities."""


# ------------------------------------------------------------------
# Syntax — propositional formulas as immutable trees
# ------------------------------------------------------------------


class Formula:
    """A propositional formula tree. Base class — use the
    constructors (``atom``, ``not_``, ``and_``…) or ``parse``."""

    __slots__ = ()

    def atoms(self) -> frozenset[str]:
        raise NotImplementedError

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        raise NotImplementedError

    def __and__(self, other: Formula) -> Formula:
        return and_(self, other)

    def __or__(self, other: Formula) -> Formula:
        return or_(self, other)

    def __invert__(self) -> Formula:
        return not_(self)

    def __rshift__(self, other: Formula) -> Formula:
        """``p >> q`` reads as implication: p → q."""
        return implies(self, other)


@dataclass(frozen=True, slots=True)
class Atom(Formula):
    name: str

    def atoms(self) -> frozenset[str]:
        return frozenset({self.name})

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return bool(model.get(self.name, False))

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class _Const(Formula):
    value: bool

    def atoms(self) -> frozenset[str]:
        return frozenset()

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        return self.value

    def __repr__(self) -> str:
        return "⊤" if self.value else "⊥"


TOP = _Const(True)
BOTTOM = _Const(False)


@dataclass(frozen=True, slots=True)
class _Nary(Formula):
    """And/Or share one node: ``op`` is "and" or "or"."""

    op: str
    parts: tuple[Formula, ...]

    def atoms(self) -> frozenset[str]:
        out: set[str] = set()
        for p in self.parts:
            out |= p.atoms()
        return frozenset(out)

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        if self.op == "and":
            return all(p.evaluate(model) for p in self.parts)
        return any(p.evaluate(model) for p in self.parts)

    def __repr__(self) -> str:
        join = " ∧ " if self.op == "and" else " ∨ "
        return "(" + join.join(_render(p) for p in self.parts) + ")"


@dataclass(frozen=True, slots=True)
class _Unary(Formula):
    """Not / Implies / Iff share one node keyed by ``op``."""

    op: str
    a: Formula
    b: Formula | None = None

    def atoms(self) -> frozenset[str]:
        out = self.a.atoms()
        if self.b is not None:
            out |= self.b.atoms()
        return out

    def evaluate(self, model: Mapping[str, bool]) -> bool:
        if self.op == "not":
            return not self.a.evaluate(model)
        assert self.b is not None
        if self.op == "implies":
            return (not self.a.evaluate(model)) or self.b.evaluate(model)
        return self.a.evaluate(model) == self.b.evaluate(model)

    def __repr__(self) -> str:
        if self.op == "not":
            return f"¬{_render(self.a)}"
        assert self.b is not None
        sym = " → " if self.op == "implies" else " ↔ "
        return f"({_render(self.a)}{sym}{_render(self.b)})"


def _render(f: Formula) -> str:
    if isinstance(f, (Atom, _Const)):
        return repr(f)
    return repr(f)


def atom(name: str) -> Atom:
    """Wrap ``name`` as a propositional atom — the leaf every formula composes from."""
    return Atom(name)


def not_(f: Formula) -> Formula:
    if isinstance(f, _Unary) and f.op == "not":
        return f.a  # ¬¬p collapses
    return _Unary("not", f)


def and_(*fs: Formula) -> Formula:
    flat: list[Formula] = []
    for f in fs:
        if isinstance(f, _Nary) and f.op == "and":
            flat.extend(f.parts)
        elif f is not TOP:
            flat.append(f)
    if not flat:
        return TOP
    if len(flat) == 1:
        return flat[0]
    if BOTTOM in flat:
        return BOTTOM
    return _Nary("and", tuple(flat))


def or_(*fs: Formula) -> Formula:
    flat: list[Formula] = []
    for f in fs:
        if isinstance(f, _Nary) and f.op == "or":
            flat.extend(f.parts)
        elif f is not BOTTOM:
            flat.append(f)
    if not flat:
        return BOTTOM
    if len(flat) == 1:
        return flat[0]
    if TOP in flat:
        return TOP
    return _Nary("or", tuple(flat))


def implies(a: Formula, b: Formula) -> Formula:
    return _Unary("implies", a, b)


def iff(a: Formula, b: Formula) -> Formula:
    return _Unary("iff", a, b)


def all_of(fs: Iterable[Formula]) -> Formula:
    return and_(*fs)


def any_of(fs: Iterable[Formula]) -> Formula:
    return or_(*fs)


# ------------------------------------------------------------------
# Parser — "~a & (b | c) -> d" / "¬ ∧ ∨ → ↔", precedence: ¬ > ∧ > ∨ > → > ↔
# ------------------------------------------------------------------


def parse(text: str) -> Formula:
    """Parse a propositional formula string into a ``Formula``.

    Accepts both ASCII and symbolic connectives — ``~a & (b | c) -> d``
    and ``¬a ∧ (b ∨ c) → d`` are equivalent — with precedence
    ``¬ > ∧ > ∨ > → > ↔`` and right-associative implication.
    ``true``/``⊤`` and ``false``/``⊥`` parse to the constants.
    Raises ``ValueError`` on unbalanced parens, trailing tokens, or a
    character the lexer cannot read.
    """
    tokens = _lex(text)
    pos = 0

    def peek() -> str:
        return tokens[pos] if pos < len(tokens) else ""

    def eat() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of formula")
        t = tokens[pos]
        pos += 1
        return t

    def p_iff() -> Formula:
        f = p_imp()
        while peek() in ("<->", "↔", "=="):
            eat()
            f = iff(f, p_imp())
        return f

    def p_imp() -> Formula:
        f = p_or()
        while peek() in ("->", "→", "=>"):
            eat()
            f = implies(f, p_imp())  # right-associative: a → b → c = a → (b → c)
        return f

    def p_or() -> Formula:
        f = p_and()
        while peek() in ("|", "∨", "or"):
            eat()
            f = or_(f, p_and())
        return f

    def p_and() -> Formula:
        f = p_not()
        while peek() in ("&", "∧", "and"):
            eat()
            f = and_(f, p_not())
        return f

    def p_not() -> Formula:
        if peek() in ("~", "¬", "!", "not"):
            eat()
            return not_(p_not())
        return p_leaf()

    def p_leaf() -> Formula:
        if peek() == "(":
            eat()
            f = p_iff()
            if eat() != ")":
                raise ValueError("unbalanced parens")
            return f
        name = eat()
        if not name or name == ")":
            raise ValueError(f"expected atom, got {name!r}")
        if name in ("true", "⊤"):
            return TOP
        if name in ("false", "⊥"):
            return BOTTOM
        return atom(name)

    result = p_iff()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens: {tokens[pos:]}")
    return result


def _lex(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("<->", i) or text.startswith("->", i) or text.startswith("=>", i):
            n = 3 if text.startswith("<->", i) else 2
            out.append(text[i : i + n])
            i += n
        elif c == "=" and text.startswith("==", i):
            out.append("==")
            i += 2
        elif c in "()" or c in "&|~¬∧∨→↔!":
            out.append(c)
            i += 1
        else:
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] in "_:.-"):
                j += 1
            if j == i:
                raise ValueError(f"bad character {c!r}")
            out.append(text[i:j])
            i = j
    return out


# ------------------------------------------------------------------
# Model theory — mental-models enumeration
# ------------------------------------------------------------------

Model = dict[str, bool]


def _valuations(atoms: Iterable[str]) -> Iterator[Model]:
    names = sorted(atoms)
    if len(names) > MAX_ATOMS:
        raise ValueError(
            f"{len(names)} atoms exceeds MAX_ATOMS={MAX_ATOMS}; "
            "the mental-model bound is a feature — split the problem"
        )
    for bits in product((False, True), repeat=len(names)):
        yield dict(zip(names, bits, strict=True))


def models(formulas: Formula | Iterable[Formula], *, limit: int | None = None) -> Iterator[Model]:
    """Yield the valuations in which every formula holds — the set
    of mental models the premises leave open."""
    if isinstance(formulas, Formula):
        formulas = (formulas,)
    fs = tuple(formulas)
    atoms: set[str] = set()
    for f in fs:
        atoms |= f.atoms()
    n = 0
    for m in _valuations(atoms):
        if all(f.evaluate(m) for f in fs):
            yield m
            n += 1
            if limit is not None and n >= limit:
                return


def satisfiable(formulas: Iterable[Formula]) -> bool:
    return next(models(formulas, limit=1), None) is not None


def entails(kb: Iterable[Formula], q: Formula) -> bool:
    """KB ⊨ q iff every model of KB is a model of q — equivalently
    KB ∪ {¬q} is unsatisfiable (refutation)."""
    kb = tuple(kb)
    return not satisfiable([*kb, not_(q)])


@dataclass(frozen=True, slots=True)
class Deduction:
    """Result of a validity check — the answer plus its warrant.
    ``premises``/``conclusion`` are untyped so the same record serves
    the propositional engine (``Formula``) and the categorical engine
    (``Categorical``); ``countermodel`` is a truth assignment for the
    former, an inhabited-region set for the latter."""

    valid: bool
    premises: tuple
    conclusion: object
    countermodel: dict | None = None
    models_examined: int = 0


def verify(premises: Iterable[Formula], conclusion: Formula) -> Deduction:
    """Deductive validity via mental models: valid iff the conclusion
    holds in every model of the premises; else the first countermodel
    found — the possibility where premises hold and it fails."""
    premises = tuple(premises)
    atoms: set[str] = set(conclusion.atoms())
    for f in premises:
        atoms |= f.atoms()
    examined = 0
    for m in _valuations(atoms):
        if all(p.evaluate(m) for p in premises):
            examined += 1
            if not conclusion.evaluate(m):
                return Deduction(False, premises, conclusion, m, examined)
    return Deduction(True, premises, conclusion, None, examined)


# ------------------------------------------------------------------
# AGM — expansion, contraction, revision over belief sets
# ------------------------------------------------------------------

BeliefSet = frozenset[Formula]


def _depth(f: Formula) -> int:
    """Default entrenchment — literals (atoms and their negations) are
    observation-level and cost nothing to surrender; compound claims
    carry structure whose loss is a larger mutilation (Gärdenfors:
    rules outrank singular facts)."""
    if isinstance(f, (Atom, _Const)):
        return 0
    if isinstance(f, _Unary):
        if f.op == "not" and isinstance(f.a, (Atom, _Const)):
            return 0  # a negated atom is still just a literal
        return 1 + _depth(f.a) + (_depth(f.b) if f.b is not None else 0)
    assert isinstance(f, _Nary)
    return 1 + sum(_depth(p) for p in f.parts)


def expand(kb: Iterable[Formula], p: Formula) -> BeliefSet:
    """K + p — add the belief; closure is implicit (entailment is
    computed on demand rather than stored)."""
    return frozenset({*kb, p})


def contract(
    kb: Iterable[Formula],
    p: Formula,
    entrenchment: Mapping[Formula, float] | None = None,
) -> BeliefSet:
    """K ÷ p — remove beliefs until p is no longer entailed.

    Approximates partial-meet contraction by surrendering the least
    entrenched sentences first (minimal mutilation). ``entrenchment``
    maps formula → resistance; unmapped formulas surrender first.
    A tautological p cannot be contracted — returns kb unchanged.
    """
    kb_set = frozenset(kb)
    if entails((), p) or not entails(kb_set, p):
        return kb_set  # vacuity: nothing to give up / impossible
    rank = {f: (entrenchment or {}).get(f, float(_depth(f))) for f in kb_set}
    out = set(kb_set)
    # repeatedly cut the least entrenched member of the remaining set
    while entails(frozenset(out), p):
        victim = min(out, key=lambda f: (rank[f], repr(f)))
        out.discard(victim)
    return frozenset(out)


def revise(
    kb: Iterable[Formula],
    p: Formula,
    entrenchment: Mapping[Formula, float] | None = None,
) -> BeliefSet:
    """K * p — the Levi identity: contract ¬p, then expand by p.
    Guarantees consistency whenever p is consistent."""
    return expand(contract(kb, not_(p), entrenchment), p)


def consistent(kb: Iterable[Formula]) -> bool:
    return satisfiable(kb)


def ramsey(
    kb: Iterable[Formula],
    p: Formula,
    q: Formula,
    entrenchment: Mapping[Formula, float] | None = None,
) -> bool:
    """The Ramsey test — "if p then q" is acceptable in K iff q is
    entailed after revising K by p. Counterfactual evaluation is the
    same operator applied when ¬p is currently believed."""
    return entails(revise(kb, p, entrenchment), q)


# ------------------------------------------------------------------
# Abduction — minimal explanans
# ------------------------------------------------------------------


def explain(
    kb: Iterable[Formula],
    q: Formula,
    candidates: Iterable[Formula],
) -> frozenset[Formula] | None:
    """Inference to a best explanation: the smallest subset of
    ``candidates`` consistent with KB that entails q — Occam applied
    to model theory. Returns None when no subset explains."""
    kb_set = frozenset(kb)
    if entails(kb_set, q):
        return frozenset()  # already explained
    cands = list(candidates)
    # smallest-first search: subsets of size 1, 2, … (bounded by count)
    for size in range(1, len(cands) + 1):
        for combo in _combinations(cands, size):
            trial = kb_set | frozenset(combo)
            if consistent(trial) and entails(trial, q):
                return frozenset(combo)
    return None


def _combinations(items: list[Formula], size: int) -> Iterator[tuple[Formula, ...]]:
    if size == 0:
        yield ()
        return
    for i in range(len(items) - size + 1):
        for rest in _combinations(items[i + 1 :], size - 1):
            yield (items[i], *rest)


# ------------------------------------------------------------------
# Categorical / syllogistic — Venn-region model theory
# ------------------------------------------------------------------


class Quantifier(Enum):
    ALL = "all"  # ∀x: A(x) → B(x)      — region A∧¬B empty
    NO = "no"  # ∀x: A(x) → ¬B(x)    — region A∧B empty
    SOME = "some"  # ∃x: A(x) ∧ B(x)    — region A∧B inhabited
    SOME_NOT = "some_not"  # ∃x: A(x) ∧ ¬B(x) — region A∧¬B inhabited


@dataclass(frozen=True, slots=True)
class Categorical:
    """An Aristotelian proposition over two terms: Quantifier(A, B).
    Modern semantics — no existential import: "all A are B" does not
    assert that A exists."""

    quantifier: Quantifier
    a: str
    b: str

    def terms(self) -> frozenset[str]:
        return frozenset({self.a, self.b})


def _region_mask(term: str, order: tuple[str, str, str]) -> int:
    """Bitmask (over the 8 Venn regions of three terms) of the cells
    where ``term`` holds."""
    mask = 0
    for cell in range(8):
        if cell & (1 << order.index(term)):
            mask |= 1 << cell
    return mask


def _categorical_ok(cat: Categorical, inhabited: int, order: tuple[str, str, str]) -> bool:
    a = _region_mask(cat.a, order)
    b = _region_mask(cat.b, order)
    if cat.quantifier == Quantifier.ALL:
        return inhabited & a & ~b == 0
    if cat.quantifier == Quantifier.NO:
        return inhabited & a & b == 0
    if cat.quantifier == Quantifier.SOME:
        return inhabited & a & b != 0
    return inhabited & a & ~b != 0


def syllogism(p1: Categorical, p2: Categorical, conclusion: Categorical) -> Deduction:
    """Validity of a two-premise categorical argument over the 256
    Venn-region patterns — the categorical subsystem of deduction
    (temporal/semantic in the brain, distinct from the sentential
    propositional engine above)."""
    terms = sorted(p1.terms() | p2.terms() | conclusion.terms())
    if len(terms) != 3:
        raise ValueError("a syllogism has exactly three terms")
    order = (terms[0], terms[1], terms[2])
    examined = 0
    for inhabited in range(1, 256):  # a nonempty world: ≥1 region lives
        if _categorical_ok(p1, inhabited, order) and _categorical_ok(p2, inhabited, order):
            examined += 1
            if not _categorical_ok(conclusion, inhabited, order):
                regions = frozenset(order[i] for i in range(3) if inhabited & (1 << i))
                return Deduction(
                    False,
                    (p1, p2),
                    conclusion,
                    {"inhabited_regions": regions},
                    examined,
                )
    return Deduction(True, (p1, p2), conclusion, None, examined)
