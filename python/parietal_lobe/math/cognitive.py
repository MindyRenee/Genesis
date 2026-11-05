"""Cognitive routing for the native parietal mathematics faculties.

This module is the language-facing bridge between the mind's mathematical
reasoning entry point and the domain faculties in this package. It accepts
only explicit, supported operation forms; it never guesses missing operands
or substitutes a numerical estimate for an exact result.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParietalAnswer:
    """A domain-faculty result normalized for the frontal reasoning layer."""

    operation: str
    answer: str
    steps: list[str] = field(default_factory=list)
    success: bool = True


def _answer(result: Any, operation: str | None = None) -> ParietalAnswer:
    """Normalize the shared result contract used by the parietal faculties."""
    ok = bool(result.ok)
    output = (
        result.output
        if ok
        else (result.error or "The mathematical faculty could not solve that form.")
    )
    steps = [f"{operation or result.operation}: {result.input}"]
    if result.data:
        steps.append(f"Result data: {result.data}")
    return ParietalAnswer(operation or result.operation, output, steps, ok)


def _literal_after(text: str, start: int = 0) -> tuple[str, int] | None:
    """Return one balanced Python-literal bracket expression and its end."""
    opening = next((i for i in range(start, len(text)) if text[i] in "[({"), None)
    if opening is None:
        return None
    pairs = {"[": "]", "(": ")", "{": "}"}
    stack: list[str] = []
    for index in range(opening, len(text)):
        char = text[index]
        if char in pairs:
            stack.append(pairs[char])
        elif char in "])}":
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return text[opening : index + 1], index + 1
    return None


def _integer_values(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"(?<![.\w])-?\d+(?![.\w])", text)]


def route_parietal_math(question: str) -> ParietalAnswer | None:
    """Route a clearly expressed higher-math question to its native faculty.

    Returning None means this router does not own the request, allowing the
    existing arithmetic/algebra/proof/discovery pathway to handle it.
    """
    text = question.strip()
    lower = text.lower()
    if not text:
        return None

    # The facade is the native parietal API. These operations are dispatched
    # here before the narrower frontal expression solver.
    from . import ParietalMath

    faculty = ParietalMath()

    # Foundational arithmetic: only explicit calculation language or
    # operator-bearing expressions enter the exact arithmetic faculty.
    arithmetic_text = text
    spoken_operations = (
        (r"\badd\s+(.+?)\s+and\s+(.+?)[?.!]*$", r"\1 + \2"),
        (r"\bsubtract\s+(.+?)\s+from\s+(.+?)[?.!]*$", r"\2 - \1"),
        (r"\bmultiply\s+(.+?)\s+by\s+(.+?)[?.!]*$", r"\1 * \2"),
        (r"\bdivide\s+(.+?)\s+by\s+(.+?)[?.!]*$", r"\1 / \2"),
        (r"\b(-?\d+(?:\.\d+)?)\s*percent\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"(\1 / 100) * \2"),
        (r"\bhalf\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 / 2"),
        (r"\bdouble\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 * 2"),
        (r"\btriple\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 * 3"),
        (r"\bsquare\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 ** 2"),
        (r"\bcube\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 ** 3"),
    )
    for pattern, replacement in spoken_operations:
        normalized = re.sub(pattern, replacement, lower)
        if normalized != lower:
            arithmetic_text = normalized
            break

    arithmetic_cue = re.search(
        r"\b(calculate|compute|evaluate|what is|what's|add|subtract|"
        r"multiply|divide|plus|minus|times|divided by|square root|sqrt|"
        r"raised to|power of|percent|half|double|triple|square of|cube of)\b",
        lower,
    )
    arithmetic_expression = re.search(r"\d\s*[+*/%^()]|[+*/%^()]\s*\d", lower)
    if arithmetic_cue or arithmetic_expression:
        result = faculty.evaluate(arithmetic_text)
        if result.ok:
            return ParietalAnswer(
                "arithmetic",
                str(result.value),
                [f"{result.expression} = {result.value}"],
            )
        if arithmetic_cue and result.error != "no arithmetic found":
            return ParietalAnswer("arithmetic", result.error, [result.expression], success=False)

    # Basic number sense compares magnitudes without pretending its
    # approximate faculty is exact arithmetic outside the subitizing range.
    comparison = re.search(
        r"\b(?:compare|which is larger|which is greater|which is smaller|"
        r"which is less|is (?:\d+(?:\.\d+)?) greater than|is (?:\d+(?:\.\d+)?) less than)\b.*?(-?\d+(?:\.\d+)?).*?(-?\d+(?:\.\d+)?)",
        lower,
    )
    if comparison:
        left, right = map(float, comparison.groups())
        judgment = faculty.compare(left, right)
        return ParietalAnswer(
            "number sense",
            f"{judgment.a:g} is {judgment.relation} than {judgment.b:g}",
            [f"confidence={judgment.confidence:.3f}", f"exact={judgment.exact}"],
        )

    # Symbolic algebra and calculus retain their native expression trees.
    if re.search(r"\b(simplify|expand|factor(?:ize)?|solve)\b", lower) and re.search(r"[a-zA-Z]", lower) and "[[" not in text:
        if re.search(r"\bsimplify\b", lower):
            return _answer(faculty.simplify(text), "algebra")
        if re.search(r"\bexpand\b", lower):
            return _answer(faculty.expand(text), "algebra")
        if re.search(r"\bsolve\b", lower):
            return _answer(faculty.solve(text), "algebra")
        return _answer(faculty.factor(text), "algebra")

    # Statistics accepts an explicit numeric collection; it never invents
    # a sample from surrounding prose.
    if re.search(r"\b(mean|average|median|variance|standard deviation|summarize)\b", lower):
        literal = _literal_after(text)
        if literal is not None:
            try:
                values = ast.literal_eval(literal[0])
                if not isinstance(values, (list, tuple)) or not values:
                    raise ValueError("provide a non-empty numeric list")
                numeric = [float(value) for value in values]
            except (SyntaxError, ValueError, TypeError) as error:
                return ParietalAnswer("statistics", str(error), success=False)
            summary = faculty.summarize(numeric)
            if "median" in lower:
                answer = f"median = {summary.median:g}"
            elif "variance" in lower:
                answer = f"population variance = {summary.variance:g}"
            elif "standard deviation" in lower:
                answer = f"population standard deviation = {summary.standard_deviation:g}"
            elif "summarize" in lower:
                answer = str(summary)
            else:
                answer = f"arithmetic mean = {faculty.mean(numeric):g}"
            return ParietalAnswer("statistics", answer, [f"Values: {numeric}"])

    # Trigonometry: explicit ratio and angle; angle parsing remains in trig.py.
    trig = re.search(r"\b(sin|cos|tan|asin|acos|atan)\s*(?:of\s*)?(.+?)\s*[?.!]*$", lower)
    if trig and any(token in lower for token in ("sin", "cos", "tan")):
        from .trig import ratio_text

        result = ratio_text(trig.group(1), trig.group(2).strip(" ()"))
        return _answer(result, "trigonometry")

    # Linear algebra: dispatch supported matrix operations through the
    # exact Fraction-based faculty. Each operand must be an explicit literal.
    if any(word in lower for word in ("determinant", "matrix rank", "rank of", "matrix", "linear system", "solve the system")) or "[[" in text:
        from .linear_algebra import determinant_text, matmul_text, rank_text, solve_text

        first = _literal_after(text)
        if first is None:
            return ParietalAnswer("linear algebra", "A matrix literal is required.", success=False)
        matrix, first_end = first
        if "multiply" in lower or "matrix product" in lower:
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra", "Two matrix literals are required for multiplication.", success=False
                )
            return _answer(matmul_text(matrix, second[0]), "linear algebra")
        if "solve" in lower or "linear system" in lower:
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra", "A right-hand-side vector literal is required.", success=False
                )
            return _answer(solve_text(matrix, second[0]), "linear algebra")
        if "determinant" in lower or re.search(r"\bdet\b", lower):
            return _answer(determinant_text(matrix), "linear algebra")
        if "rank" in lower:
            return _answer(rank_text(matrix), "linear algebra")

    # Abstract algebra: cycle notation is the input language of this faculty.
    has_group_words = re.search(r"\b(permutation|cycle|cycles|order of)\b", lower)
    has_cycle_literal = re.search(r"\(\s*\d+(?:\s+\d+)+\s*\)", lower)
    if has_group_words and has_cycle_literal:
        from .abstract_algebra import describe_text, order_text

        if "order" in lower:
            return _answer(order_text(text), "abstract algebra")
        return _answer(describe_text(text), "abstract algebra")

    # Real analysis: explicit point of approach is required for a limit.
    limit = re.search(
        r"\blimit\s+of\s+(.+?)\s+at\s+(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)\s*[?.!]*$",
        lower,
    )
    if limit:
        from .real_analysis import limit_text

        return _answer(limit_text(limit.group(1).strip(), float(limit.group(2))), "real analysis")

    # Complex analysis: description, integer powers, and roots of unity.
    if (
        "complex" in lower
        or re.search(r"\b(roots? of unity|polar form|argument of)\b", lower)
        or re.search(
            r"(?<![\w.])-?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?i\b",
            lower,
        )
    ):
        from .complex_analysis import describe_text, power_text, roots_unity_text

        unity = re.search(r"\b(\d+)\s*(?:th|st|nd|rd)?\s+roots? of unity", lower)
        if unity:
            return _answer(roots_unity_text(int(unity.group(1))), "complex analysis")
        power = re.search(r"([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?i)\s*\^\s*(-?\d+)", lower)
        if power:
            return _answer(power_text(power.group(1), int(power.group(2))), "complex analysis")
        value = re.search(r"([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?i)", lower)
        if value:
            return _answer(describe_text(value.group(1)), "complex analysis")
        return ParietalAnswer(
            "complex analysis",
            "A supported complex-number expression is required.",
            success=False,
        )

    # Number theory: integer operands are explicit and validated by the faculty.
    if re.search(
        r"\b(prime|primality|factor|factorization|prime factor|gcd|greatest common divisor)\b",
        lower,
    ):
        from .number_theory import factor_text, gcd_text, prime_text

        values = _integer_values(lower)
        if "gcd" in lower or "greatest common divisor" in lower:
            if len(values) < 2:
                return ParietalAnswer(
                    "number theory", "Two integer operands are required for gcd.", success=False
                )
            return _answer(gcd_text(values[-2], values[-1]), "number theory")
        if not values:
            return ParietalAnswer("number theory", "An integer operand is required.", success=False)
        if "factor" in lower:
            return _answer(factor_text(values[-1]), "number theory")
        return _answer(prime_text(values[-1]), "number theory")

    # Euclidean geometry: the faculty validates point counts and computes.
    if re.search(r"\b(distance|polygon area|area of|convex hull)\b", lower):
        from .geometry import area_text, distance_text, hull_text

        if "distance" in lower:
            return _answer(distance_text(text), "geometry")
        if "hull" in lower:
            return _answer(hull_text(text), "geometry")
        return _answer(area_text(text), "geometry")

    # Discrete structures: relation properties over an explicit set of pairs.
    if re.search(r"\b(relation|reflexive|symmetric|antisymmetric|transitive)\b", lower):
        from .discrete import relation_text

        literal = _literal_after(text)
        if literal is None:
            return ParietalAnswer(
                "discrete mathematics",
                "A relation literal such as {(1, 1), (1, 2)} is required.",
                success=False,
            )
        return _answer(relation_text(literal[0]), "discrete mathematics")

    # Combinatorics: choose/arrange/derange are unambiguous operation names.
    if re.search(r"\b(choose|combination|combinations|arrange|permutation|derangement)\b", lower):
        from .combinatorics import arrange_text, choose_text, derange_text

        choose = re.search(r"\bchoose\s+(-?\d+)\s+(?:from|of)\s+(-?\d+)", lower)
        if choose:
            k, n = map(int, choose.groups())
            return _answer(choose_text(n, k), "combinatorics")
        arrange = re.search(r"\barrange\s+(?:(-?\d+)\s+of\s+)?(-?\d+)", lower)
        if arrange:
            k_text, n_text = arrange.groups()
            return _answer(
                arrange_text(int(n_text), int(k_text) if k_text else None), "combinatorics"
            )
        derange = re.search(r"\bderangements?\s+of\s+(-?\d+)", lower)
        if derange:
            return _answer(derange_text(int(derange.group(1))), "combinatorics")
        return ParietalAnswer(
            "combinatorics",
            "State a supported choose, arrange, or derangement operation.",
            success=False,
        )

    # Finite topological spaces are represented by their open-set family.
    finite_topology = re.search(r"\btopology on (\d+) points?\b", lower)
    if finite_topology:
        literal = _literal_after(text)
        if literal is None:
            return ParietalAnswer(
                "topology", "An explicit family of open sets is required.", success=False
            )
        try:
            opens = ast.literal_eval(literal[0])
        except (SyntaxError, ValueError) as error:
            return ParietalAnswer("topology", str(error), success=False)
        return _answer(faculty.is_topology(int(finite_topology.group(1)), opens), "topology")

    # The propositional faculty consumes its documented structured tuple
    # representation; this is not a natural-language formula parser.
    if "tautology" in lower or "satisfiable" in lower:
        literal = _literal_after(text)
        if literal is None:
            return ParietalAnswer(
                "propositional proof", "A structured proposition tuple is required.", success=False
            )
        try:
            proposition = ast.literal_eval(literal[0])
        except (SyntaxError, ValueError) as error:
            return ParietalAnswer("propositional proof", str(error), success=False)
        if not isinstance(proposition, tuple):
            return ParietalAnswer(
                "propositional proof", "The proposition must be a tuple.", success=False
            )
        if "tautology" in lower:
            return _answer(faculty.is_tautology(proposition, words=text), "propositional proof")
        return _answer(faculty.is_satisfiable(proposition, words=text), "propositional proof")

    # Topology's Euler characteristic is directly expressible as V, E, F.
    euler = re.search(
        r"\beuler characteristic\b.*?\b(\d+)\s+vertices?\b.*?"
        r"(\d+)\s+edges?\b.*?\b(\d+)\s+faces?\b",
        lower,
    )
    if euler:
        from .topology import euler_surface

        return _answer(euler_surface(*(int(value) for value in euler.groups())), "topology")

    # Cayley-table axiom classification requires a literal square table.
    if re.search(r"\b(cayley table|classify.*structure|algebraic axioms)\b", lower):
        from .axioms import classify_text

        literal = _literal_after(text)
        if literal is None:
            return ParietalAnswer(
                "axioms", "A square Cayley-table literal is required.", success=False
            )
        try:
            table = ast.literal_eval(literal[0])
        except (SyntaxError, ValueError) as error:
            return ParietalAnswer("axioms", str(error), success=False)
        return _answer(classify_text(table), "axioms")

    return None
