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
        (r"\b(-?\d+(?:\.\d+)?)%\s*of\s*(-?\d+(?:\.\d+)?)[?.!]*$", r"(\1 / 100) * \2"),
        (r"\bhalf\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 / 2"),
        (r"\bdouble\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 * 2"),
        (r"\btriple\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 * 3"),
        (r"\bsquare\s+root\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"sqrt(\1)"),
        (r"(-?\d+(?:\.\d+)?)\s+to\s+the\s+power\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 ** \2"),
        (r"\bsquare\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 ** 2"),
        (r"\bcube\s+of\s+(-?\d+(?:\.\d+)?)[?.!]*$", r"\1 ** 3"),
    )
    for pattern, replacement in spoken_operations:
        normalized = re.sub(pattern, replacement, lower)
        if normalized != lower:
            arithmetic_text = normalized
            break

    matrix_language = any(
        term in lower
        for term in (
            "matrix",
            "matrices",
            "determinant",
            "matrix rank",
            "rank of",
            "linear system",
            "transpose",
            "inverse",
            "trace",
            "dot product",
            "inner product",
            "vector",
            "cross product",
        )
    )
    arithmetic_cue = re.search(
        r"\b(calculate|compute|evaluate|add|subtract|"
        r"multiply|divide|plus|minus|times|divided by|square root|sqrt|"
        r"raised to|power of|percent|fraction|half|halves|thirds?|quarters?|"
        r"double|triple|square of|cube of)\b",
        lower,
    )
    question_arithmetic_match = re.match(r"^(?:what is|what's)\s+(.+)$", lower)
    if question_arithmetic_match is not None:
        candidate = question_arithmetic_match.group(1)
        domain_terms = (
            "mean",
            "average",
            "median",
            "variance",
            "standard deviation",
            "matrix",
            "determinant",
            "rank",
            "limit",
            "sin",
            "cos",
            "tan",
            "complex",
            "prime",
            "factor",
            "topology",
            "permutation",
            "relation",
            "euler characteristic",
        )
        is_question_arithmetic = (
            bool(re.search(r"\d", candidate))
            and bool(
                re.search(
                    r"[+*/%^]|\b(plus|minus|times|divided|percent|power|sqrt)\b",
                    candidate,
                )
            )
            and not any(term in candidate for term in domain_terms)
        )
    else:
        is_question_arithmetic = False
    arithmetic_expression = not re.search(r"[a-zA-Z]", lower) and bool(
        re.search(r"\d\s*[+*/%^()]|[+*/%^()]\s*\d", lower)
    )
    symbolic_input = re.search(r"\b(?:[t-z]|pi|e)\b", lower)
    if (
        (
            arithmetic_cue
            or arithmetic_expression
            or is_question_arithmetic
            or arithmetic_text != text
        )
        and not symbolic_input
        and not matrix_language
    ):
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
        r"\b(?:compare|which is larger|which is greater|which is smaller|which is less)\b"
        r".*?(-?\d+(?:\.\d+)?).*?(-?\d+(?:\.\d+)?)",
        lower,
    )
    ordered_comparison = re.search(
        r"\bis\s+(-?\d+(?:\.\d+)?)\s+(greater|larger|less|smaller)\s+than\s+(-?\d+(?:\.\d+)?)\b",
        lower,
    )
    if comparison is not None or ordered_comparison is not None:
        if ordered_comparison is not None:
            left, _, right = ordered_comparison.groups()
            left_value, right_value = float(left), float(right)
            result_relation = (
                "greater"
                if left_value > right_value
                else "less"
                if left_value < right_value
                else "equal"
            )
            judgment = faculty.compare(left_value, right_value)
            answer = f"{left_value:g} is {result_relation} than {right_value:g}"
        else:
            assert comparison is not None
            left, right = map(float, comparison.groups())
            judgment = faculty.compare(left, right)
            answer = f"{judgment.a:g} is {judgment.relation} than {judgment.b:g}"
        return ParietalAnswer(
            "number sense",
            answer,
            [f"confidence={judgment.confidence:.3f}", f"exact={judgment.exact}"],
        )

    # Symbolic algebra and calculus retain their native expression trees.
    algebra_cue = re.search(r"\b(simplify|expand|factor(?:ize)?|solve)\b", lower)
    symbolic_factor = ("factor" not in lower and "factorize" not in lower) or re.search(
        r"\b[a-zA-Z]\b", lower.split("factor", 1)[-1]
    )
    if (
        algebra_cue
        and symbolic_factor
        and re.search(r"[a-zA-Z]", lower)
        and "[[" not in text
        and not re.search(r"\b(mod|modulo|congruence)\b", lower)
    ):
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

    # Calculus: keep symbolic differentiation and antiderivatives in the
    # native expression-tree faculty rather than sending them to arithmetic.
    if re.search(r"\b(differentiate|derivative|integral|integrate|antiderivative)\b", lower):
        if re.search(r"\b(differentiate|derivative)\b", lower):
            return _answer(faculty.differentiate(text), "calculus")
        return _answer(faculty.integrate(text), "calculus")

    # Trigonometry: explicit ratio and angle; angle parsing remains in trig.py.
    trig = re.search(r"\b(sin|cos|tan|asin|acos|atan)\s*(?:of\s*)?(.+?)\s*[?.!]*$", lower)
    if trig and "limit" not in lower and any(token in lower for token in ("sin", "cos", "tan")):
        from .trig import ratio_text

        trig_result = ratio_text(trig.group(1), trig.group(2).strip(" ()"))
        return _answer(trig_result, "trigonometry")

    # Angle conversion and triangle solving share the native trig faculty.
    conversion = re.search(
        r"\b(?:convert\s+)?(.+?)\s+to\s+(degrees?|deg|radians?|rad)\b[?.!]*$",
        lower,
    )
    if conversion and any(unit in lower for unit in ("degree", "deg", "radian", "rad")):
        from .trig import convert_text

        target = "degrees" if conversion.group(2).startswith(("deg", "degree")) else "radians"
        return _answer(convert_text(conversion.group(1).strip(), target), "trigonometry")
    if "triangle" in lower and "side" in lower:
        from .trig import triangle_text

        side_text = re.search(r"\bsides?\b(.*)$", lower)
        sides = (
            [float(value) for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", side_text.group(1))]
            if side_text
            else []
        )
        return _answer(triangle_text(sides), "trigonometry")

    # Linear algebra: dispatch supported matrix operations through the
    # exact Fraction-based faculty. Each operand must be an explicit literal.
    matrix_cue = any(
        word in lower
        for word in (
            "determinant",
            "matrix rank",
            "rank of",
            "matrix",
            "matrices",
            "linear system",
            "solve the system",
            "transpose",
            "inverse",
            "trace",
            "dot product",
            "inner product",
            "vector",
            "cross product",
        )
    )
    if matrix_cue or "[[" in text:
        from .linear_algebra import (
            determinant_text,
            inverse_text,
            matmul_text,
            matrix_add_text,
            matrix_scale_text,
            rank_text,
            solve_text,
            trace_text,
            transpose_text,
            vector_add_text,
            vector_cross_text,
            vector_dot_text,
        )

        first = _literal_after(text)
        if first is None:
            return ParietalAnswer("linear algebra", "A matrix literal is required.", success=False)
        matrix, first_end = first
        if "transpose" in lower:
            return _answer(transpose_text(matrix), "linear algebra")
        if "inverse" in lower:
            return _answer(inverse_text(matrix), "linear algebra")
        if "trace" in lower:
            return _answer(trace_text(matrix), "linear algebra")
        if "cross product" in lower:
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra",
                    "Two 3D vector literals are required for a cross product.",
                    success=False,
                )
            return _answer(vector_cross_text(matrix, second[0]), "linear algebra")
        if "dot product" in lower or "inner product" in lower:
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra",
                    "Two vector literals are required for a dot product.",
                    success=False,
                )
            return _answer(vector_dot_text(matrix, second[0]), "linear algebra")
        if "vector" in lower and ("add" in lower or "subtract" in lower):
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra",
                    "Two vector literals are required for vector addition.",
                    success=False,
                )
            return _answer(
                vector_add_text(matrix, second[0], subtract="subtract" in lower),
                "linear algebra",
            )
        if re.search(r"\bmatri(?:x|ces)\b", lower) and ("add" in lower or "subtract" in lower):
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra",
                    "Two matrix literals are required for matrix addition.",
                    success=False,
                )
            return _answer(
                matrix_add_text(matrix, second[0], subtract="subtract" in lower),
                "linear algebra",
            )
        if ("scale" in lower or "scalar" in lower) and "matrix" in lower:
            scalar = re.search(r"\bby\s+(-?\d+(?:/\d+)?(?:\.\d+)?)\b", lower)
            if scalar is None:
                return ParietalAnswer(
                    "linear algebra", "An explicit rational scalar is required.", success=False
                )
            return _answer(matrix_scale_text(matrix, scalar.group(1)), "linear algebra")
        if "multiply" in lower or "matrix product" in lower:
            second = _literal_after(text, first_end)
            if second is None:
                return ParietalAnswer(
                    "linear algebra",
                    "Two matrix literals are required for multiplication.",
                    success=False,
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
        from .abstract_algebra import describe_text as describe_permutation_text
        from .abstract_algebra import order_text

        if "order" in lower:
            return _answer(order_text(text), "abstract algebra")
        return _answer(describe_permutation_text(text), "abstract algebra")

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
        from .complex_analysis import describe_text as describe_complex_text
        from .complex_analysis import power_text, roots_unity_text

        root_match = re.search(
            r"\b(\d+)(?:st|nd|rd|th)\s+roots?\s+of\s+(.+?)[?.!]*$",
            lower,
        )
        if root_match and "unity" not in lower:
            from .complex_analysis import roots_text

            return _answer(
                roots_text(root_match.group(2), int(root_match.group(1))),
                "complex analysis",
            )
        named_root = re.search(
            r"\b(square|cube|fourth)\s+roots?\s+of\s+(.+?)[?.!]*$",
            lower,
        )
        if named_root:
            from .complex_analysis import roots_text

            degree = {"square": 2, "cube": 3, "fourth": 4}[named_root.group(1)]
            return _answer(roots_text(named_root.group(2), degree), "complex analysis")

        unity = re.search(r"\b(\d+)\s*(?:th|st|nd|rd)?\s+roots? of unity", lower)
        if unity:
            return _answer(roots_unity_text(int(unity.group(1))), "complex analysis")
        power = re.search(r"([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?i)\s*\^\s*(-?\d+)", lower)
        if power:
            return _answer(power_text(power.group(1), int(power.group(2))), "complex analysis")
        value = re.search(r"([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?i)", lower)
        if value:
            return _answer(describe_complex_text(value.group(1)), "complex analysis")
        return ParietalAnswer(
            "complex analysis",
            "A supported complex-number expression is required.",
            success=False,
        )

    # Modular arithmetic is distinct from polynomial equation solving.
    congruence = re.search(
        r"\bsolve\s+(-?\d+)\s*x\s*[≡=]\s*(-?\d+)\s*"
        r"\(\s*mod(?:ulo)?\s+(-?\d+)\s*\)",
        lower,
    )
    if congruence:
        from .number_theory import solve_congruence_text

        return _answer(
            solve_congruence_text(*(int(value) for value in congruence.groups())),
            "number theory",
        )
    if re.search(r"\b(totient|euler.?s phi)\b", lower):
        from .number_theory import totient

        values = _integer_values(lower)
        if not values:
            return ParietalAnswer(
                "number theory", "An integer operand is required for totient.", success=False
            )
        n = values[-1]
        if n < 1:
            return ParietalAnswer(
                "number theory", "Totient is defined here for positive integers.", success=False
            )
        totient_value = totient(n)
        return ParietalAnswer(
            "number theory",
            f"φ({n}) = {totient_value}",
            [f"φ({n}) = {totient_value}"],
        )
    if "chinese remainder" in lower or re.search(r"\bcrt\b", lower):
        from .number_theory import crt_text

        literal = _literal_after(text)
        if literal is None:
            return ParietalAnswer(
                "number theory",
                "Provide congruence pairs as [(residue, modulus), ...].",
                success=False,
            )
        try:
            pairs = ast.literal_eval(literal[0])
            if not isinstance(pairs, (list, tuple)) or not pairs:
                raise ValueError("at least one congruence pair is required")
            if any(
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in pair)
                for pair in pairs
            ):
                raise ValueError("each congruence must be an integer (residue, modulus) pair")
        except (SyntaxError, ValueError) as error:
            return ParietalAnswer("number theory", str(error), success=False)
        return _answer(crt_text([tuple(pair) for pair in pairs]), "number theory")

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

    # Point-in-polygon uses two explicit coordinate literals: one point,
    # followed by the polygon vertices.
    if re.search(r"\b(inside|outside)\b", lower) and "polygon" in lower:
        from .geometry import inside_text

        point = _literal_after(text)
        polygon = _literal_after(text, point[1]) if point else None
        if point is None or polygon is None:
            return ParietalAnswer(
                "geometry",
                "Provide a point literal followed by at least three polygon vertices.",
                success=False,
            )
        return _answer(inside_text(point[0], polygon[0]), "geometry")

    # Euclidean geometry: the faculty validates point counts and computes.
    if re.search(r"\b(distance|polygon area|area of|convex hull)\b", lower):
        from .geometry import area_text, distance_text, hull_text

        if "distance" in lower:
            return _answer(distance_text(text), "geometry")
        if "hull" in lower:
            return _answer(hull_text(text), "geometry")
        return _answer(area_text(text), "geometry")

    # Graph connectivity consumes explicit edge pairs; omitted vertices are
    # not inferred from the surrounding prose.
    if "graph" in lower and re.search(r"\b(connected|connectivity)\b", lower):
        from .discrete import connected_text

        literal = _literal_after(text)
        if literal is None:
            return ParietalAnswer(
                "discrete mathematics",
                "An explicit edge-pair literal is required.",
                success=False,
            )
        return _answer(connected_text(literal[0]), "discrete mathematics")

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

    if re.search(r"\bstirling(?: number)?\b", lower):
        from .combinatorics import stirling2

        values = _integer_values(lower)
        if len(values) < 2:
            return ParietalAnswer(
                "combinatorics", "Two integers n and k are required for S(n, k).", success=False
            )
        n, k = values[-2:]
        if n < 0 or k < 0 or k > n:
            return ParietalAnswer(
                "combinatorics", "Require integers with 0 ≤ k ≤ n.", success=False
            )
        stirling_value = stirling2(n, k)
        return ParietalAnswer("combinatorics", f"S({n}, {k}) = {stirling_value}")
    if "stars and bars" in lower:
        from .combinatorics import stars_bars

        values = _integer_values(lower)
        if len(values) < 2:
            return ParietalAnswer(
                "combinatorics",
                "Specify the number of identical objects and the number of bins.",
                success=False,
            )
        objects, bins = values[-2:]
        if objects < 0 or bins < 1:
            return ParietalAnswer(
                "combinatorics", "Require objects ≥ 0 and bins ≥ 1.", success=False
            )
        stars_bars_value = stars_bars(objects, bins)
        return ParietalAnswer(
            "combinatorics",
            f"C({objects + bins - 1}, {bins - 1}) = {stars_bars_value}",
        )

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
