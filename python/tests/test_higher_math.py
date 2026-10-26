"""Tests for the parietal faculty's higher math — algebra.py and
calculus.py, the endogenous symbolic layer.

The invariant under test: every result is derived by structure —
parsed into trees, manipulated by rule, rendered canonically. No
``eval()``, no external CAS: ``symbolic.py`` is the oracle that
verifies claims; these modules are the claims themselves, and each
test is a claim checked against ground truth computed by hand.
"""

from __future__ import annotations

import math

from parietal_lobe.math import (
    ParietalMath,
    looks_like_algebra,
    looks_like_calculus,
)
from parietal_lobe.math.algebra import (
    Add,
    Mul,
    Num,
    Pow,
    Var,
    evaluate,
    expand,
    factor_text,
    parse,
    parse_equation,
    render,
    simplify,
    simplify_text,
    solve,
    variables,
)
from parietal_lobe.math.calculus import (
    derivative_at_text,
    derivative_text,
    integral_between_text,
    integral_text,
    integrate_between,
    taylor,
)


def s(text: str) -> str:
    """Parse → simplify → render, the canonical round trip."""
    return render(simplify(parse(text)))


# ─── Parser ──────────────────────────────────────────────────────


def test_parse_builds_the_tree():
    assert parse("x + 2") == Add((Var("x"), Num(2.0)))


def test_implicit_multiplication():
    assert parse("2x") == Mul((Num(2.0), Var("x")))
    assert render(simplify(parse("x sin(x)"))) == "x*sin(x)"


def test_power_is_right_associative():
    assert parse("x^2^3") == Pow(Var("x"), Pow(Num(2.0), Num(3.0)))


def test_parse_equation():
    left, right = parse_equation("2*x + 3 = 7")
    assert left == Add((Mul((Num(2.0), Var("x"))), Num(3.0)))
    assert right == Num(7.0)


# ─── Simplify — canonical form ───────────────────────────────────


def test_like_terms_collect():
    assert s("x + x + 3") == "2*x + 3"
    assert s("2*x + 3*x") == "5*x"


def test_degree_orders_the_sum():
    assert s("3*x + x**3 + 3*x**2 + 1") == "x**3 + 3*x**2 + 3*x + 1"


def test_like_factors_become_powers():
    assert s("x * x * y**2") == "x**2*y**2"


def test_identities_erase():
    assert s("x + 0") == "x"
    assert s("x * 1") == "x"
    assert s("x**0 + 0*y") == "1"
    assert s("x**1") == "x"


def test_negative_terms_render_as_subtraction():
    assert s("x - 1 + x - 1") == "2*x - 2"
    assert s("3 - 2*x") == "-2*x + 3"


def test_constants_fold():
    assert s("2 + 3*4") == "14"
    assert s("sin(0)") == "0"


def test_factorial():
    assert s("5!") == "120"


def test_rational_coefficients_render_as_fractions():
    assert s("x/3") == "1/3*x"


# ─── Expand ──────────────────────────────────────────────────────


def test_expand_distributes():
    assert render(expand(parse("(x + 1)*(x - 1)"))) == "x**2 - 1"


def test_expand_binomial_power():
    assert render(expand(parse("(x + 1)**3"))) == "x**3 + 3*x**2 + 3*x + 1"


def test_expand_and_simplify_round_trip():
    assert render(expand(parse("(x+1)*(x-1) - x**2"))) == "-1"


# ─── Factor ──────────────────────────────────────────────────────


def test_factor_quadratic():
    assert factor_text("x**2 - 5*x + 6").output == "(x - 2)*(x - 3)"


def test_factor_cubic():
    assert factor_text("x**3 - 6*x**2 + 11*x - 6").output == "(x - 1)*(x - 2)*(x - 3)"


def test_factor_keeps_the_leading_coefficient():
    out = factor_text("2*x**2 - 8").output
    assert set(out.replace(" ", "").split("*")) == {"2", "(x+2)", "(x-2)"} or out.startswith("2*")
    assert abs(evaluate(parse(out), {"x": 0}) - (-8.0)) < 1e-9


def test_factor_irreducible_stays_whole():
    assert factor_text("x**2 + 1").output == "x**2 + 1"


# ─── Solve ───────────────────────────────────────────────────────


def test_solve_linear():
    r = solve("2*x + 3 = 7")
    assert r.ok and r.data["roots"] == [2.0]


def test_solve_quadratic_rational_roots():
    r = solve("x**2 - 5*x + 6 = 0")
    assert sorted(r.data["roots"]) == [2.0, 3.0]


def test_solve_quadratic_surds():
    r = solve("x**2 - 2 = 0")
    assert r.output == "x = sqrt(2) or x = -sqrt(2)"
    assert abs(r.data["roots"][0] - -math.sqrt(2)) < 1e-12


def test_solve_quadratic_shifted():
    r = solve("x**2 - 2*x - 1 = 0")
    assert r.output == "x = 1 + sqrt(2) or x = 1 - sqrt(2)"


def test_solve_double_root():
    r = solve("x**2 - 2*x + 1 = 0")
    assert r.data["roots"] == [1.0]


def test_solve_complex_roots():
    r = solve("x**2 + 1 = 0")
    assert r.ok and r.data["complex"] == (0.0, 1.0)


def test_solve_transcendental_numerically():
    r = solve("cos(x) = x")
    assert r.ok
    assert abs(r.data["roots"][0] - 0.7390851332) < 1e-6


def test_solve_higher_degree_numerically():
    r = solve("x**3 + x + 1 = 0")
    assert abs(r.data["roots"][0] - -0.6823278038) < 1e-6


def test_solve_contradiction():
    r = solve("2 = 3")
    assert r.ok and r.data["roots"] == []


def test_solve_identity():
    r = solve("x = x")
    assert r.ok and r.data["roots"] == "all"


# ─── Differentiation ─────────────────────────────────────────────


def test_power_rule():
    assert derivative_text("x**3").output == "3*x**2"


def test_constant_and_variable_rules():
    assert derivative_text("5").output == "0"
    assert derivative_text("x").output == "1"
    assert derivative_text("y", var="x").output == "0"


def test_sum_rule():
    assert derivative_text("x**2 + 3*x").output == "2*x + 3"


def test_product_rule():
    assert derivative_text("x**2 * sin(x)").output == "x**2*cos(x) + 2*x*sin(x)"


def test_chain_rule():
    assert derivative_text("exp(x**2)").output == "2*x*exp(x**2)"
    assert derivative_text("sin(2*x)").output == "2*cos(2*x)"
    assert derivative_text("ln(x**2)").output == "2*x**-1"


def test_trig_rules():
    assert derivative_text("sin(x)").output == "cos(x)"
    assert derivative_text("cos(x)").output == "-sin(x)"
    assert derivative_text("exp(x)").output == "exp(x)"


def test_derivative_at():
    r = derivative_at_text("x**2", "x", 3.0)
    assert r.ok and r.data["value"] == 6.0


# ─── Integration ─────────────────────────────────────────────────


def test_power_antiderivative():
    assert integral_text("x**2").output == "1/3*x**3 + C"


def test_trig_antiderivative_with_linear_inner():
    assert integral_text("sin(2*x)").output == "-1/2*cos(2*x) + C"


def test_exp_antiderivative():
    assert integral_text("exp(3*x)").output == "1/3*exp(3*x) + C"


def test_integral_linearity():
    out = integral_text("x + cos(x)").output
    assert "1/2*x**2" in out and "sin(x)" in out and out.endswith("+ C")


def test_integral_constant_factors():
    assert integral_text("3*x**2").output == "x**3 + C"


def test_integral_reciprocal():
    assert integral_text("x**-1").output == "ln(abs(x)) + C"


def test_no_closed_form_is_honest():
    # e^(x^2) has no elementary antiderivative — the table says so.
    r = integral_text("exp(x**2)")
    assert not r.ok and "no closed antiderivative" in r.error


def test_definite_integral_exact():
    r = integral_between_text("x**2", "x", 0.0, 3.0)
    assert r.data["exact"] and r.data["value"] == 9.0


def test_definite_integral_numeric_fallback():
    # exp(x**2) has no closed form — Simpson carries it.
    r = integral_between_text("exp(x**2)", "x", 0.0, 1.0)
    assert not r.data["exact"]
    assert abs(r.data["value"] - 1.4626517459) < 1e-6


def test_simpson_accuracy():
    v = integrate_between(parse("sin(x)"), "x", 0.0, math.pi)
    assert abs(v - 2.0) < 1e-9


# ─── Taylor ──────────────────────────────────────────────────────


def test_taylor_of_sin():
    out = render(taylor(parse("sin(x)"), "x", 0.0, 5))
    assert out == "1/120*x**5 - 1/6*x**3 + x"


def test_taylor_of_exp():
    out = render(taylor(parse("exp(x)"), "x", 0.0, 3))
    assert out == "1/6*x**3 + 1/2*x**2 + x + 1"


# ─── The parse tree is honest ────────────────────────────────────


def test_evaluate_binds_variables():
    assert evaluate(parse("2*x + 1"), {"x": 4}) == 9.0


def test_named_constants():
    assert abs(evaluate(parse("pi")) - math.pi) < 1e-15
    assert abs(evaluate(parse("sin(pi/2)")) - 1.0) < 1e-15


def test_variables_finds_the_unknown():
    assert variables(parse("x + pi + y")) == {"x", "y"}


def test_simplify_text_and_facade():
    assert simplify_text("x + x").output == "2*x"


# ─── The facade — detection and routing ──────────────────────────


def test_looks_like_algebra():
    assert looks_like_algebra("solve x^2 - 4 = 0")
    assert looks_like_algebra("expand (x+1)^3")
    assert looks_like_algebra("what is 2x + 3x")
    assert not looks_like_algebra("what is 7 + 5")


def test_looks_like_calculus():
    assert looks_like_calculus("derivative of x^2")
    assert looks_like_calculus("integrate sin(x)")
    assert looks_like_calculus("what is the slope of x**2")
    assert not looks_like_calculus("solve x = 2")


def test_facade_solve():
    pm = ParietalMath()
    r = pm.solve("solve 2x + 3 = 7")
    assert r.ok and r.data["roots"] == [2.0]


def test_facade_differentiate():
    pm = ParietalMath()
    assert pm.differentiate("the derivative of x**3").output == "3*x**2"


def test_facade_integrate():
    pm = ParietalMath()
    assert pm.integrate("integral of x**2").output == "1/3*x**3 + C"


def test_facade_integrate_between():
    pm = ParietalMath()
    r = pm.integrate_between("x**2 from 0 to 3", 0.0, 3.0)
    assert r.ok and r.data["value"] == 9.0


def test_facade_taylor():
    pm = ParietalMath()
    r = pm.taylor_series("sin(x)", at=0.0, order=3)
    assert r.ok and "x" in r.output


def test_no_eval_anywhere():
    """The faculty manipulates structure — ``eval``/``exec`` never."""
    import ast
    import inspect

    import parietal_lobe.math.algebra as alg
    import parietal_lobe.math.calculus as calc

    for mod in (alg, calc):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec")
