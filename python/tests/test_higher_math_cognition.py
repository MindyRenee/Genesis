"""End-to-end tests for cognitive routing into native parietal mathematics."""

from __future__ import annotations

import pytest

from frontal_lobe.reasoning.math_reasoning import try_math


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("sin of pi/6", "1/2"),
        ("differentiate x**3", "3*x**2"),
        ("integral of x**2", "1/3*x**3"),
        ("simplify x + x", "2*x"),
        ("mean of [2, 4, 6]", "4"),
        ("determinant of [[1, 2], [3, 4]]", "-2"),
        ("order of permutation (1 2 3)", "3"),
        ("limit of sin(x)/x at 0", "1"),
        ("describe complex number 3+4i", "|z| = 5"),
        ("is 91 prime", "not prime"),
        ("factor 360", "360 ="),
        ("Euler characteristic of a surface with 4 vertices, 6 edges, 4 faces", "2"),
        ("distance between (0,0) and (3,4)", "5"),
        ("choose 3 from 10", "120"),
    ],
)
def test_cognitive_entry_point_reaches_native_parietal_faculty(question: str, expected: str):
    result = try_math(question)
    assert result is not None
    assert result.success
    assert expected in result.answer


def test_cognitive_entry_point_preserves_domain_steps():
    result = try_math("determinant of [[1, 2], [3, 4]]")
    assert result is not None
    assert result.result_type == "linear algebra"
    assert result.steps
    assert "linear algebra" in result.steps[0]


def test_native_parietal_router_does_not_claim_ordinary_prose():
    result = try_math("tell me about the history of mathematics")
    assert result is None


def test_existing_arithmetic_path_remains_available():
    result = try_math("what is 2 + 2")
    assert result is not None
    assert result.success
    assert result.answer == "4"
