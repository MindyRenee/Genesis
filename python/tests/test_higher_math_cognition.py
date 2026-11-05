"""End-to-end tests for cognitive routing into native parietal mathematics."""

from __future__ import annotations

import pytest
from frontal_lobe.reasoning.math_reasoning import try_math
from parietal_lobe.math.linear_algebra import solve_text

@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("sin of pi/6", "1/2"),
        ("what is 2 + 2", "4"),
        ("what is 12 minus 5", "7"),
        ("what is 3/4 + 1/4", "1"),
        ("what is one half", "0.5"),
        ("one half plus one quarter", "0.75"),
        ("three quarters of 20", "15"),
        ("two and a half", "2.5"),
        ("what is 2 to the power of 3", "8"),
        ("square root of 81", "9"),
        ("what is -3 + 8", "5"),
        ("what is 2 * (3 + 4)", "14"),
        ("seven plus five", "12"),
        ("25% of 80", "20"),
        ("add 2 and 3", "5"),
        ("subtract 3 from 10", "7"),
        ("multiply 4 by 5", "20"),
        ("divide 20 by 4", "5"),
        ("25 percent of 80", "20"),
        ("square of 5", "25"),
        ("cube of 3", "27"),
        ("compare 3 and 9", "3 is less than 9"),
        ("is 5 greater than 3", "5 is greater than 3"),
        ("matrix product [[1, 2], [3, 4]] and [[5, 6], [7, 8]]", "[[19, 22], [43, 50]]"),
        ("add matrices [[1, 2], [3, 4]] and [[5, 6], [7, 8]]", "[[6, 8], [10, 12]]"),
        ("subtract matrices [[5, 6], [7, 8]] and [[1, 2], [3, 4]]", "[[4, 4], [4, 4]]"),
        ("scale matrix [[1, 2], [3, 4]] by 3", "[[3, 6], [9, 12]]"),
        ("calculate cross product of [1, 0, 0] and [0, 1, 0]", "(0, 0, 1)"),
        ("transpose [[1, 2, 3], [4, 5, 6]]", "[[1, 4], [2, 5], [3, 6]]"),
        ("inverse of [[4, 7], [2, 6]]", "[[3/5, -7/10], [-1/5, 2/5]]"),
        ("trace of [[1, 2], [3, 4]]", "5"),
        ("dot product of [1, 2, 3] and [4, 5, 6]", "32"),
        ("add vectors [1, 2] and [3, 4]", "(4, 6)"),
        ("subtract vectors [5, 7] and [2, 3]", "(3, 4)"),
        ("solve [[2, 1], [1, -1]] x = [5, 1]", "x = (2, 1)"),
        ("differentiate x**3", "3*x**2"),
        ("integral of x**2", "1/3*x**3"),
        ("simplify x + x", "2*x"),
        ("solve 2*x + 3 = 7", "x = 2"),
        ("what is the mean of [2, 4, 6]", "4"),
        ("mean of [2, 4, 6]", "4"),
        ("determinant of [[1, 2], [3, 4]]", "-2"),
        ("what is the determinant of [[1, 2], [3, 4]]", "-2"),
        ("order of permutation (1 2 3)", "3"),
        ("limit of sin(x)/x at 0", "1"),
        ("describe complex number 3+4i", "|z| = 5"),
        ("is 91 prime", "not prime"),
        ("factor 360", "360 ="),
        ("Euler characteristic of a surface with 4 vertices, 6 edges, 4 faces", "2"),
        ("topology on 2 points [[], [0], [1], [0, 1]]", "a topology"),
        ("is this a tautology: ('imp', ('atom', 'P'), ('atom', 'P'))", "tautology"),
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

def test_linear_algebra_rejects_singular_inverse():
    result = try_math("inverse of [[1, 2], [2, 4]]")
    assert result is not None
    assert not result.success
    assert "nonsingular square matrix" in result.answer

def test_linear_algebra_rejects_dimension_mismatch():
    result = try_math("dot product of [1, 2] and [3, 4, 5]")
    assert result is not None
    assert not result.success
    assert "dimensions must agree" in result.answer

def test_matrix_addition_rejects_dimension_mismatch():
    result = try_math("add matrices [[1, 2]] and [[1], [2]]")
    assert result is not None
    assert not result.success
    assert "dimensions must agree" in result.answer

def test_cross_product_requires_three_dimensions():
    result = try_math("cross product of [1, 2] and [3, 4]")
    assert result is not None
    assert not result.success
    assert "3D vectors" in result.answer

@pytest.mark.parametrize("right_hand_side", ["5", "[[5], [1]]", "{'x': 5}"])
def test_linear_system_rejects_non_vector_right_hand_side(right_hand_side: str):
    result = solve_text("[[2, 1], [1, -1]]", right_hand_side)
    assert not result.ok
    assert result.error
