"""The parietal arithmetic faculty — digits and number-words folded
into one token stream, parsed and computed without eval()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parietal_lobe.math import arithmetic


def test_digit_and_symbol_dialect():
    assert arithmetic.evaluate("7 + 5").value == 12
    assert arithmetic.evaluate("3 * (4 + 2)").value == 18
    assert arithmetic.evaluate("2 ** 10").value == 1024
    assert arithmetic.evaluate("(2 + 3) * 4 - 1").value == 19
    assert arithmetic.evaluate("10 // 3").value == 3
    assert arithmetic.evaluate("10 % 3").value == 1


def test_word_dialect():
    assert arithmetic.evaluate("seven plus five").value == 12
    assert arithmetic.evaluate("nine times eight").value == 72
    assert arithmetic.evaluate("a hundred minus two").value == 98
    assert arithmetic.evaluate("three point five squared").value == 12.25
    assert arithmetic.evaluate("two to the power of ten").value == 1024
    assert arithmetic.evaluate("what is 7 plus 5?").value == 12
    assert arithmetic.evaluate("twenty three plus one").value == 24
    assert arithmetic.evaluate("half of eight").value == 4


def test_precedence_and_associativity():
    assert arithmetic.evaluate("2 + 3 * 4").value == 14
    # Power is right-associative: 2 ** 3 ** 2 = 2 ** 9 = 512.
    assert arithmetic.evaluate("2 ** 3 ** 2").value == 512
    assert arithmetic.evaluate("-3 + 5").value == 2
    assert arithmetic.evaluate("negative five plus five").value == 0


def test_roots_and_postfix_powers():
    assert arithmetic.evaluate("sqrt 16").value == 4
    assert arithmetic.evaluate("square root of nine").value == 3
    assert arithmetic.evaluate("root 9").value == 3
    assert arithmetic.evaluate("five squared").value == 25
    assert arithmetic.evaluate("two cubed").value == 8


def test_failures_are_worded():
    assert arithmetic.evaluate("hello world").error == "no arithmetic found"
    assert arithmetic.evaluate("ten divided by zero").error == "division by zero"
    assert arithmetic.evaluate("(2 + 3").error == "missing closing parenthesis"
    assert arithmetic.evaluate("3 +").error == "unexpected end of expression"
    assert arithmetic.evaluate("7 plus").error == "unexpected end of expression"
    assert arithmetic.evaluate("2 ** 100001").error == "exponent too large"
    assert arithmetic.evaluate("sqrt of negative four").error == "square root of a negative number"


def test_normalized_expression_reads_clean():
    assert arithmetic.evaluate("seven plus five").expression == "7 + 5"
    assert arithmetic.evaluate("2 to the power of 10").expression == "2 ^ 10"
    assert arithmetic.evaluate("sqrt( nine )").expression == "sqrt(9)"


def test_lone_number_is_a_valid_expression():
    assert arithmetic.evaluate("42").value == 42
    assert arithmetic.evaluate("give me one hundred").value == 100
