"""Arithmetic — exact calculation, parietal lobe.

Exact symbolic arithmetic is the cortical layer on top of the
approximate number system: where the ANS estimates, this computes.
Implemented as a recursive-descent evaluator over a token stream —
no ``eval()``, no Python expression execution, only numbers and the
operators defined here.

Two input dialects:

- **Digits and symbols**: ``7 + 5``, ``3 * (4 + 2)``, ``2 ** 10``
- **Words**: ``seven plus five``, ``nine times eight``, ``a hundred
  minus two``, ``three point five squared``

The evaluator deliberately stays small — the point is a real,
inspectable calculation faculty (tokenize → parse → evaluate), not
a stolen calculator. Each step is explicit so her arithmetic errors
are traceable the way a person's slips are.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CalcResult:
    """The outcome of a calculation."""

    expression: str          # normalized expression text ("7 + 5")
    value: float | int | None
    ok: bool
    error: str = ""


# ── Number words ─────────────────────────────────────────────────

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

_OPERATOR_WORDS = {
    "plus": "+", "add": "+",
    "minus": "-", "subtract": "-", "less": "-",
    "times": "*", "multiplied": "*", "multiply": "*", "of": "*",
    "x": "*",
    "divided": "/", "divide": "/", "over": "/", "quotient": "/",
    "mod": "%", "modulo": "%",
    "power": "**", "pow": "**",
}
_POSTFIX_WORDS = {"squared": "** 2", "cubed": "** 3", "square": "** 2", "cube": "** 3"}

_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"a", "an"}
_TOKEN_RE = re.compile(
    r"""
    \d+\.\d+|\d+          # numbers
    | \*\*|//             # multi-char operators first
    | [+\-*/x×÷^%()≈]     # symbol operators and parens
    | [a-zA-Z']+          # words
    | [.,!?]              # punctuation
    """,
    re.VERBOSE,
)


def _words_to_number(words: list[str]) -> tuple[float | None, int]:
    """Parse a run of number words into a value.

    Returns (value, words_consumed). Handles units, tens, and scales:
    ``seven`` → 7, ``twenty three`` → 23, ``two hundred`` → 200,
    ``a thousand`` → 1000, ``three point five`` → 3.5,
    ``half`` → 0.5.
    """
    total = 0.0
    current = 0.0
    i = 0
    saw = False
    while i < len(words):
        w = words[i]
        if w in ("a", "an"):
            current += 1
            saw = True
        elif w in _UNITS:
            current += _UNITS[w]
            saw = True
        elif w in _TENS:
            current += _TENS[w]
            saw = True
        elif w == "hundred":
            current = max(1.0, current) * 100
            saw = True
        elif w in ("thousand", "million"):
            total += max(1.0, current) * _SCALES[w]
            current = 0.0
            saw = True
        elif w == "half":
            current += 0.5
            saw = True
        elif w == "point":
            # Decimal tail: consume digit words after "point".
            i += 1
            digits = ""
            while i < len(words) and words[i] in _UNITS:
                digits += str(_UNITS[words[i]])
                i += 1
            if digits:
                current += float(f"0.{digits}")
                saw = True
            continue
        else:
            break
        i += 1
    if not saw:
        return None, 0
    return total + current, i


def tokenize(text: str) -> list[str]:
    """Convert natural text into an operator/number token stream.

    Word numbers are folded into numeric tokens; operator words into
    symbols. Anything that isn't math is skipped, so callers can pass
    whole questions ("what is 7 plus 5?") directly.
    """
    raw = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    tokens: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        # Numbers
        if re.fullmatch(r"\d+\.\d+|\d+", tok):
            tokens.append(tok)
            i += 1
            continue
        # Symbol operators
        if tok in ("**", "//"):
            tokens.append(tok)
            i += 1
            continue
        if tok in "+-*/%^()":
            tokens.append("**" if tok == "^" else tok)
            i += 1
            continue
        if tok in "x×÷":
            tokens.append("*" if tok in "x×" else "/")
            i += 1
            continue
        # "square root" is a sqrt phrase, not "square" postfix + "root"
        if tok == "square" and i + 1 < len(raw) and raw[i + 1] == "root":
            tokens.append("__sqrt__")
            i += 2
            continue
        # Postfix power words
        if tok in _POSTFIX_WORDS:
            tokens.extend(_POSTFIX_WORDS[tok].split())
            i += 1
            continue
        # sqrt prefix
        if tok in ("sqrt", "root"):
            tokens.append("__sqrt__")
            i += 1
            continue
        # unary minus ("negative five")
        if tok == "negative":
            tokens.append("-")
            i += 1
            continue
        # "to the power of" / "raised to"
        if tok == "power" or (
            tok == "to" and raw[i + 1 : i + 4] == ["the", "power", "of"]
        ):
            if tok == "to":
                i += 3
            tokens.append("**")
            i += 1
            continue
        # Operator words ("divided by", "multiplied by")
        if tok in _OPERATOR_WORDS:
            op = _OPERATOR_WORDS[tok]
            if tok in ("divided", "multiplied") and i + 1 < len(raw) and raw[i + 1] == "by":
                i += 1
            # "of" is only an operator between math tokens; skip it
            # otherwise ("what is the color of ...")
            if tok == "of" and not (
                tokens and (tokens[-1][0].isdigit() or tokens[-1] in ")*")
            ):
                i += 1
                continue
            tokens.append(op)
            i += 1
            continue
        # Number words — consume the longest run
        if tok in _NUMBER_WORDS or tok in ("point", "half"):
            value, consumed = _words_to_number(raw[i:])
            if consumed and value is not None:
                tokens.append(str(int(value) if float(value).is_integer() else value))
                i += consumed
                continue
        # Anything else — filler or unknown word; skip.
        i += 1
    return tokens


# ── Recursive-descent evaluator ──────────────────────────────────

_BINOPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "//": operator.floordiv,
    "%": operator.mod, "**": operator.pow,
}
_MAX_POW_ABS = 10_000  # guard absurd exponent sizes


class _Parser:
    """Recursive-descent parser for the arithmetic token stream."""

    def __init__(self, tokens: list[str]) -> None:
        """Initialize the parser at the first token."""
        self.toks = tokens
        self.pos = 0

    def peek(self) -> str | None:
        """Return the current token without consuming it."""
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def next(self) -> str | None:
        """Consume and return the current token, if any."""
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def parse(self) -> float:
        """Parse a complete expression and reject trailing tokens."""
        value = self._expr()
        if self.pos != len(self.toks):
            raise ValueError(f"trailing tokens: {self.toks[self.pos:]}")
        return value

    def _expr(self) -> float:
        """Parse addition and subtraction."""
        value = self._term()
        while self.peek() in ("+", "-"):
            op = self.next()
            assert op is not None
            value = _BINOPS[op](value, self._term())
        return value

    def _term(self) -> float:
        """Parse multiplication, division, floor division, and modulo."""
        value = self._factor()
        while self.peek() in ("*", "/", "//", "%"):
            op = self.next()
            assert op is not None
            rhs = self._factor()
            if op in ("/", "//", "%") and rhs == 0:
                raise ZeroDivisionError("division by zero")
            value = _BINOPS[op](value, rhs)
        return value

    def _factor(self) -> float:
        """Parse a right-associative exponentiation factor."""
        base = self._unary()
        if self.peek() == "**":
            self.next()
            exp = self._factor()  # right-associative
            if abs(exp) > _MAX_POW_ABS:
                raise ValueError("exponent too large")
            return _BINOPS["**"](base, exp)
        return base

    def _unary(self) -> float:
        """Parse unary signs and square-root operands."""
        tok = self.peek()
        if tok == "-":
            self.next()
            return -self._unary()
        if tok == "+":
            self.next()
            return self._unary()
        if tok == "__sqrt__":
            self.next()
            operand = self._unary()
            if operand < 0:
                raise ValueError("square root of a negative number")
            return operand ** 0.5
        return self._atom()

    def _atom(self) -> float:
        """Parse a literal number or parenthesized expression."""
        tok = self.next()
        if tok is None:
            raise ValueError("unexpected end of expression")
        if tok == "(":
            value = self._expr()
            if self.next() != ")":
                raise ValueError("missing closing parenthesis")
            return value
        if re.fullmatch(r"\d+\.\d+|\d+", tok):
            return float(tok)
        raise ValueError(f"unexpected token: {tok}")


def normalize(tokens: list[str]) -> str:
    """Render the token stream back as a clean expression string."""
    text = " ".join("^" if t == "**" else "sqrt" if t == "__sqrt__" else t for t in tokens)
    return (
        text.replace("( ", "(").replace(" )", ")").replace("sqrt (", "sqrt(")
    )


def evaluate(text: str) -> CalcResult:
    """Evaluate arithmetic embedded in natural text.

    Tokenizes (folding number/operator words), parses, and computes.
    Returns a CalcResult with ``ok=False`` when the text contains no
    parseable expression or the expression is invalid.
    """
    tokens = tokenize(text)
    # Need at least a number; a lone number is a valid expression.
    if not tokens or not any(t[0].isdigit() for t in tokens if t):
        return CalcResult("", None, False, "no arithmetic found")
    expr_text = normalize(tokens)
    try:
        value = _Parser(tokens).parse()
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return CalcResult(expr_text, None, False, str(e))
    if isinstance(value, complex):
        # Negative base to a fractional power (e.g. (-1) ** 0.5) —
        # this engine is real-valued; report instead of crashing on
        # float(complex).
        return CalcResult(
            expr_text, None, False, "result is not a real number"
        )
    result: float | int = int(value) if float(value).is_integer() else value
    return CalcResult(expr_text, result, True)
