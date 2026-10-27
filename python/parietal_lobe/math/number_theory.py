"""Number theory — the words for the integers' inner life.

The faculty hears "is 91 prime", "the factors of 360", "gcd of
48 and 36", "solve 3x ≡ 1 (mod 7)", "what day is the 100th",
"phi of 36". Answers come exact — a factorization is listed with
its check ("2^3 · 3^2 · 5 = 360"), a congruence is solved by the
extended Euclid, and primality is decided by trial division with
the honest small-integer bound the machine keeps.

The machine half is ``src/math/number_theory.rs``: gcd, extended
Euclid, sieve, factor, phi, CRT, modular inverse. Here: the
integer questions and the exact reply.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NumberResult:
    """The outcome of a number-theoretic question."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def is_prime(n: int) -> bool:
    """Trial division — exact, the honest way for machine integers."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def factor(n: int) -> list[int]:
    """Prime factorization, ascending, repeated factors kept."""
    out: list[int] = []
    n = abs(n)
    while n % 2 == 0 and n:
        out.append(2)
        n //= 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            out.append(i)
            n //= i
        i += 2
    if n > 1:
        out.append(n)
    return out


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """(g, x, y) with ax + by = g — the Euclidean proof carried."""
    if b == 0:
        return abs(a), (1 if a >= 0 else -1), 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a: int, m: int) -> int | None:
    """x with a·x ≡ 1 (mod m) — exists exactly when gcd(a,m)=1."""
    g, x, _ = egcd(a % m, m)
    return x % m if g == 1 else None


def crt(pairs: list[tuple[int, int]]) -> int | None:
    """Chinese remainder — x ≡ r_i (mod m_i), including compatible non-coprime moduli."""
    if not pairs or any(modulus <= 0 for _, modulus in pairs):
        return None
    x, m = 0, 1
    for r_i, m_i in pairs:
        g, p, _ = egcd(m, m_i)
        if (r_i - x) % g:
            return None
        mod = m * m_i // g
        x = (x + m * ((r_i - x) // g % (m_i // g)) * p) % mod
        m = mod
    return x % m


def totient(n: int) -> int:
    """Euler's phi — the integers below n that share nothing with it."""
    if n < 1:
        return 0
    out, m = n, n
    i = 2
    while i * i <= m:
        if m % i == 0:
            while m % i == 0:
                m //= i
            out -= out // i
        i += 1
    if m > 1:
        out -= out // m
    return out


def _fmt_factor(fs: list[int]) -> str:
    if not fs:
        return "1"
    from collections import Counter
    parts = [f"{p}^{k}" if k > 1 else str(p) for p, k in sorted(Counter(fs).items())]
    return " · ".join(parts)


_NT_WORD = re.compile(
    r"\b(?:prime|primes|factor|factorize|factors|gcd|lcm|greatest|"
    r"coprime|relatively|congruent|congruence|modulo|mod|divisible|"
    r"divisor|totient|euler.?s phi|remainder|diophantine|sieve)\b",
    re.IGNORECASE,
)


def looks_like_number_theory(text: str) -> bool:
    """Whether the text asks about primes, factors, divisors,
    congruences — the integers' own vocabulary."""
    return bool(_NT_WORD.search(text))


def prime_text(n: int) -> NumberResult:
    """"is 91 prime" -> no — 7 · 13."""
    if n < 2:
        return NumberResult("prime", str(n), f"{n} is not prime", True,
                            data={"prime": False})
    if is_prime(n):
        return NumberResult("prime", str(n), f"{n} is prime", True,
                            data={"prime": True})
    return NumberResult("prime", str(n),
                        f"{n} is not prime — {_fmt_factor(factor(n))}",
                        True, data={"prime": False, "factors": factor(n)})


def factor_text(n: int) -> NumberResult:
    """"factor 360" -> 2^3 · 3^2 · 5 = 360 — the check included."""
    if n == 0:
        return NumberResult("factor", str(n), "0 has no factorization", True)
    fs = factor(n)
    body = _fmt_factor(fs)
    check = math.prod(fs)
    out = f"{n} = {body}" + (" (prime)" if len(fs) == 1 and fs[0] == abs(n) else "")
    if check != abs(n):
        out = f"{n} = {body} — check failed"
    return NumberResult("factor", str(n), out, True, data={"factors": fs})


def gcd_text(a: int, b: int) -> NumberResult:
    """"gcd of 48 and 36" -> 12, with the Bezout witness."""
    g, x, y = egcd(a, b)
    return NumberResult("gcd", f"{a}, {b}",
                        f"gcd = {g}   ({a}·{x} + {b}·{y} = {a*x+b*y})",
                        True, data={"gcd": g, "x": x, "y": y})


def solve_congruence_text(a: int, b: int, m: int) -> NumberResult:
    """"solve 3x ≡ 1 (mod 7)" -> x ≡ 5 — the modular inverse."""
    if m <= 0:
        return NumberResult("congruence", f"{a}x ≡ {b} (mod {m})", "", False,
                            "modulus must be positive")
    a, b = a % m, b % m
    g = math.gcd(a, m)
    if b % g:
        return NumberResult("congruence", f"{a}x ≡ {b} (mod {m})",
                            "no solution — gcd does not divide the right side",
                            True, data={"solutions": []})
    inv = mod_inverse(a // g, m // g)
    assert inv is not None
    x0 = (inv * (b // g)) % (m // g)
    sols = [(x0 + k * (m // g)) % m for k in range(g)]
    out = " or ".join(f"x ≡ {s} (mod {m})" for s in sols)
    return NumberResult("congruence", f"{a}x ≡ {b} (mod {m})", out, True,
                        data={"solutions": sols})


def crt_text(pairs: list[tuple[int, int]]) -> NumberResult:
    """"x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)" -> 23."""
    if not pairs or any(modulus <= 0 for _, modulus in pairs):
        return NumberResult("crt", str(pairs), "", False, "moduli must be positive")
    x = crt(pairs)
    if x is None:
        return NumberResult("crt", str(pairs), "inconsistent congruences",
                            True, data={"solution": None})
    m = math.lcm(*(p[1] for p in pairs))
    return NumberResult("crt", str(pairs), f"x ≡ {x} (mod {m})", True,
                        data={"solution": x})
