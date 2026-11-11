"""Math lab — statistics, number theory, linear algebra, probability.

Stdlib `statistics` for descriptive work, exact integer arithmetic
for number theory (Miller–Rabin deterministic under 2^64), plain
Gaussian elimination for small matrices. Exact answers, stated
domains — no silent approximations.

Docs: https://docs.python.org/3/library/statistics.html ·
https://docs.python.org/3/library/math.html
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from tools.framework import ToolResult


def _parse_numbers(raw: str) -> list[float]:
    parts = raw.replace(",", " ").split()
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"not a number list: {e}") from e


def statistics_lab(data: str, action: str = "describe") -> ToolResult:
    """Descriptive statistics over a comma/space separated list.

    ``action``: describe (n, mean, median, stdev, min, max),
    correlation/regression (``data`` = 'x1 y1, x2 y2, …' pairs —
    returns Pearson r, slope, intercept). Sample statistics
    (n−1 denominators), as statistics.stdev documents.
    https://docs.python.org/3/library/statistics.html
    """
    try:
        if action in ("correlation", "regression"):
            pairs: list[tuple[float, float]] = []
            for chunk in data.split(","):
                xy = _parse_numbers(chunk)
                if len(xy) != 2:
                    return ToolResult(success=False, error="pairs need 'x y, x y, …'")
                pairs.append((xy[0], xy[1]))
            if len(pairs) < 2:
                return ToolResult(success=False, error="need ≥2 pairs")
            xs_t, ys_t = zip(*pairs, strict=True)
            r = statistics.correlation(xs_t, ys_t)
            slope, intercept = statistics.linear_regression(xs_t, ys_t)
            return ToolResult(
                success=True,
                output=f"r={r:.6g}, y = {slope:.6g}x + {intercept:.6g} (n={len(pairs)})",
                data={"r": r, "slope": slope, "intercept": intercept, "n": len(pairs)},
            )
        xs = _parse_numbers(data)
        if not xs:
            return ToolResult(success=False, error="empty data")
        out: dict[str, Any] = {
            "n": len(xs),
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "min": min(xs),
            "max": max(xs),
        }
        if len(xs) >= 2:
            out["stdev"] = statistics.stdev(xs)
            out["variance"] = statistics.variance(xs)
        return ToolResult(
            success=True,
            output=", ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()
            ),
            data=out,
        )
    except (ValueError, statistics.StatisticsError) as e:
        return ToolResult(success=False, error=str(e))


def _is_prime(n: int) -> bool:
    """Deterministic Miller–Rabin for n < 2^64."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def number_theory(action: str, n: int = 0, k: int = 0, m: int = 0) -> ToolResult:
    """Exact integer mathematics.

    ``action``: is_prime (n; deterministic Miller–Rabin < 2^64),
    factor (n; trial division — fine to ~10^12), primes (n → all
    primes < n via sieve), gcd (n, k), lcm, modpow (n^k mod m),
    modinv (n mod m), egcd (n, k → x, y, g with xn+yk=g).
    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
    """
    try:
        if action == "is_prime":
            out = _is_prime(abs(n))
            return ToolResult(
                success=True, output=f"{n} {'is' if out else 'is not'} prime", data={"prime": out}
            )
        if action == "factor":
            n0 = abs(n)
            if n0 < 2:
                return ToolResult(success=False, error="factor needs |n| ≥ 2")
            f: dict[int, int] = {}
            d, v = 2, n0
            while d * d <= v:
                while v % d == 0:
                    f[d] = f.get(d, 0) + 1
                    v //= d
                d += 1 if d == 2 else 2
            if v > 1:
                f[v] = f.get(v, 0) + 1
            pretty = " · ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
            return ToolResult(success=True, output=f"{n0} = {pretty}", data={"factors": f})
        if action == "primes":
            if n < 2 or n > 10_000_000:
                return ToolResult(success=False, error="primes needs 2 ≤ n ≤ 10^7")
            sieve = bytearray(b"\x01") * (n + 1)
            sieve[:2] = b"\x00\x00"
            for i in range(2, math.isqrt(n) + 1):
                if sieve[i]:
                    sieve[i * i :: i] = b"\x00" * (((n - i * i) // i) + 1)
            ps = [i for i, v in enumerate(sieve) if v]
            return ToolResult(
                success=True, output=str(ps[:500]), data={"count": len(ps), "primes": ps[:500]}
            )
        if action == "gcd":
            return ToolResult(
                success=True, output=str(math.gcd(n, k)), data={"value": math.gcd(n, k)}
            )
        if action == "lcm":
            return ToolResult(
                success=True, output=str(math.lcm(n, k)), data={"value": math.lcm(n, k)}
            )
        if action == "modpow":
            return ToolResult(success=True, output=str(pow(n, k, m)), data={"value": pow(n, k, m)})
        if action == "modinv":
            return ToolResult(
                success=True, output=str(pow(n, -1, m)), data={"value": pow(n, -1, m)}
            )
        if action == "egcd":
            g, x, y = _egcd(n, k)
            return ToolResult(
                success=True,
                output=f"{x}·{n} + {y}·{k} = {g}",
                data={"x": x, "y": y, "g": g},
            )
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: is_prime, factor, "
            "primes, gcd, lcm, modpow, modinv, egcd",
        )
    except (ValueError, OverflowError) as e:
        return ToolResult(success=False, error=str(e))


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x, y = _egcd(b, a % b)
    return (g, y, x - (a // b) * y)


def matrix(action: str, a: str, b: str = "") -> ToolResult:
    """Small dense-matrix computation over JSON-style row lists.

    ``a``/``b``: '[[1,2],[3,4]]'. ``action``: multiply, transpose,
    det, inverse, solve (a·x = b-column). Gaussian elimination with
    partial pivoting; singular matrices report honestly.
    """
    import ast as _ast

    def _mat(raw: str) -> list[list[float]]:
        obj = _ast.literal_eval(raw)
        rows = [[float(x) for x in row] for row in obj]
        if not rows or len({len(r) for r in rows}) != 1:
            raise ValueError("ragged or empty matrix")
        return rows

    try:
        A = _mat(a)
        if action == "transpose":
            out = [list(r) for r in zip(*A, strict=True)]
            return ToolResult(success=True, output=str(out), data={"result": out})
        if action == "multiply":
            B = _mat(b)
            if len(A[0]) != len(B):
                return ToolResult(success=False, error="inner dimensions differ")
            out = [
                [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
                for i in range(len(A))
            ]
            return ToolResult(success=True, output=str(out), data={"result": out})
        if action in ("det", "inverse", "solve"):
            n = len(A)
            if any(len(r) != n for r in A):
                return ToolResult(success=False, error="needs a square matrix")
            aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
            det = 1.0
            for col in range(n):
                piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
                if abs(aug[piv][col]) < 1e-15:
                    return ToolResult(success=False, error="singular matrix")
                if piv != col:
                    aug[col], aug[piv] = aug[piv], aug[col]
                    det = -det
                det *= aug[col][col]
                for r in range(n):
                    if r != col and aug[r][col] != 0.0:
                        f = aug[r][col] / aug[col][col]
                        aug[r] = [x - f * y for x, y in zip(aug[r], aug[col], strict=True)]
            if action == "det":
                return ToolResult(success=True, output=f"{det:g}", data={"value": det})
            inv = [[aug[i][n + j] / aug[i][i] for j in range(n)] for i in range(n)]
            if action == "inverse":
                return ToolResult(success=True, output=str(inv), data={"result": inv})
            B = _mat(b)
            if len(B) != n or len(B[0]) != 1:
                return ToolResult(success=False, error="b must be an n×1 column")
            # x = A⁻¹·b via the inverse just computed.
            x = [sum(inv[i][j] * B[j][0] for j in range(n)) for i in range(n)]
            return ToolResult(success=True, output=str(x), data={"result": x})
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: multiply, transpose, det, inverse, solve",
        )
    except (ValueError, SyntaxError, TypeError) as e:
        return ToolResult(success=False, error=f"bad matrix: {e}")


MATH_LAB_TOOLS = (
    (
        "statistics_lab",
        "Descriptive statistics: data= '1 2 3' or '1,2,3' → n, mean, "
        "median, stdev, min, max. action='correlation'|'regression' "
        "with data= 'x y, x y, …' → Pearson r and least-squares line.",
        statistics_lab,
    ),
    (
        "number_theory",
        "Exact integer math: is_prime(n), factor(n), primes(<n sieve), "
        "gcd(n,k), lcm, modpow(n,k,m), modinv(n,m), egcd(n,k). "
        "Miller–Rabin is deterministic below 2^64.",
        number_theory,
    ),
    (
        "matrix",
        "Dense-matrix ops on '[[a,b],[c,d]]' literals: multiply(a,b), "
        "transpose(a), det(a), inverse(a), solve(a,b-column). Partial-"
        "pivot Gaussian elimination; singular inputs fail honestly.",
        matrix,
    ),
)
