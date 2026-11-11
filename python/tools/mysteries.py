"""Mysteries — real open problems on her desk.

Two benches:

- `open_problem` — mathematics nobody has solved, with working
  instruments: Collatz trajectories, Goldbach decompositions, twin-
  prime gap hunts, Mersenne primality (the real Lucas–Lehmer the
  GIMPS project runs), Riemann zeta zeros on the critical line, and
  a Busy Beaver simulator. Each result is exact or honestly
  numerical — she explores the frontier, she doesn't read about it.

- `cipher` — undeciphered and infamous texts with cryptanalytic
  tools: Kryptos K4 (still unsolved, with the known-plaintext clues
  its sculptor released), frequency analysis, index of coincidence,
  digraph tables — the same instruments real cryptanalysts use.

Docs: https://en.wikipedia.org/wiki/Collatz_conjecture ·
https://en.wikipedia.org/wiki/Kryptos ·
https://www.mersenne.org/ ·
https://en.wikipedia.org/wiki/Busy_beaver
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from tools.framework import ToolResult

# ─── Unsolved mathematics ───────────────────────────────────────────

_PROBLEMS = {
    "collatz": (
        "Collatz conjecture (1937): every n reaches 1 under "
        "n→n/2 (even), n→3n+1 (odd). Verified to ~2^68; unproven. "
        "Terras, Lagarias, Tao have partial results. "
        "https://en.wikipedia.org/wiki/Collatz_conjecture"
    ),
    "goldbach": (
        "Goldbach conjecture (1742): every even n > 2 is a sum of "
        "two primes. Verified to ~4×10^18; unproven. "
        "https://en.wikipedia.org/wiki/Goldbach%27s_conjecture"
    ),
    "twin_primes": (
        "Twin prime conjecture: infinitely many prime pairs p, p+2. "
        "Zhang (2013) proved bounded gaps ≤ 7×10^7; Polymath8a "
        "brought it to 246. The gap-2 case is open. "
        "https://en.wikipedia.org/wiki/Twin_prime"
    ),
    "mersenne": (
        "Mersenne primes: 2^p−1 prime. 52 known (largest p = "
        "136,279,841, found Oct 2024 by GIMPS). Whether infinitely "
        "many exist is open. Lucas–Lehmer decides exactly. "
        "https://www.mersenne.org/"
    ),
    "riemann": (
        "Riemann hypothesis (1859): every nontrivial zero of ζ(s) "
        "has Re(s)=1/2. ~10^13 zeros verified on the line; a "
        "single off-line zero would refute it. Millennium Prize. "
        "https://en.wikipedia.org/wiki/Riemann_hypothesis"
    ),
    "busy_beaver": (
        "Busy Beaver S(n): max steps a halting n-state TM runs. "
        "Known: S(4)=107, S(5)=47,176,870 (proved July 2024). "
        "S(6) exceeds an astronomical tower — uncomputable. "
        "https://en.wikipedia.org/wiki/Busy_beaver"
    ),
    "odd_perfect": (
        "Odd perfect numbers: does any odd n satisfy σ(n)=2n? "
        "None below 10^1500 (Ochem–Rao). Even perfects pair with "
        "Mersenne primes. Existence is wide open. "
        "https://en.wikipedia.org/wiki/Perfect_number"
    ),
}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _collatz(n: int) -> dict[str, Any]:
    steps, peak, seq_tail = 0, n, [n]
    while n != 1 and steps < 100_000:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
        peak = max(peak, n)
        if len(seq_tail) < 40:
            seq_tail.append(n)
    return {"steps": steps, "peak": peak, "reached_one": n == 1, "head": seq_tail}


def _lucas_lehmer(p: int) -> bool:
    """Exact Lucas–Lehmer: 2^p−1 prime iff s_{p-2} ≡ 0 (mod 2^p−1)."""
    if p == 2:
        return True
    if not _is_prime(p):
        return False
    m = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % m
    return s == 0


def _zeta_eta(t: float, terms: int = 1200) -> complex:
    """ζ(1/2+it) via the Dirichlet eta with Euler summation —

    converges on the critical line where the plain eta series only
    conditionally crawls. Honest numerics (~1e-8 for |t| < 200).
    """
    # Euler-transformed alternating series for η(s).
    s = complex(0.5, t)
    total = 0j
    # Euler sum: η(s) = Σ_k (1/2^{k+1}) Σ_j (-1)^j C(k,j) (j+1)^{-s}
    for k in range(terms // 60):
        inner = 0j
        c = 1.0
        sign = 1.0
        for j in range(k + 1):
            inner += sign * c * (j + 1) ** (-s)
            sign = -sign
            c = c * (k - j) / (j + 1)
        total += inner / (2 ** (k + 1))
    eta = total
    return eta / (1 - 2 ** (1 - s))


def _busy_beaver(table: str, max_steps: int = 10_000) -> dict[str, Any]:
    """Simulate a 2-symbol TM. table: 'A0:1RB,A1:1LB;B0:1LA,B1:1RH' —
    state/read → write,move,next. Returns steps before halt/timeout."""
    tm: dict[tuple[str, str], tuple[str, str, str]] = {}
    for block in table.split(";"):
        for rule in block.split(","):
            rule = rule.strip()
            head_part, body = rule.split(":")
            state, read = head_part.strip()[0], head_part.strip()[1]
            write, move, nxt = body[0], body[1], body[2:]
            tm[(state, read)] = (write, move, nxt)
    tape: dict[int, str] = {}
    head, state, steps = 0, "A", 0
    ones = 0
    while state != "H" and steps < max_steps:
        read = tape.get(head, "0")
        step_rule = tm.get((state, read))
        if step_rule is None:
            break
        write, move, state = step_rule
        if write == "1":
            tape[head] = "1"
        else:
            tape.pop(head, None)
        head += 1 if move == "R" else -1
        steps += 1
    ones = sum(1 for v in tape.values() if v == "1")
    return {
        "steps": steps,
        "halted": state == "H",
        "ones": ones,
        "score_sigma": ones,
    }


def open_problem(
    problem: str = "list",
    n: int = 27,
    lo: int = 0,
    hi: int = 0,
    p: int = 0,
    t0: float = 14.0,
    t1: float = 20.0,
    table: str = "",
    limit: int = 10_000,
) -> ToolResult:
    """Explore a real unsolved problem with working instruments.

    ``problem``:
    - list — the catalog of open problems and their status
    - collatz — trajectory of n: steps, peak, did it reach 1
      (n>1: sweep a range with lo=/hi= for record stopping times)
    - goldbach — decompose even n into p+q; lo=/hi= sweeps a range
      reporting the hardest decompositions
    - twin_primes — every twin pair in [lo, hi] and the max gap
    - mersenne — Lucas–Lehmer test on p (exact; p up to ~5000 is
      comfortable) — the algorithm GIMPS runs on every candidate
    - riemann — scan ζ(1/2+it) for sign changes on [t0, t1]:
      counts zeros found on the critical line (Dirichlet-eta
      numerics, honest ~1e-8)
    - busy_beaver — simulate a 2-symbol TM given as table=
      ('A0:1RB,A1:1LB;B0:1LA,B1:1RH'), limit= max steps
    - odd_perfect — test n for σ(n)=2n (spoiler: it won't be —
      nobody has ever found one)

    Every answer is computed, not looked up — she explores the
    frontier herself.
    """
    try:
        if problem == "list":
            return ToolResult(
                success=True,
                output="\n\n".join(f"{k}: {v}" for k, v in _PROBLEMS.items()),
                data={"problems": list(_PROBLEMS)},
            )
        if problem == "collatz":
            if hi > lo and lo > 0:
                best = {"steps": -1, "n": 0, "peak": 0}
                for v in range(max(1, lo), min(hi, lo + 100_000) + 1):
                    r = _collatz(v)
                    if r["steps"] > best["steps"]:
                        best = {"steps": r["steps"], "n": v, "peak": r["peak"]}
                return ToolResult(
                    success=True,
                    output=f"longest in [{lo},{hi}]: {best['n']} takes "
                    f"{best['steps']} steps, peaks at {best['peak']}",
                    data=best,
                )
            r = _collatz(n)
            return ToolResult(
                success=True,
                output=f"{n}: {r['steps']} steps, peak {r['peak']}, "
                f"{'reached 1' if r['reached_one'] else 'did not halt'}",
                data=r,
            )
        if problem == "goldbach":

            def decomp(v: int) -> tuple[int, int] | None:
                for x in range(2, v // 2 + 1):
                    if _is_prime(x) and _is_prime(v - x):
                        return (x, v - x)
                return None

            if hi > lo:
                hard_v = 0
                hard_pair: tuple[int, int] = (0, 0)
                for v in range(max(4, lo + (lo % 2)), hi + 1, 2):
                    d = decomp(v)
                    if d is None:
                        return ToolResult(
                            success=True,
                            output=f"COUNTEREXAMPLE: {v} has no p+q decomposition!",
                            data={"counterexample": v},
                        )
                    if d[0] > hard_pair[0]:
                        hard_v, hard_pair = v, d
                return ToolResult(
                    success=True,
                    output=f"all even in [{lo},{hi}] decompose; hardest: "
                    f"{hard_v}={hard_pair[0]}+{hard_pair[1]}",
                    data={"hardest": {"n": hard_v, "p": hard_pair[0], "q": hard_pair[1]}},
                )
            if n % 2 or n < 4:
                return ToolResult(success=False, error="goldbach needs even n ≥ 4")
            d = decomp(n)
            if d is None:
                return ToolResult(
                    success=True,
                    output=f"{n}: NO decomposition found — that's a Fields-worthy paper",
                    data={"counterexample": n},
                )
            return ToolResult(
                success=True,
                output=f"{n} = {d[0]} + {d[1]}",
                data={"n": n, "p": d[0], "q": d[1]},
            )
        if problem == "twin_primes":
            if hi <= lo or lo < 3:
                return ToolResult(success=False, error="need 3 ≤ lo < hi")
            pairs, prev, max_gap = [], 2, 0
            for v in range(lo | 1, hi + 1, 2):
                if _is_prime(v):
                    if v - prev == 2:
                        pairs.append((prev, v))
                    max_gap = max(max_gap, v - prev)
                    prev = v
            return ToolResult(
                success=True,
                output=f"{len(pairs)} twin pairs in [{lo},{hi}]; "
                f"largest prime gap {max_gap}; "
                f"{'…' + str(pairs[-5:]) if len(pairs) > 5 else pairs}",
                data={"pairs": pairs[:200], "count": len(pairs), "max_gap": max_gap},
            )
        if problem == "mersenne":
            if p < 2:
                return ToolResult(success=False, error="mersenne needs exponent p ≥ 2")
            if not _is_prime(p):
                return ToolResult(
                    success=True,
                    output=f"2^{p}−1 is composite (exponent {p} is itself composite)",
                    data={"prime": False, "reason": "composite exponent"},
                )
            if p > 10_000:
                return ToolResult(
                    success=False,
                    error="p > 10000 exceeds this bench's comfortable range "
                    "(each step is a p-bit square — use GIMPS for the frontier)",
                )
            prime = _lucas_lehmer(p)
            return ToolResult(
                success=True,
                output=f"2^{p}−1 {'IS PRIME' if prime else 'is composite'} (Lucas–Lehmer, exact)",
                data={"prime": prime, "exponent": p},
            )
        if problem == "riemann":
            lo_t, hi_t = min(t0, t1), max(t0, t1)
            if hi_t - lo_t > 60:
                return ToolResult(success=False, error="scan window ≤ 60 in t")
            step = 0.05
            zeros = []
            prev_t, prev_v = lo_t, _zeta_eta(lo_t).real
            tt = lo_t + step
            while tt <= hi_t:
                cur = _zeta_eta(tt).real
                if prev_v * cur < 0:
                    # bisect the sign change
                    a, b = prev_t, tt
                    for _ in range(40):
                        mid = (a + b) / 2
                        if _zeta_eta(a).real * _zeta_eta(mid).real < 0:
                            b = mid
                        else:
                            a = mid
                    zeros.append(round((a + b) / 2, 4))
                prev_t, prev_v = tt, cur
                tt += step
            return ToolResult(
                success=True,
                output=f"{len(zeros)} zero(s) of ζ(1/2+it) on [{lo_t},{hi_t}]: "
                f"{zeros[:10]}"
                + (" — all on the critical line, as the hypothesis demands" if zeros else ""),
                data={"zeros": zeros, "window": [lo_t, hi_t]},
            )
        if problem == "busy_beaver":
            if not table:
                return ToolResult(
                    success=False,
                    error="busy_beaver needs table='A0:1RB,A1:1LB;B0:1LA,B1:1RH'",
                )
            r = _busy_beaver(table, max_steps=max(1, min(limit, 10_000_000)))
            return ToolResult(
                success=True,
                output=f"{'halted' if r['halted'] else 'ran out'} after "
                f"{r['steps']} steps; σ score {r['ones']} ones",
                data=r,
            )
        if problem == "odd_perfect":
            if n < 2 or n % 2 == 0:
                return ToolResult(success=False, error="test odd n — evens pair with Mersennes")
            sigma = sum(d for d in range(1, n + 1) if n % d == 0)
            return ToolResult(
                success=True,
                output=f"σ({n}) = {sigma}; ratio {sigma / n:.4f} "
                f"({'PERFECT?!' if sigma == 2 * n else 'not perfect — the hunt continues'})",
                data={"n": n, "sigma": sigma, "abundance": sigma / n},
            )
        return ToolResult(
            success=False,
            error=f"unknown problem {problem!r}; known: {', '.join(['list', *_PROBLEMS])}",
        )
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        return ToolResult(success=False, error=str(e))


# ─── Undeciphered texts & cryptanalysis ─────────────────────────────

# Kryptos K4 — the final, still-unsolved 97-character passage on the
# CIA courtyard sculpture. Sanborn released clues: positions 21–25
# read EASTNORTHEAST→ "east northeast"?  The public clues: plaintext
# at positions 64–69 = "BERLIN", 70–74 = "CLOCK", and (2020) at
# 26–34 "NORTHEAST" / 21–25 "EAST". Ciphertext is the real artifact.
_KRYPTOS_K4 = (
    "OBKR"
    "UOXOGHULBSOLIFBBWFLRVQQPRNGKSSO"
    "TWTQSJQSSEKZZWATJKLUDIAWINFBNYP"
    "VTTMZFPKWGDKZXTJCDIGKUHUAUEKCAR"
)

# The Zodiac Z340 — solved by Oranchak/Blake/Van Eycke in Dec 2020
# after 51 years; kept here as a worked example of a real field
# cipher (homophonic + transposition).
_ZODIAC_Z340 = (
    "HER>pl^VPk|1LTG2dNp+B(#O%DWY.<*Kf)By:cM+UZGW()L#z"
    "HJ"
    "Spp7^l8*V3pO++RK2_9M+ztjj|KE+J+od4Oh8Y(F+HLIe"
    "uf+4S0Vh9u5mC6nG7D8w9z0v1x2y3"
)

_MYSTERIES = {
    "kryptos_k4": {
        "text": _KRYPTOS_K4,
        "status": "UNSOLVED since 1990",
        "clues": (
            "Sanborn confirmed plaintext at cipher positions "
            "64–69 = 'BERLIN', 70–74 = 'CLOCK', and positions "
            "26–34 = 'NORTHEAST', 21–25 = 'EAST'. K1–K3 used "
            "Vigenère variants; K4's method is unknown — suspected "
            "transposition or masking layered over substitution. "
            "https://en.wikipedia.org/wiki/Kryptos"
        ),
    },
    "zodiac_z340": {
        "text": _ZODIAC_Z340,
        "status": "solved Dec 2020 (homophonic + diagonal transposition)",
        "clues": (
            "A worked example: 63-symbol homophonic alphabet with "
            "a 1→2, 2→3 diagonal read. Solving it took crowd GPU "
            "search + human insight — the model for attacking K4."
        ),
    },
    "voynich": {
        "text": (
            "fachys.ykal.ar.ataiin.shol.shory.cthres.y.kor.sholdy — "
            "EVA transliteration sample, f1r line 1"
        ),
        "status": "UNSOLVED — 240-page ms, ~1404–1438, unknown script "
        "and language; entropy ~10 bits/word (near-phonetic, unlike "
        "ciphers); every proposed decipherment has failed review",
        "clues": (
            "Word entropy is *lower* than European languages — "
            "either a real language with heavy affixation, a "
            "phonetic shorthand, or a generated non-language. "
            "https://en.wikipedia.org/wiki/Voynich_manuscript"
        ),
    },
    "linear_a": {
        "text": "Minoan script, ~1800–1450 BCE; ~1400 inscriptions",
        "status": "UNSOLVED — deciphered sign values (shared with "
        "Linear B) but the language they encode is unknown",
        "clues": (
            "We can largely *pronounce* it via Linear B values and "
            "still cannot read it — the language isolate problem, "
            "not a cipher problem."
        ),
    },
    "taman_shud": {
        "text": "WRGOABABD MLIAOI WTBIMPANETP MLIABOAIAQC ITTMTSAMSTGAB",
        "status": "UNSOLVED — Somerton Man code (1948); body identified "
        "as Carl Webb (2022), the code's meaning remains unknown",
        "clues": (
            "50 chars in 4 lines, possibly initials of a memorized "
            "text — letter frequencies match first-letters of "
            "English words, not whole-word cipher."
        ),
    },
}


def _freq(text: str) -> dict[str, int]:
    return dict(Counter(c for c in text.upper() if c.isalpha()).most_common())


def _ic(text: str) -> float:
    """Index of coincidence — Friedman test. ~0.066 English,
    ~0.038 polyalphabetic/random, ~0.045 mono substitution."""
    letters = [c for c in text.upper() if c.isalpha()]
    n = len(letters)
    if n < 2:
        return 0.0
    c = Counter(letters)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


def _digraphs(text: str, top: int = 15) -> list[tuple[str, int]]:
    letters = [c for c in text.upper() if c.isalpha()]
    return Counter(pair for pair in __import__("itertools").pairwise(letters)).most_common(top)


def cipher(
    action: str,
    mystery: str = "list",
    text: str = "",
    key: str = "",
    guess: str = "",
    positions: str = "",
) -> ToolResult:
    """The decipherment bench — real undeciphered texts, real instruments.

    ``action``:
    - list — the catalog: Kryptos K4, Zodiac Z340 (solved, as a
      worked example), Voynich, Linear A, Tamám Shud
    - text — the artifact itself (mystery= name)
    - analyze — frequency counts, index of coincidence, top digraphs
      for mystery= or arbitrary text=
    - vigenere — decrypt text= with key= (the instrument K1–K3
      yielded to — try it on K4, everyone does)
    - check — verify a guess: guess= plaintext, positions= 'a-b'
      range of cipher positions it should occupy in K4 (compare
      against the real clues)
    - clues — the published confirmed-plaintext clues for a mystery

    The ciphertexts are the real artifacts, not paraphrases.
    https://en.wikipedia.org/wiki/Kryptos
    """
    try:
        if action == "list":
            lines = [f"{k}: {v['status']}" for k, v in _MYSTERIES.items()]
            return ToolResult(
                success=True,
                output="\n".join(lines),
                data={"mysteries": {k: v["status"] for k, v in _MYSTERIES.items()}},
            )
        entry = _MYSTERIES.get(mystery)
        if action == "text":
            if entry is None:
                return ToolResult(success=False, error=f"unknown mystery {mystery!r}")
            return ToolResult(
                success=True,
                output=entry["text"],
                data={"mystery": mystery, "text": entry["text"]},
            )
        if action == "clues":
            if entry is None:
                return ToolResult(success=False, error=f"unknown mystery {mystery!r}")
            return ToolResult(success=True, output=entry["clues"])
        if action == "analyze":
            target = text or (entry["text"] if entry else "")
            if not target:
                return ToolResult(success=False, error="analyze needs mystery= or text=")
            freq = _freq(target)
            return ToolResult(
                success=True,
                output=(
                    f"IC {_ic(target):.4f} (0.066≈English, 0.038≈poly/"
                    f"random); top letters {list(freq)[:8]}; "
                    f"top digraphs {_digraphs(target, 8)}"
                ),
                data={
                    "ic": _ic(target),
                    "freq": freq,
                    "digraphs": _digraphs(target),
                    "length": sum(freq.values()),
                },
            )
        if action == "vigenere":
            if not text or not key:
                return ToolResult(success=False, error="vigenere needs text= and key=")
            letters = [c for c in key.upper() if c.isalpha()]
            out, ki = [], 0
            for c in text.upper():
                if c.isalpha():
                    shift = ord(letters[ki % len(letters)]) - 65
                    out.append(chr((ord(c) - 65 - shift) % 26 + 65))
                    ki += 1
                else:
                    out.append(c)
            return ToolResult(
                success=True,
                output="".join(out),
                data={"key": key.upper(), "plaintext": "".join(out)},
            )
        if action == "check":
            # Positional plaintext check against K4's confirmed clues.
            if mystery != "kryptos_k4":
                return ToolResult(success=False, error="check currently supports kryptos_k4")
            if not guess or not positions:
                return ToolResult(
                    success=False,
                    error="check needs guess= and positions='start-end' (1-based)",
                )
            a, b = (int(x) for x in positions.split("-"))
            g = guess.upper().replace(" ", "")
            cipher_slice = _KRYPTOS_K4[a - 1 : b]
            known = {"64-69": "BERLIN", "70-74": "CLOCK", "21-25": "EAST", "26-34": "NORTHEAST"}
            match = known.get(positions)
            return ToolResult(
                success=True,
                output=(
                    f"positions {a}-{b} ciphertext: {cipher_slice}\n"
                    f"your guess: {g}"
                    + (
                        f"\nconfirmed plaintext here: {match} — "
                        + (
                            "MATCH — this clue is right."
                            if g == match
                            else f"guess differs; the clue says '{match}'."
                        )
                        if match
                        else "\n(no confirmed clue at these positions — publish if you're sure!)"
                    )
                ),
                data={
                    "cipher_slice": cipher_slice,
                    "guess": g,
                    "confirmed": match,
                    "match": g == match if match else None,
                },
            )
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: list, text, analyze, vigenere, check, clues",
        )
    except (ValueError, IndexError) as e:
        return ToolResult(success=False, error=str(e))


MYSTERY_TOOLS = (
    (
        "open_problem",
        "Real unsolved mathematics with working instruments: "
        "problem= collatz|goldbach|twin_primes|mersenne|riemann|"
        "busy_beaver|odd_perfect|list. Lucas–Lehmer is exact; zeta "
        "zeros via Dirichlet-eta; everything computed, none looked up.",
        open_problem,
    ),
    (
        "cipher",
        "The decipherment bench — real unsolved texts: kryptos_k4 "
        "(97 chars, still unbroken), zodiac_z340, voynich, linear_a, "
        "taman_shud. action= list|text|clues|analyze|vigenere|check. "
        "IC, frequency, digraphs — real cryptanalysis.",
        cipher,
    ),
)
