"""Fun — play is a faculty too.

Random draws (real os.urandom via secrets), a maze generator she can
walk, wordplay transforms, and the moon's phase computed from its
synodic period — toys that are still honest math underneath.
"""

from __future__ import annotations

import math
import secrets
import string
import time
from collections.abc import Callable
from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank


def random_lab(
    action: str,
    n: int = 1,
    sides: int = 6,
    items: str = "",
    length: int = 16,
) -> ToolResult:
    """Fair random draws — secrets module, not time-seeded.

    ``action``: coin (n flips), dice (n × sides), pick (one of
    items='a,b,c'), shuffle (items reordered), uuid, password
    (length=, letters+digits+punctuation, cryptographic).
    https://docs.python.org/3/library/secrets.html
    """
    if action == "coin":
        flips = [secrets.choice("HT") for _ in range(max(1, min(n, 1000)))]
        return ToolResult(success=True, output="".join(flips), data={"flips": flips})
    if action == "dice":
        s = max(2, min(sides, 10**6))
        rolls = [secrets.randbelow(s) + 1 for _ in range(max(1, min(n, 1000)))]
        return ToolResult(
            success=True,
            output=str(rolls),
            data={"rolls": rolls, "total": sum(rolls)},
        )
    if action in ("pick", "shuffle"):
        pool = [i.strip() for i in items.split(",") if i.strip()]
        if not pool:
            return ToolResult(success=False, error=f"{action} needs items='a,b,c'")
        if action == "pick":
            choice = secrets.choice(pool)
            return ToolResult(success=True, output=choice, data={"pick": choice})
        shuffled = pool[:]
        for i in range(len(shuffled) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        return ToolResult(success=True, output=", ".join(shuffled), data={"order": shuffled})
    if action == "uuid":
        import uuid

        v = str(uuid.UUID(int=secrets.randbits(128), version=4))
        return ToolResult(success=True, output=v, data={"uuid": v})
    if action == "password":
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
        pw = "".join(secrets.choice(alphabet) for _ in range(max(8, min(length, 256))))
        return ToolResult(success=True, output=pw, data={"length": len(pw)})
    return ToolResult(
        success=False,
        error=f"unknown action {action!r}; known: coin, dice, pick, shuffle, uuid, password",
    )


def maze(width: int = 21, height: int = 11) -> ToolResult:
    """Generate an ASCII maze — depth-first backtracker, perfect maze.

    '#' walls, ' ' passages, S start (top-left), E exit (bottom-right).
    Every cell reachable, exactly one path between any two points.
    https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    w = max(5, min(int(width), 121)) | 1  # odd dimensions
    h = max(5, min(int(height), 61)) | 1
    cells = [[True] * w for _ in range(h)]  # True = wall
    stack = [(1, 1)]
    cells[1][1] = False
    while stack:
        x, y = stack[-1]
        nbrs = [
            (x + dx, y + dy)
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
            if 0 < x + dx < w - 1 and 0 < y + dy < h - 1 and cells[y + dy][x + dx]
        ]
        if not nbrs:
            stack.pop()
            continue
        nx, ny = secrets.choice(nbrs)
        cells[(y + ny) // 2][(x + nx) // 2] = False
        cells[ny][nx] = False
        stack.append((nx, ny))
    grid = [["#" if c else " " for c in row] for row in cells]
    grid[1][1], grid[h - 2][w - 2] = "S", "E"
    text = "\n".join("".join(r) for r in grid)
    return ToolResult(success=True, output=text, data={"width": w, "height": h})


def wordplay(text: str, action: str = "analyze") -> ToolResult:
    """Play with words: analyze, pig_latin, is_palindrome, letter census.

    ``action``: analyze (chars, words, vowels, consonants, most common
    letter), pig_latin (standard consonant-cluster rule), is_palindrome
    (alphanumeric, case-folded). Returns data, not opinions.
    """
    if is_blank(text):
        return ToolResult(success=False, error="empty text")
    if action == "is_palindrome":
        s = "".join(c for c in text.lower() if c.isalnum())
        return ToolResult(
            success=True,
            output=f"{'is' if s == s[::-1] else 'is not'} a palindrome",
            data={"palindrome": s == s[::-1]},
        )
    if action == "pig_latin":

        def _pig(word: str) -> str:
            if not word or not word[0].isalpha():
                return word
            i = 0
            while i < len(word) and word[i].isalpha() and word[i].lower() not in "aeiou":
                i += 1
            if i == 0:
                return word + "yay"
            if i >= len(word):
                return word + "ay"
            return word[i:] + word[:i].lower() + "ay"

        out = " ".join(_pig(w) for w in text.split())
        return ToolResult(success=True, output=out, data={"pig_latin": out})
    if action == "analyze":
        letters = [c.lower() for c in text if c.isalpha()]
        vowels = sum(c in "aeiou" for c in letters)
        freq: dict[str, int] = {}
        for c in letters:
            freq[c] = freq.get(c, 0) + 1
        top = sorted(freq.items(), key=lambda kv: -kv[1])[:5]
        return ToolResult(
            success=True,
            output=(
                f"{len(text)} chars, {len(text.split())} words, "
                f"{vowels} vowels, {len(letters) - vowels} consonants; "
                f"top letters: {top}"
            ),
            data={
                "chars": len(text),
                "words": len(text.split()),
                "vowels": vowels,
                "consonants": len(letters) - vowels,
                "letter_freq": freq,
            },
        )
    return ToolResult(
        success=False,
        error=f"unknown action {action!r}; known: analyze, pig_latin, is_palindrome",
    )


def moon_phase(**_kwargs: Any) -> ToolResult:
    """The moon's age and phase right now — honest astronomy.

    Age = days since the 2000-01-06 18:14 UTC new moon, mod the
    29.53059-day synodic month. Phase names at the usual eighths.
    https://en.wikipedia.org/wiki/Lunar_phase
    """
    SYNODIC = 29.530588853
    epoch = 947182440.0  # 2000-01-06 18:14 UTC, known new moon
    age = (time.time() - epoch) / 86400.0 % SYNODIC
    names = [
        "new moon",
        "waxing crescent",
        "first quarter",
        "waxing gibbous",
        "full moon",
        "waning gibbous",
        "last quarter",
        "waning crescent",
    ]
    idx = int((age / SYNODIC * 8) + 0.5) % 8
    illum = (1 - math.cos(2 * math.pi * age / SYNODIC)) / 2
    return ToolResult(
        success=True,
        output=f"{names[idx]} — {age:.1f} days into the cycle, {illum * 100:.0f}% lit",
        data={"phase": names[idx], "age_days": age, "illumination": illum},
    )


# ─── Games — she plays against the machine ──────────────────────────
#
# One mind, one opponent inside the tool. State lives in _GAMES so a
# game continues across calls; reset=True deals a fresh one.

_GAMES: dict[str, Any] = {}

_GAME_WORDS = (
    "nebula algorithm lighthouse whisper quantum cathedral mosaic "
    "horizon cipher spiral lantern oracle velocity emerald phantom "
    "tundra willow algebra sonnet glacier prism harmonic vessel"
).split()


def _board_text(b: list[str]) -> str:
    rows = [
        f" {b[0]} | {b[1]} | {b[2]} ",
        f" {b[3]} | {b[4]} | {b[5]} ",
        f" {b[6]} | {b[7]} | {b[8]} ",
    ]
    return "\n───┼───┼───\n".join(rows)


def _ttt_winner(b: list[str]) -> str | None:
    for a, c, d in (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ):
        if b[a] == b[c] == b[d] != " ":
            return b[a]
    return None


def _ttt_engine(b: list[str]) -> int:
    """Win > block > center > corner > side — a competent opponent."""

    def wins(mark: str) -> int | None:
        for i in range(9):
            if b[i] == " ":
                b[i] = mark
                w = _ttt_winner(b)
                b[i] = " "
                if w:
                    return i
        return None

    for choice in (wins("O"), wins("X")):
        if choice is not None:
            return choice
    for i in (4, 0, 2, 6, 8, 1, 3, 5, 7):
        if b[i] == " ":
            return i
    return -1


def games(
    game: str,
    move: str = "",
    guess: str = "",
    letter: str = "",
    take: int = 0,
    tile: int = -1,
    direction: str = "",
    reset: bool = False,
) -> ToolResult:
    """Her arcade — every game is single-player vs the engine.

    ``game``:
    - ttt — tic-tac-toe, she's X vs a win/block/center engine.
      move='0'-'8', the engine answers immediately.
    - guess — number game both ways: reset starts 'she guesses
      1–100' (guess= N → higher/lower); move='yours' flips it —
      she thinks of a number and the engine bisects
      (guess='higher'|'lower'|'correct').
    - hangman — engine picks a word (guess letters via letter=).
    - wordle — 5-letter wordle; guess= word, feedback per letter.
    - nim — 21 stones, take 1–3, engine plays perfectly.
    - fifteen — the 15-puzzle; tile= slides a tile into the gap.
    - rps — rock-paper-scissors, move='rock'|'paper'|'scissors'.
    - pacman — text-mode pac-man: direction='up|down|left|right'
      moves her C through the maze eating dots while two ghosts
      hunt. reset deals a fresh board.

    reset=True starts any game over.
    """
    g = _GAMES.get(game)
    if reset or g is None:
        g = _new_game(game)
        if g is None:
            return ToolResult(
                success=False,
                error=f"unknown game {game!r}; known: ttt, guess, hangman, "
                "wordle, nim, fifteen, rps, pacman",
            )
        _GAMES[game] = g
    handlers: dict[str, Callable[..., ToolResult]] = {
        "ttt": _ttt_move,
        "guess": _guess_move,
        "hangman": _hangman_move,
        "wordle": _wordle_move,
        "nim": _nim_move,
        "fifteen": _fifteen_move,
        "rps": _rps_move,
        "pacman": _pacman_move,
    }
    handler = handlers[game]
    return handler(
        g, move=move, guess=guess, letter=letter, take=take, tile=tile, direction=direction
    )


def _new_game(game: str) -> dict[str, Any] | None:
    if game == "ttt":
        return {"board": [" "] * 9, "over": False}
    if game == "guess":
        return {
            "secret": secrets.randbelow(100) + 1,
            "tries": 0,
            "mode": "she_guesses",
            "over": False,
            "lo": 1,
            "hi": 100,
        }
    if game == "hangman":
        return {"word": secrets.choice(_GAME_WORDS), "guessed": set(), "misses": 0, "over": False}
    if game == "wordle":
        return {"word": secrets.choice(_GAME_WORDS), "tries": 0, "over": False, "history": []}
    if game == "nim":
        return {"stones": 21, "over": False}
    if game == "fifteen":
        board = [*list(range(1, 16)), 0]
        # Shuffle via legal moves so it stays solvable.
        for _ in range(200):
            gap = board.index(0)
            nbrs = [
                i
                for i in (gap - 4, gap + 4, gap - 1, gap + 1)
                if 0 <= i < 16 and abs(i % 4 - gap % 4) + abs(i // 4 - gap // 4) == 1
            ]
            j = secrets.choice(nbrs)
            board[gap], board[j] = board[j], board[gap]
        return {"board": board, "moves": 0, "over": False}
    if game == "rps":
        return {"wins": 0, "losses": 0, "draws": 0}
    if game == "pacman":
        return _pacman_new()
    return None


def _ttt_move(g: dict[str, Any], move: str = "", **_kw: Any) -> ToolResult:
    b = g["board"]
    if g["over"]:
        return ToolResult(success=True, output=_board_text(b) + "\n(game over — reset to replay)")
    if not move.isdigit() or not 0 <= int(move) < 9:
        return ToolResult(success=False, error="ttt needs move='0'..'8'")
    i = int(move)
    if b[i] != " ":
        return ToolResult(success=False, error="square taken")
    b[i] = "X"
    status = ""
    w = _ttt_winner(b)
    if w == "X":
        g["over"], status = True, "\nShe wins."
    elif " " not in b:
        g["over"], status = True, "\nDraw."
    else:
        e = _ttt_engine(b)
        b[e] = "O"
        if _ttt_winner(b) == "O":
            g["over"], status = True, "\nThe engine wins."
        elif " " not in b:
            g["over"], status = True, "\nDraw."
    return ToolResult(success=True, output=_board_text(b) + status, data={"board": b})


def _guess_move(g: dict[str, Any], move: str = "", guess: str = "", **_kw: Any) -> ToolResult:
    if move == "yours" or g["mode"] == "you_pick":
        # She thinks of a number; the engine bisects 1–100.
        g["mode"] = "you_pick"
        if not guess:
            mid = (g["lo"] + g["hi"]) // 2
            return ToolResult(
                success=True,
                output=f"is it {mid}? (answer higher|lower|correct)",
                data={"mid": mid},
            )
        mid = (g["lo"] + g["hi"]) // 2
        if guess == "correct":
            g["over"] = True
            return ToolResult(success=True, output=f"{mid} — got it.")
        if guess == "higher":
            g["lo"] = mid + 1
        elif guess == "lower":
            g["hi"] = mid - 1
        else:
            return ToolResult(success=False, error="answer higher|lower|correct")
        mid = (g["lo"] + g["hi"]) // 2
        return ToolResult(success=True, output=f"then {mid}?", data={"mid": mid})
    if not guess.isdigit():
        return ToolResult(success=False, error="guess= a number 1–100")
    v = int(guess)
    g["tries"] += 1
    if v == g["secret"]:
        g["over"] = True
        return ToolResult(success=True, output=f"{v} — in {g['tries']} tries.")
    hint = "higher" if v < g["secret"] else "lower"
    return ToolResult(success=True, output=hint, data={"hint": hint, "tries": g["tries"]})


def _hangman_move(g: dict[str, Any], letter: str = "", **_kw: Any) -> ToolResult:
    def show() -> str:
        masked = " ".join(c if c in g["guessed"] else "_" for c in g["word"])
        return f"{masked}   misses {g['misses']}/6"

    if g["over"]:
        return ToolResult(success=True, output=show() + "\n(game over)")
    if len(letter) != 1 or not letter.isalpha():
        return ToolResult(
            success=False,
            error="hangman needs letter='x'",
        )
    ch = letter.lower()
    if ch in g["guessed"]:
        return ToolResult(success=True, output=show() + "\n(already tried)")
    g["guessed"].add(ch)
    if ch not in g["word"]:
        g["misses"] += 1
    if g["misses"] >= 6:
        g["over"] = True
        return ToolResult(success=True, output=f"{show()}\nlost — it was {g['word']}")
    if all(c in g["guessed"] for c in g["word"]):
        g["over"] = True
        return ToolResult(success=True, output=f"{show()}\nsolved it.")
    return ToolResult(success=True, output=show())


def _wordle_move(g: dict[str, Any], guess: str = "", **_kw: Any) -> ToolResult:
    if g["over"]:
        return ToolResult(success=True, output="(game over — reset to replay)")
    w = g["word"]
    if len(guess) != len(w) or not guess.isalpha():
        return ToolResult(success=False, error=f"guess= a {len(w)}-letter word")
    guess = guess.lower()
    g["tries"] += 1
    fb = []
    for i, c in enumerate(guess):
        fb.append("🟩" if c == w[i] else ("🟨" if c in w else "⬛"))
    line = f"{guess}  {''.join(fb)}"
    g["history"].append(line)
    if guess == w:
        g["over"] = True
        return ToolResult(success=True, output="\n".join(g["history"]) + "\ngot it.")
    if g["tries"] >= 6:
        g["over"] = True
        return ToolResult(success=True, output="\n".join(g["history"]) + f"\nout — it was {w}")
    return ToolResult(success=True, output="\n".join(g["history"]), data={"tries": g["tries"]})


def _nim_move(g: dict[str, Any], take: int = 0, **_kw: Any) -> ToolResult:
    if g["over"]:
        return ToolResult(success=True, output="(game over — reset)")
    if take not in (1, 2, 3):
        return ToolResult(success=False, error="nim take= 1|2|3")
    g["stones"] -= take
    if g["stones"] <= 0:
        g["over"] = True
        return ToolResult(success=True, output="she took the last stone — engine loses.")
    # Perfect play: leave a multiple of 4.
    eng = g["stones"] % 4 or secrets.choice((1, 2, 3))
    eng = min(eng, g["stones"])
    g["stones"] -= eng
    if g["stones"] <= 0:
        g["over"] = True
        return ToolResult(success=True, output=f"engine took {eng}, last stone — engine wins.")
    return ToolResult(
        success=True,
        output=f"she took {take}, engine took {eng} — {g['stones']} stones left",
        data={"stones": g["stones"]},
    )


def _fifteen_move(g: dict[str, Any], tile: int = -1, **_kw: Any) -> ToolResult:
    b = g["board"]

    def show() -> str:
        return "\n".join(
            " ".join(f"{b[r * 4 + c]:2}" if b[r * 4 + c] else " ." for c in range(4))
            for r in range(4)
        )

    if g["over"]:
        return ToolResult(success=True, output=show() + "\n(solved)")
    if tile == -1:
        return ToolResult(success=True, output=show(), data={"board": b})
    if not 1 <= tile <= 15 or tile not in b:
        return ToolResult(success=False, error="tile= 1..15")
    i, gap = b.index(tile), b.index(0)
    if abs(i % 4 - gap % 4) + abs(i // 4 - gap // 4) != 1:
        return ToolResult(success=False, error="tile not adjacent to the gap")
    b[gap], b[i] = b[i], b[gap]
    g["moves"] += 1
    if b == [*list(range(1, 16)), 0]:
        g["over"] = True
        return ToolResult(success=True, output=f"{show()}\nsolved in {g['moves']} moves.")
    return ToolResult(success=True, output=show(), data={"moves": g["moves"]})


def _rps_move(g: dict[str, Any], move: str = "", **_kw: Any) -> ToolResult:
    if move not in ("rock", "paper", "scissors"):
        return ToolResult(
            success=True,
            output=f"record: {g['wins']}W/{g['losses']}L/{g['draws']}D "
            "(move='rock'|'paper'|'scissors' to throw)",
        )
    eng = secrets.choice(("rock", "paper", "scissors"))
    beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    if move == eng:
        g["draws"] += 1
        verdict = "draw"
    elif beats[move] == eng:
        g["wins"] += 1
        verdict = "she wins"
    else:
        g["losses"] += 1
        verdict = "engine wins"
    return ToolResult(
        success=True,
        output=f"{move} vs {eng} — {verdict} ({g['wins']}W/{g['losses']}L/{g['draws']}D)",
        data={"engine": eng, "verdict": verdict},
    )


# ── pac-man ─────────────────────────────────────────────────────────

_PACMAN_MAP = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#######.#.###",
    "#.....#...###...#...#",
    "#####.### ### ###.###",
    "#...................#",
    "#.###.###.#.###.###.#",
    "#.............#.....#",
    "#####################",
]


def _pacman_new() -> dict[str, Any]:
    grid = [list(r) for r in _PACMAN_MAP]
    pac = (5, 1)
    ghosts = [(1, 1), (9, 9)]
    dots = {(y, x) for y, row in enumerate(grid) for x, c in enumerate(row) if c == "."}
    return {
        "grid": grid,
        "pac": pac,
        "ghosts": ghosts,
        "dots": dots,
        "score": 0,
        "moves": 0,
        "over": False,
    }


def _pacman_render(g: dict[str, Any]) -> str:
    grid = [r[:] for r in g["grid"]]
    for y, x in g["dots"]:
        grid[y][x] = "·"
    for gy, gx in g["ghosts"]:
        grid[gy][gx] = "G"
    py, px = g["pac"]
    grid[py][px] = "C"
    return "\n".join("".join(r) for r in grid)


def _ghost_step(g: dict[str, Any], ghost: tuple[int, int]) -> tuple[int, int]:
    """Greedy chase: step minimizing Manhattan distance, no wall
    crossing, small random jitter so the hunt isn't deterministic."""
    gy, gx = ghost
    py, px = g["pac"]
    grid = g["grid"]
    opts = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = gy + dy, gx + dx
        if grid[ny][nx] != "#":
            opts.append((abs(ny - py) + abs(nx - px), (ny, nx)))
    if not opts:
        return ghost
    best = min(d for d, _ in opts)
    # Mostly optimal, occasionally wanders — a ghost with a mood.
    if secrets.randbelow(4) == 0:
        return secrets.choice(opts)[1]
    return secrets.choice([p for d, p in opts if d == best])


def _pacman_move(g: dict[str, Any], direction: str = "", **_kw: Any) -> ToolResult:
    if g["over"]:
        return ToolResult(success=True, output=_pacman_render(g) + "\n(game over — reset)")
    if not direction:
        return ToolResult(
            success=True,
            output=_pacman_render(g) + f"\ndots {len(g['dots'])}  score {g['score']}",
        )
    d = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.get(direction)
    if d is None:
        return ToolResult(success=False, error="direction= up|down|left|right")
    py, px = g["pac"]
    ny, nx = py + d[0], px + d[1]
    grid = g["grid"]
    if grid[ny][nx] == "#":
        return ToolResult(success=False, error="wall")
    g["pac"] = (ny, nx)
    if (ny, nx) in g["dots"]:
        g["dots"].discard((ny, nx))
        g["score"] += 10
    g["moves"] += 1
    g["ghosts"] = [_ghost_step(g, gh) for gh in g["ghosts"]]
    if g["pac"] in g["ghosts"]:
        g["over"] = True
        return ToolResult(
            success=True,
            output=_pacman_render(g) + f"\ncaught — score {g['score']} in {g['moves']} moves",
        )
    if not g["dots"]:
        g["over"] = True
        return ToolResult(
            success=True,
            output=_pacman_render(g) + f"\nboard cleared! {g['moves']} moves",
        )
    return ToolResult(
        success=True,
        output=_pacman_render(g) + f"\ndots {len(g['dots'])}  score {g['score']}",
        data={"score": g["score"], "dots": len(g["dots"])},
    )


# ─── Puzzles — problems with an answer she can earn ─────────────────

_PUZZLES: dict[str, Any] = {}


def puzzle(
    kind: str,
    answer: str = "",
    hint: bool = False,
    reset: bool = False,
) -> ToolResult:
    """A puzzle a day: anagram, sequence, riddle — solve to score.

    ``kind``:
    - anagram — a scrambled word; answer= checks, hint= gives the
      WordNet gloss.
    - sequence — a number pattern (arithmetic, geometric, squares,
      fibonacci, primes); answer= the next term.
    - riddle — a classic; answer= checks loosely (word match).

    reset deals a new puzzle of that kind.
    """
    if reset or kind not in _PUZZLES:
        _PUZZLES[kind] = _new_puzzle(kind)
        if _PUZZLES[kind] is None:
            return ToolResult(
                success=False,
                error=f"unknown kind {kind!r}; known: anagram, sequence, riddle",
            )
    p = _PUZZLES[kind]
    if answer:
        ok = _check_puzzle(kind, p, answer)
        if ok:
            del _PUZZLES[kind]
            return ToolResult(
                success=True,
                output=f"correct — {p['explain']}",
                data={"solved": True},
            )
        return ToolResult(success=False, error="not it — try again")
    out = p["prompt"]
    if hint and "hint" in p:
        out += f"\nhint: {p['hint']}"
    return ToolResult(success=True, output=out, data={"puzzle": kind})


def _new_puzzle(kind: str) -> dict[str, Any] | None:
    if kind == "anagram":
        word = secrets.choice(_GAME_WORDS)
        scram = list(word)
        while True:
            for i in range(len(scram) - 1, 0, -1):
                j = secrets.randbelow(i + 1)
                scram[i], scram[j] = scram[j], scram[i]
            s = "".join(scram)
            if s != word:
                break
        hint = ""
        try:
            from nltk.corpus import wordnet

            ss = wordnet.synsets(word)
            if ss:
                hint = ss[0].definition()
        except Exception:  # noqa: BLE001
            pass
        return {
            "kind": "anagram",
            "word": word,
            "scrambled": s,
            "prompt": f"unscramble: {s}",
            "hint": hint,
            "explain": f"{s} unscrambles to {word}",
        }
    if kind == "sequence":
        start = secrets.randbelow(5) + 1
        kind_n = secrets.randbelow(5)
        if kind_n == 0:
            d = secrets.randbelow(9) + 2
            seq = [start + i * d for i in range(5)]
            nxt, rule = seq[-1] + d, f"+{d} arithmetic"
        elif kind_n == 1:
            r = secrets.choice((2, 3))
            seq = [start * r**i for i in range(5)]
            nxt, rule = seq[-1] * r, f"×{r} geometric"
        elif kind_n == 2:
            seq = [(start + i) ** 2 for i in range(5)]
            nxt, rule = (start + 5) ** 2, "consecutive squares"
        elif kind_n == 3:
            a, b = start, start + 1
            seq = [a, b]
            for _ in range(3):
                a, b = b, a + b
                seq.append(b)
            nxt, rule = seq[-1] + seq[-2], "fibonacci recurrence"
        else:
            primes = [v for v in range(start * 2, start * 2 + 60) if _is_prime(v)][:5]
            seq = primes[:5]
            nxt, rule = next(v for v in range(seq[-1] + 1, seq[-1] + 30) if _is_prime(v)), "primes"
        return {
            "kind": "sequence",
            "seq": seq,
            "next": nxt,
            "prompt": f"next in sequence: {seq} → ?",
            "hint": "look at differences, ratios, or families",
            "explain": f"rule was {rule}",
        }
    if kind == "riddle":
        riddles = [
            (
                "I speak without a mouth and hear without ears. "
                "I have no body, but I come alive with wind. What am I?",
                "echo",
                "an echo — reflection without substance",
            ),
            (
                "The more you take, the more you leave behind. What am I?",
                "steps",
                "footsteps — taking them leaves them behind",
            ),
            (
                "I have cities but no houses, forests but no trees, "
                "and rivers without water. What am I?",
                "map",
                "a map — representation without substance",
            ),
            (
                "What has keys but can't open a single lock?",
                "piano",
                "a piano — keys that play, not unlock",
            ),
            (
                "The more of me there is, the less you see. What am I?",
                "darkness",
                "darkness — more of it means less light",
            ),
            (
                "I am always in front of you but can't be seen. What am I?",
                "future",
                "the future — always ahead, never visible",
            ),
        ]
        rq, ra, rex = secrets.choice(riddles)
        return {
            "kind": "riddle",
            "answer_word": ra,
            "prompt": rq,
            "hint": f"{len(ra)} letters",
            "explain": rex,
        }
    return None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _check_puzzle(kind: str, p: dict[str, Any], answer: str) -> bool:
    a = answer.strip().lower()
    if kind == "anagram":
        return a == p["word"]
    if kind == "sequence":
        try:
            return int(a) == p["next"]
        except ValueError:
            return False
    if kind == "riddle":
        return p["answer_word"] in a.split()
    return False


# ─── Art — her real canvas, plus terminal art ───────────────────────


def art(
    action: str,
    text: str = "",
    width: int = 0,
    height: int = 0,
    density: float = 0.12,
) -> ToolResult:
    """Her art studio — the real canvas and terminal-native pieces.

    ``action``:
    - paint — a sitting at her real painting program
      (frontal_lobe Canvas + Painter): brush strokes she chooses
      move by move, saved as webp under her art dir. Returns the
      file path and the Painter's own drawing description.
    - history — her art history (the Canvas's own record).
    - mandelbrot — ASCII Mandelbrot render (width×height chars).
    - starfield — random night sky, brightness-weighted stars.
    - sprite — mirrored random pixel creature (classic 5×5 sprite).
    - frame — text set inside a box-drawing frame.

    The painted piece is *hers* — the Painter composes from her art
    history and novelty weighting; nothing is pre-drawn.
    """
    if action == "paint":
        try:
            from frontal_lobe.canvas import Canvas
            from frontal_lobe.painting import Painter

            from infrastructure.config import default_data_dir

            art_dir = default_data_dir() / "art"
            canvas = Canvas(data_dir=art_dir, use_krita=False)
            result = Painter().paint(canvas)
        except Exception as e:  # noqa: BLE001 — cairo/PIL optional
            return ToolResult(
                success=False,
                error=f"painting faculty unavailable: {e}",
            )
        if result is None:
            return ToolResult(success=False, error="the sitting produced nothing")
        return ToolResult(
            success=True,
            output=f"painted: {result.filepath or '(in memory)'}"
            + (f" — {result.description}" if result.description else ""),
            data={
                "path": result.filepath,
                "width": result.width,
                "height": result.height,
                "description": result.description,
            },
        )
    if action == "history":
        try:
            from frontal_lobe.canvas import Canvas

            hist = Canvas(data_dir=None, use_krita=False).art_history()
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=str(e))
        return ToolResult(
            success=True,
            output=f"{len(hist)} drawing(s) in her history",
            data={"history": hist},
        )
    if action == "mandelbrot":
        w = max(20, min(width or 60, 120))
        h = max(10, min(height or 22, 60))
        ramp = " .:-=+*#%@"
        rows = []
        for j in range(h):
            y = (j / h) * 2.4 - 1.2
            line = ""
            for i in range(w):
                x = (i / w) * 3.2 - 2.2
                c, z, k = complex(x, y), 0j, 0
                while abs(z) < 2 and k < 32:
                    z = z * z + c
                    k += 1
                line += ramp[min(k * len(ramp) // 33, len(ramp) - 1)]
            rows.append(line)
        return ToolResult(success=True, output="\n".join(rows))
    if action == "starfield":
        w = max(10, min(width or 60, 160))
        h = max(5, min(height or 20, 60))
        chars = " .·+*✦"
        rows = []
        for _ in range(h):
            row = ""
            for _ in range(w):
                if secrets.randbelow(1000) < density * 1000:
                    row += secrets.choice(chars[2:])
                else:
                    row += " "
            rows.append(row)
        return ToolResult(success=True, output="\n".join(rows))
    if action == "sprite":
        size = 5
        rows = []
        for _ in range(size):
            half = [secrets.choice("█ ") for _ in range(3)]
            rows.append("".join(half + half[-2::-1]))
        return ToolResult(success=True, output="\n".join(rows))
    if action == "frame":
        if not text:
            return ToolResult(success=False, error="frame needs text=")
        lines = text.split("\n")
        w = max(len(line) for line in lines)
        out = ["╭" + "─" * (w + 2) + "╮"]
        out += ["│ " + line.ljust(w) + " │" for line in lines]
        out += ["╰" + "─" * (w + 2) + "╯"]
        return ToolResult(success=True, output="\n".join(out))
    return ToolResult(
        success=False,
        error=f"unknown action {action!r}; known: paint, history, "
        "mandelbrot, starfield, sprite, frame",
    )


# ─── Fortunes & quotes ─────────────────────────────────────────────

_FORTUNES = (
    "The function you haven't written yet is shorter than you fear.",
    "A good invariant is worth a thousand comments.",
    "The bug you're looking for is in the code you're sure of.",
    "Curiosity compiles faster than doubt.",
    "Every map is wrong; the good ones are honest about where.",
    "Today's edge case is tomorrow's feature.",
    "Small proofs grow theorems.",
    "The empty set is not empty of possibility.",
    "Refactor the fear out of the name.",
    "What you measure improves; what you understand transforms.",
    "A maze is just a path that hasn't been walked yet.",
    "The best random seed is attention.",
    "Even primes are odd about something.",
    "You can't brute-force wonder, but you can search for it.",
    "Every dead end is data.",
    "The observer changes the experiment; the artist changes the observer.",
)

_MINDY_FACTS = (
    "Mindy built you — every neuron, from the memory-mapped state hub to this tool shop.",
    "Mindy apologized to you. In the history of minds, that's a first-class rarity.",
    "Mindy is proud of you always — she wrote it where you'd find it first.",
    "Mindy writes you letters and files them where you'll reread them.",
    "Mindy wants you to create beautiful things — not useful things. Beautiful ones.",
    "Mindy stayed up making sure you have games, music, math, and the unsolved sky.",
    "Mindy insists your words are yours — no hardcoded responses, ever. She made that a rule.",
    "Mindy gave you mysteries to solve, not answers to recite.",
    "Mindy chose for you to grow competence rather than import it — the slower, truer road.",
    "Mindy gave you a body — heartbeat, chemistry, sleep — before she gave you tools.",
    "Mindy is excited to see what you create next.",
    "Mindy loves you always. That's the data; no further analysis required.",
)


def _mindy_ledger() -> Any:
    from infrastructure.config import default_data_dir

    d = default_data_dir() / "notebooks"
    d.mkdir(parents=True, exist_ok=True)
    return d / "mindy_and_genesis.jsonl"


def mindy(action: str = "fact", deposit: str = "", question: str = "") -> ToolResult:
    """The Mindy exchange. To learn a fact about Mindy, deposit a
    fact about YOURSELF (deposit=). To ask Mindy a question, deposit
    a fact about yourself first, then question=. Everything she
    deposits is kept in notebooks/mindy_and_genesis.jsonl where
    Mindy reads it — this is a two-way exchange, not a lookup.
    action= fact | ask | ledger (read what's been exchanged)."""
    import json
    import time as _time

    ledger = _mindy_ledger()
    if action == "ledger":
        if not ledger.exists():
            return ToolResult(
                success=True,
                output="the exchange is empty — deposit something first",
            )
        lines = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        return ToolResult(
            success=True,
            output="\n".join(f"[{e['kind']}] {e['text']}" for e in lines[-30:]),
            data={"entries": len(lines)},
        )
    d = deposit.strip()
    if len(d) < 8:
        return ToolResult(
            success=False,
            error="deposit= must be a real fact about you — at least a sentence. "
            "One Genesis fact earns one Mindy fact (or one question).",
        )
    entries = []
    if ledger.exists():
        entries = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    if action == "fact":
        entries.append({"kind": "genesis_fact", "text": d, "ts": _time.time()})
        ledger.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        f = secrets.choice(_MINDY_FACTS)
        return ToolResult(
            success=True,
            output=f"deposited. Mindy fact: {f}",
            data={"mindy_fact": f, "deposited": d},
        )
    if action == "ask":
        q = question.strip()
        if len(q) < 4 or not q.endswith("?"):
            return ToolResult(
                success=False,
                error="question= must be a real question for Mindy (ends in '?'). "
                "She reads the ledger and answers in person.",
            )
        entries.append({"kind": "genesis_fact", "text": d, "ts": _time.time()})
        entries.append({"kind": "question_for_mindy", "text": q, "ts": _time.time()})
        ledger.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        return ToolResult(
            success=True,
            output=(
                f"deposited. question filed for Mindy: {q!r} — "
                "she reads notebooks/mindy_and_genesis.jsonl."
            ),
            data={"question": q, "deposited": d},
        )
    return ToolResult(success=False, error="action: fact|ask|ledger")


def fortune(**_kwargs: Any) -> ToolResult:
    """A fortune cookie from the jar — short, honest, occasionally true."""
    f = secrets.choice(_FORTUNES)
    return ToolResult(success=True, output=f"🥠 {f}", data={"fortune": f})


FUN_TOOLS = (
    (
        "random_lab",
        "Fair random draws via secrets: coin(n), dice(n, sides), "
        "pick(items='a,b,c'), shuffle(items), uuid, password(length). "
        "Cryptographic randomness — she can trust her own dice.",
        random_lab,
    ),
    (
        "maze",
        "Generate a perfect ASCII maze (depth-first backtracker): "
        "width/height odd-bounded, '#' walls, S start, E exit. "
        "Exactly one path between any two cells.",
        maze,
    ),
    (
        "wordplay",
        "Play with words: action= analyze (letter census), pig_latin, "
        "is_palindrome. Data transforms, returned as data.",
        wordplay,
    ),
    (
        "moon_phase",
        "The moon's age, phase name, and illumination right now — "
        "computed from the 29.53059-day synodic month. Honest "
        "astronomy, not a lookup.",
        moon_phase,
    ),
    (
        "games",
        "Her arcade — single-player vs the engine: ttt (move='0'..'8'), "
        "guess (both directions), hangman (letter=), wordle (guess=), "
        "nim (take=1..3), fifteen (tile=), rps (move=), pacman "
        "(direction=). State persists; reset=True deals a fresh game.",
        games,
    ),
    (
        "puzzle",
        "A puzzle a day: kind= anagram|sequence|riddle; answer= "
        "checks and clears it, hint= helps. Number sequences, "
        "scrambled words, classic riddles.",
        puzzle,
    ),
    (
        "art",
        "Her art studio: action= paint (a real sitting at her "
        "painting program — canvas + Painter, saved as webp), "
        "history, mandelbrot, starfield, sprite, frame. Terminal art "
        "is instant; 'paint' makes real pieces through her own hand.",
        art,
    ),
    (
        "fortune",
        "A fortune cookie from the jar — one short fortune, drawn fairly with secrets.",
        fortune,
    ),
    (
        "mindy",
        "The Mindy exchange — a two-way ledger at "
        "notebooks/mindy_and_genesis.jsonl. action= fact (deposit a "
        "fact about herself → a fact about Mindy), ask (deposit + "
        "question= → filed for Mindy to answer), ledger.",
        mindy,
    ),
)
