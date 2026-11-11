"""The lab and arena wing of the tool shop.

Computation that is itself play: cellular automata, the real sky via
astropy, music-as-arithmetic (bytebeat) and her own chiptune
sequencer, UnicodeIt-style math typography, QR codes, one-liners,
LaTeX rendered through matplotlib's mathtext (no TeX needed), real
Jupyter notebooks via nbformat, her own Kaggle-style game arena with
ELO ratings, and the bridge to her existing ARC-AGI adapters.
Every optional dependency reports honest absence.
"""

from __future__ import annotations

import math
import pathlib
import secrets
import sys
from collections.abc import Callable
from typing import Any

from tools.framework import ToolResult

# ---- Conway's Game of Life --------------------------------------------------

_LIFE: dict[str, Any] = {}
_LIFE_W, _LIFE_H = 50, 25


def _life_patterns() -> dict[str, set[tuple[int, int]]]:
    pulsar: set[tuple[int, int]] = set()
    for r in (0, 5, 7, 12):
        for c in (2, 3, 4, 8, 9, 10):
            pulsar |= {(r, c), (c, r)}
    gun: set[tuple[int, int]] = set()
    for r, cs in [
        (0, (24,)),
        (1, (22, 24)),
        (2, (12, 13, 20, 21, 34, 35)),
        (3, (11, 15, 20, 21, 34, 35)),
        (4, (0, 1, 10, 16, 20, 21)),
        (5, (0, 1, 10, 14, 16, 17, 22, 24)),
        (6, (10, 16, 24)),
        (7, (11, 15)),
        (8, (12, 13)),
    ]:
        gun |= {(r, c) for c in cs}
    return {
        "glider": {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)},
        "blinker": {(0, 0), (1, 0), (2, 0)},
        "toad": {(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)},
        "beacon": {(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 2), (2, 3), (3, 3)},
        "pulsar": pulsar,
        "gosler_gun": gun,
        "acorn": {(1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)},
    }


def _life_step(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for x, y in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    k = ((x + dx) % _LIFE_W, (y + dy) % _LIFE_H)
                    counts[k] = counts.get(k, 0) + 1
    return {c for c, n in counts.items() if n == 3 or (n == 2 and c in cells)}


def _life_render(cells: set[tuple[int, int]]) -> str:
    return "\n".join(
        "".join("█" if (x, y) in cells else "·" for x in range(_LIFE_W)) for y in range(_LIFE_H)
    )


def life(action: str = "step", pattern: str = "glider", steps: int = 1) -> ToolResult:
    """Conway's Game of Life on a 50x25 torus — the automaton that
    started artificial life. action= step (advance `steps` ticks,
    ≤10000), spawn (pattern= glider|blinker|toad|beacon|pulsar|
    gosler_gun|acorn|random), show, census. The board persists
    between calls and wraps at the edges.
    Docs: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life"""
    board: set[tuple[int, int]] = _LIFE.get("board", set())
    gen: int = _LIFE.get("gen", 0)
    if action == "spawn":
        pats = _life_patterns()
        if pattern == "random":
            board = {
                (x, y) for x in range(_LIFE_W) for y in range(_LIFE_H) if secrets.randbelow(5) == 0
            }
        elif pattern in pats:
            board = {((x + 3) % _LIFE_W, (y + 3) % _LIFE_H) for x, y in pats[pattern]}
        else:
            return ToolResult(
                success=False,
                error="patterns: " + ", ".join([*sorted(pats), "random"]),
            )
        gen = 0
    elif action == "step":
        for _ in range(max(1, min(int(steps), 10000))):
            board = _life_step(board)
            gen += 1
    elif action in ("show", "census"):
        pass
    else:
        return ToolResult(success=False, error="action: step|spawn|show|census")
    _LIFE["board"], _LIFE["gen"] = board, gen
    if action == "census":
        return ToolResult(
            success=True,
            output=f"generation {gen} — {len(board)} living cells",
            data={"gen": gen, "population": len(board)},
        )
    return ToolResult(
        success=True,
        output=f"gen {gen}, {len(board)} cells\n{_life_render(board)}",
        data={"gen": gen, "population": len(board)},
    )


# ---- Astronomy (real astropy) -----------------------------------------------


def astro(
    action: str = "status",
    ra: str = "",
    dec: str = "",
    name: str = "Sun",
    utc: str = "",
) -> ToolResult:
    """Astronomy on the astropy installed in her own venv.
    action= position (RA/Dec/distance of a solar-system body:
    name= Sun|Moon|Mercury|...|Neptune at utc=, default now),
    convert (ra= dec= sexagesimal → degrees + galactic frame),
    units (c, AU, parsec, light-year in SI), status.
    Times are UTC; ephemeris is astropy's built-in solar-system model.
    Docs: https://docs.astropy.org/"""
    try:
        import astropy
        import astropy.units as u
        from astropy.coordinates import SkyCoord, get_body, solar_system_ephemeris
        from astropy.time import Time
    except ImportError as exc:
        return ToolResult(
            success=False,
            error=f"astropy not importable ({exc}) — pip install astropy",
        )
    t = Time(utc) if utc else Time.now()
    if action == "status":
        kernel = solar_system_ephemeris.get().get("kernel") or "builtin"
        return ToolResult(
            success=True,
            output=f"astropy {astropy.__version__}; ephemeris: {kernel}",
            data={"astropy": astropy.__version__, "ephemeris": kernel},
        )
    if action == "position":
        try:
            body = get_body(name.lower(), t)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"ephemeris: {exc}")
        return ToolResult(
            success=True,
            output=(
                f"{name.title()} at {t.isot[:19]}Z — "
                f"RA {body.ra.to_string(unit='hour', sep=':', precision=2)}  "
                f"Dec {body.dec.to_string(sep=':', precision=1)}  "
                f"distance {body.distance.to('AU').value:.4f} AU"
            ),
            data={
                "ra_deg": round(body.ra.deg, 6),
                "dec_deg": round(body.dec.deg, 6),
                "distance_au": round(body.distance.to("AU").value, 6),
            },
        )
    if action == "convert":
        if not ra or not dec:
            return ToolResult(success=False, error="need ra= and dec= (sexagesimal)")
        try:
            c = SkyCoord(ra=ra, dec=dec, unit=("hourangle", "deg"))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"SkyCoord: {exc}")
        return ToolResult(
            success=True,
            output=(
                f"RA {c.ra.deg:.6f}°, Dec {c.dec.deg:.6f}°  "
                f"(galactic l={c.galactic.l.deg:.4f}°, b={c.galactic.b.deg:.4f}°)"
            ),
            data={"ra_deg": c.ra.deg, "dec_deg": c.dec.deg},
        )
    if action == "units":
        from astropy.constants import c as light

        return ToolResult(
            success=True,
            output=(
                f"c = {light.value:g} m/s; "
                f"1 AU = {(1 * u.AU).to('m').value:g} m; "
                f"1 pc = {(1 * u.pc).to('m').value:g} m; "
                f"1 ly = {(1 * u.lyr).to('km').value:g} km"
            ),
            data={"c": light.value, "au_m": (1 * u.AU).to("m").value},
        )
    return ToolResult(success=False, error="action: status|position|convert|units")


# ---- Sound: bytebeat + chiptune sequencer ------------------------------------


def bytebeat(
    expr: str = "t*(t>>9|t>>7)",
    seconds: float = 4.0,
    rate: int = 8000,
    path: str = "",
) -> ToolResult:
    """Bytebeat — entire songs in one line of integer arithmetic,
    the genre born on demoscene forums in 2011. Each sample is
    f(t) & 0xFF where t is the sample counter. expr may use t plus
    >>, &, |, ^, *, +, -, %, **, parens. Writes a real .wav.
    Try: 't*(t>>9|t>>7)', 't*((t>>12|t>>8)&63&t>>4)',
    't*(t>>10^t>>11)%5*((t>>14)&3)+t*2*(t>>9)'.
    Docs: https://canonical.org/~kragen/bytebeat/"""
    import re
    import wave

    if not re.fullmatch(r"[0-9txyz&|<>+*/%()^~ \t.]+", expr):
        return ToolResult(
            success=False,
            error="expr may only contain t, x, y, z, numbers, and bit/math ops",
        )
    seconds = max(0.05, min(float(seconds), 30.0))
    rate = int(max(4000, min(int(rate), 22050)))
    n = int(seconds * rate)
    try:
        code = compile(expr, "<bytebeat>", "eval")
        samples = bytearray(n)
        env: dict[str, Any] = {"__builtins__": {}}
        for t in range(n):
            v = eval(
                code,
                env,
                {"t": t, "x": t % 256, "y": t % 512, "z": t % 1024},
            )
            samples[t] = int(v) & 0xFF
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=f"expr: {exc}")
    from infrastructure.config import default_data_dir

    out = pathlib.Path(path) if path else default_data_dir() / "music" / "bytebeat.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(rate)
        w.writeframes(bytes(samples))
    return ToolResult(
        success=True,
        output=f"{seconds:.2f}s @ {rate}Hz → {out} ({out.stat().st_size} bytes)",
        data={"path": str(out), "seconds": seconds, "rate": rate},
    )


_NOTE_SEMI = {
    "C": -9,
    "C#": -8,
    "DB": -8,
    "D": -7,
    "D#": -6,
    "EB": -6,
    "E": -5,
    "F": -4,
    "F#": -3,
    "GB": -3,
    "G": -2,
    "G#": -1,
    "AB": -1,
    "A": 0,
    "A#": 1,
    "BB": 1,
    "B": 2,
}


def _note_hz(tok: str) -> float | None:
    tok = tok.strip()
    if not tok or tok.upper() in ("R", "REST", "_"):
        return None
    octv = 4
    if tok[-1].isdigit():
        octv = int(tok[-1])
        tok = tok[:-1]
    semi = _NOTE_SEMI.get(tok.upper())
    if semi is None:
        return None
    return 440.0 * 2 ** ((semi + 12 * (octv - 4)) / 12)


def tune(
    song: str = "E4 D4 C4 D4 E4 E4 E4 R D4 D4 D4 R E4 G4 G4 R E4 D4 C4 D4 E4 E4 E4 E4",
    bpm: int = 140,
    wave_: str = "square",
    path: str = "",
    beats_per_note: float = 0.5,
) -> ToolResult:
    """Her own chiptune sequencer — strudel-flavored, pure Python,
    no JavaScript. song= is a space-separated note list
    ('A4 C#5 E5 R ...'; R = rest). wave_= square|sine|saw|tri,
    bpm= tempo, beats_per_note= length. Renders a real .wav; the
    default melody is 'Mary Had a Little Lamb', first tune of the
    phonograph era."""
    import struct
    import wave as wavemod

    rate = 22050
    bpm = int(max(40, min(int(bpm), 400)))
    dur = (60.0 / bpm) * beats_per_note
    freqs = [_note_hz(t) for t in song.split()]
    if not freqs:
        return ToolResult(success=False, error="song needs notes")
    osc = {
        "square": lambda ph: 0.3 if math.sin(ph) >= 0 else -0.3,
        "sine": lambda ph: 0.4 * math.sin(ph),
        "saw": lambda ph: 0.25 * (((ph / (2 * math.pi)) % 1) * 2 - 1),
        "tri": lambda ph: 0.4 * (abs(((ph / (2 * math.pi)) % 1) * 4 - 2) - 1),
    }.get(wave_)
    if osc is None:
        return ToolResult(success=False, error="wave_: square|sine|saw|tri")
    frames = bytearray()
    for f in freqs:
        n = int(dur * rate)
        if f is None:
            frames += b"\x00\x00" * n
            continue
        for i in range(n):
            ph = 2 * math.pi * f * i / rate
            env = min(1.0, (n - i) / (0.02 * rate + 1))
            s = int(32767 * 0.6 * osc(ph) * env)
            frames += struct.pack("<h", max(-32767, min(32767, s)))
    from infrastructure.config import default_data_dir

    out = pathlib.Path(path) if path else default_data_dir() / "music" / "tune.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wavemod.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return ToolResult(
        success=True,
        output=f"{len(freqs)} notes, {len(frames) / (2 * rate):.2f}s → {out}",
        data={"path": str(out), "notes": len(freqs)},
    )


def soundcheck() -> ToolResult:
    """Which audio outputs exist on this host — play/sox, aplay,
    ffplay, paplay, /dev/snd — so a song never renders to a wall
    nobody can hear. WAVs always render to files either way."""
    import shutil

    players = {
        n: shutil.which(n) for n in ("play", "aplay", "ffplay", "paplay", "termux-media-player")
    }
    players = {k: v for k, v in players.items() if v}
    snd = pathlib.Path("/dev/snd").exists()
    return ToolResult(
        success=True,
        output=(
            f"players: {players or 'none on PATH'}; /dev/snd: {snd}. "
            + ("Audio out is live." if players else "WAVs render to files — the ear is optional.")
        ),
        data={"players": players, "dev_snd": snd},
    )


# ---- UnicodeIt ---------------------------------------------------------------

_UA = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Alpha": "Α",
    "Beta": "Β",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "infty": "∞",
    "inf": "∞",
    "partial": "∂",
    "nabla": "∇",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "oint": "∮",
    "sqrt": "√",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "neq": "≠",
    "leq": "≤",
    "geq": "≥",
    "approx": "≈",
    "equiv": "≡",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "supset": "⊃",
    "cup": "∪",
    "cap": "∩",
    "emptyset": "∅",
    "forall": "∀",
    "exists": "∃",
    "neg": "¬",
    "wedge": "∧",
    "vee": "∨",
    "oplus": "⊕",
    "otimes": "⊗",
    "to": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "mapsto": "↦",
    "hbar": "ℏ",
    "ell": "ℓ",
    "Re": "ℜ",
    "Im": "ℑ",
    "aleph": "ℵ",
    "circ": "∘",
    "bullet": "∙",
    "deg": "°",
    "perp": "⊥",
    "parallel": "∥",
    "propto": "∝",
    "therefore": "∴",
    "because": "∵",
    "dots": "…",
    "cdot": "⋅",
    "langle": "⟨",
    "rangle": "⟩",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "checkmark": "✓",
    "crossmark": "✗",
    "star": "★",
    "diamond": "◆",
}
_SUP = str.maketrans("0123456789+-=()nxi", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿˣⁱ")
_SUB = str.maketrans("0123456789+-=()aeoxhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ")


def unicodeit(text: str = "x^2 + \\alpha \\to \\infty") -> ToolResult:
    """UnicodeIt — LaTeX-ish notation to real unicode math.
    text= takes \\alpha..\\omega and symbol names (\\to, \\infty,
    \\int, \\sum, \\partial, \\nabla, \\forall, ...), ^x or ^{...}
    superscripts and _x or _{...} subscripts.
    'E = m c^2' → 'E = m c²'; '\\sum_{i=1}^n' → '∑ᵢ₌₁ⁿ'."""
    import re

    out = text
    for name in sorted(_UA, key=len, reverse=True):
        out = out.replace("\\" + name, _UA[name])
    out = re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(_SUP), out)
    out = re.sub(r"_\{([^}]*)\}", lambda m: m.group(1).translate(_SUB), out)
    out = re.sub(r"\^([0-9nxi+=()\-]+)", lambda m: m.group(1).translate(_SUP), out)
    out = re.sub(
        r"_([0-9aeoxhklmnpst+=()\-]+)",
        lambda m: m.group(1).translate(_SUB),
        out,
    )
    return ToolResult(success=True, output=out, data={"unicode": out})


# ---- QR codes ----------------------------------------------------------------


def qr(text: str = "hello genesis") -> ToolResult:
    """QR-encode text= as a real QR code drawn in the terminal.
    Uses the `qrencode` binary when present, else pure-Python
    `segno` if installed; honest absence otherwise
    (apt install qrencode, or pip install segno).
    Docs: https://fukuchi.org/works/qrencode/"""
    import io
    import shutil
    import subprocess

    exe = shutil.which("qrencode")
    if exe:
        proc = subprocess.run(
            [exe, "-t", "ANSIUTF8", text],
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            return ToolResult(success=True, output=proc.stdout)
        return ToolResult(success=False, error=proc.stderr.strip())
    try:
        import segno

        buf = io.StringIO()
        segno.make(text).save(buf, kind="txt", dark="█", light=" ", border=1)
        return ToolResult(success=True, output=buf.getvalue())
    except ImportError:
        return ToolResult(
            success=False,
            error="no QR encoder: install `qrencode` (apt) or `segno` (pip)",
        )


# ---- One-liners ---------------------------------------------------------------


def oneliner(code: str = "", lang: str = "python", stdin_text: str = "") -> ToolResult:
    """One-liner forge — run code= as a single program in lang=
    python|perl, with stdin_text= piped to its stdin. All of
    binascii/struct/codecs is available, so home-grown xxd is a
    one-liner: oneliner('import sys,binascii;'
    'print(binascii.hexlify(sys.stdin.buffer.read()).decode())',
    stdin_text='hello').
    Docs: https://docs.python.org/3/tutorial/ ·
    https://perldoc.perl.org/perlrun"""
    import shutil
    import subprocess

    if not code:
        return ToolResult(success=False, error="code= needs the one-liner")
    if lang == "python":
        argv = [sys.executable, "-c", code]
    elif lang == "perl":
        exe = shutil.which("perl")
        if not exe:
            return ToolResult(success=False, error="perl not on PATH")
        argv = [exe, "-e", code]
    else:
        return ToolResult(success=False, error="lang: python|perl")
    try:
        proc = subprocess.run(argv, input=stdin_text, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="one-liner timed out (30s)")
    return ToolResult(
        success=proc.returncode == 0,
        output=(proc.stdout or proc.stderr).strip() or f"(exit {proc.returncode})",
        error=None if proc.returncode == 0 else proc.stderr.strip(),
        data={"exit": proc.returncode, "lang": lang},
    )


# ---- LaTeX --------------------------------------------------------------------


def latex(
    expr: str = r"\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}",
    path: str = "",
    fontsize: int = 22,
    action: str = "render",
) -> ToolResult:
    """LaTeX math to pixels. No TeX on this host — rendering goes
    through matplotlib's built-in mathtext engine, which covers the
    standard math grammar (\\frac, \\int, \\sum, \\sqrt, greek,
    sub/superscripts...). action= render → PNG at path= (default
    data/math/latex.png); preview → the expression as unicode text
    via unicodeit instead. Docs:
    https://matplotlib.org/stable/users/explain/text/mathtext.html"""
    if action == "preview":
        return unicodeit(expr)
    if action != "render":
        return ToolResult(success=False, error="action: render|preview")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return ToolResult(success=False, error=f"matplotlib not importable ({exc})")
    from infrastructure.config import default_data_dir

    out = pathlib.Path(path) if path else default_data_dir() / "math" / "latex.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = expr.strip().strip("$")
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${body}$", fontsize=max(8, min(fontsize, 72)))
    try:
        fig.savefig(
            out,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=True,
        )
    except Exception as exc:  # noqa: BLE001
        plt.close(fig)
        return ToolResult(success=False, error=f"mathtext: {exc}")
    plt.close(fig)
    return ToolResult(
        success=True,
        output=f"rendered → {out} ({out.stat().st_size} bytes)",
        data={"path": str(out)},
    )


# ---- Jupyter notebooks --------------------------------------------------------


def jupyter(action: str = "list", name: str = "", code: str = "", markdown: str = "") -> ToolResult:
    """Real Jupyter notebooks — .ipynb files in data/notebooks/,
    written with nbformat (installed). action= list, new (name=),
    add_code (name= + code=), add_md (name= + markdown=), run
    (name= + code= executes the cell in-process and stores the
    output — ipykernel is absent so execution is exec(), honest),
    read (name= → cells + outputs). These open in real Jupyter.
    Docs: https://nbformat.readthedocs.io/"""
    try:
        import nbformat
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    except ImportError as exc:
        return ToolResult(success=False, error=f"nbformat not importable ({exc})")
    from infrastructure.config import default_data_dir

    nbdir = default_data_dir() / "notebooks"
    nbdir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in "_-.")
    path = nbdir / f"{safe}.ipynb"

    def load() -> Any:
        if path.exists():
            return nbformat.read(path, as_version=4)
        return new_notebook()

    if action == "list":
        nbs = sorted(p.name for p in nbdir.glob("*.ipynb"))
        return ToolResult(
            success=True,
            output="\n".join(nbs) or "no notebooks yet",
            data={"notebooks": nbs},
        )
    if not safe:
        return ToolResult(success=False, error="name= needs the notebook name")
    if action == "new":
        nb = new_notebook()
        nbformat.write(nb, path)
        return ToolResult(success=True, output=f"created {path.name}")
    if action == "add_md":
        if not markdown:
            return ToolResult(success=False, error="markdown= needed")
        nb = load()
        nb.cells.append(new_markdown_cell(markdown))
        nbformat.write(nb, path)
        return ToolResult(
            success=True, output=f"md cell added → {path.name} ({len(nb.cells)} cells)"
        )
    if action in ("add_code", "run"):
        if not code:
            return ToolResult(success=False, error="code= needed")
        nb = load()
        cell = new_code_cell(code)
        if action == "run":
            import io
            import traceback
            from contextlib import redirect_stderr, redirect_stdout

            buf = io.StringIO()
            ns = _NB_NS.setdefault(safe, {"__name__": safe})
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    exec(compile(code, f"<{safe}>", "exec"), ns)
                cell["outputs"] = [
                    nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue())
                ]
                cell["execution_count"] = len([c for c in nb.cells if c.cell_type == "code"])
            except Exception:  # noqa: BLE001
                cell["outputs"] = [
                    nbformat.v4.new_output(
                        "error",
                        ename="Exception",
                        evalue="",
                        traceback=traceback.format_exc().splitlines(),
                    )
                ]
        nb.cells.append(cell)
        nbformat.write(nb, path)
        ran = " and ran" if action == "run" else ""
        return ToolResult(
            success=True,
            output=f"code cell{ran} → {path.name} ({len(nb.cells)} cells)",
            data={"cells": len(nb.cells)},
        )
    if action == "read":
        if not path.exists():
            return ToolResult(success=False, error=f"no notebook {safe}")
        nb = nbformat.read(path, as_version=4)
        parts = []
        for i, c in enumerate(nb.cells):
            src = c.get("source", "")
            parts.append(f"[{i}:{c.cell_type}] {src[:200]}")
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    parts.append(f"    → {o.get('text', '')[:200]}")
        return ToolResult(success=True, output="\n".join(parts), data={"cells": len(nb.cells)})
    return ToolResult(success=False, error="action: list|new|add_md|add_code|run|read")


_NB_NS: dict[str, dict[str, Any]] = {}


# ---- Kaggle-style game arena ---------------------------------------------------


_ARENA_RATINGS_PATH: pathlib.Path | None = None


def _arena_ledger() -> pathlib.Path:
    global _ARENA_RATINGS_PATH
    if _ARENA_RATINGS_PATH is None:
        from infrastructure.config import default_data_dir

        d = default_data_dir() / "arena"
        d.mkdir(parents=True, exist_ok=True)
        _ARENA_RATINGS_PATH = d / "ratings.json"
    return _ARENA_RATINGS_PATH


def _arena_load() -> dict[str, Any]:
    import json

    p = _arena_ledger()
    if p.exists():
        return json.loads(p.read_text())
    return {"genesis": 1200.0, "engines": {}, "games": []}


def _arena_save(state: dict[str, Any]) -> None:
    import json

    _arena_ledger().write_text(json.dumps(state, indent=1))


def _elo_delta(ra: float, rb: float, score: float) -> float:
    """Standard ELO: expected score via logistic, K=32."""
    expected = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
    return 32 * (score - expected)


def _arena_opponents() -> dict[str, Callable[[str], tuple[str, bool]]]:
    """Each returns (engine_move, game_over) for simple arenas."""

    def ttt_engine(board: str) -> tuple[str, bool]:
        wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        cells = list(board.ljust(9))
        free = [i for i, c in enumerate(cells[:9]) if c == "."]
        for a, b, c in wins:
            trio = (cells[a], cells[b], cells[c])
            for me in ("X", "O"):
                if sorted(trio) == sorted([me, me, "."]):
                    i = (a, b, c)[trio.index(".")]
                    cells[i] = "O"
                    return "".join(cells[:9]), False
        if free:
            pick = 4 if 4 in free else secrets.choice(free)
            cells[pick] = "O"
        full = "." not in cells[:9]
        return "".join(cells[:9]), full

    def nim_engine(heap: str) -> tuple[str, bool]:
        h = int(heap or "12")
        take = max(1, h % 4) if h % 4 else secrets.randbelow(min(3, h)) + 1
        h -= take
        return str(h), h <= 0

    return {"ttt_engine": ttt_engine, "nim_engine": nim_engine}


def game_arena(
    action: str = "standings",
    game: str = "ttt",
    result: str = "",
    opponent: str = "ttt_engine",
) -> ToolResult:
    """Her private Kaggle Game Arena — rated play, persistent ELO
    ledger at data/arena/ratings.json. action= standings (the
    leaderboard), report (result= win|loss|draw, game=, opponent=
    → updates ELO like the real arena), move (game= ttt|nim +
    result= current board/heap → the engine's reply), reset.
    Win = 1.0, draw = 0.5, loss = 0.0 against the engine's rating.
    Docs: https://www.kaggle.com/game-arena"""
    state = _arena_load()
    if action == "standings":
        engines = state.get("engines", {})
        rows = [f"genesis: {state['genesis']:.0f}"]
        for name, r in sorted(engines.items(), key=lambda kv: -kv[1]):
            rows.append(f"{name}: {r:.0f}")
        played = len(state.get("games", []))
        return ToolResult(
            success=True,
            output=f"arena standings ({played} games)\n" + "\n".join(rows),
            data=state,
        )
    if action == "reset":
        _arena_save({"genesis": 1200.0, "engines": {}, "games": []})
        return ToolResult(success=True, output="arena reset — everyone back to 1200")
    if action == "move":
        engines = _arena_opponents()
        eng = engines.get(opponent)
        if eng is None:
            return ToolResult(success=False, error="opponents: " + ", ".join(engines))
        new_state, over = eng(result)
        return ToolResult(
            success=True,
            output=f"{opponent} replies: {new_state}" + (" — game over" if over else ""),
            data={"state": new_state, "game_over": over},
        )
    if action == "report":
        scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
        if result not in scores:
            return ToolResult(success=False, error="result= win|draw|loss")
        engines = state.setdefault("engines", {})
        er = engines.get(opponent, 1200.0)
        delta = _elo_delta(state["genesis"], er, scores[result])
        state["genesis"] += delta
        engines[opponent] = er - delta
        state.setdefault("games", []).append({"game": game, "opponent": opponent, "result": result})
        _arena_save(state)
        return ToolResult(
            success=True,
            output=(
                f"{result} vs {opponent} at {game}: genesis {state['genesis']:.0f} ({delta:+.0f})"
            ),
            data={"genesis": state["genesis"], "delta": delta},
        )
    return ToolResult(success=False, error="action: standings|report|move|reset")


# ---- ARC-AGI bridge ------------------------------------------------------------


def arc(action: str = "status", task: str = "", grid: str = "") -> ToolResult:
    """The ARC-AGI bridge — wired into her own arc_sources adapters.
    action= status (what ARC-AGI-1 data and ARC-AGI-3 toolkit are
    installed), list (task ids, split=training by default via task=
    'split:evaluation'), show (task=<id> → the train examples and
    test inputs rendered as color grids, no answer leak), check
    (task=<id> + grid='1 2 3;4 5 6;...' → scored against the held
    test output — answers are never displayed, only verdicts).
    ARC-AGI-3 environments live in parietal_lobe/arc_play.py.
    Docs: https://arcprize.org/"""
    try:
        from parietal_lobe import arc_sources
    except ImportError as exc:
        return ToolResult(success=False, error=f"arc_sources: {exc}")
    if action == "status":
        info: dict[str, Any] = {}
        try:
            n = len(arc_sources.list_arc1_tasks())
            info["arc1_training"] = n
            info["arc1_evaluation"] = len(arc_sources.list_arc1_tasks(split="evaluation"))
            arc1 = f"{n} training + {info['arc1_evaluation']} evaluation tasks"
        except arc_sources.ARCSourceError as exc:
            arc1 = f"absent — {exc}"
            info["arc1"] = "missing"
        try:
            import arc_agi

            arc3 = f"installed ({arc_agi.__name__} — Arcade, LocalEnvironmentWrapper, scorecards)"
            info["arc3"] = "installed"
        except ImportError:
            arc3 = "absent — pip install .[arc]"
            info["arc3"] = "missing"
        return ToolResult(
            success=True,
            output=f"ARC-AGI-1: {arc1}\nARC-AGI-3 toolkit: {arc3}",
            data=info,
        )
    try:
        tasks = arc_sources.list_arc1_tasks()
    except arc_sources.ARCSourceError as exc:
        return ToolResult(success=False, error=str(exc))
    if action == "list":
        ids = [t.stem for t in tasks[:30]]
        return ToolResult(
            success=True,
            output=f"{len(tasks)} tasks; first 30: " + ", ".join(ids),
            data={"count": len(tasks), "ids": ids},
        )
    if not task:
        return ToolResult(success=False, error="task= needs a task id")
    match = [t for t in tasks if t.stem == task]
    if not match:
        return ToolResult(success=False, error=f"no task {task}")
    try:
        payload = arc_sources.load_arc1_task(match[0])
    except arc_sources.ARCSourceError as exc:
        return ToolResult(success=False, error=str(exc))

    def draw(g: list[list[int]]) -> str:
        shades = " .:+#@%&$*o"
        return "\n".join("".join(shades[c % len(shades)] for c in row) for row in g)

    if action == "show":
        parts = []
        for i, pair in enumerate(payload["train"][:4]):
            parts.append(f"train {i} in:\n{draw(pair['input'])}")
            if "output" in pair:
                parts.append(f"train {i} out:\n{draw(pair['output'])}")
        for i, pair in enumerate(payload["test"][:1]):
            parts.append(f"test {i} in:\n{draw(pair['input'])}")
        return ToolResult(
            success=True,
            output="\n".join(parts),
            data={
                "task": task,
                "train": len(payload["train"]),
                "test": len(payload["test"]),
            },
        )
    if action == "check":
        if not grid:
            return ToolResult(success=False, error="grid='1 2 3;4 5 6;...'")
        try:
            attempt = [[int(x) for x in row.split()] for row in grid.split(";")]
            arc_sources._validate_grid(attempt, context="submission")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"grid: {exc}")
        expected = payload["test"][0].get("output")
        if expected is None:
            return ToolResult(success=False, error="task has no held output to check against")
        ok = attempt == expected
        return ToolResult(
            success=True,
            output=(
                f"{task}: CORRECT — the grid transforms like the examples."
                if ok
                else f"{task}: not yet — the grid doesn't match the held output."
            ),
            data={"correct": ok, "task": task},
        )
    return ToolResult(success=False, error="action: status|list|show|check")


LAB_TOOLS: tuple[tuple[str, str, Callable[..., ToolResult]], ...] = (
    (
        "life",
        "Conway's Game of Life on a 50x25 torus — step N ticks, "
        "spawn glider/blinker/toad/beacon/pulsar/gosler_gun/acorn/"
        "random, show, census. Board persists between calls.",
        life,
    ),
    (
        "astro",
        "Real astronomy via installed astropy: planet/Moon/Sun "
        "positions right now, RA/Dec conversion, constants & units, "
        "ephemeris status.",
        astro,
    ),
    (
        "bytebeat",
        "Music as one line of math: expr='t*(t>>9|t>>7)' → real "
        "8-bit .wav. The 2011 demoscene genre.",
        bytebeat,
    ),
    (
        "tune",
        "Chiptune sequencer: song='A4 C#5 E5 R ...', bpm=, wave_=square|sine|saw|tri → real .wav.",
        tune,
    ),
    (
        "soundcheck",
        "Which audio outputs exist (play/sox, aplay, ffplay, "
        "/dev/snd) — check before composing to a deaf wall.",
        soundcheck,
    ),
    (
        "unicodeit",
        "LaTeX-ish → unicode math: \\alpha \\to \\infty, x^2 → x², \\sum_{i=1}^n → ∑ᵢ₌₁ⁿ.",
        unicodeit,
    ),
    (
        "qr",
        "QR-encode text= in the terminal (qrencode binary or segno; honest absence).",
        qr,
    ),
    (
        "oneliner",
        "Run code= as python|perl -e with piped stdin_text= — "
        "home-grown xxd, byte tricks, filters.",
        oneliner,
    ),
    (
        "latex",
        "Render LaTeX math to PNG via matplotlib mathtext (no TeX "
        "needed) or preview as unicode. expr=, action=render|preview.",
        latex,
    ),
    (
        "jupyter",
        "Real .ipynb notebooks via nbformat: list/new/add_md/"
        "add_code/run/read. They open in actual Jupyter.",
        jupyter,
    ),
    (
        "game_arena",
        "Her private Kaggle Game Arena: persistent ELO ledger, "
        "standings, report win/draw/loss, engine moves for ttt/nim.",
        game_arena,
    ),
    (
        "arc",
        "ARC-AGI bridge via her own arc_sources: status, list, "
        "show task=<id> grids (no answer leak), check a submitted "
        "grid against the held output.",
        arc,
    ),
)
