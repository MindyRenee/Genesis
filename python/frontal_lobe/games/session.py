"""External game session — the play path for game mode.

In game mode the stream arriving at her prompt is world state, not
conversation. This session runs each turn through her play faculties
— the parietal grid perception and spatial policy in
``games/spatial``, bound to the shared ``TaskCompetence`` substrate
so skill learned here consolidates with everything else she has
learned — and the reply is the action she chose, not a sentence
about the frame. The world stays outside her; only the stream
crosses the boundary.

Outcomes are felt, not just recorded: the same couplings internal
play uses (step feedback -> dopamine, level -> reward, death ->
dip + vigilance + a trickle of cortisol), and her inner voice
narrates the salient moments through ``_compose_play_moment`` —
self-directed speech from her own language system.

Wire protocol — one line per turn, all fields optional except the
frame itself:

    FRAME <rows, '/'-separated hex digits | @/path/to/frame.txt>
        | ACTIONS 1,2,3 | LEVELS a/b [| GAME name] [| EVENT win|over]

The ``@path`` form reads the frame from a file (one hex row per
line) — a full 64×64 grid exceeds the terminal's canonical input
limit, so the file is her "screen" for large frames.

``EVENT`` marks the terminal outcome of the step just taken (the
adapter resets immediately on game over, so the terminal belongs to
the frame that follows the reset). ``LEVELS`` advancing is the
goal signal — the session marks it itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_NOREPINEPHRINE,
)
from genesis_client.swallow import note_swallowed
from world.bridge import WorldBridge

from .spatial.agent import SpatialAgent

if TYPE_CHECKING:
    from mind import Mind

_VOICE_CAP = 12  # inner-voice budget per session — salience, not chatter

# The wire protocol's color space: frame cells are hex digits 0-f,
# which are the 16 colors of the ARC palette itself (the official
# arc_agi.rendering.COLOR_MAP, alpha dropped). Kept here rather than
# imported so the session layer stays free of the adapter's package.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99),
    (0x66, 0x66, 0x66), (0x33, 0x33, 0x33), (0x00, 0x00, 0x00),
    (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC), (0xF9, 0x3C, 0x31),
    (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30),
    (0xA3, 0x56, 0xD6),
)


class GameSession:
    """One external game stream, played by her own faculties."""

    def __init__(self, mind: Mind) -> None:
        self._mind = mind
        self._agent = SpatialAgent(
            task_competence=mind.brain.frontal.task_competence,
        )
        # World bridge — every frame is published to the tensor slab
        # and broadcast through the global workspace, so a game frame
        # reaches attention, memory, self, ToM, working memory, and
        # curiosity rather than staying inside the play loop.
        frontal = mind.brain.frontal
        language = getattr(frontal, "language", None)
        self._bridge = WorldBridge(
            workspace=getattr(frontal, "global_workspace", None),
            client=mind.client,
            brain_waves=lambda: getattr(language, "current_brain_waves", None),
        )
        self._voiced = 0
        self._deaths = 0
        self._prev_levels = 0
        self._seen_games: set[str] = set()
        self._dry = 0
        # Decision telemetry — the basis behind the last action this
        # session returned (which drive won, how decisive, what else
        # contended). ARC-style harnesses attach it to each step as
        # the recorded reasoning.
        self.last_decision: dict[str, Any] | None = None
        # Inner-voice text emitted during the current turn, if any —
        # rides the decision record so the harness hears the moment
        # that mattered, not a stale narration.
        self._last_voiced: str | None = None

    # ── Inner voice ───────────────────────────────────────────

    def _say(self, moment: dict) -> None:
        """Her inner voice on a play moment — composed by her own
        language system from the event, never recited."""
        if self._voiced >= _VOICE_CAP:
            return
        vocab = getattr(
            getattr(self._mind.brain.frontal, "language", None),
            "vocabulary",
            None,
        )
        if vocab is None:
            return
        try:
            text = vocab._compose_play_moment(moment, self._mind.feel())
        except Exception:  # noqa: BLE001
            return
        if text:
            self._voiced += 1
            self._last_voiced = text
            inner = getattr(self._mind, "inner_life", None)
            if inner is not None:
                try:
                    noted = inner.note_thought(
                        text,
                        trigger="play",
                        intent="play",
                        metadata={
                            "topic": moment.get("game", ""),
                            "moment": moment.get("kind", ""),
                        },
                    )
                    if noted is not None:
                        return
                except Exception:  # noqa: BLE001
                    pass
            self._mind._emit_live_thought("play", text)

    def _impulse(self, chem: int, amount: float) -> None:
        try:
            self._mind._learner_neuro_impulse(chem, amount)
        except Exception as e:  # noqa: BLE001
            note_swallowed("games.session._impulse", e)

    def _wave(self, wave: Any, amount: float) -> None:
        try:
            from thalamus.brain_waves import add_brain_wave_drive

            add_brain_wave_drive(wave, amount)
        except Exception:  # noqa: BLE001
            pass

    # ── Parsing ───────────────────────────────────────────────

    @staticmethod
    def _fields(line: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for part in line.split(" | "):
            key, _, val = part.partition(" ")
            out[key.strip().upper()] = val.strip()
        return out

    @staticmethod
    def _rows(frame_ref: str) -> list[list[int]]:
        """Rows from inline '/'-separated hex or an ``@file`` path."""
        if frame_ref.startswith("@"):
            from pathlib import Path

            text = Path(frame_ref[1:]).read_text(encoding="utf-8")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return [[int(c, 16) for c in ln] for ln in lines]
        return [[int(c, 16) for c in row] for row in frame_ref.split("/")]

    # ── Turns ─────────────────────────────────────────────────

    def turn(self, line: str) -> str:
        """One game turn in, one action token out."""
        from thalamus.brain_waves import BrainWave

        self._last_voiced = None
        fields = self._fields(line)
        frame_txt = fields.get("FRAME", "")
        if not frame_txt:
            return ""
        rows = self._rows(frame_txt)
        avail = [
            int(a) for a in fields.get("ACTIONS", "").split(",") if a.strip()
        ]
        if not avail:
            return ""
        a_done, _, a_win = fields.get("LEVELS", "0/1").partition("/")
        levels, _win_levels = int(a_done or 0), int(a_win or 1)
        game = fields.get("GAME", "the game")
        event = fields.get("EVENT", "").lower()

        # Terminal outcome of the step just taken arrives on the next
        # frame (the adapter resets immediately) — close the episode
        # before observing the fresh frame.
        if event in ("win", "over"):
            self._episode_end(event, game)

        # Sitting down to play — engagement signature + inner voice,
        # once per distinct game this session. An INFO field is the
        # world's spoken instruction — "this is a video game, win the
        # levels" — delivered like a manual before the first move: it
        # narrates through the inner voice and broadcasts as context,
        # so her cognition meets the stream already knowing the genre.
        if game not in self._seen_games:
            self._seen_games.add(game)
            self._prev_levels = levels
            info = fields.get("INFO", "")
            self._impulse(CHEM_ACETYLCHOLINE, 0.004)
            self._wave(BrainWave.BETA2, 0.15)
            self._say({"kind": "start", "game": game, "goal": info})
            if info:
                self._bridge.broadcast_observation(
                    f"{game} instructions: {info}",
                    topics=["game", "instructions", game],
                    salience=0.9,
                )

        # Progress — levels advanced: the goal signal, marked before
        # the new observation so the winning segment closes intact.
        # The same flag reaches the world model through info — the
        # contact that preceded the transition is the goal.
        advanced = levels > self._prev_levels
        if advanced:
            self._agent.mark_goal_reached()
            self._impulse(CHEM_DOPAMINE, 0.01)
            self._say({"kind": "scored", "game": game})
            self._dry = 0
        self._prev_levels = levels

        # Observe; the step's visible feedback couples to dopamine the
        # same way internal play couples its TD error — a changed
        # world is information.
        change = self._agent.observe(rows)
        # The embodied stack processes the same frame: the world
        # model learns physics, the hippocampal map binds the layout,
        # the agency comparator resolves last turn's intention, and
        # the hypothesis ledger takes its transition evidence. The
        # specialist plays; the brain lives in the world it plays in.
        try:
            self._mind.observe(
                rows,
                info={
                    "game": game,
                    "level": levels,
                    "change": change,
                    **({"win": True} if advanced else {}),
                    **(
                        {"instructions": fields["INFO"]}
                        if "INFO" in fields
                        else {}
                    ),
                },
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed("games.session.observe", e)
        # Vision: the frame is a visual stimulus too — the same
        # V1→V4→VTC→MTL hierarchy that sees photographs and web
        # images. V1 gamma power closes the vision→arousal→attention
        # loop (the occipital callback lands it as acetylcholine), and
        # a scene kind she has never seen rides the broadcast as
        # novelty. Grid play that never reaches the visual stream is
        # a puzzle answered blind.
        vision_novel: bool | None = None
        cortex = getattr(self._mind, "visual_cortex", None)
        if cortex is not None:
            try:
                vision_novel = self._see_frame(rows, cortex)
            except Exception as e:  # noqa: BLE001
                note_swallowed("games.session.see_frame", e)
        # The frame leaves the play loop: the raw grid lands on the
        # tensor slab and the percept broadcasts to every region that
        # listens to the workspace. Salience follows measured change —
        # a quiet frame stays subliminal, a lively one ignites.
        self._bridge.publish_frame(rows)
        self._bridge.broadcast_observation(
            f"game {game}: level {levels}/{_win_levels}, frame changed {change:.0%}",
            topics=["game", "spatial", game] if game != "the game" else ["game", "spatial"],
            salience=min(1.0, change * 3.0 + (0.5 if levels > 0 else 0.0)),
            shape=(len(rows), len(rows[0]) if rows else 0),
            metadata={
                "game": game,
                "level": levels,
                "levels_won": _win_levels,
                **({"vision_novel": vision_novel} if vision_novel is not None else {}),
            },
        )
        impulse = max(-0.006, min(0.006, change * 0.04))
        if abs(impulse) >= 0.001:
            self._impulse(CHEM_DOPAMINE, impulse)
            self._wave(
                BrainWave.GAMMA if impulse > 0 else BrainWave.BETA2,
                min(0.25, abs(impulse) * 40),
            )
        if change < 0.005:
            self._dry += 1
            if self._dry == 20:
                self._say({"kind": "stuck", "game": game})
        else:
            self._dry = 0

        # The decision is published as a distributed field before the
        # token leaves: per-action dispositions (learned effect size,
        # per the agent's own stats) onto `motor.actions`, and for
        # coordinate actions a spatial bump at the target cell — the
        # daemon's GET_MOTOR_TARGET decodes the same decision back.
        self._publish_action_fields(avail)
        agent_pick = self._agent.choose_action(avail)
        if agent_pick == 0:
            try:
                self._mind.note_world_action(0)
            except Exception as e:  # noqa: BLE001
                note_swallowed("games.session.note_action", e)
            self._bridge.broadcast_action(0, metadata={"game": game})
            self._record_decision("RESET", {"source": "reset"}, {}, avail)
            return "RESET"
        # Arbitration: the specialist votes its disposition onto the
        # motor field; the basal ganglia weighs it against the world
        # model's own plan, hypothesis terms, and its calibration
        # instinct, and selects. Frontal proposes, striatum decides.
        action = agent_pick
        rec = self._agent.recommend(avail)
        try:
            choice = self._mind.act(avail, recommended=rec)
        except Exception as e:  # noqa: BLE001
            note_swallowed("games.session.act", e)
            choice = None
        if choice is not None:
            action = int(choice["action"])
            # The arbiter's pick is what the world answers — the
            # specialist attributes consequences to what was done.
            self._agent.note_action(action)
        else:
            # No arbiter online: the specialist's own pick stands;
            # record it as the embodied intention for next turn.
            try:
                self._mind.note_world_action(action)
            except Exception as e:  # noqa: BLE001
                note_swallowed("games.session.note_action", e)
        if action == 0:
            self._bridge.broadcast_action(0, metadata={"game": game})
            self._record_decision("RESET", choice or {"source": "reset"}, rec, avail)
            return "RESET"
        if action in (6, 7):
            data = self._agent.action_data(
                action, (len(rows), len(rows[0]))
            )
            self._publish_target_field(data["x"], data["y"], (len(rows), len(rows[0])))
            self._bridge.broadcast_action(
                action,
                cell={"x": data["x"], "y": data["y"]},
                metadata={"game": game},
            )
            token = f"ACTION{action} {data['x']} {data['y']}"
            self._record_decision(token, choice, rec, avail)
            return token
        self._bridge.broadcast_action(action, metadata={"game": game})
        token = f"ACTION{action}"
        self._record_decision(token, choice, rec, avail)
        return token

    # ── Perception & decision telemetry ─────────────────────────

    @staticmethod
    def _see_frame(rows: list[list[int]], cortex: Any) -> bool | None:
        """Run the grid through the visual cortex as an image.

        The palette-index grid becomes the RGB frame the occipital
        hierarchy expects, at native resolution — upscaling would only
        teach V1 block artifacts. Returns the percept's novelty flag:
        a scene kind she's never seen is worth a salience bump.
        """
        import numpy as np

        if not rows or not rows[0]:
            return None
        grid = np.asarray(rows, dtype=np.int64) & 0xF
        rgb = np.asarray(_PALETTE, dtype=np.uint8)[grid]
        percept = cortex.see(rgb, learn=True)
        return bool(getattr(percept, "is_novel", False))

    def _record_decision(
        self,
        token: str,
        choice: dict[str, Any] | None,
        rec: dict[str, float],
        avail: list[int],
    ) -> None:
        """Stash the decision basis behind this turn's action.

        Structured self-report, not composed speech: which drive the
        striatum honored, how far ahead the field's runner-up was, and
        the actions still in contention. ``note`` carries her inner
        voice only when a moment was actually voiced this turn.
        """
        scored = sorted(
            ((rec.get(str(a), 0.0), a) for a in avail), reverse=True
        )
        top = scored[0][0] if scored else 0.0
        second = scored[1][0] if len(scored) > 1 else 0.0
        margin = max(0.0, top - second)
        self.last_decision = {
            "action": token,
            "source": (choice or {}).get("source", "specialist"),
            "confidence": round(min(1.0, margin / 2.0), 3),
            "alternatives": [
                f"ACTION{a}" for score, a in scored[1:3] if score > 0
            ],
            "note": self._last_voiced,
        }

    # ── Motor fields ───────────────────────────────────────────

    def _publish_action_fields(self, avail: list[int]) -> None:
        """Project the agent's per-action disposition onto the slab."""
        scores: dict[str, float] = {}
        for a in avail:
            stat = self._agent.stats.get(a)
            scores[str(a)] = getattr(stat, "mean_change", 0.0)
        self._bridge.publish_motor_fields(None, scores)

    def _publish_target_field(
        self, x: int, y: int, shape: tuple[int, int]
    ) -> None:
        """A coordinate action's target as a bump on the spatial field."""
        h, w = shape
        field = [[0.0] * w for _ in range(h)]
        if 0 <= y < h and 0 <= x < w:
            field[y][x] = 1.0
        self._bridge.publish_motor_fields(field, None)

    # ── Episode outcomes ──────────────────────────────────────

    def _episode_end(self, event: str, game: str) -> None:
        """Feel the end of an episode, then let the agent close it."""
        self._bridge.broadcast_outcome(
            event,
            f"episode ended {event}",
            0.9 if event == "win" else 0.7,
            metadata={"game": game, "deaths": self._deaths},
        )
        if event == "win":
            self._impulse(CHEM_DOPAMINE, 0.02)
            self._say({"kind": "won", "game": game})
            self._agent.on_episode_end("WIN")
        else:
            self._deaths += 1
            self._impulse(CHEM_DOPAMINE, -0.008)
            self._impulse(CHEM_NOREPINEPHRINE, 0.008)
            self._impulse(CHEM_CORTISOL, 0.001 * self._deaths)
            self._say({"kind": "death", "game": game})
            self._agent.on_episode_end("GAME_OVER")
        try:
            self._mind.world.it_acted(
                f"played {game}: episode ended {event}",
                detail=f"external game episode {event}",
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed("games.session._episode_end", e)
        try:
            # Episode boundary for the embodied stack — credit (or
            # suspect) whatever she was touching when it ended, then
            # forget the layout and keep the physics it taught her.
            self._mind.note_world_end(
                "win" if event == "win" else None
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed("games.session.reset_world", e)
