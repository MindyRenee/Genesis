"""Psychoanalysis — the unconscious that keeps working while she rests.

Freud's model, grown from the architecture she already has rather
than bolted on:

- **The dynamic unconscious** is her emotional-memory charge crossed
  with her concept network's access. A concept carrying heavy
  amygdala-tagged charge that the conscious network never activates
  is *repressed*: the charge is real, the ideational representative
  is held out of awareness. The register here tracks those
  complexes and their return-pressure.
- **Dream work** is interpretation of the dream engine's manifest
  content. Condensation (several manifest elements converging on
  one latent node) and displacement (a low-charge manifest element
  standing in for a charged neighbor) are read off the network's
  own topology — spread activation *is* the associative path the
  dream work walks. A dream that carries repressed material
  discharges some of its pressure: catharsis, literally.
- **Wish fulfillment** maps latent content onto her unmet drives —
  social isolation, accumulated urges — so the analysis names what
  the dream was *for*, in her own drives, not in a dictionary's.
- **Free association** is a charge-weighted walk on the association
  graph: the censor's looseness scales how far affective pull can
  bend each step away from the literal path. Relaxed censorship
  (rest, REM) lets the chain wander further toward the complex.
- **Parapraxes** — slips — are repressed concepts surfacing unbidden
  in her spontaneous thought stream; each return of the repressed
  relieves a little pressure.

Nothing here speaks. The analysis produces structure — which
concepts are complexes, what the dream discharged, where the
association chain ran — and the rest of her architecture decides
what, if anything, becomes a thought.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from memory.memory import _emotional_intensity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


@dataclass
class RepressedTrace:
    """One registered complex: a charged concept kept out of consciousness.

    Attributes:
        concept_id: The repressed ideational representative.
        charge: Current affective charge [0..1], recomputed each scan
            from the emotional-memory record.
        pressure: Accumulated return-pressure [0..1] — how hard the
            complex is pressing for discharge. Builds while the
            repression holds, relieved by dreams and slips.
        first_seen: Epoch seconds when the complex first registered.
        slips: How many times it has surfaced unbidden.
    """

    concept_id: str
    charge: float
    pressure: float
    first_seen: float
    slips: int = 0


@dataclass
class DreamInterpretation:
    """The latent read of one dream's manifest content.

    Attributes:
        manifest: The dream's surface content (concept ids), as given.
        latent: The charged latent content the analysis reached —
            what the manifest elements were carrying, ranked.
        wish: The unmet drive the latent content serves, if one could
            be traced through the network; ``None`` when no drive
            overlap cleared the floor — an honest "no wish found",
            not a fabricated one.
        discharged: Repressed concepts whose pressure this dream
            relieved.
        mechanisms: The dream-work evidence — ``condensation`` is the
            latent nodes reached by multiple manifest elements,
            ``displacement`` maps manifest stand-ins to the charged
            concepts they carry.
        confidence: How much of the manifest content resolved into
            the network at all, scaled by the charge the analysis
            actually found. Low confidence means the dream was thin
            or the network doesn't reach it — say less, not more.
    """

    manifest: list[str] = field(default_factory=list)
    latent: list[str] = field(default_factory=list)
    wish: str | None = None
    discharged: list[str] = field(default_factory=list)
    mechanisms: dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0


class Psychoanalysis:
    """The analytic engine — repression, dream work, association, slips.

    Constructed with the live concept network and the emotional
    memory system; both are read, never owned. ``get_drives`` is a
    callable returning the current unmet-drive map so the wish-fulfilment
    read follows her real state instead of a snapshot taken at
    construction.
    """

    # An emotional association below this charge is a memory with
    # feeling, not a complex. Sized so one strong conditioning trial
    # (association ~0.75 on a negative-biased tag, intensity ~0.3)
    # clears it while incidental weak associations stay out.
    CHARGE_FLOOR = 0.2
    # Conscious access scale: concept activation of ~1/3 or more reads
    # as fully accessible — the idea is in awareness, not repressed.
    ACCESS_SCALE = 3.0
    # charge × (1 - access) below this isn't registered — mild
    # discomfort the conscious mind still holds isn't repression.
    REPRESSION_FLOOR = 0.15
    # How much of the current repression becomes return-pressure per
    # register scan. Pressure accumulates while the complex holds.
    PRESSURE_STEP = 0.5
    # Quieting rate for complexes whose charge fell away — the
    # repression lifted, so the pressure relaxes rather than
    # vanishing outright.
    PRESSURE_DECAY = 0.85
    # Pressure relief fractions: a dream that carries a complex
    # relieves more than a waking slip — the censor is down in the
    # dream, so the discharge is fuller.
    DREAM_DISCHARGE = 0.6
    SLIP_DISCHARGE = 0.4
    # Bound the register — only the most pressing complexes kept.
    REGISTER_MAX = 64
    # Latent concepts returned per interpretation.
    LATENT_TOP = 10
    # Dream-work thresholds: condensation needs at least this many
    # manifest seeds converging on one latent node; a manifest
    # stand-in must itself be low-charge while the content it
    # carries is a registered complex.
    CONDENSATION_MIN_HITS = 2
    DISPLACEMENT_MANIFEST_MAX_CHARGE = 0.15
    # Minimum deficit × latent-overlap for a wish to be named —
    # below this the drive's reach into the dream's latent pool is
    # noise, not a connection.
    WISH_FLOOR = 0.02
    # Spread-activation parameters for the latent search — shallow
    # enough that the latent pool stays adjacent to the manifest.
    _SPREAD_AMOUNT = 0.3
    _SPREAD_DEPTH = 2

    def __init__(
        self,
        *,
        network,
        emotional_memory,
        get_drives: Callable[[], Mapping[str, float]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._network = network
        self._emotional = emotional_memory
        self._get_drives = get_drives or (lambda: {})
        self._rng = rng or random.Random()
        self._register: dict[str, RepressedTrace] = {}
        self._charges: dict[str, float] | None = None
        # The register and charge map are touched by the metronome
        # (REM rescan), the wake path (interpretation), and live-thought
        # threads (slips) — guard them.
        self._lock = threading.Lock()

    # ── Charge ─────────────────────────────────────────────────

    def charge_of(self, concept_id: str) -> float:
        """Affective charge on a concept — its emotional-memory weight.

        The strongest non-extinguished association on the stimulus
        sets the charge: association strength scaled by the chemical
        intensity stamped at encoding.
        """
        charges = self._charge_map()
        return charges.get(concept_id, 0.0)

    def _charge_map(self) -> dict[str, float]:
        charges = self._charges
        if charges is None:
            self._rescan_charges()
            charges = self._charges or {}
        return charges

    def _rescan_charges(self) -> None:
        charges: dict[str, float] = {}
        for em in self._emotional.get_all():
            if em.is_extinct:
                continue
            charge = min(1.0, em.association_strength * _emotional_intensity(em.emotional_tag))
            if charge > charges.get(em.stimulus, 0.0):
                charges[em.stimulus] = charge
        with self._lock:
            self._charges = charges

    # ── The repression register ────────────────────────────────

    @property
    def register(self) -> dict[str, RepressedTrace]:
        """The current complexes, concept id → trace."""
        return self._register

    def repressed(self) -> list[str]:
        """Concept ids currently held out of consciousness."""
        return list(self._register)

    def update_register(self) -> None:
        """Rescan emotional memory and refresh the register.

        Called on the REM-side of the cycle — the register is the
        unconscious's bookkeeping, and REM is when the books are
        worked. A concept registers when its affective charge is
        real but the conscious network barely touches it: charge ×
        (1 - access) is the repression, and it converts to
        return-pressure while it holds. Complexes whose charge has
        decayed (extinction, reconsolidation) relax rather than
        vanish — a quieted complex is not a deleted one.
        """
        self._rescan_charges()
        charges = self._charges or {}
        now = time.time()
        register: dict[str, RepressedTrace] = {}
        with self._lock:
            for cid, charge in charges.items():
                if charge < self.CHARGE_FLOOR:
                    continue
                concept = self._network.get_concept(cid)
                access = concept.activation if concept is not None else 0.0
                repression = charge * (1.0 - min(1.0, access * self.ACCESS_SCALE))
                if repression < self.REPRESSION_FLOOR:
                    continue
                trace = self._register.get(cid)
                if trace is None:
                    trace = RepressedTrace(
                        concept_id=cid,
                        charge=charge,
                        pressure=min(1.0, repression),
                        first_seen=now,
                    )
                else:
                    trace.charge = charge
                    trace.pressure = min(1.0, trace.pressure + repression * self.PRESSURE_STEP)
                register[cid] = trace
            # Complexes that no longer repress decay quietly.
            for cid, trace in self._register.items():
                if cid in register:
                    continue
                trace.pressure *= self.PRESSURE_DECAY
                if trace.pressure > self.REPRESSION_FLOOR * 0.5:
                    register[cid] = trace
            # Bound: keep the most pressing complexes.
            top = sorted(register.values(), key=lambda t: t.pressure, reverse=True)
            self._register = {t.concept_id: t for t in top[: self.REGISTER_MAX]}

    # ── Dream work ─────────────────────────────────────────────

    def interpret_dreams(self, manifest: Iterable[str]) -> DreamInterpretation:
        """Read a dream's manifest content back to its latent content.

        The manifest elements seed spread activation; the latent pool
        is what the spread reaches, weighted by affective charge —
        the charged neighbor is what the innocuous image stands in
        for. Condensation and displacement are read off the topology,
        the wish off the unmet drives, and any repressed material the
        dream carried is discharged.
        """
        manifest_list = [str(c) for c in manifest]
        manifest_ids = [cid for cid in manifest_list if self._network.get_concept(cid) is not None]
        coverage = len(manifest_ids) / max(1, len(manifest_list))
        manifest_set = set(manifest_ids)

        reach: dict[str, float] = {}
        hits: dict[str, int] = {}
        for seed in manifest_ids:
            spread = self._network.spread_activation(
                [seed], amount=self._SPREAD_AMOUNT, depth=self._SPREAD_DEPTH
            )
            for cid, activation in spread.items():
                if cid in manifest_set:
                    continue
                # Charge amplifies reach: the affective pull is what
                # bends the dream toward the complex.
                reach[cid] = reach.get(cid, 0.0) + activation * (0.2 + self.charge_of(cid))
                hits[cid] = hits.get(cid, 0) + 1

        latent_ranked = sorted(reach.items(), key=lambda kv: kv[1], reverse=True)
        latent = [cid for cid, _ in latent_ranked[: self.LATENT_TOP]]
        latent_set = set(latent)

        condensation = [
            cid for cid in latent if hits.get(cid, 0) >= self.CONDENSATION_MIN_HITS
        ]
        # A manifest element displaces onto the strongest charged
        # latent it stands in for — keep the best target per stand-in.
        displacement: dict[str, str] = {}
        charged_latents = [cid for cid in latent_set if self.charge_of(cid) >= self.CHARGE_FLOOR]
        if charged_latents:
            for seed in manifest_ids:
                if self.charge_of(seed) < self.DISPLACEMENT_MANIFEST_MAX_CHARGE:
                    displacement[seed] = max(charged_latents, key=self.charge_of)

        wish = self._trace_wish(reach)

        # Catharsis — repressed content the dream carried discharges.
        discharged: list[str] = []
        with self._lock:
            for cid in manifest_set | latent_set:
                trace = self._register.get(cid)
                if trace is None:
                    continue
                trace.pressure = max(0.0, trace.pressure - trace.charge * self.DREAM_DISCHARGE)
                discharged.append(cid)
            self._register = {
                cid: t for cid, t in self._register.items() if t.pressure > 0.01
            }

        found_charge = sum(reach.values())
        confidence = coverage * min(1.0, found_charge)
        return DreamInterpretation(
            manifest=manifest_ids,
            latent=latent,
            wish=wish,
            discharged=discharged,
            mechanisms={
                "condensation": condensation,
                "displacement": displacement,
            },
            confidence=confidence,
        )

    def _trace_wish(self, reach: dict[str, float]) -> str | None:
        """Which unmet drive does the latent content serve?

        Each drive that resolves to a network concept spreads its own
        activation; the overlap with the dream's latent pool, scaled
        by how unmet the drive is, names the wish. No overlap, no
        wish — the analysis declines rather than fabricates.
        """
        best_score = self.WISH_FLOOR
        wish: str | None = None
        try:
            drives = self._get_drives()
        except Exception:  # noqa: BLE001
            return None
        for name, deficit in drives.items():
            try:
                deficit = float(deficit)
            except (TypeError, ValueError):
                continue
            if deficit <= 0.0 or self._network.get_concept(name) is None:
                continue
            spread = self._network.spread_activation(
                [name], amount=self._SPREAD_AMOUNT, depth=self._SPREAD_DEPTH
            )
            overlap = sum(reach.get(cid, 0.0) for cid in spread)
            score = deficit * overlap
            if score > best_score:
                best_score = score
                wish = name
        return wish

    # ── Free association ───────────────────────────────────────

    def free_associate(
        self,
        seed: str,
        *,
        max_steps: int = 6,
        looseness: float = 0.5,
    ) -> list[str]:
        """Walk the association graph from ``seed``, charge-weighted.

        Each step picks among neighbors weighted by edge weight times
        affective pull — ``looseness`` is the censor's grip: 0 is the
        literal path (edge weights only), 1 lets the complexes bend
        the chain hard. Relaxed censorship (rest, REM, drowsiness)
        loosens the grip; a waking analytic stance holds it.

        Returns the chain including the seed, bounded at
        ``max_steps`` edges.
        """
        chain = [seed]
        seen = {seed}
        current = seed
        for _ in range(max(0, max_steps)):
            neighbors = self._network.get_associations(current)
            weighted = [
                (n.concept, n.weight * (1.0 + self.charge_of(n.concept) * looseness * 3.0))
                for n in neighbors
                if n.concept not in seen and n.weight > 0.0
            ]
            if not weighted:
                break
            total = sum(w for _, w in weighted)
            pick_at = self._rng.random() * total
            current = weighted[-1][0]
            acc = 0.0
            for cid, w in weighted:
                acc += w
                if acc >= pick_at:
                    current = cid
                    break
            chain.append(current)
            seen.add(current)
        return chain

    @staticmethod
    def censorship(resting: bool) -> float:
        """The censor's grip as a free-association looseness.

        Rest (sleep, meditation, the descent) relaxes censorship;
        waking holds it. Feed the return as ``looseness``.
        """
        return 0.85 if resting else 0.3

    # ── Return of the repressed ────────────────────────────────

    def note_surfaced(self, concepts: Iterable[str]) -> list[str]:
        """A concept surfaced unbidden — a slip, a dream fragment.

        Every repressed concept in ``concepts`` gets its slip counted
        and its pressure partially relieved (waking discharge is
        shallower than dream discharge — the censor is up). Returns
        the ids that surfaced from the register so the caller can
        mark the event.
        """
        slipped: list[str] = []
        with self._lock:
            for cid in concepts:
                trace = self._register.get(str(cid))
                if trace is None:
                    continue
                trace.slips += 1
                trace.pressure = max(0.0, trace.pressure - trace.charge * self.SLIP_DISCHARGE)
                slipped.append(trace.concept_id)
            self._register = {
                cid: t for cid, t in self._register.items() if t.pressure > 0.01
            }
        return slipped
