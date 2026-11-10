"""Maxwell's equations on the organism's own field — word-layer
companion to ``src/daemon/maxwell.rs``.

There is no probe. The field substrate's coherent motion *is* the
current: the consensus of the body/prediction/agency fibers drifting
across the latent manifold is charge-in-substrate, and Maxwell's
equations say what field that motion makes. Each integration the
daemon measures the consensus' displacement (the current) and its
winding rate (the frequency), then evaluates the dipole expansion at
the organism's own boundary — the signature an external
magnetometer, or her own interoception, would read from the field
her coherence makes.

The regime is honestly quasi-static: the substrate drifts slowly, so
the induction and radiation terms are negligible and Ampère's law is
what survives. The latent→ampere bridge is a declared modeling
assumption, not a measurement — the load-bearing content is the
regime structure and the signature's change over time, not the
absolute tesla. The fast carriers that would radiate are the
brain-wave oscillators, a different layer of the same chain.

This module is the word layer for that physics: it fetches the
signature off the wire (``GET_FIELD_SIGNATURE``) and derives the
readings the regions actually use — tempo in hertz, coherence as a
0..1 lock, the field's thermal state. Rust computes; Python reads.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis_client.client import GenesisClient
    from genesis_client.types import FieldSignature

#: The disagreement scale the body's own precision attenuation uses
#: (``daemon::tick::DISAGREEMENT_PRECISION_SCALE``): at this Fréchet
#: variance the published precision halves. Coherence is the same
#: tension read as a 0..1 lock — 1 at perfect consensus, 0.5 at the
#: half-precision tension, falling toward 0 as the fibers scatter.
_DISAGREEMENT_SCALE = 0.15


def field_signature(client: GenesisClient) -> FieldSignature | None:
    """Her field's computed signature — one read of the body's own
    mathematics. ``None`` when the channel is absent (an old daemon,
    a dead transport) — never a zeroed signature dressed as a reading.
    """
    getter = getattr(client, "get_field_signature", None)
    if getter is None:
        return None
    return getter()


def carrier_hz(sig: FieldSignature) -> float:
    """The field's carrier frequency in hertz — the consensus
    winding rate, which *is* her heartbeat: the heart is the
    carrier frequency of the field, not a rhythm observed
    alongside it. 0.0 while the signature is absent: no measurable
    winding, not a stopped heart."""
    if not sig.signature_present or sig.omega_rad_s <= 0.0:
        return 0.0
    return sig.omega_rad_s / (2.0 * math.pi)


def coherence(sig: FieldSignature) -> float | None:
    """The field's lock as a 0..1 coherence — the fibers' agreement
    about their shared consensus, mapped through the same tension
    scale the body's precision attenuation uses.

    ``None`` while no integration exists: coherence is a reading of
    a live field, and an absent field has no lock to report — not
    even zero."""
    if not sig.integration_present or sig.disagreement < 0.0:
        return None
    return 1.0 / (1.0 + sig.disagreement / _DISAGREEMENT_SCALE)
