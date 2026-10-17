"""World bridge — world state pushed to every region that can hear it.

An observation arriving through ``Mind.observe`` or an external game
session is not just input to a policy: under global-workspace theory
the salient parts of it are *conscious content*, and the whole brain
gets two channels to it:

- **Broadcast (push)** — ``GlobalWorkspace.broadcast`` with
  ``source="world"`` and a topic list, so attention, semantic memory,
  working memory, the core self, theory of mind, and curiosity all
  receive the same event at once. Routine frames stay subliminal in
  the workspace's pending buffer; salient change ignites.

- **Tensor slab (pull)** — the raw observation is published to the
  daemon's named tensor slab (``world.frame``), so any region — and
  any Rust-side process — can read the full multidimensional state
  rather than a textual summary of it. Decisions travel back the same
  way: ``motor.spatial`` / ``motor.actions`` are the distributed motor
  fields the daemon decodes into a concrete target
  (``GET_MOTOR_TARGET``).

Every call is best-effort: a missing daemon or a failing consumer
degrades the bridge, never the play loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from genesis_client.swallow import note_swallowed
from genesis_client.tensor import DType, Tensor

logger = logging.getLogger(__name__)

# Well-known slab names — mirror `src/field/motor.rs`.
WORLD_FRAME = "world.frame"
MOTOR_SPATIAL = "motor.spatial"
MOTOR_ACTIONS = "motor.actions"
MOTOR_ACTION_IDS = "motor.action_ids"

# Consecutive tensor failures before the slab channel is given up for
# the rest of the bridge's life — an absent daemon should cost a flag
# check per frame, not a caught exception.
_TENSOR_FAILURE_LIMIT = 3


class WorldBridge:
    """Publishes world state to the workspace and the tensor slab.

    Both channels are optional: constructed with only a workspace it
    still broadcasts; with only a client it still publishes tensors.
    """

    def __init__(
        self,
        workspace: Any = None,
        client: Any = None,
        brain_waves: Callable[[], Any] | None = None,
    ) -> None:
        self._workspace = workspace
        self._client = client
        self._brain_waves = brain_waves
        self._tensor_disabled = client is None
        self._tensor_failures = 0

    # ── Tensor channel (pull side of global state) ────────────

    def publish_tensor(self, name: str, tensor: Tensor) -> bool:
        """PUT a tensor onto the daemon's slab. Returns success."""
        if self._tensor_disabled:
            return False
        try:
            ok = self._client.put_tensor(name, tensor)
        except Exception as e:  # noqa: BLE001
            self._tensor_failures += 1
            if self._tensor_failures >= _TENSOR_FAILURE_LIMIT:
                self._tensor_disabled = True
                logger.debug(f"tensor slab unreachable — publishing off: {e}")
            else:
                note_swallowed("thalamus.bridge.publish_tensor", e)
            return False
        self._tensor_failures = 0
        return ok

    def publish_frame(self, rows: list[list[int]]) -> bool:
        """Publish a grid observation as ``world.frame`` (u8, H×W)."""
        if not rows or not rows[0]:
            return False
        flat = bytes(min(255, max(0, int(v))) for row in rows for v in row)
        return self.publish_tensor(
            WORLD_FRAME,
            Tensor(
                dtype=DType.U8,
                shape=(len(rows), len(rows[0])),
                payload=flat,
            ),
        )

    def publish_motor_fields(
        self,
        spatial: list[list[float]] | None,
        action_scores: dict[str, float] | None,
    ) -> None:
        """Publish the decision laid out as distributed fields.

        ``spatial`` is an H×W field over the world grid (row-major);
        ``action_scores`` maps action tokens to dispositions — tokens
        that parse as small ints also land in ``motor.action_ids`` so
        the daemon can name the winning action.
        """
        if spatial:
            h, w = len(spatial), len(spatial[0])
            payload = bytearray()
            for row in spatial:
                for v in row:
                    payload += _f32(v)
            self.publish_tensor(
                MOTOR_SPATIAL,
                Tensor(dtype=DType.F32, shape=(h, w), payload=bytes(payload)),
            )
        if action_scores:
            ids: list[int] = []
            scores = bytearray()
            id_payload = bytearray()
            for token, score in action_scores.items():
                try:
                    aid = int(str(token))
                except (TypeError, ValueError):
                    continue  # non-numeric tokens can't ride the u8 map
                if not 0 <= aid <= 255:
                    continue
                ids.append(aid)
                id_payload.append(aid)
                scores += _f32(float(score))
            if ids:
                self.publish_tensor(
                    MOTOR_ACTION_IDS,
                    Tensor(dtype=DType.U8, shape=(len(ids),), payload=bytes(id_payload)),
                )
                self.publish_tensor(
                    MOTOR_ACTIONS,
                    Tensor(dtype=DType.F32, shape=(len(ids),), payload=bytes(scores)),
                )

    # ── Workspace channel (push side of global state) ─────────

    def broadcast(
        self,
        content: Any,
        kind: str,
        activation: float,
        topics: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """One world broadcast into the global workspace.

        ``kind`` distinguishes observation/action/outcome in metadata;
        ``topics`` is what the registered consumers actually read
        (attentional capture, semantic priming, core-self trigger).
        """
        if self._workspace is None:
            return False
        meta = {"kind": kind, "topics": list(topics)}
        if metadata:
            meta.update(metadata)
        try:
            return self._workspace.broadcast(
                content=content,
                source="world",
                activation=max(0.0, min(1.0, activation)),
                brain_waves=self._brain_waves() if self._brain_waves else None,
                metadata=meta,
            )
        except Exception as e:  # noqa: BLE001 — broadcast must never break the loop
            note_swallowed("thalamus.bridge.broadcast", e)
            return False

    def broadcast_observation(
        self,
        summary: str,
        *,
        topics: list[str],
        salience: float,
        shape: tuple[int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Broadcast a world observation.

        ``salience`` scales activation: routine frames stay below the
        ignition threshold (subliminal, pending-buffer content), while
        real change — a win, a death, an unexplained transform —
        crosses it and reaches every registered region.
        """
        activation = min(1.0, 0.3 + 0.6 * max(0.0, min(1.0, salience)))
        meta = dict(metadata or {})
        if shape is not None:
            meta["grid"] = {"height": shape[0], "width": shape[1]}
            meta["tensor"] = WORLD_FRAME
        return self.broadcast(
            summary,
            "observation",
            activation,
            ["world", "observation", *topics],
            meta,
        )

    def broadcast_action(
        self,
        action: Any,
        *,
        topics: list[str] | None = None,
        cell: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Broadcast the chosen action — the decision as content."""
        meta = dict(metadata or {})
        if cell is not None:
            meta["cell"] = cell
            meta["tensor"] = MOTOR_SPATIAL
        return self.broadcast(
            f"act {action}",
            "action",
            0.55,
            ["world", "action", *(topics or [])],
            meta,
        )

    def broadcast_outcome(
        self,
        kind: str,
        detail: str,
        salience: float,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Broadcast a world outcome (win, death, transform, reset)."""
        activation = min(1.0, 0.45 + 0.55 * max(0.0, min(1.0, salience)))
        return self.broadcast(
            detail,
            "outcome",
            activation,
            ["world", "outcome", kind],
            metadata,
        )


class WorldBroadcastSink:
    """Workspace module adapter: world broadcasts forwarded to a hook.

    Regions that own no ``WorkspaceModule`` of their own (volition
    context, the inner-life stream) still need to know what world
    content is currently conscious. The sink is the fan-in point —
    registered once, it forwards only ``source="world"`` items.
    """

    def __init__(self, on_world_broadcast: Callable[[Any], None]) -> None:
        self._on_world_broadcast = on_world_broadcast

    def receive_broadcast(self, item: Any) -> None:
        if getattr(item, "source", None) != "world":
            return
        try:
            self._on_world_broadcast(item)
        except Exception as e:  # noqa: BLE001
            note_swallowed("thalamus.bridge.sink", e)


def _f32(v: float) -> bytes:
    import struct

    return struct.pack("<f", float(v))
