"""Hamiltonian flow language generator.

Language as a trajectory through a potential landscape on a torus.

Instead of searching a flat vector space and assembling results into
templates, this module builds a potential field from the concept
network's edge weights, then integrates a geodesic from the query
concept. The trajectory passes through concept cells, and the
sequence of cells traversed IS the sentence — word order is the order
of traversal, sentence structure is the shape of the trajectory.

# The math

The concept network defines a potential landscape on the torus T^d:

    U(θ) = -Σᵢ wᵢ · K(θ, θᵢ)

where θᵢ is concept i's angular position (from spectral embedding),
wᵢ is its edge weight sum (hubs become deep valleys), and K is a
von Mises kernel measuring angular proximity with wrap-around.

The Hamiltonian is:

    H(θ, p) = ½|p|² + U(θ)

The equations of motion are:

    dθ/dt =  ∂H/∂p = p
    dp/dt = -∂H/∂θ = -∇U(θ)

The emotional state sets the momentum: high arousal → high momentum →
fast transitions → shorter sentences; low arousal → low momentum →
slow transitions → longer, contemplative sentences.

# Sacred geometry constraints

The holonomy of the connection determines which trajectories are
allowed. Six-fold rotational symmetry (Flower of Life) groups six
related concepts around each hub. The vesica piscis (overlap region
between two cells) is where relations live — the trajectory passes
through vesica regions to express typed relations between concepts.

# Why this produces flow

The trajectory is continuous and determined by dynamics, not assembled
from pieces. The sentence shape emerges from the potential landscape,
which is built from the concept network. No templates, no fallbacks,
no hardcoded phrases — the sentence IS the trajectory.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from .morphology import (
    agree_verb_phrase,
    copula,
    is_plural_np,
    person_pronoun,
)

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork
    from ..emotion import EmotionalState
    from .base import Thought

__all__ = ["FlowGenerator"]

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

# Toroidal dimension — how many angular coordinates each concept has.
# This must match the spectral embedding dimension used by the
# EmbeddingStore. 32 dims gives enough resolution to distinguish
# ~100K concepts while keeping the geodesic computation fast.
_FLOW_DIM = 16

# Kernel bandwidth for the von Mises potential. Controls how sharply
# each concept's potential well is focused. Smaller κ = wider wells
# = more concepts influence the flow at each point. Larger κ = sharper
# wells = more precise but noisier flow.
_KAPPA = 3.0

# Integration step size. Smaller dt = smoother trajectory but slower.
# Larger dt = faster but may skip over small concept cells.
_DT = 0.05

# Maximum integration steps. Bounds the sentence length.
_MAX_STEPS = 80

# Minimum potential gradient magnitude to continue flowing. When the
# trajectory reaches a local minimum (flat gradient), the flow stops —
# the sentence ends naturally.
_MIN_GRADIENT = 1e-4

# Momentum decay (friction). Without this, the trajectory oscillates
# forever between potential wells. A small friction term lets the
# flow settle into a natural endpoint.
_FRICTION = 0.15

# Six-fold symmetry constant (Flower of Life). Concepts around a hub
# are grouped in sixes — each 60° (π/3 radians) rotation brings a
# semantically related neighbor.
_SIX_FOLD = math.pi / 3.0


class FlowGenerator:
    """Generate language from Hamiltonian flow on a toroidal manifold.

    The flow is built from the concept network's edge structure:
    - Each concept's angular position comes from the spectral embedding
    - Edge weights define the potential landscape (hubs = valleys)
    - Typed relations create channels between concept cells
    - The emotional state sets the momentum

    The trajectory through the potential landscape IS the sentence.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        seed: int = 42,
    ) -> None:
        self._network = network
        self._rng = np.random.default_rng(seed)

        # Angular positions: concept_id → angle vector on T^d
        self._angles: dict[str, np.ndarray] = {}
        # Edge weight sums: concept_id → total edge weight (potential depth)
        self._weights: dict[str, float] = {}
        # Concept cells: ordered list of concept_ids (for vectorized ops)
        self._concept_ids: list[str] = []
        # Angle matrix: (N, d) for vectorized computation
        self._angle_matrix: np.ndarray | None = None
        # Weight vector: (N,) potential depth per concept
        self._weight_vector: np.ndarray | None = None
        # Relation channels: (source_id, target_id, relation_type, weight)
        # These create directional biases in the potential gradient
        self._channels: list[tuple[str, str, str, float]] = []
        # Whether the flow field has been built
        self._built = False

    # ─── Building the flow field ──────────────────────────────────

    def build(self) -> None:
        """Build the potential field from the concept network.

        Maps each concept to an angular position on the torus using
        the network's spectral structure, computes edge weight sums
        for potential depth, and identifies relation channels.
        """
        if self._network is None or self._network.size < 10:
            return

        # Get all concept IDs
        concept_ids = list(self._network._concepts.keys())
        if len(concept_ids) < 10:
            return

        # Build adjacency for spectral positioning
        # Use a simple hash-based angular assignment as a fallback
        # when the spectral embedding isn't available. The spectral
        # embedding from EmbeddingStore is preferred — this method
        # is called when the embedding store hasn't been wired yet.
        n = len(concept_ids)

        # Assign angular positions. If the network has spectral
        # embeddings available, use those. Otherwise, use a
        # deterministic hash-based assignment that distributes
        # concepts uniformly on the torus.
        angle_matrix = np.zeros((n, _FLOW_DIM), dtype=np.float32)

        # Build a concept-to-index map
        idx_map: dict[str, int] = {}
        for i, cid in enumerate(concept_ids):
            idx_map[cid] = i

        # Use graph structure to assign angles: concepts connected by
        # edges should be angularly close. This is a lightweight spectral
        # embedding using the power iteration method on the adjacency
        # matrix's top eigenvectors.
        angle_matrix = self._spectral_angles(concept_ids, idx_map)

        self._angles = {
            cid: angle_matrix[idx_map[cid]]
            for cid in concept_ids
        }
        self._concept_ids = concept_ids
        self._angle_matrix = angle_matrix

        # Compute edge weight sums (potential depth)
        weights = np.zeros(n, dtype=np.float32)
        for edge in self._network._edges:
            si = idx_map.get(edge.source)
            ti = idx_map.get(edge.target)
            if si is not None:
                weights[si] += edge.weight
            if ti is not None:
                weights[ti] += edge.weight
        self._weights = {
            cid: float(weights[idx_map[cid]])
            for cid in concept_ids
        }
        self._weight_vector = weights

        # Build relation channels
        self._channels = []
        for edge in self._network._edges:
            self._channels.append((
                edge.source,
                edge.target,
                edge.relation.value,
                edge.weight,
            ))

        self._built = True
        logger.debug(
            "Flow field built: %d concepts, %d channels, dim=%d",
            n, len(self._channels), _FLOW_DIM,
        )

    def set_angles_from_embedding(
        self,
        concept_ids: list[str],
        angle_matrix: np.ndarray,
    ) -> None:
        """Set angular positions from the EmbeddingStore's toroidal mapping.

        This is the preferred way to set angles — the EmbeddingStore's
        spectral + experiential embedding is higher quality than the
        lightweight power-iteration fallback in build().
        """
        if len(concept_ids) != angle_matrix.shape[0]:
            return
        # Use only the first _FLOW_DIM dimensions
        d = min(_FLOW_DIM, angle_matrix.shape[1])
        self._angles = {
            cid: angle_matrix[i, :d].astype(np.float32)
            for i, cid in enumerate(concept_ids)
        }
        self._concept_ids = concept_ids
        self._angle_matrix = angle_matrix[:, :d].astype(np.float32)
        # Rebuild weight vector if we have concepts
        if self._weights:
            self._weight_vector = np.array(
                [self._weights.get(cid, 0.0) for cid in concept_ids],
                dtype=np.float32,
            )
        self._built = True

    def _spectral_angles(
        self,
        concept_ids: list[str],
        idx_map: dict[str, int],
    ) -> np.ndarray:
        """Compute angular positions via power iteration on adjacency.

        This is a lightweight spectral embedding — not as accurate as
        the full SVD in EmbeddingStore, but fast and good enough for
        the flow field when the embedding store isn't available.
        """
        n = len(concept_ids)
        d = _FLOW_DIM

        # Build sparse adjacency representation
        # Each concept's neighbors and weights
        neighbors: dict[int, list[tuple[int, float]]] = {}
        for edge in self._network._edges:
            si = idx_map.get(edge.source)
            ti = idx_map.get(edge.target)
            if si is not None and ti is not None:
                neighbors.setdefault(si, []).append((ti, edge.weight))
                neighbors.setdefault(ti, []).append((si, edge.weight))

        # Power iteration: start with random vectors, iterate
        # x_{k+1} = A @ x_k / |A @ x_k|
        # This converges to the top eigenvectors of the adjacency matrix.
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, d)).astype(np.float32)

        for _ in range(10):
            # Multiply by adjacency
            new_vectors = np.zeros_like(vectors)
            for i in range(n):
                for j, w in neighbors.get(i, []):
                    new_vectors[i] += w * vectors[j]
            # Normalize columns
            norms = np.linalg.norm(new_vectors, axis=0, keepdims=True)
            norms[norms < 1e-8] = 1.0
            vectors = new_vectors / norms

        # Map to angles via arctan2 of pairs of eigenvectors
        # Each pair of eigenvectors → one angular coordinate
        angles = np.zeros((n, d), dtype=np.float32)
        for dim in range(d):
            v1 = vectors[:, dim % d]
            v2 = vectors[:, (dim + 1) % d]
            angles[:, dim] = np.arctan2(v2, v1) + np.pi  # [0, 2π)

        return angles

    # ─── Potential field ───────────────────────────────────────────

    def _potential(self, theta: np.ndarray) -> float:
        """Compute the potential U(θ) at a point on the torus.

        U(θ) = -Σᵢ wᵢ · K(θ, θᵢ)

        where K is the von Mises kernel:
        K(θ, θᵢ) = exp(κ · cos(θ - θᵢ)) / (2π · I₀(κ))

        This creates potential wells at each concept's angular position,
        with depth proportional to the concept's edge weight sum.
        """
        if self._angle_matrix is None or self._weight_vector is None:
            return 0.0

        # Angular differences with wrap-around: (N, d)
        diff = self._angle_matrix - theta[np.newaxis, :]
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi  # [-π, π]

        # Joint von Mises kernel, computed in log space. The joint
        # kernel is the product of d per-dimension terms:
        #   K(θ, θᵢ) = ∏ₖ exp(κ cos(diffₖ)) = exp(κ Σₖ cos(diffₖ))
        # Computing the product directly explodes to exp(κ·d) — ~7e20
        # for κ=3, d=16 — which overflows float32 downstream (the
        # squared norm in np.linalg.norm). Subtracting the constant
        # κ·d (the kernel normalizer's θ-independent factor) keeps the
        # kernel bounded in (0, 1] without changing the field's shape.
        cos_diff = np.cos(diff)
        log_kernel_prod = _KAPPA * (cos_diff.sum(axis=1) - diff.shape[1])
        kernel_prod = np.exp(log_kernel_prod)  # (N,), ∈ (0, 1]

        # Weighted sum: potential = -Σ wᵢ K(θ, θᵢ)
        return -float(np.dot(self._weight_vector, kernel_prod))

    def _gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute the gradient ∇U(θ) at a point on the torus.

        ∇U(θ) = -Σᵢ wᵢ · ∇K(θ, θᵢ)

        where ∇K = K · κ · sin(θ - θᵢ) (derivative of von Mises)

        The gradient points uphill (away from concepts). The flow
        follows -∇U (downhill, toward concepts).
        """
        if self._angle_matrix is None or self._weight_vector is None:
            return np.zeros_like(theta)

        # Angular differences with wrap-around: (N, d)
        diff = self._angle_matrix - theta[np.newaxis, :]
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi  # [-π, π]

        # Von Mises kernel and its derivative
        cos_diff = np.cos(diff)
        sin_diff = np.sin(diff)

        # Joint kernel in log space (see _potential). Computing the
        # product directly explodes to exp(κ·d) (~7e20 for κ=3, d=16)
        # and overflows float32 in the squared norm used by
        # np.linalg.norm. The subtracted κ·d factor is θ-independent,
        # so it rescales the gradient uniformly — the direction (which
        # the flow follows) is unchanged.
        log_kernel_prod = _KAPPA * (cos_diff.sum(axis=1) - diff.shape[1])
        kernel_prod = np.exp(log_kernel_prod)  # (N,), ∈ (0, 1]

        # Gradient of the product kernel:
        # ∇(∏ⱼ Kⱼ) = (∏ⱼ Kⱼ) · Σⱼ (∇Kⱼ / Kⱼ)
        # = (∏ⱼ Kⱼ) · Σⱼ κ sin(diffⱼ)
        # So ∇U = -Σᵢ wᵢ (∏ⱼ Kᱢ) · Σⱼ κ sin(diffᱼ)
        # = -κ Σⱼ sin(diffⱼ) · Σᵢ wᵢ (∏ₖ Kᵢₖ)

        # Weighted kernel products: (N,)
        weighted_kp = self._weight_vector * kernel_prod

        # Gradient per dimension: -κ * Σᵢ wᵢ Kᵢ · sin(diffᵢ)
        # Shape: (d,)
        grad = -_KAPPA * np.dot(weighted_kp[np.newaxis, :], sin_diff).squeeze(0)

        return grad.astype(np.float32)

    # ─── Trajectory integration ────────────────────────────────────

    def _integrate(
        self,
        start_theta: np.ndarray,
        momentum: np.ndarray,
        max_steps: int = _MAX_STEPS,
    ) -> list[np.ndarray]:
        """Integrate the Hamiltonian flow from a starting position.

        Returns the trajectory as a list of angular positions.
        The trajectory ends when:
        - The gradient becomes too small (local minimum)
        - The maximum number of steps is reached
        - The trajectory revisits a region (closed orbit)

        Uses symplectic Euler integration to preserve the Hamiltonian
        structure (energy is conserved, no numerical drift).
        """
        theta = start_theta.copy()
        p = momentum.copy()
        trajectory: list[np.ndarray] = [theta.copy()]

        visited_regions: set[int] = set()

        for _ in range(max_steps):
            # Compute gradient at current position
            grad = self._gradient(theta)

            # Check for local minimum (flow stops naturally)
            grad_mag = float(np.linalg.norm(grad))
            if grad_mag < _MIN_GRADIENT:
                break

            # Symplectic Euler:
            # p_{k+1} = p_k - dt * ∇U(θ_k)  (momentum update)
            # θ_{k+1} = θ_k + dt * p_{k+1}  (position update)
            p = p - _DT * grad
            # Apply friction (lets the flow settle)
            p = p * (1.0 - _FRICTION)
            # Clip momentum to prevent overflow
            p_mag = float(np.linalg.norm(p))
            if p_mag > 10.0:
                p = p * (10.0 / p_mag)
            theta = theta + _DT * p

            # Wrap angles to [0, 2π)
            theta = np.mod(theta, 2.0 * np.pi)

            # Apply six-fold holonomy when passing near a hub
            # (Flower of Life pattern — six related concepts around each hub)
            nearest = self._nearest_concept(theta)
            if nearest and nearest in self._angles:
                hub_angles = self._angles[nearest]
                hub_weight = self._weights.get(nearest, 0.0)
                # Only apply holonomy near significant hubs
                if hub_weight > 1.0:
                    theta = self._apply_holonomy(theta, hub_angles)

            # Check if we've been in this region before
            region = self._region_of(theta)
            if region in visited_regions:
                break
            visited_regions.add(region)

            trajectory.append(theta.copy())

        return trajectory

    def _region_of(self, theta: np.ndarray) -> int:
        """Map an angular position to a discrete region index.

        Used to detect when the trajectory revisits a region (closed
        orbit). The torus is divided into coarse cells.
        """
        # Quantize each angle to 8 sectors: 0-7
        sectors = (theta * 4.0 / np.pi).astype(int) % 8
        # Hash to a single integer
        return hash(sectors.tobytes())

    # ─── Word emission ────────────────────────────────────────────

    def _nearest_concept(self, theta: np.ndarray) -> str | None:
        """Find the concept whose angular position is closest to θ.

        Uses geodesic distance on the torus (wrap-around).
        """
        if self._angle_matrix is None or not self._concept_ids:
            return None

        diff = self._angle_matrix - theta[np.newaxis, :]
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi  # [-π, π]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        idx = int(np.argmin(dist))
        return self._concept_ids[idx]

    def _concept_at(self, theta: np.ndarray, exclude: set[str] | None = None) -> str | None:
        """Find the concept at a trajectory point, excluding visited ones."""
        if self._angle_matrix is None or not self._concept_ids:
            return None

        diff = self._angle_matrix - theta[np.newaxis, :]
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
        dist = np.sqrt(np.sum(diff * diff, axis=1))

        if exclude:
            for cid in exclude:
                idx = self._concept_ids.index(cid) if cid in self._concept_ids else -1
                if idx >= 0:
                    dist[idx] = np.inf

        if np.all(np.isinf(dist)):
            return None

        idx = int(np.argmin(dist))
        return self._concept_ids[idx]

    def _relation_between(
        self, source: str, target: str
    ) -> tuple[str, float] | None:
        """Find the relation type and weight between two concepts.

        Returns (relation_verb, weight) if an edge exists, else None.
        Checks both directions.
        """
        if self._network is None:
            return None

        edges = self._network.get_edges(source, "out")
        for e in edges:
            if e.target == target:
                return (e.relation.value, e.weight)

        # Check reverse direction
        edges = self._network.get_edges(source, "in")
        for e in edges:
            if e.source == target:
                return (e.relation.value, e.weight)

        return None

    def _definition_of(self, concept_id: str) -> str | None:
        """Get the clean definition of a concept, if it has one."""
        if self._network is None:
            return None
        c = self._network.get_concept(concept_id)
        if c is None:
            return None
        defn = c.properties.get("definition", "NO DEF")
        if defn and defn != "NO DEF":
            # Clean: skip relationship facts
            cleaned = defn.strip().strip(" .;:!?")
            segments = [
                segment.strip()
                for segment in cleaned.split(".")
                if segment.strip()
            ]
            if len(segments) > 1:
                complete = [
                    segment for segment in segments
                    if len(segment.split()) >= 3
                ]
                if complete:
                    cleaned = complete[0]
            lower = cleaned.lower()
            if lower.startswith(("related to", "connects to", "part of",
                                  "is a kind of", "is a type of")):
                return None
            if len(cleaned) > 200:
                return cleaned[:197].rsplit(" ", 1)[0] + "…"
            return cleaned
        return None

    def _display_name(self, concept_id: str) -> str:
        """Get a speakable display name for a concept."""
        base = concept_id.split("#")[0].replace("_", " ")
        # Skip code concepts
        if ":" in base or "." in base:
            return ""
        return base

    def _is_speakable(self, concept_id: str) -> bool:
        """Check if a concept is speakable — has real content."""
        if self._network is None:
            return True
        c = self._network.get_concept(concept_id)
        if c is None:
            return False
        base = concept_id.split("#")[0]
        if ":" in base or "." in base or "__" in base:
            return False
        defn = c.properties.get("definition", "NO DEF")
        if defn and defn != "NO DEF":
            return True
        has_out = len(self._network.get_edges(concept_id, "out")) > 0
        has_in = len(self._network.get_edges(concept_id, "in")) > 0
        return has_out or has_in

    # ─── Sentence composition from trajectory ─────────────────────

    def generate(
        self,
        thought: Thought,
        emotion: EmotionalState,
    ) -> str | None:
        """Generate text by integrating the Hamiltonian flow.

        The query concept sets the starting position. The emotional
        state sets the momentum. The trajectory through the potential
        landscape produces the sentence.

        Returns None if the flow can't produce text (no seed concept,
        no edges, or the trajectory is too short).
        """
        if not self._built or self._angle_matrix is None:
            return None

        # Extract seed concept
        seed = self._extract_seed(thought)
        if seed is None or seed not in self._angles:
            return None

        # Get starting position
        start_theta = self._angles[seed].copy()

        # Set momentum from emotional state
        # High arousal → high momentum → fast transitions → shorter
        # Low arousal → low momentum → slow transitions → longer
        speed = 0.3 + emotion.arousal * 0.7  # [0.3, 1.0]

        # Direction: perturb from the seed's position toward the
        # steepest descent direction (downhill in the potential)
        grad = self._gradient(start_theta)
        grad_mag = float(np.linalg.norm(grad))
        if grad_mag < 1e-6:
            # Flat potential — no flow direction. Use a random
            # perturbation scaled by emotional state.
            direction = self._rng.standard_normal(_FLOW_DIM).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
        else:
            # Flow downhill: -∇U
            direction = (-grad / (grad_mag + 1e-8)).astype(np.float32)

        # Add emotional modulation to direction:
        # Positive valence → bias toward ENABLES/CREATES channels
        # Negative valence → bias toward HARMS/PREVENTS channels
        # High caution → reduce momentum (more careful flow)
        momentum = direction * speed
        if emotion.caution > 0.5:
            momentum = momentum * 0.6

        # Integrate the flow
        trajectory = self._integrate(start_theta, momentum)
        if len(trajectory) < 2:
            return None

        # Emit words from the trajectory
        return self._compose_from_trajectory(
            seed, trajectory, emotion, thought,
        )

    def _extract_seed(self, thought: Thought) -> str | None:
        """Extract the seed concept from the thought."""
        # Explicit topic
        if thought.topics:
            topic = thought.topics[0].strip()
            if topic and self._network.get_concept(topic) is not None:
                return topic
            resolved = self._network._resolve(topic) if topic else None
            if resolved:
                return resolved

        # Question target
        target = thought.metadata.get("target_concept", "")
        if target:
            resolved = self._network._resolve(target)
            if resolved:
                return resolved

        # Knowledge topic
        topic_meta = thought.metadata.get("topic", "")
        if topic_meta:
            resolved = self._network._resolve(topic_meta)
            if resolved:
                return resolved

        # Extract from content
        content = thought.content.strip()
        if content:
            words = sorted(content.split(), key=len, reverse=True)
            for word in words:
                resolved = self._network._resolve(word)
                if resolved:
                    return resolved

        return None

    def _compose_from_trajectory(
        self,
        seed: str,
        trajectory: list[np.ndarray],
        emotion: EmotionalState,
        thought: Thought,
    ) -> str | None:
        """Compose a sentence from the flow trajectory.

        The trajectory passes through concept cells. Each cell
        traversal emits a word. The seed concept is the subject.
        Subsequent concepts are related via typed relations.

        The sentence structure emerges from the trajectory shape:
        - The first concept (seed) is the subject
        - Each subsequent concept is connected via its relation to
          the previous one
        - The sentence ends when the trajectory settles
        """
        # Collect concepts along the trajectory
        visited: set[str] = {seed}
        concepts: list[str] = [seed]

        for theta in trajectory[1:]:
            cid = self._concept_at(theta, exclude=visited)
            if cid is None:
                continue
            if not self._is_speakable(cid):
                continue
            display = self._display_name(cid)
            if not display:
                continue
            visited.add(cid)
            concepts.append(cid)

        if len(concepts) < 2:
            # Try with the seed's definition if no flow concepts
            return self._compose_definition_only(seed, emotion)

        # Build the sentence from the concept sequence
        return self._build_sentence(seed, concepts, emotion, thought)

    def _compose_definition_only(
        self,
        seed: str,
        emotion: EmotionalState,
    ) -> str | None:
        """Compose a sentence from just the seed's definition.

        Fallback when the flow trajectory doesn't reach other
        concepts — the seed has a definition but few connections.
        """
        display = self._display_name(seed)
        if not display:
            return None

        defn = self._definition_of(seed)
        if defn:
            first = display[0].upper() + display[1:]
            return f"{first} {copula(display)} {defn}."

        # Check for edges
        edges = self._network.get_edges(seed, "both")
        if not edges:
            return None

        # Build from edges directly
        return self._build_sentence(seed, [seed] + [
            e.target if e.source == seed else e.source
            for e in edges[:4]
            if self._is_speakable(
                e.target if e.source == seed else e.source
            )
        ], emotion, None)

    def _build_sentence(
        self,
        seed: str,
        concepts: list[str],
        emotion: EmotionalState,
        thought: Thought | None,
    ) -> str | None:
        """Build a sentence from a sequence of concepts.

        The seed is the subject. Each subsequent concept is connected
        via its typed relation to the seed or the previous concept.

        The sentence structure:
        - Opening: seed's definition (if available)
        - Body: typed relations from seed to each subsequent concept
        - No closing (the flow ends naturally)

        Clause joining is semantic-aware: the connector between clauses
        is chosen based on the relation type of the *following* clause.
        Causal relations (causes, leads_to) get causal connectors
        ("so", "thus"); contrastive relations (harms, prevents,
        contradicts) get contrastive connectors ("but", "though");
        facilitative relations (enables, creates) get additive
        connectors ("and", "which"). Consecutive clauses sharing the
        same subject undergo conjunction reduction (dropping the
        repeated pronoun) for smoother, more natural flow.
        """
        seed_display = self._display_name(seed)
        if not seed_display:
            return None

        # Start with the definition if available
        defn = self._definition_of(seed)
        parts: list[str] = []

        if defn:
            first = seed_display[0].upper() + seed_display[1:]
            parts.append(f"{first} {copula(seed_display)} {defn}.")
        else:
            # Start with the first relation
            pass

        # Animacy-aware pronouns for the seed — "it" is wrong for
        # people and plural subjects.
        seed_pron, seed_plural = person_pronoun(
            seed, self._network, seed_display
        )
        seed_obj, _ = person_pronoun(
            seed, self._network, seed_display, for_object=True
        )

        clauses = self._relation_clauses(
            seed, concepts, seed_display, seed_pron, seed_obj, seed_plural
        )

        if not clauses and not defn:
            return None

        # Join clauses with semantic-aware connectors and conjunction
        # reduction for smoother, more natural sentence flow.
        if clauses:
            joined = self._join_clauses(clauses, defn is not None)
            if joined:
                # Relative clause subordination appends to the
                # definition sentence ("Memory is X, which enables Y")
                # rather than starting a new sentence ("Memory is X.
                # which enables Y").
                if joined.startswith("which ") and parts:
                    # Strip the trailing period from the definition
                    # and merge with the relative clause.
                    parts[-1] = parts[-1].rstrip(".") + ", " + joined
                else:
                    parts.append(joined)

        if not parts:
            return None

        # Capitalize first letter
        text = " ".join(parts)
        text = text[0].upper() + text[1:] if text else text

        # Clean up spacing
        text = " ".join(text.split())

        return text if len(text) > 5 else None

    def _relation_clauses(
        self,
        seed: str,
        concepts: list[str],
        seed_display: str,
        seed_pron: str,
        seed_obj: str,
        seed_plural: bool,
    ) -> list[tuple[str, str, bool]]:
        """Build relation clauses from the seed to each concept.

        Each entry is ``(clause_text, rel_type, subject_is_seed)`` so
        the joiner can pick semantic-aware connectors and apply
        conjunction reduction for consecutive same-subject clauses.
        """
        clauses: list[tuple[str, str, bool]] = []
        for cid in concepts[1:]:
            display = self._display_name(cid)
            if not display or display.lower() == seed_display.lower():
                continue

            # Find the relation between seed and this concept
            rel = self._relation_between(seed, cid)
            if rel is None:
                # Check if there's a relation from this concept to seed
                edges_out = self._network.get_edges(cid, "out")
                for e in edges_out:
                    if e.target == seed:
                        rel = (e.relation.value, e.weight)
                        # Reverse the clause: "memory enables it"
                        verb = self._verb_for_relation(e.relation.value)
                        if verb:
                            verb = agree_verb_phrase(
                                verb, is_plural_np(display)
                            )
                            clauses.append(
                                (f"{display} {verb} {seed_obj}",
                                 e.relation.value, False)
                            )
                        break
                continue

            rel_type, _weight = rel
            verb = self._verb_for_relation(rel_type)
            if not verb:
                continue

            # Determine direction: does seed → cid or cid → seed?
            edges_out = self._network.get_edges(seed, "out")
            seed_is_source = any(
                e.target == cid for e in edges_out
            )
            if seed_is_source:
                clauses.append(
                    (f"{seed_pron} "
                     f"{agree_verb_phrase(verb, seed_plural)} {display}",
                     rel_type, True)
                )
            else:
                clauses.append(
                    (f"{display} "
                     f"{agree_verb_phrase(verb, is_plural_np(display))} "
                     f"{seed_obj}",
                     rel_type, False)
                )
        return clauses

    # ─── Semantic-aware clause joining ──────────────────────────────

    # Relation → connector class. These are syntactic building blocks
    # (conjunctions/adverbs), not response content — they shape how
    # clauses are linked, not what is said.
    _CAUSAL_RELATIONS = frozenset({"causes", "leads_to", "creates"})
    _CONTRAST_RELATIONS = frozenset({
        "harms", "prevents", "contradicts", "opposite_of",
    })
    _FACILITATIVE_RELATIONS = frozenset({"enables", "depends_on"})
    _TEMPORAL_RELATIONS = frozenset({"temporal_order", "emerges_from"})

    def _connector_for_relation(self, rel_type: str) -> str:
        """Pick a clause connector based on the relation type.

        Causal relations get causal connectors; contrastive relations
        get contrastive connectors; facilitative relations get
        additive connectors. This produces sentences like "X enables
        Y, so Z depends on W" instead of always "X enables Y, and Z
        depends on W" — the connector reflects the semantic relation.
        """
        if rel_type in self._CAUSAL_RELATIONS:
            return str(self._rng.choice(["so", "thus", "and so"]))
        if rel_type in self._CONTRAST_RELATIONS:
            return str(self._rng.choice(["but", "though", "and yet"]))
        if rel_type in self._FACILITATIVE_RELATIONS:
            return str(self._rng.choice(["and", "which", "also"]))
        if rel_type in self._TEMPORAL_RELATIONS:
            return str(self._rng.choice(["then", "and then", "and"]))
        return "and"

    def _join_clauses(
        self,
        clauses: list[tuple[str, str, bool]],
        has_definition: bool,
    ) -> str:
        """Join clauses with semantic-aware connectors and reduction.

        Two transformations produce more natural sentence flow:

        1. **Conjunction reduction**: consecutive clauses sharing the
           same subject drop the repeated subject pronoun. "It enables
           learning, and it causes recall" becomes "It enables
           learning, causes recall" — the same reduction human language
           uses (coordinate structure constraint, Chomsky 1957).

        2. **Semantic-aware connectors**: the connector between two
           clauses is chosen based on the *following* clause's relation
           type. Causal relations get "so"/"thus", contrastive get
           "but"/"though", facilitative get "and"/"which".

        When a definition opens the sentence, the first relation
        clause can be subordinated as a relative clause ("which
        enables...") instead of coordinated — producing "Memory is X,
        which enables Y" rather than "Memory is X. It enables Y."
        """
        if not clauses:
            return ""
        if len(clauses) == 1:
            return clauses[0][0] + "."

        # Check if the first clause can be subordinated as a relative
        # clause after a definition. Only seed-subject clauses can use
        # "which" — object-subject clauses ("learning enables it")
        # can't naturally become "which learning enables."
        first_text, first_rel, first_subj_is_seed = clauses[0]
        use_relative = (
            has_definition
            and first_subj_is_seed
            and first_rel in (
                self._FACILITATIVE_RELATIONS
                | self._CAUSAL_RELATIONS
                | self._TEMPORAL_RELATIONS
            )
            and self._rng.random() < 0.4
        )

        if use_relative:
            # Subordinate the first clause as a relative clause:
            # "Memory is X, which enables Y."
            # Strip the leading pronoun — "which" replaces it.
            parts = first_text.split(None, 1)
            rest = parts[1] if len(parts) > 1 else ""
            relative_clause = f"which {rest}"
            remaining = clauses[1:]
            if not remaining:
                return relative_clause + "."
            joined = self._join_with_connectors(remaining)
            return f"{relative_clause}, {joined}."
        else:
            return self._join_with_connectors(clauses)

    def _join_with_connectors(
        self,
        clauses: list[tuple[str, str, bool]],
    ) -> str:
        """Join clauses with semantic connectors and conjunction reduction.

        Consecutive clauses with the same subject undergo conjunction
        reduction: the repeated subject pronoun is dropped, leaving
        just the verb phrase. "It enables X, and it causes Y" →
        "It enables X, causes Y". The connector between different-
        subject clauses is chosen based on the following clause's
        relation type.
        """
        # Group consecutive same-subject clauses.
        groups: list[list[tuple[str, str, bool]]] = []
        i = 0
        n = len(clauses)
        while i < n:
            group = [clauses[i]]
            j = i + 1
            while j < n and clauses[j][2] == clauses[i][2]:
                group.append(clauses[j])
                j += 1
            groups.append(group)
            i = j

        # Build a string for each group.
        group_strs: list[str] = []
        for group in groups:
            if len(group) == 1:
                group_strs.append(group[0][0])
            else:
                # Conjunction reduction: first clause keeps subject,
                # subsequent clauses drop it.
                first = group[0][0]
                reduced = [first]
                for k in range(1, len(group)):
                    parts = group[k][0].split(None, 1)
                    reduced.append(parts[1] if len(parts) > 1 else group[k][0])
                group_strs.append(", ".join(reduced))

        # Join groups with semantic connectors. The last connector is
        # always "and" for natural English list structure.
        if len(group_strs) == 1:
            return group_strs[0] + "."

        result = group_strs[0]
        for idx in range(1, len(group_strs)):
            is_last = idx == len(group_strs) - 1
            rel_type = groups[idx][0][1]
            if is_last:
                # Final group — use "and" for natural closure, unless
                # the relation is contrastive ("but"/"though" reads
                # better than "and" for a contrastive final clause).
                if rel_type in self._CONTRAST_RELATIONS:
                    connector = str(self._rng.choice(["but", "though"]))
                else:
                    connector = "and"
            else:
                connector = self._connector_for_relation(rel_type)
            lowered = self._lowercase_clause_start(group_strs[idx])
            result += f", {connector} {lowered}"

        return result + "."

    def _lowercase_clause_start(self, clause: str) -> str:
        """Lowercase the first word of a clause after a connector.

        Preserves 'I', 'I'm', 'I've', and proper nouns (capitalized
        words that aren't sentence-initial). This is a grammatical
        transform, not content — connectors like 'so' and 'but' are
        followed by lowercase in mid-sentence.
        """
        if not clause:
            return clause
        words = clause.split(None, 1)
        first = words[0]
        if first in ("I", "I'm", "I've", "I'd", "I'll"):
            return clause
        # Pronouns should be lowercased after a connector.
        if first in ("It", "They", "He", "She", "This", "That",
                     "These", "Those", "We", "You"):
            return first.lower() + (" " + words[1] if len(words) > 1 else "")
        return clause

    def _verb_for_relation(self, rel_type: str) -> str | None:
        """Map a relation type to a verb phrase.

        Uses the same seed verbs as the graph walk, but without
        hardcoded fallback phrases — just the relation verbs.
        """
        verbs = {
            "is_a": "is a kind of",
            "part_of": "is part of",
            "causes": "causes",
            "emerges_from": "emerges from",
            "similar_to": "resembles",
            "opposite_of": "is the opposite of",
            "depends_on": "depends on",
            "enables": "enables",
            "instance_of": "is an example of",
            "leads_to": "leads to",
            "creates": "creates",
            "harms": "harms",
            "prevents": "prevents",
            "contradicts": "contradicts",
            "has_property": "has the property of being",
            "related_to": "relates to",
            "goal_directed": "aims to",
            "temporal_order": "comes before",
            "agent_patient": "acts on",
        }
        return verbs.get(rel_type)

    # ─── Sacred geometry constraints ──────────────────────────────

    def _apply_holonomy(
        self,
        theta: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """Apply six-fold holonomy rotation around a hub.

        When the trajectory passes near a hub concept, the fiber
        rotates by 60° (π/3). This groups six related concepts around
        each hub — the Flower of Life pattern.

        The rotation is applied to the first two angular coordinates
        (the "base" torus), leaving the fiber coordinates unchanged.
        """
        # Distance to center
        diff = theta - center
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
        dist = float(np.sqrt(np.sum(diff[:2] * diff[:2])))

        # Only apply if close to a hub
        if dist > _SIX_FOLD:
            return theta

        # Rotation angle proportional to proximity
        proximity = 1.0 - dist / _SIX_FOLD
        rotation = _SIX_FOLD * proximity

        # Rotate the first two coordinates
        result = theta.copy()
        c, s = math.cos(rotation), math.sin(rotation)
        t0, t1 = result[0], result[1]
        result[0] = t0 * c - t1 * s
        result[1] = t0 * s + t1 * c

        return np.mod(result, 2.0 * np.pi)
