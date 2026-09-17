"""Concept dynamics — spreading activation and cortical ticks."""

from __future__ import annotations

import heapq
import random
from collections import deque
from typing import TYPE_CHECKING, Any

from .classify import _column_of
from .types import ConceptCategory, ConceptModality, Edge


class DynamicsMixin:
    """Mixin for :class:`ConceptNetwork` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def ground_activation(self, boost: float = 0.05) -> dict[str, float]:
        """Apply modality-specific activation boosts (sensorimotor grounding).

        When concepts in a given modality are active, other concepts in
        the same modality receive a small activation boost. This models
        within-modality priming — the sensorimotor grounding of concepts
        means that activating one visual concept slightly primes other
        visual concepts, one motor concept primes other motor concepts, etc.

        This is the mechanism by which action concepts get sensorimotor
        grounding: when a MOTOR concept activates, it slightly activates
        other MOTOR concepts (within-modality spreading), simulating the
        weak activation of related motor representations.

        Only concepts with activation above a threshold (0.1) contribute
        to the priming. The boost is proportional to the mean activation
        of active concepts in that modality.

        Args:
            boost: Maximum boost amount per modality (default 0.05).

        Returns:
            A dict mapping modality names to the boost applied.
        """
        ACTIVATION_THRESHOLD = 0.1

        # Group active concepts by modality
        modality_active: dict[ConceptModality, list[tuple[str, float]]] = {}
        for cid, concept in list(self._concepts.items()):
            if concept.modality == ConceptModality.UNKNOWN:
                continue
            if concept.modality == ConceptModality.MIXED:
                continue
            if concept.activation > ACTIVATION_THRESHOLD:
                modality_active.setdefault(concept.modality, []).append((cid, concept.activation))

        boosts_applied: dict[str, float] = {}

        for modality, active_list in modality_active.items():
            if not active_list:
                continue

            # Compute mean activation of active concepts in this modality
            mean_activation = sum(a for _, a in active_list) / len(active_list)
            modality_boost = boost * mean_activation

            # Apply boost to all inactive concepts in this modality
            active_ids = {cid for cid, _ in active_list}
            for cid, concept in list(self._concepts.items()):
                if concept.modality != modality:
                    continue
                if cid in active_ids or concept.activation >= ACTIVATION_THRESHOLD:
                    continue
                concept.activation = min(1.0, concept.activation + modality_boost)

            boosts_applied[modality.value] = modality_boost

        return boosts_applied
    def spread_activation(
        self,
        concepts: list[str],
        amount: float = 0.2,
        depth: int = 2,
        column_context: set[str] | None = None,
    ) -> dict[str, float]:
        """Spread activation from given concepts through the network.

        This is spreading activation — a classic cognitive science
        model where activating one concept primes related concepts.
        Used to find what's relevant to the current conversation.

        Column-aware spreading (cortical column model):
        - Activation spreads densely within the same column (full strength)
        - Activation crosses bridge edges to other columns at reduced
          strength (BRIDGE_ATTENUATION = 0.3), mirroring how cortical
          association fibers are weaker than intra-column connections
        - When column_context is provided (the VIP disinhibition signal),
          concepts in context columns receive full activation while
          concepts in non-context columns receive attenuated activation

        Returns a dict of {concept_id: activation_level} for all
        activated concepts.
        """
        (bridge_attenuation, context_penalty, category_spread_multipliers,
         hub_spread_multiplier, modality_spread_multipliers,
        ) = self._spread_activation_constants()

        activated: dict[str, float] = {}

        # Start with the given concepts
        frontier: deque[tuple[str, int]] = deque()
        for name in concepts:
            cid = self._resolve(name)
            if cid:
                activated[cid] = amount
                frontier.append((cid, 0))

        # Spread activation with fan-out normalization and depth decay.
        #
        # Activation decays with path length (depth) using exponential
        # decay: spread = source_activation * (edge_weight / total_weight)
        # times decay to the power of depth.
        #
        # The fan-out normalization (edge_weight / total_weight) ensures
        # that a concept with many neighbors doesn't spread more total
        # activation than one with few neighbors — each neighbor gets a
        # share proportional to the edge weight. This is the standard
        # spreading activation model (Anderson & Pirolli, 1984).
        decay = 0.5  # per-depth attenuation
        while frontier:
            current, d = frontier.popleft()
            if d >= depth:
                continue

            edges = self._edge_index.get(current, [])
            if not edges:
                continue
            total_weight = sum(e.weight for e in edges)
            if total_weight < 1e-8:
                continue

            current_cols = self.get_columns_for(current)

            # Category-specific spread multiplier for the source concept.
            # Living things spread faster; abstract concepts spread more
            # widely. Applied to the base spread before column attenuation.
            category_mult, hub_mult, modality_mult = self._get_concept_multipliers(
                current, category_spread_multipliers, hub_spread_multiplier,
                modality_spread_multipliers)

            for edge in edges:
                self._spread_one_edge(
                    activated, frontier, current, edge, d, current_cols,
                    column_context, decay, total_weight, bridge_attenuation,
                    context_penalty, category_mult, hub_mult, modality_mult)

            # NOTE: Holographic associations are NOT queried during
            # spreading activation. The holographic graph query involves
            # FFT-based circular correlation + dot products against all
            # ~3000 registered concepts, which is too expensive to run
            # for every concept in the frontier at every depth level.
            # With depth=2 and a large starting set, this would trigger
            # hundreds of FFT queries and blow past the 15s cognition
            # timeout. Holographic associations are still accessible via
            # get_neighbors() for direct lookups (which are less frequent
            # and can benefit from caching).

        return activated
    def _spread_activation_constants(
        self,
    ) -> tuple[float, float, dict[ConceptCategory, float], float, dict[ConceptModality, float]]:
        """Return spread activation constants."""
        # Bridge attenuation: inter-column spread is weaker than
        # intra-column spread. This mirrors the difference between
        # local cortical connections (strong) and long-range
        # association fibers (weaker).
        #
        # Biological basis: intra-columnar connections have ~3-10x
        # higher synaptic density than inter-columnar association
        # fibers (Binzegger et al., 2004; Douglas & Martin, 2004).
        # A bridge attenuation of 0.3 means cross-column spread is
        # 30% of intra-column strength — within the biological range.
        BRIDGE_ATTENUATION = 0.3

        # Context penalty: when column_context is provided, concepts
        # NOT in context columns receive a SUBTRACTIVE penalty on top
        # of bridge attenuation. This is applied additively, not
        # multiplicatively, to avoid compounding below the biological
        # range. A penalty of 0.2 means non-context cross-column
        # spread is 30% - 20% = 10% of intra-column strength (still
        # within the 10-33% biological range, at the low end).
        #
        # This mirrors how VIP interneuron disinhibition works: the
        # context signal doesn't change the association fiber strength,
        # it changes how much inhibition the target column receives.
        # The net effect is a moderate bias, not a complete shutdown.
        CONTEXT_PENALTY = 0.2

        # Category-specific activation multipliers.
        #
        # Different categories of concepts have different spreading
        # dynamics, modeling category-specific cortical organization:
        #
        # - LIVING things: slightly faster spreading activation. The
        #   temporal subsystem (where living things are processed) has dense
        #   sensory association networks, and biological motion detection
        #   is fast (superior temporal sulcus responds to biological
        #   motion within ~100ms). A 1.15x multiplier captures this.
        #   (Puce & Perrett, 2003; Beauchamp et al., 2002)
        #
        # - ABSTRACT concepts: more distributed activation, recruiting
        #   more neighbors. Abstract concepts don't have a single
        #   sensory-motor anchor — they rely on distributed networks
        #   across prefrontal and temporal cortex. A 1.2x multiplier
        #   on spread means they activate a wider neighborhood.
        #   (Wang et al., 2010; Binder & Desai, 2011)
        #
        # NON_LIVING and EVENT categories use the default (1.0) — no
        # special multiplier. UNKNOWN also defaults to 1.0.
        CATEGORY_SPREAD_MULTIPLIERS: dict[ConceptCategory, float] = {
            ConceptCategory.LIVING: 1.15,
            ConceptCategory.ABSTRACT: 1.2,
        }

        # Semantic hub spread multiplier.
        #
        # Semantic hubs (high-betweenness convergence zones, Damasio 1989)
        # have stronger activation influence — they integrate information
        # from diverse sources and broadcast to a wider neighborhood.
        # A 1.3x spread multiplier means hubs spread activation 30%
        # more strongly than non-hub concepts.
        HUB_SPREAD_MULTIPLIER = 1.3

        # Modality-specific spread multipliers.
        #
        # Visual concepts activate faster — the visual system has the
        # fastest processing pathways (magnocellular ~100ms). A 1.1x
        # multiplier captures this faster spread.
        # Motor concepts have more focused spreading — motor representations
        # are tightly coupled to action sequences, so they spread to fewer
        # but stronger neighbors. We keep the default (1.0) for motor.
        MODALITY_SPREAD_MULTIPLIERS: dict[ConceptModality, float] = {
            ConceptModality.VISUAL: 1.1,
        }

        return (BRIDGE_ATTENUATION, CONTEXT_PENALTY, CATEGORY_SPREAD_MULTIPLIERS,
                HUB_SPREAD_MULTIPLIER, MODALITY_SPREAD_MULTIPLIERS)
    def _get_concept_multipliers(
        self,
        current: str,
        category_spread_multipliers: dict[ConceptCategory, float],
        hub_spread_multiplier: float,
        modality_spread_multipliers: dict[ConceptModality, float],
    ) -> tuple[float, float, float]:
        """Return category, hub, and modality multipliers for a concept."""
        current_concept = self._concepts.get(current)
        category_mult = 1.0
        hub_mult = 1.0
        modality_mult = 1.0
        if current_concept is not None:
            category_mult = category_spread_multipliers.get(current_concept.category, 1.0)
            # Semantic hubs get stronger spread (1.3x)
            if current_concept.is_semantic_hub:
                hub_mult = hub_spread_multiplier
            # Modality-specific spread (visual concepts spread faster)
            modality_mult = modality_spread_multipliers.get(current_concept.modality, 1.0)
        return category_mult, hub_mult, modality_mult
    def _spread_one_edge(
        self,
        activated: dict[str, float],
        frontier: deque[tuple[str, int]],
        current: str,
        edge: Edge,
        d: int,
        current_cols: set[str],
        column_context: set[str] | None,
        decay: float,
        total_weight: float,
        bridge_attenuation: float,
        context_penalty: float,
        category_mult: float,
        hub_mult: float,
        modality_mult: float,
    ) -> None:
        """Compute spread for one edge and update activation/frontier."""
        share = edge.weight / total_weight
        # Decay applies to the edge being traversed (d+1),
        # so direct neighbors (d=0) get decay^1 = 0.5,
        # second-degree neighbors get decay^2 = 0.25, etc.
        spread = (
            activated[current]
            * share
            * (decay ** (d + 1))
            * category_mult
            * hub_mult
            * modality_mult
        )

        # Column-aware attenuation: if this edge crosses columns
        # (bridge edge or target in different column), attenuate
        target_cols = self.get_columns_for(edge.target)
        same_column = current_cols & target_cols  # intersection

        if not same_column:
            # Inter-column spread — apply bridge attenuation
            # Save the pre-bridge spread for the context penalty
            # calculation below, avoiding a redundant recomputation
            # of the same product.
            pre_bridge_spread = spread
            spread *= bridge_attenuation

            # Context penalty (VIP): if column_context is
            # provided and the target is NOT in a context column,
            # apply a SUBTRACTIVE penalty. This avoids the
            # compounding problem where bridge * context
            # attenuation drops below the biological range.
            if column_context is not None and not (target_cols & column_context):
                # Reduce by context_penalty fraction of the
                # original (pre-bridge) spread, not the current
                # spread. This keeps the total within biological
                # range.
                spread -= pre_bridge_spread * context_penalty
                spread = max(0.0, spread)

        if edge.target not in activated or activated[edge.target] < spread:
            activated[edge.target] = max(activated.get(edge.target, 0), spread)
            frontier.append((edge.target, d + 1))
    def decay_activation(self, rate: float = 0.05) -> None:
        """Decay all concept activations using exponential decay.

        Exponential decay: a(t+1) = a(t) * (1 - rate)
        This is biologically plausible — neural activation decays
        proportionally to its current level, not by a fixed amount.
        A concept at activation 1.0 loses 0.05, but a concept at 0.1
        loses only 0.005. This preserves recently-activated concepts
        longer than ones that were barely activated.

        Semantic hubs decay more slowly (0.7x decay rate) — they are
        convergence zones that maintain activation longer to integrate
        information across the network (Damasio, 1989).

        Motor concepts have shorter decay (1.3x decay rate) — motor
        representations are transient, activating briefly for action
        execution then fading quickly (motor cortex shows rapid
        onset-offset dynamics).
        """
        # Hub decay multiplier: hubs decay slower (0.7x rate)
        HUB_DECAY_MULTIPLIER = 0.7
        # Motor modality decay multiplier: motor concepts decay faster
        MOTOR_DECAY_MULTIPLIER = 1.3

        for concept in list(self._concepts.values()):
            if concept.activation is None:
                concept.activation = 0.0
            effective_rate = rate
            # Semantic hubs decay slower
            if concept.is_semantic_hub:
                effective_rate *= HUB_DECAY_MULTIPLIER
            # Motor concepts decay faster
            if concept.modality == ConceptModality.MOTOR:
                effective_rate *= MOTOR_DECAY_MULTIPLIER
            decay_factor = 1.0 - effective_rate
            concept.activation *= decay_factor
    def cortical_tick(
        self,
        arousal: float = 0.5,
        gaba: float = 0.3,
        ach: float = 0.3,
        serotonin: float = 0.4,
        noise: float = 0.01,
    ) -> dict[str, float]:
        """Evolve the network's activation state by one tick.

        This is the cortical tick — the continuous dynamics that make
        the concept network a living system rather than an inert data
        structure. Every call advances activation by one step:

        1. **Spreading**: Each concept with activation above a
           threshold spreads a fraction of its activation to
           neighbors, proportional to edge weight with fan-out
           normalization. This is the same principle as
           ``spread_activation`` but applied continuously to the
           stored activations, not as a one-shot BFS.

        2. **Decay**: All activations decay exponentially. The decay
           rate is modulated by GABA (more GABA → faster decay, she's
           calming down) and arousal (more arousal → slower decay,
           she's alert and things stay in mind).

        3. **Noise**: Small random perturbations are added to each
           concept's activation. This is spontaneous neural noise —
           the background chatter of a living brain. It prevents the
           network from settling into a dead state and allows
           serendipitous activation of dormant concepts. The noise
           amplitude is modulated by serotonin (low serotonin →
           more noise, modeling the noisy, ruminative quality of
           low-serotonin states).

        4. **Neurochemical modulation**:
           - **Arousal** (NE + DA): scales the spread rate. High
             arousal → activation spreads faster and further.
           - **GABA**: scales the decay rate. High GABA → faster
             decay, the mind quiets down.
           - **ACh** (acetylcholine): focuses activation. High ACh
             sharpens the winner-take-all dynamics — the most active
             concepts get a boost, the rest get suppressed. This
             models the cholinergic attention signal from the basal
             forebrain.
           - **Serotonin**: modulates noise. Low serotonin → more
             noise (noisy, ruminative). High serotonin → less noise
             (stable, calm).

        Returns a dict of {concept_id: activation} for concepts that
        are currently above the activation threshold (0.1). This lets
        the caller know what's "on her mind" right now.
        """
        if not self._concepts:
            return {}

        # ─── Modulated parameters ───────────────────────────────
        spread_rate, decay_rate, noise_amp, ach_focus = (
            self._compute_tick_parameters(arousal, gaba, ach, serotonin, noise)
        )

        # ─── 1. Spreading ───────────────────────────────────────
        spread_delta = self._compute_spread_delta(spread_rate)

        # ─── 2. Decay + noise + apply spread ────────────────────
        self._apply_decay_noise_spread(decay_rate, noise_amp, spread_delta)

        # ─── 3. ACh focus (winner-take-all) ─────────────────────
        self._apply_ach_focus(ach_focus)

        # ─── 4. Return what's on her mind ───────────────────────
        return {
            cid: c.activation
            for cid, c in list(self._concepts.items())
            if c.activation > 0.1
        }
    def _compute_tick_parameters(
        self,
        arousal: float,
        gaba: float,
        ach: float,
        serotonin: float,
        noise: float,
    ) -> tuple[float, float, float, float]:
        """Compute modulated parameters for the cortical tick.

        Returns (spread_rate, decay_rate, noise_amp, ach_focus).
        """
        # Base spread rate: how much of a concept's activation flows
        # to neighbors each tick. Arousal scales this up; GABA scales
        # it down. Range: ~0.02 (drowsy) to ~0.15 (highly alert).
        base_spread = 0.06
        spread_rate = base_spread * (0.5 + arousal) * (1.0 - gaba * 0.5)
        spread_rate = max(0.01, min(0.20, spread_rate))

        # Base decay rate: how fast activations fade. GABA increases
        # decay (the mind quiets); arousal decreases it (things stay
        # in mind when alert). Range: ~0.02 (alert) to ~0.12 (calm).
        base_decay = 0.05
        decay_rate = base_decay * (1.0 + gaba * 0.8) * (1.5 - arousal)
        decay_rate = max(0.01, min(0.20, decay_rate))

        # Noise amplitude: low serotonin → more noise. Range: ~0.002
        # (stable) to ~0.02 (noisy/ruminative).
        noise_amp = noise * (2.0 - serotonin * 1.5)
        noise_amp = max(0.001, min(0.05, noise_amp))

        # ACh focus: high ACh → stronger winner-take-all. The top
        # concepts get boosted, the rest get suppressed.
        ach_focus = ach * 0.3  # 0 to 0.3 boost for top concepts

        return spread_rate, decay_rate, noise_amp, ach_focus
    def _compute_spread_delta(self, spread_rate: float) -> dict[str, float]:
        """Collect spread contributions from all active concepts.

        This avoids order-dependence (all spreading happens
        "simultaneously" based on the pre-tick state).
        """
        spread_delta: dict[str, float] = {}
        SPREAD_THRESHOLD = 0.08  # concepts below this don't spread

        for cid, concept in list(self._concepts.items()):
            if concept.activation < SPREAD_THRESHOLD:
                continue
            edges = self._edge_index.get(cid, [])
            if not edges:
                continue
            total_weight = sum(e.weight for e in edges)
            if total_weight < 1e-8:
                continue

            # How much activation this concept spreads in total
            total_spread = concept.activation * spread_rate
            # Category/hub/modality multipliers
            cat_mult, hub_mult, mod_mult = self._get_concept_multipliers(
                cid,
                {ConceptCategory.LIVING: 1.15, ConceptCategory.ABSTRACT: 1.2},
                1.3,
                {ConceptModality.VISUAL: 1.1},
            )
            total_spread *= cat_mult * hub_mult * mod_mult

            current_cols = self.get_columns_for(cid)
            for edge in edges:
                # Fan-out normalization + bridge attenuation
                share = (edge.weight / total_weight) * total_spread
                target_cols = self.get_columns_for(edge.target)
                if not current_cols.intersection(target_cols):
                    share *= 0.3  # bridge attenuation
                if share > 1e-6:
                    spread_delta[edge.target] = (
                        spread_delta.get(edge.target, 0.0) + share
                    )

        return spread_delta
    def _apply_decay_noise_spread(
        self,
        decay_rate: float,
        noise_amp: float,
        spread_delta: dict[str, float],
    ) -> None:
        """Apply decay, spread, and noise to all concepts in-place."""
        HUB_DECAY_MULTIPLIER = 0.7
        MOTOR_DECAY_MULTIPLIER = 1.3

        for cid, concept in list(self._concepts.items()):
            # Guard against None activation from corrupted/old state
            if concept.activation is None:
                concept.activation = 0.0
            # Decay
            effective_decay = decay_rate
            if concept.is_semantic_hub:
                effective_decay *= HUB_DECAY_MULTIPLIER
            if concept.modality == ConceptModality.MOTOR:
                effective_decay *= MOTOR_DECAY_MULTIPLIER
            concept.activation *= (1.0 - effective_decay)

            # Apply spread
            if cid in spread_delta:
                concept.activation = min(1.0, concept.activation + spread_delta[cid])

            # Noise (Gaussian-like via uniform sum)
            n = (random.random() + random.random() - 1.0) * noise_amp
            concept.activation = max(0.0, min(1.0, concept.activation + n))
    def _apply_ach_focus(self, ach_focus: float) -> None:
        """Apply ACh winner-take-all dynamics: boost top concepts, suppress rest."""
        if ach_focus <= 0.01:
            return
        # Boost top concepts, suppress the rest
        activated = [
            (cid, c.activation)
            for cid, c in self._concepts.items()
            if c.activation > 0.05
        ]
        if activated:
            activated.sort(key=lambda x: x[1], reverse=True)
            # Top 20% get boosted
            top_n = max(1, len(activated) // 5)
            top_ids = {cid for cid, _ in activated[:top_n]}
            for cid, concept in list(self._concepts.items()):
                if concept.activation is None:
                    concept.activation = 0.0
                if cid in top_ids:
                    concept.activation = min(
                        1.0, concept.activation + ach_focus
                    )
                elif concept.activation > 0.05:
                    concept.activation *= (1.0 - ach_focus * 0.5)
    def _compute_effective_inhibition(
        self,
        column: str,
        context_columns: set[str] | None,
        inhibition_strength: float,
    ) -> float:
        """Compute the effective inhibition strength for a column.

        Context disinhibition: columns in context_columns receive
        reduced inhibition (the VIP interneuron signal).
        """
        if context_columns and column in context_columns:
            return inhibition_strength * 0.2  # 80% reduction
        else:
            return inhibition_strength
    def _inhibit_column_concepts(
        self, column: str, inhib_factor: float, dominant_column: str
    ) -> None:
        """Suppress concepts in a column by the multiplicative inhibition factor.

        Uses the column index for O(k) instead of O(n) scan.
        Concepts that also belong to the dominant column are skipped.
        """
        column_cids = self._column_index.get(column, set())
        for cid in list(column_cids):
            concept = self._concepts.get(cid)
            if concept is None:
                continue
            cols = concept.columns or {_column_of(concept.origin, cid)}
            if dominant_column in cols:
                continue
            if concept.activation is None:
                concept.activation = 0.0
            concept.activation *= inhib_factor
    def apply_lateral_inhibition(
        self,
        context_columns: set[str] | None = None,
        inhibition_strength: float = 0.4,
    ) -> dict[str, float]:
        """Apply lateral inhibition between cortical columns (SOM interneurons).

        When one column is highly active, other columns are suppressed.
        This is the mechanism that makes columns functionally separate —
        the dictionary column doesn't compete with the code column because
        the code column actively inhibits it.

        The inhibition is MULTIPLICATIVE, not additive. Each concept in a
        non-dominant column has its activation scaled by:
            (1 - inhibition_strength * dominance_ratio)
        where dominance_ratio = dominant_mean_activation / concept_activation.

        This is biologically grounded: inhibitory interneurons reduce
        firing rate proportionally, not by a fixed amount. A concept at
        activation 0.8 in a suppressed column loses more absolute
        activation than one at 0.1, but the relative reduction is the
        same — mirroring how inhibition scales with excitatory input.

        Context disinhibition (VIP): columns in context_columns receive
        reduced inhibition (multiplied by 0.2, an 80% reduction). This
        is the top-down attention signal — the prefrontal cortex sends
        a signal that disinhibits the relevant column, allowing it to
        win the competition even against a larger, more active column.

        Returns a dict of {column: mean_activation} for all columns
        (measured BEFORE inhibition is applied).
        """
        # Calculate mean column activation levels
        column_activations = self._compute_column_activations()

        if not column_activations:
            return {}

        # Find the dominant column (highest MEAN activation)
        dominant_column = max(column_activations, key=lambda k: column_activations[k])
        dominant_activation = column_activations[dominant_column]

        if dominant_activation < 1e-8:
            return column_activations  # nothing to inhibit with

        # Apply inhibition to non-dominant columns
        for column, _activation in column_activations.items():
            if column == dominant_column:
                continue

            effective_inhibition = self._compute_effective_inhibition(
                column, context_columns, inhibition_strength
            )

            # Inhibition ratio: how much stronger the dominant column is
            # relative to this column's mean. Clamped to [0, 1].
            col_mean = column_activations[column]
            if col_mean < 1e-8:
                continue  # nothing to inhibit
            dominance_ratio = min(1.0, dominant_activation / max(col_mean, 1e-8))

            # Multiplicative inhibition factor
            inhib_factor = 1.0 - effective_inhibition * dominance_ratio

            # Suppress concepts in this column (multiplicatively)
            self._inhibit_column_concepts(column, inhib_factor, dominant_column)

        return column_activations
    def _compute_column_activations(self) -> dict[str, float]:
        """Calculate mean activation level for each cortical column."""
        column_activations: dict[str, float] = {}
        for column, cids in self._column_index.items():
            total = 0.0
            count = 0
            for cid in list(cids):
                concept = self._concepts.get(cid)
                if concept:
                    total += concept.activation
                    count += 1
            if count > 0:
                column_activations[column] = total / count
        return column_activations
    def apply_winner_take_all(
        self,
        column: str,
        top_k: int = 3,
        suppression: float = 0.5,
    ) -> None:
        """Apply winner-take-all dynamics within a column (PV interneurons).

        Within a cortical column, the most activated concepts suppress
        less-activated competitors. This sharpens the response — instead
        of many weakly-activated concepts, you get a few strongly-activated
        ones. This is what makes a column's output coherent rather than
        diffuse.

        Distance-dependent suppression: instead of a flat suppression for
        all concepts below top_k, each concept is suppressed proportional
        to its distance from the winner's activation level. Concepts close
        to the winner are lightly suppressed; concepts far below are
        heavily suppressed. This mirrors how PV interneurons create a
        suppression gradient — the winner's inhibitory output scales with
        its own activation, so nearby competitors receive more inhibition
        than distant ones.

        Formula for concept at rank i (0-indexed, sorted by activation):
            if i < top_k: preserved at full activation
            if i >= top_k: activation *= (1 - suppression * gap_ratio)
            where gap_ratio = (winner_activation - concept_activation) / winner_activation

        gap_ratio ∈ [0, 1]: 0 means the concept is as active as the
        winner (minimal suppression), 1 means it's at zero (maximal
        suppression). This is the standard winner-take-all circuit from
        computational neuroscience (Douglas & Martin, 2004).
        """
        # Get all concepts in this column using the column index
        # (avoids O(N) scan of the entire concept network).
        column_cids = self._column_index.get(column, set())
        column_concepts = [
            (cid, self._concepts[cid].activation)
            for cid in column_cids
            if cid in self._concepts
        ]

        if not column_concepts:
            return

        # Find the top_k winners without sorting the entire column.
        winners = heapq.nlargest(top_k, column_concepts, key=lambda x: x[1])
        winner_activation = winners[0][1]
        if winner_activation < 1e-8:
            return  # nothing to compete with

        winner_ids = {cid for cid, _ in winners}

        # Suppress non-winners with distance-dependent scaling.
        # No need to sort — the suppression factor depends only on
        # the winner's activation and each concept's own activation.
        for cid, activation in column_concepts:
            if cid in winner_ids:
                continue
            gap_ratio = (winner_activation - activation) / winner_activation
            gap_ratio = max(0.0, min(1.0, gap_ratio))
            factor = 1.0 - suppression * gap_ratio
            concept = self._concepts.get(cid)
            if concept is not None:
                if concept.activation is None:
                    concept.activation = 0.0
                concept.activation *= factor
    def most_activated(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the N most activated concepts."""
        ranked = heapq.nlargest(n, list(self._concepts.items()), key=lambda x: x[1].activation)
        return [(cid, c.activation) for cid, c in ranked]
