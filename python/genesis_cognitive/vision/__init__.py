"""Occipital subsystem — the visual processing center of Genesis's brain.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The occipital subsystem is the primary visual processing center of the
brain. It receives input from the retina (via the LGN) and contains
the earliest stages of the cortical visual hierarchy. In Genesis, the
occipital subsystem implements the full ventral ("what") stream pipeline,
from raw pixels to concept recognition.

The ventral visual stream is a hierarchical predictive coding system.
Each level predicts the activity of the level below; prediction errors
drive both recognition (bottom-up) and learning (top-down). This
bidirectional flow is the core computational principle.

Pipeline:

    retina frame (H x W x 3, uint8)
        |
        v
    [V1] Primary visual cortex (Brodmann area 17)
        |   Sparse coding with Gabor dictionary
        |   Recurrent ISTA inference
        |   Plastic horizontal connections (Hebbian)
        |   Contour grouping (union-find, co-circular)
        |   Produces: Z1 (n_patches x n_v1_features) + VisualField
        v
    [V4] Mid-level visual cortex (Brodmann area 19)
        |   Pools V1 latents into larger receptive fields
        |   Adds chromatic channels (V4 color-selective neurons)
        |   Learns curvature/texture/shape dictionary via sparse coding
        |   Produces: Z4 (n_v4_units x n_v4_features)
        v
    [VTC] Ventral temporal cortex (anatomically temporal subsystem)
        |   PCA of V4 activations -> axis-based feature space
        |   Incremental PCA with EMA forgetting
        |   Produces: vtc_vector (n_vtc_dims) -- global visual descriptor
        v
    [MTL bridge] Medial temporal subsystem (anatomically temporal subsystem)
        |   Ridge-regression projection: VTC -> concept embedding space
        |   One-shot learning from natural experience
        |   Produces: recognized concept + confidence
        v
    [Concept network] -- nearest-neighbor lookup

    -- Predictive loop (bidirectional) --
    Top-down: concept -> W^T -> vtc_prediction -> V4 dictionary^T -> V1 prediction
    Bottom-up: image -> V1 -> V4 -> VTC -> concept
    Error at each level drives learning.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

V1 — Primary Visual Cortex (sparse coding)
------------------------------------------

The V1 model is based on the 2026 sparse-coding account of cortical
inference (Yun, Belsten, Olshausen et al., arXiv:2607.15693). The
dictionary Phi is a bank of Gabor wavelets at multiple orientations,
scales, and phases. Inference is recurrent ISTA with lateral
interactions.

Dictionary construction (Gabor filter bank):

    For each orientation theta_k, spatial frequency f_j, phase phi_l:

        g(x, y; theta, f, phi) = exp(-(x^2 + y^2) / (2*sigma^2))
                                  * cos(2*pi * x_r / f + phi)

    where x_r = x*cos(theta) + y*sin(theta)

    Each filter is vectorized and L2-normalized to form a column of
    Phi. Shape: (patch_size^2, n_features) where
    n_features = orientations * scales * phases.

Inference (recurrent ISTA with lateral interactions):

    z_{t+1} = ReLU( z_t + eta * [ Phi^T * x
                                  -  (Phi^T*Phi + sigma^2 * gamma * M) * z_t
                                  -  gamma * lambda ] )

    where:
        x       = whitened image patch (mean-subtracted, std-normalized)
        Phi     = Gabor dictionary
        Phi^T*Phi = data-likelihood Gram matrix (precomputed)
        M       = horizontal interaction matrix (plastic, co-circular prior)
        sigma^2 = 1 (clean inference)
        gamma   = recurrent_scale (lateral interaction strength)
        lambda  = sparsity (L1 weight)
        eta     = ISTA step size

    The ReLU enforces non-negative firing rates. Damping (0.7*z + 0.3*z_new)
    prevents oscillation. Convergence in ~40 iterations.

Horizontal connection plasticity (Hebbian / Oja-like):

    cov = (active^T * active) / n_patches     (co-activation matrix)
    delta_M = lr * (cov - 0.05 * M)           (Oja-like decay)
    M += delta_M, clipped to [-0.5, 0.5]

    The prior M is structured: features at the same scale with similar
    orientations (co-circular, within pi/n_orientations) facilitate each
    other. Same-phase pairs get a 1.2x boost. The diagonal is zero (no
    self-excitation). Experience refines this prior online.

Contour grouping:

    Co-active features in adjacent patches with compatible orientations
    are linked via union-find. Two features are connected if:
        - |drow| <= 1 and |dcol| <= 1 (spatially adjacent)
        - angle_difference(theta_i, theta_j) < pi / n_orientations

    Groups of >= 2 features become contours. Contours are sorted by
    mean feature strength (salience).

V1 gamma power:

    gamma = clip( mean(active) + std(active) * 0.3, 0, 1 )

    where active = z[z > 0]. This is the binding signal sent to the
    brain wave system. Stronger, more coherent sparse activity produces
    higher gamma, reflecting the neural signature of feature binding.


V4 — Mid-level Visual Cortex (sparse coding on V1 latents)
-----------------------------------------------------------

V4 pools V1 latents into larger receptive fields and learns a
dictionary of mid-level features (curvature, texture, shape complexes,
color blobs) via unsupervised sparse coding.

Pooling:

    V1 latents are reshaped into a grid and grouped pool_size x
    pool_size. Each group's latent vectors are concatenated, and
    color statistics (mean RGB, std RGB, HSV stats) are appended.

    input_dim = n_v1_features * pool_size^2 + color_dims

Inference (ISTA, same form as V1 but without lateral interactions):

    z_{t+1} = ReLU( z_t + eta * [ Phi4^T * x  -  (Phi4^T * Phi4) * z_t  -  lambda ] )

Dictionary learning (gradient descent on reconstruction error):

    recon = Z * Phi4^T
    error = X - recon
    grad = error^T * Z
    Phi4 += lr * grad
    Phi4 = normalize_columns(Phi4)
    Phi4^T*Phi4 recomputed every 50 samples


VTC — Ventral Temporal Cortex (incremental PCA)
------------------------------------------------

PCA of V4 activations creates a low-dimensional feature space whose
axes capture the principal variations in mid-level visual features.
This is the "neural feature space" described in the 2026 Nature
Communications paper on VTC->MTL coding.

Global descriptor:

    descriptor = mean(Z4, axis=0)    (average V4 activation across all units)

Incremental PCA with EMA forgetting:

    mean = alpha * mean + (1 - alpha) * descriptor
    scatter = alpha * scatter + outer(delta, delta)
    where delta = descriptor - mean, alpha = forgetting_factor

    Every 10 samples:
        eigenvalues, eigenvectors = eigh(scatter)
        components = top-k eigenvectors (sorted descending)

Projection:

    vtc = components @ (descriptor - mean)
    vtc = vtc / ||vtc||    (L2 normalize)


MTL Bridge — VTC to Concept Embedding Projection (ridge regression)
---------------------------------------------------------------------

The MTL bridge maps VTC feature vectors to concept embedding space
via a learned linear projection W. Learning is one-shot: collect
(vtc_vector, concept_embedding) pairs and solve ridge regression.

Ridge regression:

    W = E^T * V * (V^T * V + lambda * I)^{-1}

    where:
        E = matrix of concept embeddings (n x embedding_dim)
        V = matrix of VTC vectors (n x vtc_dim)
        W = projection matrix (embedding_dim x vtc_dim)
        lambda = ridge regularization parameter

Recognition (bottom-up):

    predicted = W @ vtc_vector
    concept = argmax_k  cosine_sim(predicted, embedding_k)

Generative path (top-down, for imagination):

    vtc_predicted = W^+ @ concept_embedding
    where W^+ = pinv(W)  (Moore-Penrose pseudo-inverse)


Predictive Coding Errors
-------------------------

At each level, the prediction error drives both recognition and
learning. The top-down prediction from the level above is compared to
the actual bottom-up signal.

    v1_error = ||Z1 - Z1_predicted|| / n_patches
    v4_error = ||Z4 - Z4_predicted|| / n_v4_units
    vtc_error = ||vtc - vtc_predicted||

    surprise = v1_error + v4_error + vtc_error
    is_novel = surprise > 0.5 AND confidence < 0.3

When a concept is recognized with confidence > 0.3, the top-down
generative path updates predictions at each level for the next frame.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The occipital subsystem generates and is modulated by three primary brain
wave bands. These are not abstract labels — they correspond to
measurable oscillatory dynamics in the visual cortex, and Genesis's
brain wave system (brain_waves.py) reads the V1 gamma signal produced
here.

Alpha (8-12 Hz) — The resting rhythm of visual cortex
-------------------------------------------------------

Alpha is the dominant oscillation in the occipital subsystem during
relaxed wakefulness with eyes closed. When visual attention is
engaged, alpha power decreases (desynchronizes) at the retinotopic
location of the stimulus — this is "alpha suppression" and it is
spatially tuned (eLife, 2023).

Alpha oscillations are a spectral "fingerprint" of hierarchical
predictive coding. Under neural delays, predictive coding networks
naturally generate oscillations in the alpha range (J Neurosci,
2023). Alpha power carries stimulus-related information in a
temporally predictive fashion — future position representations are
activated even without direct visual input.

In Genesis: alpha is the baseline state when V1 is not actively
processing. High alpha = idling/resting visual cortex. Alpha
suppression marks the transition to active visual processing. The
brain wave system tracks this as the occipital alpha component.

Gamma (30-100 Hz) — The active processing rhythm
---------------------------------------------------

Gamma is the signature of active visual computation. Gamma power
increases with stimulus-driven activity in V1, driven by the
coherence and strength of sparse-coding activity (Springer, 2021).
Gamma is associated with feature binding — the synchronous firing
that binds together the features of a single object into a coherent
percept.

In Genesis: V1 gamma power is computed directly from sparse-coding
activity:

    gamma = clip(mean(active) + std(active) * 0.3, 0, 1)

This signal is sent to the brain wave system via the gamma callback
(mind.py: _on_v1_gamma), which boosts acetylcholine — the
neuromodulator of attention. The brain wave system then assesses
gamma synchrony as part of its overall cortical state assessment.

Beta (13-30 Hz) — The dorsal stream rhythm
--------------------------------------------

Beta oscillations in the occipital subsystem are associated with the
dorsal ("where") visual pathway — the magnocellular-dominated stream
that projects to the parietal subsystem. Beta supports spatial
reorganization of visual inputs, form-motion integration, and
top-down influences on the slower ventral stream (Frontiers in
Psychology, 2023).

In Genesis: beta is associated with the dorsal stream output of V1
— the motion and spatial structure signals that feed into the
parietal subsystem (not yet implemented as a separate module). The
occipital subsystem's contour grouping and spatial layout analysis are
the substrate for beta-band activity.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

Strictly, the occipital subsystem contains V1, V2, and V3. V4 sits at the
occipital-temporal boundary. VTC and the MTL bridge are anatomically
in the temporal subsystem. However, the ventral visual stream is a
continuous functional pipeline that originates in V1, and splitting
it across folders would break the computational unity of the
predictive coding hierarchy.

This folder therefore contains the full ventral stream pipeline
(V1 -> V4 -> VTC -> MTL bridge), with the understanding that VTC and
MTL are anatomically temporal subsystem structures that are functionally
part of the occipital subsystem's processing stream. When the temporal
subsystem folder is created, it will reference these modules as
projections FROM the occipital subsystem.

The dorsal stream (V1 -> MT/V5 -> parietal) will be organized under
the parietal subsystem folder when created.


════════════════════════════════════════════════════════════════════════
MODULE ORGANIZATION
════════════════════════════════════════════════════════════════════════

    v1.py              Primary visual cortex — sparse coding, Gabor
                       dictionary, ISTA inference, plastic horizontal
                       connections, contour grouping, gamma power.

    v4.py              Mid-level visual cortex — pools V1 latents,
                       adds color, learns curvature/texture/shape
                       dictionary via sparse coding.

    vtc.py             Ventral temporal cortex — incremental PCA of
                       V4 activations, axis-based feature space.

    memory_bridge.py      MTL bridge — ridge-regression projection from
                       VTC space to concept embeddings, one-shot
                       learning, generative path for imagination.

    visual_cortex.py   Full hierarchical predictive coding system —
                       orchestrates V1 -> V4 -> VTC -> MTL, computes
                       prediction errors at each level, produces
                       VisualPercept.

    image_utils.py     Image loading and resizing utilities.

Top-level modules referenced by this subsystem:

    (perception/) retina.py
        Retina — zero-copy access to the Rust retina shared-memory
        camera feed. This is the eye: the retina captures photons
        (camera frames) and passes them to V1 for processing.
        Anatomically the retina is not part of the occipital subsystem
        (it's in the eye), but functionally it's the input to the
        visual pathway. The LGN (lateral geniculate nucleus) relay
        is not separately modeled — the retina feeds V1 directly.

    (perception/) vision.py
        Vision — the bridge between the raw camera feed (retina.py)
        and Genesis's cognitive experience. Produces rich, human-like
        scene descriptions combining color, objects, faces, lighting,
        and spatial layout. This is the visual association cortex —
        the occipital-temporal-parietal integration that produces
        the unified visual percept. Anatomically spans occipital,
        temporal (object recognition), and parietal (spatial layout),
        but the primary processing is occipital.

    (top-level) perception/
        IntegratedPerception, MultisensoryIntegrator — multi-modal
        perception integration. Not purely occipital (includes
        auditory and other modalities), but visual perception is
        the dominant input. FaceRecognizer (perception/recognition.py)
        is the fusiform face area (occipital-temporal).

    (top-level) brain_waves.py
        Brain wave system — tracks alpha (occipital's dominant
        rhythm), gamma (active visual processing), beta (dorsal
        stream). Referenced by every subsystem.
"""

from __future__ import annotations

from .._views import view_getattr
from .image_utils import decode_image_bytes, load_image, resize_for_vision
from .memory_bridge import MemoryBridge, TrainingExample
from .v1 import V1Model, VisualContour, VisualFeature, VisualField
from .v4 import V4Model
from .visual_cortex import VisualCortex, VisualPercept
from .vtc import VTCFeatureSpace

# The retina (the eye's sensor sheet) and the visual association
# cortex (unified scene perception). These are lazily re-exported:
# vision.py does `from .vision import V1Model` and
# `from .auditory import ...`, so eagerly importing it here
# creates a cycle. PEP 562 resolution means `Retina` and `Vision`
# are bound on first attribute access, after package init.
_EXPORTS: dict[str, str] = {
    "Retina": "perception.retina",
    "Vision": "perception.vision",
}

__all__ = [
    "MemoryBridge",
    "Retina",
    "TrainingExample",
    "V1Model",
    "V4Model",
    "VTCFeatureSpace",
    "Vision",
    "VisualContour",
    "VisualCortex",
    "VisualFeature",
    "VisualField",
    "VisualPercept",
    "decode_image_bytes",
    "load_image",
    "resize_for_vision",
]

__getattr__ = view_getattr(_EXPORTS, __name__)
