"""Structural form perception — scene organization as a coupled variational flow.

This is the layer above ``scene.py``. ``perceive`` binds pixels into
objects; ``structure`` binds objects into *organization* — the answer
to "what kind of thing arranges this scene?" A board, a sequence, a
cycle, a hierarchy, a cloud — each is a different structural *form*,
and discovering which one a scene instantiates is itself the
computation (Kemp & Tenenbaum, *The Discovery of Structural Form*,
PNAS 2008).

Biologically this is the parietal / entorhinal–hippocampal function:
the HEC system encodes abstract relational maps with the same grid-like
code it uses for space — structure perceived, not assumed. And like the
cortex, this module is **reentrant, not a pipeline**: grouping, form,
and constraint are co-evolving beliefs in one joint probabilistic
model. There is no stage order — each sweep, every variable is updated
conditioned on the others' current posterior, and the interpretation
is the attractor the flow settles into. An initial spectral pass seeds
the settle the way the first feedforward sweep seeds recurrent visual
processing; after that, everything flows.

Model — a mean-field free-energy minimization over:

- ``z_i``  soft group assignment of element i  (q_i(k))
- ``F``    structural form ∈ {cluster, chain, ring, tree, lattice}  (q(F))
- ``θ``    constraint evidence — Dirichlet-multinomial posteriors over
           cell values and adjacency relations, persistent across scenes

Each relaxation sweep applies the coordinate updates:

    W_ij   local-scaling affinity (Zelnik-Manor & Perona 2004):
           W_ij = exp(−d_ij² / (σ_i σ_j)) · κ(F_i, F_j)
           with σ_i the distance to element i's k-th nearest neighbour —
           no global bandwidth, scale is read off the scene itself —
           and κ a feature kernel (colour/shape agreement).

    Ã_ij   expected linkage: Σ_k q_i(k) q_j(k) (W_ij + λ Φ̄_ij)
           where Φ̄_ij = Σ_F q(F) Φ^F_ij is the edge probability the
           current form posterior predicts. Form feeds grouping the
           same way grouping feeds form.

    q(F)   ∝ exp(−½ BIC_F)   — a proper model posterior, not a vote:
           BIC_F = −2 log ∏_p Bern(a_p | θ̂_p^F) + dof_F log M
           over all element pairs p, with a_p the observed adjacency
           (symmetrised k-nearest linkage) and θ̂^F Beta-smoothed.

    q(z_i) ∝ exp( β_t Σ_j [W_ij + λ Φ̄_ij] q_j )   — a Potts mean-field
           step under deterministic annealing: β_t rises each sweep,
           so assignments flow from soft to committed, never clamped.

    θ      conjugate Dirichlet updates — counts + α. No evidence
           thresholds: the posterior's own concentration *is* the
           confidence. (What the old heuristic called "three agreeing
           observations" is here a credible interval.)

Convergence: the free energy falls monotonically — every update is a
coordinate descent — and a sweep cap bounds the settle. What remains
is not the output of a stage; it is the state the scene relaxes into.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.linalg import eigh

from .scene import PerceivedObject, Scene

#: k for the local-scaling σ_i (Zelnik-Manor & Perona use 7; capped by n−1).
_SCALE_K = 7
#: Degree bound of the observed adjacency graph — a lattice node has at
#: most four structural neighbours, so four is the natural linkage degree.
_NEIGHBOR_K = 4
#: Linkage gate in units of first-neighbour distance. On a square lattice
#: orthogonal and diagonal neighbours sit at ratio √2 ≈ 1.414 — the gate
#: must live strictly inside (1, √2) to link the 4-neighbourhood and no
#: further; 1.3 sits in that gap.
_DIAG_GAP = 1.3
#: Annealing schedule: β_t = β_0 + (β_1 − β_0) t/T. Soft at first — the
#: flow can rearrange — committed by the last sweeps.
_BETA_0 = 2.0
_BETA_1 = 24.0
#: Settle bound and the free-energy stall that ends it early.
_MAX_SWEEPS = 12
_SETTLE_TOL = 1e-4
#: Form-evidence mixing: λ on Φ̄ in the linkage energy — form is a prior
#: on linkage, weighted below the sensory affinity it explains.
_FORM_LAMBDA = 0.5
#: Dirichlet prior strength for learned constraints (Jeffreys-flavoured,
#: weak — the scene's own evidence should dominate quickly).
_DIR_ALPHA = 0.5
#: Eigen-spectrum scan bound for the group-count eigengap.
_GAP_SCAN = 6
#: A group of this many or fewer elements cannot instantiate a lattice.
_MIN_LATTICE = 4


class StructuralForm(Enum):
    """The generative forms the posterior ranges over."""

    CLUSTER = "cluster"  # undifferentiated group — all-pairs linkage
    CHAIN = "chain"  # linear order — each node links its successor
    RING = "ring"  # closed chain
    TREE = "tree"  # spanning hierarchy — MST linkage
    LATTICE = "lattice"  # rank-2 periodic array — a board

    @property
    def dof(self) -> int:
        """Free parameters of the form's embedding — the BIC penalty term."""
        return {
            StructuralForm.CLUSTER: 1,
            StructuralForm.CHAIN: 2,
            StructuralForm.RING: 2,
            StructuralForm.TREE: 2,
            StructuralForm.LATTICE: 3,
        }[self]


@dataclass
class LatticeFit:
    """A rank-2 index lattice fitted to element centroids.

    ``row_lines``/``col_lines`` are the line coordinates (scene space);
    ``pos`` maps element index → (row, col) lattice index. ``fill`` is
    the fraction of the bounding R×C lattice the elements actually
    occupy — a sparse fit is a weak board.
    """

    row_lines: list[float]
    col_lines: list[float]
    pos: dict[int, tuple[int, int]]
    pitch: tuple[float, float]
    fill: float
    rms_snap: float

    @property
    def signature(self) -> tuple[int, int, int, int]:
        """Identity for evidence accumulation — same arrangement, same θ."""
        r, c = self.pitch
        return (len(self.row_lines), len(self.col_lines), round(r), round(c))


@dataclass
class Violation:
    """An element whose observed state contradicts the learned constraint.

    ``surprise`` is −log P(observed | evidence); ``predicted`` the MAP
    state; ``confidence`` the posterior mass on that prediction.
    """

    element: int
    position: tuple[int, int]
    observed: int
    predicted: int
    surprise: float
    confidence: float


@dataclass
class GroupStructure:
    """One settled group: its form posterior and fitted structure."""

    members: list[int]
    form: StructuralForm
    form_posterior: dict[StructuralForm, float]
    lattice: LatticeFit | None = None


@dataclass
class SceneStructure:
    """The interpretation a scene relaxes into — the flow's attractor."""

    groups: list[GroupStructure]
    violations: list[Violation]
    sweeps: int
    free_energy: float


def _local_sigma(dists: np.ndarray, i: int, n: int) -> float:
    """Zelnik-Manor & Perona local scale: distance to the k-th neighbour."""
    k = min(_SCALE_K, n - 1)
    if k <= 0:
        return 1.0
    s = float(np.partition(dists[i], k)[k])
    return max(s, 1e-6)


def affinity(objects: list[PerceivedObject]) -> np.ndarray:
    """Pairwise affinity W over scene elements — the sensory evidence field.

    W_ij = exp(−d_ij²/(σ_i σ_j)) · κ(i,j): spatial proximity on each
    element's own scale, multiplied by a feature kernel κ ∈ [1/3, 1]
    averaging colour and shape agreement. W is computed once — it is
    the *data* the flow conditions on, not a stage.
    """
    n = len(objects)
    if n == 0:
        return np.zeros((0, 0))
    cent = np.array([o.centroid for o in objects], dtype=float)
    d2 = np.sum((cent[:, None, :] - cent[None, :, :]) ** 2, axis=-1)
    dists = np.sqrt(d2)
    W = np.zeros((n, n))
    for i in range(n):
        si = _local_sigma(dists, i, n)
        for j in range(i + 1, n):
            sj = _local_sigma(dists, j, n)
            kappa = (
                1.0
                + float(objects[i].color == objects[j].color)
                + float(objects[i].shape_signature == objects[j].shape_signature)
            ) / 3.0
            W[i, j] = W[j, i] = math.exp(-d2[i, j] / (si * sj)) * kappa
    return W


def _spectral_seed(objects: list[PerceivedObject], W: np.ndarray, max_k: int) -> np.ndarray:
    """The feedforward pass: Ng–Jordan–Weiss embedding → hard seed groups.

    L_sym = I − D^{−1/2} W D^{−1/2}; the k smallest eigenvectors embed
    elements so that cutting the embedding approximates min Ncut. k is
    chosen by the eigengap — the scene says how many groups it holds.
    """
    n = W.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    # Zero-coupling is a hard boundary — elements with no perceptual path
    # between them can never belong to one organization, no matter what
    # the flow does later. The boundary is spatial: a small cell-gap
    # (scaled to element size, the way pixel-near tiles read as one
    # board) links elements; scene-distant elements split first.
    components = _linked_components(objects)
    labels = np.zeros(n, dtype=int)
    offset = 0
    for comp in components:
        if len(comp) < 2:
            for i in comp:
                labels[i] = offset
            offset += 1
            continue
        sub = W[np.ix_(comp, comp)]
        labels[np.asarray(comp)] = _eigengap_labels(sub) + offset
        offset += int(labels[np.asarray(comp)].max()) + 1
    return labels


def _cell_gap(a: PerceivedObject, b: PerceivedObject) -> int:
    """Chebyshev gap between two cell sets — 0 touching, 1 one cell apart."""
    ar = {r for r, _ in a.cells}
    ac = {c for _, c in a.cells}
    br = {r for r, _ in b.cells}
    bc = {c for _, c in b.cells}
    dr = max(0, max(min(ar) - max(br), min(br) - max(ar)) - 1)
    dc = max(0, max(min(ac) - max(bc), min(bc) - max(ac)) - 1)
    return max(dr, dc)


def _linked_components(objects: list[PerceivedObject]) -> list[list[int]]:
    """Perceptual linkage components: elements within a small cell-gap —
    max(2, √size) — of each other cohere into one organization; gaps an
    order larger are separate scenes' worth of space between them.
    """
    n = len(objects)
    seen = [False] * n
    comps: list[list[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in range(n):
                if seen[j]:
                    continue
                margin = max(2.0, math.sqrt(min(objects[i].size, objects[j].size)))
                if _cell_gap(objects[i], objects[j]) <= margin:
                    seen[j] = True
                    stack.append(j)
        comps.append(comp)
    return comps


def _eigengap_labels(W: np.ndarray, max_k: int = _GAP_SCAN) -> np.ndarray:
    """Eigengap group count + NJW k-means on one connected component."""
    n = W.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    d = W.sum(axis=1)
    d = np.where(d <= 0, 1e-12, d)
    dm = np.diag(d**-0.5)
    L = np.eye(n) - dm @ W @ dm
    kmax = max(2, min(max_k + 1, n))
    vals, vecs = eigh(L, subset_by_index=[0, kmax - 1])
    gaps = np.diff(vals)
    if gaps.size == 0:
        return np.zeros(n, dtype=int)
    # k groups = eigenvalues before the largest gap (von Luxburg's
    # eigengap heuristic); k=1 stays reachable — one coherent scene.
    k = int(np.argmax(gaps) + 1)
    k = max(1, min(k, n))
    X = vecs[:, :k]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms <= 0, 1.0, norms)
    return _kmeans(X, k)


def _kmeans(X: np.ndarray, k: int, iters: int = 60) -> np.ndarray:
    """Lloyd's algorithm with deterministic furthest-point seeding."""
    n = X.shape[0]
    if k >= n:
        return np.arange(n)
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[0]
    dist = np.sum((X - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        centroids[c] = X[int(np.argmax(dist))]
        dist = np.minimum(dist, np.sum((X - centroids[c]) ** 2, axis=1))
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmin(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for c in range(k):
            members = X[assign == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return assign


def _observed_adjacency(objects: list[PerceivedObject], W: np.ndarray) -> np.ndarray:
    """Observed linkage a_ij ∈ {0,1}: symmetrised top-k affinity edges,
    gated by first-neighbour distance — links stay within _DIAG_GAP ×
    d_nearest, the geometric gap separating the 4-neighbourhood from the
    diagonals beyond it (√2 ≈ 1.414 on a square lattice).

    The rank gate alone degenerates on small scenes (k ≥ n−1 links
    everything); the scale gate is what lets a line's endpoints be
    non-adjacent — which is exactly what distinguishes a chain from a
    clique.
    """
    n = W.shape[0]
    a = np.zeros((n, n))
    if n < 2:
        return a
    cent = np.array([o.centroid for o in objects], dtype=float)
    dists = np.sqrt(np.sum((cent[:, None, :] - cent[None, :, :]) ** 2, axis=-1))
    k = min(_NEIGHBOR_K, n - 1)
    for i in range(n):
        nearest = np.partition(np.delete(dists[i], i), 0)[0]
        gate = _DIAG_GAP * max(nearest, 1e-6)
        top = np.argsort(W[i])[::-1]
        for j in [j for j in top if j != i][:k]:
            if dists[i, j] <= gate:
                a[i, j] = 1.0
    return np.maximum(a, a.T)


def _fit_lines(coords: np.ndarray) -> tuple[list[float], float] | None:
    """Cluster 1-D coordinates into periodic lines; returns lines + pitch.

    The pitch is read off the scene: the median gap between consecutive
    occupied coordinates. Lines are agglomerated where the gap stays
    under half a pitch — the same way a row of tiles reads as one row.
    """
    u = np.unique(np.round(coords, 1))
    if u.size < 2:
        return None
    gaps = np.diff(u)
    pitch = float(np.median(gaps))
    if pitch <= 0:
        return None
    tol = max(1.5, 0.5 * pitch)
    lines: list[list[float]] = [[float(u[0])]]
    for v in u[1:]:
        if v - lines[-1][-1] <= tol:
            lines[-1].append(float(v))
        else:
            lines.append([float(v)])
    return [float(np.mean(g)) for g in lines], pitch


def fit_lattice(objects: list[PerceivedObject], members: list[int]) -> LatticeFit | None:
    """Fit a rank-2 axis-aligned lattice to a group's centroids.

    Elements snap to the nearest row/column line; the fit is real only
    if it has ≥2 lines per axis, ≥4 members, snap residual under a
    quarter-pitch, and the occupied fraction of the index lattice above
    half — a handful of colinear points is a chain, not a board.
    """
    if len(members) < _MIN_LATTICE:
        return None
    cent = np.array([objects[i].centroid for i in members])
    rows = _fit_lines(cent[:, 0])
    cols = _fit_lines(cent[:, 1])
    if rows is None or cols is None:
        return None
    row_lines, pitch_r = rows
    col_lines, pitch_c = cols
    if len(row_lines) < 2 or len(col_lines) < 2:
        return None
    pos: dict[int, tuple[int, int]] = {}
    err = 0.0
    for m_i, e_i in enumerate(members):
        ri = int(np.argmin([abs(cent[m_i, 0] - rl) for rl in row_lines]))
        ci = int(np.argmin([abs(cent[m_i, 1] - cl) for cl in col_lines]))
        err += (cent[m_i, 0] - row_lines[ri]) ** 2 + (cent[m_i, 1] - col_lines[ci]) ** 2
        pos[e_i] = (ri, ci)
    rms = math.sqrt(err / len(members))
    if rms > 0.25 * min(pitch_r, pitch_c):
        return None
    occupancy = len(set(pos.values()))
    fill = occupancy / (len(row_lines) * len(col_lines))
    if fill < 0.5:
        return None
    return LatticeFit(
        row_lines=row_lines,
        col_lines=col_lines,
        pos=pos,
        pitch=(pitch_r, pitch_c),
        fill=fill,
        rms_snap=rms,
    )


def _form_edges(
    form: StructuralForm,
    objects: list[PerceivedObject],
    members: list[int],
    lattice: LatticeFit | None,
) -> dict[tuple[int, int], float]:
    """The edges a form predicts over a group — Φ^F as probabilities."""
    n = len(members)
    edges: dict[tuple[int, int], float] = {}

    def put(i: int, j: int, p: float = 1.0) -> None:
        if i != j:
            edges[(i, j)] = p
            edges[(j, i)] = p

    if form is StructuralForm.CLUSTER:
        for a in range(n):
            for b in range(a + 1, n):
                put(members[a], members[b])
    elif form is StructuralForm.CHAIN or form is StructuralForm.RING:
        cent = np.array([objects[i].centroid for i in members])
        if form is StructuralForm.CHAIN:
            mu = cent.mean(axis=0)
            cov = np.cov((cent - mu).T)
            w, v = eigh(cov)
            axis = v[:, int(np.argmax(w))]
            order = np.argsort((cent - mu) @ axis)
        else:
            mu = cent.mean(axis=0)
            angles = np.arctan2(cent[:, 0] - mu[0], cent[:, 1] - mu[1])
            order = np.argsort(angles)
        for t in range(n - 1):
            put(members[int(order[t])], members[int(order[t + 1])])
        if form is StructuralForm.RING and n > 2:
            put(members[int(order[-1])], members[int(order[0])])
    elif form is StructuralForm.TREE:
        cent = np.array([objects[i].centroid for i in members])
        d2 = np.sum((cent[:, None, :] - cent[None, :, :]) ** 2, axis=-1)
        in_tree = {0}
        while len(in_tree) < n:
            best = min(
                ((d2[a, b], a, b) for a in in_tree for b in range(n) if b not in in_tree),
                key=lambda t: t[0],
            )
            _, a, b = best
            in_tree.add(b)
            put(members[a], members[b])
    elif form is StructuralForm.LATTICE and lattice is not None:
        by_pos = {p: e for e, p in lattice.pos.items()}
        for (ri, ci), e in by_pos.items():
            for dr, dc in ((1, 0), (0, 1)):
                other = by_pos.get((ri + dr, ci + dc))
                if other is not None:
                    put(e, other)
    return edges


def _form_logl(edges: dict[tuple[int, int], float], adj: np.ndarray) -> float:
    """log P(a | F): Beta-Bernoulli likelihood over all element pairs.

    Predicted-edge pairs and unpredicted pairs get separate smoothed
    Bernoulli parameters — the form earns likelihood only by correctly
    separating what links from what doesn't.
    """
    n = adj.shape[0]
    hit_p = miss_p = 0
    hit_n = miss_n = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred = edges.get((i, j), 0.0) > 0.5
            obs = adj[i, j] > 0.5
            if pred:
                hit_p += int(obs)
                hit_n += 1
            else:
                miss_p += int(obs)
                miss_n += 1
    p1 = (1 + hit_p) / (2 + hit_n)
    p0 = (1 + miss_p) / (2 + miss_n)
    return (
        hit_p * math.log(p1)
        + (hit_n - hit_p) * math.log(1 - p1)
        + miss_p * math.log(p0)
        + (miss_n - miss_p) * math.log(1 - p0)
    )


def _form_posterior(
    objects: list[PerceivedObject], members: list[int], adj: np.ndarray
) -> tuple[StructuralForm, dict[StructuralForm, float], LatticeFit | None]:
    """q(F) ∝ exp(−½ BIC_F) — posterior over structural forms.

    Each form's embedding is computed analytically (PCA order for the
    chain, angular order for the ring, MST for the tree, line-fit for
    the lattice, complete for the cluster), so no search is needed —
    the BIC selects, and the softmax of −½BIC is the posterior.
    """
    n = len(members)
    lattice = fit_lattice(objects, members)
    forms = [f for f in StructuralForm if f is not StructuralForm.LATTICE or lattice]
    if n < 3:
        forms = [StructuralForm.CLUSTER]
    m_pairs = max(1, n * (n - 1) // 2)
    bics: dict[StructuralForm, float] = {}
    for f in forms:
        edges = _form_edges(f, objects, members, lattice)
        bics[f] = -2.0 * _form_logl(edges, adj) + f.dof * math.log(m_pairs)
    best = min(bics, key=bics.get)  # type: ignore[arg-type]
    mx = max(bics.values())
    ws = {f: math.exp(-0.5 * (b - mx)) for f, b in bics.items()}
    z = sum(ws.values())
    return best, {f: w / z for f, w in ws.items()}, lattice


class StructurePerceiver:
    """Session-state structural perception — the flowing half of parietal.

    Each ``observe`` settles the current scene into a joint
    interpretation: soft group assignments, a form posterior, fitted
    structure, and constraint posteriors all co-evolving until the
    free energy stalls. Learned constraint evidence (θ) persists
    across scenes keyed by lattice signature — the same arrangement
    accrues evidence; a re-arranged scene starts fresh.
    """

    def __init__(self, sweeps: int = _MAX_SWEEPS) -> None:
        self.sweeps = sweeps
        self._value_counts: dict[tuple, Counter[int]] = {}
        self._lattice_sig: tuple[int, int, int, int] | None = None
        self.structure: SceneStructure | None = None

    def observe(self, scene: Scene) -> SceneStructure:
        """Settle a scene into its structural interpretation.

        One observe = one flow: the spectral pass seeds assignments,
        then sweeps co-update q(z), q(F), the lattice fit and the
        constraint evidence until assignments stop moving. Returns
        the settled ``SceneStructure`` — also held at ``.structure``.
        """
        objects = scene.objects
        n = len(objects)
        if n == 0:
            self.structure = SceneStructure(groups=[], violations=[], sweeps=0, free_energy=0.0)
            return self.structure

        W = affinity(objects)
        adj = _observed_adjacency(objects, W)
        z = _spectral_seed(objects, W, _GAP_SCAN)
        K = int(z.max()) + 1
        # Soft assignments — the mean-field state the flow carries.
        q = np.zeros((n, K))
        q[np.arange(n), z] = 1.0

        groups: list[list[int]] = []
        sweeps_run = 0
        prev_labels = z.copy()
        phibar = np.zeros((n, n))

        for t in range(self.sweeps):
            sweeps_run = t + 1
            beta = _BETA_0 + (_BETA_1 - _BETA_0) * t / max(1, self.sweeps - 1)
            groups = [[i for i in range(n) if int(q[i].argmax()) == k] for k in range(K)]
            groups = [g for g in groups if g]

            # Form beliefs conditioned on current grouping.
            phibar = np.zeros((n, n))
            for members in groups:
                slot = {e: s for s, e in enumerate(members)}
                sub = np.ix_(members, members)
                _best, post, lat = _form_posterior(objects, members, adj[sub])
                edge_mat = np.zeros((len(members), len(members)))
                for f, p in post.items():
                    e = _form_edges(f, objects, members, lat)
                    for (a, b), w in e.items():
                        if a in slot and b in slot:
                            edge_mat[slot[a], slot[b]] += p * w
                phibar[sub] = edge_mat

            # Mean-field Potts step: element i's group energy is its
            # weighted linkage plus the linkage the forms predict.
            energy = (W + _FORM_LAMBDA * phibar) @ q
            energy -= energy.max(axis=1, keepdims=True)
            new_q = np.exp(beta * energy)
            new_q /= new_q.sum(axis=1, keepdims=True)
            delta = float(np.abs(new_q - q).max())
            q = new_q
            labels = q.argmax(axis=1)
            if np.array_equal(labels, prev_labels) and delta < _SETTLE_TOL and t >= 2:
                break
            prev_labels = labels

        # Final settle state.
        groups = [[i for i in range(n) if int(q[i].argmax()) == k] for k in range(K)]
        groups = [g for g in groups if g]
        out_groups: list[GroupStructure] = []
        all_violations: list[Violation] = []
        for members in groups:
            sub = np.ix_(members, members)
            best, post, lat = _form_posterior(objects, members, adj[sub])
            out_groups.append(
                GroupStructure(members=members, form=best, form_posterior=post, lattice=lat)
            )
            if lat is not None:
                all_violations.extend(self._update_constraints(objects, members, lat))

        # Mean-field free energy at the settled state:
        # F = −E_q[linkage energy] + Σ_i Σ_k q_ik log q_ik
        link = float(((W + _FORM_LAMBDA * phibar) * (q @ q.T)).sum())
        entropy = float((q * np.log(np.maximum(q, 1e-12))).sum())
        self.structure = SceneStructure(
            groups=out_groups,
            violations=sorted(all_violations, key=lambda v: -v.surprise),
            sweeps=sweeps_run,
            free_energy=-link + entropy,
        )
        return self.structure

    def _update_constraints(
        self,
        objects: list[PerceivedObject],
        members: list[int],
        lattice: LatticeFit,
    ) -> list[Violation]:
        """Dirichlet evidence update + violation readout for one lattice.

        Per-position value model — a product of experts over three
        conditionally independent witnesses: the position's Dirichlet
        history (persistent across scenes), and the leave-one-out row
        and column value distributions (within-scene regularity, the
        element's own value excluded so it can't testify for itself):

            P(v | p) ∝ P_pos(v | p) · P_row^LOO(v) · P_col^LOO(v)

        A violation is an element whose MAP under that posterior is a
        different value — detected from the very first frame when the
        board's own regularity is the only evidence there is.
        """
        sig = lattice.signature
        if sig != self._lattice_sig:
            # A re-arranged scene: fresh evidence, keyed to the new
            # arrangement — old boards' beliefs don't leak in.
            self._value_counts = {}
            self._lattice_sig = sig

        pos_val: dict[tuple[int, int], int] = {}
        pos_elem: dict[tuple[int, int], int] = {}
        row_vals: dict[int, Counter[int]] = defaultdict(Counter)
        col_vals: dict[int, Counter[int]] = defaultdict(Counter)
        values: Counter[int] = Counter()
        for e in members:
            if e not in lattice.pos:
                continue
            p = lattice.pos[e]
            v = objects[e].color
            pos_val[p] = v
            pos_elem[p] = e
            row_vals[p[0]][v] += 1
            col_vals[p[1]][v] += 1
            values[v] += 1

        violations: list[Violation] = []
        for p, observed in pos_val.items():
            counts = self._value_counts.get(p, Counter())
            vocab = set(values) | set(counts)
            r, c = p
            total = sum(counts.values())
            # Leave-one-out denominators — the element's own value is
            # excluded from its row/column witnesses.
            row_n = sum(row_vals[r].values()) - 1
            col_n = sum(col_vals[c].values()) - 1
            nv = max(1, len(vocab))
            logw: dict[int, float] = {}
            for v in vocab:
                lp_pos = math.log((counts.get(v, 0) + _DIR_ALPHA) / (total + _DIR_ALPHA * nv))
                row_loo = row_vals[r].get(v, 0) - int(v == observed)
                col_loo = col_vals[c].get(v, 0) - int(v == observed)
                lp_row = math.log((row_loo + _DIR_ALPHA) / (row_n + _DIR_ALPHA * nv))
                lp_col = math.log((col_loo + _DIR_ALPHA) / (col_n + _DIR_ALPHA * nv))
                logw[v] = lp_pos + lp_row + lp_col
            mx = max(logw.values())
            ws = {v: math.exp(ll - mx) for v, ll in logw.items()}
            zsum = sum(ws.values())
            post = {v: w / zsum for v, w in ws.items()}
            predicted = max(post, key=post.get)  # type: ignore[arg-type]
            if predicted != observed and post[predicted] > 0.5:
                violations.append(
                    Violation(
                        element=pos_elem[p],
                        position=p,
                        observed=observed,
                        predicted=predicted,
                        surprise=-math.log(max(post.get(observed, 1e-12), 1e-12)),
                        confidence=post[predicted],
                    )
                )
            self._value_counts.setdefault(p, Counter())[observed] += 1
        return violations

    def violations(self) -> list[Violation]:
        """The settled scene's constraint violations, most surprising first."""
        return list(self.structure.violations) if self.structure else []

    def form(self) -> StructuralForm | None:
        """The dominant group's winning form, if any."""
        if not self.structure or not self.structure.groups:
            return None
        dominant = max(self.structure.groups, key=lambda g: len(g.members))
        return dominant.form


def discover_structure(scene: Scene) -> SceneStructure:
    """One-shot structural perception — a fresh settle, no session memory."""
    return StructurePerceiver().observe(scene)
