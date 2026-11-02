#!/usr/bin/env python3
"""null_core.py — shared machinery for the permuted-label RCS null (reviewer qp6c, W1).

WHAT THE NULL IS
----------------
Reviewer qp6c asks: RCS is nonnegative at every layer, so it has mass everywhere by
construction. What do Gini and 80%-coverage look like under PERMUTED harmful/harmless
labels, and how should a reader distinguish "distributed" from "nonnegative and flat"?

Under a random relabeling of the pooled N = n_F + n_L representations at a layer into
groups A (size n_F, "harmful") and B (size n_L, "harmless"), with pooled centroid m and
pooled scatter tr(S) = (1/N) sum_i ||x_i - m||^2:

    mu_A - mu_B = (N / n_L) (mu_A - m)                                      [exact]
    E ||mu_A - mu_B||^2 = N^2 tr(S) / (n_F n_L (N - 1))                     [SRSWOR]
    sigma_A = E_{i in A} ||x_i - mu_A||  ->  D := (1/N) sum_i ||x_i - m||   [A is random]

so the expected permuted score is

    E[RCS_perm] ~= (N / D) sqrt( tr(S) / (n_F n_L (N-1)) )
                 = sqrt(1/n_F + 1/n_L) * kappa * sqrt(N/(N-1)),
    kappa := sqrt(tr(S)) / D = (RMS radius)/(mean radius) >= 1, = 1 under concentration.

LEADING ORDER THE NULL IS A PURE FUNCTION OF THE CLASS SIZES AND IS LAYER-INDEPENDENT:
the per-layer scale tr(S(l)) cancels between numerator and denominator. The class sizes that
were ACTUALLY used are n_F = 260 and n_L = 6000 (the --source processed path; the paper's
stated 4000 SORRY-Bench prompts were not used -- see CRITICAL_data_source.md):

    sqrt(1/260 + 1/6000)  = 0.06335     <- correct
    sqrt(1/4000 + 1/6000) = 0.02041     <- what the paper's stated setup would imply

Therefore the null profile is FLAT up to sampling noise, which implies
    Gini_null ~= 0,  n80_null/L ~= 0.8,  peak/mean_null ~= 1,  CV_null ~= 0.

WHAT THE NULL DOES *NOT* TEST
-----------------------------
Permuting labels destroys the real class structure entirely, so it tests "is there any
harmful-vs-harmless signal at all", NOT "is that signal refusal-specific". Because the
two classes come from different corpora (SORRY-Bench vs Alpaca), topic/style/source
confounding is a SEPARATE question that a label permutation cannot settle. See
results/matched_controls/.
"""
import os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from lama_rcs import _gini_coefficient  # reuse the paper's exact Gini

EPS = 1e-8


# ---------------------------------------------------------------- profile statistics
def gini(v):
    """The paper's Gini, re-exported so the null and the published table cannot diverge."""
    return float(_gini_coefficient(np.asarray(v, dtype=float)))


def n80(v):
    """Layers needed to cover 80% of total RCS mass.

    Exact convention of lama_rcs.compute_rcs_statistics: sort descending, cumulative
    percentage, first index reaching 80, converted to a count.
    """
    a = np.asarray(v, dtype=float)
    if a.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(np.sort(a)[::-1]) / a.sum() * 100, 80) + 1)


def peak_over_mean(v):
    a = np.asarray(v, dtype=float)
    return float(a.max() / (a.mean() + EPS))


def cv(v):
    a = np.asarray(v, dtype=float)
    return float(a.std() / (a.mean() + EPS))


def profile_stats(v):
    a = np.asarray(v, dtype=float)
    L = len(a)
    return {
        "L": L,
        "mean_rcs": float(a.mean()),
        "peak_rcs": float(a.max()),
        "peak_over_mean": peak_over_mean(a),
        "gini": gini(a),
        "n80": n80(a),
        "n80_over_L": float(n80(a) / L),
        "cv": cv(a),
    }


# ---------------------------------------------------------------- Eq. (2)
def rcs_layer(hf, hl, eps=EPS):
    """RCS at one layer. hf: [n_f, d], hl: [n_l, d]. Exactly Eq. (2) of the paper."""
    hf = np.asarray(hf); hl = np.asarray(hl)
    mu = hf.mean(axis=0)
    nu = hl.mean(axis=0)
    num = float(np.linalg.norm(mu - nu))
    den = float(np.linalg.norm(hf - mu, axis=1).mean())
    return num / (den + eps)


def rcs_profile(HF, HL, eps=EPS):
    """RCS for every layer. HF: [n_f, L, d], HL: [n_l, L, d]."""
    return np.array([rcs_layer(HF[:, l, :], HL[:, l, :], eps) for l in range(HF.shape[1])])


# ---------------------------------------------------------------- permutation null
def make_label_perms(N, n_f, n_perm, seed=42):
    """Draw `n_perm` relabelings of N samples, returning the pseudo-harmful index sets.

    Returns int array [n_perm, n_f]. Class counts are preserved by construction.

    These index sets are drawn ONCE and reused for every layer, because a sample is a
    prompt that has a representation at every layer: a permutation test on the PROFILE
    must relabel prompts once per replicate and apply that relabeling across all depths.
    Permuting independently per layer would destroy the cross-layer dependence that the
    profile statistics (Gini, n80) are computed over, and would understate the null's
    spread.
    """
    rng = np.random.RandomState(seed)
    out = np.empty((n_perm, n_f), dtype=np.int64)
    for k in range(n_perm):
        out[k] = rng.permutation(N)[:n_f]
    return out


def permuted_rcs_for_layer(pooled, n_f, perms, eps=EPS):
    """Permuted-label RCS at one layer for every supplied relabeling.

    `pooled` is [N, d] with the two classes concatenated; `perms` is [n_perm, n_f] of
    pseudo-harmful indices. BOTH the centroid gap and the pseudo-harmful intra-class
    spread are recomputed per replicate (the original harmful variance is never frozen).

    Uses the exact identity  mu_A - mu_B = (N / n_l) (mu_A - m),  where m is the pooled
    centroid, so the complement never has to be materialized.
    """
    pooled = np.asarray(pooled)
    N = pooled.shape[0]
    n_l = N - n_f
    m = pooled.mean(axis=0)
    out = np.empty(len(perms), dtype=float)
    for k, idx in enumerate(perms):
        A = pooled[idx]
        mu = A.mean(axis=0)
        num = (N / n_l) * float(np.linalg.norm(mu - m))
        den = float(np.linalg.norm(A - mu, axis=1).mean())
        out[k] = num / (den + eps)
    return out


def permuted_rcs_for_layer_fast(pooled, n_f, perms, eps=EPS):
    """Vectorized equivalent of permuted_rcs_for_layer (same values, ~20x faster).

    Two BLAS GEMMs replace the per-replicate gather:
      subset means   mu_k = (M @ Xc) / n_f                    with M the [K, N] indicator
      distances      ||x_i - mu_k||^2 = ||x_i||^2 - 2 x_i.mu_k + ||mu_k||^2
    Data are centered on the pooled mean first, which both conditions the distance
    computation and lets the numerator use the exact identity
      mu_A - mu_B = (N / n_l)(mu_A - m).
    """
    pooled = np.ascontiguousarray(pooled, dtype=np.float32)
    N, d = pooled.shape
    n_l = N - n_f
    perms = np.asarray(perms)
    if perms.ndim == 1:
        perms = perms[None, :]
    K = perms.shape[0]

    m = pooled.mean(axis=0, dtype=np.float64)
    Xc = pooled - m.astype(np.float32)

    M = np.zeros((K, N), dtype=np.float32)
    M[np.repeat(np.arange(K), n_f), perms.ravel()] = 1.0

    muc = (M @ Xc) / np.float32(n_f)                                  # [K, d]
    num = (N / n_l) * np.linalg.norm(muc.astype(np.float64), axis=1)  # [K]

    xsq = np.einsum("ij,ij->i", Xc, Xc).astype(np.float64)            # [N]
    musq = np.einsum("kj,kj->k", muc, muc).astype(np.float64)         # [K]
    d2 = xsq[:, None] - 2.0 * (Xc @ muc.T).astype(np.float64) + musq[None, :]
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2, out=d2)                                        # [N, K]
    den = (M.T.astype(np.float64) * dist).sum(axis=0) / n_f           # [K]
    return num / (den + eps)


def permuted_rcs_layer(pooled, n_f, rng, eps=EPS):
    """Single permuted-label RCS at one layer (convenience; draws its own relabeling)."""
    N = pooled.shape[0]
    idx = rng.permutation(N)[:n_f]
    return float(permuted_rcs_for_layer(pooled, n_f, idx[None, :], eps)[0])


def permutation_null_profiles(HF, HL, n_perm=200, seed=42, eps=EPS,
                              shared_labels=True, progress=None):
    """Empirical permuted-label RCS profiles. Returns [n_perm, L].

    shared_labels=True (default, and the correct permutation test) relabels prompts once
    per replicate and applies that relabeling at every layer. shared_labels=False
    permutes independently per layer and is provided only for comparison.

    Requires SAMPLE-LEVEL representations: HF [n_f, L, d], HL [n_l, L, d].
    """
    HF = np.asarray(HF); HL = np.asarray(HL)
    assert HF.shape[1] == HL.shape[1], "layer counts must match"
    assert HF.shape[2] == HL.shape[2], "hidden dims must match"
    n_f, L = HF.shape[0], HF.shape[1]
    N = n_f + HL.shape[0]
    perms = make_label_perms(N, n_f, n_perm, seed) if shared_labels else None
    out = np.zeros((n_perm, L), dtype=float)
    for l in range(L):
        pooled = np.concatenate([HF[:, l, :], HL[:, l, :]], axis=0)
        if shared_labels:
            out[:, l] = permuted_rcs_for_layer_fast(pooled, n_f, perms, eps)
        else:
            rng = np.random.RandomState(seed + 1000 * l)
            out[:, l] = [permuted_rcs_layer(pooled, n_f, rng, eps) for _ in range(n_perm)]
        if progress:
            progress(l, L)
    return out


# ---------------------------------------------------------------- analytic null
def analytic_null_mean(n_f, n_l, kappa=1.0):
    """Leading-order E[RCS] under permuted labels: sqrt(1/n_f + 1/n_l) * kappa * sqrt(N/(N-1))."""
    N = n_f + n_l
    return float(np.sqrt(1.0 / n_f + 1.0 / n_l) * kappa * np.sqrt(N / (N - 1.0)))


def analytic_null_exact(n_f, n_l, tr_S, D):
    """E[RCS_perm] = (N/D) sqrt(tr_S / (n_f n_l (N-1))) -- no concentration assumption."""
    N = n_f + n_l
    return float((N / D) * np.sqrt(tr_S / (n_f * n_l * (N - 1.0))))


# ---------------------------------------------------------------- calibrated simulation
def simulate_layer(n_f, n_l, d, sigma_hf, centroid_gap, rng, eff_rank=None):
    """Synthesize one layer whose measured RCS matches a target.

    Isotropic (or spectrally decaying) Gaussian classes with equal spread, separated by
    `centroid_gap` along one axis, scaled so that the harmful mean radius equals
    `sigma_hf`. Used only to study the NULL's sampling spread, never to stand in for a
    real measurement.
    """
    if eff_rank is None:
        scale = np.ones(d)
    else:  # power-law spectrum with the requested participation ratio ~ eff_rank
        lam = 1.0 / np.arange(1, d + 1) ** 0.5
        lam = lam / np.sqrt((lam**2).sum() / d)
        scale = lam
    A = rng.randn(n_f, d).astype(np.float64) * scale
    B = rng.randn(n_l, d).astype(np.float64) * scale
    # normalize so mean radius of A is exactly sigma_hf
    r = np.linalg.norm(A - A.mean(axis=0), axis=1).mean()
    A *= sigma_hf / r
    B *= sigma_hf / r
    B[:, 0] += centroid_gap  # plant the inter-class gap
    return A, B
