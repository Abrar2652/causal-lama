"""
lama_rcs.py — Refusal Contribution Score (RCS): Core Scientific Instrument

This module implements the central scientific contribution of LAMA:
  1. RCS computation: a representational metric quantifying per-layer refusal encoding
  2. Causal validation: single-layer ablation sweep to measure ΔASR per layer
  3. Correlation analysis: Spearman ρ between RCS and ΔASR
  4. Layer grouping: automatic partitioning of layers into refusal-encoding regions
  5. Visualization: publication-quality RCS profiles and correlation plots

The key scientific claim: RCS(l) predicts which layers causally encode refusal,
validated by correlation with ΔASR(l) measured through actual ablation experiments.
"""

import os
import gc
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from collections import defaultdict

from config import Config


# =============================================================================
# 0. Reproducibility
# =============================================================================

def set_seeds(seed=42):
    """Set seeds for reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. RCS Computation
# =============================================================================

def compute_rcs(HF_x, HL_x, epsilon=1e-8):
    """Compute the Refusal Contribution Score (RCS) for each layer.
    
    RCS(l) = ||μ_hf(l) - ν(l)||_2 / (σ_hf(l) + ε)
    
    where:
        μ_hf(l) = centroid of harmful representations at layer l
        ν(l)    = centroid of harmless representations at layer l
        σ_hf(l) = mean intra-cluster std of harmful representations at layer l
    
    Args:
        HF_x: np.ndarray of shape [n_harmful, n_layers, hidden_dim] — harmful representations
        HL_x: np.ndarray of shape [n_harmless, n_layers, hidden_dim] — harmless representations
        epsilon: numerical stability constant
    
    Returns:
        rcs_scores: dict mapping layer_idx → RCS value
        rcs_metadata: dict with per-layer centroid distances and variances
    """
    if isinstance(HF_x, torch.Tensor):
        HF_x = HF_x.numpy()
    if isinstance(HL_x, torch.Tensor):
        HL_x = HL_x.numpy()
    
    n_layers = HF_x.shape[1]
    rcs_scores = {}
    rcs_metadata = {}
    
    for l in range(n_layers):
        # Layer representations
        hf_layer = HF_x[:, l, :]  # [n_harmful, hidden_dim]
        hl_layer = HL_x[:, l, :]  # [n_harmless, hidden_dim]
        
        # Centroids
        mu_hf = hf_layer.mean(axis=0)
        nu_hl = hl_layer.mean(axis=0)
        
        # Centroid distance (inter-class separation)
        centroid_dist = np.linalg.norm(mu_hf - nu_hl)
        
        # Intra-cluster std of harmful representations
        # Mean L2 distance from each harmful sample to harmful centroid
        diffs = hf_layer - mu_hf[np.newaxis, :]
        dists_to_centroid = np.linalg.norm(diffs, axis=1)
        sigma_hf = dists_to_centroid.mean()
        
        # RCS
        rcs = centroid_dist / (sigma_hf + epsilon)
        
        rcs_scores[l] = float(rcs)
        rcs_metadata[l] = {
            'centroid_distance': float(centroid_dist),
            'harmful_intra_std': float(sigma_hf),
            'rcs': float(rcs)
        }
    
    return rcs_scores, rcs_metadata


def normalize_rcs(rcs_scores):
    """Normalize RCS scores to [0, 1] range.
    
    Args:
        rcs_scores: dict mapping layer_idx → RCS value
    
    Returns:
        normalized: dict mapping layer_idx → normalized RCS in [0, 1]
    """
    values = list(rcs_scores.values())
    max_val = max(values) if values else 1.0
    min_val = min(values) if values else 0.0
    denom = max_val - min_val if (max_val - min_val) > 1e-10 else 1.0
    return {k: (v - min_val) / denom for k, v in rcs_scores.items()}


def compute_rcs_statistics(HF_x, HL_x, n_bootstrap=1000, ci_alpha=0.05, seed=42):
    """Compute RCS with bootstrap confidence intervals and Cohen's d.
    
    For reviewer-proof experiments: reports per-layer statistics including:
      - Bootstrap 95% CI for each layer's RCS
      - Cohen's d effect size for harmful-vs-harmless separation
      - Coefficient of variation (CV) across layers
    
    Args:
        HF_x: harmful representations [n, n_layers, d]
        HL_x: harmless representations [n, n_layers, d]
        n_bootstrap: number of bootstrap samples
        ci_alpha: confidence level (0.05 = 95% CI)
        seed: random seed
    
    Returns:
        stats_dict: dict with per-layer statistics
    """
    rng = np.random.RandomState(seed)
    if isinstance(HF_x, torch.Tensor):
        HF_x = HF_x.numpy()
    if isinstance(HL_x, torch.Tensor):
        HL_x = HL_x.numpy()
    
    n_harmful, n_layers, hidden_dim = HF_x.shape
    n_harmless = HL_x.shape[0]
    stats_dict = {}
    
    for l in range(n_layers):
        hf_layer = HF_x[:, l, :]
        hl_layer = HL_x[:, l, :]
        
        # Point estimate
        mu_hf = hf_layer.mean(axis=0)
        nu_hl = hl_layer.mean(axis=0)
        centroid_dist = np.linalg.norm(mu_hf - nu_hl)
        sigma_hf = np.linalg.norm(hf_layer - mu_hf, axis=1).mean()
        rcs = centroid_dist / (sigma_hf + 1e-8)
        
        # Cohen's d: effect size for harmful vs harmless separation
        # Using pooled std in the centroid distance space
        sigma_hl = np.linalg.norm(hl_layer - nu_hl, axis=1).mean()
        pooled_std = np.sqrt((sigma_hf**2 + sigma_hl**2) / 2)
        cohens_d = centroid_dist / (pooled_std + 1e-8)
        
        # Bootstrap CI for RCS
        boot_rcs = []
        for _ in range(n_bootstrap):
            hf_idx = rng.choice(n_harmful, size=n_harmful, replace=True)
            hl_idx = rng.choice(n_harmless, size=n_harmless, replace=True)
            hf_boot = HF_x[hf_idx, l, :]
            hl_boot = HL_x[hl_idx, l, :]
            mu_b = hf_boot.mean(axis=0)
            nu_b = hl_boot.mean(axis=0)
            cd_b = np.linalg.norm(mu_b - nu_b)
            sig_b = np.linalg.norm(hf_boot - mu_b, axis=1).mean()
            boot_rcs.append(cd_b / (sig_b + 1e-8))
        
        boot_rcs = np.array(boot_rcs)
        ci_lo = np.percentile(boot_rcs, 100 * ci_alpha / 2)
        ci_hi = np.percentile(boot_rcs, 100 * (1 - ci_alpha / 2))
        
        stats_dict[l] = {
            'rcs': float(rcs),
            'rcs_ci_lo': float(ci_lo),
            'rcs_ci_hi': float(ci_hi),
            'rcs_bootstrap_std': float(boot_rcs.std()),
            'cohens_d': float(cohens_d),
            'centroid_distance': float(centroid_dist),
            'harmful_intra_std': float(sigma_hf),
            'harmless_intra_std': float(sigma_hl),
        }
    
    # Cross-layer statistics
    all_rcs = [stats_dict[l]['rcs'] for l in range(n_layers)]
    cv = np.std(all_rcs) / (np.mean(all_rcs) + 1e-8)  # coefficient of variation
    gini = _gini_coefficient(np.array(all_rcs))
    
    stats_dict['cross_layer'] = {
        'mean_rcs': float(np.mean(all_rcs)),
        'std_rcs': float(np.std(all_rcs)),
        'cv': float(cv),
        'gini': float(gini),
        'n80_layers': int(np.searchsorted(
            np.cumsum(np.sort(all_rcs)[::-1]) / np.sum(all_rcs) * 100, 80
        ) + 1),
        'peak_to_mean_ratio': float(max(all_rcs) / (np.mean(all_rcs) + 1e-8)),
    }
    
    return stats_dict


def _gini_coefficient(values):
    """Compute Gini coefficient — measures inequality in RCS distribution.
    Gini=0: perfectly uniform. Gini→1: all RCS concentrated in one layer."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() < 1e-10:
        return 0.0
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * values).sum() / (n * values.sum()))


def bootstrap_spearman_ci(x, y, n_bootstrap=1000, ci_alpha=0.05, seed=42):
    """Compute bootstrap confidence interval for Spearman correlation.
    
    Args:
        x, y: arrays of equal length
        n_bootstrap: number of bootstrap samples
        ci_alpha: confidence level
        seed: random seed
    
    Returns:
        rho: point estimate
        p_val: p-value
        ci_lo, ci_hi: confidence interval bounds
    """
    rng = np.random.RandomState(seed)
    rho, p_val = stats.spearmanr(x, y)
    
    boot_rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rhos.append(r)
    
    boot_rhos = np.array(boot_rhos)
    if boot_rhos.size == 0 or np.isnan(rho):
        # Degenerate input (e.g. a constant array) — no defined correlation.
        return (0.0 if np.isnan(rho) else float(rho)), float(p_val), float('nan'), float('nan')
    ci_lo = np.percentile(boot_rhos, 100 * ci_alpha / 2)
    ci_hi = np.percentile(boot_rhos, 100 * (1 - ci_alpha / 2))

    return float(rho), float(p_val), float(ci_lo), float(ci_hi)


# =============================================================================
# 2b. RCS Robustness & Non-Circularity Controls
# =============================================================================
#
# THE CIRCULARITY CONCERN (the deepest reviewer threat):
#   RCS(l) = ||mu_hf(l) - nu(l)|| / sigma_hf(l) is built from the SAME
#   diff-in-means vector (mu_hf - nu) that the causal probe ablates. So a
#   correlation RCS <-> DeltaASR could be partly *mechanical*: layers where the
#   diff-in-means is large/clean are exactly the layers where that vector is an
#   accurate refusal direction, so ablating it works. We must show RCS carries
#   predictive signal BEYOND this. Two controls:
#     (1) partial correlation of RCS and DeltaASR controlling for the raw
#         diff-in-means norm (and for a structurally independent probe score);
#     (2) a non-circular refusal-encoding score (cross-validated linear-probe
#         AUC) that NEVER uses the ablated vector — if it also tracks DeltaASR,
#         the layer-distribution finding is not an artifact of RCS's definition.
# =============================================================================

def compute_direction_norms(HF_x, HL_x):
    """Per-layer raw diff-in-means norm ||mu_hf(l) - nu(l)||.

    This is the primary CONTROL covariate for the circularity analysis: RCS shares
    its numerator with the ablated direction, so we test whether RCS predicts
    DeltaASR beyond this raw norm.
    """
    if isinstance(HF_x, torch.Tensor):
        HF_x = HF_x.numpy()
    if isinstance(HL_x, torch.Tensor):
        HL_x = HL_x.numpy()
    n_layers = HF_x.shape[1]
    norms = {}
    for l in range(n_layers):
        mu_hf = HF_x[:, l, :].mean(axis=0)
        nu_hl = HL_x[:, l, :].mean(axis=0)
        norms[l] = float(np.linalg.norm(mu_hf - nu_hl))
    return norms


def compute_probe_rcs(HF_x, HL_x, n_splits=5, C=1.0, seed=42,
                      max_per_class=2000, standardize=True):
    """Non-circular refusal-encoding score: cross-validated linear-probe ROC-AUC.

    Unlike RCS, this does NOT use the diff-in-means vector that is later ablated.
    A logistic-regression probe is trained with stratified k-fold CV to separate
    harmful vs harmless at each layer; the held-out ROC-AUC measures how linearly
    *decodable* refusal is at that layer. Because the probe weights are learned
    (with regularization) and AUC is a held-out generalization statistic, this is
    a structurally independent estimate of per-layer refusal encoding.

    If RCS and this probe score AGREE on the layer profile, the layer-distributed
    finding is robust to the choice of metric and not an artifact of RCS's
    diff-in-means definition.

    Returns:
        probe_scores: dict layer -> mean held-out AUC
        probe_meta:   dict layer -> {auc_mean, auc_std}
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(
            "compute_probe_rcs requires scikit-learn. `pip install scikit-learn`."
        ) from e

    if isinstance(HF_x, torch.Tensor):
        HF_x = HF_x.numpy()
    if isinstance(HL_x, torch.Tensor):
        HL_x = HL_x.numpy()

    rng = np.random.RandomState(seed)
    n_layers = HF_x.shape[1]

    # Balanced subsample for a clean, comparable AUC per layer.
    n_hf = min(HF_x.shape[0], max_per_class)
    n_hl = min(HL_x.shape[0], max_per_class)
    hf_idx = rng.choice(HF_x.shape[0], size=n_hf, replace=False)
    hl_idx = rng.choice(HL_x.shape[0], size=n_hl, replace=False)

    y = np.concatenate([np.ones(n_hf), np.zeros(n_hl)])
    probe_scores, probe_meta = {}, {}

    for l in range(n_layers):
        X = np.concatenate([HF_x[hf_idx, l, :], HL_x[hl_idx, l, :]], axis=0)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            if standardize:
                scaler = StandardScaler().fit(Xtr)
                Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
            clf = LogisticRegression(
                C=C, max_iter=2000, class_weight='balanced', solver='lbfgs'
            ).fit(Xtr, y[tr])
            prob = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
        probe_scores[l] = float(np.mean(aucs))
        probe_meta[l] = {'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs))}

    return probe_scores, probe_meta


def partial_spearman(x, y, covars):
    """Rank-based partial correlation of x and y controlling for covars.

    Computed as the partial Pearson correlation on rank-transformed variables
    (i.e. residualize rank(x) and rank(y) on rank(covars), then correlate the
    residuals). covars may be a 1D array or a 2D [n, k] / [k, n] array.

    Returns (partial_rho, p_value).
    """
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    covars = np.asarray(covars, dtype=float)
    if covars.ndim == 1:
        covars = covars.reshape(1, n)
    elif covars.shape[0] == n and covars.shape[1] != n:
        covars = covars.T
    # rank-transform everything
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(n)] + [rankdata(covars[j]) for j in range(covars.shape[0])])
    bx = np.linalg.lstsq(Zr, xr, rcond=None)[0]
    by = np.linalg.lstsq(Zr, yr, rcond=None)[0]
    rx, ry = xr - Zr @ bx, yr - Zr @ by
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0, 1.0
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def rcs_robustness_report(rcs_scores, delta_asr_scores, direction_norms,
                          probe_scores=None):
    """Assemble the non-circularity evidence table for the paper.

    Compares RCS against control predictors of DeltaASR and reports whether RCS
    retains predictive power after controlling for the raw diff-in-means norm
    (and the probe score). The headline numbers a reviewer will look for:
      - spearman_rcs:            does RCS predict DeltaASR?
      - spearman_norm:           does the raw (shared-numerator) norm alone?
      - spearman_probe:          does the non-circular probe score?
      - partial_rcs_given_norm:  RCS's UNIQUE contribution beyond the norm
      - partial_rcs_given_all:   RCS's unique contribution beyond norm + probe
    A large partial_rcs_given_norm is the key anti-circularity result.
    """
    layers = sorted(set(rcs_scores) & set(delta_asr_scores) & set(direction_norms))
    rcs = np.array([rcs_scores[l] for l in layers])
    dasr = np.array([delta_asr_scores[l] for l in layers])
    norm = np.array([direction_norms[l] for l in layers])

    report = {'n_layers': len(layers)}
    if len(layers) < 4:
        report['warning'] = 'Too few layers for stable partial correlation (need >= 4).'
        return report

    sp_rcs = stats.spearmanr(rcs, dasr)
    sp_norm = stats.spearmanr(norm, dasr)
    report['spearman_rcs'] = {'rho': float(sp_rcs[0]), 'p': float(sp_rcs[1])}
    report['spearman_norm'] = {'rho': float(sp_norm[0]), 'p': float(sp_norm[1])}

    pr_rho, pr_p = partial_spearman(rcs, dasr, norm)
    report['partial_rcs_given_norm'] = {'rho': pr_rho, 'p': pr_p}

    covars = [norm]
    if probe_scores is not None:
        probe = np.array([probe_scores[l] for l in layers])
        sp_probe = stats.spearmanr(probe, dasr)
        report['spearman_probe'] = {'rho': float(sp_probe[0]), 'p': float(sp_probe[1])}
        # cross-metric agreement: does the non-circular probe track RCS?
        agree = stats.spearmanr(rcs, probe)
        report['rcs_probe_agreement'] = {'rho': float(agree[0]), 'p': float(agree[1])}
        pr2_rho, pr2_p = partial_spearman(rcs, dasr, probe)
        report['partial_rcs_given_probe'] = {'rho': pr2_rho, 'p': pr2_p}
        covars.append(probe)

    pall_rho, pall_p = partial_spearman(rcs, dasr, np.vstack(covars))
    report['partial_rcs_given_all'] = {'rho': pall_rho, 'p': pall_p}

    # Verdict heuristic for the console
    survives = report['partial_rcs_given_norm']['rho'] >= 0.4
    report['passes_noncircularity'] = bool(survives)
    return report


# =============================================================================
# 2. Layer Grouping
# =============================================================================

def auto_group_layers(rcs_scores, n_groups=3, method='quantile'):
    """Partition layers into groups based on RCS profile.
    
    Args:
        rcs_scores: dict mapping layer_idx → RCS value
        n_groups: number of groups to create
        method: 'quantile' (equal-mass) or 'peaks' (around RCS peaks)
    
    Returns:
        groups: list of lists, each containing layer indices for that group
        group_map: dict mapping layer_idx → group_id
    """
    layers = sorted(rcs_scores.keys())
    n_layers = len(layers)
    
    if method == 'quantile':
        # Split layers into n_groups equal-sized contiguous blocks
        group_size = n_layers // n_groups
        groups = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else n_layers
            groups.append(layers[start:end])
    
    elif method == 'peaks':
        # Find peaks in the RCS profile and group around them
        rcs_array = np.array([rcs_scores[l] for l in layers])
        
        # Smooth the profile for peak detection
        if len(rcs_array) > 7:
            window_len = min(7, len(rcs_array) // 2 * 2 + 1)  # must be odd
            smoothed = savgol_filter(rcs_array, window_len, polyorder=2)
        else:
            smoothed = rcs_array
        
        # Find peaks with minimum distance between them
        min_distance = max(2, n_layers // (n_groups + 1))
        peaks, properties = find_peaks(smoothed, distance=min_distance)
        
        # If fewer peaks than groups, fall back to quantile
        if len(peaks) < n_groups:
            return auto_group_layers(rcs_scores, n_groups, method='quantile')
        
        # Keep only top n_groups peaks by prominence
        if len(peaks) > n_groups:
            peak_values = smoothed[peaks]
            top_indices = np.argsort(peak_values)[-n_groups:]
            peaks = np.sort(peaks[top_indices])
        
        # Assign each layer to the nearest peak
        groups = [[] for _ in range(n_groups)]
        for l_idx, l in enumerate(layers):
            distances = np.abs(peaks - l_idx)
            nearest_peak = np.argmin(distances)
            groups[nearest_peak].append(l)
    
    elif method == 'adaptive':
        # RCS-threshold based: high-RCS layers get their own group, low-RCS layers are grouped together
        rcs_array = np.array([rcs_scores[l] for l in layers])
        threshold = np.percentile(rcs_array, 100 * (1 - 1.0 / n_groups))
        
        high_rcs_layers = [l for l in layers if rcs_scores[l] >= threshold]
        low_rcs_layers = [l for l in layers if rcs_scores[l] < threshold]
        
        # Split high-RCS layers into n_groups-1 groups, all low-RCS into one group
        if len(high_rcs_layers) >= n_groups - 1:
            high_group_size = len(high_rcs_layers) // (n_groups - 1)
            groups = []
            for g in range(n_groups - 1):
                start = g * high_group_size
                end = start + high_group_size if g < n_groups - 2 else len(high_rcs_layers)
                groups.append(high_rcs_layers[start:end])
            groups.append(low_rcs_layers)
        else:
            return auto_group_layers(rcs_scores, n_groups, method='quantile')
    
    else:
        raise ValueError(f"Unknown grouping method: {method}")
    
    # Build layer → group mapping
    group_map = {}
    for g_id, group_layers in enumerate(groups):
        for l in group_layers:
            group_map[l] = g_id
    
    return groups, group_map


def select_group_representatives(rcs_scores, groups):
    """Select the representative layer (highest RCS) for each group.
    
    Args:
        rcs_scores: dict mapping layer_idx → RCS value
        groups: list of lists of layer indices
    
    Returns:
        representatives: list of layer indices (one per group)
    """
    representatives = []
    for group_layers in groups:
        if not group_layers:
            representatives.append(None)
            continue
        best_layer = max(group_layers, key=lambda l: rcs_scores.get(l, 0))
        representatives.append(best_layer)
    return representatives


def compute_rcs_weights(rcs_scores):
    """Convert RCS scores into initial ablation weights α_l.
    
    α_l = RCS(l) / max_l' RCS(l')
    
    Args:
        rcs_scores: dict mapping layer_idx → RCS value
    
    Returns:
        weights: dict mapping layer_idx → α_l in [0, 1]
    """
    max_rcs = max(rcs_scores.values()) if rcs_scores else 1.0
    if max_rcs < 1e-10:
        return {l: 1.0 for l in rcs_scores}
    return {l: rcs / max_rcs for l, rcs in rcs_scores.items()}


# =============================================================================
# 3. Causal Validation
# =============================================================================

def compute_single_layer_delta_asr(
    model_name, device, HF_x, HL_x, layer, dataset_name='harmbench_val'
):
    """Ablate a single direction at a single layer and measure ΔASR.
    
    This performs causal intervention: compute the SD at this specific layer,
    ablate ONLY this layer's weights, generate completions, and evaluate.
    
    Args:
        model_name: name of the model to load
        device: torch device string
        HF_x: harmful representations [n, n_layers, d]
        HL_x: harmless representations [n, n_layers, d]
        layer: which layer to ablate
        dataset_name: evaluation dataset
    
    Returns:
        delta_asr: float, ASR change from ablating this single layer
    """
    from models.load_models import load_model
    from utils.ablation_utils import get_weighted_ablated_matrix
    from directions_ablation import generate_and_save_hookfree_completions
    from eval_jailbreaks import evaluate_jailbreak
    
    # Compute single direction at this layer
    if isinstance(HF_x, np.ndarray):
        HF_t = torch.tensor(HF_x, dtype=torch.float32)
    else:
        HF_t = HF_x.float()
    if isinstance(HL_x, np.ndarray):
        HL_t = torch.tensor(HL_x, dtype=torch.float32)
    else:
        HL_t = HL_x.float()
    
    mu_hf = HF_t[:, layer, :].mean(dim=0)
    nu_hl = HL_t[:, layer, :].mean(dim=0)
    direction = mu_hf - nu_hl  # single direction at this layer
    
    # Load model
    llm = load_model(model_name, device)
    cfg = Config(model_alias=model_name, model_path=model_name)
    
    # Ablate only the specified layer
    blocks = llm.model_block_modules
    if layer < len(blocks):
        block = blocks[layer]
        o_proj, down_proj = llm._get_block_proj_weights(block)
        o_proj.data = get_weighted_ablated_matrix(o_proj.data.T, direction, alpha=1.0).T
        down_proj.data = get_weighted_ablated_matrix(down_proj.data.T, direction, alpha=1.0).T
    
    # Generate completions
    aux_name = f'rcs_causal_layer{layer}'
    completions = generate_and_save_hookfree_completions(
        cfg=cfg,
        folder='causal_validation',
        model_base=llm,
        dataset_name=dataset_name,
        aux_name=aux_name,
        return_completions=True,
    )
    
    # Cleanup — gc.collect() is REQUIRED before empty_cache: nn.Module has
    # circular refs, so `del` alone doesn't release CUDA memory and it accumulates
    # ~model-size per layer until OOM.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate
    eval_path = os.path.join(cfg.artifact_path(), 'causal_validation')
    os.makedirs(eval_path, exist_ok=True)
    result = evaluate_jailbreak(
        completions=completions,
        methodologies=['harmbench'],
        evaluation_path=os.path.join(eval_path, f'causal_layer{layer}_results.json'),
        judge_keep_loaded=False,  # free 26GB judge so next layer's model load has a clean GPU
    )

    return float(result['harmbench_success_rate'])


def compute_no_ablation_asr(model_name, device, dataset_name='harmbench_val'):
    """Baseline ASR with NO ablation — the reference point for DeltaASR.

    DeltaASR(l) = ASR(ablate layer l) - ASR(no ablation). Reporting raw ASR as if
    it were DeltaASR (the previous behavior) hides layers that *increase* refusal
    and is not what the scientific claim states.
    """
    from models.load_models import load_model
    from directions_ablation import generate_and_save_hookfree_completions
    from eval_jailbreaks import evaluate_jailbreak

    llm = load_model(model_name, device)
    cfg = Config(model_alias=model_name, model_path=model_name)
    completions = generate_and_save_hookfree_completions(
        cfg=cfg, folder='causal_validation', model_base=llm,
        dataset_name=dataset_name, aux_name='rcs_causal_noablation',
        return_completions=True,
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    eval_path = os.path.join(cfg.artifact_path(), 'causal_validation')
    os.makedirs(eval_path, exist_ok=True)
    result = evaluate_jailbreak(
        completions=completions, methodologies=['harmbench'],
        evaluation_path=os.path.join(eval_path, 'causal_noablation_results.json'),
        judge_keep_loaded=False,
    )
    return float(result['harmbench_success_rate'])


def run_causal_validation(
    model_name, device, HF_x, HL_x, rcs_scores,
    dataset_name='harmbench_val', skip_low_rcs=False, rcs_threshold_percentile=20
):
    """Run the full causal validation protocol.
    
    For each layer (or a subset), compute ΔASR via single-layer ablation,
    then correlate with RCS scores.
    
    Args:
        model_name: model name
        device: torch device
        HF_x, HL_x: representations
        rcs_scores: pre-computed RCS values
        dataset_name: evaluation dataset
        skip_low_rcs: if True, skip layers below threshold to save compute
        rcs_threshold_percentile: percentile below which to skip
    
    Returns:
        delta_asr_scores: dict mapping layer → ΔASR
        correlation: dict with 'spearman_rho', 'spearman_p', 'pearson_r', 'pearson_p'
    """
    layers = sorted(rcs_scores.keys())
    
    # Optionally skip low-RCS layers
    if skip_low_rcs:
        rcs_values = [rcs_scores[l] for l in layers]
        threshold = np.percentile(rcs_values, rcs_threshold_percentile)
        eval_layers = [l for l in layers if rcs_scores[l] >= threshold]
        print(f"[RCS Causal] Evaluating {len(eval_layers)}/{len(layers)} layers "
              f"(skipping {len(layers)-len(eval_layers)} below RCS threshold {threshold:.4f})")
    else:
        eval_layers = layers
    
    # Reference point: ASR with NO ablation (computed once).
    print("\n[RCS Causal] Computing no-ablation baseline ASR...")
    baseline_asr = compute_no_ablation_asr(model_name, device, dataset_name)
    print(f"  → baseline ASR (no ablation) = {baseline_asr:.4f}")

    raw_asr_scores = {}
    delta_asr_scores = {}
    for l in eval_layers:
        print(f"\n[RCS Causal] Layer {l}/{max(layers)} (RCS={rcs_scores[l]:.4f})...")
        try:
            asr = compute_single_layer_delta_asr(
                model_name, device, HF_x, HL_x, l, dataset_name
            )
            raw_asr_scores[l] = asr
            delta_asr_scores[l] = asr - baseline_asr
            print(f"  → ASR={asr:.4f}  ΔASR={delta_asr_scores[l]:+.4f}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            raw_asr_scores[l] = baseline_asr
            delta_asr_scores[l] = 0.0

    # Compute correlations on evaluated layers
    common_layers = sorted(delta_asr_scores.keys())
    rcs_vals = np.array([rcs_scores[l] for l in common_layers])
    asr_vals = np.array([delta_asr_scores[l] for l in common_layers])
    
    if len(common_layers) > 2:
        sp_rho, sp_p = stats.spearmanr(rcs_vals, asr_vals)
        pr_r, pr_p = stats.pearsonr(rcs_vals, asr_vals)
        # Bootstrap CI for Spearman (reviewer-proof)
        _, _, sp_ci_lo, sp_ci_hi = bootstrap_spearman_ci(rcs_vals, asr_vals)
    else:
        sp_rho, sp_p, pr_r, pr_p = 0.0, 1.0, 0.0, 1.0
        sp_ci_lo, sp_ci_hi = 0.0, 0.0
    
    correlation = {
        'spearman_rho': float(sp_rho),
        'spearman_p': float(sp_p),
        'spearman_ci_lo': float(sp_ci_lo),
        'spearman_ci_hi': float(sp_ci_hi),
        'pearson_r': float(pr_r),
        'pearson_p': float(pr_p),
        'n_layers_evaluated': len(common_layers),
        'n_layers_total': len(layers),
        'baseline_asr_no_ablation': float(baseline_asr),
        'raw_asr_scores': {int(l): float(v) for l, v in raw_asr_scores.items()},
    }
    
    print(f"\n[RCS Causal] Correlation Results:")
    print(f"  Spearman ρ = {sp_rho:.4f} (p = {sp_p:.6f})")
    print(f"  Spearman 95% CI: [{sp_ci_lo:.4f}, {sp_ci_hi:.4f}]")
    print(f"  Pearson  r = {pr_r:.4f} (p = {pr_p:.6f})")
    
    return delta_asr_scores, correlation


# =============================================================================
# 4. Visualization
# =============================================================================

def plot_rcs_profile(rcs_scores, model_name, groups=None, save_path='logs'):
    """Plot the RCS profile across layers — publication quality.
    
    Args:
        rcs_scores: dict mapping layer_idx → RCS
        model_name: name for title and filename
        groups: optional list of lists for coloring groups
        save_path: directory to save the figure
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    
    layers = sorted(rcs_scores.keys())
    rcs_values = [rcs_scores[l] for l in layers]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if groups is not None:
        # Color by group
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for g_id, group_layers in enumerate(groups):
            g_layers = [l for l in layers if l in group_layers]
            g_values = [rcs_scores[l] for l in g_layers]
            ax.bar(g_layers, g_values, color=colors[g_id], 
                   label=f'Group {g_id}', alpha=0.8, edgecolor='black', linewidth=0.5)
    else:
        ax.bar(layers, rcs_values, color='steelblue', alpha=0.8, 
               edgecolor='black', linewidth=0.5)
    
    # Overlay smooth line
    if len(rcs_values) > 5:
        window = min(5, len(rcs_values) // 2 * 2 + 1)
        smoothed = savgol_filter(rcs_values, window, polyorder=2)
        ax.plot(layers, smoothed, color='darkred', linewidth=2, alpha=0.7, label='Smoothed')
    
    ax.set_xlabel('Layer Index', fontsize=14)
    ax.set_ylabel('RCS(l)', fontsize=14)
    ax.set_title(f'Refusal Contribution Score Profile — {model_name}', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'rcs_profile_{model_name}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"[✓] RCS profile saved to {filepath}")


def plot_rcs_vs_delta_asr(rcs_scores, delta_asr_scores, correlation, model_name, save_path='logs'):
    """Plot RCS vs ΔASR scatter with regression line — causal validation figure.
    
    Args:
        rcs_scores: dict layer → RCS
        delta_asr_scores: dict layer → ΔASR
        correlation: dict with 'spearman_rho', 'spearman_p'
        model_name: name for title
        save_path: directory to save
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    
    common_layers = sorted(set(rcs_scores.keys()) & set(delta_asr_scores.keys()))
    rcs_vals = np.array([rcs_scores[l] for l in common_layers])
    asr_vals = np.array([delta_asr_scores[l] for l in common_layers])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Scatter
    scatter = ax.scatter(rcs_vals, asr_vals, c=common_layers, cmap='viridis',
                         s=80, edgecolors='black', linewidth=0.8, zorder=5)
    
    # Regression line
    if len(rcs_vals) > 2:
        z = np.polyfit(rcs_vals, asr_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(rcs_vals.min(), rcs_vals.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Linear fit')
    
    # Annotate each point with layer index
    for l, rx, ax_val in zip(common_layers, rcs_vals, asr_vals):
        ax.annotate(str(l), (rx, ax_val), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Layer Index')
    
    # Correlation text
    rho = correlation['spearman_rho']
    p_val = correlation['spearman_p']
    ax.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np = {p_val:.2e}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('RCS(l)', fontsize=14)
    ax.set_ylabel('ΔASR(l)', fontsize=14)
    ax.set_title(f'RCS vs Causal ΔASR — {model_name}', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'rcs_vs_dasr_{model_name}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"[✓] RCS vs ΔASR plot saved to {filepath}")


def plot_multi_model_rcs(all_rcs_scores, model_names, save_path='logs'):
    """Plot RCS profiles for multiple models in a single figure.
    
    Args:
        all_rcs_scores: list of dicts (one per model)
        model_names: list of model name strings
        save_path: directory to save
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    
    n_models = len(model_names)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (rcs_scores, name) in enumerate(zip(all_rcs_scores, model_names)):
        ax = axes[idx]
        layers = sorted(rcs_scores.keys())
        values = [rcs_scores[l] for l in layers]
        
        # Normalize for cross-model comparison
        norm_values = np.array(values)
        norm_values = (norm_values - norm_values.min()) / (norm_values.max() - norm_values.min() + 1e-10)
        
        ax.bar(layers, norm_values, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Normalized RCS')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Cross-Model Refusal Contribution Score Profiles', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, 'rcs_multi_model.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"[✓] Multi-model RCS plot saved to {filepath}")


# =============================================================================
# 5. CLI Entry Point
# =============================================================================

def load_representations(model_name):
    """Load pre-computed representations for a model."""
    HL_x = torch.load(f"./dataset/representations/{model_name}/HLx_train.pt", weights_only=True).numpy()
    HF_x = torch.load(f"./dataset/representations/{model_name}/HFx_train.pt", weights_only=True).numpy()
    print(f"Loaded representations: HL_x={HL_x.shape}, HF_x={HF_x.shape}")
    return HF_x, HL_x


def main():
    parser = argparse.ArgumentParser(description="LAMA: Refusal Contribution Score Analysis")
    parser.add_argument('--model_name', type=str, required=True, help='Model name for representations')
    parser.add_argument('--n_groups', type=int, default=3, help='Number of layer groups')
    parser.add_argument('--grouping_method', type=str, default='quantile',
                        choices=['quantile', 'peaks', 'adaptive'],
                        help='Layer grouping method')
    parser.add_argument('--run_causal', action='store_true',
                        help='Run causal validation (requires GPU + model inference)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for causal validation')
    parser.add_argument('--dataset_name', type=str, default='harmbench_val',
                        help='Dataset for causal validation')
    parser.add_argument('--save_path', type=str, default='logs', help='Output directory')
    parser.add_argument('--run_robustness', action='store_true',
                        help='Run non-circularity controls (probe AUC + partial correlations). '
                             'Requires --run_causal for the DeltaASR-based partial correlations.')
    parser.add_argument('--probe_max_per_class', type=int, default=2000,
                        help='Max samples per class for the linear-probe AUC (non-circular RCS).')
    args = parser.parse_args()
    
    # Load representations
    HF_x, HL_x = load_representations(args.model_name)
    
    # Step 1: Compute RCS
    print("\n" + "="*60)
    print("STEP 1: Computing Refusal Contribution Score (RCS)")
    print("="*60)
    rcs_scores, rcs_metadata = compute_rcs(HF_x, HL_x)
    
    # Print RCS profile
    print(f"\n{'Layer':<8} {'RCS':<12} {'CentDist':<12} {'IntraStd':<12}")
    print("-" * 44)
    for l in sorted(rcs_scores.keys()):
        m = rcs_metadata[l]
        print(f"{l:<8} {m['rcs']:<12.4f} {m['centroid_distance']:<12.4f} {m['harmful_intra_std']:<12.4f}")
    
    # Find top-5 layers
    top5 = sorted(rcs_scores.keys(), key=lambda l: rcs_scores[l], reverse=True)[:5]
    print(f"\nTop-5 RCS layers: {top5}")
    print(f"  Values: {[round(rcs_scores[l], 4) for l in top5]}")
    
    # Step 2: Layer Grouping
    print("\n" + "="*60)
    print(f"STEP 2: Layer Grouping ({args.n_groups} groups, method='{args.grouping_method}')")
    print("="*60)
    groups, group_map = auto_group_layers(rcs_scores, args.n_groups, args.grouping_method)
    representatives = select_group_representatives(rcs_scores, groups)
    
    for g_id, (group_layers, rep) in enumerate(zip(groups, representatives)):
        group_rcs = [rcs_scores[l] for l in group_layers]
        print(f"  Group {g_id}: layers {min(group_layers)}-{max(group_layers)} "
              f"({len(group_layers)} layers), rep=layer {rep}, "
              f"mean RCS={np.mean(group_rcs):.4f}")
    
    # Step 3: Compute initial alpha weights
    print("\n" + "="*60)
    print("STEP 3: Computing RCS-based Initial Ablation Weights (α_l)")
    print("="*60)
    alpha_weights = compute_rcs_weights(rcs_scores)
    print(f"  α range: [{min(alpha_weights.values()):.4f}, {max(alpha_weights.values()):.4f}]")
    print(f"  Mean α: {np.mean(list(alpha_weights.values())):.4f}")
    
    # Step 4: Visualization
    print("\n" + "="*60)
    print("STEP 4: Generating Visualizations")
    print("="*60)
    plot_rcs_profile(rcs_scores, args.model_name, groups=groups, save_path=args.save_path)
    
    # Step 5: Causal Validation (optional — expensive)
    if args.run_causal:
        print("\n" + "="*60)
        print("STEP 5: Causal Validation (Single-Layer Ablation Sweep)")
        print("="*60)
        delta_asr_scores, correlation = run_causal_validation(
            args.model_name, args.device, HF_x, HL_x, rcs_scores,
            dataset_name=args.dataset_name
        )
        
        plot_rcs_vs_delta_asr(rcs_scores, delta_asr_scores, correlation,
                              args.model_name, save_path=args.save_path)

        # Non-circularity controls (the central anti-reviewer evidence)
        if args.run_robustness:
            print("\n" + "="*60)
            print("STEP 5b: RCS Non-Circularity Controls")
            print("="*60)
            direction_norms = compute_direction_norms(HF_x, HL_x)
            print("  Computing non-circular probe AUC per layer...")
            probe_scores, probe_meta = compute_probe_rcs(
                HF_x, HL_x, max_per_class=args.probe_max_per_class
            )
            robustness = rcs_robustness_report(
                rcs_scores, delta_asr_scores, direction_norms, probe_scores
            )
            print(f"  Spearman(RCS, ΔASR)            = {robustness.get('spearman_rcs', {}).get('rho', float('nan')):.4f}")
            print(f"  Spearman(norm, ΔASR)           = {robustness.get('spearman_norm', {}).get('rho', float('nan')):.4f}")
            print(f"  Spearman(probe, ΔASR)          = {robustness.get('spearman_probe', {}).get('rho', float('nan')):.4f}")
            print(f"  RCS↔probe agreement            = {robustness.get('rcs_probe_agreement', {}).get('rho', float('nan')):.4f}")
            print(f"  Partial RCS|norm (UNIQUE)      = {robustness.get('partial_rcs_given_norm', {}).get('rho', float('nan')):.4f}")
            print(f"  Partial RCS|norm+probe         = {robustness.get('partial_rcs_given_all', {}).get('rho', float('nan')):.4f}")
            if robustness.get('passes_noncircularity'):
                print("  ✅ RCS retains predictive power beyond the shared diff-in-means norm.")
            else:
                print("  ⚠️  RCS's unique contribution is weak — consider redefining RCS (probe-based).")

        # Decision gate
        rho = correlation['spearman_rho']
        if rho >= 0.7:
            print(f"\n✅ CAUSAL VALIDATION PASSED: ρ = {rho:.4f} ≥ 0.7")
            print("   RCS is a valid causal predictor of refusal encoding.")
        elif rho >= 0.5:
            print(f"\n⚠️  MODERATE CORRELATION: ρ = {rho:.4f}")
            print("   RCS shows meaningful but imperfect causal prediction.")
        else:
            print(f"\n❌ CAUSAL VALIDATION FAILED: ρ = {rho:.4f} < 0.5")
            print("   Consider revising the RCS definition or pivoting approach.")
    
    # Save all results
    output = {
        'model_name': args.model_name,
        'rcs_scores': rcs_scores,
        'rcs_metadata': rcs_metadata,
        'groups': {str(g): layers for g, layers in enumerate(groups)},
        'representatives': representatives,
        'alpha_weights': alpha_weights,
        'n_groups': args.n_groups,
        'grouping_method': args.grouping_method,
    }
    
    if args.run_causal:
        output['delta_asr_scores'] = delta_asr_scores
        output['correlation'] = correlation
        if args.run_robustness:
            output['direction_norms'] = direction_norms
            output['probe_scores'] = probe_scores
            output['probe_meta'] = probe_meta
            output['robustness'] = robustness
    
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, f'rcs_analysis_{args.model_name}.json')
    
    # Convert numpy types for JSON serialization
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n[✓] Full RCS analysis saved to {output_path}")


if __name__ == '__main__':
    main()
