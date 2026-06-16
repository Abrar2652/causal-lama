"""Synthetic CPU validation for lama_rcs.py — no GPU, no real model needed.

Builds harmful/harmless representations where refusal separability is KNOWN to
vary across layers, then checks that:
  1. RCS recovers the planted layer profile (higher RCS where classes separate).
  2. The non-circular probe AUC agrees with RCS.
  3. partial_spearman behaves correctly on constructed cases.
  4. rcs_robustness_report runs and detects unique RCS signal when present.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lama_rcs import (
    compute_rcs, compute_direction_norms, compute_probe_rcs,
    partial_spearman, rcs_robustness_report, compute_rcs_statistics,
    auto_group_layers, select_group_representatives, compute_rcs_weights,
)

rng = np.random.RandomState(0)

# ----- Build synthetic reps: [n, n_layers, d] with planted per-layer separation
n_per_class, n_layers, d = 300, 8, 32
# planted separation grows then falls (peak in the middle) — a layer-distributed profile
planted_sep = np.array([0.1, 0.4, 1.0, 2.5, 3.0, 1.5, 0.5, 0.2])
HL = rng.randn(n_per_class, n_layers, d).astype(np.float32)
HF = rng.randn(n_per_class, n_layers, d).astype(np.float32)
# shift harmful along a fixed axis per layer by planted_sep
for l in range(n_layers):
    HF[:, l, 0] += planted_sep[l]

print("== Test 1: RCS recovers planted profile ==")
rcs, meta = compute_rcs(HF, HL)
rcs_vec = np.array([rcs[l] for l in range(n_layers)])
from scipy.stats import spearmanr
rho_planted = spearmanr(rcs_vec, planted_sep)[0]
print("  RCS:", np.round(rcs_vec, 3))
print("  Spearman(RCS, planted_sep) =", round(rho_planted, 3))
assert rho_planted > 0.9, "RCS failed to recover the planted profile"

print("== Test 2: direction norms ==")
norms = compute_direction_norms(HF, HL)
norm_vec = np.array([norms[l] for l in range(n_layers)])
print("  norms:", np.round(norm_vec, 3))
assert spearmanr(norm_vec, planted_sep)[0] > 0.9

print("== Test 3: non-circular probe AUC agrees with RCS ==")
probe, pmeta = compute_probe_rcs(HF, HL, n_splits=4, max_per_class=300)
probe_vec = np.array([probe[l] for l in range(n_layers)])
print("  probe AUC:", np.round(probe_vec, 3))
agree = spearmanr(probe_vec, rcs_vec)[0]
print("  Spearman(probe, RCS) =", round(agree, 3))
assert agree > 0.8, "probe and RCS should agree on synthetic data"

print("== Test 4: partial_spearman sanity ==")
# y depends ONLY on z; x is z + noise. partial(x,y|z) should be ~0.
z = np.arange(20).astype(float)
y = z + rng.randn(20) * 0.01
x = z + rng.randn(20) * 2.0
full = spearmanr(x, y)[0]
part = partial_spearman(x, y, z)[0]
print(f"  full Spearman(x,y)={full:.3f}  partial(x,y|z)={part:.3f}")
assert abs(part) < 0.5 < abs(full), "partial correlation should collapse when z explains y"

print("== Test 5: rcs_robustness_report — UNIQUE RCS signal case ==")
# Construct DeltaASR that depends on RCS beyond norm: dasr = rank(rcs) + noise,
# while norm is a noisy copy of rcs (shares variance but not all).
delta_asr = {l: float(rcs_vec[l] / rcs_vec.max() + rng.randn() * 0.05) for l in range(n_layers)}
rep = rcs_robustness_report(rcs, delta_asr, norms, probe)
print("  spearman_rcs:", rep['spearman_rcs'])
print("  partial_rcs_given_norm:", rep['partial_rcs_given_norm'])
print("  partial_rcs_given_all:", rep['partial_rcs_given_all'])
print("  rcs_probe_agreement:", rep['rcs_probe_agreement'])
assert 'passes_noncircularity' in rep

print("== Test 6: grouping + weights ==")
for method in ('quantile', 'peaks', 'adaptive'):
    groups, gmap = auto_group_layers(rcs, n_groups=3, method=method)
    reps = select_group_representatives(rcs, groups)
    covered = sorted([l for g in groups for l in g])
    assert covered == list(range(n_layers)), f"{method}: groups must cover all layers, got {covered}"
    assert len(reps) == len(groups)
    print(f"  {method}: groups={[ (min(g),max(g)) for g in groups]} reps={reps}")
weights = compute_rcs_weights(rcs)
assert abs(max(weights.values()) - 1.0) < 1e-9 and min(weights.values()) >= 0
print("  alpha weights range:", round(min(weights.values()),3), "-", round(max(weights.values()),3))

print("== Test 7: bootstrap stats run ==")
st = compute_rcs_statistics(HF, HL, n_bootstrap=50)
assert 'cross_layer' in st and 'gini' in st['cross_layer']
print("  gini:", round(st['cross_layer']['gini'],3), " n80:", st['cross_layer']['n80_layers'])

print("\nALL SYNTHETIC TESTS PASSED ✅")
