#!/usr/bin/env python3
"""Deterministic tests for the permuted-label RCS null. CPU only, no model, no GPU.

Run: python3 analysis/test_permutation_null.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)

import null_core as nc
from lama_rcs import compute_rcs

fails = []
def check(name, cond, info=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{info}]" if info else ""))
    if not cond: fails.append(name)

print("== T1: Gini = 0 for equal positive layer scores ==")
for L in (4, 8, 28, 42):
    check(f"gini(flat L={L}) == 0", abs(nc.gini([3.7]*L)) < 1e-12)

print("== T2: n80 == ceil(0.8 L) for a uniform profile ==")
for L in (4, 10, 24, 28, 32, 36, 42):
    exp = int(np.ceil(0.8*L))
    got = nc.n80([1.0]*L)
    check(f"n80(flat L={L}) == {exp}", got == exp, f"got {got}")

print("== T3: Gini increases with concentration; n80 decreases ==")
flat = np.ones(28)
conc = np.zeros(28); conc[:3] = 10.0; conc[3:] = 0.1
check("gini(concentrated) > gini(flat)", nc.gini(conc) > nc.gini(flat), f"{nc.gini(conc):.3f} > {nc.gini(flat):.3f}")
check("n80(concentrated) < n80(flat)", nc.n80(conc) < nc.n80(flat), f"{nc.n80(conc)} < {nc.n80(flat)}")

print("== T4: permutation preserves class counts exactly ==")
rng = np.random.RandomState(0)
pooled = rng.randn(500, 16)
seen = []
for _ in range(50):
    N = pooled.shape[0]; n_f = 200
    idx = rng.permutation(N)
    seen.append((len(idx[:n_f]), len(idx[n_f:])))
check("every split is (200, 300)", all(s == (200, 300) for s in seen))

print("== T5: rcs_profile is identical to the paper's compute_rcs ==")
rng = np.random.RandomState(1)
n_f, n_l, L, d = 120, 180, 6, 32
HF = rng.randn(n_f, L, d).astype(np.float32)
HL = rng.randn(n_l, L, d).astype(np.float32)
for l in range(L):
    HF[:, l, 0] += 0.3 * (l + 1)
mine = nc.rcs_profile(HF, HL)
theirs_d, _ = compute_rcs(HF, HL)
theirs = np.array([theirs_d[l] for l in range(L)])
check("max |mine - compute_rcs| < 1e-5", np.max(np.abs(mine - theirs)) < 1e-5,
      f"{np.max(np.abs(mine-theirs)):.2e}")

print("== T6: permuted RCS matches the analytic null on concentrated data ==")
# high dimension -> strong concentration -> kappa ~ 1 -> null ~ sqrt(1/n_f + 1/n_l)
rng = np.random.RandomState(2)
n_f, n_l, d = 400, 600, 2048
A, B = nc.simulate_layer(n_f, n_l, d, sigma_hf=40.0, centroid_gap=55.0, rng=rng)
true_rcs = nc.rcs_layer(A, B)
pooled = np.concatenate([A, B], axis=0)
prng = np.random.RandomState(3)
perm = np.array([nc.permuted_rcs_layer(pooled, n_f, prng) for _ in range(300)])
pred = nc.analytic_null_mean(n_f, n_l)
rel = abs(perm.mean() - pred) / pred
check("permuted mean within 10% of analytic", rel < 0.10,
      f"perm={perm.mean():.5f} analytic={pred:.5f} rel={rel:.3f}")
check("true-label RCS is >> null", true_rcs > 20 * perm.mean(),
      f"true={true_rcs:.3f} null={perm.mean():.5f} ratio={true_rcs/perm.mean():.0f}x")
check("null spread is small (CV < 0.25)", perm.std()/perm.mean() < 0.25,
      f"cv={perm.std()/perm.mean():.4f}")

print("== T7: the null RECOMPUTES the denominator (does not freeze harmful variance) ==")
# Build classes with very different spreads. If the denominator were frozen at the
# original harmful spread, the permuted score would be systematically mis-scaled.
rng = np.random.RandomState(4)
tight = rng.randn(300, 256) * 1.0
wide  = rng.randn(300, 256) * 6.0
pooled2 = np.concatenate([tight, wide], axis=0)
prng = np.random.RandomState(5)
perm2 = np.array([nc.permuted_rcs_layer(pooled2, 300, prng) for _ in range(200)])
frozen_den = np.linalg.norm(tight - tight.mean(axis=0), axis=1).mean()
prng2 = np.random.RandomState(5)
perm2_frozen = []
for _ in range(200):
    N = pooled2.shape[0]; idx = prng2.permutation(N)
    Aa = pooled2[idx[:300]]; Bb = pooled2[idx[300:]]
    perm2_frozen.append(float(np.linalg.norm(Aa.mean(0) - Bb.mean(0))) / (frozen_den + nc.EPS))
perm2_frozen = np.array(perm2_frozen)
check("recomputed != frozen-denominator variant", abs(perm2.mean() - perm2_frozen.mean()) > 1e-3,
      f"recomputed={perm2.mean():.5f} frozen={perm2_frozen.mean():.5f}")

print("== T8: determinism ==")
p1 = np.array([nc.permuted_rcs_layer(pooled, n_f, np.random.RandomState(7)) for _ in range(5)])
p2 = np.array([nc.permuted_rcs_layer(pooled, n_f, np.random.RandomState(7)) for _ in range(5)])
check("same seed -> identical", np.allclose(p1, p2))

print("== T9: permuted PROFILE is flat -> Gini ~ 0, n80/L ~ 0.8 ==")
rng = np.random.RandomState(8)
n_f, n_l, L, d = 300, 450, 24, 1024
HF2 = np.zeros((n_f, L, d), dtype=np.float32); HL2 = np.zeros((n_l, L, d), dtype=np.float32)
for l in range(L):
    a, b = nc.simulate_layer(n_f, n_l, d, sigma_hf=10.0 + 2.0*l, centroid_gap=12.0 + 0.5*l, rng=rng)
    HF2[:, l, :] = a; HL2[:, l, :] = b
nulls = nc.permutation_null_profiles(HF2, HL2, n_perm=40, seed=11)
ginis = np.array([nc.gini(r) for r in nulls])
n80s  = np.array([nc.n80(r)/L for r in nulls])
obs = nc.rcs_profile(HF2, HL2)
check("null Gini is near zero (< 0.05)", ginis.mean() < 0.05, f"mean null gini={ginis.mean():.4f}")
check("null n80/L is near 0.8", abs(n80s.mean() - 0.8) < 0.06, f"mean null n80/L={n80s.mean():.3f}")
check("observed Gini exceeds null Gini", nc.gini(obs) > ginis.max(),
      f"obs={nc.gini(obs):.4f} max-null={ginis.max():.4f}")

print()
if fails:
    print(f"{len(fails)} TEST(S) FAILED:", fails); sys.exit(1)
print("ALL PERMUTATION-NULL TESTS PASSED")
