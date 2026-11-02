#!/usr/bin/env python3
"""Depth interval for the effect-size-matched length control (reviewers PKr6, p6Gj-W1).

Part 2 of matched_controls/RESULTS.md reports eight non-refusal contrasts, but only seven
carry the bootstrap depth intervals that Part 3 compares against refusal: length_effectmatched
was measured before depth statistics were added. The claim "refusal's centre of mass is
disjoint from all eight benign intervals" therefore overcounted by one. This script measures
the missing interval instead of narrowing the claim.

The split is reconstructed, not re-drawn: nonrefusal_controls.py tunes beta with a fresh
RandomState(SEED) inside each bisection step, so the search is independent of outer rng state
and reproduces exactly. The reconstruction is checked against the recorded mean RCS, peak layer
and Gini before any new number is reported; a mismatch means the split differs and aborts.

Depth is summarised exactly as deep_benign_contrasts.py does, so the interval is directly
comparable: centre of mass over normalised depth, 95% bootstrap over the small class with the
reference centroid held fixed.

Writes matched_controls/length_matched_depth_qwen2-7b.json
"""
import json, os, sys
import numpy as np

REPO = "/nas/home/jahin/causal-lama"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))

import null_core as nc
from nonrefusal_controls import (HOLD, OUT, SEED, N_A, N_B, TARGET_DELTA,
                                 weighted_length_split, cliffs_delta)

N_BOOT, EPS = 1000, 1e-8
# refusal's interval, from deep_benign_qwen2-7b.json
REF_COM, REF_LO, REF_HI = 0.5823, 0.5769, 0.5870


def rcs_fast(A, nu):
    mu = A.mean(axis=0)
    gap = np.linalg.norm(mu - nu, axis=1)
    sig = np.linalg.norm(A - mu[None, :, :], axis=2).mean(axis=0)
    return gap / (sig + EPS)


def depth(v):
    L = len(v); w = v / v.sum()
    return int(np.argmax(v)), float((np.arange(L) * w).sum() / (L - 1))


def main():
    import torch
    from transformers import AutoTokenizer

    H = torch.load(os.path.join(HOLD, "HLholdout_train.pt"),
                   weights_only=True).numpy().astype(np.float32)
    prompts = json.load(open(os.path.join(HOLD, "prompts.json")))["prompts"]
    assert len(prompts) == H.shape[0]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    lens = np.array([len(tok(p).input_ids) for p in prompts])

    # reconstruct the tuned split, bit for bit
    lo, hi, best = 0.0, 6.0, None
    for _ in range(18):
        mid = (lo + hi) / 2
        a, b = weighted_length_split(lens, np.random.RandomState(SEED), mid)
        d = cliffs_delta(lens[a], lens[b])
        best = (mid, a, b, d)
        if d < TARGET_DELTA: lo = mid
        else: hi = mid
    beta, ia, ib, delta = best
    print(f"[rebuild] beta={beta:.3f}  Cliff's delta={delta:+.3f} (target {TARGET_DELTA})")

    A, B = H[ia], H[ib]
    v = rcs_fast(A, B.mean(axis=0))
    L = len(v)

    # guard: does the reconstruction match what Part 2 recorded?
    rec = json.load(open(os.path.join(
        OUT, "nonrefusal_controls_qwen2-7b.json")))["contrasts"]["length_effectmatched"]
    dm = abs(rec["mean_rcs"] - float(v.mean()))
    dg = abs(rec["gini"] - float(nc.gini(v)))
    print(f"[check] vs recorded: mean {rec['mean_rcs']:.4f} vs {v.mean():.4f} (d={dm:.2e}), "
          f"peak L{rec['peak_layer']} vs L{int(np.argmax(v))}, "
          f"gini {rec['gini']:.4f} vs {nc.gini(v):.4f} (d={dg:.2e})")
    assert dm < 1e-3 and dg < 1e-3 and rec["peak_layer"] == int(np.argmax(v)), \
        "reconstruction does not match the recorded split; do not report this interval"

    nu = B.mean(axis=0)
    rng = np.random.RandomState(SEED); n = A.shape[0]
    pk = np.zeros(N_BOOT, int); cm = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        pk[i], cm[i] = depth(rcs_fast(A[rng.randint(0, n, n)], nu))
    peak, com = depth(v)
    q = lambda x, p: float(np.percentile(x, p))
    null = nc.analytic_null_mean(N_A, len(ib))
    disjoint = bool(q(cm, 97.5) < REF_LO or q(cm, 2.5) > REF_HI)

    res = {"contrast": "length_effectmatched", "beta": float(beta),
           "cliffs_delta_length": float(delta), "n_a": int(len(ia)), "n_b": int(len(ib)),
           "mean_rcs": float(v.mean()), "gini": float(nc.gini(v)),
           "ratio_obs_over_null": float(v.mean() / null),
           "peak_layer": peak, "peak_q2.5": q(pk, 2.5), "peak_q97.5": q(pk, 97.5),
           "com_depth": com, "com_q2.5": q(cm, 2.5), "com_q97.5": q(cm, 97.5),
           "n_boot": N_BOOT, "seed": SEED, "L": int(L),
           "refusal_com": REF_COM, "refusal_com_q2.5": REF_LO, "refusal_com_q97.5": REF_HI,
           "com_disjoint_from_refusal": disjoint,
           "note": ("the eighth benign contrast's depth interval, added so the "
                    "all-eight disjointness claim is measured rather than assumed")}
    print(f"\nlength_effectmatched  {res['ratio_obs_over_null']:.1f}x null  "
          f"gini={res['gini']:.4f}  peak=L{peak}[{res['peak_q2.5']:.0f},{res['peak_q97.5']:.0f}]  "
          f"com={com:.3f}[{res['com_q2.5']:.3f},{res['com_q97.5']:.3f}]")
    print(f"refusal com {REF_COM:.3f}[{REF_LO:.3f},{REF_HI:.3f}]  ->  "
          f"{'DISJOINT' if disjoint else 'OVERLAPS'}")
    with open(os.path.join(OUT, "length_matched_depth_qwen2-7b.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}/length_matched_depth_qwen2-7b.json")


if __name__ == "__main__":
    main()
