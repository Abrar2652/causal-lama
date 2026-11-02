#!/usr/bin/env python3
"""Permuted-label RCS null for the nine profiled models (reviewer qp6c, W1).

The paper's sample-level residual-stream tensors (dataset/representations/<model>/
{HFx,HLx}_train.pt) are NOT present in this repository, so a permutation test cannot be
run on the paper's own activations. For Qwen2.5-7B that empirical run now exists and agrees
with this one to within 1%; see docs/permutation_null.md.

What this script does instead, and what it is worth:

  (1) ANALYTIC NULL  -- exact, assumption-light. Under a random relabeling, the per-layer
      scale cancels between RCS's numerator and denominator, leaving
          E[RCS_perm] ~= sqrt(1/n_f + 1/n_l) * kappa * sqrt(N/(N-1)),  kappa >= 1,
      a quantity independent of the layer, the model and the representation geometry.
      For the n_f=260, n_l=6000 split actually used this is 0.0634 (see
      the analysis/, results/ and docs/ trees/CRITICAL_data_source.md); the paper's stated 4000/6000 would give
      0.0204. A flat null implies Gini ~ 0 and n80/L ~ 0.8.

  (2) CALIBRATED SIMULATION -- supplies the null's SAMPLING SPREAD, which the analytic
      argument does not give. For each layer we synthesize pooled data at the paper's real
      (n_f, n_l, d) whose TRUE-label RCS equals the model's measured RCS at that layer
      (planting the measured centroid gap and the measured harmful spread, both read from
      logs/rcs_analysis_<model>.json), then run the shared-label permutation test on it.

  This is a simulation calibrated to measured summary statistics, NOT a measurement of the
  paper's activations. Its one real assumption is the within-class geometry (Gaussian with
  a chosen eigenvalue spectrum); --eff_rank sweeps that assumption, since the null's spread
  -- unlike its mean -- depends on the effective rank.
"""
import argparse, json, os, sys, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import null_core as nc

HIDDEN_DIM = {            # hidden size per model (README / HF configs)
    "qwen2-7b": 3584, "qwen2-3b": 2048, "llama2-7b": 4096, "llama31-8b": 4096,
    "gemma2-9b": 3584, "mistral-7b-rr": 4096, "deepseek-r1-qwen-7b": 3584,
    "ministral-8b": 4096, "gpt-oss-20b": 2880,
}
PAPER_NAME = {
    "qwen2-7b":"Qwen2.5-7B","qwen2-3b":"Qwen2.5-3B","llama2-7b":"Llama-2-7B",
    "llama31-8b":"Llama-3.1-8B","gemma2-9b":"Gemma2-9B","mistral-7b-rr":"Mistral-7B-RR",
    "deepseek-r1-qwen-7b":"DeepSeek-R1-7B","ministral-8b":"Ministral-8B","gpt-oss-20b":"GPT-oss-20B",
}
ALL = list(HIDDEN_DIM)


def spectrum(d, eff_rank, dtype=np.float32):
    """Per-coordinate std whose COVARIANCE has participation ratio ~= eff_rank.

    The returned array multiplies standard normals, so covariance eigenvalues are
    scale**2. We fit a power law lam_j ~ j**-a to hit a target participation ratio
    PR(lam) = (sum lam)**2 / sum lam**2, then return sqrt(lam) normalized so the mean
    coordinate variance is 1. eff_rank >= d gives the isotropic case.

    Returns (scale, achieved_participation_ratio).
    """
    if eff_rank is None or eff_rank >= d:
        return np.ones(d, dtype=dtype), float(d)
    j = np.arange(1, d + 1, dtype=np.float64)
    pr = lambda a: (np.sum(j ** -a) ** 2) / np.sum(j ** (-2.0 * a))
    lo, hi = 1e-6, 8.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pr(mid) > eff_rank: lo = mid
        else: hi = mid
    a = 0.5 * (lo + hi)
    lam = j ** -a
    lam = lam / lam.mean()                 # mean coordinate variance = 1
    return np.sqrt(lam).astype(dtype), float(pr(a))


def run_model(alias, n_f, n_l, n_perm, eff_rank, rho, seed, verbose=True):
    """Calibrated permutation null for one model. Returns dict of results."""
    cache = os.path.join(REPO, "logs", f"rcs_analysis_{alias}.json")
    d_full = json.load(open(cache))
    layers = sorted(int(k) for k in d_full["rcs_scores"])
    obs_rcs = np.array([d_full["rcs_scores"][str(l)] for l in layers])
    meta = d_full["rcs_metadata"]
    gaps  = np.array([meta[str(l)]["centroid_distance"] for l in layers])
    spreads = np.array([meta[str(l)]["harmful_intra_std"] for l in layers])
    L = len(layers)
    d = HIDDEN_DIM[alias]
    N = n_f + n_l

    perms = nc.make_label_perms(N, n_f, n_perm, seed=seed)
    scale, achieved_pr = spectrum(d, eff_rank)
    rng = np.random.default_rng(seed)
    # shared per-sample latent gives cross-layer dependence (same prompt at every depth)
    z = rng.standard_normal((N, d), dtype=np.float32) if rho > 0 else None

    null = np.zeros((n_perm, L), dtype=float)
    recon = np.zeros(L)
    t0 = time.time()
    for i, l in enumerate(layers):
        e = rng.standard_normal((N, d), dtype=np.float32)
        X = (np.float32(np.sqrt(rho)) * z + np.float32(np.sqrt(1.0 - rho)) * e) if rho > 0 else e
        X = X * scale
        # calibrate: harmful-class mean radius == measured spread at this layer
        A0 = X[:n_f]
        r = float(np.linalg.norm(A0 - A0.mean(axis=0), axis=1).mean())
        X *= (spreads[i] / r)
        X[:n_f, 0] += gaps[i]           # plant the measured centroid gap
        recon[i] = nc.rcs_layer(X[:n_f], X[n_f:])
        null[:, i] = nc.permuted_rcs_for_layer_fast(X, n_f, perms)
        del X, e
    dt = time.time() - t0

    stat = lambda f: np.array([f(row) for row in null])
    res = {
        "alias": alias, "model": PAPER_NAME[alias], "L": L, "d": d,
        "n_f": n_f, "n_l": n_l, "n_perm": n_perm,
        "eff_rank": ("isotropic" if (eff_rank is None or eff_rank >= d) else eff_rank),
        "achieved_participation_ratio": round(achieved_pr, 1),
        "rho_cross_layer": rho, "seed": seed, "elapsed_s": round(dt, 1),
        "analytic_null_mean": nc.analytic_null_mean(n_f, n_l),
        "observed": nc.profile_stats(obs_rcs),
        "calibration_max_abs_rcs_err": float(np.max(np.abs(recon - obs_rcs))),
        "null_per_layer_rcs": {
            "mean": float(null.mean()), "std": float(null.std()),
            "q2.5": float(np.percentile(null, 2.5)), "q97.5": float(np.percentile(null, 97.5)),
            "min": float(null.min()), "max": float(null.max()),
        },
        "null_profile_stats": {},
    }
    for name, fn in [("gini", nc.gini), ("n80", nc.n80), ("peak_over_mean", nc.peak_over_mean), ("cv", nc.cv)]:
        v = stat(fn).astype(float)
        if name == "n80":
            v = v / L; name = "n80_over_L"
        res["null_profile_stats"][name] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "q2.5": float(np.percentile(v, 2.5)), "q50": float(np.percentile(v, 50)),
            "q97.5": float(np.percentile(v, 97.5)),
            "min": float(v.min()), "max": float(v.max()),
        }
    # one-sided empirical tail probabilities (descriptive, not pre-registered)
    o = res["observed"]
    res["tail_prob_observed_vs_null"] = {
        "gini_ge_observed": float(np.mean(stat(nc.gini) >= o["gini"])),
        "n80_over_L_le_observed": float(np.mean(stat(nc.n80) / L <= o["n80_over_L"])),
        "peak_over_mean_ge_observed": float(np.mean(stat(nc.peak_over_mean) >= o["peak_over_mean"])),
        "min_layer_rcs_ratio_observed_over_null_q97.5":
            float(obs_rcs.min() / res["null_per_layer_rcs"]["q97.5"]),
    }
    if verbose:
        print(f"  {PAPER_NAME[alias]:<15} L={L:<3} d={d:<5} {dt:6.1f}s  "
              f"calib_err={res['calibration_max_abs_rcs_err']:.2e}  "
              f"null/layer={null.mean():.4f}+-{null.std():.4f}  "
              f"nullGini={res['null_profile_stats']['gini']['mean']:.4f} "
              f"(obs {o['gini']:.4f})  nullN80/L={res['null_profile_stats']['n80_over_L']['mean']:.3f} "
              f"(obs {o['n80_over_L']:.3f})")
    return res, null


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_f", type=int, default=4000, help="harmful count (SORRY-Bench, supplement C)")
    ap.add_argument("--n_l", type=int, default=6000, help="harmless count (Alpaca, supplement C)")
    ap.add_argument("--eff_rank", default="isotropic", help="'isotropic' or an integer effective rank")
    ap.add_argument("--rho", type=float, default=0.9, help="cross-layer latent correlation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--outdir", default=os.path.join(REPO, "results", "permutation_null"))
    a = ap.parse_args()

    models = ALL if a.models == "all" else [m.strip() for m in a.models.split(",")]
    eff = None if a.eff_rank == "isotropic" else int(a.eff_rank)
    os.makedirs(a.outdir, exist_ok=True)

    print(f"analytic null mean = {nc.analytic_null_mean(a.n_f, a.n_l):.5f}  "
          f"(n_f={a.n_f}, n_l={a.n_l}, n_perm={a.n_perm}, eff_rank={a.eff_rank}, rho={a.rho})")
    results, dists = [], {}
    for m in models:
        r, null = run_model(m, a.n_f, a.n_l, a.n_perm, eff, a.rho, a.seed)
        results.append(r); dists[m] = null

    cfg = {"argv": sys.argv, "models": models, "n_perm": a.n_perm, "n_f": a.n_f, "n_l": a.n_l,
           "eff_rank": a.eff_rank, "rho": a.rho, "seed": a.seed,
           "numpy": np.__version__, "python": sys.version.split()[0],
           "analytic_null_mean": nc.analytic_null_mean(a.n_f, a.n_l),
           "note": "Calibrated simulation, not a measurement of the paper's activations."}
    json.dump(cfg, open(os.path.join(a.outdir, f"config_{a.tag}.json"), "w"), indent=2)
    json.dump(results, open(os.path.join(a.outdir, f"results_{a.tag}.json"), "w"), indent=2)

    rows = []
    for r in results:
        o = r["observed"]; n = r["null_profile_stats"]; pl = r["null_per_layer_rcs"]
        rows.append({
            "model": r["model"], "L": r["L"], "d": r["d"], "eff_rank": r["eff_rank"],
            "achieved_pr": r["achieved_participation_ratio"],
            "obs_rcs_min": round(min(o["peak_rcs"], o["mean_rcs"]), 4) if False else None,
            "obs_rcs_mean": round(o["mean_rcs"], 4), "obs_rcs_peak": round(o["peak_rcs"], 4),
            "null_rcs_mean": round(pl["mean"], 5), "null_rcs_q97.5": round(pl["q97.5"], 5),
            "rcs_ratio_obsmean_over_null": round(o["mean_rcs"] / pl["mean"], 1),
            "obs_gini": round(o["gini"], 4),
            "null_gini_mean": round(n["gini"]["mean"], 4),
            "null_gini_q97.5": round(n["gini"]["q97.5"], 4),
            "gini_tail_p": r["tail_prob_observed_vs_null"]["gini_ge_observed"],
            "obs_n80_over_L": round(o["n80_over_L"], 3),
            "null_n80_over_L_mean": round(n["n80_over_L"]["mean"], 3),
            "null_n80_over_L_q2.5": round(n["n80_over_L"]["q2.5"], 3),
            "n80_tail_p": r["tail_prob_observed_vs_null"]["n80_over_L_le_observed"],
            "obs_peak_over_mean": round(o["peak_over_mean"], 3),
            "null_peak_over_mean_mean": round(n["peak_over_mean"]["mean"], 3),
            "null_peak_over_mean_q97.5": round(n["peak_over_mean"]["q97.5"], 3),
            "calib_max_abs_err": f"{r['calibration_max_abs_rcs_err']:.2e}",
        })
    for r in rows: r.pop("obs_rcs_min", None)
    out_csv = os.path.join(a.outdir, f"summary_{a.tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    np.savez_compressed(os.path.join(a.outdir, f"permutation_distribution_{a.tag}.npz"),
                        **{k: v for k, v in dists.items()})
    print("\nwrote:", out_csv)
    print("       ", os.path.join(a.outdir, f"results_{a.tag}.json"))
    print("       ", os.path.join(a.outdir, f"permutation_distribution_{a.tag}.npz"))


if __name__ == "__main__":
    main()
