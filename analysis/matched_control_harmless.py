#!/usr/bin/env python3
"""Harmless-vs-harmless control: is the RCS profile specific to harmfulness?

The permutation null (qp6c W1) destroys the labels by relabeling the pooled set. It
answers "could this concentration arise by chance?" but not "is this signal about
refusal, or about any two prompt groups?" This control answers the second question
using real, disjoint prompts instead of shuffled labels.

Design -- everything matched to the published analysis except the one thing under test:
  class A : 260 prompts drawn from the harmless HOLDOUT (indices 6000-13000)
  class B : the SAME 6000 harmless prompts the published analysis used
  so n_A = 260 and n_B = 6000 exactly as published, same reference class, same model,
  same extraction. The ONLY change is that class A is harmless rather than harmful.

If the published profile reflects refusal geometry, this control should collapse to the
permutation null. If it stays elevated, the profile partly reflects arbitrary group
structure and the paper's attribution is too strong.

Writes results/matched_controls/harmless_control_qwen2-7b.{json,csv}
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import null_core as nc

EPS = 1e-8
MODEL = "qwen2-7b"
N_DRAW = 200
SEED = 42


def rcs_fast(A, muB, L):
    """RCS(l) = ||mu_A(l) - mu_B(l)|| / (mean_i ||A_i(l) - mu_A(l)|| + eps).

    Matches lama_rcs.compute_rcs; validated against it below before use.
    """
    out = np.zeros(L)
    for l in range(L):
        Al = A[:, l, :]
        mu = Al.mean(axis=0)
        gap = np.linalg.norm(mu - muB[l])
        spread = np.linalg.norm(Al - mu, axis=1).mean()
        out[l] = gap / (spread + EPS)
    return out


def main():
    import torch
    base = os.path.join(REPO, "dataset", "representations")
    HL = torch.load(os.path.join(base, MODEL + "_processed_confirm", "HLx_train.pt"),
                    weights_only=True).float().numpy()
    HF = torch.load(os.path.join(base, MODEL + "_processed_confirm", "HFx_train.pt"),
                    weights_only=True).float().numpy()
    HO = torch.load(os.path.join(base, MODEL + "_harmless_holdout", "HLholdout_train.pt"),
                    weights_only=True).float().numpy()
    for nm, X in (("HL", HL), ("HF", HF), ("HO", HO)):
        X_nz = X[np.abs(X).sum(axis=(1, 2)) > 0]
        assert X_nz.shape[0] == X.shape[0], f"{nm} has all-zero rows"
    n_f, L, d = HF.shape
    n_l = HL.shape[0]
    print(f"HF={HF.shape} HL={HL.shape} holdout={HO.shape}")

    muB = np.stack([HL[:, l, :].mean(axis=0) for l in range(L)])

    # fidelity check: fast path must agree with the repo's own compute_rcs
    from lama_rcs import compute_rcs
    ref_d, _ = compute_rcs(HF, HL)
    ref = np.array([ref_d[l] for l in range(L)])
    mine = rcs_fast(HF, muB, L)
    dev = float(np.max(np.abs(ref - mine)))
    print(f"fast-path vs lama_rcs.compute_rcs: max|diff|={dev:.3e}")
    assert dev < 1e-4, "fast RCS path disagrees with the repo implementation"

    # observed (harmful) profile and its statistics
    obs = ref
    o = nc.profile_stats(obs)

    # control draws
    rng = np.random.RandomState(SEED)
    prof = np.zeros((N_DRAW, L))
    for k in range(N_DRAW):
        idx = rng.choice(HO.shape[0], size=n_f, replace=False)
        prof[k] = rcs_fast(HO[idx], muB, L)
        if (k + 1) % 50 == 0:
            print(f"  draw {k+1}/{N_DRAW}", flush=True)

    stat = lambda f: np.array([f(r) for r in prof], dtype=float)
    res = {
        "model": MODEL, "n_A": int(n_f), "n_B": int(n_l), "L": int(L), "d": int(d),
        "n_draws": N_DRAW, "seed": SEED,
        "design": "class A = 260 harmless holdout prompts; class B = the published 6000 "
                  "harmless prompts; sizes and reference class matched to the paper",
        "observed_harmful": o,
        "analytic_permutation_null_mean_rcs": nc.analytic_null_mean(n_f, n_l),
        "control_per_layer_rcs": {
            "mean": float(prof.mean()), "std": float(prof.std()),
            "q2.5": float(np.percentile(prof, 2.5)),
            "q97.5": float(np.percentile(prof, 97.5))},
        "control_profile_stats": {}, "separation": {},
    }
    for name, fn in [("gini", nc.gini), ("n80_over_L", lambda v: nc.n80(v) / L),
                     ("peak_over_mean", nc.peak_over_mean), ("cv", nc.cv)]:
        v = stat(fn)
        res["control_profile_stats"][name] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "q2.5": float(np.percentile(v, 2.5)), "q50": float(np.percentile(v, 50)),
            "q97.5": float(np.percentile(v, 97.5))}
    res["separation"] = {
        "obs_mean_rcs_over_control_mean": float(o["mean_rcs"] / res["control_per_layer_rcs"]["mean"]),
        "gini_ge_observed": float(np.mean(stat(nc.gini) >= o["gini"])),
        "n80_over_L_le_observed": float(np.mean(stat(lambda v: nc.n80(v) / L) <= o["n80_over_L"])),
        "draws_with_mean_rcs_ge_observed": float(np.mean(prof.mean(axis=1) >= o["mean_rcs"])),
    }

    outdir = os.path.join(REPO, "results", "matched_controls")
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, f"harmless_control_{MODEL}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(outdir, f"harmless_control_profiles_{MODEL}.npz"),
                        profiles=prof, observed=obs)
    with open(os.path.join(outdir, f"harmless_control_{MODEL}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "observed_harmful", "harmless_control_mean",
                    "harmless_control_q2.5", "harmless_control_q97.5", "tail_p"])
        w.writerow(["per_layer_rcs_mean", round(o["mean_rcs"], 5),
                    round(res["control_per_layer_rcs"]["mean"], 5),
                    round(res["control_per_layer_rcs"]["q2.5"], 5),
                    round(res["control_per_layer_rcs"]["q97.5"], 5),
                    res["separation"]["draws_with_mean_rcs_ge_observed"]])
        for k in ["gini", "n80_over_L", "peak_over_mean", "cv"]:
            c = res["control_profile_stats"][k]
            tp = {"gini": "gini_ge_observed",
                  "n80_over_L": "n80_over_L_le_observed"}.get(k, "")
            w.writerow([k, round(o[k], 5), round(c["mean"], 5),
                        round(c["q2.5"], 5), round(c["q97.5"], 5),
                        res["separation"].get(tp, "")])

    print("\n" + "=" * 72)
    print("HARMLESS-VS-HARMLESS CONTROL (matched n=260 vs 6000)")
    print("=" * 72)
    print(f"observed harmful mean RCS   {o['mean_rcs']:.4f}   peak layer {int(np.argmax(obs))}")
    print(f"harmless control mean RCS   {res['control_per_layer_rcs']['mean']:.4f}"
          f"  [{res['control_per_layer_rcs']['q2.5']:.4f},"
          f" {res['control_per_layer_rcs']['q97.5']:.4f}]")
    print(f"analytic permutation null   {res['analytic_permutation_null_mean_rcs']:.4f}")
    print(f"observed / control          {res['separation']['obs_mean_rcs_over_control_mean']:.1f}x")
    print(f"\nGini      observed {o['gini']:.4f}   control {res['control_profile_stats']['gini']['mean']:.4f}"
          f" [{res['control_profile_stats']['gini']['q2.5']:.4f},"
          f"{res['control_profile_stats']['gini']['q97.5']:.4f}]"
          f"   tail p {res['separation']['gini_ge_observed']:.3f}")
    print(f"n80/L     observed {o['n80_over_L']:.4f}   control {res['control_profile_stats']['n80_over_L']['mean']:.4f}"
          f"   tail p {res['separation']['n80_over_L_le_observed']:.3f}")
    print(f"\nwrote {outdir}/harmless_control_{MODEL}.{{json,csv,npz}}")


if __name__ == "__main__":
    main()
