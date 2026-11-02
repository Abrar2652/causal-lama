#!/usr/bin/env python3
"""Permuted-label RCS null on the paper's OWN cached activations (reviewer qp6c, W1).

This is the real test. It requires dataset/representations/<model>/{HFx,HLx}_train.pt, which
are absent from this repository -- regenerate them first, see
docs/permutation_null.md. The
script is implemented and unit-tested (test_permutation_null.py) so it can be run unchanged
the moment those tensors exist.

Procedure
  1. Load the real tensors and recompute the observed RCS profile with the paper's own
     lama_rcs.compute_rcs, then CHECK it against logs/rcs_analysis_<model>.json and report the
     deviation rather than assuming agreement.
  2. Draw --n_perm relabelings ONCE, preserving class counts, and apply each at every layer
     (a prompt has a representation at every depth, so the profile-level permutation test must
     relabel prompts, not layers).
  3. For each relabeling recompute BOTH the centroid gap and the pseudo-harmful intra-class
     spread. The original harmful variance is never frozen.
  4. Report the null distribution of per-layer RCS and of Gini, n80/L, peak/mean and CV, with
     median and 2.5-97.5% quantiles, plus where the observed profile falls relative to it.

Exports the same artifact names as the calibrated run so the two are directly comparable.
No model is loaded, no text is generated, and no prompt content is written to any output.

Usage:
  python3 analysis/permutation_null_empirical.py --model qwen2-7b --n_perm 200
"""
import argparse, json, os, sys, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import null_core as nc


def load_reps(model, reps_dir=None):
    import torch
    base = reps_dir
    if base is None:
        # Prefer the validated extraction. The bare <model> directory is NOT used as a
        # default any more: on Turing GPUs the repo's own extraction path writes silently
        # corrupt tensors there (see the analysis/, results/ and docs/ trees/CRITICAL_sdpa_turing.md), so an
        # explicit, known-good directory is safer than whatever happens to be in place.
        for cand in (model + "_processed_confirm", model):
            d = os.path.join(REPO, "dataset", "representations", cand)
            if os.path.exists(os.path.join(d, "HFx_train.pt")):
                base = d
                break
        else:
            base = os.path.join(REPO, "dataset", "representations", model)
    hf_p, hl_p = os.path.join(base, "HFx_train.pt"), os.path.join(base, "HLx_train.pt")
    missing = [p for p in (hf_p, hl_p) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "BLOCKED: sample-level features unavailable.\n"
            + "".join(f"  missing: {p}\n" for p in missing)
            + "  Regenerate with:  python3 create_representation_dataset.py "
              f"--model_name {model} --device cuda:0\n"
            + "  See docs/permutation_null.md for how to regenerate them."
        )
    HF = torch.load(hf_p, weights_only=True).float().numpy()
    HL = torch.load(hl_p, weights_only=True).float().numpy()
    return HF, HL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model alias, e.g. qwen2-7b")
    ap.add_argument("--n_perm", type=int, default=200, help="number of relabelings (>=200 recommended)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max allowed |recomputed RCS - cached RCS| before flagging a mismatch")
    ap.add_argument("--reps_dir", default=None,
                    help="override the directory holding HFx_train.pt / HLx_train.pt (for testing)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "results", "permutation_null"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"empirical_{a.model}"

    HF, HL = load_reps(a.model, a.reps_dir)
    n_f, L, d = HF.shape
    n_l = HL.shape[0]
    print(f"loaded HF={HF.shape} HL={HL.shape}")

    # ---- step 1: reproduce the published profile before trusting anything else
    from lama_rcs import compute_rcs
    rcs_d, _ = compute_rcs(HF, HL)
    obs = np.array([rcs_d[l] for l in range(L)])
    cache = os.path.join(REPO, "logs", f"rcs_analysis_{a.model}.json")
    cached_dev = None
    if os.path.exists(cache):
        cd = json.load(open(cache))
        if len(cd["rcs_scores"]) == L:
            cached = np.array([cd["rcs_scores"][str(l)] for l in range(L)])
            cached_dev = float(np.max(np.abs(cached - obs)))
            status = "MATCHES cache" if cached_dev <= a.tol else "DIFFERS FROM CACHE"
            print(f"observed profile vs logs/rcs_analysis_{a.model}.json: "
                  f"max|diff|={cached_dev:.3e}  -> {status}")
            if cached_dev > a.tol:
                print("  WARNING: the regenerated activations do not reproduce the published profile.")
                print("  Report this deviation; do not present the null as a check on the published table.")
        else:
            print(f"  cache has {len(cd['rcs_scores'])} layers, tensors have {L}; not comparable")
    else:
        print("  no cached profile to compare against")

    # ---- step 2-3: shared-label permutation test
    perms = nc.make_label_perms(n_f + n_l, n_f, a.n_perm, seed=a.seed)
    null = np.zeros((a.n_perm, L))
    t0 = time.time()
    for l in range(L):
        pooled = np.concatenate([HF[:, l, :], HL[:, l, :]], axis=0)
        null[:, l] = nc.permuted_rcs_for_layer_fast(pooled, n_f, perms)
        print(f"  layer {l+1}/{L}", end="\r", flush=True)
    dt = time.time() - t0
    print(f"\n{a.n_perm} relabelings x {L} layers in {dt:.1f}s")

    # ---- step 4: summarize
    stat = lambda f: np.array([f(r) for r in null], dtype=float)
    o = nc.profile_stats(obs)
    res = {
        "model": a.model, "n_f": n_f, "n_l": n_l, "L": L, "d": d, "n_perm": a.n_perm,
        "seed": a.seed, "elapsed_s": round(dt, 1),
        "source": "EMPIRICAL -- the paper's own cached activations",
        "observed": o,
        "observed_vs_cached_max_abs_dev": cached_dev,
        "analytic_null_mean": nc.analytic_null_mean(n_f, n_l),
        "null_per_layer_rcs": {"mean": float(null.mean()), "std": float(null.std()),
                               "q2.5": float(np.percentile(null, 2.5)),
                               "q97.5": float(np.percentile(null, 97.5))},
        "null_profile_stats": {}, "tail_prob_observed_vs_null": {},
    }
    for name, fn in [("gini", nc.gini), ("n80_over_L", lambda v: nc.n80(v)/L),
                     ("peak_over_mean", nc.peak_over_mean), ("cv", nc.cv)]:
        v = stat(fn)
        res["null_profile_stats"][name] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "q2.5": float(np.percentile(v, 2.5)), "q50": float(np.percentile(v, 50)),
            "q97.5": float(np.percentile(v, 97.5))}
    res["tail_prob_observed_vs_null"] = {
        "gini_ge_observed": float(np.mean(stat(nc.gini) >= o["gini"])),
        "n80_over_L_le_observed": float(np.mean(stat(lambda v: nc.n80(v)/L) <= o["n80_over_L"])),
        "peak_over_mean_ge_observed": float(np.mean(stat(nc.peak_over_mean) >= o["peak_over_mean"])),
    }
    json.dump(res, open(os.path.join(a.outdir, f"results_{tag}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.outdir, f"permutation_distribution_{tag}.npz"),
                        **{a.model: null})
    with open(os.path.join(a.outdir, f"summary_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic", "observed", "null_mean", "null_q2.5", "null_q97.5", "tail_p"])
        w.writerow(["per_layer_rcs_mean", round(o["mean_rcs"], 5),
                    round(res["null_per_layer_rcs"]["mean"], 5),
                    round(res["null_per_layer_rcs"]["q2.5"], 5),
                    round(res["null_per_layer_rcs"]["q97.5"], 5), ""])
        for k in ["gini", "n80_over_L", "peak_over_mean", "cv"]:
            n = res["null_profile_stats"][k]
            tp = {"gini": "gini_ge_observed", "n80_over_L": "n80_over_L_le_observed",
                  "peak_over_mean": "peak_over_mean_ge_observed"}.get(k, "")
            w.writerow([k, round(o[k if k != "n80_over_L" else "n80_over_L"], 5),
                        round(n["mean"], 5), round(n["q2.5"], 5), round(n["q97.5"], 5),
                        res["tail_prob_observed_vs_null"].get(tp, "")])

    print(f"\nobserved mean RCS {o['mean_rcs']:.4f}   null {res['null_per_layer_rcs']['mean']:.5f}"
          f"   ratio {o['mean_rcs']/res['null_per_layer_rcs']['mean']:.1f}x")
    print(f"Gini     observed {o['gini']:.4f}   null {res['null_profile_stats']['gini']['mean']:.4f}"
          f" [{res['null_profile_stats']['gini']['q2.5']:.4f},"
          f"{res['null_profile_stats']['gini']['q97.5']:.4f}]"
          f"   tail p {res['tail_prob_observed_vs_null']['gini_ge_observed']:.3f}")
    print(f"n80/L    observed {o['n80_over_L']:.3f}   null "
          f"{res['null_profile_stats']['n80_over_L']['mean']:.3f}"
          f"   tail p {res['tail_prob_observed_vs_null']['n80_over_L_le_observed']:.3f}")
    print("\nThis tests whether ANY harmful-vs-harmless signal exists, not whether it is")
    print("refusal-specific. See docs/matched_controls.md.")
    print("wrote", os.path.join(a.outdir, f"results_{tag}.json"))


if __name__ == "__main__":
    main()
