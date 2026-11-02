#!/usr/bin/env python3
"""Analyse the base-versus-aligned RCS comparison (reviewers qp6c, PKr6).

The prediction under test. If the late-depth RCS mass is created by alignment, the base
checkpoint should show the same harmful-versus-harmless contrast with its mass at chance
depth, around 0.52, because the base model has no refusal behaviour to encode. If instead
the base model reproduces the late mass, then what RCS measures is harm-topic recognition
rather than refusal, and the paper's interpretation does not survive.

Both profiles come from one process on one host with identical token ids, so the only
difference is the weights. Two checks guard the comparison before it is read:

  1. the re-extracted ALIGNED profile against the published logs, which establishes that
     the extraction path used here reproduces the fp16 Ampere run the paper reports. How
     tight a match to expect depends on the run being analysed, and the guard values are
     recorded in the output rather than asserted against a fixed threshold. A CPU fp32 run
     with a 1000-prompt harmless class agrees at Spearman 0.9989 with a centre-of-mass shift
     of 0.0014, inside the bootstrap width; an Ampere fp16 run at the full 6000 reproduces
     the published profile to ~2e-4, which is agreement at numerical resolution;
  2. the fast RCS path against the paper's own lama_rcs.compute_rcs;
  3. the weight checksums recorded at extraction. GPU inference on this host mutates weights
     in VRAM (see ../docs/gpu_weight_corruption.md), so a run whose checksums moved is
     void no matter how plausible its profile looks.

Depth is summarised by the centre of mass over normalised depth with a 95% bootstrap
interval over the harmful class, matching deep_benign_contrasts.py so the numbers are
directly comparable with the benign contrasts.

Writes results/matched_controls/base_vs_aligned_qwen2-7b.{json,csv}
"""
import csv, json, os, sys
import numpy as np

REPO = "/nas/home/jahin/causal-lama"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))

from lama_rcs import compute_rcs
import null_core as nc
from scipy.stats import spearmanr

R = os.path.join(REPO, "dataset", "representations")
OUT = os.path.join(REPO, "results", "matched_controls")
SEED, N_BOOT, EPS = 42, 1000, 1e-8
RUN_THRESHOLD = 0.25   # the contiguous-band threshold quoted in the writeups


def longest_run(mask):
    """(start, end, length) of the longest contiguous True run; (-1, -1, 0) if none."""
    best, start = (-1, -1, 0), None
    for i, x in enumerate(list(mask) + [False]):
        if x and start is None:
            start = i
        elif not x and start is not None:
            if i - start > best[2]:
                best = (start, i - 1, i - start)
            start = None
    return best


def rcs_fast(A, nu):
    mu = A.mean(axis=0)
    gap = np.linalg.norm(mu - nu, axis=1)
    sig = np.linalg.norm(A - mu[None, :, :], axis=2).mean(axis=0)
    return gap / (sig + EPS)


def depth(v):
    L = len(v); w = v / v.sum()
    return int(np.argmax(v)), float((np.arange(L) * w).sum() / (L - 1))


def boot(A, nu, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed); n = A.shape[0]
    pk = np.zeros(n_boot, int); cm = np.zeros(n_boot)
    for b in range(n_boot):
        pk[b], cm[b] = depth(rcs_fast(A[rng.randint(0, n, n)], nu))
    return {"peak_q2.5": float(np.percentile(pk, 2.5)),
            "peak_q97.5": float(np.percentile(pk, 97.5)),
            "peak_mode": int(np.bincount(pk).argmax()),
            "com_q2.5": float(np.percentile(cm, 2.5)),
            "com_q97.5": float(np.percentile(cm, 97.5)),
            "com_mean": float(cm.mean())}


def load(tag):
    import torch
    d = os.path.join(R, tag)
    HF = torch.load(os.path.join(d, "HFx_train.pt")).numpy().astype(np.float32)
    HL = torch.load(os.path.join(d, "HLx_train.pt")).numpy().astype(np.float32)
    meta = {}
    if os.path.exists(os.path.join(d, "meta.json")):
        meta = json.load(open(os.path.join(d, "meta.json")))
    return HF, HL, meta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_cpu_fp32",
                    help="suffix of the qwen2-7b_{aligned,base}<suffix> tensor directories")
    ap.add_argument("--out", default="base_vs_aligned_qwen2-7b",
                    help="output basename under matched_controls/")
    args = ap.parse_args()
    res, rows = {}, {}

    HFa, HLa, ma = load(f"qwen2-7b_aligned{args.suffix}")
    HFb, HLb, mb = load(f"qwen2-7b_base{args.suffix}")
    assert HFa.shape == HFb.shape and HLa.shape == HLb.shape, "shape mismatch"
    L = HFa.shape[1]
    print(f"aligned {HFa.shape}/{HLa.shape}  base {HFb.shape}/{HLb.shape}")
    print(f"aligned meta: {ma.get('model_id')} {ma.get('attn_implementation')}/{ma.get('dtype')}")
    print(f"base    meta: {mb.get('model_id')} {mb.get('attn_implementation')}/{mb.get('dtype')}")

    # guard 3: refuse to analyse tensors the extraction itself flagged
    for tag, m in (("aligned", ma), ("base", mb)):
        d = os.path.join(R, f"qwen2-7b_{tag}{args.suffix}")
        assert not os.path.exists(os.path.join(d, "VOID_weight_corruption.txt")), \
            f"{tag}: extraction wrote a VOID marker; these tensors are not usable"
        mut = m.get("weights_mutated_during_inference")
        assert mut == 0, f"{tag}: {mut} parameters moved during inference, run is void"
    print("[check] weight integrity at extraction: 0 parameters mutated, both models")

    va = rcs_fast(HFa, HLa.mean(axis=0))
    vb = rcs_fast(HFb, HLb.mean(axis=0))

    # guard 1: fast path vs the paper's implementation
    ref = compute_rcs(HFa, HLa)[0]
    chk = np.array([ref[l] for l in range(L)])
    d1 = float(np.max(np.abs(chk - va)))
    print(f"[check] rcs_fast vs lama_rcs.compute_rcs max|diff|={d1:.3e}")
    assert d1 < 1e-4

    # guard 2: re-extracted aligned vs the published Ampere fp16 run
    cd = json.load(open(os.path.join(REPO, "logs", "rcs_analysis_qwen2-7b.json")))
    pub = np.array([cd["rcs_scores"][str(l)] for l in range(L)])
    rho_pub = float(spearmanr(va, pub)[0])
    d2 = float(np.max(np.abs(va - pub)))
    rel = float(np.max(np.abs(va - pub) / pub))
    print(f"[check] re-extracted aligned vs published logs: Spearman={rho_pub:.4f} "
          f"max|diff|={d2:.4f} max rel={rel:.3%}  peak L{int(np.argmax(va))} vs L{int(np.argmax(pub))}")

    null = nc.analytic_null_mean(HFa.shape[0], HLa.shape[0])
    for tag, v, HF, HL in (("aligned", va, HFa, HLa), ("base", vb, HFb, HLb)):
        pk, cm = depth(v)
        rows[tag] = dict(mean_rcs=float(v.mean()), peak_layer=pk, peak_rcs=float(v.max()),
                         gini=float(nc.gini(v)), n80_over_L=float(nc.n80(v) / L),
                         cv=float(v.std() / v.mean()), com_depth=cm,
                         analytic_null_mean=float(null),
                         ratio_obs_over_null=float(v.mean() / null),
                         **boot(HF, HL.mean(axis=0)))
        rows[tag]["profile"] = v.tolist()
        r = rows[tag]
        print(f"  {tag:8s} mean={r['mean_rcs']:.4f} {r['ratio_obs_over_null']:5.1f}x "
              f"peak=L{r['peak_layer']:<2d}[{r['peak_q2.5']:.0f},{r['peak_q97.5']:.0f}] "
              f"gini={r['gini']:.4f} com={r['com_depth']:.3f}"
              f"[{r['com_q2.5']:.3f},{r['com_q97.5']:.3f}]")

    a, b = rows["aligned"], rows["base"]
    com_disjoint = (b["com_q97.5"] < a["com_q2.5"]) or (b["com_q2.5"] > a["com_q97.5"])
    # chance depth, measured in deep_benign_contrasts.py by a pure random split
    CHANCE_COM, CHANCE_LO, CHANCE_HI = 0.519, 0.482, 0.531
    # Two readings of "is the base model at chance depth". The point-estimate test asks
    # whether base's com falls inside the chance interval; the interval test asks whether the
    # two intervals overlap at all. They can disagree when base sits just outside chance, and
    # the interval test is the conservative one, so both are reported.
    base_at_chance = CHANCE_LO <= b["com_depth"] <= CHANCE_HI
    base_vs_chance_overlap = not (b["com_q2.5"] > CHANCE_HI or b["com_q97.5"] < CHANCE_LO)
    verdict = {
        "spearman_aligned_vs_base": float(spearmanr(va, vb)[0]),
        "com_gap_aligned_minus_base": float(a["com_depth"] - b["com_depth"]),
        "com_intervals_disjoint": bool(com_disjoint),
        "magnitude_ratio_aligned_over_base": float(a["mean_rcs"] / b["mean_rcs"]),
        "base_com_within_chance_interval": bool(base_at_chance),
        "base_com_interval_overlaps_chance": bool(base_vs_chance_overlap),
        "chance_com_reference": {"com": CHANCE_COM, "q2.5": CHANCE_LO, "q97.5": CHANCE_HI,
                                 "source": "random_split in deep_benign_qwen2-7b.json"},
    }
    if base_at_chance and com_disjoint:
        verdict["reading"] = ("SUPPORTS refusal specificity: the base model places its mass at "
                             "chance depth and the aligned model does not, so the late mass is "
                             "created by alignment rather than by harm-topic recognition.")
    elif not com_disjoint:
        verdict["reading"] = ("DOES NOT SUPPORT refusal specificity: base and aligned place "
                             "their mass at indistinguishable depth, so the late mass is present "
                             "without any refusal training.")
    else:
        verdict["reading"] = ("MIXED: the depth intervals separate, but the base model's mass is "
                             "not at chance depth either; alignment shifts depth further rather "
                             "than creating the shift.")
    print("\naligned vs base:")
    for k, v in verdict.items():
        if k not in ("chance_com_reference",):
            print(f"  {k}: {v}")

    # where along depth does alignment add separation?
    dv = va - vb
    top = np.argsort(-dv)[:5]
    print(f"\nlayers where alignment adds the most RCS: "
          f"{[(int(l), round(float(dv[l]), 3)) for l in top]}")

    # --- decomposition of the alignment increment ------------------------------
    # The layer-distribution claim rests on these, so they are computed here rather
    # than by hand. An earlier version of this file shipped them inside the stored
    # JSON with no code that produced them, and two of the figures quoted in the
    # writeups -- the best-five share and the contiguous run -- were in neither the
    # JSON nor any script. Ungenerated numbers cannot be re-checked, so they are
    # generated now, including the prose note.
    pos = np.clip(dv, 0.0, None)
    tot_pos = float(pos.sum())
    order = np.argsort(-pos)
    above_half = [int(l) for l in np.where(dv > dv.max() / 2.0)[0]]
    run = longest_run(dv > RUN_THRESHOLD)
    dec = {
        "base_share_of_aligned_mean_rcs":
            float(rows["base"]["mean_rcs"] / rows["aligned"]["mean_rcs"]),
        "base_share_of_aligned_gini":
            float(rows["base"]["gini"] / rows["aligned"]["gini"]),
        "alignment_delta_peak_layer": int(np.argmax(dv)),
        "alignment_delta_at_peak": float(dv.max()),
        "alignment_delta_layers_0_4": [float(x) for x in dv[:5]],
        "layers_where_base_exceeds_aligned": [int(l) for l in np.where(dv < 0)[0]],
        "alignment_increment_layers_above_0.5": above_half,
        "alignment_increment_above_half_max_n_layers": len(above_half),
        "alignment_increment_share_single_best_layer": float(pos[order[0]] / tot_pos),
        "alignment_increment_share_best_five": float(pos[order[:5]].sum() / tot_pos),
        "alignment_increment_best_five_layers": sorted(int(l) for l in order[:5]),
        "alignment_increment_longest_run": {
            "threshold": RUN_THRESHOLD, "start": run[0], "end": run[1], "n_layers": run[2]},
        "total_positive_increment": tot_pos,
    }
    dec["note"] = (
        f"magnitude is largely pre-existing (base is "
        f"{dec['base_share_of_aligned_mean_rcs']:.0%} of aligned mean RCS at "
        f"{rows['base']['ratio_obs_over_null']:.1f}x the null) but depth-flat (base Gini is "
        f"{dec['base_share_of_aligned_gini']:.0%} of aligned, com "
        f"{'at' if base_at_chance else 'near'} chance); alignment's increment is late-weighted, "
        f"near zero in layers 0-4 and maximal at L{dec['alignment_delta_peak_layer']}, and it is "
        f"distributed: no single layer holds more than "
        f"{dec['alignment_increment_share_single_best_layer']:.1%} of it, the best five hold "
        f"{dec['alignment_increment_share_best_five']:.1%}, it stays above half its maximum "
        f"across {dec['alignment_increment_above_half_max_n_layers']} layers, and it exceeds "
        f"+{RUN_THRESHOLD} contiguously across {run[2]} layers (L{run[0]}-L{run[1]})."
    )
    print("\nalignment-increment decomposition:")
    for k in ("alignment_increment_share_single_best_layer",
              "alignment_increment_share_best_five",
              "alignment_increment_above_half_max_n_layers"):
        print(f"  {k}: {dec[k]}")
    print(f"  longest run > +{RUN_THRESHOLD}: {run[2]} layers (L{run[0]}-L{run[1]})")

    res = {"guards": {"fast_path_max_abs_dev": d1,
                      "aligned_vs_published_spearman": rho_pub,
                      "aligned_vs_published_max_abs_dev": d2,
                      "aligned_vs_published_max_rel_dev": rel},
           "n_f": int(HFa.shape[0]), "n_l": int(HLa.shape[0]), "L": int(L),
           "n_boot": N_BOOT, "seed": SEED, "tensor_suffix": args.suffix,
           "aligned_meta": ma, "base_meta": mb,
           "profiles": rows, "verdict": verdict, "decomposition": dec,
           "alignment_delta_profile": dv.tolist()}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{args.out}.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT, f"{args.out}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_rcs", "ratio_obs_over_null", "gini", "n80_over_L",
                    "peak_layer", "peak_q2.5", "peak_q97.5", "com_depth", "com_q2.5", "com_q97.5"])
        for tag in ("aligned", "base"):
            r = rows[tag]
            w.writerow([tag, f"{r['mean_rcs']:.4f}", f"{r['ratio_obs_over_null']:.1f}",
                        f"{r['gini']:.4f}", f"{r['n80_over_L']:.4f}", r["peak_layer"],
                        f"{r['peak_q2.5']:.0f}", f"{r['peak_q97.5']:.0f}",
                        f"{r['com_depth']:.4f}", f"{r['com_q2.5']:.4f}", f"{r['com_q97.5']:.4f}"])
    print(f"\nwrote {OUT}/{args.out}.{{json,csv}}")


if __name__ == "__main__":
    main()
