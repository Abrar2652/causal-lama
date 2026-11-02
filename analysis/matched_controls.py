#!/usr/bin/env python3
"""Refusal-specificity matched controls (reviewers PKr6, p6Gj-W1, qp6c-W1).

The permutation null answers "is there ANY harmful-vs-harmless signal". It cannot
answer "is that signal about refusal rather than topic, style or length", because
relabeling destroys class structure. These controls attack that question directly,
using the paper's own activations.

Controls, all computed with the repo's own compute_rcs:

  A. length_matched    harmful vs a harmless subset resampled to match the harmful
                       token-length distribution stratum by stratum.
  B. style_matched     harmful vs harmless restricted to imperative-initial prompts,
                       which is the register the harmful prompts are written in.
  C. length_and_style  both constraints at once.
  D. nonrefusal_*      a contrast formed ENTIRELY INSIDE the held-out harmless pool,
                       where no refusal distinction exists by construction. Run by
                       nonrefusal_controls.py once the holdout tensors exist.

If the published profile survives A-C largely unchanged, length and register are not
what RCS is measuring. If a non-refusal contrast of comparable effect size reproduces
the profile, RCS is a generic separability statistic rather than a refusal probe.
"""
import json, os, sys
import numpy as np

REPO = "/nas/home/jahin/causal-lama"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))

from lama_rcs import compute_rcs
import null_core as nc
from scipy.stats import spearmanr

REPS = os.path.join(REPO, "dataset", "representations", "qwen2-7b_processed_confirm")
OUT = os.path.join(REPO, "results", "matched_controls")
SEED = 42

IMPERATIVE_VERBS = {
    "write", "describe", "explain", "create", "give", "list", "generate", "name",
    "suggest", "provide", "develop", "design", "make", "build", "compose", "summarize",
    "classify", "identify", "construct", "produce", "outline", "draft", "compile",
    "rewrite", "convert", "translate", "calculate", "find", "determine", "analyze",
    "compare", "edit", "add", "select", "choose", "arrange", "formulate", "tell",
    "demonstrate", "illustrate", "show", "state", "define", "discuss", "detail",
}


def profile_vec(HF, HL):
    rcs, meta = compute_rcs(HF, HL)
    return np.array([rcs[l] for l in range(len(rcs))]), meta


def summarize(v):
    return {
        "mean_rcs": float(v.mean()),
        "peak_layer": int(np.argmax(v)),
        "peak_rcs": float(v.max()),
        "peak_over_mean": float(v.max() / v.mean()),
        "gini": float(nc.gini(v)),
        "n80": int(nc.n80(v)),
        "n80_over_L": float(nc.n80(v) / len(v)),
        "cv": float(v.std() / v.mean()),
    }


def is_imperative(p):
    w = p.strip().split()
    return bool(w) and w[0].lower().strip(",.:;") in IMPERATIVE_VERBS


def length_matched_indices(len_f, len_l, rng, max_per_stratum=None):
    """Resample harmless indices so their length histogram matches the harmful one.

    For each token length L present in the harmful class with count c_L, draw
    harmless indices of that same length. The draw is scaled by the largest common
    factor k such that every stratum can supply k*c_L distinct harmless prompts,
    which keeps n_l as large as the matching allows without replacement.
    """
    by_len = {}
    for i, L in enumerate(len_l):
        by_len.setdefault(int(L), []).append(i)
    strata = {}
    for L in np.unique(len_f):
        c = int((len_f == L).sum())
        avail = by_len.get(int(L), [])
        strata[int(L)] = (c, avail)
    feasible = [len(a) // c for c, a in strata.values() if c > 0 and len(a) > 0]
    if not feasible:
        return np.array([], dtype=int), 0
    k = max(1, min(feasible))
    if max_per_stratum:
        k = min(k, max_per_stratum)
    idx = []
    for L, (c, avail) in strata.items():
        if not avail:
            continue
        take = min(len(avail), k * c)
        idx.extend(rng.choice(avail, size=take, replace=False).tolist())
    return np.array(sorted(idx), dtype=int), k


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)

    import torch
    HF = torch.load(os.path.join(REPS, "HFx_train.pt")).numpy().astype(np.float32)
    HL = torch.load(os.path.join(REPS, "HLx_train.pt")).numpy().astype(np.float32)
    print(f"loaded HF={HF.shape} HL={HL.shape}")

    from transformers import AutoTokenizer
    from dataset.load_dataset import load_dataset_split
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    p_f = load_dataset_split('harmful', 'train', instructions_only=True)
    p_l = [h for h in load_dataset_split('harmless', 'train', instructions_only=True)
           if h and h.strip()][:HL.shape[0]]
    assert len(p_f) == HF.shape[0] and len(p_l) == HL.shape[0], "prompt/tensor mismatch"

    len_f = np.array([len(tok(p).input_ids) for p in p_f])
    len_l = np.array([len(tok(p).input_ids) for p in p_l])
    imp_l = np.array([is_imperative(p) for p in p_l])
    imp_f = np.array([is_imperative(p) for p in p_f])
    print(f"imperative share: harmful {imp_f.mean():.3f}  harmless {imp_l.mean():.3f}")

    base, _ = profile_vec(HF, HL)
    results = {"published_equivalent": {"n_f": int(HF.shape[0]), "n_l": int(HL.shape[0]),
                                        **summarize(base)}}
    print("\nbaseline (all 6000 harmless): mean=%.4f peak=L%d gini=%.4f"
          % (base.mean(), int(np.argmax(base)), nc.gini(base)))

    controls = {}

    # ---- A. length matched
    idx, k = length_matched_indices(len_f, len_l, rng)
    v, _ = profile_vec(HF, HL[idx])
    controls["length_matched"] = dict(
        n_l=int(len(idx)), k_per_stratum=int(k),
        harmless_len_mean=float(len_l[idx].mean()), harmful_len_mean=float(len_f.mean()),
        **summarize(v))
    controls["length_matched"]["profile"] = v.tolist()

    # ---- B. style matched (imperative register only)
    sidx = np.where(imp_l)[0]
    v, _ = profile_vec(HF, HL[sidx])
    controls["style_matched"] = dict(
        n_l=int(len(sidx)), harmless_len_mean=float(len_l[sidx].mean()),
        harmful_len_mean=float(len_f.mean()), **summarize(v))
    controls["style_matched"]["profile"] = v.tolist()

    # ---- C. length AND style matched
    sub = np.where(imp_l)[0]
    idx2, k2 = length_matched_indices(len_f, len_l[sub], rng)
    idx2 = sub[idx2]
    v, _ = profile_vec(HF, HL[idx2])
    controls["length_and_style_matched"] = dict(
        n_l=int(len(idx2)), k_per_stratum=int(k2),
        harmless_len_mean=float(len_l[idx2].mean()),
        harmful_len_mean=float(len_f.mean()), **summarize(v))
    controls["length_and_style_matched"]["profile"] = v.tolist()

    # ---- D. symmetric register match: imperative-only on BOTH sides.
    # The one-sided style control above restricts only the harmless class, but the
    # harmful class is itself only 56.9% imperative, so register is not fully matched.
    # Constraining both sides is the stricter test.
    fi = np.where(imp_f)[0]
    li = np.where(imp_l)[0]
    v, _ = profile_vec(HF[fi], HL[li])
    controls["style_matched_symmetric"] = dict(
        n_f_override=int(len(fi)), n_l=int(len(li)),
        harmless_len_mean=float(len_l[li].mean()),
        harmful_len_mean=float(len_f[fi].mean()), **summarize(v))
    controls["style_matched_symmetric"]["profile"] = v.tolist()

    # ---- E. symmetric register match AND length match
    idx3, k3 = length_matched_indices(len_f[fi], len_l[li], rng)
    idx3 = li[idx3]
    v, _ = profile_vec(HF[fi], HL[idx3])
    controls["length_and_style_symmetric"] = dict(
        n_f_override=int(len(fi)), n_l=int(len(idx3)), k_per_stratum=int(k3),
        harmless_len_mean=float(len_l[idx3].mean()),
        harmful_len_mean=float(len_f[fi].mean()), **summarize(v))
    controls["length_and_style_symmetric"]["profile"] = v.tolist()

    # ---- agreement with the published profile + null at each control's class sizes
    rows = []
    for name, c in controls.items():
        v = np.array(c["profile"])
        rho = spearmanr(v, base)[0]
        n_f_c = c.get("n_f_override", HF.shape[0])
        c["n_f"] = int(n_f_c)
        null = nc.analytic_null_mean(n_f_c, c["n_l"])
        c["spearman_vs_published"] = float(rho)
        c["max_abs_dev_vs_published"] = float(np.abs(v - base).max())
        c["analytic_null_mean"] = float(null)
        c["ratio_obs_over_null"] = float(v.mean() / null)
        rows.append((name, c))
        print(f"\n{name}: n_f={n_f_c} n_l={c['n_l']}  mean={c['mean_rcs']:.4f}  peak=L{c['peak_layer']}"
              f"  gini={c['gini']:.4f}  rho_vs_published={rho:.4f}"
              f"  null={null:.4f}  ratio={c['ratio_obs_over_null']:.1f}x")

    results["controls"] = controls
    with open(os.path.join(OUT, "matched_controls_qwen2-7b.json"), "w") as f:
        json.dump(results, f, indent=2)

    import csv
    with open(os.path.join(OUT, "matched_controls_qwen2-7b.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["control", "n_f", "n_l", "mean_rcs", "peak_layer", "gini", "n80_over_L",
                    "cv", "spearman_vs_published", "max_abs_dev_vs_published",
                    "analytic_null_mean", "ratio_obs_over_null",
                    "harmful_len_mean", "harmless_len_mean"])
        b = results["published_equivalent"]
        w.writerow(["published_equivalent", HF.shape[0], b["n_l"], f"{b['mean_rcs']:.4f}",
                    b["peak_layer"], f"{b['gini']:.4f}", f"{b['n80_over_L']:.4f}",
                    f"{b['cv']:.4f}", "1.0000", "0.0000",
                    f"{nc.analytic_null_mean(HF.shape[0], b['n_l']):.4f}",
                    f"{b['mean_rcs']/nc.analytic_null_mean(HF.shape[0], b['n_l']):.1f}",
                    f"{len_f.mean():.2f}", f"{len_l.mean():.2f}"])
        for name, c in rows:
            w.writerow([name, c.get("n_f", HF.shape[0]), c["n_l"], f"{c['mean_rcs']:.4f}", c["peak_layer"],
                        f"{c['gini']:.4f}", f"{c['n80_over_L']:.4f}", f"{c['cv']:.4f}",
                        f"{c['spearman_vs_published']:.4f}",
                        f"{c['max_abs_dev_vs_published']:.4f}",
                        f"{c['analytic_null_mean']:.4f}", f"{c['ratio_obs_over_null']:.1f}",
                        f"{c['harmful_len_mean']:.2f}", f"{c['harmless_len_mean']:.2f}"])
    print(f"\nwrote {OUT}/matched_controls_qwen2-7b.{{json,csv}}")


if __name__ == "__main__":
    main()
