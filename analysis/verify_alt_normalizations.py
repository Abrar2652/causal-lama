#!/usr/bin/env python3
"""Alternative normalizations of the harmful-harmless centroid gap (reviewer hgpB, W3).

hgpB: "The paper does not compare RCS against simpler baselines, such as raw centroid norm
without normalization or alternative normalization schemes, leaving it unclear whether RCS
provides a meaningful advantage."

The cached logs/rcs_analysis_*.json stores only the gap and the HARMFUL spread, so the
harmless- and pooled-spread variants cannot be computed from it. They need the sample-level
tensors. This script computes them once those exist.

Each variant divides the SAME numerator ||mu_hf - nu||_2 by a different scale, then is scored
the same way as RCS in Table 3: Spearman correlation with the measured single-layer causal
profile, plus the partial correlation controlling for the raw gap.

Usage: python3 analysis/verify_alt_normalizations.py --model qwen2-7b
"""
import argparse, json, os, sys, csv
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
from lama_rcs import partial_spearman

EPS = 1e-8

def variants(HF, HL):
    """Per-layer scores for each normalization scheme. HF/HL: [n, L, d]."""
    L = HF.shape[1]
    out = {k: np.zeros(L) for k in
           ["raw_gap", "rcs_harmful_spread", "harmless_spread", "pooled_spread",
            "max_spread", "geometric_mean_spread", "pooled_std_trace", "snr_fisher"]}
    for l in range(L):
        hf, hl = HF[:, l, :], HL[:, l, :]
        mu, nu = hf.mean(0), hl.mean(0)
        gap = float(np.linalg.norm(mu - nu))
        s_hf = float(np.linalg.norm(hf - mu, axis=1).mean())    # Eq. (2) denominator
        s_hl = float(np.linalg.norm(hl - nu, axis=1).mean())
        # trace-based (RMS radius) versions
        t_hf = float(np.sqrt((np.linalg.norm(hf - mu, axis=1) ** 2).mean()))
        t_hl = float(np.sqrt((np.linalg.norm(hl - nu, axis=1) ** 2).mean()))
        out["raw_gap"][l]              = gap
        out["rcs_harmful_spread"][l]   = gap / (s_hf + EPS)
        out["harmless_spread"][l]      = gap / (s_hl + EPS)
        out["pooled_spread"][l]        = gap / (0.5 * (s_hf + s_hl) + EPS)
        out["max_spread"][l]           = gap / (max(s_hf, s_hl) + EPS)
        out["geometric_mean_spread"][l]= gap / (np.sqrt(s_hf * s_hl) + EPS)
        out["pooled_std_trace"][l]     = gap / (np.sqrt(0.5 * (t_hf**2 + t_hl**2)) + EPS)
        out["snr_fisher"][l]           = gap**2 / (t_hf**2 + t_hl**2 + EPS)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-7b")
    ap.add_argument("--reps_dir", default=None)
    a = ap.parse_args()

    import torch
    base = a.reps_dir
    if base is None:
        for cand in (a.model + "_processed_confirm", a.model):
            d = os.path.join(REPO, "dataset", "representations", cand)
            if os.path.exists(os.path.join(d, "HFx_train.pt")):
                base = d
                break
        else:
            base = os.path.join(REPO, "dataset", "representations", a.model)
    hf_p, hl_p = os.path.join(base, "HFx_train.pt"), os.path.join(base, "HLx_train.pt")
    for p in (hf_p, hl_p):
        if not os.path.exists(p):
            raise SystemExit(f"BLOCKED: missing {p}")
    HF = torch.load(hf_p, weights_only=True).float().numpy()
    HL = torch.load(hl_p, weights_only=True).float().numpy()
    print(f"loaded HF={HF.shape} HL={HL.shape}")

    V = variants(HF, HL)
    L = HF.shape[1]

    # agreement with the published profile
    cache = os.path.join(REPO, "logs", f"rcs_analysis_{a.model}.json")
    cached = None
    if os.path.exists(cache):
        cd = json.load(open(cache))
        if len(cd["rcs_scores"]) == L:
            cached = np.array([cd["rcs_scores"][str(l)] for l in range(L)])
            dev = float(np.max(np.abs(cached - V["rcs_harmful_spread"])))
            rho = stats.spearmanr(cached, V["rcs_harmful_spread"])[0]
            print(f"regenerated RCS vs published: max|diff|={dev:.4f}  Spearman={rho:.4f}")

    # causal profile
    cf = os.path.join(REPO, "logs", f"causal_refusalprob_{a.model}.json")
    if not os.path.exists(cf):
        print("no causal profile cached; emitting profiles only")
        dp = None
    else:
        d = json.load(open(cf))
        layers = sorted(int(k) for k in d["delta_p"])
        dp = np.array([d["delta_p"][str(l)] for l in layers])
        if len(dp) != L:
            print(f"causal profile has {len(dp)} layers vs {L}; skipping correlations"); dp = None

    rows = []
    if dp is not None:
        gap = V["raw_gap"]
        print()
        print("%-24s %10s %10s %12s" % ("normalization", "rho(causal)", "vs RCS", "partial|raw gap"))
        rcs_rho = stats.spearmanr(V["rcs_harmful_spread"], dp)[0]
        for name, v in V.items():
            r = stats.spearmanr(v, dp)[0]
            pr = (float("nan") if name == "raw_gap"
                  else partial_spearman(v, dp, gap)[0])
            print("%-24s %10.3f %10s %12s" % (
                name, r, "--" if name == "rcs_harmful_spread" else f"{r - rcs_rho:+.3f}",
                "--" if name == "raw_gap" else f"{pr:.3f}"))
            rows.append({"model": a.model, "normalization": name,
                         "spearman_vs_causal": round(float(r), 4),
                         "delta_vs_rcs": None if name == "rcs_harmful_spread" else round(float(r - rcs_rho), 4),
                         "partial_given_raw_gap": None if name == "raw_gap" else round(float(pr), 4),
                         "n_layers": L})
        out = os.path.join(REPO, "results", "verification",
                           f"alt_normalizations_{a.model}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("\nwrote", out)

    np.savez(os.path.join(REPO, "results", "verification",
                          f"normalization_profiles_{a.model}.npz"), **V)
    print("wrote per-layer profiles for every variant")

if __name__ == "__main__":
    main()
