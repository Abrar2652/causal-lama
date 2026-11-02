#!/usr/bin/env python3
"""Three-way test of the data-source diagnosis.

The paper's Supplement C states the harmful class is 4000 SORRY-Bench prompts. All
nine extraction logs record `[processed source] harmful=260 harmless=6000`, i.e. the
in-repo split. This script decides between the two accounts by regenerating the RCS
profile under each source and comparing both against the PUBLISHED cached profile.

Compared with rank statistics and shape statistics rather than exact equality, because
fp16 inference is not bitwise reproducible across GPU architectures (the published runs,
the SORRY-Bench run and this run did not all use the same card). Spearman rank
agreement, peak-layer agreement and Gini are robust to that drift; a raw max|diff| is not.

Writes results/verification/data_source_audit.csv
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import null_core as nc
from lama_rcs import compute_rcs
from scipy.stats import spearmanr

MODEL = "qwen2-7b"
CAND = [
    ("processed (260 harmful / 6000 harmless, in-repo split)",
     os.path.join(REPO, "dataset", "representations", MODEL + "_processed_confirm")),
    ("sorry-bench (4000 harmful / 6000 harmless, as the paper states)",
     os.path.join(REPO, "dataset", "representations", MODEL + "_hfsource")),
]


def profile_from(d):
    import torch
    hf_p, hl_p = os.path.join(d, "HFx_train.pt"), os.path.join(d, "HLx_train.pt")
    if not (os.path.exists(hf_p) and os.path.exists(hl_p)):
        return None, None
    HF = torch.load(hf_p, weights_only=True).float().numpy()
    HL = torch.load(hl_p, weights_only=True).float().numpy()
    # drop all-zero rows: convert() pre-allocates zeros and skips None prompts
    nzf = np.abs(HF).sum(axis=(1, 2)) > 0
    nzl = np.abs(HL).sum(axis=(1, 2)) > 0
    HF, HL = HF[nzf], HL[nzl]
    rcs_d, _ = compute_rcs(HF, HL)
    L = HF.shape[1]
    return np.array([rcs_d[l] for l in range(L)]), (HF.shape, HL.shape)


def stats(v, pub):
    L = len(v)
    return {
        "L": L,
        "peak_layer": int(v.argmax()),
        "mean_rcs": round(float(v.mean()), 4),
        "peak_rcs": round(float(v.max()), 4),
        "gini": round(float(nc.gini(v)), 4),
        "n80_over_L": round(float(nc.n80(v) / L), 4),
        "peak_over_mean": round(float(nc.peak_over_mean(v)), 4),
        "cv": round(float(nc.cv(v)), 4),
        "spearman_vs_published": round(float(spearmanr(v, pub)[0]), 4),
        "spearman_vs_published_reversed": round(float(spearmanr(v, pub[::-1])[0]), 4),
        "max_abs_dev_vs_published": round(float(np.max(np.abs(v - pub))), 4),
        "mean_abs_dev_vs_published": round(float(np.mean(np.abs(v - pub))), 4),
    }


def main():
    pub = json.load(open(os.path.join(REPO, "logs", f"rcs_analysis_{MODEL}.json")))
    rs = pub["rcs_scores"]
    P = np.array([rs[str(i)] for i in range(len(rs))])

    rows = []
    rows.append({"source": "PUBLISHED (logs/rcs_analysis_%s.json)" % MODEL,
                 "shapes": "n/a (cached profile)", **stats(P, P)})

    for label, d in CAND:
        v, shapes = profile_from(d)
        if v is None:
            rows.append({"source": label, "shapes": "ABSENT -- not regenerated",
                         **{k: "" for k in rows[0] if k not in ("source", "shapes")}})
            continue
        if len(v) != len(P):
            rows.append({"source": label, "shapes": str(shapes),
                         **{k: "" for k in rows[0] if k not in ("source", "shapes")}})
            continue
        rows.append({"source": label, "shapes": "HF=%s HL=%s" % shapes, **stats(v, P)})

    out = os.path.join(REPO, "results", "verification", "data_source_audit.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    keys = ["peak_layer", "mean_rcs", "peak_rcs", "gini", "n80_over_L",
            "spearman_vs_published", "spearman_vs_published_reversed",
            "mean_abs_dev_vs_published"]
    print("\n%-58s %s" % ("source", "  ".join("%s" % k for k in keys)))
    for r in rows:
        print("%-58s %s" % (r["source"][:58],
                            "  ".join(str(r.get(k, "")) for k in keys)))
    print("\nshapes:")
    for r in rows:
        print("  %-58s %s" % (r["source"][:58], r["shapes"]))
    print(f"\nwrote {out}")

    # explicit verdict
    proc = next((r for r in rows if r["source"].startswith("processed")), None)
    if proc and proc.get("spearman_vs_published") != "":
        rho = float(proc["spearman_vs_published"])
        same_peak = proc["peak_layer"] == rows[0]["peak_layer"]
        print("\nVERDICT: processed-source Spearman vs published = %.4f, "
              "peak layer %s vs published %s -> %s"
              % (rho, proc["peak_layer"], rows[0]["peak_layer"],
                 "DIAGNOSIS CONFIRMED" if (rho > 0.95 and same_peak)
                 else "NOT confirmed; report the deviation"))


if __name__ == "__main__":
    main()
