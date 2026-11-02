#!/usr/bin/env python3
"""Audit published Table 1 (RCS concentration statistics) against cached per-layer
RCS profiles in logs/rcs_analysis_<model>.json.

Recomputes, with EXACTLY the conventions in lama_rcs.compute_rcs_statistics:
  peak layer, peak/mean, Gini, n80, n80/L, CV
and checks internal consistency of RCS = centroid_distance / (harmful_intra_std + eps).

Writes results/verification/table1_audit.csv
"""
import json, os, sys, csv
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from lama_rcs import _gini_coefficient

# Published Table 1 (main paper, p.5). key -> (L, peak, l_star, peak_mean, gini, n80, CV)
PUBLISHED = {
    "qwen2-7b":           (28, 19, 15, 1.58, 0.188, 19, 0.33),
    "qwen2-3b":           (36, 30, 23, 1.54, 0.208, 25, 0.37),
    "llama2-7b":          (32, 15, 11, 1.49, 0.200, 21, 0.37),
    "llama31-8b":         (32, 21, 11, 1.41, 0.192, 21, 0.34),
    "gemma2-9b":          (42, 38, 23, 1.52, 0.226, 27, 0.40),
    "mistral-7b-rr":      (32,  9, 23, 1.69, 0.231, 21, 0.42),
    "deepseek-r1-qwen-7b":(28, 22,  8, 1.22, 0.084, 21, 0.15),
    "ministral-8b":       (36, 16, 26, 1.35, 0.132, 26, 0.24),
    "gpt-oss-20b":        (24, 16, 10, 1.40, 0.110, 18, 0.20),
}
PAPER_NAME = {
    "qwen2-7b":"Qwen2.5-7B","qwen2-3b":"Qwen2.5-3B","llama2-7b":"Llama-2-7B",
    "llama31-8b":"Llama-3.1-8B","gemma2-9b":"Gemma2-9B","mistral-7b-rr":"Mistral-7B-RR",
    "deepseek-r1-qwen-7b":"DeepSeek-R1-7B","ministral-8b":"Ministral-8B","gpt-oss-20b":"GPT-oss-20B",
}

def n80_of(vals):
    """Exact convention from lama_rcs.compute_rcs_statistics."""
    a = np.asarray(vals, dtype=float)
    return int(np.searchsorted(np.cumsum(np.sort(a)[::-1]) / np.sum(a) * 100, 80) + 1)

rows = []
for alias, pub in PUBLISHED.items():
    p = os.path.join(REPO, "logs", f"rcs_analysis_{alias}.json")
    if not os.path.exists(p):
        rows.append({"model": PAPER_NAME[alias], "alias": alias, "status": "MISSING_CACHE"})
        continue
    d = json.load(open(p))
    layers = sorted(int(k) for k in d["rcs_scores"])
    rcs = np.array([d["rcs_scores"][str(l)] for l in layers], dtype=float)
    meta = d.get("rcs_metadata", {})

    # internal consistency: RCS == dist/(sigma+1e-8)
    recon_err = np.nan
    if meta:
        cd = np.array([meta[str(l)]["centroid_distance"] for l in layers])
        sg = np.array([meta[str(l)]["harmful_intra_std"] for l in layers])
        recon = cd / (sg + 1e-8)
        recon_err = float(np.max(np.abs(recon - rcs)))

    L, peak_pub, lstar_pub, pm_pub, gini_pub, n80_pub, cv_pub = pub
    peak = int(layers[int(np.argmax(rcs))])
    pm   = float(rcs.max() / (rcs.mean() + 1e-8))
    gini = float(_gini_coefficient(rcs))
    n80  = n80_of(rcs)
    cv   = float(rcs.std() / (rcs.mean() + 1e-8))

    rows.append({
        "model": PAPER_NAME[alias], "alias": alias, "status": "OK",
        "L_cached": len(layers), "L_pub": L, "L_match": len(layers) == L,
        "peak_cached": peak, "peak_pub": peak_pub, "peak_match": peak == peak_pub,
        "peakmean_cached": round(pm, 3), "peakmean_pub": pm_pub, "peakmean_absdiff": round(abs(pm - pm_pub), 3),
        "gini_cached": round(gini, 4), "gini_pub": gini_pub, "gini_absdiff": round(abs(gini - gini_pub), 4),
        "n80_cached": n80, "n80_pub": n80_pub, "n80_match": n80 == n80_pub,
        "n80_over_L_cached": round(n80 / len(layers), 3),
        "cv_cached": round(cv, 3), "cv_pub": cv_pub, "cv_absdiff": round(abs(cv - cv_pub), 3),
        "rcs_min": round(float(rcs.min()), 4), "rcs_max": round(float(rcs.max()), 4),
        "rcs_mean": round(float(rcs.mean()), 4),
        "eq2_recon_max_abs_err": None if np.isnan(recon_err) else float(f"{recon_err:.3e}"),
        # uniform-profile reference values (the permutation-null expectation, see analytic null)
        "gini_if_uniform": 0.0,
        "n80_if_uniform": int(np.ceil(0.8 * len(layers))),
        "n80_over_L_if_uniform": round(np.ceil(0.8*len(layers))/len(layers), 3),
    })

out = os.path.join(REPO, "results", "verification", "table1_audit.csv")
keys = sorted({k for r in rows for k in r})
order = ["model","alias","status","L_cached","L_pub","L_match","peak_cached","peak_pub","peak_match",
         "peakmean_cached","peakmean_pub","peakmean_absdiff","gini_cached","gini_pub","gini_absdiff",
         "n80_cached","n80_pub","n80_match","n80_over_L_cached","cv_cached","cv_pub","cv_absdiff",
         "rcs_min","rcs_mean","rcs_max","eq2_recon_max_abs_err",
         "gini_if_uniform","n80_if_uniform","n80_over_L_if_uniform"]
order += [k for k in keys if k not in order]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=order); w.writeheader()
    for r in rows: w.writerow(r)

# console report
print(f"{'model':<16}{'L':>4}{'peak':>12}{'peak/mean':>20}{'Gini':>20}{'n80':>12}{'CV':>16}  eq2err")
for r in rows:
    if r["status"] != "OK":
        print(f"{r['model']:<16}  {r['status']}"); continue
    print(f"{r['model']:<16}{r['L_cached']:>4}"
          f"{str(r['peak_cached'])+'/'+str(r['peak_pub']):>12}"
          f"{str(r['peakmean_cached'])+' vs '+str(r['peakmean_pub']):>20}"
          f"{str(r['gini_cached'])+' vs '+str(r['gini_pub']):>20}"
          f"{str(r['n80_cached'])+'/'+str(r['n80_pub']):>12}"
          f"{str(r['cv_cached'])+' vs '+str(r['cv_pub']):>16}"
          f"  {r['eq2_recon_max_abs_err']}")
ok = [r for r in rows if r["status"]=="OK"]
print()
print("models audited:", len(ok), "of", len(PUBLISHED))
print("exact matches  L:", sum(r["L_match"] for r in ok),
      " peak:", sum(r["peak_match"] for r in ok),
      " n80:", sum(r["n80_match"] for r in ok))
if ok:
    print("max |Gini diff|:", max(r["gini_absdiff"] for r in ok),
          " max |peak/mean diff|:", max(r["peakmean_absdiff"] for r in ok),
          " max |CV diff|:", max(r["cv_absdiff"] for r in ok))
print("wrote", out)
