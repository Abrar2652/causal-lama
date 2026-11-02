#!/usr/bin/env python3
"""Behavior-level paired tests on seed-averaged outcomes -- the defensible multi-seed
analysis when the statistical unit is the behavior (as supplement C states).

For each behavior b we form the mean outcome over available seeds for each method, then
paired-bootstrap the difference over behaviors (n = 159). This keeps the pairing, does not
treat (behavior, seed) pairs as independent units, and matches the point estimates in
Tables 2 and 4.
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
from verify_table2 import find_eval, load_cls, MODELS, SEEDS, PAPER_NAME

# published Table 4 deltas vs MD (HarmBench), with the paper's marks
PUB_T4 = {
 "qwen2-7b": {"LAMA-uniform":(+3.8,"dagger"), "LAMA-RCS":(+8.8,"star"), "LAMA-BO":(+1.9,"")},
 "qwen2-3b": {"LAMA-uniform":(+3.4,"dagger"), "LAMA-RCS":(+2.9,""),     "LAMA-BO":(+5.5,"star")},
 "gemma2-9b":{"LAMA-uniform":(-27.0,"star"),  "LAMA-RCS":(-45.3,"star"),"LAMA-BO":(-40.3,"star")},
}

def stack(model, method, judge="hb"):
    cs = []
    for s in SEEDS:
        p = find_eval(model, method, s, judge)
        if p:
            c = load_cls(p, judge)
            if c.size: cs.append(c)
    if not cs or len({len(c) for c in cs}) != 1:
        return None
    return np.vstack(cs).astype(float)

def paired_behavior_bootstrap(va, vb, B=10000, seed=42, alpha=0.05):
    """Paired bootstrap over behaviors on seed-averaged per-behavior outcomes."""
    a = np.asarray(va, float).mean(axis=0)
    b = np.asarray(vb, float).mean(axis=0)
    n = len(a); rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(B, n))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(boot, 100*alpha/2), np.percentile(boot, 100*(1-alpha/2))
    p = float(min(1.0, 2*min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return float(a.mean()-b.mean()), float(lo), float(hi), p

def wilcoxon_paired(va, vb):
    from scipy.stats import wilcoxon
    a = np.asarray(va, float).mean(axis=0); b = np.asarray(vb, float).mean(axis=0)
    d = a - b
    if np.allclose(d, 0): return 1.0
    try: return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception: return float("nan")

print("%-14s %-13s %7s %9s %20s %9s %9s  %-9s %s" % (
    "model","variant","pubΔ","mark","recomputed Δ [95% CI]","boot p","wilcox p","paper says","behavior-level verdict"))
rows=[]
for m in ["qwen2-7b","qwen2-3b","gemma2-9b","mistral-7b-rr","llama2-7b"]:
    MD = stack(m,"MD")
    if MD is None: continue
    for meth in ["LAMA-uniform","LAMA-RCS","LAMA-BO"]:
        V = stack(m,meth)
        if V is None or V.shape[1]!=MD.shape[1]: continue
        ns = min(V.shape[0], MD.shape[0])
        d, lo, hi, p = paired_behavior_bootstrap(V[:ns], MD[:ns])
        wp = wilcoxon_paired(V[:ns], MD[:ns])
        pub, mark = PUB_T4.get(m,{}).get(meth,(None,""))
        claim = {"star":"p<0.05","dagger":"p<0.10",""  :"n.s."}[mark]
        sig = "significant (p<0.05)" if p<0.05 else ("marginal (p<0.10)" if p<0.10 else "not significant")
        agree = ("AGREES" if ((mark=="star" and p<0.05) or (mark=="dagger" and 0.05<=p<0.10) or
                              (mark=="" and p>=0.10)) else "DIFFERS")
        print("%-14s %-13s %7s %9s  %+6.2f [%+6.2f,%+6.2f] %9.4f %9.4f  %-9s %s -> %s" % (
            PAPER_NAME[m],meth,pub,claim,100*d,100*lo,100*hi,p,wp,claim,sig,agree))
        rows.append({"model":PAPER_NAME[m],"variant":meth,"n_seeds_used":ns,
                     "n_behaviors":int(MD.shape[1]),
                     "published_delta_pp":pub,"paper_mark":mark or "none","paper_claim":claim,
                     "recomputed_delta_pp":round(100*d,2),
                     "ci_lo_pp":round(100*lo,2),"ci_hi_pp":round(100*hi,2),
                     "paired_boot_p":round(p,5),"wilcoxon_p":round(wp,5) if wp==wp else None,
                     "behavior_level_verdict":sig,"agreement_with_paper":agree})

out = os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification","significance_multiseed_audit.csv")
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nagreement summary: %d AGREES, %d DIFFERS, of %d testable cells" % (
    sum(r["agreement_with_paper"]=="AGREES" for r in rows),
    sum(r["agreement_with_paper"]=="DIFFERS" for r in rows), len(rows)))
print("wrote", out)
