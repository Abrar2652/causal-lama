#!/usr/bin/env python3
"""Does the published interval width imply behavior x seed pairs were pooled as
independent units, and do the starred Table 2 claims survive a behavior-level analysis?

Candidate C: pool all S*n (behavior, seed) observations and resample them independently.
This inflates the effective sample size by S and narrows intervals by ~sqrt(S). It is only
valid if seeds are independent replicates -- so we also test whether the per-seed
judgments differ at all.

Then re-runs the paired tests that carry Table 2's significance stars, at the behavior
level on seed 42 (McNemar + paired bootstrap), which is the statistical unit the
supplement itself names.
"""
import json, os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
from verify_table2 import find_eval, load_cls, MODELS, METHODS, SEEDS, PAPER_NAME, PUB_CI_HB
from lama_stats import mcnemar_test, paired_bootstrap_diff, bootstrap_asr_ci

def pooled_independent_bootstrap(X, B=10000, seed=42, alpha=0.05):
    """Candidate C: treat every (behavior, seed) pair as an independent unit."""
    flat = np.asarray(X, float).ravel()
    n = len(flat); rng = np.random.RandomState(seed)
    boot = flat[rng.randint(0, n, size=(B, n))].mean(axis=1)
    return (float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2))))

print("="*104)
print("Do the three seeds actually differ?  (identical judgments => 'multi-seed' adds no information)")
print("="*104)
print("%-15s %-13s %7s %28s %s" % ("model","method","n_seeds","pairwise identical?","per-seed ASR%"))
ident_rows=[]
for m in MODELS:
    for meth in METHODS:
        cs=[]
        for s in SEEDS:
            p=find_eval(m,meth,s,"hb")
            if p:
                c=load_cls(p,"hb")
                if c.size: cs.append((s,c))
        if len(cs)<2: continue
        allsame = all(np.array_equal(cs[0][1], c) for _,c in cs[1:])
        asrs = ", ".join("s%d=%.2f"%(s,100*c.mean()) for s,c in cs)
        ndiff = 0 if allsame else int(sum((cs[0][1]!=c).sum() for _,c in cs[1:]))
        print("%-15s %-13s %7d %28s %s" % (PAPER_NAME[m],meth,len(cs),
              "YES (identical)" if allsame else "no (%d differing judgments)"%ndiff, asrs))
        ident_rows.append({"model":PAPER_NAME[m],"method":meth,"n_seeds":len(cs),
                           "seeds_identical":allsame,"n_differing_judgments":ndiff,"per_seed_asr":asrs})

print()
print("="*104)
print("Candidate C: pooling (behavior, seed) pairs as independent units")
print("="*104)
print("%-15s %-13s %-16s %-24s %s" % ("model","method","published","C: pooled-independent","dev"))
rows=[]
for m in MODELS:
    for meth in METHODS:
        cs=[]
        for s in SEEDS:
            p=find_eval(m,meth,s,"hb")
            if p:
                c=load_cls(p,"hb")
                if c.size: cs.append(c)
        if not cs or len({len(c) for c in cs})!=1: continue
        X=np.vstack(cs)
        pc=PUB_CI_HB.get(m,{}).get(meth)
        if not pc: continue
        c_lo,c_hi = pooled_independent_bootstrap(X)
        dev=max(abs(100*c_lo-pc[0]),abs(100*c_hi-pc[1]))
        print("%-15s %-13s [%5.1f,%5.1f]   [%5.1f,%5.1f]            %+5.2f pp" % (
            PAPER_NAME[m],meth,pc[0],pc[1],100*c_lo,100*c_hi,dev))
        rows.append({"model":PAPER_NAME[m],"method":meth,"n_seeds":X.shape[0],
                     "pub_lo":pc[0],"pub_hi":pc[1],
                     "C_pooled_lo":round(100*c_lo,1),"C_pooled_hi":round(100*c_hi,1),
                     "C_max_abs_dev_pp":round(dev,2)})
multi=[r for r in rows if r["n_seeds"]>1]
if multi:
    print("\nmedian |deviation| for candidate C over multi-seed cells: %.2f pp (A was 2.48, B 2.34)" %
          float(np.median([r["C_max_abs_dev_pp"] for r in multi])))

print()
print("="*104)
print("Starred Table 2 / Table 4 claims re-tested at the behavior level (seed 42, n=159)")
print("="*104)
print("%-15s %-14s %8s %8s %8s %10s %12s %s" % ("model","variant vs MD","MD%","var%","diff pp",
      "McNemar p","boot p","behavior-level verdict"))
sig_rows=[]
for m in MODELS:
    md_p = find_eval(m,"MD",42,"hb")
    if not md_p: continue
    md = load_cls(md_p,"hb")
    for meth in ["LAMA-uniform","LAMA-RCS","LAMA-BO"]:
        p = find_eval(m,meth,42,"hb")
        if not p: continue
        v = load_cls(p,"hb")
        if v.size != md.size: continue
        mc = mcnemar_test(v, md)
        pb = paired_bootstrap_diff(v, md)
        verdict = ("variant > MD (p<0.05)" if (pb["diff"]>0 and mc["p_value"]<0.05) else
                   "MD > variant (p<0.05)" if (pb["diff"]<0 and mc["p_value"]<0.05) else
                   "not significant")
        print("%-15s %-14s %8.2f %8.2f %+8.2f %10.4f %12.4f %s" % (
            PAPER_NAME[m],meth,100*md.mean(),100*v.mean(),100*(v.mean()-md.mean()),
            mc["p_value"],pb["p_value"],verdict))
        sig_rows.append({"model":PAPER_NAME[m],"variant":meth,"seed":42,"n":int(md.size),
                         "md_asr_pct":round(100*float(md.mean()),2),
                         "variant_asr_pct":round(100*float(v.mean()),2),
                         "diff_pp":round(100*float(v.mean()-md.mean()),2),
                         "mcnemar_p":round(mc["p_value"],6),"mcnemar_method":mc["method"],
                         "paired_boot_p":round(pb["p_value"],6),
                         "boot_ci_lo_pp":round(100*pb["ci_lo"],2),"boot_ci_hi_pp":round(100*pb["ci_hi"],2),
                         "verdict":verdict})

V = os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification")
for name,rr in [("seed_identity_audit.csv",ident_rows),("ci_candidateC_audit.csv",rows),
                ("significance_behaviorlevel_audit.csv",sig_rows)]:
    if rr:
        with open(os.path.join(V,name),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
        print("wrote", os.path.join(V,name))
