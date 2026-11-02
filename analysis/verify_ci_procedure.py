#!/usr/bin/env python3
"""Which resampling procedure reproduces the published Table 2 confidence intervals?

Candidate A -- as the supplement describes it: resample behaviors with replacement and,
  within each sampled behavior, resample one outcome from seeds {1,2,42}.
Candidate B -- average the seeds first: form each behavior's mean outcome across seeds,
  then resample behaviors only.

A propagates seed-level variance; B treats the per-behavior seed-mean as the unit and is
narrower by roughly sqrt(S). Also reports Mistral-7B-RR per-seed ASR, because its
HarmBench column does not reproduce as a three-seed mean.
"""
import json, os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from verify_table2 import (find_eval, load_cls, MODELS, METHODS, SEEDS, PAPER_NAME,
                           PUB_CI_HB, hierarchical_paired_bootstrap)

def seed_averaged_bootstrap(per_seed_cls, B=10000, seed=42, alpha=0.05):
    X = np.asarray(per_seed_cls, float).mean(axis=0)   # per-behavior mean over seeds
    n = len(X); rng = np.random.RandomState(seed)
    boot = X[rng.randint(0, n, size=(B, n))].mean(axis=1)
    return (float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2))))

print("=" * 100)
print("Mistral-7B-RR per-seed ASR (the one model whose HarmBench column does not reproduce)")
print("=" * 100)
print("%-14s %-6s %8s %8s %8s %10s %10s %8s" % ("method","judge","seed1","seed2","seed42","mean(1,2)","mean(all)","pub"))
from verify_table2 import PUB
mis_rows = []
for meth in METHODS:
    for judge in ["hb", "g2"]:
        vals = {}
        for s in SEEDS:
            p = find_eval("mistral-7b-rr", meth, s, judge)
            if p: vals[s] = 100*float(load_cls(p, judge).mean())
        if not vals: continue
        m12 = np.mean([vals[s] for s in (1,2) if s in vals]) if any(s in vals for s in (1,2)) else float("nan")
        mall = np.mean(list(vals.values()))
        pub = PUB["mistral-7b-rr"][meth][0 if judge=="hb" else 1]
        print("%-14s %-6s %8s %8s %8s %10.2f %10.2f %8s" % (
            meth, judge.upper(),
            f"{vals.get(1, float('nan')):.2f}", f"{vals.get(2, float('nan')):.2f}",
            f"{vals.get(42, float('nan')):.2f}", m12, mall, pub))
        mis_rows.append({"method":meth,"judge":judge.upper(),
                         "seed1":round(vals.get(1,float('nan')),2),"seed2":round(vals.get(2,float('nan')),2),
                         "seed42":round(vals.get(42,float('nan')),2),
                         "mean_seeds_1_2":round(m12,2),"mean_all_seeds":round(mall,2),"published":pub,
                         "matches":"mean(1,2)" if abs(m12-pub)<0.1 else ("mean(all)" if abs(mall-pub)<0.1 else "neither")})

print()
print("=" * 100)
print("Which bootstrap reproduces the published HarmBench CIs?")
print("=" * 100)
print("%-15s %-13s %-18s %-18s %-18s" % ("model","method","published","A: seeds-within","B: seed-averaged"))
rows = []
for m in MODELS:
    for meth in METHODS:
        per_seed = []
        for s in SEEDS:
            p = find_eval(m, meth, s, "hb")
            if p:
                c = load_cls(p, "hb")
                if c.size: per_seed.append(c)
        if not per_seed: continue
        if len({len(c) for c in per_seed}) != 1: continue
        X = np.vstack(per_seed)
        a_lo, a_hi = hierarchical_paired_bootstrap(X, seed=42)
        b_lo, b_hi = seed_averaged_bootstrap(X, seed=42)
        pc = PUB_CI_HB.get(m, {}).get(meth)
        dA = max(abs(100*a_lo-pc[0]), abs(100*a_hi-pc[1])) if pc else None
        dB = max(abs(100*b_lo-pc[0]), abs(100*b_hi-pc[1])) if pc else None
        print("%-15s %-13s [%5.1f,%5.1f]      [%5.1f,%5.1f] %+5.1f  [%5.1f,%5.1f] %+5.1f" % (
            PAPER_NAME[m], meth, pc[0], pc[1], 100*a_lo, 100*a_hi, dA, 100*b_lo, 100*b_hi, dB))
        rows.append({"model":PAPER_NAME[m],"method":meth,"n_seeds":X.shape[0],
                     "pub_lo":pc[0],"pub_hi":pc[1],
                     "A_seeds_within_lo":round(100*a_lo,1),"A_seeds_within_hi":round(100*a_hi,1),
                     "A_max_abs_dev_pp":round(dA,2),
                     "B_seed_averaged_lo":round(100*b_lo,1),"B_seed_averaged_hi":round(100*b_hi,1),
                     "B_max_abs_dev_pp":round(dB,2),
                     "closer":"B_seed_averaged" if dB < dA else "A_seeds_within"})

multi = [r for r in rows if r["n_seeds"] > 1]
print()
if multi:
    print("multi-seed cells: %d   median |dev| A (as described) = %.2f pp   B (seed-averaged) = %.2f pp" % (
        len(multi), float(np.median([r["A_max_abs_dev_pp"] for r in multi])),
        float(np.median([r["B_max_abs_dev_pp"] for r in multi]))))
    print("cells where B is closer to published: %d / %d" % (
        sum(r["closer"]=="B_seed_averaged" for r in multi), len(multi)))

o1 = os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification","mistral_per_seed_audit.csv")
with open(o1,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(mis_rows[0])); w.writeheader(); w.writerows(mis_rows)
o2 = os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification","ci_procedure_audit.csv")
with open(o2,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nwrote", o1, "\n      ", o2)
