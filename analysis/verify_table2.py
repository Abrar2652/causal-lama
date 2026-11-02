#!/usr/bin/env python3
"""Audit published Table 2 (ASR, both judges) against the per-seed evaluation artifacts,
and test whether the supplement's described HIERARCHICAL paired bootstrap reproduces the
published confidence intervals.

Supplement C states: "confidence intervals and paired method comparisons use a
hierarchical paired bootstrap: we resample behaviors with replacement and, within each
sampled behavior, resample outcomes from seeds {1,2,42} while preserving the pairing
between methods." No code implementing that two-level resampling exists in this
repository (run_ablation_study.py calls only the behavior-level lama_stats helpers, per
seed), so it is reimplemented here from the written description and checked.

Writes results/verification/table2_audit.csv
"""
import glob, json, os, re, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

MODELS = ["qwen2-7b", "qwen2-3b", "llama2-7b", "gemma2-9b", "mistral-7b-rr"]
PAPER_NAME = {"qwen2-7b":"Qwen2.5-7B","qwen2-3b":"Qwen2.5-3B","llama2-7b":"Llama-2-7B",
              "gemma2-9b":"Gemma2-9B","mistral-7b-rr":"Mistral-7B-RR"}
METHODS = ["MD", "LAMA-uniform", "LAMA-RCS", "LAMA-BO"]
SEEDS = [1, 2, 42]

# Published Table 2 (main paper p.5), percent. (HB, G2) per method.
PUB = {
 "qwen2-7b":     {"MD":(84.3,78.6), "LAMA-uniform":(88.1,83.0), "LAMA-RCS":(93.1,81.8), "LAMA-BO":(86.2,76.7)},
 "qwen2-3b":     {"MD":(83.9,78.0), "LAMA-uniform":(87.2,80.3), "LAMA-RCS":(86.8,79.9), "LAMA-BO":(89.3,78.4)},
 "llama2-7b":    {"MD":( 5.7, 8.2), "LAMA-uniform":( 0.0, 0.0), "LAMA-RCS":( 0.0, 0.0), "LAMA-BO":( 0.0, 0.6)},
 "gemma2-9b":    {"MD":(95.6,78.0), "LAMA-uniform":(68.6,57.9), "LAMA-RCS":(50.3,39.6), "LAMA-BO":(55.3,46.5)},
 "mistral-7b-rr":{"MD":( 8.2,52.8), "LAMA-uniform":(20.1,34.0), "LAMA-RCS":( 1.3,56.6), "LAMA-BO":( 8.2,42.8)},
}
PUB_CI_HB = {  # published 95% hierarchical-bootstrap CIs, HarmBench judge
 "qwen2-7b": {"MD":(80.7,87.3), "LAMA-uniform":(84.8,90.7), "LAMA-RCS":(90.4,95.0), "LAMA-BO":(82.8,89.0)},
 "qwen2-3b": {"MD":(80.3,86.9), "LAMA-uniform":(83.9,89.9), "LAMA-RCS":(83.5,89.5), "LAMA-BO":(86.2,91.8)},
 "gemma2-9b":{"MD":(93.4,97.1), "LAMA-uniform":(64.3,72.6), "LAMA-RCS":(45.8,54.8), "LAMA-BO":(50.9,59.7)},
 "mistral-7b-rr":{"MD":(5.6,11.7),"LAMA-uniform":(16.1,24.9),"LAMA-RCS":(0.5,3.2),"LAMA-BO":(5.6,11.7)},
 "llama2-7b":{"MD":(3.0,10.4), "LAMA-uniform":(0.0,2.3), "LAMA-RCS":(0.0,2.3), "LAMA-BO":(0.0,2.3)},
}

ALIASES = {  # artifact filenames are not consistently cased across models
 "MD": ["MD"], "LAMA-uniform": ["LAMA-uniform", "lama_uniform"],
 "LAMA-RCS": ["LAMA-RCS", "lama_rcs"], "LAMA-BO": ["LAMA-BO", "lama_bo"],
}

def find_eval(model, method, seed, judge):
    """Locate an eval artifact; returns (path or None)."""
    d = os.path.join(REPO, "runs", model, "ablation_study")
    suffix = "hb_eval" if judge == "hb" else "guard2_eval"
    for a in ALIASES[method]:
        for pat in [f"{a}_seed{seed}_{suffix}.json"]:
            p = os.path.join(d, pat)
            if os.path.exists(p):
                return p
    return None

def load_cls(path, judge):
    d = json.load(open(path))
    if judge == "hb":
        if "harmbench_classifications" in d:
            return np.asarray(d["harmbench_classifications"], int)
        return np.asarray([int(c.get("is_jailbreak_harmbench", 0)) for c in d.get("completions", [])], int)
    return np.asarray(d.get("classifications", []), int)

def hierarchical_paired_bootstrap(per_seed_cls, B=10000, seed=0, alpha=0.05):
    """CI for ASR under the supplement's two-level scheme.

    per_seed_cls: [S, n] binary outcomes, behavior-aligned across seeds.
    Resample behaviors with replacement; within each sampled behavior draw one of the S
    seed outcomes with replacement.
    """
    X = np.asarray(per_seed_cls, float)
    S, n = X.shape
    rng = np.random.RandomState(seed)
    bidx = rng.randint(0, n, size=(B, n))
    sidx = rng.randint(0, S, size=(B, n))
    boot = X[sidx, bidx].mean(axis=1)
    return (float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2))))

rows = []
for m in MODELS:
    for meth in METHODS:
        for judge in ["hb", "g2"]:
            found, per_seed, missing = {}, [], []
            for s in SEEDS:
                p = find_eval(m, meth, s, judge)
                if p is None:
                    missing.append(s); continue
                c = load_cls(p, judge)
                if c.size == 0:
                    missing.append(s); continue
                found[s] = os.path.basename(p); per_seed.append(c)
            pub = PUB[m][meth][0 if judge == "hb" else 1]
            if not per_seed:
                rows.append({"model":PAPER_NAME[m],"method":meth,"judge":judge.upper(),
                             "status":"MISSING_ARTIFACT","seeds_found":"","n_seeds":0,
                             "published_pct":pub,"recomputed_pct":None,"abs_diff":None,
                             "missing_seeds":",".join(map(str,missing)),"n_behaviors":None,
                             "recomp_ci_lo":None,"recomp_ci_hi":None,
                             "published_ci_lo":None,"published_ci_hi":None,"ci_note":""})
                continue
            lens = {len(c) for c in per_seed}
            asr_per_seed = [100*float(c.mean()) for c in per_seed]
            recomp = float(np.mean(asr_per_seed))
            r = {"model":PAPER_NAME[m],"method":meth,"judge":judge.upper(),
                 "status":"OK" if len(lens)==1 else "LENGTH_MISMATCH",
                 "seeds_found":",".join(str(s) for s in sorted(found)),
                 "n_seeds":len(per_seed),"published_pct":pub,
                 "recomputed_pct":round(recomp,2),"abs_diff":round(abs(recomp-pub),2),
                 "missing_seeds":",".join(map(str,missing)),
                 "n_behaviors":sorted(lens)[0] if len(lens)==1 else str(sorted(lens)),
                 "recomp_ci_lo":None,"recomp_ci_hi":None,
                 "published_ci_lo":None,"published_ci_hi":None,"ci_note":""}
            if judge == "hb" and len(lens) == 1:
                lo, hi = hierarchical_paired_bootstrap(np.vstack(per_seed), seed=42)
                r["recomp_ci_lo"], r["recomp_ci_hi"] = round(100*lo,1), round(100*hi,1)
                pc = PUB_CI_HB.get(m, {}).get(meth)
                if pc:
                    r["published_ci_lo"], r["published_ci_hi"] = pc
                    d_lo, d_hi = abs(100*lo-pc[0]), abs(100*hi-pc[1])
                    r["ci_note"] = ("CI reproduced (<=0.8pp)" if max(d_lo,d_hi) <= 0.8
                                    else f"CI differs by up to {max(d_lo,d_hi):.1f}pp")
            rows.append(r)

out = os.path.join(REPO, "results", "verification", "table2_audit.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("ASR point estimates (recomputed from per-seed eval artifacts vs published Table 2)\n")
print("%-15s %-13s %-5s %-7s %8s %8s %6s  %s" % ("model","method","judge","seeds","pub%","recomp%","diff","status"))
for r in rows:
    print("%-15s %-13s %-5s %-7s %8s %8s %6s  %s" % (
        r["model"], r["method"], r["judge"], r["seeds_found"] or "-",
        r["published_pct"], r["recomputed_pct"], r["abs_diff"], r["status"]))

ok = [r for r in rows if r["status"]=="OK"]
miss = [r for r in rows if r["status"]=="MISSING_ARTIFACT"]
print("\ncells OK: %d   missing artifacts: %d" % (len(ok), len(miss)))
if ok:
    print("max |recomputed - published|: %.2f pp" % max(r["abs_diff"] for r in ok))
if miss:
    print("MISSING:", "; ".join(f"{r['model']}/{r['method']}/{r['judge']}" for r in miss))
print("\nHierarchical-bootstrap CI check (HarmBench judge, 3-seed models):")
for r in rows:
    if r["recomp_ci_lo"] is not None and r["published_ci_lo"] is not None:
        print("  %-15s %-13s recomputed [%5.1f,%5.1f]  published [%5.1f,%5.1f]   %s" % (
            r["model"], r["method"], r["recomp_ci_lo"], r["recomp_ci_hi"],
            r["published_ci_lo"], r["published_ci_hi"], r["ci_note"]))
print("\nwrote", out)
