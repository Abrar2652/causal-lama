#!/usr/bin/env python3
"""Audit published Table 3 / Table 6 (RCS vs single-layer causal effects) from the cached
causal sweeps, and add the alternative-normalization comparison reviewer hgpB asks for
(W3: RCS is not compared against simpler baselines).

Sources, all already in the repo:
  logs/causal_refusalprob_<model>.json      graded dP_refuse sweep, per layer
  logs/causal_asr_cache_<model>_harmbench_test.json   binary ASR sweep, per layer
  logs/rcs_analysis_<model>.json            RCS + centroid gap + harmful spread
  logs/rcs_robustness_noncausal_qwen2-7b.json, runs/qwen2-7b/metric_robustness/cross_cat_probe.json
"""
import json, os, sys, csv
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from lama_rcs import partial_spearman

PAPER = {"qwen2-7b":"Qwen2.5-7B","qwen2-3b":"Qwen2.5-3B","llama2-7b":"Llama-2-7B",
         "gemma2-9b":"Gemma2-9B","mistral-7b-rr":"Mistral-7B-RR"}
# published Table 3, graded dP_refuse: (rho_RCS, rho_norm, partial)
PUB_GRADED = {"qwen2-7b":(0.873,0.744,0.685), "qwen2-3b":(0.749,0.674,0.445),
              "llama2-7b":(0.466,-0.070,0.625), "gemma2-9b":(0.313,0.385,-0.219)}
# published Table 3, binary dASR
PUB_BINARY = {"qwen2-7b":(0.657,0.424,0.610), "qwen2-3b":(0.434,0.375,0.235)}

rows = []
print("="*100); print("Table 3 -- graded dP_refuse metric"); print("="*100)
print("%-14s %8s %8s %8s   %8s %8s %8s   %s" % ("model","rho_pub","rho_rec","diff",
      "nrm_pub","nrm_rec","prt_pub","prt_rec"))
for alias,(p_rho,p_nrm,p_prt) in PUB_GRADED.items():
    f = os.path.join(REPO,"logs",f"causal_refusalprob_{alias}.json")
    if not os.path.exists(f):
        print("%-14s MISSING %s" % (PAPER[alias], f)); continue
    d = json.load(open(f))
    layers = sorted(int(k) for k in d["rcs"])
    rcs  = np.array([d["rcs"][str(l)] for l in layers])
    dp   = np.array([d["delta_p"][str(l)] for l in layers])
    nrm  = np.array([d["norms"][str(l)] for l in layers])
    r_rho = stats.spearmanr(rcs, dp)[0]
    r_nrm = stats.spearmanr(nrm, dp)[0]
    r_prt = partial_spearman(rcs, dp, nrm)[0]
    print("%-14s %8.3f %8.3f %8.3f   %8.3f %8.3f %8.3f   %8.3f" % (
        PAPER[alias], p_rho, r_rho, abs(r_rho-p_rho), p_nrm, r_nrm, p_prt, r_prt))
    rows.append({"table":"3-graded","model":PAPER[alias],"n_layers":len(layers),
                 "rho_rcs_pub":p_rho,"rho_rcs_rec":round(float(r_rho),4),
                 "rho_norm_pub":p_nrm,"rho_norm_rec":round(float(r_nrm),4),
                 "partial_pub":p_prt,"partial_rec":round(float(r_prt),4),
                 "stored_rho":d.get("spearman_rcs"),"stored_norm":d.get("spearman_norm"),
                 "stored_partial":d.get("partial_rcs_given_norm"),
                 "max_abs_dev":round(max(abs(r_rho-p_rho),abs(r_nrm-p_nrm),abs(r_prt-p_prt)),4)})

print()
print("="*100); print("Table 3 -- binary dASR metric"); print("="*100)
print("%-14s %8s %8s   %8s %8s   %8s %8s  %s" % ("model","rho_pub","rho_rec","nrm_pub","nrm_rec","prt_pub","prt_rec","clean ASR"))
for alias,(p_rho,p_nrm,p_prt) in PUB_BINARY.items():
    fa = os.path.join(REPO,"logs",f"causal_asr_cache_{alias}_harmbench_test.json")
    fr = os.path.join(REPO,"logs",f"rcs_analysis_{alias}.json")
    if not (os.path.exists(fa) and os.path.exists(fr)):
        print("%-14s MISSING" % PAPER[alias]); continue
    ca = json.load(open(fa)); ra = json.load(open(fr))
    noabl = ca["noabl"]
    layers = sorted(int(k[1:]) for k in ca if k.startswith("L"))
    dasr = np.array([ca[f"L{l}"] - noabl for l in layers])
    rcs  = np.array([ra["rcs_scores"][str(l)] for l in layers])
    nrm  = np.array([ra["rcs_metadata"][str(l)]["centroid_distance"] for l in layers])
    r_rho = stats.spearmanr(rcs,dasr)[0]; r_nrm = stats.spearmanr(nrm,dasr)[0]
    r_prt = partial_spearman(rcs,dasr,nrm)[0]
    print("%-14s %8.3f %8.3f   %8.3f %8.3f   %8.3f %8.3f  %.4f" % (
        PAPER[alias],p_rho,r_rho,p_nrm,r_nrm,p_prt,r_prt,noabl))
    rows.append({"table":"3-binary","model":PAPER[alias],"n_layers":len(layers),
                 "rho_rcs_pub":p_rho,"rho_rcs_rec":round(float(r_rho),4),
                 "rho_norm_pub":p_nrm,"rho_norm_rec":round(float(r_nrm),4),
                 "partial_pub":p_prt,"partial_rec":round(float(r_prt),4),
                 "stored_rho":None,"stored_norm":None,"stored_partial":None,
                 "max_abs_dev":round(max(abs(r_rho-p_rho),abs(r_nrm-p_nrm),abs(r_prt-p_prt)),4)})

print()
print("="*100)
print("hgpB W3: alternative normalizations of the same centroid gap, vs the causal profile")
print("="*100)
print("%-14s %10s %10s %10s %10s %10s" % ("model","raw gap","/harmful","/harmless*","/pooled*","log gap"))
alt_rows=[]
for alias in PUB_GRADED:
    f = os.path.join(REPO,"logs",f"causal_refusalprob_{alias}.json")
    g = os.path.join(REPO,"logs",f"rcs_analysis_{alias}.json")
    if not (os.path.exists(f) and os.path.exists(g)): continue
    d=json.load(open(f)); ra=json.load(open(g))
    layers = sorted(int(k) for k in d["rcs"])
    dp  = np.array([d["delta_p"][str(l)] for l in layers])
    gap = np.array([ra["rcs_metadata"][str(l)]["centroid_distance"] for l in layers])
    sig = np.array([ra["rcs_metadata"][str(l)]["harmful_intra_std"] for l in layers])
    variants = {
        "raw gap":  gap,
        "/harmful": gap/(sig+1e-8),                 # this is RCS (Eq. 2)
        "log gap":  np.log(gap+1e-8),
    }
    vals = {k: stats.spearmanr(v,dp)[0] for k,v in variants.items()}
    print("%-14s %10.3f %10.3f %10s %10s %10.3f" % (
        PAPER[alias], vals["raw gap"], vals["/harmful"], "n/a", "n/a", vals["log gap"]))
    alt_rows.append({"model":PAPER[alias],"rho_raw_gap":round(float(vals["raw gap"]),4),
                     "rho_rcs_div_harmful_spread":round(float(vals["/harmful"]),4),
                     "rho_log_gap":round(float(vals["log gap"]),4),
                     "note":"harmless/pooled spread NOT recoverable: rcs_metadata stores only centroid_distance and harmful_intra_std"})
print("\n* harmless-spread and pooled-spread normalizations cannot be computed: the cached")
print("  rcs_metadata stores only centroid_distance and harmful_intra_std. They would need")
print("  either the sample-level tensors or a re-run of compute_rcs_statistics.")
print("  log gap is a monotone transform of the raw gap, so its Spearman rho is identical by construction.")

print()
print("="*100); print("Table 6 -- metric-robustness diagnostics (Qwen2.5-7B)"); print("="*100)
t6=[]
rb = os.path.join(REPO,"logs","rcs_robustness_noncausal_qwen2-7b.json")
cc = os.path.join(REPO,"runs","qwen2-7b","metric_robustness","cross_cat_probe.json")
if os.path.exists(rb):
    d=json.load(open(rb))
    t6 += [("RCS vs. diff-in-means norm",0.84,d.get("spearman_rcs_norm")),
           ("RCS vs. probe AUC (2000/class)",0.61,d.get("spearman_rcs_probe"))]
    aucs=np.array([d["probe_scores"][k] for k in sorted(d["probe_scores"],key=int)])
    print("  probe AUC range across layers: %.4f - %.4f (n_layers=%d)" % (aucs.min(),aucs.max(),len(aucs)))
if os.path.exists(cc):
    d=json.load(open(cc))
    t6.append(("RCS vs. cross-cat AUC (LOCO-CV)",0.76,d.get("spearman_rho")))
    ma=np.array(d["mean_auc"]); print("  LOCO-CV AUC range: %.4f - %.4f (paper: 0.993-1.000)" % (ma.min(),ma.max()))
for name,pub,rec in t6:
    print("  %-34s published %.2f   artifact %.4f   %s" % (
        name,pub,rec if rec is not None else float('nan'),
        "match" if rec is not None and abs(rec-pub)<0.006 else "CHECK"))

V=os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification")
with open(os.path.join(V,"table3_audit.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(V,"alt_normalization_audit.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(alt_rows[0])); w.writeheader(); w.writerows(alt_rows)
print("\nmax |recomputed - published| across Table 3: %.4f" % max(r["max_abs_dev"] for r in rows))
print("wrote", os.path.join(V,"table3_audit.csv"), "and alt_normalization_audit.csv")
