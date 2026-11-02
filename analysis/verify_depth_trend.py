#!/usr/bin/env python3
"""What the RCS denominator removes: the shared depth trend (reviewer hgpB).

Saves the numbers cited in reused_evidence.md section 5.
"""
import json, os, sys, csv
import numpy as np
from scipy import stats
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if False else \
       os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "/nas/home/jahin/causal-lama"

MODELS = [("qwen2-7b","Qwen2.5-7B"),("qwen2-3b","Qwen2.5-3B"),("llama2-7b","Llama-2-7B"),
          ("gemma2-9b","Gemma2-9B"),("mistral-7b-rr","Mistral-7B-RR"),("llama31-8b","Llama-3.1-8B"),
          ("deepseek-r1-qwen-7b","DeepSeek-R1-7B"),("ministral-8b","Ministral-8B"),("gpt-oss-20b","GPT-oss-20B")]
rows=[]
print("%-16s %18s %18s %16s | %12s %12s %10s" % (
    "model","rho(gap,depth)","rho(spread,depth)","rho(RCS,depth)","gap max/min","spread max/min","RCS max/min"))
for a,n in MODELS:
    d=json.load(open(os.path.join(REPO,"logs",f"rcs_analysis_{a}.json")))
    L=sorted(int(k) for k in d["rcs_scores"]); idx=np.array(L,float)
    gap=np.array([d["rcs_metadata"][str(l)]["centroid_distance"] for l in L])
    sp =np.array([d["rcs_metadata"][str(l)]["harmful_intra_std"] for l in L])
    r  =np.array([d["rcs_scores"][str(l)] for l in L])
    g,s,c = (stats.spearmanr(gap,idx)[0], stats.spearmanr(sp,idx)[0], stats.spearmanr(r,idx)[0])
    print("%-16s %18.3f %18.3f %16.3f | %12.1f %12.1f %10.2f" % (
        n,g,s,c,gap.max()/gap.min(),sp.max()/sp.min(),r.max()/r.min()))
    rows.append({"model":n,"n_layers":len(L),
                 "spearman_gap_vs_depth":round(float(g),4),
                 "spearman_harmfulspread_vs_depth":round(float(s),4),
                 "spearman_rcs_vs_depth":round(float(c),4),
                 "gap_dynamic_range":round(float(gap.max()/gap.min()),1),
                 "spread_dynamic_range":round(float(sp.max()/sp.min()),1),
                 "rcs_dynamic_range":round(float(r.max()/r.min()),2)})
out=os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification","depth_trend_audit.csv")
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
g=[r["spearman_gap_vs_depth"] for r in rows]; s=[r["spearman_harmfulspread_vs_depth"] for r in rows]
c=[r["spearman_rcs_vs_depth"] for r in rows]
print("\nranges cited in reused_evidence.md section 5:")
print("  rho(gap,depth)    %.3f - %.3f" % (min(g),max(g)))
print("  rho(spread,depth) %.3f - %.3f" % (min(s),max(s)))
print("  rho(RCS,depth)    %.3f - %.3f" % (min(c),max(c)))
print("  gap range    %.0fx - %.0fx" % (min(r['gap_dynamic_range'] for r in rows), max(r['gap_dynamic_range'] for r in rows)))
print("  spread range %.0fx - %.0fx" % (min(r['spread_dynamic_range'] for r in rows), max(r['spread_dynamic_range'] for r in rows)))
print("  RCS range    %.1fx - %.1fx" % (min(r['rcs_dynamic_range'] for r in rows), max(r['rcs_dynamic_range'] for r in rows)))
print("wrote", out)
