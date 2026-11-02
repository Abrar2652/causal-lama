#!/usr/bin/env python3
"""Audit published Table 5 (benign capability cost) and decompose the RQ6 attribution that
reviewer qp6c challenges in W2 (and p6Gj in W3).

The question: is the benign-perplexity reduction relative to MD caused by RCS DOWN-WEIGHTING
of low-RCS layers, or by DEPTH-SPECIFIC DIRECTION CONSTRUCTION?

LAMA-uniform is the decisive condition. It uses the group-wise directions but sets alpha=1
at every layer, so it intervenes at exactly the baseline's strength everywhere. Any benign
cost it avoids cannot be due to reduced intervention strength.
"""
import json, os, sys, csv
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(HERE))

PUB5 = {
 "Qwen2.5-7B": {"Clean":(82.43,None),"MD":(366.81,+1.493),"LAMA-uniform":(72.11,-0.134),
                "LAMA-RCS":(77.64,-0.060),"LAMA-BO":(81.26,-0.015)},
 "Qwen2.5-3B": {"Clean":(71.87,None),"MD":(188.27,+0.963),"LAMA-uniform":(90.62,+0.232),
                "LAMA-RCS":(90.59,+0.232),"LAMA-BO":(91.07,+0.237)},
}
rows=[]

def add(model, setting, pub_ppl, pub_dnll, got_ppl, src, note=""):
    d = None if (got_ppl is None or pub_ppl is None) else round(abs(got_ppl-pub_ppl),3)
    rows.append({"model":model,"setting":setting,"published_ppl":pub_ppl,
                 "artifact_ppl":None if got_ppl is None else round(got_ppl,4),
                 "abs_diff":d,"published_dNLL":pub_dnll,"source_artifact":src,"note":note})

# ---- Qwen2.5-7B
a42=json.load(open(f"{REPO}/runs/qwen2-7b/ablation_study/ablation_seed42.json"))
cmu=json.load(open(f"{REPO}/runs/qwen2-7b/ablation_study/clean_md_utility.json"))
mdu=json.load(open(f"{REPO}/runs/qwen2-7b/ablation_study/md_utility.json"))
add("Qwen2.5-7B","Clean",82.43,None,a42["utility_clean"]["perplexity"],"ablation_seed42.json:utility_clean")
add("Qwen2.5-7B","MD",366.81,+1.493,mdu["perplexity"],"md_utility.json")
for k,lbl in [("uniform","LAMA-uniform"),("rcs","LAMA-RCS"),("bo","LAMA-BO")]:
    add("Qwen2.5-7B",lbl,PUB5["Qwen2.5-7B"][lbl][0],PUB5["Qwen2.5-7B"][lbl][1],
        cmu[k]["perplexity"],"clean_md_utility.json")
# ---- Qwen2.5-3B
b42=json.load(open(f"{REPO}/runs/qwen2-3b/ablation_study/ablation_seed42.json"))
bmd=json.load(open(f"{REPO}/runs/qwen2-3b/ablation_study/md_utility.json"))
add("Qwen2.5-3B","Clean",71.87,None,b42["utility_clean"]["perplexity"],"ablation_seed42.json:utility_clean")
add("Qwen2.5-3B","MD",188.27,+0.963,bmd["perplexity"],"md_utility.json")
add("Qwen2.5-3B","LAMA-BO",91.07,+0.237,b42["utility_lama_bo"]["perplexity"],"ablation_seed42.json:utility_lama_bo")
for lbl,val in [("LAMA-uniform",90.622),("LAMA-RCS",90.592)]:
    add("Qwen2.5-3B",lbl,PUB5["Qwen2.5-3B"][lbl][0],PUB5["Qwen2.5-3B"][lbl][1],val,
        "logs/chain_qwen2-3b.log (log line only)",
        "NO RELEASED JSON: written to runs/qwen2-3b/lama_utility/, a directory absent from the repo")

print("%-12s %-14s %12s %12s %9s  %s" % ("model","setting","published","artifact","|diff|","source"))
for r in rows:
    print("%-12s %-14s %12s %12s %9s  %s" % (r["model"],r["setting"],r["published_ppl"],
          r["artifact_ppl"],r["abs_diff"],r["source_artifact"]))

print("\n" + "="*96)
print("RQ6 / qp6c-W2: what actually drives the benign-cost reduction on Qwen2.5-7B?")
print("="*96)
md, uni, rcs, bo, clean = 366.81, 72.11, 77.64, 81.26, 82.43
print(f"  MD (single-layer directions, alpha=1 everywhere)       PPL {md:7.2f}")
print(f"  LAMA-uniform (GROUP directions, alpha=1 everywhere)    PPL {uni:7.2f}   <- same strength as MD")
print(f"  LAMA-RCS     (group directions, RCS-weighted alpha)    PPL {rcs:7.2f}")
print(f"  LAMA-BO      (group directions, BO-weighted alpha)     PPL {bo:7.2f}")
print(f"  Clean                                                  PPL {clean:7.2f}")
print()
print(f"  reduction MD -> LAMA-uniform, at IDENTICAL intervention strength : {md-uni:7.2f} PPL  "
      f"({100*(md-uni)/(md-uni):.0f}% of the total)")
print(f"  further change from adding RCS down-weighting                    : {uni-rcs:+7.2f} PPL  "
      f"(RCS weighting makes benign cost WORSE, not better)")
print(f"  further change from adding BO weighting                          : {uni-bo:+7.2f} PPL")
print()
print("  => The entire MD->LAMA benign-cost reduction is already achieved by LAMA-uniform,")
print("     which applies full strength at every layer. Re-weighting by RCS costs +5.53 PPL")
print("     relative to uniform. The reduction is attributable to depth-specific direction")
print("     construction, NOT to reduced intervention at low-RCS layers.")
print()
print("  Cross-check against supplement Figure 5(b), the weighting-only exposure diagnostic:")
print("     exposure sum_l alpha_l*KL_l : MD-schedule 78  ->  LAMA-RCS schedule 47  (-40%)")
print("     but measured perplexity     : uniform 72.11   <  LAMA-RCS 77.64")
print("     The exposure score and the measured benign cost are ordered OPPOSITELY within the")
print("     LAMA family, so Figure 5(b) does not license the Table 5 mechanism claim.")

print("\n" + "="*96)
print("Clean-reference inconsistency on Qwen2.5-7B (affects the stated multiplier and dNLL)")
print("="*96)
print(f"  ablation_seed42.json utility_clean : PPL {a42['utility_clean']['perplexity']:.2f}  "
      f"NLL {a42['utility_clean']['mean_nll']:.4f}   <- the reference Table 5 uses")
print(f"  clean_md_utility.json / _v4 clean  : PPL {cmu['clean']['perplexity']:.2f}  "
      f"NLL {cmu['clean']['mean_nll']:.4f}   <- a second, disagreeing clean run")
print(f"  ratio MD/clean is {md/a42['utility_clean']['perplexity']:.2f}x against the first and "
      f"{md/cmu['clean']['perplexity']:.2f}x against the second (paper states 4.5x).")
print("  The supplement says the 200 benign responses are generated once and held fixed, so a")
print("  single clean NLL should exist. Both files are in the release; the authoritative one")
print("  should be documented and the other removed or explained.")

out=os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification","table5_audit.csv")
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok=[r for r in rows if r["abs_diff"] is not None]
print("\ncells checked: %d   max |diff|: %.3f PPL" % (len(ok), max(r["abs_diff"] for r in ok)))
print("wrote", out)
