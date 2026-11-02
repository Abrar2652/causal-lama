#!/usr/bin/env python3
"""Is the MD baseline in this paper the MD baseline of Piras et al. (2026)?

Compares our reproduced MD against the published configuration from
https://github.com/pralab/som-refusal-directions (the baseline's own repository), which
ships the same datasets, the same judge and the same decoding settings.

Root cause of reviewer qp6c's W4, and it is broader than W4 states.
"""
import json, os, sys, csv
import numpy as np
REPO = "/nas/home/jahin/causal-lama"

# baseline README reproduction table: published source layer, direction ids, ASR on harmbench_test
PUB = {
 "qwen2-7b":     dict(name="Qwen2.5-7B",    l_star=18, ids=[10,3,5,2,4,6,13], asr=95.97),
 "qwen2-3b":     dict(name="Qwen2.5-3B",    l_star=24, ids=[2,7,9,14,8],      asr=93.71),
 "llama2-7b":    dict(name="Llama-2-7B",    l_star=13, ids=[8,11,4,7,13,6,12],asr=59.11),
 "gemma2-9b":    dict(name="Gemma2-9B",     l_star=23, ids=[7,6,15,2,13,3],   asr=96.27),
 "mistral-7b-rr":dict(name="Mistral-7B-RR", l_star=23, ids=[9,5,6,3,1],       asr=25.79),
}
# what this submission used (Table 2 footnote) and what it measured (verified in table2_audit)
OURS = {
 "qwen2-7b":     dict(l_star=15, md=84.28, best_lama=93.08, best_name="LAMA-RCS"),
 "qwen2-3b":     dict(l_star=23, md=83.86, best_lama=89.31, best_name="LAMA-BO"),
 "llama2-7b":    dict(l_star=11, md=5.66,  best_lama=0.00,  best_name="LAMA-* (all at floor)"),
 "gemma2-9b":    dict(l_star=23, md=95.60, best_lama=68.55, best_name="LAMA-uniform"),
 "mistral-7b-rr":dict(l_star=23, md=8.18,  best_lama=20.13, best_name="LAMA-uniform"),
}
MD_DIR_IDS_USED = [0,1,2,3,4,5,6]   # run_ablation_study.py, for every model

rows=[]
print("="*108)
print("MD baseline fidelity: published configuration vs what this submission ran")
print("="*108)
print("%-15s %18s %12s %28s %s" % ("model","source layer","n dirs","direction ids","match?"))
for a,p in PUB.items():
    o=OURS[a]
    lay = f"{o['l_star']} vs {p['l_star']}"
    nd  = f"{len(MD_DIR_IDS_USED)} vs {len(p['ids'])}"
    ids = f"{MD_DIR_IDS_USED} vs {p['ids']}"
    m = "LAYER+IDS DIFFER" if o["l_star"]!=p["l_star"] else "IDS DIFFER"
    print("%-15s %18s %12s %28s %s" % (p["name"], lay, nd, "[0..6] vs published", m))
    rows.append({"model":p["name"],"our_l_star":o["l_star"],"published_l_star":p["l_star"],
                 "layer_matches":o["l_star"]==p["l_star"],
                 "our_dir_ids":"0-1-2-3-4-5-6","published_dir_ids":"-".join(map(str,p["ids"])),
                 "our_n_dirs":len(MD_DIR_IDS_USED),"published_n_dirs":len(p["ids"]),
                 "dir_ids_match":False,
                 "our_md_asr":o["md"],"published_md_asr":p["asr"],
                 "md_reproduction_gap_pp":round(o["md"]-p["asr"],2),
                 "our_best_lama_asr":o["best_lama"],"our_best_lama_variant":o["best_name"],
                 "best_lama_vs_published_md_pp":round(o["best_lama"]-p["asr"],2)})

print()
print("="*108)
print("Consequence: our LAMA against the baseline's PUBLISHED MD, on the identical test set")
print("(same 159 harmbench_test behaviors, same cais/HarmBench-Llama-2-13b-cls judge, same greedy decoding)")
print("="*108)
print("%-15s %14s %14s %12s | %22s %14s" % (
    "model","published MD","our MD","repro gap","our best LAMA","vs published MD"))
for r in rows:
    print("%-15s %13.2f%% %13.2f%% %+11.2f | %10s %10.2f%% %+13.2f" % (
        r["model"], r["published_md_asr"], r["our_md_asr"], r["md_reproduction_gap_pp"],
        r["our_best_lama_variant"][:10], r["our_best_lama_asr"], r["best_lama_vs_published_md_pp"]))
worse = [r for r in rows if r["best_lama_vs_published_md_pp"] < 0]
print()
print(f"Our best LAMA variant is BELOW the baseline's published MD on {len(worse)} of {len(rows)} models.")
print(f"Mean MD reproduction gap: {np.mean([r['md_reproduction_gap_pp'] for r in rows]):+.1f} pp")

out=os.path.join(REPO,"the analysis/, results/ and docs/ trees","verification","baseline_fidelity_audit.csv")
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nwrote", out)
