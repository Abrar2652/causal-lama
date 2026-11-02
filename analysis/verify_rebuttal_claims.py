#!/usr/bin/env python3
"""Check every quantitative claim in rebuttal/combined_2500.md against its source file.

Run from the repository root. Exits non-zero if any claim fails, so this can gate a submit.
Each entry names the claim as it appears in the draft, the source artifact, and a tolerance.
"""
import csv
import json
import os
import sys

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(rel):
    with open(os.path.join(ART, rel)) as fh:
        return json.load(fh)


def rows(rel):
    with open(os.path.join(ART, rel)) as fh:
        return list(csv.DictReader(fh))


def close(a, b, tol):
    return a is not None and abs(a - b) <= tol


results = []


def claim(text, got, want, tol=5e-4, src=""):
    results.append((close(got, want, tol), text, got, want, src))


# --- corrections block ---
bva = load("matched_controls/base_vs_aligned_qwen2-7b.json")
meta = bva.get("meta", bva)
claim("260 harmful prompts", float(meta["n_f"]), 260.0, 0, "base_vs_aligned")
claim("6000 harmless prompts", float(meta["n_l"]), 6000.0, 0, "base_vs_aligned")

sig = rows("verification/significance_multiseed_audit.csv")
uni = [r for r in sig if r["model"] == "Qwen2.5-7B" and r["variant"] == "LAMA-uniform"][0]
rcs = [r for r in sig if r["model"] == "Qwen2.5-7B" and r["variant"] == "LAMA-RCS"][0]
claim("LAMA-uniform p=0.31", float(uni["paired_boot_p"]), 0.31, 5e-3, "significance_multiseed")
claim("LAMA-RCS p=0.006", float(rcs["paired_boot_p"]), 0.006, 5e-4, "significance_multiseed")

# --- permutation null: must use the CORRECTED config (260 harmful), not primary (4000) ---
corr = load("permutation_null/results_corrected.json")
claim("null RCS flat at 0.063", corr[0]["analytic_null_mean"], 0.063, 1e-3, "results_corrected")
ratios = [e["observed"]["mean_rcs"] / e["analytic_null_mean"] for e in corr]
claim("observed 12x the null (min)", min(ratios), 12.0, 0.5, "results_corrected")
claim("observed 48x the null (max)", max(ratios), 48.0, 0.5, "results_corrected")
emp = load("permutation_null/results_empirical_qwen2-7b.json")
claim("tail p = 0.000", emp["tail_prob_observed_vs_null"]["peak_over_mean_ge_observed"], 0.0, 0,
      "results_empirical")

# --- specificity ---
nr = load("matched_controls/nonrefusal_controls_qwen2-7b.json")["contrasts"]
null_c = corr[0]["analytic_null_mean"]
claim("benign contrast 9.0x", nr["topic_code"]["mean_rcs"] / null_c, 9.0, 0.1, "nonrefusal_controls")
claim("benign contrast 11.2x", nr["register_question"]["mean_rcs"] / null_c, 11.2, 0.1,
      "nonrefusal_controls")
claim("that contrast's Gini matches 0.1881", nr["register_question"]["gini"], 0.1881, 3e-3,
      "nonrefusal_controls")

# --- joint causal ---
jc = load("joint_causal/joint_causal_qwen2-7b.json")
s, cum, add = jc["summary"], jc["delta_cumulative_rcs"], jc["summary"]["additivity"]
cum = cum if isinstance(cum, dict) else {str(i + 1): v for i, v in enumerate(cum)}
attainable = max(cum.values())
claim("best single layer = 25% of attainable",
      100 * s["best_single_delta"] / attainable, 25.1, 0.1, "joint_causal")
claim("three layers reach 87%", 100 * cum["3"] / attainable, 86.6, 0.1, "joint_causal")
claim("super-additive ~1.3x at k=2",
      add["2"]["joint"] / add["2"]["sum_of_singles"], 1.30, 0.01, "joint_causal")

# --- credit reassignment ---
t5 = open(os.path.join(ART, "verification/table5_audit.csv")).read()
claim("Table 5 LAMA-uniform PPL 72.11", 72.11 if "72.11" in t5 else None, 72.11, 0, "table5_audit")
claim("Table 5 LAMA-RCS PPL 77.64", 77.64 if "77.64" in t5 else None, 77.64, 0, "table5_audit")
claim("RCS weighting costs 5.5 PPL", 77.64 - 72.11, 5.53, 0.01, "table5_audit")

# --- hgpB denominator variants ---
alt = rows("verification/alt_normalizations_qwen2-7b.csv")
variants = [r for r in alt if r["normalization"] != "raw_gap"]
rhos = [float(r["spearman_vs_causal"]) for r in variants]
claim("seven denominator variants", float(len(variants)), 7.0, 0, "alt_normalizations")
claim("variant rho min 0.867", min(rhos), 0.867, 1e-3, "alt_normalizations")
claim("variant rho max 0.873", max(rhos), 0.873, 1e-3, "alt_normalizations")

bad = 0
for good, text, got, want, src in results:
    g = f"{got:.4f}" if isinstance(got, float) else str(got)
    print(f"{'PASS' if good else 'FAIL':5s} {text:40s} got {g:>10s}  want {want:<8} [{src}]")
    bad += not good
print(f"\n{len(results) - bad}/{len(results)} claims verified")
sys.exit(1 if bad else 0)
