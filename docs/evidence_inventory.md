# Evidence inventory — AAAI-27 #43976

Repository revision `203160e` (github.com/Abrar2652/causal-lama, 2 commits).
Audit date 2026-10-31. Host ckg10, Python 3.10.12, NumPy 2.2.6, SciPy 1.13.1.

Columns: **Concern** / **Reviewer** / **Existing source** / **New data truly missing?** /
**Action taken** / **Blocker** / **Artifact**.

## A. Answered by new evidence produced here

| Concern | Reviewer | Existing source | New data missing? | Action | Blocker | Artifact |
|---|---|---|---|---|---|---|
| Gini and 80%-coverage under permuted labels | qp6c W1 | none — no null reported | Yes; needs sample-level activations, which the release omits | Derived a closed-form null, ran a calibrated 200-relabeling test for all 9 models, then **regenerated the activations and ran the empirical 200-relabeling null** on Qwen2.5-7B | — resolved; the closed form it validates (to 0.95%) is what carries the other 8 models | `permutation_null.md`, `scripts/permutation_null_*.py` |
| Is the benign-cost gain caused by RCS down-weighting? | qp6c W2, p6Gj W3 | Table 5, supplement Fig. 5 | No — existing tables settle it | Decomposed the attribution; found the paper's mechanism claim contradicted by its own Table 5 | — | `../results/verification/table5_audit.csv` |
| What does RCS add beyond the raw centroid norm? | hgpB W3/Q | Table 3, Table 6 | No | Quantified the depth trend the denominator removes, across all 9 models | — | `reused_evidence.md` §5 |
| Llama-2-7B baseline is implausibly weak | qp6c W4 | Table 2, baseline repository | No | **Fully diagnosed** against github.com/pralab/som-refusal-directions: wrong direction ids and wrong source layer; decoding/judge/template ruled out | — | `baseline_fidelity.md` |
| How many independent alignment points? | qp6c W3, eKk5 W3 | Table 3, Fig. 4 | No | Counted; also found the seed replication is largely degenerate | — | `../results/verification/seed_identity_audit.csv` |

## B. Verification of what the paper already reports

| Published object | Reproduced from cache? | Max deviation | Artifact |
|---|---|---|---|
| Table 1, all 9 models | **Yes, exactly** (L, peak, n80 exact) | Gini 0.0005, peak/mean 0.004, CV 0.005 | `../results/verification/table1_audit.csv` |
| Eq. (2) identity RCS = gap / (spread + eps) | **Yes** | 2.1e-7 (float32) | same |
| Table 3, graded and binary rows | **Yes, exactly** | 0.0004 | `../results/verification/table3_audit.csv` |
| Table 6 diagnostics | **Yes** | 0.004 | `../results/verification/table3_audit.csv` |
| Table 2 point ASRs, 4 of 5 models | **Yes** | 0.05 pp | `../results/verification/table2_audit.csv` |
| Table 2 point ASRs, Mistral-7B-RR HarmBench | **No** — published column is a 2-seed mean | 6.7 pp | `../results/verification/mistral_per_seed_audit.csv` |
| Table 4 component deltas | **Yes** | 0.05 pp | `../results/verification/significance_multiseed_audit.csv` |
| Table 5, all cells | **Yes** | 0.005 PPL | `../results/verification/table5_audit.csv` |
| Table 2 confidence intervals | **No** — not the procedure the supplement describes | see §C | `../results/verification/ci_procedure_audit.csv` |

## C. Discrepancies found between the release and the manuscript

| # | Finding | Evidence |
|---|---|---|
| **0a** | **The paper misstates its data.** Main Sec. 5 and supplement C state 4000 SORRY-Bench harmful prompts; all nine extraction logs record `[processed source] harmful=260 harmless=6000`. Regenerating with the stated setup gives a profile anti-correlated with Table 1 (Spearman -0.78, peak layer 6 vs 19). Table 1's intervals rest on **260** harmful samples | `data_source.md` |
| **0b** | **The MD baseline is not the published MD baseline.** Direction ids `[0..6]` instead of the baseline's BO-selected sets on all five models; source layer differs on three (Llama-2-7B 11 vs 13, Qwen2.5-3B 23 vs 24, Qwen2.5-7B 15 vs 18). Mean reproduction gap **-18.7 pp**; our best LAMA is below published MD on **5 of 5** models | `baseline_fidelity.md`, `../results/verification/baseline_fidelity_audit.csv` |
| 1 | Supplement C states **128 BO trials**; every result file records **48** (`*_results_t48.json`, `n_trials: 48`), for all five models | `runs/*/lama_optimizations/` |
| 2 | Llama-2-7B BO `best_score = 0.0` — no trial found any validation jailbreak, so its layer weights carry no optimization signal | `runs/llama2-7b/lama_optimizations/` |
| 3 | Table 2 CIs match pooling (behavior, seed) pairs as **independent** units (median deviation 0.28 pp) rather than the hierarchical scheme the supplement describes (2.48 pp). No hierarchical-bootstrap code exists in the release | `../results/verification/ci_procedure_audit.csv`, `../results/verification/ci_candidateC_audit.csv` |
| 4 | Seeds are largely degenerate: Qwen2.5-7B and Gemma2-9B give **identical** per-behavior judgments across all three seeds; so pooling inflates n from 159 to 477 without new information | `../results/verification/seed_identity_audit.csv` |
| 5 | Mistral-7B-RR HarmBench column averages seeds {1,2} only (seed 42 is 0.0 for every method), while the caption claims three seeds and the Guard-2 column uses all three | `../results/verification/mistral_per_seed_audit.csv` |
| 6 | The two marginal daggers (LAMA-uniform on both Qwen models) do not survive a behavior-level paired test (p = 0.31 and 0.18) | `../results/verification/significance_multiseed_audit.csv` |
| 7 | Two disagreeing clean references for Qwen2.5-7B: PPL 82.43 (used by Table 5) and 133.18. The stated 4.5x MD multiplier becomes 2.75x against the second | `../results/verification/table5_audit.csv` |
| 8 | Missing release artifacts: Llama-2-7B LAMA HarmBench evals (3 Table 2 cells), and the Qwen2.5-3B LAMA-uniform/RCS utility JSONs (traceable only to a log line) | `../results/verification/table2_audit.csv`, `../results/verification/table5_audit.csv` |
| 9 | Supplement reports 8x RTX A6000 48 GB, driver 535.274.02; the current host has 8x RTX A5000 24 GB, driver 580.178.04, so timings are not comparable here | `execution_summary.md` |

## D. Requested but NOT delivered (and why)

| Request | Reviewer | Status | Prerequisite |
|---|---|---|---|
| Permutation null on the paper's own activations | qp6c W1 | **DONE** | observed 18.7x the null, tail p = 0.000 on both concentration statistics; `permutation_null.md` |
| Refusal-specific matched controls | PKr6, p6Gj W1 | **DONE (partial)** | length and register matched away without moving the profile (peak stays L19-20, rho >= 0.974); 8 non-refusal contrasts with bootstrap depth intervals. No single statistic is refusal-specific: a benign register contrast reaches 11.0x the null and ties the published Gini, and a pure random split correlates with the published profile at +0.844. What is specific is the *combination* of large magnitude with late mass: refusal's center of mass, 0.582, is disjoint from all eight benign intervals (widest reaches 0.531), a margin of ~1.3 layers. `matched_controls.md`. Topic matching is keyword based because the category field is uniformly "None" |
| Base-versus-aligned RCS profile | qp6c W1 | **DONE** | `Qwen/Qwen2.5-7B` vs `-Instruct`, identical prompts and token ids, **full published class sizes 260/6000**, fp16 sdpa on Ampere, 0/339 weights mutated in either, aligned arm reproducing the published profile at Spearman 1.0000 (max|diff| 0.0002). Splits the claim: base reaches 60.2% of aligned mean RCS at 11.4x the null (so magnitude is **not** refusal-specific) but only 52.3% of the Gini with com 0.537 [0.531, 0.542] overlapping chance, while alignment's own increment runs from ~0 in layers 0-4 to +1.005 at L18 (so the late **concentration** is alignment's). `matched_controls.md` Part 4 |
| Joint / multi-layer causal interventions | PKr6 | **DONE** | ablation of layer *sets* on the paper's own graded metric; the harness reproduces the published single-layer probe to 2.8e-04 and restores to baseline exactly. Best single layer = 25.1% of the joint effect; joint/sum-of-singles = 1.30 at k=2; RCS beats raw norm 8x and the best of 5 random draws 3.1x at k=3; a random direction at the same layers moves refusal < 0.001; harmless entropy +0.05 nats at worst. **Narrows the title**: 3 layers reach 86.6% of the achievable effect, so causal concentration is far tighter than the representational n80/L = 0.679. `joint_causal.md` |
| Broader benchmarks, larger benign sets, more models | eKk5, PKr6 | **DEFERRED** | new full-pipeline runs |
| Cheap alignment proxy that predicts divergence | qp6c W5, p6Gj W4 | **NOT ESTABLISHED** | none found; do not claim one exists |

## E. Drafted author response

the drafted author response (2471 chars for one response box, plus a per-reviewer variant), which is kept unpublished
(five self-contained blocks, 1340-2471 chars each). Both are URL-free. Every number in them is
checked against its source artifact by `../analysis/verify_rebuttal_claims.py` (20/20).

Two cautions are recorded with that draft: the AAAI-27 feedback window is stated
inconsistently on AAAI's own pages (Oct 19-25 in the review-process prose, Oct 24-31 in the
timetable, deadlines anywhere-on-earth), and the AAAI-27 rebuttal FAQ is a 404, so the
2500-character target is AAAI-26's cap rather than a confirmed AAAI-27 one.
