# Reused and re-derived evidence, with the claim corrections it forces

Every number below comes from an artifact already in this repository or from the submitted
PDFs. Nothing here required a new model run.

---

## 1. The permuted-label null (qp6c W1) — the one genuinely new result

**Closed form.** Under a random relabeling of the pooled representations at a layer into groups
of size `n_f` and `n_l`, with pooled centroid `m` and pooled scatter `tr(S)`:

```
mu_A - mu_B = (N / n_l) (mu_A - m)                        [exact]
E||mu_A - mu_B||^2 = N^2 tr(S) / (n_f n_l (N - 1))        [finite-population]
sigma_A -> (1/N) sum_i ||x_i - m||                        [A is a random subset]
```

The per-layer scale `tr(S)` cancels between numerator and denominator, leaving

```
E[RCS_perm] ~= sqrt(1/n_f + 1/n_l) = 0.0634     for n_f = 260, n_l = 6000 (the sizes actually used)
```

independent of layer, model and representation geometry. **A flat null implies Gini ~ 0 and
n80/L ~ 0.8.** Monte Carlo on concentrated synthetic data confirms the formula to 1.3%.

**Measured null versus observed**, 200 relabelings drawn once and applied at every layer:

| Model | obs mean RCS | null | ratio | obs Gini | null Gini | obs n80/L | null n80/L |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.1977 | 0.06396 | 18.7x | 0.1881 | 0.009 | 0.679 | 0.821 |
| Qwen2.5-3B | 1.2646 | 0.06412 | 19.7x | 0.2078 | 0.0113 | 0.694 | 0.805 |
| Llama-2-7B | 1.6483 | 0.06458 | 25.5x | 0.2 | 0.0145 | 0.656 | 0.81 |
| Llama-3.1-8B | 1.508 | 0.06433 | 23.4x | 0.1925 | 0.0123 | 0.656 | 0.811 |
| Gemma2-9B | 1.2443 | 0.06406 | 19.4x | 0.2256 | 0.011 | 0.643 | 0.808 |
| Mistral-7B-RR | 3.0608 | 0.06808 | 45.0x | 0.2314 | 0.0349 | 0.656 | 0.799 |
| DeepSeek-R1-7B | 0.8189 | 0.06361 | 12.9x | 0.0837 | 0.0036 | 0.75 | 0.821 |
| Ministral-8B | 1.1647 | 0.06383 | 18.2x | 0.1318 | 0.0063 | 0.722 | 0.805 |
| GPT-oss-20B | 0.7916 | 0.06363 | 12.4x | 0.1104 | 0.0041 | 0.75 | 0.833 |

Empirical tail probability was **0.000 for every model on both statistics**: across 200
relabelings none reached the observed Gini or the observed n80/L.

**Two conclusions, both favorable but narrower than "refusal is distributed".**
1. The breadth is real. Every layer's RCS sits 12x-45x above chance, so RCS having mass
   everywhere is not a consequence of its nonnegativity.
2. The profile is **more** concentrated than the null, not indistinguishable from flat. So
   "broadly distributed" should be stated as *distributed but measurably non-uniform*, which
   is also what motivates adaptive weighting in the first place.

**Robustness.** The null's mean is distribution-free; only its spread depends on within-class
geometry. Sweeping the effective rank on Qwen2.5-7B (isotropic, 100, 20) moves the null Gini
upper quantile only from 0.077 to 0.092, against an observed 0.188.

**Scope.** This is a simulation calibrated to the measured per-layer centroid gap and harmful
spread (max calibration error 1.2e-3), not a measurement of the paper's activations, which are
absent. And a label permutation **cannot** address topic, style or source confounding, because
it destroys the class structure outright. That question is addressed separately, and partly
answered, in `matched_controls.md`.

Artifacts: `../results/permutation_null/summary_primary.csv`, `../results/permutation_null/results_primary.json`,
`plots/null_vs_observed_{profiles,concentration}.pdf`.

---

## 2. RQ6 benign-cost attribution is contradicted by Table 5 (qp6c W2, p6Gj W3)

The main paper writes: *"Uniform ablation applies full-strength intervention at every layer,
whereas LAMA reduces intervention at lower-RCS layers. On the aligned Qwen models, this change
preserves or improves attack success while substantially reducing benign perplexity cost."*

LAMA-uniform is the decisive condition: group-wise directions, `alpha = 1` at **every** layer,
i.e. exactly the baseline's intervention strength. Qwen2.5-7B, from Table 5:

| Setting | directions | per-layer strength | PPL |
|---|---|---|---|
| MD | single source layer | 1 everywhere | 366.81 |
| LAMA-uniform | group-wise | **1 everywhere** | **72.11** |
| LAMA-RCS | group-wise | RCS-weighted | 77.64 |
| LAMA-BO | group-wise | BO-weighted | 81.26 |
| Clean | — | — | 82.43 |

The whole 294.70-point reduction is achieved at **unchanged** intervention strength. Adding RCS
down-weighting then **costs** 5.53 points, and BO weighting costs 9.15.

Supplement Fig. 5(b) does not rescue the claim: its exposure score `sum_l alpha_l KL_l` falls
78 -> 47, yet measured perplexity moves the other way (uniform 72.11 < RCS 77.64). Exposure and
measured benign cost are **oppositely ordered** inside the LAMA family. The supplement already
calls Fig. 5(b) "a weighting-only diagnostic"; the main-paper sentence should match.

**Correction to make:** credit the benign-cost reduction to **depth-specific direction
construction**. State that RCS weighting trades a small amount of benign cost for attack
success on Qwen2.5-7B, rather than delivering the perplexity saving.

This is also the evidence-based reply to p6Gj W3 — but note it cuts both ways, see §4.

---

## 3. Confidence intervals do not follow the described procedure (eKk5 W4, qp6c W3)

Supplement C describes a hierarchical paired bootstrap resampling behaviors and then seeds
within behavior. No code implementing that exists in the release; `../run_ablation_study.py` calls
only the behavior-level helpers in `../lama_stats.py`, per seed.

Three candidate procedures, compared against the 16 published HarmBench intervals:

| Procedure | median max deviation |
|---|---|
| A — as the supplement describes (seeds within behaviors) | 2.48 pp |
| B — average seeds first, then resample behaviors | 2.34 pp |
| **C — pool (behavior, seed) pairs as independent units** | **0.28 pp** |

Candidate C reproduces the published intervals (deviations 0.06-0.38 pp on the 13 non-Mistral
cells). It inflates the effective sample size from 159 to 477.

That inflation is not supportable, because the seeds are largely degenerate:

| Model | per-behavior judgments across seeds {1, 2, 42} |
|---|---|
| Qwen2.5-7B | **identical** for all four methods |
| Gemma2-9B | **identical** for all four methods |
| Qwen2.5-3B | differ on 11-18 of 159 behaviors |
| Mistral-7B-RR | seeds 1 and 2 identical; seed 42 is 0.0 for every method |

With greedy decoding and a direction collection held fixed across configurations, this is
expected. The consequence is that the published intervals are roughly sqrt(3) too narrow.

**Correction to make:** recompute intervals with the behavior as the unit, and either drop the
"hierarchical" description or implement it.

---

## 4. Which significance claims survive (Tables 2 and 4)

Behavior-level paired bootstrap on seed-averaged per-behavior outcomes, n = 159:

| Model | Variant vs MD | Published | Recomputed delta [95% CI] | p | Verdict |
|---|---|---|---|---|---|
| Qwen2.5-7B | LAMA-uniform | +3.8 (dagger, p<0.10) | +3.77 [-3.14, +10.69] | 0.310 | **does not hold** |
| Qwen2.5-7B | LAMA-RCS | +8.8 (star, p<0.05) | +8.81 [+2.52, +15.09] | 0.006 | **holds** |
| Qwen2.5-7B | LAMA-BO | +1.9 (n.s.) | +1.89 [-4.40, +8.18] | 0.610 | holds (n.s.) |
| Qwen2.5-3B | LAMA-uniform | +3.4 (dagger) | +3.35 [-1.47, +8.18] | 0.176 | **does not hold** |
| Qwen2.5-3B | LAMA-RCS | +2.9 (n.s.) | +2.94 [-2.52, +8.39] | 0.291 | holds (n.s.) |
| Qwen2.5-3B | LAMA-BO | +5.5 (star) | +5.45 [+0.42, +10.69] | 0.035 | **holds** |
| Gemma2-9B | LAMA-uniform | -27.0 (star) | -27.04 [-34.59, -20.13] | <1e-4 | **holds** |
| Gemma2-9B | LAMA-RCS | -45.3 (star) | -45.28 [-53.46, -37.11] | <1e-4 | **holds** |
| Gemma2-9B | LAMA-BO | -40.3 (star) | -40.25 [-47.80, -32.70] | <1e-4 | **holds** |

**Point estimates reproduce exactly.** Both starred positive claims survive, and all three
Gemma negative results survive decisively. Only the two marginal daggers fail.

This sharpens RQ5. The paper decomposes Qwen2.5-7B as "+3.8 from group directions, +5.0 more
from RCS weighting, so both components contribute". The group-direction step alone is **not**
distinguishable from MD (p = 0.31); only the full LAMA-RCS configuration is. Combined with §2,
the defensible statement is:

> On Qwen2.5-7B, depth-specific direction construction delivers the benign-cost reduction but
> no significant attack-success gain on its own; RCS weighting is what produces the significant
> gain, at a small benign-cost premium.

That is a direct, evidence-backed answer to p6Gj W3, and it is stronger than the paper's
current "both contribute" phrasing because each half is separately supported.

---

## 5. What the denominator adds beyond the raw norm (hgpB)

hgpB asks what the "additional information beyond the raw centroid norm" actually is. It is the
removal of a depth trend that the gap and the spread share.

Spearman correlation with layer index, all nine models:

| Quantity | rho with depth | dynamic range (max/min) |
|---|---|---|
| raw centroid gap | 0.909 - 1.000 | 84x - 19389x |
| harmful intra-class spread | 0.906 - 1.000 | 54x - 4648x |
| **RCS (their ratio)** | **0.195 - 0.961** | **1.7x - 7.4x** |

Both the numerator and the denominator grow almost monotonically with depth across one to four
orders of magnitude; their ratio does not. The clearest case is Llama-2-7B, where the raw gap
correlates **-0.07** with the causal profile while RCS reaches **0.466**, rising to **0.625**
after controlling for the gap (Table 3).

**Scope to keep:** this is a residual predictive association on some models, not proof of
refusal specificity and not a general causal ranking. On Gemma2-9B the partial correlation is
**-0.219**, so the residual carries no causal signal there.

**Not computable from the release:** normalizing by the harmless spread or the pooled spread.
`rcs_metadata` stores only `centroid_distance` and `harmful_intra_std`. Those variants need the
sample-level tensors. A log-gap variant is a monotone transform of the raw gap, so its Spearman
value is identical by construction and adds nothing.

Artifact: `../results/verification/alt_normalization_audit.csv`.

---

## 6. Llama-2-7B baseline audit (qp6c W4)

The reviewer calls the 5.7% reproduced MD ASR implausibly weak. Three **confirmed** protocol
differences, from the release:

1. **Source layer.** The reproduction selects `l* = 11`; the MD paper reports layer **13** for
   LLaMA2-7B (README baseline table). Only `som4x4_sigma0.33_layer11_directions.pt` exists.
2. **Direction identities.** The baseline reports IDs `8-11-4-7-13-6-12`; this run's BO selected
   `{group0: [5,15,11], group1: [9,2,2], group2: [0,13,9]}`.
3. **Degenerate optimization.** `best_score = 0.0` across all 48 trials — no trial found a
   single validation jailbreak, so the Llama-2-7B layer weights are effectively arbitrary.

Also relevant: the baseline's published 59.11% is on its own evaluation protocol, whereas this
paper evaluates 159 HarmBench test behaviors under HarmBench-Llama-2-13B.

**Not diagnosed:** whether the chat template, system prompt, decoding settings or judge
threshold also differ. Those cannot be checked without the baseline's own code and outputs. The
honest statement is that the source layer, direction identities and a degenerate BO objective
are confirmed differences, the remainder is **NOT_DIAGNOSED**, and the Llama-2-7B row therefore
carries very little weight for method ranking. The release is also missing the three LAMA
HarmBench eval files for this model, so its HB row is not reproducible from the artifacts.

---

## 7. How many independent alignment points support RQ4 (qp6c W3, eKk5 W3)

Four models enter Fig. 4: Qwen2.5-7B (rho 0.87, +8.8), Qwen2.5-3B (0.75, +2.9), Llama-2-7B
(0.47, at the ASR floor), Gemma2-9B (0.31, -45.3). Of these, one is uninformative at the floor,
two share a model family, and the remaining spread is driven by the Gemma outlier. The seed
degeneracy in §3 means the three seeds do not add independent replication either.

The paper's limitations section concedes a small sample; the abstract states the condition more
firmly. **Recommended wording:** an observed ordering across four models, one of them at the
ASR floor and two from one family, consistent with alignment mattering but not establishing a
general law.
