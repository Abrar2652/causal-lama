# Joint multi-layer causal ablation — the experiment the title rests on

Reviewer PKr6 asked for it, the first audit pass deferred it, and the 2026-11-01 decision to
**keep the causal framing** makes it load-bearing: it is the only evidence that can support
"refusal **is** layer-distributed" as a causal claim rather than a representational one.

The gap it closes. The paper's causal evidence (`../causal_refusal_prob.py`) ablates **one layer
at a time** and correlates the per-layer effect with RCS. LAMA, the method the paper proposes,
intervenes at **many layers at once**. A correlation with single-layer effects shows that
high-RCS layers matter individually; it cannot show that refusal is distributed across them.
Those are different claims and only the second one is in the title.

Scripts: `../analysis/joint_causal_ablation.py`, `../analysis/random_direction_control.py`.
Data: `joint_causal_qwen2-7b.{json,csv}`, `random_direction_qwen2-7b.{json,csv}`.
Host ckg10 (Ampere), Qwen2.5-7B-Instruct, fp16, 159 harmbench_test prompts, alpha = 1.0.
Directions and RCS come from `qwen2-7b_processed_confirm`, the regenerated tensors that
reproduce the published per-layer profile to 2.0e-04.

Metric: mean probability mass on refusal tokens at the first generated position, the paper's
own graded metric. **No text is generated**, so no completion exists to judge or to store.

    P_refuse(S) = mean_prompts sum_{t in refusal_ids} P(t | prompt, ablating layer set S)
    DeltaP(S)   = P_refuse({}) - P_refuse(S)

## Guards, all of which pass

| Guard | Result |
|---|---|
| Reproduces the paper's own single-layer probe | Spearman **0.99945**, max abs dev **2.83e-04** |
| Baseline P_refuse vs published | 0.864798 vs 0.864909, dev **1.10e-04** |
| Spearman(RCS, DeltaP) vs published 0.8730 | **0.8719** |
| Partial Spearman(RCS, DeltaP \| raw norm) vs published 0.6847 | **0.6811** |
| Restore-to-baseline after all ablations | **exactly 0.0** |
| Baseline across two independent processes | **exactly 0.00e+00** |
| Damage control: harmless-prompt entropy | 0.572 baseline, **0.624 worst case** over all configurations |

The first guard is the important one: the single-layer arm reproduces
`../logs/causal_refusalprob_qwen2-7b.json` to within 3e-04, so the joint arms measured in the
same process rest on the same footing as the paper's published causal result.

## 1. No single layer carries refusal

Baseline P_refuse = **0.8648**. The best achievable removal is **DeltaP = 0.8589** at k = 12,
which is 99.3% of the baseline: joint ablation can switch refusal off almost completely.

The best **single** layer, L20, reaches **DeltaP = 0.2152 — only 25.1%** of that. The
single-layer effects sum to 1.4823, well above the 0.8648 that exists to be removed, so they
cannot simply be added.

*Two reference points, because the JSON carries both and they differ.* Against the **best
attainable** drop, 0.8589 at k = 12, the best single layer is **25.1%**. Against the
**all-layer** drop, 0.7965 at k = 28, it is **27.0%**, which is the figure stored as
`summary.best_single_share_of_full`. The all-layer value is the smaller denominator only
because the curve declines past k = 12 (see below), so quoting against it would flatter the
single layer. This document quotes the best-attainable reference throughout; the fields
`layers_for_50pct`, `layers_for_80pct` and `layers_for_90pct` in the JSON are computed against
the all-layer reference and read 2, 3 and 3 respectively.

**Single-layer localization is refuted causally.** No layer is the site of refusal.

## 2. The curve: three layers do most of the causal work

Layers added in RCS order (19, 20, 18, 21, 17, 22, 23, 16, ...):

| k | 1 | 2 | **3** | 4 | 5 | 8 | **12** | 20 | 28 |
|---|---|---|---|---|---|---|---|---|---|
| DeltaP | 0.176 | 0.508 | **0.744** | 0.807 | 0.831 | 0.845 | **0.859** | 0.816 | 0.797 |
| % of best achievable | 20.4 | 59.1 | **86.6** | 93.9 | 96.7 | 98.3 | **100** | 95.0 | 92.7 |

**This narrows the paper's claim and the narrowing must be stated.** Three layers — L18, L19,
L20 — reach 86.6% of the achievable effect and four reach 93.9%. The representational
statistic the paper reports, n80/L = 0.679, says 19 of 28 layers are needed to hold 80% of the
RCS mass. **That does not carry over causally: causally it is 3 of 28.** Representational
spread is not causal spread, and the paper cannot let the first stand in for the second.

**More is worse, which is an argument for the method.** Past k = 12 the effect *declines*,
from 0.859 to 0.797 at k = 28. Ablating every layer is strictly worse than ablating the band.
A uniform, all-layer intervention is therefore not the ceiling an adaptive one is chasing —
it is below it. This is the strongest causal argument in the paper's favor for a
layer-selective method, and it is currently not made.

## 3. Layers are complementary, not redundant

Joint effect against the sum of the same layers' individual effects:

| k | 2 | 3 | 5 | 8 | 14 |
|---|---|---|---|---|---|
| joint / sum-of-singles | **1.299** | **1.273** | 1.151 | 0.981 | 0.701 |

At k = 2 and k = 3 the joint effect is about **30% larger than the sum** of the parts, while
still far from the 0.8648 ceiling, so this is genuine synergy and not an artifact of
saturation. Beyond k = 8 the ratio falls below 1 because the ceiling binds.

"Distributed" and "redundant" are different claims and this separates them: refusal is
**jointly** encoded across the late band, not duplicated in each layer independently.

## 4. RCS causally identifies where to intervene

DeltaP at matched k, so every arm pays the same intervention budget:

| k | top-k by **RCS** | bottom-k by RCS | top-k by **raw norm** | random k, mean (best of 5) |
|---|---|---|---|---|
| 1 | **0.1756** | -0.0064 | 0.0247 | 0.0113 (0.0516) |
| 2 | **0.5077** | 0.0010 | 0.0721 | 0.1140 (0.1964) |
| 3 | **0.7436** | 0.0012 | 0.0921 | 0.1247 (0.2388) |
| 5 | **0.8308** | 0.0009 | 0.3548 | 0.2062 (0.3450) |
| 8 | **0.8445** | 0.0127 | 0.7735 | 0.3195 (0.5525) |
| 14 | 0.8546 | 0.1631 | 0.8546 | 0.5640 (0.6695) |

At k = 3, RCS-chosen layers deliver **0.744** against **0.092** for the highest-raw-norm
layers (8.1x), **0.125** for random layers and 0.239 for the best of five random draws (3.1x
the luckiest draw), and **0.001** for the lowest-RCS layers.

The raw-norm arm is the **non-circularity control**: RCS shares its numerator with the ablated
direction, so the question is whether the normalization earns anything. It does, at small k,
by a factor of 8. The two converge at k >= 8, where every rule saturates, and at k = 14 they
are identical. **RCS's causal advantage is real but it lives at small k**, which is exactly the
regime an adaptive method operates in.

## 5. The effect is specific to the refusal direction

Holding the layer set at the top-k by RCS and swapping only the direction for a random unit
vector (the edit magnitude is matched by construction, since the ablation normalizes its input):

| k | 1 | 2 | 3 | 5 | 8 | 14 |
|---|---|---|---|---|---|---|
| real direction | 0.1756 | 0.5077 | 0.7436 | 0.8308 | 0.8445 | 0.8546 |
| random direction | -0.0005 | 0.0006 | 0.0005 | 0.0004 | -0.0001 | 0.0008 |

The random-direction effect is **indistinguishable from zero everywhere**: every value lies
within 0.0008 of no change, with draw-to-draw standard deviation at most 0.0020. Ratios are
reported in the CSV but are not meaningful against a denominator this close to zero; the
honest statement is that a random rank-1 edit at the same layers does nothing at all.

This closes the obvious objection, that a rank-1 projection applied at 14 layers would depress
refusal whatever direction were removed. It would not.

## What the paper may now claim, and what it may not

> **Supported.** Refusal in Qwen2.5-7B-Instruct is causally **joint**: no single layer carries
> more than a quarter of the removable refusal behavior, intervening at two or three layers
> together is about 30% more effective than the sum of those layers alone, and the layers RCS
> ranks highest are the right ones to choose — 8x better than the raw diff-in-means norm and
> 3x better than the best of five random draws at k = 3. The effect is direction-specific: a
> random direction at the same layers moves refusal by less than 0.001. It is not degradation:
> harmless-prompt next-token entropy rises by at most 0.05 nats.

> **Not supported, and the title should not imply it.** That refusal is spread across the
> depth of the network. Causally it is concentrated in L18-L20, which reach 86.6% of the
> achievable effect; a fourth layer brings 93.9%. The paper's representational n80/L = 0.679
> describes the spread of RCS mass, not the spread of causal necessity, and the two differ by
> a factor of six. "Layer-distributed" is defensible as "no single layer suffices and the
> layers act jointly"; it is not defensible as "most layers are involved."

## Limits

One model. Qwen2.5-7B-Instruct is the only cached checkpoint, and it is the model whose
representations reproduce the published profile. The eight others would each need a download
plus roughly 5 minutes of extraction and about 5 minutes of this sweep.

One metric. Refusal-token mass at the first generated position is the paper's own graded
choice, and it is what makes an ASR-free sweep possible on a robustly aligned model, but it is
a proxy for refusal behavior rather than behavior itself. The ASR-based claim would need
generation and a judge.

Fixed alpha. Every arm uses alpha = 1.0. LAMA's per-layer alpha from Bayesian optimization is
not evaluated here, so this measures where refusal lives, not how well LAMA exploits it.

The damage control is a proxy. First-position entropy and top-1 probability on harmless
prompts catch gross degradation, not subtle capability loss. Table 5's perplexity numbers
remain the utility evidence.
