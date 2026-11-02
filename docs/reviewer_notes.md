# Reviewer-by-reviewer response notes

Working notes for drafting the rebuttal. Each entry separates (a) what is **confirmed from the
paper or the release**, (b) what is **newly measured here**, (c) what remains **unresolved**,
and (d) the **safe next step**. These are not drafted rebuttal prose, and nothing here claims a
rating change. Where the audit found a defect in our own work, it is stated plainly: conceding
a real error early is worth more than defending it.

---

## qp6c — rating 5, confidence 3

This reviewer is the most tractable: they say their central objections need no new experiment
beyond a permutation control, and that control now exists.

**W1 — "distributed" asserted without a null.**
- *Newly measured.* Under permuted labels RCS collapses to a flat **0.0634**, a closed-form
  quantity depending only on the class sizes, because the per-layer scale cancels between
  numerator and denominator. A flat null means **Gini ~ 0 and n80/L ~ 0.8**. Against that, all
  nine observed profiles sit **12x-48x higher in magnitude** and are **more concentrated**
  (Gini 0.084-0.231 vs null 0.010-0.068; n80/L 0.643-0.750 vs null 0.780-0.825), with empirical
  tail probability **0.000** at 200 relabelings. Robust down to effective rank 20. **Quote the
  12x-48x range, not the 38x-148x from `../results/permutation_null/results_primary.json`:** that config assumes the
  paper's stated 4000 harmful prompts, whose null is 0.0204. At the real 260 the null is
  0.0634 and every ratio falls by 3.2x. Pairing 0.0634 with 38x-148x is self-contradictory,
  and a reviewer who evaluates sqrt(1/260 + 1/6000) will see it.
- *Concede.* The null was run on data calibrated to our measured per-layer statistics, not on
  the cached activations themselves, which are not in the release. Say so explicitly.
- *Concede.* The reviewer is right that a permutation null says nothing about topic
  separability. It tests whether any harmful-vs-harmless signal exists, not whether it is
  refusal-specific. The cross-corpus design (SORRY-Bench vs Alpaca) leaves that open.
- *Now run, and it helps us.* The base-versus-aligned comparison they ask for. `Qwen/Qwen2.5-7B`
  against `Qwen/Qwen2.5-7B-Instruct`, identical prompts, identical token ids, at the **full
  published class sizes of 260 and 6000**, with 0/339 parameters mutated in either model and the
  re-extracted aligned arm reproducing the published profile at Spearman
  1.0000. It separates two claims we had been making as one. The base
  model reaches **60.2%** of the aligned mean RCS at **11.4x** the null, so score magnitude is
  not refusal-specific and we must stop implying it is. But that pre-existing separation is
  flat: Gini 0.0984 against 0.1881, n80/L *higher* at 0.786, and center of mass 0.537
  [0.531, 0.542] overlapping the chance interval [0.482, 0.531]. Alignment's own contribution
  is near zero in layers 0-4 (negative at 0, 2 and 3) and peaks at **+1.005 at layer 18**. So
  the late concentration the paper reports is created by alignment. See
  `matched_controls.md` Part 4.
- *Next step.* Add the null to the paper. On the title, the 2026-11-01 decision is to **keep the
  causal framing**, which the joint multi-layer ablation now supports directly: no single layer
  carries more than 25.1% of the removable refusal and two or three layers act super-additively
  (`joint_causal.md`). What must change is *breadth*, not *causality*. Say refusal is
  **jointly encoded across a late band**, since three layers reach 86.6% of the attainable causal
  effect, and stop implying it is spread across the network, which the representational
  n80/L = 0.679 would suggest but the causal evidence contradicts by a factor of six.

**W2 — benign-cost gain miscredited.**
- *The reviewer is correct, and our own Table 5 proves it.* LAMA-uniform applies full strength
  at every layer and reaches PPL 72.11, below LAMA-RCS at 77.64 and far below MD at 366.81. The
  entire reduction comes from depth-specific direction construction; RCS weighting costs 5.53
  points on top. Supplement Fig. 5(b)'s exposure score (78 -> 47) is ordered **oppositely** to
  measured perplexity within the LAMA family, so it cannot support the mechanism claim.
- *Next step.* Rewrite the RQ6 mechanism sentence. No new run.

**W3 — the alignment claim rests on three informative points.**
- *Confirmed*, and worse than stated: the three seeds give **identical** per-behavior judgments
  on Qwen2.5-7B and Gemma2-9B, so they add no independent replication.
- *Next step.* Match the abstract's wording to the limitations section.

**W4 — Llama-2-7B baseline undiagnosed.**
- *Fully diagnosed, and the reviewer deserves a direct answer.* Comparing against the baseline's
  own repository (identical datasets, judge and decoding), the gap has two identified causes,
  and neither is the template, decoding or judge version they hypothesized. Our MD control takes
  the top-7 SOM directions by harmful-win count rather than the authors' Bayesian-optimization
  selected set, and selects the source layer by refusal score alone, giving layer 11 where the
  KL-constrained rule gives the published 13.
- *Position.* We keep this control, because every condition in our study is matched to it, and
  we state plainly in the paper that it is a matched control rather than a reproduction. We
  report the published numbers alongside so the reviewer can see the difference.
- *Tone.* Answer this one by conceding and disclosing, not by defending. The reviewer asked
  exactly the right question and we now have the exact answer.

**W5 — knowing when to trust RCS needs the expensive sweep.**
- *Concede.* No cheap validated gating criterion was found and none should be claimed. The
  fallback-to-MD-on-divergence idea they propose is sensible and untested.

---

## PKr6 — rating 6, confidence 3

**Matched controls for refusal specificity.**
- *Resolved, and it reshapes the claim.* Six matched controls and five non-refusal contrasts
  were run; see `matched_controls.md`. Matching away prompt length and imperative
  register leaves the profile intact: peak stays at L19-20, rho >= 0.974 against the published
  profile, margin over the null 15-19x. A length contrast tuned to the real Cliff's delta of
  0.148 reaches only 1.3x the null, so length is eliminated outright.
- *What the controls cost us.* Magnitude and concentration are **not** refusal-specific. A
  code-vs-prose contrast built inside held-out harmless prompts reaches 9.0x the null and a
  question-vs-imperative contrast 11.2x, and the latter's Gini **matches** the published 0.1881.
  Say "matches", not "exceeds": the two draws of that contrast give 0.1908 and 0.1877, which
  straddle 0.1881, so "exceeds" is draw-dependent and a reviewer re-running may get the other
  side. "Matches" is true either way and makes the same point. Do not argue refusal specificity from a high RCS or a concentrated profile.
- *What they buy us.* A joint statement, not a single statistic. Refusal's center of mass sits
  at 0.582 of normalized depth with a bootstrap interval of [0.577, 0.587], disjoint from all
  eight benign contrasts, the highest of which reaches 0.519. The benign contrast that comes
  closest in magnitude, 11.0x, places its mass *earliest* of all at 0.408. Refusal is the only
  contrast that is both large and late. Argue that conjunction, not depth alone.
- *Two sentences we must not use.* An earlier version of these notes claimed every non-refusal
  contrast peaks at layers 4-5 and anti-correlates with the published profile. Both are false.
  Deep benign contrasts peak at layers 17, 18 and 21, and four of seven correlate *positively*
  with the published profile. Worst of all, a pure random split at 1.1x the null reaches
  rho = +0.844, because any two profiles sharing the common depth trend correlate. **Never cite
  rho against the published profile as evidence of refusal specificity**; a reviewer who checks
  it will find it is satisfied by noise.
- *What the deep contrasts do show.* Semantic depth alone raises the center of mass only to
  about 0.51, exactly where the random split sits, and produces flat profiles, Gini 0.035-0.086
  against the published 0.188. The benign moral contrast, topically adjacent to harm, reaches
  only 3.3x the null. Harm-topic adjacency and reasoning depth are therefore not sufficient to
  reproduce the profile, which is the specific alternative explanation qp6c and PKr6 raise.
- *Size the effect honestly.* Refusal's mass is roughly 1.3 layers later than chance. The
  intervals are disjoint over 1000 resamples, but this is not a large effect, and calling it
  one invites the obvious check.
- *Still genuinely unavailable.* Real topic labels, since the `category` field is uniformly
  `"None"` and the topic split is therefore keyword-based and reported as coarse. The
  base-versus-aligned comparison qp6c also requests is now **run**; see above and Part 4 of
  `matched_controls.md`.

**Multi-layer / joint causal interventions.** *Run, after the 2026-11-01 decision to keep the
causal framing made this the experiment the title stands on.* See `joint_causal.md`.

- *Concede the distinction first, then answer it.* The reviewer is right that the published
  probe ablates one layer at a time while LAMA intervenes jointly, so a correlation with
  single-layer effects could only show that representation is distributed, not control. We
  therefore ablated layer **sets**. The harness reproduces the paper's own single-layer probe to
  `max|diff| = 2.8e-04` (Spearman 0.99945) and returns to baseline exactly, so the joint arms
  stand on the published result's footing rather than on a new pipeline.
- *The headline.* No single layer carries refusal: the best one, L20, removes **25.1%** of what
  joint ablation removes (baseline P_refuse 0.8648, best attainable drop 0.8589 at k = 12).
  Single-layer localization is refuted causally rather than by inference.
- *The mechanism.* Two or three layers together exceed the **sum** of their individual effects by
  about 30% (joint/sum = 1.30 at k = 2, 1.27 at k = 3), while still far from the ceiling, so this
  is genuine synergy. Refusal is **jointly** encoded, not duplicated layer by layer. This is the
  sentence that separates "distributed" from "redundant", and we should use it.
- *RCS is causally load-bearing, at matched budget.* At k = 3, RCS-chosen layers remove 0.7436
  against **0.0921** for the top-k by raw diff-in-means norm (8.1x), 0.1247 for random layers,
  0.2388 for the luckiest of five draws, and 0.0012 for the bottom-k. The raw-norm arm is the
  non-circularity control, so this is the direct answer to "RCS is just the ablated direction's
  norm".
- *And it is the direction, not the damage.* Holding the layer set fixed and substituting a
  random unit direction moves refusal by **< 0.001** at every k. Harmless-prompt next-token
  entropy rises at most 0.05 nats across all 99 configurations, so none of this is degradation.
- *The honest narrowing, which we must state before a reviewer does.* Three layers (L18-L20)
  reach **86.6%** of the attainable effect and four reach 93.9%, while the representational
  n80/L = 0.679 would imply 19 of 28. Causal concentration is about six times tighter than
  representational spread. Claim "no single layer suffices and the layers act jointly"; do not
  claim "refusal is distributed across the network", and do not let n80/L stand in for a causal
  statement.
- *A gift we are currently not using.* Past k = 12 the effect **declines**, 0.859 to 0.797 at
  k = 28, so ablating every layer is strictly worse than ablating the band. That is a causal
  argument in favor of a layer-selective method, i.e. for LAMA's premise, and it is stronger
  than anything in the current draft. It also means "more ablation is better" is false.
- *Limits to state.* One model, alpha fixed at 1.0 so LAMA's optimised per-layer alphas are not
  evaluated, and refusal-token mass rather than ASR as the outcome.

**Transfer across models and judges.**
- *Confirmed.* Five models in the full pipeline, two judges, with genuine disagreement on
  Mistral-7B-RR. Broader task accuracy, reasoning and instruction-following were not evaluated.
  State the utility claim as perplexity-and-compliance on 200 prompts, nothing wider.

---

## eKk5 — rating 4, confidence 5

The most severe review, and the one least addressable by a permutation null. Treat its points
as largely valid and respond by narrowing claims, not by pushing back.

**W1 — separation does not imply causal importance; Gemma proves it.**
- *Concede.* This is the paper's own boundary case: RCS peaks at layer 38, the causal hotspot is
  layer 23, and LAMA-RCS loses 45.3 points there. Our audit confirms that negative result is
  rock solid (p < 1e-4 under a correct behavior-level test).

**W2 — validation protocol does not match the deployed intervention.**
- *Concede*, same substance as PKr6. Spell out the operator difference explicitly and state what
  the correlations do and do not test.

**W3 — limited full-pipeline evidence.**
- *Confirmed and sharpened.* Five models, four informative alignment points, one at the ASR
  floor, two sharing a family. Plus the seed degeneracy above.

**W4 — benchmark, benign-set and seed coverage.**
- *Partly already done, and should be cited as such:* 159 HarmBench behaviors, two judges, three
  seeds for four models. *But* our audit shows the seed replication is largely vacuous, the
  Mistral HarmBench column uses only two seeds while the caption claims three, and the published
  intervals treat 477 non-independent observations as independent. Correct these rather than
  defending them.
- *Unresolved.* Additional safety benchmarks and broader benign tasks were not run.

**W5 — cost and practical reliability.**
- RCS itself is seconds of CPU once hidden states are cached. The causal sweep loads the model
  once per layer and is the expensive step; BO loads it once per trial. Note the discrepancy we
  found: the supplement says 128 trials, every artifact says **48**.
- *Concede.* The applicability condition is diagnosable only via the expensive sweep.

**W6 — presentation.**
- Accept. A single table separating representational claims from causal claims, and a clear
  statement of which configuration differs from which in exactly what, would address most of it.

---

## p6Gj — rating 6, confidence 3

**W1 — Eq. 2 would give a layer distribution for any two classes; SORRY-Bench vs Alpaca may be
topic or source.**
- *Largely concede.* The permutation null answers the narrower half: a random two-class split of
  the same data gives a flat profile at 0.0634, so the observed structure is not automatic. But
  the reviewer's real point is cross-corpus confounding, which the null does not touch and which
  we cannot currently test.

**W2 — effectiveness is inconsistent across models.**
- *Confirmed.* Do not claim general effectiveness. The honest summary: significant gain on
  Qwen2.5-7B with LAMA-RCS, significant gain on Qwen2.5-3B only with LAMA-BO, floor on
  Llama-2-7B, significant loss on Gemma2-9B, judge disagreement on Mistral-7B-RR.

**W3 — the independent contribution of RCS weighting is unclear; gains may come from the
multi-region directions.**
- *This is where we have new evidence, and it partly favors us.* On attack success, the
  group-direction step alone (LAMA-uniform, +3.77) is **not** significant (p = 0.31); only the
  full LAMA-RCS configuration is (+8.81, p = 0.006). On benign cost the reviewer's intuition is
  right: the group directions deliver the whole reduction and RCS weighting costs a little.
  Both halves should be stated together — they are separately supported and they explain the
  division of labour between the two components.
- *Note.* This also means the paper's "+3.8 then +5.0, so both contribute" decomposition in RQ5
  needs rewording, since the +3.8 is not distinguishable from zero.

**W4 — the applicability condition is post hoc.**
- *Concede.* If the causal profile is already available, using it directly for the weights is
  indeed more sensible; the paper's own LAMA-BO is a step in that direction. No cheap proxy
  exists.

---

## hgpB — rating 5, confidence 3

**W1 — incremental over Arditi et al. and Piras et al.**
- Position the contribution as the per-layer **strength** axis plus the depth-grouped direction
  construction, and note the weighted operator reduces to prior work at full strength. Do not
  claim the primitives are new.

**W2 — on Qwen2.5-3B the correlation drops 0.75 -> 0.45 after the norm control.**
- *Confirmed exactly* (recomputed 0.749 / 0.674 / 0.445). Concede that a substantial share is
  shared with the norm on that model. The counter-example worth citing is Llama-2-7B, where the
  raw norm correlates **-0.07** and RCS **rises** to 0.625 after the control. And state the
  failure openly: on Gemma2-9B the partial correlation is **-0.219**.

**W3 — RCS is not compared against simpler baselines.**
- *Newly measured.* The denominator removes a shared depth trend: the raw gap and the harmful
  spread both correlate 0.91-1.00 with depth and span 84x-19389x, while their ratio spans only
  1.7x-7.4x. That is the "additional information".
- *No longer a gap — now measured.* The tensors were regenerated (they reproduce Table 1 to
  `max|diff| = 2.0e-04`), so all seven denominator variants are computed in
  `../results/verification/alt_normalizations_qwen2-7b.csv`. Every one gives rho(causal) between 0.867 and
  0.873 against the published RCS at 0.873; the raw gap alone gives 0.744; the partial
  correlation controlling for the raw gap stays between 0.666 and 0.685. **State this as a
  robustness win and concede what it costs:** the ranking does not depend on which spread sits
  in the denominator, which also means RCS is not a *distinguished* choice and the paper should
  not argue that the specific form of Eq. (2) is necessary. A log-gap variant is a monotone
  transform and adds nothing by construction.

**Q — what exactly is retained beyond the norm?**
- Answer with the depth-detrending above, and keep the scope narrow: a residual predictive
  association on some models, not proof of refusal specificity, and not a general causal ranking.

**W4 — readability, undefined symbols.**
- Accept. Define ΔASR at first use.

---

## Cross-cutting corrections to make regardless of reviewer

1. Describe MD accurately: a matched single-layer **control** using a top-k direction heuristic,
   not a reproduction of the published method. Fix the Supplement C source-layer sentence and
   the Table 5 caption. Wording is in `baseline_fidelity.md`.
2. Recredit the benign-cost reduction to depth-specific direction construction (qp6c W2).
3. Recompute Table 2 intervals with the behavior as the unit; drop or implement the
   "hierarchical" description. Remove the two marginal daggers that do not survive.
4. Fix the Mistral-7B-RR caption: its HarmBench column averages two seeds, not three.
5. Fix the BO trial count: 48 in every artifact, 128 in the supplement.
6. Disclose that seeds produce identical judgments on two of five models.
7. Resolve the two disagreeing clean references for Qwen2.5-7B (82.43 vs 133.18).
8. Keep the causal framing (decided 2026-11-01) and correct its **breadth**, not its kind. The
   joint multi-layer ablation supports causality directly: no single layer carries more than
   25.1% of the removable refusal, two or three layers act super-additively at ~1.3x the sum of
   their parts, RCS-chosen layers beat the raw-norm control 8x at matched budget, and a random
   direction at the same layers does nothing. What the title must stop implying is breadth:
   three layers reach 86.6% of the attainable causal effect, so write "jointly encoded across a
   late band", never "distributed across the network", and never let the representational
   n80/L = 0.679 stand in for a causal statement. See `joint_causal.md`.
9. State the corrected data source wherever Supplement C describes it: 260 in-repo harmful
   prompts and 6000 harmless, not 4000 SORRY-Bench. This is the correction a reviewer can check
   fastest, since every extraction log prints it in plain text.
