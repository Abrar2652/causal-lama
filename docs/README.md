# Audit, controls, and additional experiments

**The drafted author response is deliberately not published here.** It is kept with the
authors until the response is filed. What follows is the evidence it rests on. Note that the
AAAI-27 feedback window and length cap could not be confirmed, and all 20 quantitative claims in
the drafts are gated by `../analysis/verify_rebuttal_claims.py`.

> ## READ FIRST (six findings)
>
> **1. `data_source.md` — the paper misstates its own data.** The paper states the
> harmful representations are 4000 SORRY-Bench prompts. All nine extraction logs record
> `[processed source] harmful=260 harmless=6000`: the harmful class is 260 prompts from a
> different, in-repo corpus. **Now confirmed by reproduction**, not only by logs. Regenerating
> with the actual in-repo split reproduces the published profile to `max|diff| = 2.0e-4`
> (Spearman 1.0000), with layer-0 RCS agreeing to eight decimal places. A 28-layer match at that
> precision identifies the source beyond coincidence. The SORRY-Bench counter-run that originally
> corroborated this is now void (see finding 4), and the positive match never needed it. Every
> per-layer statistic and interval in Table 1 rests on 260 samples, not 4000. This must be
> corrected in the manuscript.
>
> **2. `matched_controls.md` — the refusal-specificity objection has an answer, and it
> is narrower than it first looked.** Matching away prompt length and imperative register leaves
> the profile essentially untouched (peak stays at layer 19-20, rho >= 0.974). Eight benign
> contrasts built inside held-out harmless prompts then test specificity. **No single statistic
> is refusal-specific**: a benign register contrast reaches 11.0x the null and ties the published
> Gini at 0.188, and a pure random split containing no contrast at all correlates with the
> published profile at rho = +0.844, which retires rank correlation as a specificity measure.
> What is specific to refusal is the **combination of a large score with a late center of mass**:
> 0.582 of normalized depth, disjoint from all eight benign intervals, the highest of which is
> 0.519 and whose widest interval reaches 0.531. An earlier draft of this document claimed
> every benign contrast peaks at layers 4-5 and anti-correlates with refusal. Part 3 disproves
> both halves and the file records the correction.
>
> **3. `matched_controls.md` Part 4 — the base-versus-aligned control splits the paper's
> claim in two.** The control qp6c and PKr6 ask for directly has been run: `Qwen/Qwen2.5-7B`
> against `-Instruct`, identical prompts, asserted token-id identity, at the **full published
> class sizes 260/6000**, fp16 `sdpa` on Ampere, with 0 of 339 parameters mutated in either arm
> and the aligned arm reproducing the published profile at Spearman 1.0000 (max|diff| 2.0e-4).
> **Score magnitude is largely pre-existing**: the base checkpoint, never trained to refuse,
> separates harmful from harmless at 11.4x the null and reaches 60.2% of the aligned mean RCS, so
> a high RCS is not evidence of refusal and the harm-topic-recognition
> alternative is confirmed as real. **Depth structure is not pre-existing**: that separation is
> nearly flat, 52.3% of the aligned Gini with a *higher* n80/L, and its center of mass
> 0.537 [0.531, 0.542] overlaps the chance interval [0.482, 0.531]. Alignment's own per-layer
> increment is approximately zero in layers 0-4, negative at 0, 2 and 3, and maximal at +1.005
> at layer 18. The late concentration is built by alignment; the magnitude is not. What
> alignment builds is also **distributed**: no single layer holds more than 7.4% of the total
> increment, the best five hold 34.7%, and it stays above half its maximum across 14 layers.
> That is the title's layer-distribution claim, measured on the alignment difference rather
> than on the aligned profile alone.
>
> **4. `baseline_fidelity.md`** — our MD comparison uses a top-k direction heuristic and
> an unconstrained source-layer rule, so it is a **matched single-layer control**, not a
> reproduction of Piras et al. (2026). The author decision (2026-10-31) is to keep this setup
> and describe it accurately; paste-ready disclosure wording is in that document. Comparisons
> should be framed as isolating the two design axes within a matched pipeline, not as improving
> on the published method. The RCS analysis and the permutation null are unaffected.
>
> **5. `gpu_weight_corruption.md` — inference silently corrupts model weights in GPU
> memory on ckg08, which voids every GPU extraction done there.** Repeated byte-identical
> forward passes return different answers: 25 runs of one prompt gave 25 distinct results,
> drifting from the correct 15.96 to 160.60 at layer 1, and to 998 at layer 19 against a true
> 109.5. Checksumming shows 12 of 340 parameters change during a no-grad eval forward, by up to
> 23%, with no NaN and no error, and these cards report ECC counters as `[N/A]`. Ruled out: the
> attention backend, the dtype, the KV cache, our extraction code, and the arithmetic units (200
> identical matmuls are bitwise equal). **This supersedes the earlier `sdpa_turing_superseded.md`,
> which blamed `sdpa` and recommended `eager` + bfloat16; that diagnosis and that fix are both
> wrong.** Verified-sound paths are CPU fp32 (0.03-0.06% against the Ampere reference) or an
> Ampere host **with `--attn sdpa`** -- `eager` + fp16 is independently unsafe, see finding 5's
> addendum in `sdpa_turing_superseded.md`. Nothing in this audit is affected: every conclusion
> derives from Ampere-extracted
> tensors that reproduce the published profile to 2.018e-04. Void tensors are quarantined with
> `DO_NOT_USE.md` markers. **The card needs reporting to whoever administers the machine.**
>
> **6. `joint_causal.md` -- the causal claim in the title now has direct evidence, and
> it is narrower than the title implies.** With the 2026-11-01 decision to keep the causal
> framing, PKr6's deferred request became load-bearing and has been run: ablation of layer
> *sets*, not one layer at a time. The harness reproduces the paper's own single-layer probe to
> `max|diff| = 2.8e-04` (Spearman 0.99945) and restores to baseline exactly, so the joint arms
> stand on the published result's footing. **Supported:** no single layer carries refusal (the
> best, L20, reaches only **25.1%** of the removable effect); two or three layers together are
> about **30% more effective than the sum** of their individual effects, so the layers are
> complementary rather than redundant; RCS picks the right ones, beating the raw diff-in-means
> norm **8x** and the best of five random draws **3.1x** at k = 3; and the effect is
> direction-specific, since a random direction at the same layers moves refusal by **< 0.001**.
> It is not degradation: harmless-prompt entropy rises at most 0.05 nats. **Not supported:**
> that refusal is spread across the network's depth. Causally **three layers (L18-L20) reach
> 86.6%** of the achievable effect against the representational n80/L = 0.679, which would
> suggest 19 of 28 -- the two differ by a factor of six. Also useful to the paper: past k = 12
> the effect *declines* (0.859 to 0.797 at k = 28), so a uniform all-layer intervention is
> strictly worse than the band, a causal argument for a layer-selective method that the paper
> does not currently make.

Evidence audit for AAAI-27 submission 43976, built on 2026-10-31 from the released repository,
the submitted paper and the technical supplement. Read in this order:

0. **`data_source.md`** and **`baseline_fidelity.md`** — the two defects that
   change what the paper may claim. Read before anything else.
1. **`evidence_inventory.md`** — the reviewer matrix: what exists, what is genuinely missing,
   what was done, and the discrepancies found between the release and the manuscript.
2. **`reused_evidence.md`** — the numbers, the permuted-label null, and the specific claim
   corrections the evidence forces.
3. **`reviewer_notes.md`** — per-reviewer working notes for drafting the response.
4. **`execution_summary.md`** — commands, tests, environment, timings, blockers.

New measurements, all on Qwen2.5-7B, the one cached checkpoint:

- `permutation_null.md` — the empirical label-permutation null, run on activations that
  provably match Table 1. Observed RCS sits 18.7x above it; tail p = 0.000 on both concentration
  statistics. The closed form predicted 0.0634 and the measurement gives 0.0640.
- `matched_controls.md` — six matched controls and eight non-refusal contrasts, with
  bootstrap depth intervals on all eight.
- `../results/verification/alt_normalizations_qwen2-7b.csv` — seven alternative denominators for Eq. (2).
- `matched_controls.md` Part 4 — base versus aligned at the **full published class
  sizes (260/6000)**: magnitude 60.2% pre-existing, concentration 52.3%, alignment's
  increment maximal at layer 18. The re-extracted aligned arm reproduces the published profile at
  Spearman 1.0000, max|diff| 2e-04.
- `joint_causal.md` — joint multi-layer ablation: the cumulative RCS-ranked curve,
  matched-k selection rules (bottom-k, raw-norm, random-k), an additivity test, a
  random-direction control and a degradation control.
- `../tables/rebuttal_tables.tex` — paper-ready tables, each with an insight and a failure mode.
  Compiles under pdfTeX with zero overfull boxes.

Nothing in the original repository was modified. Every number traces to a file under
`verification/`, `permutation_null/` or `matched_controls/`, or to a cited table in the PDFs.

## The two-column decision, per concern

| Concern | New experiment conducted | Explanation from existing evidence |
|---|---|---|
| qp6c W1 permutation null | **Yes — empirical**, on the paper's own activations (`permutation_null.md`); calibrated null still carries the other 8 models | — |
| qp6c W2 benign-cost attribution | no | **Yes** — `../results/verification/table5_audit.csv` |
| qp6c W3 sample size | no | **Yes** — `../results/verification/seed_identity_audit.csv` |
| qp6c W4 Llama-2-7B baseline | no | **Yes, fully diagnosed** — `baseline_fidelity.md` |
| qp6c W5 cheap alignment proxy | no | none exists; do not claim one |
| qp6c / PKr6 base-vs-aligned | **Yes — run at full n = 260/6000** (`matched_controls.md` Part 4): base model shows 60.2% of the magnitude at 11.4x the null but only 52.3% of the concentration, mass at chance depth | — |
| PKr6 / p6Gj-W1 matched controls | **Yes** — 6 matched controls, 8 non-refusal contrasts (`matched_controls.md`) | — |
| PKr6 joint-layer causality | **Yes — run** (`joint_causal.md`): best single layer only 25.1% of the joint effect, k=2-3 super-additive at ~1.3x, RCS beats raw norm 8x at matched k=3, random direction < 0.001 | — |
| eKk5 W1/W2 separation vs causation | no | **Yes** — Gemma boundary case, confirmed significant |
| eKk5 W4 seeds and judges | no | **Yes** — and three corrections required |
| eKk5 W5 cost | no | **Yes** — plus the 48-vs-128 BO trial discrepancy |
| p6Gj W3 component contributions | no | **Yes** — `../results/verification/significance_multiseed_audit.csv` |
| hgpB W2/W3/Q normalization | **Yes** — 7 denominators computed from regenerated activations (`../results/verification/alt_normalizations_qwen2-7b.csv`) | — |

## Claims that must be softened absent further evidence

1. That refusal **causally** is layer-distributed **in the sense of being spread across depth**.
   The joint-ablation experiment (`joint_causal.md`) supports the causal claim in its
   *joint* reading: no single layer carries more than 25.1% of the removable refusal, and two
   or three layers together exceed the sum of their parts by ~30%, so "no single layer suffices
   and the layers act jointly" is defensible. What is **not** defensible is breadth -- three
   layers reach 86.6% of the achievable effect, against the representational n80/L = 0.679 that
   would imply 19 of 28. Representational spread is not causal spread. Say "jointly encoded
   across a late band", not "distributed across the network".
2. That RCS measures refusal specifically. The controls sharpen this rather than settling it.
   Supported: the refusal profile is not explained by length or register, and it is the only
   contrast tested that is **both large in magnitude and late in depth** (18.9x the null with
   its mass at 0.582 of depth, disjoint from all eight benign intervals, whose highest point
   estimate is 0.519 and whose widest interval reaches 0.531). Not
   supported: that any single statistic is refusal specific. A high RCS is not, since benign
   contrasts reach 11x the null; a concentrated profile is not, since a benign register contrast
   ties the published Gini; and rank correlation with the published profile is not, since a pure
   random split reaches +0.844 on it. Nor is depth alone sufficient: benign contrasts peak
   anywhere from layer 4 to layer 21, so the peak layer separates nothing by itself. The
   base-versus-aligned control settles the magnitude half outright: 60.2% of the aligned mean
   RCS is present in a checkpoint never trained to refuse, so **magnitude claims must be
   dropped, not softened**. What that control does support is the depth claim, since the
   pre-existing separation is flat and at chance depth while alignment's increment is maximal
   at layer 18.
3. That RCS down-weighting reduces benign cost. Table 5 shows the opposite within the LAMA family.
4. That depth-specific directions alone improve attack success on Qwen2.5-7B (p = 0.31).
5. That the alignment-lift relationship generalizes. Four points, one at the ASR floor, two from
   one family, no independent seed replication.
6. That LAMA improves on the **published** MD method. The comparison is against a matched
   control; reframe accordingly. See `baseline_fidelity.md`.
7. That the RCS analysis used SORRY-Bench. It used the 260-prompt in-repo harmful split.
   See `data_source.md`.
8. That the harmful intra-class spread is the *principled* denominator for Eq. (2). Seven
   normalizations agree to within 0.006 in rank correlation, far below the resolution of a
   28-layer Spearman. Normalizing matters; which normalizer does not.
