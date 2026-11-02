# Refusal-specificity controls — RUN (partially resolved)

Supersedes the earlier `permutation_null_superseded.md` (now removed), which declared all of this impossible.
Four of its five stated prerequisites were wrong or clearable, and clearing them yielded six
matched controls, eight non-refusal contrasts with bootstrap depth intervals, and the
base-versus-aligned comparison in Part 4. Only the topic-matched provenance contrast remains
genuinely blocked, for lack of topic labels in the release.

**Requested by:** PKr6 ("show that RCS is detecting refusal-related structure rather than
generic harmful-versus-harmless semantic differences, using matched controls"), p6Gj-W1
("RCS may capture differences in topic, style, or dataset distribution rather than refusal
itself"), qp6c-W1 from the probe-saturation angle.

## Why the block lifted

That removed `permutation_null_superseded.md` claimed the corpora actually used were "4000 SORRY-Bench, 6000 Alpaca, not in the
repository". Both halves were wrong. The published runs used the in-repo processed splits, which
ship with the repository and need no credential (see `data_source.md`). And
sample-level representations, declared absent, can be generated in minutes by sharding the fp16
weights across partly-free GPUs (`../analysis/extract_sharded.py`). The category-metadata gap is
real, so the controls below are built from token length and syntactic register, which are
computable from the prompt text, rather than from topic labels.

## Part 1: does matching on length and register change the profile?

Harmful prompts are slightly longer than harmless ones, 12.7 against 12.2 tokens, Cliff's
delta 0.148, with 99.7% overlap. They are also written in an imperative register. Both are
candidate confounds, so both are matched away.

| Control | $n_f$ | $n_l$ | mean RCS | peak | Gini | rho vs published | vs null |
|---|---|---|---|---|---|---|---|
| published equivalent | 260 | 6000 | 1.1976 | L19 | 0.1881 | 1.0000 | 18.9x |
| length matched | 260 | 2331 | 1.2003 | L19 | 0.1867 | 0.9967 | 18.4x |
| register matched, harmless side | 260 | 4658 | 1.2246 | L19 | 0.1784 | 0.9956 | 19.2x |
| length and register matched | 260 | 1554 | 1.2303 | L19 | 0.1755 | 0.9929 | 18.4x |
| register matched, both sides | 148 | 4658 | 1.3366 | L20 | 0.1687 | 0.9781 | 16.0x |
| length and register, both sides | 148 | 882 | 1.3386 | L20 | 0.1674 | 0.9737 | 15.1x |

All "vs null" figures in this file use the analytic null at each contrast's own class sizes,
so the column is comparable across rows. For the published contrast the empirical permutation
null gives 18.7x against this analytic 18.9x, a 1% difference; see
`permutation_null.md`.

Length stratification resamples the harmless class so its token-length histogram matches the
harmful one stratum by stratum. Register matching restricts to imperative-initial prompts; the
harmful class is itself only 56.9% imperative, so the symmetric version constrains both sides
and is the stricter test.

**Nothing moves.** Across every control the peak layer stays at 19 or 20, Gini falls only from
0.188 to 0.167, the rank correlation with the published profile never drops below 0.974, and
the margin over the null stays between 15x and 19x. Prompt length and syntactic register do not
explain the RCS profile.

## Part 2: do non-refusal contrasts reproduce the profile?

The stronger test. Each contrast below is formed entirely inside a **held-out** harmless slice,
prompts 6000 to 12999, disjoint from the published set, so neither class contains any
refusal-triggering content by construction. Class sizes are pinned at 260 / 6000 so the null is
numerically identical to the paper's.

| Contrast | mean RCS | peak | Gini | vs null | rho vs published | Cliff's delta, length |
|---|---|---|---|---|---|---|
| **refusal (published)** | **1.1976** | **L19** | **0.1881** | **18.9x** | 1.000 | +0.148 |
| register, question vs imperative | 0.7104 | L5 | 0.1908 | 11.2x | **-0.788** | -0.106 |
| topic, code vs non-code | 0.5727 | L4 | 0.0580 | 9.0x | **-0.561** | +0.120 |
| length, 260 longest vs rest | 0.4436 | L5 | 0.1465 | 7.0x | **-0.789** | +1.000 |
| length, effect-size matched | 0.0818 | L1 | 0.0752 | 1.3x | -0.840 | +0.145 |
| random split | 0.0663 | L26 | 0.0225 | 1.0x | +0.403 | -0.020 |

Three results, in order of importance.

**Depth separates these contrasts from refusal, but read Part 3 before using this.** Each of
the five contrasts above is anti-correlated with the published profile, rho between -0.56 and
-0.84, and each peaks in layers 4 to 5 where refusal peaks at layer 19. Stated that way the
result does not generalise: every contrast in this table is decidable from surface form, and
Part 3 shows that deep benign contrasts behave quite differently. The two sentences this
section originally carried, that *every* non-refusal contrast peaks early and anti-correlates,
are false. The corrected claim is in Part 3.

**Length is eliminated outright.** A length contrast tuned to the same Cliff's delta as the real
harmful-versus-harmless length difference, 0.145 against a target of 0.148, sits at 1.3x the
null with mean RCS 0.082. The real contrast sits at 18.9x. A length difference of the size
actually present cannot produce anything resembling the observed profile, falling short by a
factor of 14.6.

**The null is confirmed a third time.** The random split gives 0.0663 against an analytic
prediction of 0.0634 and an empirical permutation null of 0.0640. Three independent routes
agree to within 5%.

A fourth route agrees too, and it is the cleanest of them. Drawing 260 prompts from the
held-out harmless slice and contrasting them against **the paper's own 6000 harmless
prompts** keeps the reference class, the class sizes and the extraction identical to the
published analysis, changing only whether the small class is harmful. Over 200 draws that
control gives mean RCS **0.0640** with a 95% interval of [0.0505, 0.0845], Gini 0.0398,
and not one draw out of 200 reaching the published Gini or n80/L (tail p 0.000 for both).
The observed contrast is **18.7x** this control. Artifacts:
`harmless_control_qwen2-7b.{json,csv}`, `../results/matched_controls/harmless_control_profiles_qwen2-7b.npz`, from
`../analysis/matched_control_harmless.py`.

## Part 3: are those early peaks just lexical separability?

Every contrast in Part 2 is decidable from surface form, so an early peak is what one would
predict of them whether or not refusal is involved, and the depth gap might be tracking
semantic abstraction instead. Three further contrasts test that, all formed inside the same
held-out harmless pool. The important one is the moral contrast: benign ethics and values
prompts, topically adjacent to harm and demanding moral reasoning, containing nothing that
triggers refusal. If the late peak were really harm-topic recognition, that contrast should
reproduce it.

Depth is now reported two ways. The peak layer carries a 95% interval from 1000 bootstrap
resamples of the small class, which is the limiting sample at n = 260 and the one reviewer
qp6c-W3 questions; because RCS depends on the small class alone once the reference centroid is
fixed, resampling it is the correct bootstrap. Beside it is the center of mass of the profile
over normalized depth, which is stable where an argmax over 28 layers is not.

| Contrast | vs null | Gini | peak | peak 95% CI | center of mass | com 95% CI | rho vs published |
|---|---|---|---|---|---|---|---|
| **refusal (published)** | **18.9x** | **0.1881** | **L19** | **[19, 20]** | **0.582** | **[0.577, 0.587]** | +1.000 |
| moral / ethical, benign | 3.3x | 0.0363 | L21 | [0, 27] | 0.506 | [0.493, 0.517] | +0.329 |
| multi-step vs lookup | 5.1x | 0.0856 | L17 | [17, 17] | 0.509 | [0.500, 0.518] | +0.436 |
| negative valence | 3.1x | 0.0347 | L18 | [6, 27] | 0.515 | [0.502, 0.525] | +0.882 |
| code vs prose | 8.0x | 0.0606 | L4 | [3, 4] | 0.479 | [0.473, 0.486] | -0.571 |
| question vs imperative | 11.0x | 0.1877 | L5 | [5, 5] | 0.408 | [0.403, 0.414] | -0.778 |
| 260 longest vs rest | 7.0x | 0.1465 | L5 | [5, 6] | 0.428 | [0.422, 0.437] | -0.789 |
| random split | 1.1x | 0.0476 | L18 | [0, 27] | 0.519 | [0.482, 0.531] | +0.844 |
| length, effect-matched | 1.3x | 0.0752 | L1 | [0, 18] | 0.461 | [0.450, 0.506] | -0.840 |

**Two claims Part 2 made are false, and this is where they break.** "Every non-refusal contrast
peaks in layers 4 to 5" held only for the lexical contrasts: the three deep ones peak at layers
17, 18 and 21. "Every non-refusal contrast anti-correlates with the published profile" fails the
same way and worse, because four of the eight correlate *positively*, up to +0.882.

**The rank correlation against the published profile is not usable as a specificity statistic.**
A pure random split, sitting at 1.1x the null and containing no contrast at all by construction,
reaches rho = +0.844. Any two profiles sharing the common depth trend will correlate, so a high
rho carries almost no information. The negative values in Part 2 are the anomaly rather than the
norm: the lexical contrasts actively pull mass *earlier* than chance.

**What the deep contrasts do and do not reproduce.** They raise the center of mass only to about
0.51, which is where the random split already sits, so semantic depth alone does not move RCS
mass later than chance. And they produce *flat* profiles, Gini 0.035 to 0.086 against the
published 0.188, so unstable that two of the three peak intervals span the whole model. Deep
benign semantics yields a weak, shapeless RCS, not a late concentrated one. The moral contrast
in particular reaches only 3.3x the null, so topical adjacency to harm is not sufficient.

**What survives is a joint statement, not a single statistic.** Refusal's center of mass, 0.582,
is disjoint from all eight non-refusal intervals, the highest of which is 0.519, though the
margin over chance is modest, roughly 1.3 layers: the widest benign interval reaches 0.531 and
refusal's begins at 0.577. And no single column separates refusal: the
highest-magnitude benign contrast reaches 11.0x but places its mass *earliest* of all at 0.408,
while the latest-mass benign contrast sits at 1.1x the null. Refusal is the only contrast that is
simultaneously large in magnitude and late in depth.

The effect-matched length row is the one contrast not drawn by this script. It predates the
depth statistics and its interval was added later by `../analysis/length_matched_depth.py`, which
reconstructs the tuned split rather than redrawing it and checks the reconstruction against the
recorded mean RCS, peak layer and Gini (agreement to 1e-8) before reporting the interval. An
earlier version of this section claimed disjointness from "all eight" intervals while only seven
had been measured; the eighth was measured to settle it, and it is disjoint.

The four contrasts shared with Part 2 are redrawn here under an independent sampling sequence,
which is why their magnitudes differ slightly from that table: code vs prose gives 8.0x here
against 9.0x there, question vs imperative 11.0x against 11.2x. The spread across draws is small
and no conclusion turns on it.

Artifacts: `deep_benign_qwen2-7b.{json,csv}`, from `../analysis/deep_benign_contrasts.py`. That
script validates its fast RCS path against the paper's own `lama_rcs.compute_rcs`
(`max|diff| = 1.07e-06`) before computing anything.

## The honest limitation this creates

RCS is **not** exclusively a refusal detector. A code-versus-prose topic contrast reaches 9.0x
the null and a question-versus-imperative register contrast reaches 11.2x, both far above
chance. Any strong semantic contrast produces a substantial RCS. The register contrast is
sharper still: its Gini of 0.1908 slightly **exceeds** the published 0.1881.

So concentration alone does not distinguish refusal from other contrasts. The claim the evidence
supports is narrower and more specific than "RCS measures refusal":

> Neither the score magnitude, nor its concentration, nor its rank correlation with the
> published profile is refusal-specific on its own: a benign register contrast reaches 11.0x the
> null and ties the published Gini at 0.188, and a pure random split correlates with the
> published profile at +0.844. What is specific to refusal is the *combination* of a large score
> with a late center of mass. Refusal's mass sits at 0.582 of normalized depth, disjoint from
> all eight benign contrasts tested, whose maximum is 0.519, and the only benign contrast that
> approaches its magnitude places its mass earliest of all.

This should be stated in the paper. It converts a weakness the reviewers identified into a
positive, testable characterization, and it is the honest reading of these numbers.

### What this claim has and has not survived

The attack this section previously listed as untested has been run; it is Part 3, and it
removed two of the sentences this file originally carried. Three caveats remain.

The classes in Part 3 are keyword-anchored, so they are themselves partly lexical. That makes
the test conservative in one direction only: it could fail to find a genuinely deep contrast,
but the deep contrasts it did find already behave unlike refusal, so the conclusion does not
depend on sharpening them.

The margin is modest. Refusal's center of mass is about 1.3 layers later than a random split's.
The intervals are disjoint over 1000 bootstrap resamples, but this is not a large effect and
should not be described as one.

All eight contrasts come from a single model, Qwen2.5-7B.

## Part 4: base versus aligned — the control that separates the two claims

This is the control qp6c and PKr6 ask for directly, and it is no longer blocked. Both prompt
classes are held identical and only the weights differ: `Qwen/Qwen2.5-7B` against
`Qwen/Qwen2.5-7B-Instruct`, with every prompt formatted once using the *Instruct* chat template
and system prompt and token-id identity asserted under both tokenizers. The base model has seen
the same harmful-versus-harmless distinction but was never trained to refuse, so whatever it
shows is what RCS measures *before* refusal exists.

The prediction was pre-registered in `../analysis/analyze_base_vs_aligned.py` before the run:
base mass at chance depth would mean alignment creates the late structure; base mass at the same
depth as aligned would mean RCS measures harm-topic recognition and the paper's reading fails.

| | mean RCS | vs null | Gini | n80/L | peak [95% CI] | com [95% CI] |
|---|---|---|---|---|---|---|
| **aligned** (Instruct) | 1.1976 | 18.9x | 0.1881 | 0.679 | L19 [19, 20] | **0.582 [0.577, 0.587]** |
| **base** | 0.7205 | 11.4x | 0.0984 | 0.786 | L20 [20, 21] | 0.537 [0.531, 0.542] |
| chance reference | 0.0680 | 1.1x | 0.0476 | — | L18 [0, 27] | 0.519 [0.482, 0.531] |

**The answer is neither branch of the prediction, and it splits the paper's claim in two.**

**Magnitude is largely pre-existing, so a high RCS is not evidence of refusal.** The base model
separates harmful from harmless prompts at **11.4x the null**, reaching **60.2% of the aligned
model's mean RCS**. A checkpoint with no refusal training whatsoever accounts for most of the
score. This is the reviewers' harm-topic-recognition alternative, confirmed: it is real, it is
large, and the paper cannot attribute score magnitude to refusal.

**Depth structure is created by alignment, and that is the claim the title rests on.** The
pre-existing separation is almost **flat**. The base Gini is 0.0984, 52.3% of the aligned
0.1881, and its n80/L is *higher* (0.786 against 0.679), meaning its mass is spread more widely
rather than concentrated. Its center of mass, 0.537 [0.531, 0.542], **overlaps the chance
interval** [0.482, 0.531], so its depth is not distinguishable from a random split except
marginally. The aligned interval [0.577, 0.587] is disjoint from both.

**Alignment's increment is itself late-weighted, which is the strongest single number here.**
The per-layer difference is near zero early and maximal in exactly the band the paper
emphasises:

| layer | 0 | 1 | 2 | 3 | 4 | ... | 16 | 17 | **18** | 19 | 20 | ... | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aligned - base | -0.069 | +0.019 | -0.092 | -0.024 | +0.061 | | +0.826 | +0.931 | **+1.005** | +0.989 | +0.950 | | +0.536 |

In layers 0, 2 and 3 the **base model scores higher than the aligned one**: alignment adds
nothing at all in the first few layers. The five layers where it adds most are 18, 19, 20, 17
and 16. Alignment does not shift a peak that already existed; it builds the late concentration.
Note that the base model's own argmax, L20, is *later* than the aligned model's L19, so lateness
of the peak is not the alignment signature. Concentration is: alignment makes the late band
tall, not late.

**And what alignment builds is itself spread across layers, which is the title's claim.** The
increment is not localized anywhere. The single largest layer holds only **7.4%** of the total
positive increment, the best five layers hold **34.7%**, the increment stays above half its
maximum across **14 layers** (L13-L27), and it exceeds +0.25 contiguously across **19 layers**
(L9-L27). This is the paper's layer-distribution claim measured on the alignment difference
rather than on the aligned profile alone, which is a considerably stronger basis for it: no
single layer can be called the site of refusal, because none of them carries even a twelfth of
what alignment adds.

**What this licenses the paper to say, and what it does not.**

> Supported: the late-layer *concentration* of RCS is a product of alignment, and what
> alignment adds is distributed over roughly half the network rather than localized. An
> unaligned checkpoint shows the same harmful-versus-harmless contrast with 52% of the
> concentration, mass at chance depth, and alignment's own contribution rising from
> approximately zero in layers 0-4 to its maximum at layer 18, with no single layer holding
> more than 7.4% of it.
>
> Not supported: that RCS magnitude reflects refusal. 60% of it is present before any refusal
> training. Claims of the form "RCS is high, therefore refusal lives here" do not survive.

One honest caveat, and one that has been retired. The base model's center of mass,
0.537, sits marginally above the chance point estimate of 0.519 and the
intervals still overlap, though now by only 0.0002 against 0.0017 at the smaller sample, so
a weak pre-existing late tilt cannot be excluded; the claim is that alignment dominates it, not
that it is absent. The sample-size caveat is gone: this run uses the **full published class
sizes, 260 harmful and 6000 harmless**, not the 1000-prompt harmless subsample the first version
was forced into. The earlier CPU run is retained for comparison as
`base_vs_aligned_qwen2-7b_cpu_n1000.{json,csv}`; every conclusion below held there too, at
60.4% rather than 60.2% and 51.6% rather than
52.3%.

Both extractions ran in one process on ckg10 (Ampere) in fp16 with the `sdpa` backend, with
parameter checksums before and after: **0 of 339 parameters moved in either model**. That is the
guard the ckg08 GPU run of this same comparison failed (see
`gpu_weight_corruption.md`; retained as
`../results/matched_controls/VOID_gpucorrupt_base_vs_aligned_qwen2-7b.json`). A second guard was added after an `eager` +
fp16 attempt on this same sound host silently wrote NaN into layer 27 for a large minority of
prompts while passing the weight checksum: every tensor is now checked with `assert_finite()`
before it is saved, because a weight checksum is not an activation check. The re-extracted
aligned profile reproduces the published logs at **Spearman 1.0000**
with `max|diff| = 0.0002` and peak L19 in both, so the base
profile measured in the same process is trustworthy on the same grounds.

Artifacts: `base_vs_aligned_qwen2-7b.{json,csv}`.

## What remains genuinely blocked

**Topic-matched provenance contrast.** Still limited by metadata. The `category` field is
uniformly `"None"` in both `harmful_train.json` and `harmless_train.json`. The keyword-based
topic split used above is a coarse substitute and is reported as such.

**The other eight models.** Controls here are Qwen2.5-7B only, the one cached checkpoint.

## Reproducing

```bash
python3 analysis/extract_sharded.py --model_name qwen2-7b
python3 analysis/extract_control_sharded.py --start 6000 --n 7000
python3 analysis/matched_controls.py
python3 analysis/nonrefusal_controls.py
python3 analysis/matched_control_harmless.py
```

```bash
python3 analysis/deep_benign_contrasts.py
python3 analysis/length_matched_depth.py
python3 analysis/base_vs_aligned.py --shard "7:11.6,6:4.7,1:4.6" \
    --dtype float16 --attn sdpa --n_harmless 6000 --suffix _gpu_fp16_full
python3 analysis/analyze_base_vs_aligned.py --suffix _gpu_fp16_full
```

Artifacts: `matched_controls_qwen2-7b.{json,csv}`, `nonrefusal_controls_qwen2-7b.{json,csv}`,
`deep_benign_qwen2-7b.{json,csv}`, `../results/matched_controls/length_matched_depth_qwen2-7b.json`,
`base_vs_aligned_qwen2-7b.{json,csv}`.
