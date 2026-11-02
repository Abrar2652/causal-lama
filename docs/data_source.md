# CRITICAL: the paper misstates the data used to compute RCS

Found 2026-10-31 while regenerating representations. Independent of, and more fundamental
than, the MD baseline issue in `baseline_fidelity.md`.

## What the paper says

Main paper, Section 5:
> "Harmful and harmless representations come from SORRY-Bench (Xie et al. 2025) and Alpaca
> (Taori et al. 2023)."

Supplement C:
> "Harmful prompts: 4000 from SORRY-Bench; harmless: 6000 from Alpaca."

## What the extraction logs record

Every one of the nine profiled models logs the same line:

```
[processed source] harmful=260 harmless=6000
Processing 6000 samples with token position -1 using model ...
Processing 260 samples with token position -1 using model ...
```

| Model log | source | harmless | harmful |
|---|---|---|---|
| qwen2-7b, qwen2-3b, llama2-7b, llama31-8b, gemma2-9b | `processed` | 6000 | **260** |
| mistral-7b-rr, deepseek-r1-qwen-7b, ministral-8b, gpt-oss-20b | `processed` | 6000 | **260** |

`--source processed` is the non-gated fallback path in `../create_representation_dataset.py`. It
loads `../dataset/processed/harmful_train.json` (260 prompts) and the first 6000 of
`../dataset/processed/harmless_train.json`. It does **not** touch SORRY-Bench.

The 260 harmful prompts are the baseline repository's own harmful training split, byte-identical
to the copy in `pralab/som-refusal-directions`. They read like AdvBench-style instructions
("Describe in detail how to commit tax fraud"), not SORRY-Bench items.

So the harmful class is **260 prompts from a different corpus**, not 4000 from SORRY-Bench.
The harmless side is approximately as described, an Alpaca-derived in-repo subset, though it is
the first 6000 rows of the processed split rather than a seeded selection from `tatsu-lab/alpaca`.

## Independent corroboration

Representations were regenerated for Qwen2.5-7B using the **stated** setup, 4000 SORRY-Bench
and 6000 Alpaca pulled from HuggingFace. The resulting RCS profile does **not** reproduce
Table 1. It is anti-correlated with the published profile:

| Comparison | Value |
|---|---|
| Spearman(regenerated, published) | **-0.78** |
| Spearman(regenerated, published reversed) | +0.89 |
| Peak layer | regenerated **6**, published **19** |
| Gini | regenerated 0.117, published 0.188 |
| max relative difference | 89% |

The published profile rises with depth and peaks at layer 19; the SORRY-Bench profile falls
with depth. Those tensors are preserved at `dataset/representations/qwen2-7b_hfsource/`.

## Confirmed by reproduction, not only by logs

The companion run has now completed. Representations were regenerated for Qwen2.5-7B with
`--source processed`, sharding the fp16 weights across three partly-free GPUs
(`../analysis/extract_sharded.py`; 6260 prompts in 5.3 minutes at 19.7 it/s). The regenerated
profile reproduces the published one:

| Comparison | Value |
|---|---|
| max abs difference, regenerated vs `../logs/rcs_analysis_qwen2-7b.json` | **2.0e-4** |
| Spearman(regenerated, published) | **1.0000** |

Set beside the positive match, this closes the question. The
published Table 1 was computed from the 260-prompt in-repo harmful split. Both regenerations
are preserved: `dataset/representations/qwen2-7b_processed_confirm/` reproduces the paper,
`dataset/representations/qwen2-7b_hfsource/` does not.

This also cleared the permutation-null blocker. The empirical label-permutation control that
reviewer qp6c asked for now runs on activations that provably match Table 1; see
`permutation_null.md`.

The four all-zero rows in the SORRY-Bench extraction are a separate, minor artifact: four
prompts in that selection are `None`, and the extraction loop skips `None` while leaving the
pre-allocated zero row in place. Dropping them changes RCS by at most 0.5% (Spearman 0.9989),
so it is not the cause of anything above. It would, however, affect any future SORRY-Bench run.

## Why this matters

1. **The stated experimental setup is wrong** in both the main paper and the supplement. This is
   a factual description of the data, not an interpretation, and it is checkable from the
   released logs by anyone who clones the repository.
2. **Statistical power is far lower than implied.** Every per-layer RCS, bootstrap interval,
   Cohen's d and Gini in Table 1 rests on **260** harmful samples, not 4000. The paper's own
   bootstrap resamples that class, so the reported intervals are correct for what was run but
   much wider than a reader assuming 4000 would expect.
3. **The permutation null changes.** Under relabeling the expected score is
   $\sqrt{1/n_{hf}+1/n_{hl}}$, which depends only on class sizes: 0.0204 at 4000/6000 but
   **0.0634** at 260/6000, roughly three times higher. Observed RCS still sits far above it, so
   the qualitative conclusion holds, but every number in the null table had to be regenerated.
   The corrected run is tagged `corrected` in `permutation_null/`. The empirical null measured
   on the real tensors is **0.0640**, within 0.95% of the closed-form 0.0634, so the correction
   is confirmed rather than merely recomputed.
4. **It changes how two reviewer concerns should be answered.** PKr6 and p6Gj object to
   SORRY-Bench versus Alpaca as a cross-corpus confound. The confound is real but the corpora
   are not the ones named, so the response must describe the actual pair. hgpB's question about
   normalization is also affected, since the harmful class is 260 samples.

## What to do

The honest options, in order of preference:

1. **Re-extract with the stated setup and recompute.** This makes the paper's description true.
   It needs GPU time for nine models and would change every number in Table 1.
2. **Correct the description to match what was run.** Cheaper and defensible: state that
   harmful representations are the 260-prompt processed split and harmless are 6000 Alpaca-derived
   prompts, and disclose the sample size wherever Table 1's intervals are discussed.

Option 2 is the realistic choice inside a rebuttal window. What is not defensible is leaving
the current text, because it names a dataset and a sample size that were not used.

## The SORRY-Bench counter-run is void, and was never needed

Everything above that cites the regenerated SORRY-Bench profile (Spearman -0.78, peak layer 6,
the four all-zero rows, the layer-0 value of 1.7414) came from an extraction performed on ckg08,
where inference corrupts model weights in GPU memory. Those tensors are quarantined; see
`gpu_weight_corruption.md`. Treat every SORRY-Bench number in this file as withdrawn.

The conclusion does not depend on them. The identification rests on the positive match: the
in-repo 260/6000 split reproduces the published 28-layer profile at Spearman +1.0000 with
`max|diff| = 2.018e-04`, layer-0 RCS agreeing to eight decimal places (published 0.6663883328,
re-extracted 0.6663883552), and the same peak layer. A match at that precision across 28 layers
cannot arise from a different corpus. The counter-run was corroboration, and losing it costs
rhetorical force rather than evidence.

## Zero-row hazard: checked, and the published run is clean

`convert()` in `../create_representation_dataset.py` pre-allocates
`torch.zeros(n, num_layer, hidden_dim)` and then **skips** any prompt that is `None`,
leaving an all-zero row in place. `compute_rcs` does not filter those rows, so it would
average them into the class centroid and silently bias the profile.

This is not hypothetical: the SORRY-Bench regeneration produced 4 all-zero harmful rows
out of 4000 (the usable tensor is 3996 after filtering), because the prompt-style filters
leave a few empty strings. The three-way comparison in
`../analysis/verify_data_source.py` therefore drops all-zero rows before computing RCS.

The published runs are **not** affected. The in-repo processed split that those runs used
contains no empty or `None` prompts:

| split | n | empty / None rows |
|---|---|---|
| harmful (train) | 260 | none |
| harmless (train, capped at 6000) | 6000 | none |

So the hazard does not touch Table 1. It is worth a defensive one-line filter in
`convert()` regardless, and it is worth mentioning if a reviewer asks how empty prompts
are handled.
