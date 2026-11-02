# Permutation null on the paper's own activations — RESOLVED

**Status: run and passed.** The empirical label-permutation control that reviewer qp6c asked
for (W1) has now been computed on the activations that produce the paper's Table 1, for
Qwen2.5-7B. This file supersedes the earlier `permutation_null_superseded.md`.

Artifacts: `../results/permutation_null/results_empirical_qwen2-7b.json`, `../results/permutation_null/summary_empirical_qwen2-7b.csv`,
`../results/permutation_null/permutation_distribution_empirical_qwen2-7b.npz`.

```bash
python3 analysis/permutation_null_empirical.py \
    --model qwen2-7b --n_perm 200 --seed 42 \
    --reps_dir dataset/representations/qwen2-7b_processed_confirm
```

## How the block was cleared

Two separate obstacles were listed before, and the resolution is different for each.

**The data was never gated.** The earlier note assumed the tensors had to be rebuilt from 4000
gated SORRY-Bench prompts, as Supplement C states. They do not. The extraction logs record
`harmful=260 harmless=6000` for all nine models, and the regenerated profile now reproduces the
published one to `max|diff| = 2.0e-4` with Spearman 1.0000. The harmful class is the 260-prompt
in-repo split, which needs no credential and no download. See `data_source.md`.

**The GPU limit was an artifact of insisting on one card.** A single-card fp16 load of
Qwen2.5-7B needs about 16 GiB and no card on this node had that free. Three partly-free cards
together did. `../analysis/extract_sharded.py` shards the fp16 weights with `device_map`, keeping
dtype, chat template, token position and `hidden_states[1:]` indexing identical to the repo
path. The 6260 prompts extracted in 5.3 minutes at 19.7 it/s, faster than the original
single-card run, which was contending with other jobs.

## Result

Shared-label permutation: 200 relabelings drawn once, class counts preserved, each applied at
every layer. 28 layers, n_f = 260, n_l = 6000, d = 3584. Runtime 16.1 s.

| Quantity | Observed | Null mean | Null 95% interval | Tail p |
|---|---|---|---|---|
| Mean RCS | 1.1976 | 0.0640 | [0.0504, 0.0811] | — |
| Gini | 0.1881 | 0.0407 | [0.0194, 0.0705] | 0.000 |
| n80/L | 0.6786 | 0.7946 | [0.7857, 0.8214] | 0.000 |
| Peak / mean | 1.5826 | 1.1541 | [1.0611, 1.3003] | — |
| CV | 0.3284 | 0.0740 | [0.0368, 0.1314] | — |

Observed RCS sits **18.7x** above the permutation null. Neither concentration statistic has a
single relabeling out of 200 reaching the observed value, so both tail probabilities are
below 1/200 and are reported as 0.000 rather than as exact zeros.

## The closed form is validated, not just assumed

The analytic derivation predicted `E[RCS_perm] ~ sqrt(1/n_f + 1/n_l) = 0.06335`, independent of
layer, model and representation geometry. The measured empirical null is **0.06396**, a relative
agreement of 0.95%. The calibrated simulation reported earlier is therefore no longer load
bearing for this model: it is corroborated by the real measurement. It still carries the other
eight models, where checkpoints are uncached.

## What this does and does not establish

It establishes that the layer-distributed *shape* is not a finite-sample artifact. Under a
random relabeling the null profile is flat, Gini collapses toward 0.04 and n80/L rises toward
0.80, which is what 28 near-equal layers give. The observed profile is far from that.

It does **not** establish that the signal is refusal-specific. Relabeling destroys class
structure entirely, so it answers "is there any harmful-versus-harmless signal" and not "is that
signal about refusal". The two classes also come from different corpora, so topic, style and
source confounding is a separate question that no label permutation can settle. That control
is addressed separately, and partly answered, in `matched_controls.md`.

## Still outstanding

The other eight models. The null is model-independent in expectation, and the closed form is
now validated against a real measurement, so the analytic value carries them. Running them
empirically needs eight uncached checkpoints, roughly 4 to 8 minutes of extraction each on the
sharded path.
