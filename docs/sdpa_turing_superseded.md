# Superseded: this diagnosis was wrong

This file blamed the `sdpa` attention backend on Turing GPUs for corrupt activations, and
recommended `attn_implementation="eager"` with bfloat16 as the fix. **Both the diagnosis and the
fix are wrong.** The attention backend is irrelevant; `eager` is corrupt too.

The real cause is that inference on this host silently modifies the model's own weights in GPU
memory, so repeated identical forward passes return different answers. See
**`gpu_weight_corruption.md`**, which supersedes this file and records the evidence,
what it voids, and how to extract safely.

Kept as a pointer so existing references resolve, and because the mistake is instructive: the
original comparison tested configurations on models that had been resident for different lengths
of time, which under progressive corruption cannot isolate the variable being tested.

## Addendum (2026-11-01): `eager` is independently unsafe at fp16

A second, unrelated reason not to revive this file's recommendation. On **ckg10, which is sound
Ampere hardware with none of the weight corruption described above**, extracting Qwen2.5-7B with
`attn_implementation="eager"` and `float16` wrote tensors whose **layer 27, and only layer 27**,
held NaN for 42% of the aligned harmless prompts and 17% of the base ones. Exit code 0, no
error, every other layer finite, and **0 of 339 parameters mutated**, so the weight checksum
reported a clean run. The weights were fine; the activations overflowed.

`sdpa` at the same dtype on the same host is clean and reproduces the published per-layer
profile at Spearman 1.0000 (`max|diff| = 2e-04`).

So the fix this file proposed is wrong twice over: wrong about the cause on Turing, and
actively harmful at fp16 on Ampere. Use `sdpa`, and check activations as well as weights —
`../analysis/base_vs_aligned.py` now defaults to `--attn sdpa` and calls
`assert_finite()` before saving any tensor. **A weight checksum is not an activation check.**
