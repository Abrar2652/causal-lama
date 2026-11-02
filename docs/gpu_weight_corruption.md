# Critical: inference silently corrupts model weights in GPU memory on ckg08

**Severity: this voids any GPU extraction performed on this host. No effect on the published
numbers, and no effect on any analysis in this audit, all of which derives from tensors
extracted on the Ampere host.**

**This document supersedes `sdpa_turing_superseded.md`, whose diagnosis was wrong.** That file
blamed the `sdpa` attention backend on Turing hardware and recommended `eager` + bfloat16 as
the fix. The recommended fix does not work, because the attention backend was never the cause.

## What is actually happening

Repeated inference on the same model object progressively modifies the model's own parameters.

A no-grad forward pass in eval mode has no code path that writes to a weight tensor. Checksumming
every parameter before and after 15 identical forward passes of one prompt, Qwen2.5-7B-Instruct,
fp16, `eager`, on `cuda:3`:

| Parameter | abs-sum before | abs-sum after | relative change |
|---|---|---|---|
| `model.layers.4.self_attn.v_proj.weight` | 21,675.1 | 26,703.5 | 23.2% |
| `model.layers.4.self_attn.k_proj.weight` | 29,096.8 | 34,291.9 | 17.9% |
| `model.layers.1.mlp.up_proj.weight` | 538,059 | 589,136 | 9.5% |
| `model.layers.2.mlp.down_proj.weight` | 699,543 | 727,397 | 4.0% |
| `model.layers.1.mlp.down_proj.weight` | 549,664 | 576,741 | 4.9% |

12 of 340 parameters and buffers changed. No tensor contained NaN or Inf, so nothing fails and
nothing warns.

The corrupted weights make the outputs non-reproducible. 25 byte-identical forward passes of one
prompt returned 25 distinct answers, degrading as the model stayed resident. Last-token
hidden-state norm at layer 1, where the Ampere reference is **15.963**:

```
16.10  15.93  15.96  15.96  16.17  36.01  77.27  77.25  77.25  77.27  79.55  122.69  ...  max 160.60
```

At layer 19 the same sequence ranges from 102.10 to 998.76 against a true value of 109.5. Only
the first five forwards after a fresh load were within 5% of the reference. The drift persists
across loops on the same object: a second loop began at 24.9 rather than returning to 16.

## What rules out the alternatives

| Hypothesis | Test | Result |
|---|---|---|
| `sdpa` attention backend | `eager` in both dtypes | still wrong; backend is irrelevant |
| fp16 range overflow | bfloat16, which has fp32 range | still wrong; no NaN or Inf anywhere |
| KV cache state across calls | `use_cache=False` | no change; still drifts |
| A bug in this transformers/torch pairing | a code bug would be deterministic | 25 identical calls gave 25 different answers |
| Faulty arithmetic units | 200 identical 4096x4096 matmuls, fp16 and bf16, on `cuda:1` and `cuda:3` | 0/200 non-identical, bitwise equal |
| Our extraction code | the repo's own path, a hand-written inline forward, and a CPU run | both GPU paths drift; CPU does not |

Deterministic arithmetic plus non-deterministic model output plus weights that change on their
own points at the memory holding those weights, not at the compute. The matmul probe stays clean
because it touches ~400 MB for a few seconds, where the model holds 15 GB resident for minutes.
These cards report `ecc.errors.uncorrected.volatile.total = [N/A]`, so uncorrected bit flips are
neither corrected nor counted.

What is established is that weights change during inference on this host and that outputs are
not reproducible. Whether the root cause is failing VRAM, the driver, or something in this
software stack writing into parameter storage is not isolated here, and does not change what
follows.

## What is void

| Tensors | Why |
|---|---|
| `qwen2-7b_VOID_gpucorrupt_aligned` | base-vs-aligned run, extracted here |
| `qwen2-7b_VOID_gpucorrupt_base` | same |
| `qwen2-7b_INVALID_sdpa_fp16` | the run originally misattributed to `sdpa` |
| `qwen2-7b_hfsource` | the SORRY-Bench extraction, made here at 08:31 |

Each carries a `DO_NOT_USE.md`. The practical loss is that the SORRY-Bench comparison, which
gave Spearman -0.79 against the published profile, is no longer usable as evidence.

## What is unaffected, and why

Everything this audit concludes rests on `qwen2-7b_processed_confirm` and
`qwen2-7b_harmless_holdout`, both extracted on the Ampere host (ckg10) and both unaffected.
`qwen2-7b_processed_confirm` reproduces the published per-layer RCS to a maximum absolute
deviation of **2.018e-04**, Spearman **+1.0000**, with layer-0 RCS agreeing to eight decimal
places (published 0.6663883328, re-extracted 0.6663883552). Every downstream number, the
permutation null, the matched controls, the deep-benign contrasts and the alternative
normalizations, is CPU-side NumPy over those tensors.

The data-source finding therefore stands on its positive half alone, which is the strong half: a
28-layer profile matching to four decimal places identifies the source beyond coincidence. It
never needed the SORRY-Bench contrast. See `data_source.md`.

## How to extract safely here

Two paths are verified against the Ampere reference:

- **CPU, fp32.** Matches to 0.03-0.06% per prompt. Costs 6.9 s per forward, so reduce the
  harmless class: the reference centroid converges fast, and 1000 harmless prompts reproduce the
  full-6000 profile at Spearman 0.9989 with the center of mass shifting 0.0014, far inside the
  bootstrap width of 0.010.
- **An Ampere host.** The repo default is sound there.

Either way, guard it. `../analysis/base_vs_aligned.py` now checksums every parameter before and
after extraction and writes a `VOID_weight_corruption.txt` marker if any changed. The check is
three lines and would have caught this immediately:

```python
before = {k: float(v.detach().float().abs().sum()) for k, v in m.named_parameters()}
...                                  # extraction
after  = {k: float(v.detach().float().abs().sum()) for k, v in m.named_parameters()}
assert before == after, "weights mutated during inference; this run is void"
```

## Why the first diagnosis was wrong

`sdpa_turing_superseded.md` compared configurations using a few probe prompts, each measured on a
model that had been resident for a different length of time. Under progressive corruption that
design cannot separate the variable under test from how long the weights had been decaying, and
it produced a table in which `sdpa` looked guilty and `eager` + bf16 looked clean. The giveaway
was missed: a correct implementation returns the same answer twice. Repeating one identical
forward, which costs seconds, would have exposed this at the start.

## Action needed outside this repository

`cuda:3` on ckg08 is returning wrong arithmetic results from resident memory without reporting
an error. This is worth raising with whoever administers the machine, and worth checking on the
other three cards, which could not be probed at model scale because other jobs held their
memory. Any result produced from GPU work on this host should be treated as suspect until it is
cleared.
