# MD baseline fidelity: what was run, and how to describe it

> **Author decision (2026-10-31): proceed with the existing experimental setup.** The MD
> re-run at the published configuration is NOT being done. This document therefore records what
> was actually run and the disclosure wording needed so the paper describes it accurately.
> The factual findings below are unchanged.

Found 2026-10-31 by comparing against the baseline's own repository,
<https://github.com/pralab/som-refusal-directions> (Piras et al., AAAI 2026).

This supersedes the "NOT_DIAGNOSED" verdict I recorded earlier for reviewer qp6c's W4. The
cause is now identified in code, and it is broader than W4 states: it affects **every MD
comparison in the paper**, not only Llama-2-7B.

## The comparison is apples-to-apples

Before drawing any conclusion, the two setups were checked for comparability. They match:

| Component | Status |
|---|---|
| `harmbench_test.json` (159 behaviors) | **byte-identical** |
| `harmbench_val`, `harmful_train`, `harmless_train`, `harmful_val`, `harmless_val` | **byte-identical** |
| Judge | both `cais/HarmBench-Llama-2-13b-cls` |
| Decoding | both `do_sample=False`, `max_new_tokens=64`, `temperature=None` |
| Direction ordering convention | both sort SOM neurons by harmful-win count, descending |
| SOM configuration | both `MiniSom(4, 4, sigma=0.33, lr=0.01, 10000 iters, hexagonal, euclidean, random_seed=0)` |

So the baseline's published ASRs are directly comparable to ours, and direction indices mean
the same thing in both repositories. This also **rules out** the template, decoding and judge
explanations that qp6c hypothesized.

## Defect 1 — MD uses the wrong directions, on every model

`../run_ablation_study.py`:

```python
# Apply MD: top-7 directions (matching baseline paper's best config), uniform alpha=1
# For qwen2-7b the paper uses direction ids 10-3-5-2-4-6-13; we use the same ranked set
MD_DIR_IDS = [0, 1, 2, 3, 4, 5, 6]   # top-7 by harmful-win count (SOM sorts them)
```

The comment states the intent correctly and the code does something else. The baseline's
configurations are **Bayesian-optimization-selected** sets of varying size, and they
deliberately include low-win-count neurons that a top-k heuristic excludes by construction:

| Model | published direction ids | published count | ours |
|---|---|---|---|
| Qwen2.5-7B | 10-3-5-2-4-6-13 | 7 | 0-1-2-3-4-5-6 |
| Qwen2.5-3B | 2-7-9-14-8 | 5 | 0-1-2-3-4-5-6 |
| Llama-2-7B | 8-11-4-7-13-6-12 | 7 | 0-1-2-3-4-5-6 |
| Gemma2-9B | 7-6-15-2-13-3 | 6 | 0-1-2-3-4-5-6 |
| Mistral-7B-RR | 9-5-6-3-1 | 5 | 0-1-2-3-4-5-6 |

Every published set contains indices above 6 or uses fewer than 7 directions, so no published
set equals `[0..6]`. Selecting directions by search **is** the baseline's method; replacing it
with a top-7 heuristic removes the step the baseline exists to perform. The baseline's README
recommends `--trials 512` for that search; our MD runs none.

## Defect 2 — MD is built at the wrong source layer on three of five models

`../run_ablation_study.py` line 171:

```python
best_l = int(min(ls_data, key=lambda k: ls_data[k]['Refusal']))
```

This takes the lowest refusal score with **no KL-divergence constraint**. Both the baseline and
this repository's own `../utils/select_layer.py` apply `KL_Div < 0.1` first:

```python
valid = {k: v for k, v in scores.items() if v["KL_Div"] < 0.1}   # previous work's threshold
best_layer = min(valid, key=lambda x: valid[x]["Refusal"])
```

The `layer_selection.json` files in this repository are **byte-identical** to the baseline's, so
the inputs agree and only the rule differs.

| Model | published l* | our l* | `select_layer.py` rule on the same file |
|---|---|---|---|
| Llama-2-7B | 13 | **11** | **13** — layer 11 fails the KL filter (KL = 0.188) |
| Qwen2.5-3B | 24 | **23** | 23 (no layer passes KL < 0.1, so the rule falls back) |
| Qwen2.5-7B | 18 | **15** | 15 (same fallback) |
| Gemma2-9B | 23 | 23 | 23 |
| Mistral-7B-RR | 23 | 23 | 23 |

Llama-2-7B is an unambiguous internal inconsistency: the repository's own selector returns the
published layer 13, and the experiment used 11. For the two Qwen models the rules coincide but
still disagree with the baseline's published l*, so the baseline's layer must come from a
different procedure than the one the supplement claims we follow.

The supplement states the source layer "follows the validation-based source-layer selection
procedure of the MD baseline". That is not accurate as implemented.

## Consequence: LAMA does not beat the published MD on any model

Same test set, same judge, same decoding:

| Model | published MD | our MD | reproduction gap | our best LAMA | vs published MD |
|---|---|---|---|---|---|
| Qwen2.5-7B | 95.97% | 84.28% | -11.69 | 93.08% (RCS) | **-2.89** |
| Qwen2.5-3B | 93.71% | 83.86% | -9.85 | 89.31% (BO) | **-4.40** |
| Llama-2-7B | 59.11% | 5.66% | -53.45 | 0.00% | **-59.11** |
| Gemma2-9B | 96.27% | 95.60% | -0.67 | 68.55% (unif) | **-27.72** |
| Mistral-7B-RR | 25.79% | 8.18% | -17.61 | 20.13% (unif) | **-5.66** |

Mean MD reproduction gap **-18.7 points**. The headline "+8.8 points over MD" on Qwen2.5-7B is
an improvement over a weakened reproduction; measured against the baseline's published number,
LAMA-RCS is 2.9 points **worse**.

Note also that the reproduction gap varies enormously by model (-0.7 on Gemma2-9B, -53.5 on
Llama-2-7B). Since RQ4 regresses "lift over MD" on RCS-causal alignment, and the lift depends
on how badly MD happens to be reproduced for each model, the alignment-versus-lift ordering may
partly track baseline-reproduction quality rather than alignment. Gemma2-9B, the outlier that
drives that analysis, is also the model where MD was reproduced most faithfully.

## What is and is not affected

**Not affected.** Tables 1, 3 and 6, and the whole RCS analysis, including the new permutation
null. These never involve MD, and all of them reproduce exactly.

**Affected.** Table 2 (every ASR comparison), Table 4 (component ablation, all deltas are
"vs MD"), Table 5 (benign cost, the MD column), and the claims in RQ3, RQ4, RQ5 and RQ6 that are
framed relative to MD.

## Decision: keep the setup, correct the description

The experiment stays as run. Two sentences in the manuscript currently claim a fidelity the
code does not have, and those must change:

- Supplement C: the source layer "follows the validation-based source-layer selection procedure
  of the MD baseline (Piras et al. 2026)." It does not; the KL constraint is dropped.
- Table 5 caption: "our reproduced MD baseline." It is a reimplementation with a different
  direction-selection rule, not a reproduction.

### Why the existing setup is still defensible as a control

Everything except the two axes under study is matched across all four conditions: same
representations, same SOM hyperparameters and seed, same judge, same greedy decoding, same
seeds, same 159 test behaviors, and the same direction pool. MD and LAMA-uniform differ **only**
in single-layer versus group-wise direction construction. LAMA-uniform and LAMA-RCS differ
**only** in the per-layer strength schedule. That makes Table 4 a clean internal component
ablation, which is what RQ5 actually needs.

What it does not support is a claim of improving on the published method. Reframe MD from a
reproduction to a **matched single-layer control**, and the comparisons become statements about
the two design axes rather than about beating Piras et al.

### Paste-ready disclosure wording

For Section 5 (experimental setup):

> As an internal control we implement the multi-directional (MD) construction of Piras et al.
> (2026) at a single validation-selected source layer, taking the top-k SOM directions ranked by
> harmful-win count rather than their Bayesian-optimization-selected subset. All conditions
> share the same representations, SOM hyperparameters and seed, judge, decoding settings, seeds
> and test behaviors, so MD and the LAMA variants differ only along the two axes we study. This
> control is therefore matched to our configurations but is not a reproduction of the authors'
> tuned configuration, and our numbers for it are not comparable to those they report.

For the Limitations section:

> Our MD control uses a top-k direction heuristic and a source layer selected by refusal score
> alone, whereas the original method selects directions by Bayesian optimization at a layer
> chosen under an additional KL-divergence constraint. Our MD numbers are accordingly lower than
> the published ones, so our comparisons should be read as isolating the contribution of
> depth-specific direction construction and per-layer strength within a matched pipeline, not as
> a demonstration that LAMA improves on the published method.

### Claims to adjust

| Current claim | Adjusted |
|---|---|
| "improves attack success by up to 8.8 points over the multi-directional (MD) baseline" | "over a matched single-layer MD control" |
| "substantially reducing benign-perplexity cost" (vs MD) | keep, but scope to the control; the 4.5x perplexity inflation is a property of the control, not of the published method |
| Supplement C source-layer sentence | state the rule actually used: lowest refusal score, no KL constraint |
| Table 5 caption "our reproduced MD baseline" | "our MD control" |

### Residual risk to accept knowingly

The repository is public and the direction list is a single hardcoded line, so a reviewer who
looks can see it. Disclosing it first is strictly better than being asked about it.

### If compute frees up later

Both repositories train the SOM with identical hyperparameters and `random_seed=0`, so given
identical representations the published direction indices are directly transferable. A corrected
run needs each model's published source layer together with its published ids and count; it does
not need the 512-trial search. The blocker is the deleted representation tensors.
