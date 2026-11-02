# Execution summary

**Dates** 2026-10-31 to 2026-11-01 · **Hosts** ckg10 (Ampere A5000: the audit, and all GPU
extraction) and ckg08 (Turing Quadro RTX 8000: CPU only, for the reason in the hazard note
below) · **Repository** github.com/Abrar2652/causal-lama at `203160e` (cloned fresh; no prior
local checkout existed).

## Environment actually used

| Item | Value |
|---|---|
| Python | 3.10.12 (/usr/bin/python3) |
| NumPy / SciPy / scikit-learn / PyTorch | 2.2.6 / 1.13.1 / 1.7.2 / 2.7.1+cu126 |
| CPU | 40 logical cores; BLAS capped at 12 threads (shared machine) |
| GPU (ckg10) | 8x RTX A5000 24 GB, sm_86, driver 580.178.04, shared with other tenants. Table verification and reanalysis used **no GPU**; the representation extractions used 1-3 cards, sharded with `device_map="auto"` because no single card had 17 GiB free |
| GPU (ckg08) | 8x Quadro RTX 8000 48 GB, sm_75. **Not usable for inference**: weights are mutated in VRAM mid-forward (hazard note below), so extraction on that host ran on CPU |
| Supplement's reported hardware | 8x RTX A6000 48 GB, driver 535.274.02 — a different machine, so timings here are not comparable |

Scope of what actually ran, stated precisely because an earlier version of this line was
wrong. The table verification and every reanalysis under `verification/` and
`permutation_null/` used cached artifacts only: no model, no generation, no download. The
controls added later *did* load models, because they cannot be done otherwise: regenerating
the paper's own representations and comparing a base checkpoint against an aligned one both
require forward passes. So: **two checkpoints were loaded** (`Qwen/Qwen2.5-7B-Instruct`,
already cached, and `Qwen/Qwen2.5-7B`, downloaded 2026-10-31 with the author's approval,
14.19 GiB); **no text was ever generated**, since only hidden states were read, so no model
completion, harmful or otherwise, exists in any artifact; and **nothing was uploaded**.

## Commands run

```bash
git clone https://github.com/Abrar2652/causal-lama           # 99 MB, 2 commits

# existing test suites (pre-existing code, unmodified)
python3 tests/test_rcs_synthetic.py                          # PASS (7 groups)
python3 tests/test_adaptive_ablation.py                      # PASS (3 groups)
python3 lama_stats.py                                        # PASS (self-test)

# new tests for the permutation-null machinery
python3 analysis/test_permutation_null.py  # PASS (9 groups, 25 assertions)

# verification of published tables, all from cached artifacts
python3 analysis/verify_table1.py
python3 analysis/verify_table2.py
python3 analysis/verify_ci_procedure.py
python3 analysis/verify_significance.py
python3 analysis/verify_significance_multiseed.py
python3 analysis/verify_table3.py
python3 analysis/verify_table5.py

# the permuted-label null, calibrated form: all 9 models, from cached per-layer statistics
python3 analysis/permutation_null_calibrated.py --models all --n_perm 200 \
        --eff_rank isotropic --tag primary                   # 341 s
python3 analysis/permutation_null_calibrated.py --models qwen2-7b --n_perm 200 \
        --eff_rank 100 --tag effrank100                      # 34 s
python3 analysis/permutation_null_calibrated.py --models qwen2-7b --n_perm 200 \
        --eff_rank 20 --tag effrank20                        # 33 s
python3 analysis/plot_permutation_null.py

# the empirical null, on activations regenerated from the in-repo split (qp6c W1).
# No longer blocked: the prerequisite list that declared it impossible was wrong.
python3 analysis/extract_sharded.py          # -> qwen2-7b_processed_confirm
python3 analysis/verify_data_source.py       # confirms the source by reproduction
python3 analysis/permutation_null_empirical.py --model qwen2-7b \
        --n_perm 200 --seed 42 \
        --reps_dir dataset/representations/qwen2-7b_processed_confirm

# refusal-specificity controls
python3 analysis/extract_control_sharded.py --start 6000 --n 7000
python3 analysis/matched_controls.py
python3 analysis/matched_control_harmless.py
python3 analysis/nonrefusal_controls.py
python3 analysis/deep_benign_contrasts.py
python3 analysis/length_matched_depth.py

# base versus aligned (qp6c, PKr6). The CPU run was forced by ckg08's weight corruption;
# the GPU run on ckg10 is what makes the full 6000-prompt harmless class affordable.
python3 analysis/base_vs_aligned.py --device cpu --dtype float32 \
        --attn eager --n_harmless 1000 --suffix _cpu_fp32          # ckg08, ~2.5 h per model
        # `eager` is safe ONLY at fp32; with fp16 it overflows layer 27. Do not reuse this line.
python3 analysis/base_vs_aligned.py --shard "7:11.6,6:4.7,1:4.6" \
        --dtype float16 --attn sdpa --n_harmless 6000 \
        --suffix _gpu_fp16_full                                    # ckg10, ~6 min per model
python3 analysis/analyze_base_vs_aligned.py --suffix _gpu_fp16_full \
        --out base_vs_aligned_qwen2-7b_full6000

# joint multi-layer causal ablation (PKr6) -- the control the causal framing rests on
python3 analysis/joint_causal_ablation.py --model qwen2-7b \
        --reps qwen2-7b_processed_confirm                          # 284 s, 99 configurations
python3 analysis/random_direction_control.py --model qwen2-7b

# author response drafts, and the gate on their numbers
python3 analysis/verify_rebuttal_claims.py   # 20/20, exits non-zero on drift

# paper-ready tables
python3 analysis/make_tex_tables.py              # 8 tables, 8 insights, 8 failure modes
```

## Tests

| Suite | Result |
|---|---|
| `../tests/test_rcs_synthetic.py` (pre-existing) | **PASS** |
| `../tests/test_adaptive_ablation.py` (pre-existing) | **PASS** |
| `../lama_stats.py` self-test (pre-existing) | **PASS** |
| `../analysis/test_permutation_null.py` (new) | **PASS** |
| `../analysis/analyze_base_vs_aligned.py` regression after parameterizing its input path | **PASS** -- 152 shared keys, **0 changed values** |
| `../analysis/analyze_base_vs_aligned.py` decomposition, recomputed under code | **PASS** -- reproduces all 9 stored values to 6e-08 (float32 summation order) |
| `../analysis/joint_causal_ablation.py` against the published single-layer probe | **PASS** -- Spearman 0.99945, max abs dev 2.8e-04 |
| `../analysis/joint_causal_ablation.py` restore-to-baseline after every ablation | **PASS** -- deviation exactly 0.0 |
| Baseline P_refuse across two independent processes | **PASS** -- deviation exactly 0.00e+00 |

The new suite covers: Gini = 0 on a flat profile (4 sizes); n80 = ceil(0.8L) on a uniform
profile (7 sizes); monotonicity of Gini and n80 under concentration; exact class-count
preservation; identity of our RCS with the paper's `compute_rcs` (5.2e-9); agreement of the
permuted mean with the closed form (1.3%); proof that the denominator is **recomputed** and not
frozen; determinism under a fixed seed; and flatness of the permuted profile (Gini ~ 0.018,
n80/L ~ 0.808) against a more concentrated observed profile.

A vectorized fast path was added and checked against the straightforward loop (max relative
difference 6.7e-6) before being used for any reported number.

## Sample sizes

| Quantity | n |
|---|---|
| Relabelings per null run | 200, drawn once and applied at every layer |
| Class sizes in the null | n_f = 260, n_l = 6000 (the `--source processed` path actually used; supplement C's stated 4000/6000 is incorrect, see `data_source.md`) |
| Models profiled | 9 |
| Behaviors per ASR cell | 159 |
| Bootstrap resamples in interval checks | 10,000 |
| Benign prompts in utility checks | 200 |

## Headline numerical results

- **Permuted-label null**: closed form 0.06335; simulated 0.0636-0.0681; observed per-model mean
  RCS 0.79-3.06, i.e. **12x-45x the null**. Observed Gini 0.084-0.231 against null 0.004-0.035;
  observed n80/L 0.643-0.750 against null 0.799-0.833. Empirical tail probability **0.000** on
  both statistics for all nine models. Conclusion holds down to effective rank 20.
- **Table 1** reproduces exactly for all nine models (L, peak layer and n80 exact; Gini within
  0.0005). Eq. (2) identity holds to 2.1e-7.
- **Table 3** reproduces exactly (max deviation 0.0004), both graded and binary rows.
- **Table 6** reproduces (0.8369 / 0.6132 / 0.7591 against published 0.84 / 0.61 / 0.76).
- **Table 2** point ASRs reproduce to 0.05 pp for four of five models.
- **Table 5** reproduces to 0.005 PPL for every cell.
- **Significance**: both starred positive claims and all three Gemma negatives survive a
  behavior-level paired test; the two marginal daggers do not.

## Discrepancies found

Nine, listed in `evidence_inventory.md` §C with their artifacts. The consequential ones: the
published confidence intervals follow a pooled-independent procedure rather than the described
hierarchical one and are about sqrt(3) too narrow; the seeds are degenerate on two of five
models; the Mistral HarmBench column uses two seeds while the caption says three; the BO trial
count is 48 everywhere in the release against 128 in the supplement; and Llama-2-7B's BO
objective is identically zero.

## Blockers

Four of the six items previously listed here are now resolved. The table records what cleared
and why, because two of them were never genuinely blocked and the mistaken diagnosis is worth
keeping visible.

### Cleared

| Work | How it cleared |
|---|---|
| Permutation null on the paper's own activations | Tensors regenerated from the in-repo processed split; they reproduce the published per-layer RCS to `max\|diff\| = 2.0e-04`. See `permutation_null.md`. |
| Regenerating those tensors faithfully | The prerequisite list was wrong. The published runs used the in-repo split (260/6000), so neither SORRY-Bench access nor an HF token was ever needed. The GPU ceiling was an artifact of insisting on one card; three partly-free cards sufficed. |
| Refusal-specificity controls | Length- and register-matched controls plus five non-refusal contrasts, all on held-out harmless prompts. See `matched_controls.md`. Topic matching is still keyword-based rather than label-based, and is reported as coarse. |
| Harmless- and pooled-spread normalizations | Needed sample-level tensors, which now exist. All seven denominator variants computed; see `../results/verification/alt_normalizations_qwen2-7b.csv`. |

### Still missing

| Blocked work | Exact missing input | Cost once available |
|---|---|---|
| Topic-matched provenance contrast | real topic labels; the `category` field is uniformly `"None"` in both splits | analysis only |
| The other eight models | eight uncached checkpoints | 6-20 minutes of extraction each |
| Multi-layer joint causal interventions on the other 8 models | eight uncached checkpoints | ~5 min extraction + ~5 min sweep each; the Qwen2.5-7B run is done, see `joint_causal.md` |

The base-versus-aligned comparison, previously listed here as the highest-value missing
experiment, has been run. The base checkpoint was downloaded with the author's approval
(14.19 GiB, four shards) and both models were extracted on CPU fp32 in one process with
identical prompts and asserted token-id identity, at 6.9 s per forward, about 2.5 h per model
for 1260 prompts. The GPU path was attempted first and voided by the weight corruption recorded
below. Result in `matched_controls.md` Part 4; it separates score magnitude, which is
60% pre-existing, from late concentration, which alignment creates.

## Environment hazards worth recording

**GPU inference on ckg08 corrupts model weights in memory, voiding any extraction done
there.** Repeated byte-identical forwards return different answers, and 12 of 340 parameters
change their absolute sum during a no-grad eval forward, by up to 23%, with no NaN and no error.
The attention backend, the dtype, the KV cache, our code and the arithmetic units are all ruled
out. Extract on CPU in fp32, which matches the Ampere reference to 0.03-0.06%, or on an Ampere
host, and checksum the parameters around any extraction. Full evidence in
`gpu_weight_corruption.md`. **This supersedes an earlier entry here that blamed the
`sdpa` attention backend and recommended `eager` + bfloat16; both were wrong.** Void tensors are
quarantined under `dataset/representations/qwen2-7b_VOID_gpucorrupt_*`,
`qwen2-7b_INVALID_sdpa_fp16/` and `qwen2-7b_hfsource/`.

**`eager` attention with fp16 silently poisons the final layer on Qwen2.5-7B.** A first
full-6000 extraction on ckg10 -- sound Ampere hardware, with **0 of 339 parameters mutated** --
wrote tensors whose layer 27, and only layer 27, held NaN for 42% of the aligned harmless
prompts and 17% of the base ones. No error, exit code 0, every other layer finite. The weight
checksum could not see it, because the weights were never the problem: the activations
overflowed. The `sdpa` path at the same dtype on the same host is clean and reproduces the
published profile, and `eager` had only been chosen because of the retracted
`sdpa_turing_superseded.md` diagnosis. `../analysis/base_vs_aligned.py` now takes `--attn` (default `sdpa`)
and calls `assert_finite()` on every tensor before saving, so this fails loudly instead of
three analyses downstream. The poisoned tensors are quarantined under
`qwen2-7b_{aligned,base}_VOID_eagerfp16_nan/` with `DO_NOT_USE.md` markers. **Lesson: a weight
checksum is not an activation check, and both are needed.**

**nvidia-smi works on ckg08**, contrary to an earlier note, but reports ECC counters as `[N/A]`
on these Quadro RTX 8000s, so silent bit flips are neither corrected nor counted.

**One sandbox denial.** Reading the HuggingFace token file was refused under the
credential-access policy. It was not worked around, so token availability is reported as
**unverified** rather than absent. This gates nothing: the published representations come from
the in-repo split, which needs no credential.

## Files written

Everything new lives under ``. No existing file, cached artifact, result
table, log or PDF was modified or deleted.

```

  evidence_inventory.md  reused_evidence.md  reviewer_rebuttal_notes.md  execution_summary.md
  scripts/      null_core.py, permutation_null_calibrated.py, permutation_null_empirical.py,
                test_permutation_null.py, plot_permutation_null.py, verify_table{1,2,3,5}.py,
                verify_ci_procedure.py, verify_significance.py, verify_significance_multiseed.py
  permutation_null/   RESOLVED.md, config_*.json, results_*.json, summary_*.csv,
                      permutation_distribution_*.npz, plots/
  matched_controls/   RESULTS.md, matched_controls_qwen2-7b.{json,csv},
                      nonrefusal_controls_qwen2-7b.{json,csv}
  verification/       table1_audit.csv, table2_audit.csv, table3_audit.csv, table5_audit.csv,
                      ci_procedure_audit.csv, ci_candidateC_audit.csv, mistral_per_seed_audit.csv,
                      seed_identity_audit.csv, significance_behaviorlevel_audit.csv,
                      significance_multiseed_audit.csv, alt_normalization_audit.csv
```
