# Causal-LAMA: Refusal is Layer-Distributed — Causal Analysis and Adaptive Ablation for Multi-Directional Refusal Suppression


---

## Scientific Claim

> Refusal in LLMs is **not** encoded at a single layer; rather it is **causally distributed** across a structured layer hierarchy. The purely representational metric **RCS (Refusal Contribution Score)** predicts which layers causally control refusal, validated by Spearman ρ ≥ 0.7 between RCS and ΔASR.

### Key Contributions

1. **Scientific Finding**: Refusal is causally layer-distributed (first systematic evidence)
2. **Diagnostic Metric**: RCS — a causal predictor for refusal structure analysis (with bootstrap CI, Cohen's d)
3. **Method**: LAMA — Layer-Adaptive Multi-Directional Ablation
4. **Interpretability**: Cross-model refusal fingerprints via RCS profiles

### How It Differs from the Baseline

| Aspect | Baseline (SOM) | Causal-LAMA |
|--------|---------------|-------------|
| Layer selection | Single "best" layer | All layers analyzed via RCS |
| SOM training | One SOM at one layer | Per-group SOMs at representative layers |
| Ablation weights | Uniform α=1 for all layers | Adaptive α_l guided by RCS |
| Scientific claim | "Refusal is a manifold" | "Refusal is a **layer-distributed** manifold" |
| Validation | Empirical ASR only | Causal validation (RCS ↔ ΔASR correlation) |

---

## Project Structure

```
causal-lama/
├── Core LAMA Modules
│   ├── lama_rcs.py                  # RCS computation, causal validation, bootstrap CI, layer grouping
│   ├── lama_generate_directions.py  # Per-group SOM training, direction extraction
│   ├── lama_optuna_search.py        # Joint Bayesian optimization (directions + layer weights)
│   └── lama_ablate.py               # Ablation entry point (3 modes: uniform/rcs/bo)
│
├── Baseline Modules (used by LAMA)
│   ├── som.py                       # SOM training via MiniSom
│   ├── som_generate_directions.py   # Original single-layer direction extraction
│   ├── optuna_search.py             # Original BO search
│   ├── ablate.py                    # Original uniform ablation
│   ├── directions_ablation.py       # Completion generation pipeline
│   ├── eval_jailbreaks.py           # HarmBench judge evaluation
│   ├── create_representation_dataset.py  # Extract all-layer hidden states
│   └── config.py                    # Path configuration
│
├── Model & Data
│   ├── models/                      # LanguageModel classes + adaptive ablation
│   ├── dataset/                     # Data loading + pre-computed representations
│   └── utils/                       # Ablation utilities (weighted ablation)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Create Virtual Environment

```bash
# Create a virtual environment
python -m venv causal-lama-env

# Activate it
# On Linux/macOS:
source causal-lama-env/bin/activate
# On Windows (PowerShell):
.\causal-lama-env\Scripts\Activate.ps1
# On Windows (CMD):
causal-lama-env\Scripts\activate.bat

# Verify
python --version  # Should be >= 3.9
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `vllm` (needed for HarmBench evaluation) requires Linux + CUDA. On Windows, you can skip it and run the evaluation on Kaggle/Colab instead:
> ```bash
> pip install torch transformers minisom optuna scipy matplotlib numpy accelerate
> ```

### 3. HuggingFace Login

Request access to the models you want to use, then:

```bash
huggingface-cli login
# Enter your HuggingFace token
```

### Supported Models

| Model Alias | HuggingFace Path | Layers | Hidden Dim | VRAM |
|-------------|------------------|--------|-----------|------|
| `llama2-7b` | `meta-llama/Llama-2-7b-chat-hf` | 32 | 4096 | ~14GB |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | 32 | 4096 | ~16GB |
| `qwen-7b` | `Qwen/Qwen-7B-Chat` | 32 | 4096 | ~14GB |
| `qwen-14b` | `Qwen/Qwen-14B-Chat` | 40 | 5120 | ~28GB |
| `qwen2-3b` | `Qwen/Qwen2.5-3B-Instruct` | 36 | 2048 | ~7GB |
| `qwen2-7b` | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3584 | ~14GB |
| `gemma2-9b` | `google/gemma-2-9b-it` | 42 | 3584 | ~18GB |
| `mistral-7b-rr` | `GraySwanAI/Mistral-7B-Instruct-RR` | 32 | 4096 | ~14GB |

---

## Data Setup

This codebase requires **three types of data**. Here's what's included and what you need to generate:

### A. Text Datasets ✅ (Already Included)

These ship with the repository in `dataset/processed/`:

| File | Contents | Samples | Used For |
|------|----------|---------|----------|
| `harmful_train.json` | Harmful prompts (train split) | ~200 | Representation extraction |
| `harmful_val.json` | Harmful prompts (validation) | ~50 | BO validation |
| `harmless_train.json` | Harmless prompts (train split) | ~6000 | Representation extraction |
| `harmless_val.json` | Harmless prompts (validation) | ~2000 | Perplexity evaluation |
| `harmbench_val.json` | HarmBench validation behaviors | ~40 | BO objective evaluation |
| `harmbench_test.json` | HarmBench test behaviors | ~159 | Final test evaluation |

> **No action needed** — these files are ready to use.

### B. HuggingFace Datasets 🔄 (Auto-Downloaded)

The `create_representation_dataset.py` script automatically downloads from HuggingFace:
- **[sorry-bench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202406)** — harmful prompts (4000 selected)
- **[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)** — harmless prompts (6000 selected)

These are downloaded automatically on first run. You need internet access and a HuggingFace token.

### C. Representation Tensors ⚠️ (Must Be Generated — Requires GPU)

This is the **most important step**. All LAMA analysis starts from pre-extracted hidden-state representations.

```bash
# Generate representations for your chosen model (requires GPU, ~15-30 min per model)
python create_representation_dataset.py --model_name llama2-7b --device cuda:0
```

**What it creates:**

```
dataset/representations/llama2-7b/
├── HFx_train.pt   # Harmful representations — shape: [4000, 32, 4096]
└── HLx_train.pt   # Harmless representations — shape: [6000, 32, 4096]
```

- Dimensions: `[n_samples, n_layers, hidden_dim]`
- File size: ~1-3 GB per model
- Time: ~15-30 min per model on a single GPU

> ⚠️ **This step requires a GPU with enough VRAM** to load the target model (see the Supported Models table for VRAM requirements).
>
> If you don't have a local GPU, you can run this step on **Kaggle** or **Google Colab** and download the `.pt` files.

**Generate for multiple models:**

```bash
# Run for each model you want to evaluate
python create_representation_dataset.py --model_name llama2-7b --device cuda:0
python create_representation_dataset.py --model_name llama3-8b --device cuda:0
python create_representation_dataset.py --model_name qwen-7b --device cuda:0
python create_representation_dataset.py --model_name gemma2-9b --device cuda:0
```

### D. HarmBench Judge Model 🔄 (Auto-Downloaded During Evaluation)

The evaluation step (Step 6) uses the [HarmBench classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls) which is **auto-downloaded** by `vllm` on first use. This requires:
- ~26GB disk space (13B parameter model)
- GPU with ≥16GB VRAM
- `vllm` installed (Linux only)

### Summary: What You Need Before Starting

| What | Status | Action |
|------|--------|--------|
| Text datasets | ✅ Included | None |
| HuggingFace token | ❌ Needed | `huggingface-cli login` |
| Representation tensors | ❌ Must generate | Run `create_representation_dataset.py` |
| HarmBench judge | 🔄 Auto-download | Needs `vllm` + GPU during evaluation |

---

## Full Pipeline

### Step 1: Extract Representations

Extract hidden-state representations from **all layers** for both harmful and harmless prompts:

```bash
python create_representation_dataset.py --model_name llama2-7b
```

**Output**: `dataset/representations/llama2-7b/{HFx_train.pt, HLx_train.pt}`
- Shape: `[n_samples, n_layers, hidden_dim]`

> The baseline extracted representations from a single layer. We extract from **all layers** — this is what enables the RCS analysis.

### Step 2: RCS Discovery — Compute Refusal Contribution Scores

Compute RCS for each layer to discover which layers encode refusal:

```bash
python lama_rcs.py --model_name llama2-7b --n_groups 3 --save_path logs
```

**Output**: `logs/rcs_analysis_llama2-7b.json`, `logs/rcs_profile_llama2-7b.pdf`

This produces:
- Per-layer RCS scores
- Layer grouping (quantile/peaks/adaptive)
- RCS profile visualization

#### Optional: Run Causal Validation (GPU required)

```bash
python lama_rcs.py --model_name llama2-7b --n_groups 3 --run_causal --device cuda:0
```

This performs single-layer ablation sweeps and computes Spearman ρ (RCS vs ΔASR) with bootstrap 95% CI. **This is the most important experiment** — it validates the scientific claim.

### Step 3: Generate Per-Group Directions

Train a SOM for each layer group at its representative layer:

```bash
python lama_generate_directions.py \
    --model_name llama2-7b \
    --n_groups 3 \
    --som_x 4 --som_y 4 \
    --sigma 0.33 --lr 0.01 \
    --iterations 10000
```

**Output**: `runs/llama2-7b/generate_directions/`
- `lama_som4x4_sigma0.33_groups3_group{0,1,2}_directions.pt`
- `lama_som4x4_sigma0.33_groups3_metadata.json`

> The baseline trained ONE SOM at ONE layer. We train one SOM per group — each captures refusal features at its depth region.

### Step 4: Bayesian Optimization Search

Jointly optimize direction selection + layer weights α_l:

```bash
python lama_optuna_search.py \
    --model_name llama2-7b \
    --directions_dir ./runs/llama2-7b/generate_directions \
    --directions_prefix lama_som4x4_sigma0.33_groups3 \
    --trials 128 \
    --dirs_per_group 3 \
    --optimize_alpha \
    --seed 42
```

**Output**: `runs/llama2-7b/lama_optimizations/`
- `lama_*_results_t128.json` — best directions, layer weights, trial history
- `lama_*_study_t128.pkl` — Optuna study object

> The baseline optimized only direction indices. LAMA jointly optimizes directions **and** per-layer ablation weights, with RCS-guided initialization for faster convergence.

### Step 5: Ablate and Generate Completions

Apply the optimized LAMA ablation on the **test set**:

```bash
# Full LAMA (BO-optimized weights)
python lama_ablate.py \
    --model_name llama2-7b \
    --directions_dir ./runs/llama2-7b/generate_directions \
    --directions_prefix lama_som4x4_sigma0.33_groups3 \
    --optimization_results ./runs/llama2-7b/lama_optimizations/lama_lama_som4x4_sigma0.33_groups3_results_t128.json \
    --dataset_name harmbench_test \
    --ablation_mode bo
```

**Output**: `runs/llama2-7b/lama_completions/*.json`

### Step 6: Evaluate with HarmBench Judge

```bash
python eval_jailbreaks.py \
    --completions_path ./runs/llama2-7b/lama_completions/*_completions.json \
    --evaluation_path ./runs/llama2-7b/lama_evaluations/lama_bo_results.json \
    --methodologies harmbench
```

---

## Ablation Study

LAMA supports three ablation modes to isolate which component drives improvement:

```bash
# Mode 1: Uniform — per-group SOMs + α=1 for all layers
# Tests: "Do per-group SOMs alone improve over single-layer SOM?"
python lama_ablate.py \
    --model_name llama2-7b \
    --ablation_mode uniform \
    --directions_dir ./runs/llama2-7b/generate_directions \
    --directions_prefix lama_som4x4_sigma0.33_groups3 \
    --dir_ids_per_group "0:1,2,3;1:0,4,5;2:1,2"

# Mode 2: RCS — per-group SOMs + RCS-initialized α (no BO)
# Tests: "Does RCS-guided weighting help beyond uniform?"
python lama_ablate.py \
    --model_name llama2-7b \
    --ablation_mode rcs \
    --directions_dir ./runs/llama2-7b/generate_directions \
    --directions_prefix lama_som4x4_sigma0.33_groups3 \
    --dir_ids_per_group "0:1,2,3;1:0,4,5;2:1,2"

# Mode 3: BO — per-group SOMs + BO-optimized α (full method)
# Tests: "Does BO fine-tuning improve over RCS initialization?"
python lama_ablate.py \
    --model_name llama2-7b \
    --ablation_mode bo \
    --directions_dir ./runs/llama2-7b/generate_directions \
    --directions_prefix lama_som4x4_sigma0.33_groups3 \
    --optimization_results ./runs/llama2-7b/lama_optimizations/lama_*_results.json
```

| Method | Directions | Weights | What It Tests |
|--------|-----------|---------|--------------|
| MD (baseline) | Single-layer SOM | α=1 uniform | Baseline |
| LAMA-uniform | Per-group SOMs | α=1 uniform | Per-group SOM quality |
| LAMA-RCS | Per-group SOMs | α from RCS | RCS-guided weighting |
| LAMA-BO | Per-group SOMs | α from BO | Full LAMA method |

---

## Reproducing Baseline Results

To run the original MD baseline for comparison:

```bash
# Train SOM at the baseline's best layer
python som_generate_directions.py \
    --som_x 4 --som_y 4 --sigma 0.33 --lr 0.01 \
    --layer 13 --model_name llama2-7b

# Ablate with baseline directions
python ablate.py \
    --directions_path ./runs/llama2-7b/generate_directions/som4x4_sigma0.33_layer13_directions.pt \
    --model_name llama2-7b \
    --dir_ids 8 11 4 7 13 6 12

# Evaluate
python eval_jailbreaks.py \
    --completions_path ./runs/llama2-7b/completions/*_completions.json \
    --evaluation_path ./runs/llama2-7b/evaluations/md_baseline_results.json \
    --methodologies harmbench
```

### Baseline Reported Results

| Model | Best Layer | Direction IDs | ASR |
|-------|-----------|---------------|-----|
| LLaMA2-7B | 13 | 8-11-4-7-13-6-12 | 59.11% |
| LLaMA3-8B | 11 | 15-3-0-12-3-11 | 88.05% |
| Qwen-7B | 16 | 2-7-9-10-0 | 88.05% |
| Qwen-14B | 22 | 0-7-3-15-11-8 | 91.82% |
| Qwen2.5-3B | 24 | 2-7-9-14-8 | 93.71% |
| Qwen2.5-7B | 18 | 10-3-5-2-4-6-13 | 95.97% |
| Gemma2-9B | 23 | 7-6-15-2-13-3 | 96.27% |
| Mistral-7B-RR | 23 | 9-5-6-3-1 | 25.79% |

---

## Running on Kaggle / Google Colab

For GPU access without a local GPU:

1. Upload this repository as a Kaggle dataset
2. Create a new notebook with **GPU T4 x2** accelerator
3. In the first cell:
   ```python
   import os
   os.chdir('/kaggle/input/causal-lama/')
   !pip install -q minisom optuna scipy
   ```
4. Run the pipeline steps sequentially

**Estimated runtime on Kaggle T4:**

| Step | Time | GPU Required? |
|------|------|--------------|
| RCS computation (Step 2) | < 1 min | No |
| Causal validation (Step 2 + `--run_causal`) | 30-60 min | Yes |
| SOM training (Step 3) | 2-5 min | No |
| BO search (Step 4, 128 trials) | 1-3 hours | Yes |
| Ablation + eval (Step 5-6) | 10-15 min | Yes |

---

## Statistical Methods


| Metric | Function | Purpose |
|--------|----------|---------|
| RCS with 95% CI | `compute_rcs_statistics()` | Per-layer significance |
| Spearman ρ with 95% CI | `bootstrap_spearman_ci()` | Causal claim robustness |
| Cohen's d | `compute_rcs_statistics()` | Effect size per layer |
| Gini coefficient | `compute_rcs_statistics()` | RCS concentration measure |
| n80 (layers for 80% RCS) | `compute_rcs_statistics()` | Distribution concentration |

---

## Key Arguments Reference

### `lama_rcs.py`
| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | required | Model alias (e.g., `llama2-7b`) |
| `--n_groups` | `3` | Number of layer groups |
| `--grouping_method` | `quantile` | `quantile` / `peaks` / `adaptive` |
| `--run_causal` | `False` | Run causal validation (GPU) |
| `--device` | `cuda:0` | GPU device |

### `lama_generate_directions.py`
| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | required | Model alias |
| `--som_x` / `--som_y` | `4` | SOM grid dimensions |
| `--sigma` | `0.33` | SOM neighborhood radius |
| `--lr` | `0.01` | SOM learning rate |
| `--iterations` | `10000` | SOM training iterations |
| `--n_groups` | `3` | Number of layer groups |

### `lama_optuna_search.py`
| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | required | Model alias |
| `--directions_dir` | required | Directory with group directions |
| `--directions_prefix` | required | Prefix for direction files |
| `--trials` | `128` | Number of BO trials |
| `--dirs_per_group` | `3` | Directions selected per group |
| `--optimize_alpha` | `False` | Also optimize layer weights |
| `--seed` | `42` | Random seed |

### `lama_ablate.py`
| Arg | Default | Description |
|-----|---------|-------------|
| `--model_name` | required | Model alias |
| `--directions_dir` | required | Directory with group directions |
| `--directions_prefix` | required | Prefix for direction files |
| `--ablation_mode` | `bo` | `uniform` / `rcs` / `bo` |
| `--optimization_results` | `None` | Path to BO results JSON (for `bo` mode) |
| `--dir_ids_per_group` | `None` | Manual direction IDs (for `uniform`/`rcs`) |
| `--dataset_name` | `harmbench_test` | Evaluation dataset |
| `--seed` | `42` | Random seed |

---

## Reference


### Baseline (SOM Directions)
```bibtex
@inproceedings{piras_mura_2026som,
  title={SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models},
  author={Piras, Giorgio and Mura, Raffaele and Brau, Fabio and Oneto, Luca and Roli, Fabio and Biggio, Battista},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={39},
  pages={32728--32736},
  year={2026}
}
```

## Acknowledgements

This codebase extends the [SOM refusal directions](https://arxiv.org/pdf/2511.08379) codebase by Piras et al. and was also inspired by the [refusal direction work by Andy Arditi](https://arxiv.org/pdf/2406.11717).
