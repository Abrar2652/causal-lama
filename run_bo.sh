#!/usr/bin/env bash
# run_bo.sh <gpu> <model> [trials]  — BO search for one model on one GPU.
set -u
GPU=$1; M=$2; TRIALS=${3:-48}
cd /nas/home/jahin/causal-lama
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/usr/bin/python3 lama_optuna_search.py --model_name "$M" \
  --directions_dir "./runs/$M/generate_directions" \
  --directions_prefix lama_som4x4_sigma0.33_groups3 \
  --trials "$TRIALS" --dirs_per_group 3 --optimize_alpha --seed 42
echo "[$M] BO DONE"
