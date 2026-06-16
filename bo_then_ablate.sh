#!/usr/bin/env bash
# bo_then_ablate.sh <gpu> <model> [trials] — full method pipeline for one model on one GPU:
# BO search -> 3 LAMA ablation modes (+utility) on harmbench_test -> judge ASR.
set -u
GPU=$1; M=$2; TRIALS=${3:-48}
cd /nas/home/jahin/causal-lama
echo "[$M] === BO (trials=$TRIALS, GPU=$GPU) ==="
bash run_bo.sh "$GPU" "$M" "$TRIALS"
if ! ls ./runs/$M/lama_optimizations/*results*.json >/dev/null 2>&1; then
  echo "[$M] BO produced no results JSON; abort"; exit 1
fi
echo "[$M] === ABLATE + EVAL ==="
bash ablate_eval.sh "$GPU" "$M"
echo "[$M] === PIPELINE DONE ==="
