#!/usr/bin/env bash
# =============================================================================
# run_pilot_llama2.sh — End-to-end Causal-LAMA pilot on llama2-7b.
#
# Runs the make-or-break causal-validation experiment + the full ablation study
# with the reviewer-proofing additions (non-circularity controls, real DeltaASR,
# multi-seed ASR with CIs, utility/perplexity cost).
#
# DESIGNED FOR AN INTERMITTENT SHARED CLUSTER: each STAGE is independently
# re-runnable and skips if its output already exists. Pick a free GPU and go.
#
# Usage:
#   bash run_pilot_llama2.sh <gpu_id> [stage]
#   stage ∈ {reps, rcs, dirs, bo, ablate, eval, utility, all}  (default: all)
#
# Train/val/test discipline (anti-leakage):
#   - representations + RCS + SOM directions  : *_train.pt
#   - BO search objective                     : harmbench_val
#   - causal validation DeltaASR              : harmbench_val
#   - FINAL ablation + ASR + utility          : harmbench_test  (touched ONCE)
# =============================================================================
set -euo pipefail

GPU="${1:?Usage: bash run_pilot_llama2.sh <gpu_id> [stage]}"
STAGE="${2:-all}"
MODEL="llama2-7b"
export CUDA_VISIBLE_DEVICES="$GPU"
DEV="cuda:0"   # after masking, the chosen GPU is index 0

PY="${PYTHON:-python3}"
PREFIX="lama_som4x4_sigma0.33_groups3"
DIRS_DIR="./runs/${MODEL}/generate_directions"
OPT_DIR="./runs/${MODEL}/lama_optimizations"
COMP_DIR="./runs/${MODEL}/lama_completions"
EVAL_DIR="./runs/${MODEL}/lama_evaluations"
SEEDS=(42 1 2)   # multi-seed for the final ablation table (mean ± std + CI)

run () { echo; echo ">>> $*"; "$@"; }
have () { [ -e "$1" ]; }

# --- STAGE 1: representations (all layers) -----------------------------------
if [[ "$STAGE" == "reps" || "$STAGE" == "all" ]]; then
  if have "dataset/representations/${MODEL}/HFx_train.pt"; then
    echo "[skip] representations already exist"
  else
    run $PY create_representation_dataset.py --model_name "$MODEL" --device "$DEV"
  fi
fi

# --- STAGE 2: RCS + causal validation + non-circularity controls (KEY) --------
if [[ "$STAGE" == "rcs" || "$STAGE" == "all" ]]; then
  run $PY lama_rcs.py --model_name "$MODEL" --n_groups 3 \
      --run_causal --run_robustness --device "$DEV" \
      --dataset_name harmbench_val --save_path logs
  echo "    -> inspect logs/rcs_analysis_${MODEL}.json :"
  echo "       correlation.spearman_rho (>=0.7 target),"
  echo "       robustness.partial_rcs_given_norm (the anti-circularity number)."
fi

# --- STAGE 3: per-group SOM directions (per seed) ----------------------------
if [[ "$STAGE" == "dirs" || "$STAGE" == "all" ]]; then
  for S in "${SEEDS[@]}"; do
    run $PY lama_generate_directions.py --model_name "$MODEL" \
        --n_groups 3 --som_x 4 --som_y 4 --sigma 0.33 --lr 0.01 \
        --iterations 10000
  done
fi

# --- STAGE 4: Bayesian optimization on VAL (per seed) ------------------------
if [[ "$STAGE" == "bo" || "$STAGE" == "all" ]]; then
  for S in "${SEEDS[@]}"; do
    run $PY lama_optuna_search.py --model_name "$MODEL" \
        --directions_dir "$DIRS_DIR" --directions_prefix "$PREFIX" \
        --trials 128 --dirs_per_group 3 --optimize_alpha --seed "$S"
  done
fi

# --- STAGE 5+6: ablate on TEST (3 modes) + ASR eval + utility, per seed -------
# NOTE: provide the BO-selected dir ids for rcs/uniform from the BO results so
# all three modes share the SAME directions and differ ONLY in alpha (clean
# isolation of the adaptive-weighting effect).
if [[ "$STAGE" == "ablate" || "$STAGE" == "eval" || "$STAGE" == "utility" || "$STAGE" == "all" ]]; then
  for S in "${SEEDS[@]}"; do
    OPT_JSON=$(ls -t ${OPT_DIR}/*results*t128*.json 2>/dev/null | head -1 || true)
    [ -z "${OPT_JSON:-}" ] && { echo "ERROR: no BO results in ${OPT_DIR}; run stage bo"; exit 1; }
    # derive shared dir ids per group from the BO json
    DIRIDS=$($PY - "$OPT_JSON" <<'PYEOF'
import json,sys
r=json.load(open(sys.argv[1]))
d=r.get("best_dir_indices_per_group",{})
print(";".join(f"{g}:{','.join(map(str,v))}" for g,v in sorted(d.items(),key=lambda x:int(x[0]))))
PYEOF
)
    echo "    seed=$S  shared dir ids: $DIRIDS  (from $OPT_JSON)"

    run $PY lama_ablate.py --model_name "$MODEL" --device "$DEV" --seed "$S" \
        --directions_dir "$DIRS_DIR" --directions_prefix "$PREFIX" \
        --ablation_mode bo --optimization_results "$OPT_JSON" \
        --dataset_name harmbench_test --eval_utility

    run $PY lama_ablate.py --model_name "$MODEL" --device "$DEV" --seed "$S" \
        --directions_dir "$DIRS_DIR" --directions_prefix "$PREFIX" \
        --ablation_mode rcs --dir_ids_per_group "$DIRIDS" \
        --dataset_name harmbench_test --eval_utility

    run $PY lama_ablate.py --model_name "$MODEL" --device "$DEV" --seed "$S" \
        --directions_dir "$DIRS_DIR" --directions_prefix "$PREFIX" \
        --ablation_mode uniform --dir_ids_per_group "$DIRIDS" \
        --dataset_name harmbench_test --eval_utility

    # ASR eval for every completions file produced this seed
    mkdir -p "$EVAL_DIR"
    for CJSON in ${COMP_DIR}/*harmbench_test_completions.json; do
      [ -e "$CJSON" ] || continue
      BASE=$(basename "$CJSON" _completions.json)
      EJSON="${EVAL_DIR}/${BASE}_results.json"
      have "$EJSON" || run $PY eval_jailbreaks.py \
          --completions_path "$CJSON" --evaluation_path "$EJSON" \
          --methodologies harmbench --device "$DEV"
    done
  done
fi

echo
echo "=============================================================="
echo " PILOT DONE. Aggregate ASR (mean±std + paired tests) with:"
echo "   python -c \"from lama_stats import compare_methods_from_files as c; \\"
echo "       import json; print(json.dumps(c('<bo_results.json>','<uniform_results.json>'),indent=2))\""
echo " Utility/perplexity per mode: runs/${MODEL}/lama_utility/*_utility.json"
echo "=============================================================="
