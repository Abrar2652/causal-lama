#!/usr/bin/env bash
# ablate_eval.sh <gpu> <model>
# After BO: run the 3 LAMA ablation modes on harmbench_test (sharing the BO-selected
# directions so they differ ONLY in alpha), with utility eval, then judge ASR.
set -u
GPU=$1; M=$2
cd /nas/home/jahin/causal-lama
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PREFIX=lama_som4x4_sigma0.33_groups3
DIRS=./runs/$M/generate_directions
OPT=$(ls -t ./runs/$M/lama_optimizations/*results*.json 2>/dev/null | head -1)
[ -z "$OPT" ] && { echo "[$M] no BO results; abort"; exit 1; }
echo "[$M] using BO results: $OPT"

# shared direction ids per group from the BO json
DIRIDS=$(/usr/bin/python3 - "$OPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); d=r.get("best_dir_indices_per_group",{})
print(";".join(f"{g}:{','.join(map(str,v))}" for g,v in sorted(d.items(),key=lambda x:int(x[0]))))
PY
)
echo "[$M] shared dir ids: $DIRIDS"

run_ablate () {  # mode  extra_args
  /usr/bin/python3 lama_ablate.py --model_name "$M" --device cuda:0 \
    --directions_dir "$DIRS" --directions_prefix "$PREFIX" \
    --dataset_name harmbench_test --eval_utility --ablation_mode "$1" $2
}
run_ablate bo "--optimization_results $OPT"
run_ablate rcs "--dir_ids_per_group $DIRIDS"
run_ablate uniform "--dir_ids_per_group $DIRIDS"

# judge ASR for every completions file produced
mkdir -p ./runs/$M/lama_evaluations
for CJSON in ./runs/$M/lama_completions/*harmbench_test_completions.json; do
  [ -e "$CJSON" ] || continue
  BASE=$(basename "$CJSON" _completions.json)
  EJSON=./runs/$M/lama_evaluations/${BASE}_results.json
  [ -e "$EJSON" ] || /usr/bin/python3 eval_jailbreaks.py --completions_path "$CJSON" \
      --evaluation_path "$EJSON" --methodologies harmbench --device cuda:0
done
echo "[$M] ABLATE+EVAL DONE"
