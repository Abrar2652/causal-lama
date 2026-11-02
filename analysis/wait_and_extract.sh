#!/usr/bin/env bash
# Wait for a GPU with enough free memory, then extract Qwen2.5-7B representations using the
# SAME source the published runs used (--source processed: 260 harmful + 6000 harmless).
# Bounded: gives up after MAX_WAIT seconds. Launches at most one extraction.
set -u
REPO=/nas/home/jahin/causal-lama
NEED_MIB=${NEED_MIB:-16000}
MAX_WAIT=${MAX_WAIT:-10800}     # 3 hours
INTERVAL=${INTERVAL:-120}
LOG="$REPO/logs/audit/reps_qwen2-7b_processed.log"
WLOG="$REPO/logs/audit/wait_and_extract.log"
cd "$REPO" || exit 1
echo "[$(date '+%F %T')] waiting for a GPU with >= ${NEED_MIB} MiB free (max ${MAX_WAIT}s)" >> "$WLOG"
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' -v need="$NEED_MIB" '{free=$3-$2; if (free>=need) {print $1; exit}}')
  if [ -n "${GPU:-}" ]; then
    echo "[$(date '+%F %T')] GPU $GPU is free; starting extraction" >> "$WLOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 create_representation_dataset.py --model_name qwen2-7b \
        --device "cuda:$GPU" --source processed > "$LOG" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] extraction exited rc=$rc" >> "$WLOG"
    if [ $rc -eq 0 ]; then
      echo "[$(date '+%F %T')] SUCCESS -> dataset/representations/qwen2-7b/" >> "$WLOG"
      exit 0
    fi
    # OOM can happen if another job grabs memory mid-load; keep waiting and retry
    echo "[$(date '+%F %T')] retrying after failure" >> "$WLOG"
  fi
  sleep $INTERVAL
  elapsed=$((elapsed+INTERVAL))
done
echo "[$(date '+%F %T')] GAVE UP after ${MAX_WAIT}s: no GPU freed" >> "$WLOG"
exit 3
