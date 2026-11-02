#!/usr/bin/env bash
# Wait for the sharded extraction to finish, then run the two analyses that needed
# the paper's own activations: the EMPIRICAL permutation null (qp6c W1) and the
# alternative-normalization sensitivity check (hgpB W3).
set -u
REPO=/nas/home/jahin/causal-lama
LOG="$REPO/logs/audit/reps_qwen2-7b_sharded.log"
CLOG="$REPO/logs/audit/chain_after_extract.log"
REPS="$REPO/dataset/representations/qwen2-7b_processed_confirm"
MAX=5400
cd "$REPO" || exit 1
el=0
while [ $el -lt $MAX ]; do
  if grep -q "^\[DONE\]" "$LOG" 2>/dev/null; then break; fi
  if grep -qE "^\[FATAL\]|Traceback|CUDA out of memory" "$LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] extraction FAILED; chain aborted" >> "$CLOG"; exit 2
  fi
  sleep 20; el=$((el+20))
done
if ! grep -q "^\[DONE\]" "$LOG" 2>/dev/null; then
  echo "[$(date '+%F %T')] timed out waiting for extraction" >> "$CLOG"; exit 3
fi
echo "[$(date '+%F %T')] extraction done; tensors at $REPS" >> "$CLOG"

echo "=== EMPIRICAL PERMUTATION NULL (200 relabelings) ===" >> "$CLOG"
python3 analysis/permutation_null_empirical.py \
  --model qwen2-7b --reps_dir "$REPS" --n_perm 200 --seed 42 >> "$CLOG" 2>&1
echo "[$(date '+%F %T')] empirical null rc=$?" >> "$CLOG"

echo "=== ALTERNATIVE NORMALIZATIONS (hgpB W3) ===" >> "$CLOG"
python3 analysis/verify_alt_normalizations.py \
  --model qwen2-7b --reps_dir "$REPS" >> "$CLOG" 2>&1
echo "[$(date '+%F %T')] alt normalizations rc=$?" >> "$CLOG"

echo "[$(date '+%F %T')] CHAIN COMPLETE" >> "$CLOG"
