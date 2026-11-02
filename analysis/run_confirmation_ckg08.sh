#!/usr/bin/env bash
# Confirmation pipeline on ckg08 (4x Quadro RTX 8000, sm_75).
#
# Step 1: regenerate Qwen2.5-7B representations through the REPO'S OWN unmodified
#         entrypoint with --source processed (260 harmful / 6000 harmless) -- the
#         source the extraction logs show the published runs actually used.
#         Card 3 has ~24 GB free, so no sharding is needed and the repo code path
#         is used verbatim.
# Step 2: EMPIRICAL permutation null on those activations (reviewer qp6c, W1).
# Step 3: alternative-normalization sensitivity (reviewer hgpB, W3).
set -u
REPO=/nas/home/jahin/causal-lama
DEV=${DEV:-cuda:3}
ELOG="$REPO/logs/audit/reps_qwen2-7b_ckg08.log"
CLOG="$REPO/logs/audit/confirmation_ckg08.log"
REPS="$REPO/dataset/representations/qwen2-7b"
cd "$REPO" || exit 1

echo "[$(date '+%F %T')] host=$(hostname) device=$DEV  STEP 1 extraction" > "$CLOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 create_representation_dataset.py --model_name qwen2-7b \
    --device "$DEV" --source processed > "$ELOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] extraction rc=$rc" >> "$CLOG"
if [ $rc -ne 0 ]; then
  echo "[$(date '+%F %T')] EXTRACTION FAILED -- aborting" >> "$CLOG"
  tail -20 "$ELOG" >> "$CLOG"
  exit 2
fi
for f in HFx_train.pt HLx_train.pt; do
  [ -s "$REPS/$f" ] || { echo "[$(date '+%F %T')] MISSING $REPS/$f" >> "$CLOG"; exit 2; }
done
echo "[$(date '+%F %T')] tensors present:" >> "$CLOG"
ls -la "$REPS" >> "$CLOG"

echo "" >> "$CLOG"
echo "=== STEP 2: EMPIRICAL PERMUTATION NULL (200 relabelings) ===" >> "$CLOG"
python3 analysis/permutation_null_empirical.py \
  --model qwen2-7b --reps_dir "$REPS" --n_perm 200 --seed 42 >> "$CLOG" 2>&1
echo "[$(date '+%F %T')] empirical null rc=$?" >> "$CLOG"

echo "" >> "$CLOG"
echo "=== STEP 3: ALTERNATIVE NORMALIZATIONS (hgpB W3) ===" >> "$CLOG"
python3 analysis/verify_alt_normalizations.py \
  --model qwen2-7b --reps_dir "$REPS" >> "$CLOG" 2>&1
echo "[$(date '+%F %T')] alt normalizations rc=$?" >> "$CLOG"

echo "" >> "$CLOG"
echo "[$(date '+%F %T')] CONFIRMATION PIPELINE COMPLETE" >> "$CLOG"
