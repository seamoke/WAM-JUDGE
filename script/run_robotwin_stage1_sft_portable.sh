#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "${CODE_ROOT}")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-${LINGBOT_ROOT}/datasets/robotwin-clean-aug-two-stage-seed42}"
SPLIT_MANIFEST="${PREPARED_DATA_ROOT}/split_manifest.json"

test -s "${SPLIT_MANIFEST}"
test -s "${PREPARED_DATA_ROOT}/PREPARATION_COMPLETE.json"

"${CODE_ROOT}/.venv/bin/python" \
  "${CODE_ROOT}/script/prepare_robotwin_two_stage_dataset.py" \
  --output-root "${PREPARED_DATA_ROOT}" \
  --allow-missing-latent-segments "${ALLOW_MISSING_LATENT_SEGMENTS:-8}" \
  --verify-only

export DATASET_PATH="${PREPARED_DATA_ROOT}/stage1"
export EMPTY_EMB_PATH="${PREPARED_DATA_ROOT}/stage1/empty_emb.pt"
export ROBOTWIN_SPLIT_MANIFEST="${SPLIT_MANIFEST}"
export ROBOTWIN_TRAIN_STAGE="stage1"
export TARGET_GLOBAL_BATCH=64
export NUM_STEPS=15000
export SAVE_INTERVAL=3000
export SAVE_STEPS="3000,6000,9000,12000,15000"
export LR_SCHEDULER=constant
export WARMUP_STEPS=10
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="robotwin_clean_aug_stage1_sft_global64_15000steps_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ID

exec bash "${CODE_ROOT}/script/run_robotwin_clean_train_portable.sh"
