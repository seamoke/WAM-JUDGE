#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$CODE_ROOT")}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to checkpoint_step_15000}"
MIXED_DATA_ROOT="${MIXED_DATA_ROOT:?Set MIXED_DATA_ROOT to an immutable mixed view}"
MIXED_MANIFEST="$MIXED_DATA_ROOT/mixed_view_manifest.json"

cd "$CODE_ROOT"
test -s "$STAGE1_CHECKPOINT/transformer/config.json"
test -s "$STAGE1_CHECKPOINT/transformer/diffusion_pytorch_model.safetensors"
test -s "$MIXED_MANIFEST"
test -s "$MIXED_DATA_ROOT/empty_emb.pt"

"$CODE_ROOT/.venv/bin/python" \
  -m robotwin_critic.two_stage_rft.validate_rft_dataset \
  --dataset-root "$MIXED_DATA_ROOT" \
  --loader-only \
  --max-items "${VALIDATE_ITEMS:-8}"

export MODEL_PATH="$STAGE1_CHECKPOINT"
export DATASET_PATH="$MIXED_DATA_ROOT"
export EMPTY_EMB_PATH="$MIXED_DATA_ROOT/empty_emb.pt"
export ROBOTWIN_TRAIN_STAGE="stage2_chunk_rft"
export TARGET_GLOBAL_BATCH=64
export NUM_STEPS="${RFT_NUM_STEPS:-3000}"
export SAVE_INTERVAL="${RFT_SAVE_INTERVAL:-1000}"
if [[ -n "${RFT_SAVE_STEPS:-}" ]]; then
  SAVE_STEPS="$RFT_SAVE_STEPS"
else
  SAVE_STEPS="$(seq -s, "$SAVE_INTERVAL" "$SAVE_INTERVAL" "$NUM_STEPS")"
fi
export SAVE_STEPS
export LR_SCHEDULER="${RFT_LR_SCHEDULER:-constant}"
export WARMUP_STEPS="${RFT_WARMUP_STEPS:-10}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="robotwin_stage2_chunk_rft_global64_${NUM_STEPS}steps_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ID

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf 'RFT_TRAIN_VALIDATE_ONLY_OK\n'
  printf 'model=%s\n' "$MODEL_PATH"
  printf 'dataset=%s\n' "$DATASET_PATH"
  printf 'num_steps=%s\n' "$NUM_STEPS"
  printf 'save_steps=%s\n' "$SAVE_STEPS"
  exit 0
fi

exec bash "$CODE_ROOT/script/run_robotwin_clean_train_portable.sh"
