#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$CODE_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-redacted-seed42}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot}"
VISIBLE_DATA_ROOT="${VISIBLE_DATA_ROOT:-$PREPARED_DATA_ROOT/action_visible_real}"
MODEL_PATH="${MODEL_PATH:-$LINGBOT_ROOT/models/lingbot-va-base}"

SPLIT_MANIFEST="$PREPARED_DATA_ROOT/split_manifest.json"
test -x "$CODE_ROOT/.venv/bin/python"
test -s "$SPLIT_MANIFEST"
test -s "$PREPARED_DATA_ROOT/PREPARATION_COMPLETE.json"
test -d "$SOURCE_DATA_ROOT"
test -s "$MODEL_PATH/transformer/config.json"
cd "$CODE_ROOT"

if [[ ! -s "$VISIBLE_DATA_ROOT/ACTION_VISIBLE_COMPLETE.json" ]]; then
  "$CODE_ROOT/.venv/bin/python" \
    -m robotwin_critic.two_stage_rft.prepare_action_visible_real \
    --prepared-root "$PREPARED_DATA_ROOT" \
    --source-root "$SOURCE_DATA_ROOT" \
    --output-root "$VISIBLE_DATA_ROOT" \
    --link-mode "${LINK_MODE:-hardlink}" \
    --allow-missing-latent-segments "${ALLOW_MISSING_LATENT_SEGMENTS:-8}"
fi

"$CODE_ROOT/.venv/bin/python" \
  -m robotwin_critic.two_stage_rft.prepare_action_visible_real \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --output-root "$VISIBLE_DATA_ROOT" \
  --verify-only

export DATASET_PATH="$VISIBLE_DATA_ROOT"
export EMPTY_EMB_PATH="$VISIBLE_DATA_ROOT/empty_emb.pt"
export ROBOTWIN_SPLIT_MANIFEST="$SPLIT_MANIFEST"
export ROBOTWIN_TRAIN_STAGE=stage1-stage2-base50
export MODEL_PATH
export TARGET_GLOBAL_BATCH=64
export NUM_STEPS="${NUM_STEPS:-15000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
export SAVE_STEPS="${SAVE_STEPS:-3000,6000,9000,12000,15000}"
export LR_SCHEDULER="${LR_SCHEDULER:-constant}"
export WARMUP_STEPS="${WARMUP_STEPS:-10}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}"
export LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}"
export NGPU=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="robotwin_clean_aug_base50_sft_global64_15000steps_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ID
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${LINGBOT_SWANLAB_EXPERIMENT_NAME:-$RUN_ID}"

if [[ -z "${SWANLAB_API_KEY:-}" && -s "$CODE_ROOT/.secrets/swanlab_api_key" ]]; then
  export SWANLAB_API_KEY="$(tr -d '[:space:]' < "$CODE_ROOT/.secrets/swanlab_api_key")"
fi

echo "BASE50_SFT_START $(date -Is)"
echo "dataset=$VISIBLE_DATA_ROOT"
echo "model=$MODEL_PATH"
echo "run_id=$RUN_ID"
exec bash "$CODE_ROOT/script/run_robotwin_clean_train_portable.sh"
