#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
RUN_ID="${RUN_ID:-robotwin_clean_4xh100_b1_ga32_cosine_10epoch_swanlab_20260724_0053}"
OUT="${ROOT}/train_out/robotwin/${RUN_ID}"
LOG="${OUT}/train.log"

mkdir -p "${OUT}/swanlab"
cd "${ROOT}/code"

export WAN_VA_MODEL_PATH="${ROOT}/models/lingbot-va-base"
export ROBOTWIN_DATASET_PATH="${ROOT}/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
export ROBOTWIN_EMPTY_EMB_PATH="${ROOT}/datasets/robotwin-clean-and-aug-lerobot/empty_emb.pt"
export SAVE_ROOT="${OUT}"
export LINGBOT_TRAIN_SAVE_ROOT="${OUT}"

# 2,411 valid segments are padded to 2,412 by DistributedSampler.
# 10 epochs * 2,412 segments / (4 GPUs * batch 1 * GA 32) = 188.44.
export LINGBOT_TRAIN_NUM_STEPS=189
export LINGBOT_TRAIN_BATCH_SIZE=1
export LINGBOT_GRADIENT_ACCUMULATION_STEPS=32
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING=1
export LINGBOT_MAX_EPISODE_FRAMES=500

export LINGBOT_DATASET_INIT_WORKERS=16
export LINGBOT_TRAIN_LOAD_WORKERS=8
export LINGBOT_SAVE_INTERVAL=189
export LINGBOT_GC_INTERVAL=1000
export LINGBOT_WARMUP_STEPS=19
export LINGBOT_LR_SCHEDULER=cosine
export LINGBOT_MIN_LR_RATIO=0.0

export LINGBOT_ENABLE_SWANLAB=1
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}"
export LINGBOT_SWANLAB_LOG_DIR="${OUT}/swanlab"
export LINGBOT_SWANLAB_PROJECT=lingbot-va-robotwin
export LINGBOT_SWANLAB_WORKSPACE=seamoke
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${RUN_ID}"

export NGPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${MASTER_PORT:-29529}"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

{
  printf 'RUN_ID=%s\n' "${RUN_ID}"
  printf 'START_UTC=%s\n' "$(date -u +'%F %T UTC')"
  printf 'HOST=%s\n' "$(hostname)"
  printf 'MODEL=%s\n' "${WAN_VA_MODEL_PATH}"
  printf 'DATASET=%s\n' "${ROBOTWIN_DATASET_PATH}"
  printf 'VALID_SEGMENTS=%s\n' "2411"
  printf 'DISTRIBUTED_PADDED_SEGMENTS_PER_EPOCH=%s\n' "2412"
  printf 'TARGET_EPOCHS=%s\n' "10"
  printf 'NGPU=%s\n' "${NGPU}"
  printf 'BATCH_SIZE=%s\n' "${LINGBOT_TRAIN_BATCH_SIZE}"
  printf 'GRADIENT_ACCUMULATION_STEPS=%s\n' "${LINGBOT_GRADIENT_ACCUMULATION_STEPS}"
  printf 'GLOBAL_BATCH=%s\n' "$((LINGBOT_TRAIN_BATCH_SIZE * NGPU * LINGBOT_GRADIENT_ACCUMULATION_STEPS))"
  printf 'NUM_STEPS=%s\n' "${LINGBOT_TRAIN_NUM_STEPS}"
  printf 'SAVE_INTERVAL=%s\n' "${LINGBOT_SAVE_INTERVAL}"
  printf 'LR_SCHEDULER=%s\n' "${LINGBOT_LR_SCHEDULER}"
  printf 'WARMUP_STEPS=%s\n' "${LINGBOT_WARMUP_STEPS}"
  printf 'MIN_LR_RATIO=%s\n' "${LINGBOT_MIN_LR_RATIO}"
  printf 'ACTIVATION_CHECKPOINTING=%s\n' "${LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING}"
  printf 'MAX_EPISODE_FRAMES=%s\n' "${LINGBOT_MAX_EPISODE_FRAMES}"
  printf 'SWANLAB_ENABLED=%s\n' "${LINGBOT_ENABLE_SWANLAB}"
  printf 'SWANLAB_MODE=%s\n' "${LINGBOT_SWANLAB_MODE}"
  printf 'SWANLAB_LOG_DIR=%s\n' "${LINGBOT_SWANLAB_LOG_DIR}"
} > "${OUT}/run_manifest.txt"

printf 'TRAIN_START %s\n' "$(date -u +'%F %T UTC')" | tee "${LOG}"
set +e
bash script/run_robotwin_train_4gpu.sh 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e
printf 'TRAIN_DONE rc=%s %s\n' "${rc}" "$(date -u +'%F %T UTC')" | tee -a "${LOG}"
printf '%s\n' "${rc}" > "${OUT}/exit_code"
exit "${rc}"
