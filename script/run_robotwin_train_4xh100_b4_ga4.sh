#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
RUN_ID="${RUN_ID:-robotwin_clean_aug_4xh100_b4_ga4_20260723_0320}"
OUT="${ROOT}/train_out/robotwin/${RUN_ID}"
LOG="${OUT}/train.log"

mkdir -p "${OUT}"
cd "${ROOT}/code"

export WAN_VA_MODEL_PATH="${ROOT}/models/lingbot-va-base"
export ROBOTWIN_DATASET_PATH="${ROOT}/datasets/robotwin-clean-and-aug-lerobot"
export SAVE_ROOT="${OUT}"
export LINGBOT_TRAIN_SAVE_ROOT="${OUT}"

export LINGBOT_TRAIN_NUM_STEPS=21000
export LINGBOT_TRAIN_BATCH_SIZE=4
export LINGBOT_GRADIENT_ACCUMULATION_STEPS=4
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING=1
export LINGBOT_MAX_EPISODE_FRAMES=500

export LINGBOT_DATASET_INIT_WORKERS=16
export LINGBOT_TRAIN_LOAD_WORKERS=8
export LINGBOT_SAVE_INTERVAL=3000
export LINGBOT_GC_INTERVAL=1000
export LINGBOT_WARMUP_STEPS=1000
export LINGBOT_ENABLE_SWANLAB=0

export NGPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${MASTER_PORT:-29527}"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

{
  printf 'RUN_ID=%s\n' "${RUN_ID}"
  printf 'START_UTC=%s\n' "$(date -u +'%F %T UTC')"
  printf 'HOST=%s\n' "$(hostname)"
  printf 'MODEL=%s\n' "${WAN_VA_MODEL_PATH}"
  printf 'DATASET=%s\n' "${ROBOTWIN_DATASET_PATH}"
  printf 'NGPU=%s\n' "${NGPU}"
  printf 'BATCH_SIZE=%s\n' "${LINGBOT_TRAIN_BATCH_SIZE}"
  printf 'GRADIENT_ACCUMULATION_STEPS=%s\n' "${LINGBOT_GRADIENT_ACCUMULATION_STEPS}"
  printf 'GLOBAL_BATCH=%s\n' "$((LINGBOT_TRAIN_BATCH_SIZE * NGPU * LINGBOT_GRADIENT_ACCUMULATION_STEPS))"
  printf 'NUM_STEPS=%s\n' "${LINGBOT_TRAIN_NUM_STEPS}"
  printf 'ACTIVATION_CHECKPOINTING=%s\n' "${LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING}"
  printf 'MAX_EPISODE_FRAMES=%s\n' "${LINGBOT_MAX_EPISODE_FRAMES}"
  printf 'SELECTION_REASON=%s\n' 'batch8 reached 78659 MiB on a long-sequence batch; batch4 preserves global batch 64 with safer headroom'
} > "${OUT}/run_manifest.txt"

printf 'TRAIN_START %s\n' "$(date -u +'%F %T UTC')" | tee "${LOG}"
set +e
bash script/run_robotwin_train_4gpu.sh 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e
printf 'TRAIN_DONE rc=%s %s\n' "${rc}" "$(date -u +'%F %T UTC')" | tee -a "${LOG}"
printf '%s\n' "${rc}" > "${OUT}/exit_code"
exit "${rc}"
