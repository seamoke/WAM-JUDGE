#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
RUN_ID="${RUN_ID:-robotwin_clean_zipbaseline_4xh100_b1_ga16_constant_10000steps_ckpt2000_swanlab_20260726_1655}"
OUT="${ROOT}/train_out/robotwin/${RUN_ID}"
LOG="${OUT}/train.log"

if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing run: ${OUT}" >&2
  exit 2
fi

mkdir -p "${OUT}/swanlab"
cd "${ROOT}/code"

export WAN_VA_MODEL_PATH="${ROOT}/models/lingbot-va-base"
export ROBOTWIN_DATASET_PATH="${ROOT}/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
export ROBOTWIN_EMPTY_EMB_PATH="${ROOT}/datasets/robotwin-clean-and-aug-lerobot/empty_emb.pt"
export SAVE_ROOT="${OUT}"
export LINGBOT_TRAIN_SAVE_ROOT="${OUT}"

# Match the reference ZIP training semantics.
export LINGBOT_TRAIN_NUM_STEPS=10000
export LINGBOT_TRAIN_BATCH_SIZE=1
export LINGBOT_GRADIENT_ACCUMULATION_STEPS=16
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING=1
# The current configurable loader expects an integer; this effectively disables
# the non-reference 500-frame filter while retaining its local-path fixes.
export LINGBOT_MAX_EPISODE_FRAMES=1000000000

export LINGBOT_DATASET_INIT_WORKERS=128
export LINGBOT_TRAIN_LOAD_WORKERS=16
export LINGBOT_SAVE_INTERVAL=2000
export LINGBOT_SAVE_STEPS=2000,4000,6000,8000,10000
export LINGBOT_GC_INTERVAL=50
export LINGBOT_WARMUP_STEPS=10
export LINGBOT_LR_SCHEDULER=constant
export LINGBOT_MIN_LR_RATIO=0.0

export LINGBOT_ENABLE_SWANLAB=1
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}"
export LINGBOT_SWANLAB_LOG_DIR="${OUT}/swanlab"
export LINGBOT_SWANLAB_PROJECT=lingbot-va-robotwin
export LINGBOT_SWANLAB_WORKSPACE=seamoke
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${RUN_ID}"

export NGPU=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT="${MASTER_PORT:-29531}"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

{
  printf 'RUN_ID=%s\n' "${RUN_ID}"
  printf 'REFERENCE_ZIP_COMMIT=%s\n' "46117b9b76a884e7943a72fe06752ce13e656453"
  printf 'START_UTC=%s\n' "$(date -u +'%F %T UTC')"
  printf 'HOST=%s\n' "$(hostname)"
  printf 'MODEL=%s\n' "${WAN_VA_MODEL_PATH}"
  printf 'DATASET=%s\n' "${ROBOTWIN_DATASET_PATH}"
  printf 'EXPECTED_VALID_SEGMENTS=%s\n' "2492"
  printf 'EXPECTED_DISTRIBUTED_SEGMENTS_PER_EPOCH=%s\n' "2492"
  printf 'TARGET_EPOCHS_APPROX=%s\n' "256.82"
  printf 'NGPU=%s\n' "${NGPU}"
  printf 'BATCH_SIZE=%s\n' "${LINGBOT_TRAIN_BATCH_SIZE}"
  printf 'GRADIENT_ACCUMULATION_STEPS=%s\n' "${LINGBOT_GRADIENT_ACCUMULATION_STEPS}"
  printf 'GLOBAL_BATCH=%s\n' "$((LINGBOT_TRAIN_BATCH_SIZE * NGPU * LINGBOT_GRADIENT_ACCUMULATION_STEPS))"
  printf 'NUM_STEPS=%s\n' "${LINGBOT_TRAIN_NUM_STEPS}"
  printf 'SAVE_INTERVAL=%s\n' "${LINGBOT_SAVE_INTERVAL}"
  printf 'SAVE_STEPS=%s\n' "${LINGBOT_SAVE_STEPS}"
  printf 'LR_SCHEDULER=%s\n' "${LINGBOT_LR_SCHEDULER}"
  printf 'LEARNING_RATE=%s\n' "1e-5"
  printf 'WARMUP_STEPS=%s\n' "${LINGBOT_WARMUP_STEPS}"
  printf 'ACTIVATION_CHECKPOINTING=%s\n' "${LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING}"
  printf 'MAX_EPISODE_FRAMES_EFFECTIVE=%s\n' "disabled"
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
