#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "${CODE_ROOT}")}"
MODEL_PATH="${MODEL_PATH:-${LINGBOT_ROOT}/models/lingbot-va-base}"
DATASET_PATH="${DATASET_PATH:-${LINGBOT_ROOT}/datasets/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${LINGBOT_ROOT}/datasets/robotwin-clean-and-aug-lerobot/empty_emb.pt}"

if [[ -z "${NGPU:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPU="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
  else
    NGPU="$(nvidia-smi -L | wc -l | tr -d ' ')"
  fi
fi
if [[ "${NGPU}" -lt 1 ]]; then
  echo "No visible GPU found." >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NGPU - 1)))"
fi

BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-64}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  denominator=$((BATCH_SIZE * NGPU))
  if (( TARGET_GLOBAL_BATCH % denominator != 0 )); then
    echo "TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} is not divisible by batch_size*NGPU=${denominator}." >&2
    echo "Set GRADIENT_ACCUMULATION_STEPS explicitly or use a compatible GPU count." >&2
    exit 2
  fi
  GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH / denominator))
fi
GLOBAL_BATCH=$((BATCH_SIZE * NGPU * GRADIENT_ACCUMULATION_STEPS))

NUM_STEPS="${NUM_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
if (( NUM_STEPS % SAVE_INTERVAL != 0 )); then
  echo "NUM_STEPS must be divisible by SAVE_INTERVAL." >&2
  exit 2
fi
if [[ -z "${SAVE_STEPS:-}" ]]; then
  SAVE_STEPS="$(seq -s, "${SAVE_INTERVAL}" "${SAVE_INTERVAL}" "${NUM_STEPS}")"
fi

RUN_ID="${RUN_ID:-robotwin_clean_zipbaseline_${NGPU}gpu_b${BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_constant_${NUM_STEPS}steps_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${LINGBOT_ROOT}/train_out/robotwin/${RUN_ID}}"
LOG="${OUT}/train.log"

if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite existing run: ${OUT}" >&2
  exit 2
fi
test -f "${CODE_ROOT}/.venv/bin/activate"
test -s "${MODEL_PATH}/transformer/config.json"
test -d "${DATASET_PATH}"
test -s "${EMPTY_EMB_PATH}"

"${CODE_ROOT}/.venv/bin/python" - "${MODEL_PATH}/transformer/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
if config.get("attn_mode") != "flex":
    raise SystemExit(
        f"Training requires transformer/config.json attn_mode=flex, "
        f"got {config.get('attn_mode')!r}"
    )
print("Training model attn_mode=flex")
PY

mkdir -p "${OUT}/swanlab"
cd "${CODE_ROOT}"

export WAN_VA_MODEL_PATH="${MODEL_PATH}"
export ROBOTWIN_DATASET_PATH="${DATASET_PATH}"
export ROBOTWIN_EMPTY_EMB_PATH="${EMPTY_EMB_PATH}"
export SAVE_ROOT="${OUT}"
export LINGBOT_TRAIN_SAVE_ROOT="${OUT}"
export LINGBOT_TRAIN_NUM_STEPS="${NUM_STEPS}"
export LINGBOT_TRAIN_BATCH_SIZE="${BATCH_SIZE}"
export LINGBOT_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export LINGBOT_MAX_EPISODE_FRAMES="${MAX_EPISODE_FRAMES:-1000000000}"
export LINGBOT_DATASET_INIT_WORKERS="${DATASET_INIT_WORKERS:-64}"
export LINGBOT_TRAIN_LOAD_WORKERS="${TRAIN_LOAD_WORKERS:-16}"
export LINGBOT_SAVE_INTERVAL="${SAVE_INTERVAL}"
export LINGBOT_SAVE_STEPS="${SAVE_STEPS}"
export LINGBOT_GC_INTERVAL="${GC_INTERVAL:-50}"
export LINGBOT_WARMUP_STEPS="${WARMUP_STEPS:-10}"
export LINGBOT_LR_SCHEDULER="${LR_SCHEDULER:-constant}"
export LINGBOT_MIN_LR_RATIO="${MIN_LR_RATIO:-0.0}"
export LINGBOT_ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-offline}"
export LINGBOT_SWANLAB_LOG_DIR="${OUT}/swanlab"
export LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}"
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${LINGBOT_SWANLAB_EXPERIMENT_NAME:-${RUN_ID}}"
export NGPU CUDA_VISIBLE_DEVICES
export MASTER_PORT="${MASTER_PORT:-29531}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'code_root=%s\n' "${CODE_ROOT}"
  printf 'model=%s\n' "${MODEL_PATH}"
  printf 'dataset=%s\n' "${DATASET_PATH}"
  printf 'ngpu=%s\n' "${NGPU}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'batch_size=%s\n' "${BATCH_SIZE}"
  printf 'gradient_accumulation_steps=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
  printf 'global_batch=%s\n' "${GLOBAL_BATCH}"
  printf 'num_steps=%s\n' "${NUM_STEPS}"
  printf 'save_steps=%s\n' "${SAVE_STEPS}"
  printf 'learning_rate=1e-5\n'
  printf 'warmup_steps=%s\n' "${LINGBOT_WARMUP_STEPS}"
  printf 'lr_scheduler=%s\n' "${LINGBOT_LR_SCHEDULER}"
  printf 'max_episode_frames=%s\n' "${LINGBOT_MAX_EPISODE_FRAMES}"
  printf 'swanlab_mode=%s\n' "${LINGBOT_SWANLAB_MODE}"
} > "${OUT}/run_manifest.txt"

printf 'TRAIN_START %s\n' "$(date -Is)" | tee "${LOG}"
set +e
bash script/run_robotwin_train_4gpu.sh 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
set -e
printf 'TRAIN_DONE rc=%s %s\n' "${rc}" "$(date -Is)" | tee -a "${LOG}"
printf '%s\n' "${rc}" > "${OUT}/exit_code"
exit "${rc}"
