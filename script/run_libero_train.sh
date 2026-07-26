#!/usr/bin/bash
# Launch LIBERO post-training (libero_train config).
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

NGPU=${NGPU:-4}
MASTER_PORT=${MASTER_PORT:-29501}
LOG_RANK=${LOG_RANK:-0}
SAVE_ROOT=${SAVE_ROOT:-"${ROOT}/train_out/libero"}
CONFIG_NAME=libero_train
DATASET_PATH=${DATASET_PATH:-}
EMPTY_EMB_PATH=${EMPTY_EMB_PATH:-}

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# SwanLab: load API key from env or local secret file (not committed)
if [[ -z "${SWANLAB_API_KEY:-}" && -f "${ROOT}/.secrets/swanlab_api_key" ]]; then
  export SWANLAB_API_KEY="$(tr -d '[:space:]' < "${ROOT}/.secrets/swanlab_api_key")"
fi
export LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-libero}"
export LINGBOT_SWANLAB_WORKSPACE="${LINGBOT_SWANLAB_WORKSPACE:-seamoke}"
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${LINGBOT_SWANLAB_EXPERIMENT_NAME:-libero-train}"

overrides=""
if [[ $# -gt 0 ]]; then
  overrides="$*"
fi

if [[ -n "${DATASET_PATH}" ]]; then
  overrides="dataset_path=${DATASET_PATH} ${overrides}"
  if [[ -z "${EMPTY_EMB_PATH}" && -f "${DATASET_PATH}/empty_emb.pt" ]]; then
    EMPTY_EMB_PATH="${DATASET_PATH}/empty_emb.pt"
  fi
fi

if [[ -n "${EMPTY_EMB_PATH}" ]]; then
  overrides="empty_emb_path=${EMPTY_EMB_PATH} ${overrides}"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter="${LOG_RANK}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --save-root "${SAVE_ROOT}" \
  ${overrides}
