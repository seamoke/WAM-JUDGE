#!/usr/bin/bash
# Launch RoboTwin post-training (robotwin_train config).
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1091
source .venv/bin/activate

NGPU=${NGPU:-4}
MASTER_PORT=${MASTER_PORT:-29512}
LOG_RANK=${LOG_RANK:-0}
SAVE_ROOT=${SAVE_ROOT:-"${ROOT}/train_out/robotwin"}
CONFIG_NAME=robotwin_train

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# SwanLab: load API key from env or local secret file (not committed)
set +x
if [[ -z "${SWANLAB_API_KEY:-}" && -f "${ROOT}/.secrets/swanlab_api_key" ]]; then
  export SWANLAB_API_KEY="$(tr -d '[:space:]' < "${ROOT}/.secrets/swanlab_api_key")"
fi
set -x
export LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}"
export LINGBOT_SWANLAB_WORKSPACE="${LINGBOT_SWANLAB_WORKSPACE:-seamoke}"
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${LINGBOT_SWANLAB_EXPERIMENT_NAME:-robotwin-train}"

overrides=""
if [[ $# -gt 0 ]]; then
  overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --redirects=3 \
  --tee=3 \
  --master_port "${MASTER_PORT}" \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --save-root "${SAVE_ROOT}" \
  ${overrides}
