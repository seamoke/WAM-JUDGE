#!/usr/bin/env bash
# Download LingBot-VA checkpoints via ModelScope (China-friendly).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_KEY="${1:-lingbot-va-base}"
OUT_DIR="${2:-${ROOT}/checkpoints/${MODEL_KEY}}"

declare -A MODEL_MAP=(
  ["lingbot-va-base"]="Robbyant/lingbot-va-base"
  ["lingbot-va-posttrain-robotwin"]="Robbyant/lingbot-va-posttrain-robotwin"
  ["lingbot-va-posttrain-libero-long"]="Robbyant/lingbot-va-posttrain-libero-long"
)

declare -A DATASET_MAP=(
  ["robotwin-clean-and-aug-lerobot"]="Robbyant/robotwin-clean-and-aug-lerobot"
  ["libero-long-lerobot"]="Robbyant/libero-long-lerobot"
  ["libero-vla-lerobot"]="HuggingFaceVLA/libero"
)

if [[ -n "${DATASET_MAP[$MODEL_KEY]+x}" ]]; then
  REPO_ID="${DATASET_MAP[$MODEL_KEY]}"
  REPO_TYPE="dataset"
elif [[ -n "${MODEL_MAP[$MODEL_KEY]+x}" ]]; then
  REPO_ID="${MODEL_MAP[$MODEL_KEY]}"
  REPO_TYPE="model"
else
  echo "Unknown key: $MODEL_KEY"
  echo "Models: ${!MODEL_MAP[*]}"
  echo "Datasets: ${!DATASET_MAP[*]}"
  exit 1
fi

mkdir -p "$(dirname "$OUT_DIR")"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

if ! python -c "import modelscope" 2>/dev/null; then
  echo "modelscope not installed. Run: bash script/setup_env_cn.sh"
  exit 1
fi

echo "Downloading ${REPO_TYPE} ${REPO_ID} -> ${OUT_DIR}"
python - <<PY
from modelscope import snapshot_download
snapshot_download(
    "${REPO_ID}",
    cache_dir="${OUT_DIR}",
    local_dir="${OUT_DIR}",
    repo_type="${REPO_TYPE}",
)
print("Saved to: ${OUT_DIR}")
PY

echo ""
echo "Update config, e.g.:"
echo "  wan22_pretrained_model_name_or_path = \"${OUT_DIR}\""
