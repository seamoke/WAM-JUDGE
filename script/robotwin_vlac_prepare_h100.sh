#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
DATASET_ROOT="${DATASET_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot}"
CRITIC_ROOT="${CRITIC_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin}"
BASE_PYTHON="${BASE_PYTHON:-$LINGBOT_ROOT/envs/lingbot-va-py310/bin/python}"
VENV_DIR="${VENV_DIR:-$CRITIC_ROOT/envs/vlac}"
MODEL_DIR="${MODEL_DIR:-$CRITIC_ROOT/models/VLAC-2B}"
REPO_ID="${REPO_ID:-InternRobotics/VLAC}"
HF_ENDPOINT_FALLBACK="${HF_ENDPOINT_FALLBACK:-https://hf-mirror.com}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYPI_INDEX_FALLBACK="${PYPI_INDEX_FALLBACK:-https://pypi.org/simple}"
INDEX="${INDEX:-$CRITIC_ROOT/index_rgb.jsonl}"
SMOKE_DIR="${SMOKE_DIR:-$CRITIC_ROOT/vlac_finetune/smoke_2task}"
FULL_DIR="${FULL_DIR:-$CRITIC_ROOT/vlac_finetune/full}"
BUILD_FULL="${BUILD_FULL:-1}"
INSTALL_ONLY="${INSTALL_ONLY:-0}"

mkdir -p "$CRITIC_ROOT/envs" "$CRITIC_ROOT/models" "$CRITIC_ROOT/logs"
cd "$PROJECT_ROOT"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "Base Python is missing: $BASE_PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "RoboTwin dataset is missing: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
export PATH="$VENV_DIR/bin:$PATH"

pip_install() {
  if ! PIP_INDEX_URL="$PYPI_INDEX_URL" \
    "$VENV_DIR/bin/python" -m pip install "$@"; then
    echo "pip failed through $PYPI_INDEX_URL; retrying through $PYPI_INDEX_FALLBACK" >&2
    PIP_INDEX_URL="$PYPI_INDEX_FALLBACK" \
      "$VENV_DIR/bin/python" -m pip install "$@"
  fi
}

pip_install --upgrade pip setuptools wheel
pip_install \
  "ms-swift==3.3.0" \
  "datasets==3.6.0" \
  "transformers>=4.51,<4.56" \
  "peft>=0.15.2,<0.16" \
  "huggingface_hub>=0.34,<1.0" \
  "loguru>=0.7,<1.0" \
  "sentencepiece==0.1.99" \
  "timm>=1.0,<2.0"

if [[ "$INSTALL_ONLY" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="" "$VENV_DIR/bin/python" -c \
    'import importlib.metadata as m; print({p: m.version(p) for p in ("ms-swift", "datasets", "transformers", "peft", "huggingface_hub", "loguru", "sentencepiece", "timm")})'
  exit 0
fi

if ! "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.download_model \
  --repo-id "$REPO_ID" \
  --output-dir "$MODEL_DIR"; then
  echo "Primary Hugging Face download failed; retrying through $HF_ENDPOINT_FALLBACK" >&2
  HF_ENDPOINT="$HF_ENDPOINT_FALLBACK" \
    "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.download_model \
      --repo-id "$REPO_ID" \
      --output-dir "$MODEL_DIR"
fi
[[ -s "$MODEL_DIR/config.json" ]] || {
  echo "Downloaded VLAC snapshot has no config.json: $MODEL_DIR" >&2
  exit 1
}
MODEL_WEIGHT="$(
  find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +1G -print -quit
)"
[[ -n "$MODEL_WEIGHT" ]] || {
  echo "Downloaded VLAC snapshot has no complete safetensors weight over 1 GiB" >&2
  exit 1
}

"$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.build_rgb_index \
  --dataset-root "$DATASET_ROOT" \
  --output "$INDEX"

PROJECT_ROOT="$PROJECT_ROOT" \
INDEX="$INDEX" \
OUTPUT_DIR="$SMOKE_DIR" \
  "$PROJECT_ROOT/robotwin_critic/vlac_finetune/build_smoke_data.sh"

if [[ "$BUILD_FULL" == "1" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT" \
  INDEX="$INDEX" \
  OUTPUT_DIR="$FULL_DIR" \
    nice -n 10 "$PROJECT_ROOT/robotwin_critic/vlac_finetune/build_full_data.sh"
fi

"$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.check_environment \
  --model-2b "$MODEL_DIR"
