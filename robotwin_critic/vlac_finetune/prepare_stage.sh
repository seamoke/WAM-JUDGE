#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/lingbot-va}"
CRITIC_ROOT="${CRITIC_ROOT:-$PROJECT_ROOT/train_out/critic/robotwin}"
VENV_DIR="${VENV_DIR:-$CRITIC_ROOT/envs/vlac}"
MODEL_DIR="${MODEL_DIR:-$CRITIC_ROOT/models/VLAC-2B}"
REPO_ID="${REPO_ID:-InternRobotics/VLAC}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$CRITIC_ROOT/envs" "$CRITIC_ROOT/models"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install \
  "ms-swift==3.8.3" \
  "peft>=0.17,<0.19" \
  "huggingface_hub>=0.34,<1.0"

"$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.download_model \
  --repo-id "$REPO_ID" \
  --output-dir "$MODEL_DIR"

PROJECT_ROOT="$PROJECT_ROOT" \
  "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.check_environment \
  --model-2b "$MODEL_DIR"
