#!/usr/bin/bash
# RoboTwin post-training on 4 GPUs (0-3 by default).
# Usage: bash script/run_robotwin_train_4gpu.sh
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NGPU="${NGPU:-4}"
export MASTER_PORT="${MASTER_PORT:-29512}"

mkdir -p "${ROOT}/logs"
bash "${ROOT}/script/run_robotwin_train.sh" "$@"
