#!/usr/bin/env bash
# Full official LIBERO eval for checkpoint_step_10000 — for tmux/nohup.
#
# Settings: libero (default), 50 rollouts/task, 800 max steps, 4 GPUs.
#
# Usage:
#   tmux new -s libero_eval
#   bash script/run_libero_eval_step10000.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LOG_DIR="${ROOT}/logs/libero_eval"
LOG_FILE="${LOG_DIR}/step10000_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "========== LIBERO full eval: checkpoint_step_10000 =========="
log "Log: ${LOG_FILE}"

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
[[ -f "${ROOT}/script/.libero_eval_env" ]] && source "${ROOT}/script/.libero_eval_env"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

log "Stopping GPU guard if active ..."
kill "$(cat "${ROOT}/logs/gpu_occupy.pid" 2>/dev/null)" 2>/dev/null || true
pkill -f gpu_guard.sh 2>/dev/null || true
sleep 2

export CHECKPOINT_PATH="${ROOT}/train_out/libero/checkpoints/checkpoint_step_10000"
export EVAL_CONFIG_NAME=libero
export TEST_NUM=50
export MAX_ENV_STEPS=800
export SKIP_EXISTING=0
export LIBERO_RENDER_GL=egl

log "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
log "EVAL_CONFIG_NAME=${EVAL_CONFIG_NAME}"
log "TEST_NUM=${TEST_NUM}"
log "MAX_ENV_STEPS=${MAX_ENV_STEPS}"

bash "${ROOT}/script/run_libero_eval.sh"

log "========== Full eval finished =========="
log "Results: ${ROOT}/train_out/libero/eval_results/results.md"
