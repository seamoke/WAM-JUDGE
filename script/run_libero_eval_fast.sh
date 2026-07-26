#!/usr/bin/env bash
# Fast LIBERO checkpoint eval — for tmux/nohup.
#
# Settings: LIBERO_FAST (libero_eval config, no video), skip completed checkpoints.
#
# Usage:
#   tmux new -s libero_eval
#   bash script/run_libero_eval_fast.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LOG_DIR="${ROOT}/logs/libero_eval"
LOG_FILE="${LOG_DIR}/fast_eval_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "========== LIBERO fast eval start =========="
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

export LIBERO_FAST=1
export TEST_NUM=50
export SKIP_EXISTING=1
export MAX_ENV_STEPS=400
export LIBERO_RENDER_GL=egl

log "LIBERO_FAST=${LIBERO_FAST}"
log "TEST_NUM=${TEST_NUM}"
log "SKIP_EXISTING=${SKIP_EXISTING}"
log "MAX_ENV_STEPS=${MAX_ENV_STEPS}"
log "LIBERO_RENDER_GL=${LIBERO_RENDER_GL}"

bash "${ROOT}/script/run_libero_eval.sh"

log "========== Fast eval finished =========="
log "Results: ${ROOT}/train_out/libero/eval_results/results.md"
