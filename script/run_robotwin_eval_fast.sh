#!/usr/bin/env bash
# Quick RoboTwin checkpoint eval smoke test — for tmux/nohup after training.
#
# Settings: 10 rollouts/task, first checkpoint only, demo_clean (easy).
#
# Usage:
#   bash script/run_robotwin_eval_fast.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LOG_DIR="${ROOT}/logs/robotwin_eval"
LOG_FILE="${LOG_DIR}/fast_eval_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "========== RoboTwin fast eval start =========="
log "Log: ${LOG_FILE}"

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
[[ -f "${ROOT}/script/.robotwin_eval_env" ]] && source "${ROOT}/script/.robotwin_eval_env"

log "Stopping GPU guard if active ..."
kill "$(cat "${ROOT}/logs/gpu_occupy.pid" 2>/dev/null)" 2>/dev/null || true
pkill -f gpu_guard.sh 2>/dev/null || true
sleep 2

export TEST_NUM=10
export MAX_CHECKPOINTS=1
export SKIP_EXISTING=0
export TASK_CONFIG=demo_clean
export ROBOTWIN_FAST=1

log "TEST_NUM=${TEST_NUM}"
log "MAX_CHECKPOINTS=${MAX_CHECKPOINTS}"
log "TASK_CONFIG=${TASK_CONFIG}"
log "ROBOTWIN_FAST=${ROBOTWIN_FAST}"

bash "${ROOT}/script/run_robotwin_eval.sh"

log "========== Fast eval finished =========="
log "Results: ${ROOT}/train_out/robotwin/eval_results/demo_clean/results.md"
