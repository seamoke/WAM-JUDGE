#!/usr/bin/env bash
# Install RoboTwin (if needed) then run full checkpoint evaluation. For tmux/nohup.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LOG_DIR="${ROOT}/logs/robotwin_eval"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "========== RoboTwin eval pipeline start =========="
log "Log: ${LOG_FILE}"

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
[[ -f "${ROOT}/script/.robotwin_eval_env" ]] && source "${ROOT}/script/.robotwin_eval_env"

ROBOTWIN_DIR="${ROBOTWIN_DIR:-${ROOT}/third_party/RoboTwin}"
if [[ ! -d "${ROBOTWIN_DIR}/envs" ]]; then
  log "RoboTwin not ready, running setup_robotwin_eval.sh ..."
  bash "${ROOT}/script/setup_robotwin_eval.sh"
else
  log "RoboTwin already present at ${ROBOTWIN_DIR}, skip setup"
fi

log "Stopping GPU guard / training if you run this manually after training ..."
kill "$(cat "${ROOT}/logs/gpu_occupy.pid" 2>/dev/null)" 2>/dev/null || true
pkill -f gpu_guard.sh 2>/dev/null || true
sleep 2

log "Starting evaluation (4 GPUs, 50 tasks, all checkpoints) ..."
bash "${ROOT}/script/run_robotwin_eval.sh"

log "========== Pipeline finished =========="
log "Results: ${ROOT}/train_out/robotwin/eval_results/demo_clean/results.md"
