#!/usr/bin/env bash
# Install LIBERO (if needed) then run full checkpoint evaluation. For tmux/nohup.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LOG_DIR="${ROOT}/logs/libero_eval"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "========== LIBERO eval pipeline start =========="
log "Log: ${LOG_FILE}"

# shellcheck disable=SC1091
source .venv/bin/activate
export MUJOCO_GL=osmesa
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
[[ -f "${ROOT}/script/.libero_eval_env" ]] && source "${ROOT}/script/.libero_eval_env"

if ! python -c "from libero.libero import benchmark" 2>/dev/null; then
  log "LIBERO not ready, running setup_libero_eval.sh ..."
  bash "${ROOT}/script/setup_libero_eval.sh"
else
  log "LIBERO already installed, skip setup"
fi

python -c "from libero.libero import benchmark; print('benchmarks:', sorted(benchmark.get_benchmark_dict().keys()))"

log "Stopping GPU guard if active ..."
kill "$(cat "${ROOT}/logs/gpu_occupy.pid" 2>/dev/null)" 2>/dev/null || true
pkill -f gpu_guard.sh 2>/dev/null || true
sleep 2

log "Starting evaluation (4 GPUs, 4 benchmarks, all checkpoints) ..."
bash "${ROOT}/script/run_libero_eval.sh"

log "========== Pipeline finished =========="
log "Results: ${ROOT}/train_out/libero/eval_results/results.md"
