#!/usr/bin/env bash
# Every 30 minutes, if GPUs are idle, start gpu_occupy.py on all cards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OCCUPY_PY="${SCRIPT_DIR}/gpu_occupy.py"
LOG_DIR="${ROOT}/logs"
PID_FILE="${LOG_DIR}/gpu_occupy.pid"
LOG_FILE="${LOG_DIR}/gpu_guard.log"
OCCUPY_LOG="${LOG_DIR}/gpu_occupy.log"

WATCH_INTERVAL="${GPU_GUARD_INTERVAL:-60}"   # how often to check/restart (seconds)
UTIL_THRESHOLD="${GPU_GUARD_UTIL:-5}"        # % GPU utilization
MEM_THRESHOLD="${GPU_GUARD_MEM_MB:-500}"       # MiB used
MEMORY_FRACTION="${GPU_GUARD_MEM_FRAC:-0.90}"

PYTHON="${GPU_GUARD_PYTHON:-python3}"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

occupy_pid() {
  pgrep -f "${OCCUPY_PY}" 2>/dev/null | head -1
}

is_occupy_running() {
  local pid
  pid="$(occupy_pid)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid" >"$PID_FILE"
    return 0
  fi
  rm -f "$PID_FILE"
  return 1
}

all_gpus_idle() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found"
    return 1
  fi

  local line util mem
  while IFS=',' read -r util mem; do
    util="$(echo "$util" | tr -d ' ')"
    mem="$(echo "$mem" | tr -d ' ')"
    if [[ "$util" -gt "$UTIL_THRESHOLD" ]] || [[ "$mem" -gt "$MEM_THRESHOLD" ]]; then
      return 1
    fi
  done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)

  return 0
}

start_occupy() {
  if is_occupy_running; then
    log "gpu_occupy already running (pid $(cat "$PID_FILE"))"
    return 0
  fi

  log "GPUs idle — starting gpu_occupy on all cards"
  nohup "$PYTHON" "$OCCUPY_PY" --memory-fraction "$MEMORY_FRACTION" \
    >>"$OCCUPY_LOG" 2>&1 &
  disown
  sleep 2
  local pid
  pid="$(occupy_pid)"
  if [[ -z "$pid" ]]; then
    log "ERROR: gpu_occupy failed to start, see $OCCUPY_LOG"
    return 1
  fi
  echo "$pid" >"$PID_FILE"
  log "gpu_occupy started, pid=$pid"
}

check_and_occupy() {
  if is_occupy_running; then
    log "gpu_occupy running (pid $(cat "$PID_FILE")), skip check"
    return 0
  fi

  if all_gpus_idle; then
    start_occupy
  else
    log "GPUs in use, skip occupy"
  fi
}

log "gpu_guard started (watch=${WATCH_INTERVAL}s, util<=${UTIL_THRESHOLD}%, mem<=${MEM_THRESHOLD}MiB)"

# Occupy runs forever until killed; guard re-starts it if it dies.
check_and_occupy
while true; do
  sleep "$WATCH_INTERVAL"
  check_and_occupy
done
