#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
CODE="${CODE:-$ROOT/code}"
CRITIC="${CRITIC:-$ROOT/train_out/critic/robotwin}"
LOG="${LOG:-$CRITIC/logs/vlac_delayed_full_pipeline.log}"
LOCK="${LOCK:-/tmp/robotwin-vlac-full-pipeline.lock}"
POLL_SECONDS="${POLL_SECONDS:-300}"
IDLE_CONFIRMATIONS="${IDLE_CONFIRMATIONS:-3}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[vlac-full-pipeline] $(date -Is) another pipeline owns $LOCK"
  exit 0
fi

gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d "[:space:]"
}

wait_for_stable_gpu_idle() {
  local idle_count=0
  local pids
  while (( idle_count < IDLE_CONFIRMATIONS )); do
    pids="$(gpu_compute_pids)"
    if [[ -n "$pids" ]]; then
      idle_count=0
      echo "[vlac-full-pipeline] $(date -Is) GPUs busy; waiting ${POLL_SECONDS}s"
    else
      idle_count=$((idle_count + 1))
      echo "[vlac-full-pipeline] $(date -Is) idle confirmation ${idle_count}/${IDLE_CONFIRMATIONS}"
    fi
    if (( idle_count < IDLE_CONFIRMATIONS )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

echo "[vlac-full-pipeline] $(date -Is) watcher started"
wait_for_stable_gpu_idle

if [[ "$SKIP_PREPARE" == "1" ]]; then
  echo "[vlac-full-pipeline] $(date -Is) reusing verified environment, model, and smoke data"
else
  echo "[vlac-full-pipeline] $(date -Is) preparing server-downloaded VLAC-2B and smoke data"
  BUILD_FULL=0 "$CODE/script/robotwin_vlac_prepare_h100.sh"
fi

wait_for_stable_gpu_idle
echo "[vlac-full-pipeline] $(date -Is) running zero-shot baseline, 4-GPU full-parameter smoke, and validation gates"
"$CODE/script/robotwin_vlac_train_4xh100.sh" smoke

echo "[vlac-full-pipeline] $(date -Is) smoke passed; building full RoboTwin RGB pairs"
PATH="$CRITIC/envs/vlac/bin:$PATH" \
PROJECT_ROOT="$CODE" \
INDEX="$CRITIC/index_rgb.jsonl" \
OUTPUT_DIR="$CRITIC/vlac_finetune/full" \
  nice -n 10 "$CODE/robotwin_critic/vlac_finetune/build_full_data.sh"

wait_for_stable_gpu_idle
echo "[vlac-full-pipeline] $(date -Is) full data ready; starting 4-GPU VLAC-2B full-parameter training"
"$CODE/script/robotwin_vlac_train_4xh100.sh" full
echo "[vlac-full-pipeline] $(date -Is) full training complete"
