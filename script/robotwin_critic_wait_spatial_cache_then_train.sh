#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va

LOG_DIR="train_out/critic/robotwin/logs"
mkdir -p "$LOG_DIR"

TRAIN_SRC="train_out/critic/robotwin/process_pairs_train.jsonl"
VAL_SRC="train_out/critic/robotwin/process_pairs_val.jsonl"
TRAIN_CACHED="train_out/critic/robotwin/process_pairs_train_spatial_cached.jsonl"
VAL_CACHED="train_out/critic/robotwin/process_pairs_val_spatial_cached.jsonl"
OUT_DIR="train_out/critic/robotwin/process_critic_spatial_v2_rank_only"

echo "[wait-spatial] start $(date '+%F %T %Z')"

while true; do
  src_train=$(wc -l < "$TRAIN_SRC")
  src_val=$(wc -l < "$VAL_SRC")
  cached_train=0
  cached_val=0
  [[ -f "$TRAIN_CACHED" ]] && cached_train=$(wc -l < "$TRAIN_CACHED")
  [[ -f "$VAL_CACHED" ]] && cached_val=$(wc -l < "$VAL_CACHED")

  echo "[wait-spatial] $(date '+%F %T %Z') train=${cached_train}/${src_train} val=${cached_val}/${src_val}"

  if [[ "$cached_train" -ge "$src_train" && "$cached_val" -ge "$src_val" ]]; then
    break
  fi

  if ! pgrep -af "robotwin_critic.build_spatial_feature_cache" >/dev/null; then
    if [[ "$cached_train" -gt 0 && "$cached_val" -gt 0 ]]; then
      echo "[wait-spatial] cache builder finished with skipped rows: train=${cached_train}/${src_train} val=${cached_val}/${src_val}"
      break
    fi
    echo "[wait-spatial] cache builder is not running before usable cached files were produced"
    exit 2
  fi

  sleep "${POLL_SECONDS:-60}"
done

if pgrep -af "robotwin_critic.train_process_critic_spatial_v2" >/dev/null; then
  echo "[wait-spatial] spatial training is already running; not starting a duplicate"
  exit 0
fi

if [[ -f "$OUT_DIR/best.pt" || -f "$OUT_DIR/last.pt" ]]; then
  echo "[wait-spatial] spatial checkpoint already exists in $OUT_DIR; not starting a duplicate"
  exit 0
fi

train_log="$LOG_DIR/train_spatial_rank_only_$(date +%Y%m%d_%H%M%S).log"
echo "[wait-spatial] starting training, log=$train_log"
nohup bash script/robotwin_critic_train_spatial_rank_only.sh > "$train_log" 2>&1 < /dev/null &
train_pid=$!
echo "$train_pid" > "$LOG_DIR/train_spatial_rank_only.pid"
echo "$train_log" > "$LOG_DIR/train_spatial_rank_only.logpath"
echo "[wait-spatial] training pid=$train_pid"
