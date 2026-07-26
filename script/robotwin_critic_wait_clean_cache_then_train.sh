#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va
mkdir -p train_out/critic/robotwin/logs

cache_pidfile="train_out/critic/robotwin/logs/build_feature_cache_clean.pid"
train_launcher="script/robotwin_critic_start_clean_cached_rank_only.sh"

echo "WAIT_CLEAN_CACHE_START $(date)"
echo "cache_pidfile=$cache_pidfile"

if [[ ! -f "$cache_pidfile" ]]; then
  echo "missing cache pidfile: $cache_pidfile" >&2
  exit 1
fi

cache_pid="$(cat "$cache_pidfile")"
echo "cache_pid=$cache_pid"
ps -p "$cache_pid" -o pid,stat,etime,%cpu,%mem,cmd || true

while kill -0 "$cache_pid" 2>/dev/null; do
  echo "WAIT $(date)"
  ps -p "$cache_pid" -o pid,stat,etime,%cpu,%mem,cmd || true
  if [[ -f train_out/critic/robotwin/logs/clean_feature_cache_latest.env ]]; then
    source train_out/critic/robotwin/logs/clean_feature_cache_latest.env
    wc -l "$TRAIN_JSONL" "$VAL_JSONL" 2>/dev/null || true
  fi
  sleep "${WAIT_SECONDS:-120}"
done

echo "CLEAN_CACHE_PROCESS_DONE $(date)"
if [[ -f train_out/critic/robotwin/logs/clean_feature_cache_latest.env ]]; then
  source train_out/critic/robotwin/logs/clean_feature_cache_latest.env
  if ! grep -q "CLEAN_CACHE_DONE" "$LOG"; then
    echo "cache process ended but CLEAN_CACHE_DONE marker missing; not starting training" >&2
    tail -n 80 "$LOG" >&2 || true
    exit 1
  fi
  wc -l "$TRAIN_JSONL" "$VAL_JSONL"
fi

bash "$train_launcher"
echo "WAIT_CLEAN_CACHE_TRAIN_LAUNCHED $(date)"
