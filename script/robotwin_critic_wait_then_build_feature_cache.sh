#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va

echo "START $(date)"
echo "Waiting for current v2 process critic PIDs: ${WAIT_PIDS:-967 968}"
PID_LIST="$(echo "${WAIT_PIDS:-967 968}" | tr ' ' ',')"

for pid in ${WAIT_PIDS:-967 968}; do
  echo "initial pid $pid:"
  ps -p "$pid" -o pid,stat,etime,%cpu,%mem,cmd || true
done

while true; do
  alive=0
  for pid in ${WAIT_PIDS:-967 968}; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  if [[ "$alive" == "0" ]]; then
    break
  fi
  echo "WAIT $(date)"
  ps -p "$PID_LIST" -o pid,stat,etime,%cpu,%mem,cmd || true
  sleep "${WAIT_SECONDS:-60}"
done

echo "V2_DONE $(date)"
bash script/robotwin_critic_build_feature_cache.sh
echo "CACHE_DONE $(date)"
