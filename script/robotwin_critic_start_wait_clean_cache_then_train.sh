#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va
mkdir -p train_out/critic/robotwin/logs

tag="$(date +%Y%m%d_%H%M%S)"
log="train_out/critic/robotwin/logs/wait_clean_cache_then_train_${tag}.log"
pidfile="train_out/critic/robotwin/logs/wait_clean_cache_then_train.pid"

nohup script/robotwin_critic_wait_clean_cache_then_train.sh > "$log" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$pidfile"
echo "$log" > train_out/critic/robotwin/logs/wait_clean_cache_then_train.logpath

echo "pid=$pid"
echo "log=$log"
sleep 1
tail -n 30 "$log" || true
