#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va
mkdir -p train_out/critic/robotwin/logs

ts="$(date +%Y%m%d_%H%M%S)"
log="train_out/critic/robotwin/logs/wait_then_build_feature_cache_${ts}.log"
pidfile="train_out/critic/robotwin/logs/wait_then_build_feature_cache.pid"

nohup script/robotwin_critic_wait_then_build_feature_cache.sh > "$log" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$pidfile"
echo "$log" > "train_out/critic/robotwin/logs/wait_then_build_feature_cache.logpath"

echo "pid=$pid"
echo "log=$log"
sleep 1
tail -n 20 "$log" || true
