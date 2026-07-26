#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va
mkdir -p train_out/critic/robotwin/logs

tag="${CACHE_TAG:-$(date +%Y%m%d_%H%M%S)}"
train_jsonl="train_out/critic/robotwin/process_pairs_train_cached_clean_${tag}.jsonl"
val_jsonl="train_out/critic/robotwin/process_pairs_val_cached_clean_${tag}.jsonl"
cache_root="train_out/critic/robotwin/feature_cache/process_clean_${tag}"
log="train_out/critic/robotwin/logs/build_feature_cache_clean_${tag}.log"
pidfile="train_out/critic/robotwin/logs/build_feature_cache_clean.pid"
envfile="train_out/critic/robotwin/logs/clean_feature_cache_latest.env"

cat > "$envfile" <<ENV
CACHE_TAG=$tag
TRAIN_JSONL=$train_jsonl
VAL_JSONL=$val_jsonl
CACHE_ROOT=$cache_root
LOG=$log
ENV

nohup bash -lc "
set -euo pipefail
cd /workspace/lingbot-va
echo CLEAN_CACHE_START \$(date)
python -m robotwin_critic.build_feature_cache \
  --input-jsonl train_out/critic/robotwin/process_pairs_train.jsonl \
  --output-jsonl '$train_jsonl' \
  --cache-root '$cache_root' \
  --float16 \
  --verbose
python -m robotwin_critic.build_feature_cache \
  --input-jsonl train_out/critic/robotwin/process_pairs_val.jsonl \
  --output-jsonl '$val_jsonl' \
  --cache-root '$cache_root' \
  --float16 \
  --verbose
echo CLEAN_CACHE_DONE \$(date)
" > "$log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$pidfile"
echo "pid=$pid"
echo "log=$log"
echo "env=$envfile"
sleep 1
tail -n 20 "$log" || true
