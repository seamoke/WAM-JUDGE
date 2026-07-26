#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va
source train_out/critic/robotwin/logs/clean_feature_cache_latest.env
mkdir -p train_out/critic/robotwin/logs

log="train_out/critic/robotwin/logs/train_cached_rank_only_clean_${CACHE_TAG}.log"
pidfile="train_out/critic/robotwin/logs/train_cached_rank_only_clean.pid"
out_dir="train_out/critic/robotwin/process_critic_cached_v2_rank_only_clean_${CACHE_TAG}"

nohup bash -lc "
set -euo pipefail
cd /workspace/lingbot-va
echo CACHED_RANK_ONLY_START \$(date)
CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES:-0}\" python -m robotwin_critic.train_process_critic_cached_v2 \
  --train-jsonl '$TRAIN_JSONL' \
  --val-jsonl '$VAL_JSONL' \
  --output-dir '$out_dir' \
  --device cuda \
  --batch-size 1024 \
  --num-workers 2 \
  --max-steps 3000 \
  --eval-interval 100 \
  --lr 1e-4 \
  --hidden-dim 512 \
  --progress-weight 1.0 \
  --target-scale 2.0 \
  --best-metric spearman \
  --drop-neutral \
  --init-checkpoint train_out/critic/robotwin/process_critic/last.pt
echo CACHED_RANK_ONLY_DONE \$(date)
" > "$log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$pidfile"
echo "pid=$pid"
echo "log=$log"
echo "out_dir=$out_dir"
sleep 1
tail -n 20 "$log" || true
