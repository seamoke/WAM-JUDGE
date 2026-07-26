#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m robotwin_critic.train_process_critic_spatial_v2 \
  --train-jsonl train_out/critic/robotwin/process_pairs_train_spatial_cached.jsonl \
  --val-jsonl train_out/critic/robotwin/process_pairs_val_spatial_cached.jsonl \
  --output-dir train_out/critic/robotwin/process_critic_spatial_v2_rank_only \
  --device cuda \
  --batch-size 512 \
  --num-workers 2 \
  --max-steps 3000 \
  --eval-interval 100 \
  --lr 1e-4 \
  --hidden-dim 512 \
  --progress-weight 1.0 \
  --target-scale 2.0 \
  --best-metric spearman \
  --drop-neutral
