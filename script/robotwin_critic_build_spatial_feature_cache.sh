#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va

python -m robotwin_critic.build_spatial_feature_cache \
  --input-jsonl train_out/critic/robotwin/process_pairs_train.jsonl \
  --output-jsonl train_out/critic/robotwin/process_pairs_train_spatial_cached.jsonl \
  --cache-root train_out/critic/robotwin/feature_cache/process_spatial \
  --grid 4 \
  --float16 \
  --verbose

python -m robotwin_critic.build_spatial_feature_cache \
  --input-jsonl train_out/critic/robotwin/process_pairs_val.jsonl \
  --output-jsonl train_out/critic/robotwin/process_pairs_val_spatial_cached.jsonl \
  --cache-root train_out/critic/robotwin/feature_cache/process_spatial \
  --grid 4 \
  --float16 \
  --verbose
