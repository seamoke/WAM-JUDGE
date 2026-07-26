#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lingbot-va

python -m robotwin_critic.build_feature_cache \
  --input-jsonl train_out/critic/robotwin/process_pairs_train.jsonl \
  --output-jsonl train_out/critic/robotwin/process_pairs_train_cached.jsonl \
  --cache-root train_out/critic/robotwin/feature_cache/process \
  --float16 \
  --verbose

python -m robotwin_critic.build_feature_cache \
  --input-jsonl train_out/critic/robotwin/process_pairs_val.jsonl \
  --output-jsonl train_out/critic/robotwin/process_pairs_val_cached.jsonl \
  --cache-root train_out/critic/robotwin/feature_cache/process \
  --float16 \
  --verbose
