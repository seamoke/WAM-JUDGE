#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m robotwin_critic.train_process_critic \
  --train-jsonl "${TRAIN_JSONL:-/workspace/lingbot-va/train_out/critic/robotwin/process_pairs_train.jsonl}" \
  --val-jsonl "${VAL_JSONL:-/workspace/lingbot-va/train_out/critic/robotwin/process_pairs_val.jsonl}" \
  --output-dir "${OUTPUT_DIR:-/workspace/lingbot-va/train_out/critic/robotwin/process_critic}" \
  --device "${DEVICE:-cpu}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --max-steps "${MAX_STEPS:-1000}" \
  --eval-interval "${EVAL_INTERVAL:-100}" \
  --hidden-dim "${HIDDEN_DIM:-512}"

