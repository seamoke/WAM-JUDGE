#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/lingbot-va}"
INDEX="${INDEX:-$PROJECT_ROOT/train_out/critic/robotwin/index.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/full}"

cd "$PROJECT_ROOT"
python -m robotwin_critic.vlac_finetune.build_pairs \
  --index "$INDEX" \
  --output-dir "$OUTPUT_DIR" \
  --max-tasks 0 \
  --episodes-per-task 0 \
  --groups-per-episode "${GROUPS_PER_EPISODE:-8}" \
  --eval-frames "${EVAL_FRAMES:-12}" \
  --trainer-val-samples "${TRAINER_VAL_SAMPLES:-1024}" \
  --workers "${DATA_WORKERS:-8}" \
  --seed 42

python -m robotwin_critic.vlac_finetune.validate_dataset \
  --data-dir "$OUTPUT_DIR" \
  --samples 32
