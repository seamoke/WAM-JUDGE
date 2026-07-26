#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/lingbot-va}"
MODEL_PATH="${MODEL_PATH:-/data/lingbot-va/models/vlac/VLAC-2B}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/smoke_2task}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/vlac_2b_eval}"
ADAPTER="${ADAPTER:-}"

cd "$PROJECT_ROOT"
args=(
  --model "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "${DEVICE:-cuda:0}"
  --batch-size "${BATCH_SIZE:-4}"
)
if [[ -n "$ADAPTER" ]]; then
  args+=(--adapter "$ADAPTER")
fi
python -m robotwin_critic.vlac_finetune.evaluate_vlac "${args[@]}"

