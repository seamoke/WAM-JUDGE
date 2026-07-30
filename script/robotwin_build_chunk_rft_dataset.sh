#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
SELECTED_JSONL="${SELECTED_JSONL:?Set SELECTED_JSONL to scored WAM candidates}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new, non-existing RFT dataset path}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"
test -x "$PYTHON"
"$PYTHON" -m robotwin_critic.two_stage_rft.rft_dataset \
  --selected-jsonl "$SELECTED_JSONL" \
  --output-root "$OUTPUT_ROOT" \
  --empty-embedding "$PREPARED_DATA_ROOT/stage1/empty_emb.pt" \
  --min-process-score "${MIN_PROCESS_SCORE:-0.0}" \
  --min-action-score "${MIN_ACTION_SCORE:-0.5}"

"$PYTHON" -m robotwin_critic.two_stage_rft.validate_rft_dataset \
  --dataset-root "$OUTPUT_ROOT" \
  --with-loader \
  --max-items "${VALIDATE_ITEMS:-4}"
