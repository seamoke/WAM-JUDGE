#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
RFT_DATA_ROOT="${RFT_DATA_ROOT:?Set RFT_DATA_ROOT to the selected chunk dataset}"
MIXED_DATA_ROOT="${MIXED_DATA_ROOT:?Set MIXED_DATA_ROOT to a new output path}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"
test -x "$PYTHON"
"$PYTHON" -m robotwin_critic.two_stage_rft.build_mixed_view \
  --stage1-root "$PREPARED_DATA_ROOT/stage1" \
  --rft-root "$RFT_DATA_ROOT" \
  --output-root "$MIXED_DATA_ROOT" \
  --rft-target-fraction "${RFT_TARGET_FRACTION:-0.25}"

"$PYTHON" -m robotwin_critic.two_stage_rft.validate_rft_dataset \
  --dataset-root "$MIXED_DATA_ROOT" \
  --loader-only \
  --max-items "${VALIDATE_ITEMS:-8}"
