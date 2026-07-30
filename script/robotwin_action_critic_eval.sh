#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/action_critic}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"
test -x "$PYTHON"
"$PYTHON" -m robotwin_critic.two_stage_rft.evaluate_action_critic \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --profile "$OUTPUT_ROOT/stage1_profile.json" \
  --output "$OUTPUT_ROOT/stage2_corruption_metrics.json" \
  --max-segments "${MAX_SEGMENTS:-1000}"
