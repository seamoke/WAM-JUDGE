#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/action_critic}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
EXTRA_ARGS=()
if [[ -n "${MAX_TRAJECTORIES:-}" ]]; then
  EXTRA_ARGS+=(--max-trajectories "$MAX_TRAJECTORIES")
fi
if [[ -n "${FPS:-}" ]]; then
  EXTRA_ARGS+=(--fps "$FPS")
fi

cd "$PROJECT_ROOT"
test -x "$PYTHON"
"$PYTHON" -m robotwin_critic.two_stage_rft.calibrate_action_critic \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --output "$OUTPUT_ROOT/stage1_profile.json" \
  --soft-quantile "${SOFT_QUANTILE:-0.99}" \
  --hard-quantile "${HARD_QUANTILE:-0.999}" \
  "${EXTRA_ARGS[@]}"
