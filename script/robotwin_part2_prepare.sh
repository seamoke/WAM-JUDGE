#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"
test -x "$PYTHON"
test -s "$PREPARED_DATA_ROOT/split_manifest.json"
test -s "$PREPARED_DATA_ROOT/PREPARATION_COMPLETE.json"
mkdir -p "$PART2_ROOT"

"$PYTHON" script/prepare_robotwin_two_stage_dataset.py \
  --output-root "$PREPARED_DATA_ROOT" \
  --allow-missing-latent-segments "${ALLOW_MISSING_LATENT_SEGMENTS:-8}" \
  --verify-only

"$PYTHON" -m robotwin_critic.two_stage_rft.calibrate_action_critic \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --output "$PART2_ROOT/stage1_kinematic_profile.json" \
  --soft-quantile "${SOFT_QUANTILE:-0.99}" \
  --hard-quantile "${HARD_QUANTILE:-0.999}" \
  --minimum-score "${MIN_ACTION_SCORE:-0.5}"

"$PYTHON" -m robotwin_critic.two_stage_rft.build_video_contexts \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --output "$PART2_ROOT/stage2_video_contexts.jsonl" \
  --history-frames "${HISTORY_FRAMES:-4}" \
  --max-episode-frames "${MAX_EPISODE_FRAMES:-500}" \
  --context-pool-multiplier "${CONTEXT_POOL_MULTIPLIER:-2.0}" \
  ${CONTEXT_LIMIT_ARGS:-}

"$PYTHON" -m robotwin_critic.two_stage_rft.count_pseudo_budget \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --output "$PART2_ROOT/stage2_chunk_budget.json" \
  --max-episode-frames "${MAX_EPISODE_FRAMES:-500}"

echo "ROBOTWIN_PART2_PREPARE_OK"
