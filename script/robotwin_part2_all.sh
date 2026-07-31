#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PART2_RUN_ID="${PART2_RUN_ID:-robotwin_part2_$(date +%Y%m%d_%H%M%S)}"
PART2_LOG="${PART2_LOG:-$LINGBOT_ROOT/train_out/logs/${PART2_RUN_ID}.log}"
mkdir -p "$(dirname "$PART2_LOG")"
: > "$PART2_LOG"
set -o pipefail

{
  echo "PART2_START $(date -Is)"
  echo "project_root=$PROJECT_ROOT"
  echo "part2_log=$PART2_LOG"
  nvidia-smi
  bash "$PROJECT_ROOT/script/robotwin_part2_prepare.sh"
  bash "$PROJECT_ROOT/script/robotwin_part2_generate_and_select.sh"
  bash "$PROJECT_ROOT/script/run_robotwin_action_only_rft.sh"
  nvidia-smi
  echo "PART2_DONE $(date -Is)"
} 2>&1 | tee -a "$PART2_LOG"

echo "Unified Part 2 log: $PART2_LOG"
