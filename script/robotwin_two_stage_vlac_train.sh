#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
CRITIC_ROOT="${CRITIC_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin}"
export DATA_DIR="${DATA_DIR:-$CRITIC_ROOT/vlac_finetune/two_stage_${MODE}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$CRITIC_ROOT/vlac_finetune/two_stage_${MODE}_full_4xh100}"
export MODEL_PATH="${MODEL_PATH:-$CRITIC_ROOT/models/VLAC-2B}"
export PROJECT_ROOT LINGBOT_ROOT CRITIC_ROOT

# The reused trainer is deliberately full-parameter: --train_type full,
# freeze_vit=false, and freeze_aligner=false. No LoRA adapter is involved.
exec "$PROJECT_ROOT/script/robotwin_vlac_train_4xh100.sh" "$MODE"
