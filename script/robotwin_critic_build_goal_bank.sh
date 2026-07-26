#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m robotwin_critic.build_goal_bank \
  --index "${INDEX:-/workspace/lingbot-va/train_out/critic/robotwin/index.jsonl}" \
  --output "${OUTPUT:-/workspace/lingbot-va/train_out/critic/robotwin/goal_bank.pt}" \
  --max-goals-per-task "${MAX_GOALS_PER_TASK:-64}"

