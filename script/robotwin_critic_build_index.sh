#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m robotwin_critic.build_index \
  --dataset-root "${DATASET_ROOT:-/data/lingbot-va/models/datasets/robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot}" \
  --output "${OUTPUT:-/workspace/lingbot-va/train_out/critic/robotwin/index.jsonl}" \
  ${MAX_TASKS:+--max-tasks "${MAX_TASKS}"} \
  ${MAX_EPISODES_PER_TASK:+--max-episodes-per-task "${MAX_EPISODES_PER_TASK}"}

