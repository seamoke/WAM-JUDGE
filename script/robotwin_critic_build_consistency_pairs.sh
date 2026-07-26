#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m robotwin_critic.build_consistency_pairs \
  --index "${INDEX:-/workspace/lingbot-va/train_out/critic/robotwin/index.jsonl}" \
  --train-output "${TRAIN_OUTPUT:-/workspace/lingbot-va/train_out/critic/robotwin/consistency_pairs_train.jsonl}" \
  --val-output "${VAL_OUTPUT:-/workspace/lingbot-va/train_out/critic/robotwin/consistency_pairs_val.jsonl}" \
  --samples-per-episode "${SAMPLES_PER_EPISODE:-1}" \
  --horizon "${HORIZON:-32}" \
  --val-fraction "${VAL_FRACTION:-0.05}" \
  --seed "${SEED:-42}" \
  ${MAX_TASKS:+--max-tasks "${MAX_TASKS}"} \
  ${MAX_EPISODES_PER_TASK:+--max-episodes-per-task "${MAX_EPISODES_PER_TASK}"}
