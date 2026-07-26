#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m robotwin_critic.eval_process_critic \
  --jsonl "${PROCESS_JSONL:-/workspace/lingbot-va/train_out/critic/robotwin/process_pairs_val.jsonl}" \
  --checkpoint "${PROCESS_CKPT:-/workspace/lingbot-va/train_out/critic/robotwin/process_critic/best.pt}" \
  --device "${DEVICE:-cpu}"

python -m robotwin_critic.eval_consistency_filter \
  --jsonl "${CONSISTENCY_JSONL:-/workspace/lingbot-va/train_out/critic/robotwin/consistency_pairs_val.jsonl}" \
  --checkpoint "${CONSISTENCY_CKPT:-/workspace/lingbot-va/train_out/critic/robotwin/consistency_filter/best.pt}" \
  --device "${DEVICE:-cpu}"

