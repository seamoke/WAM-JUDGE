#!/usr/bin/env bash
set -euo pipefail

: "${MODELSCOPE_API_TOKEN:?Set MODELSCOPE_API_TOKEN in the current shell}"
: "${MODELSCOPE_REPO_ID:?Set MODELSCOPE_REPO_ID to namespace/model-name}"
: "${STAGE1_RUN_DIR:?Set STAGE1_RUN_DIR to the completed Stage-1 run}"

MS_HUB="${MS_HUB:-ms-hub}"
MAX_WORKERS="${MAX_WORKERS:-8}"
CHECKPOINT_ROOT="$STAGE1_RUN_DIR/checkpoints"
CHECKPOINT_12000="$CHECKPOINT_ROOT/checkpoint_step_12000"
CHECKPOINT_15000="$CHECKPOINT_ROOT/checkpoint_step_15000"

command -v "$MS_HUB" >/dev/null

for checkpoint in "$CHECKPOINT_12000" "$CHECKPOINT_15000"; do
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint directory: $checkpoint" >&2
    exit 1
  fi
  if [[ -z "$(find "$checkpoint" -type f -print -quit)" ]]; then
    echo "Checkpoint directory is empty: $checkpoint" >&2
    exit 1
  fi
done

"$MS_HUB" whoami >/dev/null

"$MS_HUB" upload \
  "$MODELSCOPE_REPO_ID" \
  "$CHECKPOINT_12000" \
  checkpoint_step_12000 \
  --repo-type model \
  --max-workers "$MAX_WORKERS" \
  --commit-message "Upload Stage-1 checkpoint 12000"

"$MS_HUB" upload \
  "$MODELSCOPE_REPO_ID" \
  "$CHECKPOINT_15000" \
  checkpoint_step_15000 \
  --repo-type model \
  --max-workers "$MAX_WORKERS" \
  --commit-message "Upload Stage-1 checkpoint 15000"

echo "MODELSCOPE_STAGE1_CHECKPOINT_UPLOAD_OK"
echo "Repository: https://modelscope.cn/models/$MODELSCOPE_REPO_ID"
