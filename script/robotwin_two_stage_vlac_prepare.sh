#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-and-aug-lerobot}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
CRITIC_ROOT="${CRITIC_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin}"
VENV_DIR="${VENV_DIR:-$CRITIC_ROOT/envs/vlac}"
MODEL_PATH="${MODEL_PATH:-$CRITIC_ROOT/models/VLAC-2B}"
MODEL_REPO="${MODEL_REPO:-InternRobotics/VLAC}"
PAIR_ROOT="${PAIR_ROOT:-$CRITIC_ROOT/vlac_finetune/two_stage_${MODE}}"
INDEX_PATH="${INDEX_PATH:-$CRITIC_ROOT/vlac_finetune/two_stage_${MODE}_index.jsonl}"

case "$MODE" in
  smoke)
    MAX_TASKS="${MAX_TASKS:-2}"
    EPISODES_PER_TASK="${EPISODES_PER_TASK:-4}"
    GROUPS_PER_EPISODE="${GROUPS_PER_EPISODE:-2}"
    ;;
  full)
    MAX_TASKS=0
    EPISODES_PER_TASK=0
    GROUPS_PER_EPISODE="${GROUPS_PER_EPISODE:-8}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
if [[ ! -f "$PREPARED_DATA_ROOT/PREPARATION_COMPLETE.json" ]]; then
  python3 script/prepare_robotwin_two_stage_dataset.py \
    --source-root "$SOURCE_DATA_ROOT" \
    --output-root "$PREPARED_DATA_ROOT" \
    --seed 42 \
    --expected-tasks 50 \
    --per-domain-total 50 \
    --stage1-per-domain 30 \
    --link-mode "${LINK_MODE:-hardlink}" \
    --allow-missing-latent-segments "${ALLOW_MISSING_LATENTS:-8}"
else
  python3 script/prepare_robotwin_two_stage_dataset.py \
    --output-root "$PREPARED_DATA_ROOT" \
    --allow-missing-latent-segments "${ALLOW_MISSING_LATENTS:-8}" \
    --verify-only
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
if ! "$VENV_DIR/bin/python" -c \
  "import swift, peft, transformers, cv2, huggingface_hub" 2>/dev/null; then
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install \
    "ms-swift>=3.5,<4" \
    "peft>=0.17,<1" \
    "transformers>=4.51,<4.52" \
    "opencv-python-headless>=4.9" \
    "huggingface-hub>=0.30"
fi

if [[ ! -s "$MODEL_PATH/config.json" ]]; then
  "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.download_model \
    --repo-id "$MODEL_REPO" \
    --output-dir "$MODEL_PATH"
fi

"$VENV_DIR/bin/python" -m robotwin_critic.two_stage_rft.build_vlac_index \
  --prepared-root "$PREPARED_DATA_ROOT" \
  --output "$INDEX_PATH" \
  --max-tasks "$MAX_TASKS"

"$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.build_pairs \
  --index "$INDEX_PATH" \
  --output-dir "$PAIR_ROOT" \
  --max-tasks "$MAX_TASKS" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --groups-per-episode "$GROUPS_PER_EPISODE" \
  --workers "${PAIR_WORKERS:-8}"

"$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.validate_dataset \
  --data-dir "$PAIR_ROOT"

if [[ "$MODE" == "full" && "${USE_ALL_PROTOCOL_PAIRS:-1}" == "1" ]]; then
  "$VENV_DIR/bin/python" -m robotwin_critic.two_stage_rft.use_all_vlac_pairs \
    --data-dir "$PAIR_ROOT"
fi

echo "VLAC two-stage preparation complete"
echo "  model: $MODEL_PATH"
echo "  index: $INDEX_PATH"
echo "  pairs: $PAIR_ROOT"
