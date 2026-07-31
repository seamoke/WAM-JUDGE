#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to M30 checkpoint_step_15000}"
PSEUDO_JSONL="${PSEUDO_JSONL:-$PART2_ROOT/dual_rft_selected.jsonl}"
NGPU="${NGPU:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-64}"
NUM_STEPS="${RFT_NUM_STEPS:-3000}"
SAVE_INTERVAL="${RFT_SAVE_INTERVAL:-1000}"
MASTER_PORT="${MASTER_PORT:-29631}"
RUN_ID="${RUN_ID:-robotwin_dual_rft_action_only_${NUM_STEPS}steps_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$LINGBOT_ROOT/train_out/robotwin/$RUN_ID}"

cd "$PROJECT_ROOT"
test -x "$PROJECT_ROOT/.venv/bin/python"
test -s "$STAGE1_CHECKPOINT/transformer/config.json"
test -s "$PSEUDO_JSONL"
test -d "$PREPARED_DATA_ROOT/stage1"
test -s "$PREPARED_DATA_ROOT/stage1/empty_emb.pt"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing run: $OUT" >&2
  exit 2
fi
denominator=$((BATCH_SIZE * NGPU))
if (( TARGET_GLOBAL_BATCH % denominator != 0 )); then
  echo "TARGET_GLOBAL_BATCH must divide batch_size*NGPU exactly." >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH / denominator))
mkdir -p "$OUT"

export WAN_VA_MODEL_PATH="$STAGE1_CHECKPOINT"
export ROBOTWIN_DATASET_PATH="$PREPARED_DATA_ROOT/stage1"
export ROBOTWIN_EMPTY_EMB_PATH="$PREPARED_DATA_ROOT/stage1/empty_emb.pt"
export LINGBOT_TRAIN_SAVE_ROOT="$OUT"
export LINGBOT_TRAIN_NUM_STEPS="$NUM_STEPS"
export LINGBOT_TRAIN_BATCH_SIZE="$BATCH_SIZE"
export LINGBOT_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export LINGBOT_MAX_EPISODE_FRAMES="${MAX_EPISODE_FRAMES:-500}"
export LINGBOT_DATASET_INIT_WORKERS="${DATASET_INIT_WORKERS:-64}"
export LINGBOT_TRAIN_LOAD_WORKERS="${TRAIN_LOAD_WORKERS:-16}"
export LINGBOT_SAVE_INTERVAL="$SAVE_INTERVAL"
export LINGBOT_SAVE_STEPS="$(seq -s, "$SAVE_INTERVAL" "$SAVE_INTERVAL" "$NUM_STEPS")"
export LINGBOT_GC_INTERVAL="${GC_INTERVAL:-50}"
export LINGBOT_WARMUP_STEPS="${WARMUP_STEPS:-10}"
export LINGBOT_LR_SCHEDULER="${LR_SCHEDULER:-constant}"
export LINGBOT_ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-offline}"
export LINGBOT_SWANLAB_LOG_DIR="$OUT/swanlab"
export LINGBOT_SWANLAB_EXPERIMENT_NAME="$RUN_ID"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "run_id=$RUN_ID"
  echo "stage1_checkpoint=$STAGE1_CHECKPOINT"
  echo "real_dataset=$PREPARED_DATA_ROOT/stage1"
  echo "pseudo_jsonl=$PSEUDO_JSONL"
  echo "real_fraction=0.7"
  echo "pseudo_fraction=0.3"
  echo "objective=action_flow_matching_only"
  echo "trainable_modules=action_embedder,condition_embedder_action,action_proj_out"
  echo "num_steps=$NUM_STEPS"
  echo "global_batch=$TARGET_GLOBAL_BATCH"
} > "$OUT/run_manifest.txt"

set +e
"$PROJECT_ROOT/.venv/bin/python" -m torch.distributed.run \
  --nproc_per_node="$NGPU" \
  --redirects=3 \
  --tee=3 \
  --master_port "$MASTER_PORT" \
  -m robotwin_critic.two_stage_rft.train_action_only_rft \
  --config-name robotwin_train \
  --pseudo-jsonl "$PSEUDO_JSONL" \
  --real-fraction 0.7 \
  --save-root "$OUT" \
  2>&1 | tee "$OUT/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "$rc" > "$OUT/exit_code"
exit "$rc"
