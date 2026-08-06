#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to M30 checkpoint_step_15000}"
PSEUDO_JSONL="${PSEUDO_JSONL:-$PART2_ROOT/dual_rft_selected.jsonl}"
NGPU="${NGPU:-4}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-${BATCH_SIZE:-1}}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-64}"
NUM_STEPS="${RFT_NUM_STEPS:-3000}"
SAVE_INTERVAL="${RFT_SAVE_INTERVAL:-1000}"
MASTER_PORT="${MASTER_PORT:-29641}"
WAM_PYTHON="${WAM_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RFT_SELECTION_MODE="${RFT_SELECTION_MODE:-dual}"
REAL_CHUNK_MODE="${REAL_CHUNK_MODE:-full}"
REAL_FRACTION="${REAL_FRACTION:-0.5}"
OUTER_STEP="${RFT_OUTER_STEP:-0}"
SWANLAB_STEP_OFFSET="${RFT_SWANLAB_STEP_OFFSET:-0}"
RUN_ID="${RUN_ID:-robotwin_dual_rft_joint_full_${NUM_STEPS}steps_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$LINGBOT_ROOT/train_out/robotwin/$RUN_ID}"

cd "$PROJECT_ROOT"
test -x "$WAM_PYTHON"
test -s "$STAGE1_CHECKPOINT/transformer/config.json"
test -s "$PSEUDO_JSONL"
test -d "$PREPARED_DATA_ROOT/stage1"
test -s "$PREPARED_DATA_ROOT/stage1/empty_emb.pt"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing run: $OUT" >&2
  exit 2
fi
denominator=$((TRAIN_BATCH_SIZE_PER_GPU * NGPU))
if (( TARGET_GLOBAL_BATCH % denominator != 0 )); then
  echo "TARGET_GLOBAL_BATCH must be divisible by TRAIN_BATCH_SIZE_PER_GPU*NGPU." >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH / denominator))
mkdir -p "$OUT"

export WAN_VA_MODEL_PATH="$STAGE1_CHECKPOINT"
export ROBOTWIN_DATASET_PATH="$PREPARED_DATA_ROOT/stage1"
export ROBOTWIN_EMPTY_EMB_PATH="$PREPARED_DATA_ROOT/stage1/empty_emb.pt"
export LINGBOT_TRAIN_SAVE_ROOT="$OUT"
export LINGBOT_TRAIN_NUM_STEPS="$NUM_STEPS"
export LINGBOT_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE_PER_GPU"
export LINGBOT_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export LINGBOT_MAX_EPISODE_FRAMES="${MAX_EPISODE_FRAMES:-500}"
export LINGBOT_DATASET_INIT_WORKERS="${DATASET_INIT_WORKERS:-64}"
export LINGBOT_TRAIN_LOAD_WORKERS="${TRAIN_LOAD_WORKERS:-16}"
export LINGBOT_SAVE_INTERVAL="$SAVE_INTERVAL"
SAVE_STEPS="$(seq -s, "$SAVE_INTERVAL" "$SAVE_INTERVAL" "$NUM_STEPS" || true)"
if [[ ",$SAVE_STEPS," != *",$NUM_STEPS,"* ]]; then
  SAVE_STEPS="${SAVE_STEPS:+$SAVE_STEPS,}$NUM_STEPS"
fi
export LINGBOT_SAVE_STEPS="$SAVE_STEPS"
export LINGBOT_GC_INTERVAL="${GC_INTERVAL:-50}"
export LINGBOT_WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export LINGBOT_LR_SCHEDULER="${LR_SCHEDULER:-constant}"
export LINGBOT_ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}"
export LINGBOT_SWANLAB_LOG_DIR="$OUT/swanlab"
export LINGBOT_SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$RUN_ID}"
export LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}"
if [[ -z "${SWANLAB_API_KEY:-}" && -s "$PROJECT_ROOT/.secrets/swanlab_api_key" ]]; then
  export SWANLAB_API_KEY="$(tr -d '[:space:]' < "$PROJECT_ROOT/.secrets/swanlab_api_key")"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

{
  echo "run_id=$RUN_ID"
  echo "stage1_checkpoint=$STAGE1_CHECKPOINT"
  echo "real_dataset=$PREPARED_DATA_ROOT/stage1"
  echo "pseudo_jsonl=$PSEUDO_JSONL"
  echo "selection_mode=$RFT_SELECTION_MODE"
  echo "real_source_update_ratio=$REAL_FRACTION"
  echo "objective=joint_video_action_flow_matching"
  echo "trainable_scope=full_transformer"
  echo "real_chunk_mode=$REAL_CHUNK_MODE"
  echo "train_batch_size_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU"
  echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  echo "num_steps=$NUM_STEPS"
  echo "global_batch=$TARGET_GLOBAL_BATCH"
  echo "outer_step=$OUTER_STEP"
  echo "swanlab_step_offset=$SWANLAB_STEP_OFFSET"
  echo "swanlab_run_id=${LINGBOT_SWANLAB_RUN_ID:-}"
} > "$OUT/run_manifest.txt"

set +e
"$WAM_PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NGPU" \
  --redirects=3 \
  --tee=3 \
  --master_port "$MASTER_PORT" \
  -m robotwin_critic.two_stage_rft.train_joint_rft \
  --config-name robotwin_train \
  --pseudo-jsonl "$PSEUDO_JSONL" \
  --split-manifest "$PREPARED_DATA_ROOT/split_manifest.json" \
  --expected-selection-mode "$RFT_SELECTION_MODE" \
  --real-fraction "$REAL_FRACTION" \
  --real-chunk-mode "$REAL_CHUNK_MODE" \
  --outer-step "$OUTER_STEP" \
  --swanlab-step-offset "$SWANLAB_STEP_OFFSET" \
  --save-root "$OUT" \
  2>&1 | tee "$OUT/train.log"
rc=${PIPESTATUS[0]}
set -e
if (( rc == 0 )); then
  set +e
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.audit_joint_checkpoint \
    --base-transformer "$STAGE1_CHECKPOINT/transformer" \
    --checkpoint-transformer "$OUT/checkpoints/checkpoint_step_${NUM_STEPS}/transformer" \
    --output "$OUT/checkpoint_audit.json" \
    2>&1 | tee -a "$OUT/train.log"
  audit_rc=${PIPESTATUS[0]}
  set -e
  if (( audit_rc != 0 )); then
    rc=$audit_rc
  fi
fi
echo "$rc" > "$OUT/exit_code"
exit "$rc"
