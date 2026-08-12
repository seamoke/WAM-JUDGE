#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to M30 checkpoint_step_15000}"
PSEUDO_JSONL="${PSEUDO_JSONL:-$PART2_ROOT/dual_rft_selected.jsonl}"
LOCAL_NGPU="${LOCAL_NGPU:-${NGPU:-8}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-${BATCH_SIZE:-64}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-512}"
NUM_STEPS="${RFT_NUM_STEPS:-3000}"
SAVE_INTERVAL="${RFT_SAVE_INTERVAL:-1000}"
MASTER_PORT="${MASTER_PORT:-29641}"
WAM_PYTHON="${WAM_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RFT_SELECTION_MODE="${RFT_SELECTION_MODE:-dual}"
REAL_CHUNK_MODE="${REAL_CHUNK_MODE:-full}"
REAL_FRACTION="${REAL_FRACTION:-0.7}"
REAL_DATA_FRACTION="${REAL_DATA_FRACTION:-1.0}"
DATA_FRACTION_SEED="${DATA_FRACTION_SEED:-42}"
MIXING_MODE="${MIXING_MODE:-ratio}"
NUM_EPOCHS="${RFT_NUM_EPOCHS:-0}"
REAL_DATA_MODE="${REAL_DATA_MODE:-stage1-stage2-visible}"
REAL_DATA_ROOT="${REAL_DATA_ROOT:-}"
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
case "$REAL_DATA_MODE" in
  stage1)
    REAL_DATA_ROOT="${REAL_DATA_ROOT:-$PREPARED_DATA_ROOT/stage1}"
    ;;
  stage1-stage2)
    REAL_DATA_ROOT="${REAL_DATA_ROOT:-$($WAM_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "$PREPARED_DATA_ROOT/split_manifest.json")}"
    ;;
  stage1-stage2-visible)
    REAL_DATA_ROOT="${REAL_DATA_ROOT:-$PREPARED_DATA_ROOT/action_visible_real}"
    test -s "$REAL_DATA_ROOT/ACTION_VISIBLE_COMPLETE.json"
    ;;
  *)
    echo "Unsupported REAL_DATA_MODE: $REAL_DATA_MODE" >&2
    exit 2
    ;;
esac
test -d "$REAL_DATA_ROOT"
if (( NODE_RANK == 0 )); then
  if [[ -e "$OUT" ]]; then
    echo "Refusing to overwrite existing run: $OUT" >&2
    exit 2
  fi
  mkdir -p "$OUT"
else
  deadline=$(( $(date +%s) + 300 ))
  while [[ ! -s "$OUT/run_manifest.txt" ]]; do
    if (( $(date +%s) >= deadline )); then
      echo "Timed out waiting for rank-0 run manifest: $OUT" >&2
      exit 2
    fi
    sleep 1
  done
fi
WORLD_SIZE=$((LOCAL_NGPU * NNODES))
denominator=$((TRAIN_BATCH_SIZE_PER_GPU * WORLD_SIZE))
if (( GRADIENT_ACCUMULATION_STEPS <= 0 )); then
  echo "GRADIENT_ACCUMULATION_STEPS must be positive." >&2
  exit 2
fi
expected_global_batch=$((denominator * GRADIENT_ACCUMULATION_STEPS))
if (( TARGET_GLOBAL_BATCH != expected_global_batch )); then
  echo "TARGET_GLOBAL_BATCH must equal TRAIN_BATCH_SIZE_PER_GPU*WORLD_SIZE*GRADIENT_ACCUMULATION_STEPS ($expected_global_batch)." >&2
  exit 2
fi

export WAN_VA_MODEL_PATH="$STAGE1_CHECKPOINT"
export ROBOTWIN_DATASET_PATH="$REAL_DATA_ROOT"
if [[ -s "$REAL_DATA_ROOT/empty_emb.pt" ]]; then
  export ROBOTWIN_EMPTY_EMB_PATH="$REAL_DATA_ROOT/empty_emb.pt"
else
  export ROBOTWIN_EMPTY_EMB_PATH="$PREPARED_DATA_ROOT/stage1/empty_emb.pt"
fi
export LINGBOT_TRAIN_SAVE_ROOT="$OUT"
export LINGBOT_TRAIN_NUM_STEPS="$NUM_STEPS"
export LINGBOT_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE_PER_GPU"
export LINGBOT_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"
export LINGBOT_ENABLE_ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"
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
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

if (( NODE_RANK == 0 )); then
{
  echo "run_id=$RUN_ID"
  echo "stage1_checkpoint=$STAGE1_CHECKPOINT"
  echo "real_dataset=$REAL_DATA_ROOT"
  echo "real_data_mode=$REAL_DATA_MODE"
  echo "pseudo_jsonl=$PSEUDO_JSONL"
  echo "selection_mode=$RFT_SELECTION_MODE"
  echo "real_source_update_ratio=$REAL_FRACTION"
  echo "real_data_fraction=$REAL_DATA_FRACTION"
  echo "data_fraction_seed=$DATA_FRACTION_SEED"
  echo "mixing_mode=$MIXING_MODE"
  echo "num_epochs=$NUM_EPOCHS"
  echo "objective=joint_video_action_flow_matching"
  echo "trainable_scope=full_transformer"
  echo "real_chunk_mode=$REAL_CHUNK_MODE"
  echo "train_batch_size_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU"
  echo "nnodes=$NNODES"
  echo "local_ngpu=$LOCAL_NGPU"
  echo "world_size=$WORLD_SIZE"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "nccl_net=$NCCL_NET"
  echo "nccl_socket_ifname=$NCCL_SOCKET_IFNAME"
  echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  echo "num_steps=$NUM_STEPS"
  echo "global_batch=$TARGET_GLOBAL_BATCH"
  echo "outer_step=$OUTER_STEP"
  echo "swanlab_step_offset=$SWANLAB_STEP_OFFSET"
  echo "swanlab_run_id=${LINGBOT_SWANLAB_RUN_ID:-}"
} > "$OUT/run_manifest.txt"
fi

if (( NODE_RANK == 0 )); then
  TRAIN_LOG="$OUT/train.log"
else
  TRAIN_LOG="$OUT/train.node_${NODE_RANK}.log"
fi

set +e
"$WAM_PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$LOCAL_NGPU" \
  --master_addr="$MASTER_ADDR" \
  --redirects=3 \
  --tee=3 \
  --master_port "$MASTER_PORT" \
  -m robotwin_critic.two_stage_rft.train_joint_rft \
  --config-name robotwin_train \
  --pseudo-jsonl "$PSEUDO_JSONL" \
  --split-manifest "$PREPARED_DATA_ROOT/split_manifest.json" \
  --expected-selection-mode "$RFT_SELECTION_MODE" \
  --real-fraction "$REAL_FRACTION" \
  --real-data-fraction "$REAL_DATA_FRACTION" \
  --data-fraction-seed "$DATA_FRACTION_SEED" \
  --mixing-mode "$MIXING_MODE" \
  --num-epochs "$NUM_EPOCHS" \
  --real-data-mode "$REAL_DATA_MODE" \
  --real-chunk-mode "$REAL_CHUNK_MODE" \
  --outer-step "$OUTER_STEP" \
  --swanlab-step-offset "$SWANLAB_STEP_OFFSET" \
  --save-root "$OUT" \
  2>&1 | tee "$TRAIN_LOG"
rc=${PIPESTATUS[0]}
set -e
if (( NODE_RANK == 0 && rc == 0 )); then
  AUDIT_STEP="$NUM_STEPS"
  if (( NUM_EPOCHS > 0 )); then
    AUDIT_STEP="$($WAM_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["optimizer_steps"])' "$OUT/rft_dataset_report.json")"
  fi
  set +e
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.audit_joint_checkpoint \
    --base-transformer "$STAGE1_CHECKPOINT/transformer" \
    --checkpoint-transformer "$OUT/checkpoints/checkpoint_step_${AUDIT_STEP}/transformer" \
    --output "$OUT/checkpoint_audit.json" \
    2>&1 | tee -a "$OUT/train.log"
  audit_rc=${PIPESTATUS[0]}
  set -e
  if (( audit_rc != 0 )); then
    rc=$audit_rc
  fi
fi
if (( NODE_RANK == 0 )); then
  echo "$rc" > "$OUT/exit_code"
else
  echo "$rc" > "$OUT/node_${NODE_RANK}_exit_code"
fi
exit "$rc"
