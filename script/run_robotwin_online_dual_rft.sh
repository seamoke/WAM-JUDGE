#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
ONLINE_ROOT="${ONLINE_ROOT:-$PART2_ROOT/online_dual_rft}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
REAL_DATA_ROOT="${REAL_DATA_ROOT:-$PREPARED_DATA_ROOT/action_visible_real}"
REAL_DATA_MODE="${REAL_DATA_MODE:-stage1-stage2-visible}"
INITIAL_MODEL="${INITIAL_MODEL:?Set INITIAL_MODEL to a complete WAM checkpoint root}"
VLAC_MODEL="${VLAC_MODEL:?Set VLAC_MODEL to the trained VLAC checkpoint}"
VLAC_ADAPTER="${VLAC_ADAPTER:-}"
WAM_PYTHON="${WAM_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
VLAC_PYTHON="${VLAC_PYTHON:-$LINGBOT_ROOT/train_out/critic/robotwin/envs/vlac/bin/python}"

# Sampling throughput hyperparameters. With eight GPUs, defaults give 320 Q and
# 320*8=2560 candidates per collection round.
INFER_GPU_IDS="${INFER_GPU_IDS:-0,1,2,3,4,5,6,7}"
REMOTE_INFER_WORKERS="${REMOTE_INFER_WORKERS:-0}"
REMOTE_GPU_IDS="${REMOTE_GPU_IDS:-0,1}"
MULTINODE_QUEUE_ROOT="${MULTINODE_QUEUE_ROOT:-$ONLINE_ROOT/multinode_queue}"
Q_PER_ROUND="${Q_PER_ROUND:-320}"
INFER_BATCH_SIZE_PER_GPU="${INFER_BATCH_SIZE_PER_GPU:-8}"
CANDIDATES_PER_Q="${CANDIDATES_PER_Q:-8}"
VLAC_BATCH_SIZE_PER_GPU="${VLAC_BATCH_SIZE_PER_GPU:-4}"

# Eight-GPU defaults: 64 samples/GPU, no accumulation, global batch 512.
BUFFER_CAPACITY="${BUFFER_CAPACITY:-1024}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-64}"
TRAIN_GLOBAL_BATCH="${TRAIN_GLOBAL_BATCH:-512}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_ACTIVATION_CHECKPOINTING="${TRAIN_ACTIVATION_CHECKPOINTING:-0}"
PSEUDO_EPOCHS_PER_UPDATE="${PSEUDO_EPOCHS_PER_UPDATE:-3}"
REAL_FRACTION="${REAL_FRACTION:-0.7}"
UPDATE_STEPS="${UPDATE_STEPS:-}"
MAX_UPDATES="${MAX_UPDATES:-1000}"
MODEL_SAVE_EVERY_UPDATES="${MODEL_SAVE_EVERY_UPDATES:-25}"
MIN_ACTION_SCORE="${MIN_ACTION_SCORE:-0.5}"
ACTION_GATE_POLICY="${ACTION_GATE_POLICY:-score_with_safety_gates}"
ACTION_WORKSPACE_SCOPE="${ACTION_WORKSPACE_SCOPE:-global}"
MIN_PROCESS_SCORE="${MIN_PROCESS_SCORE:-5.0}"
MAX_PSEUDO_PER_CONTEXT="${MAX_PSEUDO_PER_CONTEXT:-4}"
ONE_SHOT_MODE="${ONE_SHOT_MODE:-0}"
ONE_SHOT_TARGET="${ONE_SHOT_TARGET:-25000}"
ONE_SHOT_DATA_FRACTION="${ONE_SHOT_DATA_FRACTION:-1.0}"
ONE_SHOT_COLLECT_ROOT="${ONE_SHOT_COLLECT_ROOT:-$ONLINE_ROOT/collect}"
ONE_SHOT_TRAIN_EPOCHS="${ONE_SHOT_TRAIN_EPOCHS:-3}"
ONE_SHOT_MAX_PER_EPISODE="${ONE_SHOT_MAX_PER_EPISODE:-16}"
ONE_SHOT_PROGRESS_BINS="${ONE_SHOT_PROGRESS_BINS:-5}"
ONE_SHOT_MIN_ACTION_DISTANCE="${ONE_SHOT_MIN_ACTION_DISTANCE:-0.03}"
ONE_SHOT_MIN_MEAN_LUMA="${ONE_SHOT_MIN_MEAN_LUMA:-8.0}"
ONE_SHOT_MIN_STD_LUMA="${ONE_SHOT_MIN_STD_LUMA:-4.0}"
ONE_SHOT_MAX_DARK_FRACTION="${ONE_SHOT_MAX_DARK_FRACTION:-0.98}"
ONE_SHOT_WARMUP_STEPS="${ONE_SHOT_WARMUP_STEPS:-100}"
BASE_SEED="${BASE_SEED:-42}"
WORKER_MASTER_PORT_BASE="${WORKER_MASTER_PORT_BASE:-29700}"
TRAIN_NNODES="${TRAIN_NNODES:-1}"
TRAIN_LOCAL_NGPU="${TRAIN_LOCAL_NGPU:-}"
TRAIN_MASTER_ADDR="${TRAIN_MASTER_ADDR:-127.0.0.1}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29641}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-script/run_robotwin_joint_rft.sh}"

CONTEXTS="${CONTEXTS:-$PART2_ROOT/stage2_video_contexts.jsonl}"
ACTION_PROFILE="${ACTION_PROFILE:-$PART2_ROOT/stage1_kinematic_profile.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PREPARED_DATA_ROOT/split_manifest.json}"
PSEUDO_BUDGET="${PSEUDO_BUDGET:-$PART2_ROOT/stage2_chunk_budget.json}"
STATE="$ONLINE_ROOT/state.json"
PENDING="$ONLINE_ROOT/pending_buffer.jsonl"
BUFFERS_DIR="$ONLINE_ROOT/buffers"
TASK_RETENTION_SUMMARY="$ONLINE_ROOT/task_qa_retention.json"
SWANLAB_COLLECTION_STATE="$ONLINE_ROOT/swanlab_collection_upload_state.json"
SWANLAB_COLLECTION_RUN_ID_FILE="$ONLINE_ROOT/swanlab_collection_run_id"
SWANLAB_COLLECTION_LOG_DIR="$ONLINE_ROOT/swanlab_collection"
SWANLAB_COLLECTION_IMAGES="${SWANLAB_COLLECTION_IMAGES:-4}"
SWANLAB_COLLECTION_REQUIRED="${SWANLAB_COLLECTION_REQUIRED:-0}"
SWANLAB_PARENT_DRIVER="${SWANLAB_PARENT_DRIVER:-0}"
ONE_SHOT_BUFFER="$ONLINE_ROOT/one_shot_pseudo_buffer.jsonl"
ONE_SHOT_SUMMARY="$ONLINE_ROOT/one_shot_pseudo_buffer.summary.json"
ONE_SHOT_VISUAL_CACHE="$ONLINE_ROOT/one_shot_visual_cache.jsonl"
ONE_SHOT_COMPLETE="$ONLINE_ROOT/one_shot_complete.json"
ONE_SHOT_EFFECTIVE_TARGET="$($WAM_PYTHON -c 'import math,sys; f=float(sys.argv[1]); n=int(sys.argv[2]); assert 0 < f <= 1; print(max(1, math.floor(n*f+0.5)))' "$ONE_SHOT_DATA_FRACTION" "$ONE_SHOT_TARGET")"

IFS=',' read -r -a GPUS <<< "$INFER_GPU_IDS"
IFS=',' read -r -a REMOTE_GPUS <<< "$REMOTE_GPU_IDS"
NGPU="${#GPUS[@]}"
if (( NGPU < 1 )); then
  echo "INFER_GPU_IDS must contain at least one GPU" >&2
  exit 2
fi
TRAIN_LOCAL_NGPU="${TRAIN_LOCAL_NGPU:-$NGPU}"
if (( REMOTE_INFER_WORKERS > 0 && REMOTE_INFER_WORKERS != ${#REMOTE_GPUS[@]} )); then
  echo "REMOTE_INFER_WORKERS must equal the number of REMOTE_GPU_IDS" >&2
  exit 2
fi
if (( TRAIN_LOCAL_NGPU != NGPU )); then
  echo "TRAIN_LOCAL_NGPU must equal the number of local INFER_GPU_IDS" >&2
  exit 2
fi
TOTAL_WORKERS=$((NGPU + REMOTE_INFER_WORKERS))
if (( TOTAL_WORKERS < 1 )); then
  echo "At least one local or remote inference worker is required" >&2
  exit 2
fi
if (( Q_PER_ROUND <= 0 || Q_PER_ROUND % TOTAL_WORKERS != 0 )); then
  echo "Q_PER_ROUND must be positive and divisible by total inference workers" >&2
  exit 2
fi
Q_PER_GPU=$((Q_PER_ROUND / TOTAL_WORKERS))
TRAIN_WORLD_SIZE=$((TRAIN_NNODES * TRAIN_LOCAL_NGPU))
if (( GRADIENT_ACCUMULATION_STEPS <= 0 )); then
  echo "GRADIENT_ACCUMULATION_STEPS must be positive" >&2
  exit 2
fi
EXPECTED_GLOBAL_BATCH=$((TRAIN_BATCH_SIZE_PER_GPU * TRAIN_WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( TRAIN_GLOBAL_BATCH != EXPECTED_GLOBAL_BATCH )); then
  echo "TRAIN_GLOBAL_BATCH must equal TRAIN_BATCH_SIZE_PER_GPU*training_world_size*GRADIENT_ACCUMULATION_STEPS ($EXPECTED_GLOBAL_BATCH)" >&2
  exit 2
fi
if [[ -z "$UPDATE_STEPS" ]]; then
  UPDATE_STEPS="$(awk \
    -v capacity="$BUFFER_CAPACITY" \
    -v epochs="$PSEUDO_EPOCHS_PER_UPDATE" \
    -v batch="$TRAIN_GLOBAL_BATCH" \
    -v real_fraction="$REAL_FRACTION" \
    'BEGIN {
      pseudo_fraction = 1.0 - real_fraction
      if (real_fraction <= 0.0 || real_fraction >= 1.0) exit 2
      exact = capacity * epochs / (batch * pseudo_fraction)
      steps = int(exact)
      if (steps < exact) steps += 1
      print steps
    }'
  )" || {
    echo "REAL_FRACTION must be strictly between 0 and 1" >&2
    exit 2
  }
fi
if (( UPDATE_STEPS <= 0 || MAX_UPDATES <= 0 || MODEL_SAVE_EVERY_UPDATES <= 0 )); then
  echo "UPDATE_STEPS, MAX_UPDATES, and MODEL_SAVE_EVERY_UPDATES must be positive" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
test -x "$WAM_PYTHON"
test -x "$VLAC_PYTHON"
test -s "$INITIAL_MODEL/transformer/config.json"
test -s "$CONTEXTS"
test -s "$ACTION_PROFILE"
test -s "$SPLIT_MANIFEST"
if [[ "$ONE_SHOT_MODE" == "1" ]]; then
  test -s "$PSEUDO_BUDGET"
fi
mkdir -p "$ONLINE_ROOT" "$BUFFERS_DIR" "$ONLINE_ROOT/checkpoints"
SWANLAB_RUN_ID_FILE="$ONLINE_ROOT/swanlab_run_id"
if [[ "$SWANLAB_PARENT_DRIVER" == "1" ]]; then
  SWANLAB_RUN_ID="${SWANLAB_PARENT_RUN_ID:?Parent driver must set SWANLAB_PARENT_RUN_ID}"
  printf '%s\n' "$SWANLAB_RUN_ID" > "$SWANLAB_RUN_ID_FILE"
elif [[ -s "$SWANLAB_RUN_ID_FILE" ]]; then
  SWANLAB_RUN_ID="$(tr -d '[:space:]' < "$SWANLAB_RUN_ID_FILE")"
else
  SWANLAB_RUN_ID="rft-$(date +%Y%m%d%H%M%S)-$$"
  printf '%s\n' "$SWANLAB_RUN_ID" > "$SWANLAB_RUN_ID_FILE"
fi
if [[ "$SWANLAB_PARENT_DRIVER" == "1" ]]; then
  SWANLAB_COLLECTION_RUN_ID="$SWANLAB_RUN_ID"
elif [[ -s "$SWANLAB_COLLECTION_RUN_ID_FILE" ]]; then
  SWANLAB_COLLECTION_RUN_ID="$(tr -d '[:space:]' < "$SWANLAB_COLLECTION_RUN_ID_FILE")"
else
  SWANLAB_COLLECTION_RUN_ID="rft-collect-$(date +%Y%m%d%H%M%S)-$$"
  printf '%s\n' "$SWANLAB_COLLECTION_RUN_ID" > "$SWANLAB_COLLECTION_RUN_ID_FILE"
fi
if [[ -z "${SWANLAB_API_KEY:-}" && -s "$PROJECT_ROOT/.secrets/swanlab_api_key" ]]; then
  export SWANLAB_API_KEY="$(tr -d '[:space:]' < "$PROJECT_ROOT/.secrets/swanlab_api_key")"
fi
exec > >(tee -a "$ONLINE_ROOT/online.log") 2>&1

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "ONLINE_DUAL_RFT_CONFIG local_infer_gpus=$INFER_GPU_IDS remote_infer_workers=$REMOTE_INFER_WORKERS total_infer_workers=$TOTAL_WORKERS q_per_round=$Q_PER_ROUND q_per_gpu=$Q_PER_GPU inference_batch_per_gpu=$INFER_BATCH_SIZE_PER_GPU candidates_per_q=$CANDIDATES_PER_Q buffer_capacity=$BUFFER_CAPACITY pseudo_epochs_per_update=$PSEUDO_EPOCHS_PER_UPDATE real_fraction=$REAL_FRACTION train_nodes=$TRAIN_NNODES train_local_gpus=$TRAIN_LOCAL_NGPU train_world_size=$TRAIN_WORLD_SIZE train_batch_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU train_global_batch=$TRAIN_GLOBAL_BATCH gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS activation_checkpointing=$TRAIN_ACTIVATION_CHECKPOINTING update_steps=$UPDATE_STEPS max_updates=$MAX_UPDATES model_save_every_updates=$MODEL_SAVE_EVERY_UPDATES action_gate_policy=$ACTION_GATE_POLICY action_workspace_scope=$ACTION_WORKSPACE_SCOPE min_action_score=$MIN_ACTION_SCORE min_process_score=$MIN_PROCESS_SCORE max_pseudo_per_context=$MAX_PSEUDO_PER_CONTEXT one_shot_mode=$ONE_SHOT_MODE one_shot_full_target=$ONE_SHOT_TARGET one_shot_data_fraction=$ONE_SHOT_DATA_FRACTION one_shot_effective_target=$ONE_SHOT_EFFECTIVE_TARGET one_shot_collect_root=$ONE_SHOT_COLLECT_ROOT one_shot_epochs=$ONE_SHOT_TRAIN_EPOCHS one_shot_max_per_episode=$ONE_SHOT_MAX_PER_EPISODE one_shot_progress_bins=$ONE_SHOT_PROGRESS_BINS one_shot_min_action_distance=$ONE_SHOT_MIN_ACTION_DISTANCE one_shot_visual_gate=$ONE_SHOT_MIN_MEAN_LUMA,$ONE_SHOT_MIN_STD_LUMA,$ONE_SHOT_MAX_DARK_FRACTION swanlab_run_id=$SWANLAB_RUN_ID"

submit_remote_stage() {
  local stage="$1"
  local collect_index="$2"
  local collect_dir="$3"
  local current_model="$4"
  REMOTE_JOB_ID="${stage}_collect_$(printf '%06d' "$collect_index")_$(date +%s)_$$"
  env \
    STAGE="$stage" \
    COLLECT_INDEX="$collect_index" \
    COLLECT_DIR="$collect_dir" \
    CURRENT_MODEL="$current_model" \
    REMOTE_WORKER_OFFSET="$NGPU" \
    REMOTE_GPU_IDS="$REMOTE_GPU_IDS" \
    WORKER_MASTER_PORT_BASE="$WORKER_MASTER_PORT_BASE" \
    "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.multinode_worker submit \
      --queue-root "$MULTINODE_QUEUE_ROOT" \
      --job-id "$REMOTE_JOB_ID" \
      --cwd "$PROJECT_ROOT" \
      -- bash script/run_robotwin_remote_collect_stage.sh
}

wait_remote_stage() {
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.multinode_worker wait \
    --queue-root "$MULTINODE_QUEUE_ROOT" \
    --job-id "$REMOTE_JOB_ID" \
    --timeout 0
}

log_collection_swanlab() {
  local collect_index="${1:-}"
  if [[ "$SWANLAB_PARENT_DRIVER" == "1" ]]; then
    if [[ -n "$collect_index" ]]; then
      printf 'SWANLAB_COLLECTION_EVENT {"collect_index": %s}\n' "$collect_index"
    fi
    return 0
  fi
  local rc=0
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.log_online_collection_swanlab \
    --online-root "$ONLINE_ROOT" \
    --state "$SWANLAB_COLLECTION_STATE" \
    --run-id "$SWANLAB_COLLECTION_RUN_ID" \
    --project "${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}" \
    --group "${LINGBOT_SWANLAB_GROUP:-robotwin-stage1-real-stage2-pseudo}" \
    --name "${SWANLAB_COLLECTION_NAME:-robotwin-stage1-15000-dual-rft-collection}" \
    --log-dir "$SWANLAB_COLLECTION_LOG_DIR" \
    --max-images-per-collect "$SWANLAB_COLLECTION_IMAGES" || rc=$?
  if (( rc != 0 )); then
    echo "SWANLAB_COLLECTION_LOG_FAILED rc=$rc"
    if [[ "$SWANLAB_COLLECTION_REQUIRED" == "1" ]]; then
      return "$rc"
    fi
  fi
}

"$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration init \
  --state "$STATE" \
  --contexts "$CONTEXTS" \
  --model "$INITIAL_MODEL" \
  --base-seed "$BASE_SEED"

"$WAM_PYTHON" -m robotwin_critic.two_stage_rft.summarize_online_task_retention \
  --collect-root "$ONLINE_ROOT/collect" \
  --output "$TASK_RETENTION_SUMMARY"
log_collection_swanlab

run_ready_update() {
  local ready update_index current_model update_root train_root staged_model
  local completed_update checkpoint_link previous_root previous_name previous_digits previous_update
  ready="$("$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration field --state "$STATE" --name ready_buffer)"
  if [[ -z "$ready" ]]; then
    return 1
  fi
  update_index="$("$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration field --state "$STATE" --name update_index)"
  current_model="$("$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration field --state "$STATE" --name current_model)"
  update_root="$ONLINE_ROOT/updates/update_$(printf '%06d' "$update_index")"
  train_root="$update_root/train"
  staged_model="$update_root/model"
  mkdir -p "$update_root"
  echo "ONLINE_UPDATE_START index=$update_index buffer=$ready model=$current_model"
  if ! env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  LINGBOT_ROOT="$LINGBOT_ROOT" \
  PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
  REAL_DATA_ROOT="$REAL_DATA_ROOT" \
  PART2_ROOT="$PART2_ROOT" \
  STAGE1_CHECKPOINT="$current_model" \
  PSEUDO_JSONL="$ready" \
  LOCAL_NGPU="$TRAIN_LOCAL_NGPU" \
  NNODES="$TRAIN_NNODES" \
  MASTER_ADDR="$TRAIN_MASTER_ADDR" \
  MASTER_PORT="$TRAIN_MASTER_PORT" \
  TRAIN_NNODES="$TRAIN_NNODES" \
  TRAIN_LOCAL_NGPU="$TRAIN_LOCAL_NGPU" \
  TRAIN_MASTER_ADDR="$TRAIN_MASTER_ADDR" \
  TRAIN_MASTER_PORT="$TRAIN_MASTER_PORT" \
  TRAIN_LOCAL_GPU_IDS="$INFER_GPU_IDS" \
  TRAIN_REMOTE_GPU_IDS="$REMOTE_GPU_IDS" \
  MULTINODE_QUEUE_ROOT="$MULTINODE_QUEUE_ROOT" \
  TRAIN_BATCH_SIZE_PER_GPU="$TRAIN_BATCH_SIZE_PER_GPU" \
  TARGET_GLOBAL_BATCH="$TRAIN_GLOBAL_BATCH" \
  RFT_NUM_STEPS="$UPDATE_STEPS" \
  RFT_SAVE_INTERVAL="$UPDATE_STEPS" \
  RFT_OUTER_STEP="$update_index" \
  RFT_SWANLAB_STEP_OFFSET="$((update_index * UPDATE_STEPS))" \
  REAL_FRACTION="$REAL_FRACTION" \
  REAL_DATA_MODE="$REAL_DATA_MODE" \
  REAL_CHUNK_MODE=first-transition \
  ACTIVATION_CHECKPOINTING="$TRAIN_ACTIVATION_CHECKPOINTING" \
  WARMUP_STEPS=1 \
  ENABLE_SWANLAB=1 \
  LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}" \
  LINGBOT_SWANLAB_RUN_ID="$SWANLAB_RUN_ID" \
  LINGBOT_SWANLAB_EXTERNAL="$SWANLAB_PARENT_DRIVER" \
  LINGBOT_SWANLAB_GROUP="${LINGBOT_SWANLAB_GROUP:-robotwin-stage1-real-stage2-pseudo}" \
  LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}" \
  SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-robotwin-stage1-15000-dual-rft-1000}" \
  RUN_ID="online_update_$(printf '%06d' "$update_index")" \
  OUT="$train_root" \
  CUDA_VISIBLE_DEVICES="$INFER_GPU_IDS" \
  bash "$TRAIN_LAUNCHER"; then
    echo "ONLINE_UPDATE_TRAIN_FAILED index=$update_index buffer=$ready" >&2
    return 1
  fi
  if ! "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.stage_updated_model \
    --base-model "$current_model" \
    --transformer "$train_root/checkpoints/checkpoint_step_${UPDATE_STEPS}/transformer" \
    --output "$staged_model" \
    --move-transformer; then
    echo "ONLINE_UPDATE_STAGE_FAILED index=$update_index train_root=$train_root" >&2
    return 1
  fi
  if ! "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration complete-update \
    --state "$STATE" \
    --model "$staged_model"; then
    echo "ONLINE_UPDATE_STATE_FAILED index=$update_index model=$staged_model" >&2
    return 1
  fi

  # The trainer checkpoint is only an intermediate used to assemble the next
  # complete WAM. Keep logs, but do not retain this duplicate transformer copy.
  rm -rf -- "$train_root/checkpoints"

  completed_update=$((update_index + 1))
  if (( completed_update % MODEL_SAVE_EVERY_UPDATES == 0 )); then
    checkpoint_link="$ONLINE_ROOT/checkpoints/rft_update_$(printf '%06d' "$completed_update")"
    ln -sfn "../updates/update_$(printf '%06d' "$update_index")/model" "$checkpoint_link"
    printf '%s\n' "$completed_update" > "$staged_model/rft_checkpoint_update"
    echo "ONLINE_MODEL_CHECKPOINT_SAVED update=$completed_update model=$staged_model link=$checkpoint_link"
  fi

  # Once the new model is active, discard the previous rolling model unless it
  # is a retained 50-update milestone. Initial/external models are never touched.
  previous_root="$(dirname "$current_model")"
  previous_name="$(basename "$previous_root")"
  if [[ "$previous_root" == "$ONLINE_ROOT"/updates/update_* && "$previous_name" == update_* ]]; then
    previous_digits="${previous_name#update_}"
    if [[ "$previous_digits" =~ ^[0-9]+$ ]]; then
      previous_update=$((10#$previous_digits + 1))
      if (( previous_update % MODEL_SAVE_EVERY_UPDATES != 0 )); then
        rm -rf -- "$current_model"
        echo "ONLINE_ROLLING_MODEL_REMOVED update=$previous_update model=$current_model"
      fi
    fi
  fi

  echo "ONLINE_UPDATE_OK index=$update_index completed_update=$completed_update model=$staged_model"
  return 0
}

run_one_shot_training() {
  local train_root final_model actual_steps
  train_root="$ONLINE_ROOT/one_shot_train"
  final_model="$ONLINE_ROOT/final_model"
  if [[ -s "$ONE_SHOT_COMPLETE" ]]; then
    echo "ONE_SHOT_RFT_ALREADY_COMPLETE marker=$ONE_SHOT_COMPLETE"
    return 0
  fi
  if [[ -e "$train_root" || -e "$final_model" ]]; then
    echo "Refusing to overwrite partial one-shot training artifacts under $ONLINE_ROOT" >&2
    return 2
  fi
  echo "ONE_SHOT_RFT_TRAIN_START pseudo=$ONE_SHOT_BUFFER epochs=$ONE_SHOT_TRAIN_EPOCHS model=$INITIAL_MODEL"
  if ! env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  LINGBOT_ROOT="$LINGBOT_ROOT" \
  PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
  REAL_DATA_ROOT="$REAL_DATA_ROOT" \
  PART2_ROOT="$PART2_ROOT" \
  STAGE1_CHECKPOINT="$INITIAL_MODEL" \
  PSEUDO_JSONL="$ONE_SHOT_BUFFER" \
  LOCAL_NGPU="$TRAIN_LOCAL_NGPU" \
  NNODES="$TRAIN_NNODES" \
  MASTER_ADDR="$TRAIN_MASTER_ADDR" \
  MASTER_PORT="$TRAIN_MASTER_PORT" \
  TRAIN_BATCH_SIZE_PER_GPU="$TRAIN_BATCH_SIZE_PER_GPU" \
  TARGET_GLOBAL_BATCH="$TRAIN_GLOBAL_BATCH" \
  RFT_NUM_STEPS=1 \
  RFT_SAVE_INTERVAL=1 \
  RFT_NUM_EPOCHS="$ONE_SHOT_TRAIN_EPOCHS" \
  MIXING_MODE=union \
  REAL_DATA_MODE="$REAL_DATA_MODE" \
  REAL_CHUNK_MODE=all-transitions \
  REAL_FRACTION="$REAL_FRACTION" \
  REAL_DATA_FRACTION="$ONE_SHOT_DATA_FRACTION" \
  DATA_FRACTION_SEED="$BASE_SEED" \
  ACTIVATION_CHECKPOINTING="$TRAIN_ACTIVATION_CHECKPOINTING" \
  WARMUP_STEPS="$ONE_SHOT_WARMUP_STEPS" \
  ENABLE_SWANLAB=1 \
  LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}" \
  LINGBOT_SWANLAB_RUN_ID="$SWANLAB_RUN_ID" \
  LINGBOT_SWANLAB_EXTERNAL="$SWANLAB_PARENT_DRIVER" \
  LINGBOT_SWANLAB_GROUP="${LINGBOT_SWANLAB_GROUP:-robotwin-stage1-real-stage2-pseudo}" \
  LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}" \
  SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-robotwin-stage2-one-shot-dual-rft}" \
  RUN_ID="one_shot_fraction_${ONE_SHOT_DATA_FRACTION}_3epochs" \
  OUT="$train_root" \
  CUDA_VISIBLE_DEVICES="$INFER_GPU_IDS" \
  bash "$TRAIN_LAUNCHER"; then
    echo "ONE_SHOT_RFT_TRAIN_FAILED train_root=$train_root" >&2
    return 1
  fi
  actual_steps="$($WAM_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["optimizer_steps"])' "$train_root/rft_dataset_report.json")"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.stage_updated_model \
    --base-model "$INITIAL_MODEL" \
    --transformer "$train_root/checkpoints/checkpoint_step_${actual_steps}/transformer" \
    --output "$final_model"
  "$WAM_PYTHON" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"complete": True, "pseudo_buffer": sys.argv[2], "train_root": sys.argv[3], "final_model": sys.argv[4], "optimizer_steps": int(sys.argv[5]), "epochs": int(sys.argv[6])}, indent=2)+"\n")' \
    "$ONE_SHOT_COMPLETE" "$ONE_SHOT_BUFFER" "$train_root" "$final_model" "$actual_steps" "$ONE_SHOT_TRAIN_EPOCHS"
  echo "ONE_SHOT_RFT_DONE pseudo=$ONE_SHOT_BUFFER model=$final_model steps=$actual_steps epochs=$ONE_SHOT_TRAIN_EPOCHS"
}

refresh_one_shot_buffer() {
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.finalize_one_shot_buffer \
    --collect-root "$ONE_SHOT_COLLECT_ROOT" \
    --budget "$PSEUDO_BUDGET" \
    --output "$ONE_SHOT_BUFFER" \
    --summary "$ONE_SHOT_SUMMARY" \
    --visual-cache "$ONE_SHOT_VISUAL_CACHE" \
    --target "$ONE_SHOT_EFFECTIVE_TARGET" \
    --max-per-context "$MAX_PSEUDO_PER_CONTEXT" \
    --max-per-episode "$ONE_SHOT_MAX_PER_EPISODE" \
    --progress-bins "$ONE_SHOT_PROGRESS_BINS" \
    --min-action-distance "$ONE_SHOT_MIN_ACTION_DISTANCE" \
    --min-mean-luma "$ONE_SHOT_MIN_MEAN_LUMA" \
    --min-std-luma "$ONE_SHOT_MIN_STD_LUMA" \
    --max-dark-fraction "$ONE_SHOT_MAX_DARK_FRACTION"
}

if [[ "$ONE_SHOT_MODE" == "1" && -d "$ONE_SHOT_COLLECT_ROOT" ]]; then
  refresh_one_shot_buffer
  one_shot_ready="$($WAM_PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["ready"]))' "$ONE_SHOT_SUMMARY")"
  if [[ "$one_shot_ready" == "1" ]]; then
    run_one_shot_training
    exit 0
  fi
fi

while true; do
  if [[ "$ONE_SHOT_MODE" == "1" && -s "$ONE_SHOT_COMPLETE" ]]; then
    echo "ONE_SHOT_RFT_DONE marker=$ONE_SHOT_COMPLETE"
    break
  fi
  if [[ "$ONE_SHOT_MODE" != "1" ]] && run_ready_update; then
    completed="$("$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration field --state "$STATE" --name update_index)"
    if (( MAX_UPDATES > 0 && completed >= MAX_UPDATES )); then
      echo "ONLINE_DUAL_RFT_DONE updates=$completed"
      break
    fi
    continue
  fi

  collect_index="$("$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration field --state "$STATE" --name collect_index)"
  current_model="$("$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration field --state "$STATE" --name current_model)"
  collect_dir="$ONLINE_ROOT/collect/collect_$(printf '%06d' "$collect_index")"
  if [[ ! -s "$collect_dir/collect_manifest.json" ]]; then
    "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration prepare \
      --state "$STATE" \
      --output-dir "$collect_dir" \
      --workers "$TOTAL_WORKERS" \
      --q-per-worker "$Q_PER_GPU"
  fi
  echo "ONLINE_COLLECT_START index=$collect_index model=$current_model"

  if (( REMOTE_INFER_WORKERS > 0 )); then
    submit_remote_stage generate "$collect_index" "$collect_dir" "$current_model"
  fi
  pids=()
  for worker in "${!GPUS[@]}"; do
    gpu="${GPUS[$worker]}"
    worker_dir="$collect_dir/worker_$(printf '%02d' "$worker")"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$((WORKER_MASTER_PORT_BASE + worker))" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    WAN_VA_DISABLE_WORLD1_FSDP=1 \
    "$WAM_PYTHON" \
      -m robotwin_critic.two_stage_rft.generate_wam_candidates \
      --contexts "$collect_dir/contexts_worker_$(printf '%02d' "$worker").jsonl" \
      --model "$current_model" \
      --output-dir "$worker_dir" \
      --candidates-per-context "$CANDIDATES_PER_Q" \
      --inference-batch-size "$INFER_BATCH_SIZE_PER_GPU" \
      --base-seed "$((BASE_SEED + collect_index * 1000000 + worker * 10000))" \
      --resume \
      > "$collect_dir/generate_worker_$(printf '%02d' "$worker").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  if (( REMOTE_INFER_WORKERS > 0 )); then wait_remote_stage; fi

  if (( REMOTE_INFER_WORKERS > 0 )); then
    submit_remote_stage decode "$collect_index" "$collect_dir" "$current_model"
  fi
  pids=()
  for worker in "${!GPUS[@]}"; do
    gpu="${GPUS[$worker]}"
    worker_dir="$collect_dir/worker_$(printf '%02d' "$worker")"
    CUDA_VISIBLE_DEVICES="$gpu" "$WAM_PYTHON" \
      -m robotwin_critic.two_stage_rft.decode_wam_candidates \
      --input "$worker_dir/candidates.jsonl" \
      --model "$current_model" \
      --device cuda:0 \
      > "$collect_dir/decode_worker_$(printf '%02d' "$worker").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  if (( REMOTE_INFER_WORKERS > 0 )); then wait_remote_stage; fi

  merge_args=()
  for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
    merge_args+=(--input "$collect_dir/worker_$(printf '%02d' "$worker")/candidates.jsonl")
  done
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration merge \
    "${merge_args[@]}" --output "$collect_dir/candidates.jsonl"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.score_action_chunks \
    --input "$collect_dir/candidates.jsonl" \
    --profile "$ACTION_PROFILE" \
    --output "$collect_dir/action_scored.jsonl" \
    --min-score "$MIN_ACTION_SCORE" \
    --gate-policy "$ACTION_GATE_POLICY" \
    --workspace-scope "$ACTION_WORKSPACE_SCOPE"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration split \
    --input "$collect_dir/action_scored.jsonl" \
    --output-dir "$collect_dir/vlac_shards" \
    --workers "$TOTAL_WORKERS"

  if (( REMOTE_INFER_WORKERS > 0 )); then
    submit_remote_stage vlac "$collect_index" "$collect_dir" "$current_model"
  fi
  pids=()
  for worker in "${!GPUS[@]}"; do
    gpu="${GPUS[$worker]}"
    adapter_args=()
    if [[ -n "$VLAC_ADAPTER" ]]; then adapter_args+=(--adapter "$VLAC_ADAPTER"); fi
    CUDA_VISIBLE_DEVICES="$gpu" "$VLAC_PYTHON" \
      -m robotwin_critic.two_stage_rft.score_vlac_candidates \
      --input "$collect_dir/vlac_shards/shard_$(printf '%02d' "$worker").jsonl" \
      --model "$VLAC_MODEL" \
      --output "$collect_dir/vlac_scored_$(printf '%02d' "$worker").jsonl" \
      --device cuda:0 \
      --batch-size "$VLAC_BATCH_SIZE_PER_GPU" \
      "${adapter_args[@]}" \
      > "$collect_dir/vlac_worker_$(printf '%02d' "$worker").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  if (( REMOTE_INFER_WORKERS > 0 )); then wait_remote_stage; fi

  merge_args=()
  for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
    merge_args+=(--input "$collect_dir/vlac_scored_$(printf '%02d' "$worker").jsonl")
  done
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration merge \
    "${merge_args[@]}" --output "$collect_dir/dual_scored.jsonl"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration commit \
    --state "$STATE" \
    --collect-dir "$collect_dir" \
    --scored "$collect_dir/dual_scored.jsonl" \
    --pending "$PENDING" \
    --buffers-dir "$BUFFERS_DIR" \
    --capacity "$([[ "$ONE_SHOT_MODE" == "1" ]] && printf 0 || printf '%s' "$BUFFER_CAPACITY")" \
    --min-action-score "$MIN_ACTION_SCORE" \
    --min-process-score "$MIN_PROCESS_SCORE" \
    --max-per-context "$MAX_PSEUDO_PER_CONTEXT" \
    --split-manifest "$SPLIT_MANIFEST"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.summarize_online_task_retention \
    --collect-root "$ONLINE_ROOT/collect" \
    --output "$TASK_RETENTION_SUMMARY"
  log_collection_swanlab "$collect_index"
  echo "ONLINE_COLLECT_OK index=$collect_index"
  if [[ "$ONE_SHOT_MODE" == "1" ]]; then
    refresh_one_shot_buffer
    "$WAM_PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("ONE_SHOT_BUFFER_EVENT "+json.dumps({"collect_index": int(sys.argv[2]), "selected": d["selected"], "target": d["target"], "ready": d["ready"], "unique_contexts": d["unique_contexts"], "unique_episodes": d["unique_episodes"], "visual_rejected": d["visual_rejected"], "action_duplicate_rejected": d["action_duplicate_rejected"], "groups_with_quota_shortfall": len(d["quota_shortfalls_before_backfill"]), "groups_overfilled": len(d["quota_overfill_after_backfill"])}))' "$ONE_SHOT_SUMMARY" "$collect_index"
    one_shot_ready="$($WAM_PYTHON -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["ready"]))' "$ONE_SHOT_SUMMARY")"
    if [[ "$one_shot_ready" == "1" ]]; then
      run_one_shot_training
      break
    fi
  fi
done
