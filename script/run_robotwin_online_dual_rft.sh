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

# Sampling throughput hyperparameters. With four GPUs, defaults give 128 Q and
# 128*8=1024 candidates per collection round.
INFER_GPU_IDS="${INFER_GPU_IDS:-0,1,2,3}"
Q_PER_ROUND="${Q_PER_ROUND:-128}"
INFER_BATCH_SIZE_PER_GPU="${INFER_BATCH_SIZE_PER_GPU:-1}"
CANDIDATES_PER_Q="${CANDIDATES_PER_Q:-8}"
VLAC_BATCH_SIZE_PER_GPU="${VLAC_BATCH_SIZE_PER_GPU:-4}"

# Replay/update hyperparameters. Increase TRAIN_BATCH_SIZE_PER_GPU on 280GB GPUs.
BUFFER_CAPACITY="${BUFFER_CAPACITY:-1024}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-8}"
TRAIN_GLOBAL_BATCH="${TRAIN_GLOBAL_BATCH:-32}"
TRAIN_ACTIVATION_CHECKPOINTING="${TRAIN_ACTIVATION_CHECKPOINTING:-1}"
PSEUDO_EPOCHS_PER_UPDATE="${PSEUDO_EPOCHS_PER_UPDATE:-3}"
REAL_FRACTION="${REAL_FRACTION:-0.5}"
UPDATE_STEPS="${UPDATE_STEPS:-}"
MAX_UPDATES="${MAX_UPDATES:-1000}"
MODEL_SAVE_EVERY_UPDATES="${MODEL_SAVE_EVERY_UPDATES:-50}"
MIN_ACTION_SCORE="${MIN_ACTION_SCORE:-0.5}"
ACTION_GATE_POLICY="${ACTION_GATE_POLICY:-score_with_safety_gates}"
ACTION_WORKSPACE_SCOPE="${ACTION_WORKSPACE_SCOPE:-global}"
MIN_PROCESS_SCORE="${MIN_PROCESS_SCORE:-5.0}"
MAX_PSEUDO_PER_CONTEXT="${MAX_PSEUDO_PER_CONTEXT:-0}"
BASE_SEED="${BASE_SEED:-42}"
WORKER_MASTER_PORT_BASE="${WORKER_MASTER_PORT_BASE:-29700}"

CONTEXTS="${CONTEXTS:-$PART2_ROOT/stage2_video_contexts.jsonl}"
ACTION_PROFILE="${ACTION_PROFILE:-$PART2_ROOT/stage1_kinematic_profile.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$PREPARED_DATA_ROOT/split_manifest.json}"
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

IFS=',' read -r -a GPUS <<< "$INFER_GPU_IDS"
NGPU="${#GPUS[@]}"
if (( NGPU < 1 )); then
  echo "INFER_GPU_IDS must contain at least one GPU" >&2
  exit 2
fi
if (( Q_PER_ROUND <= 0 || Q_PER_ROUND % NGPU != 0 )); then
  echo "Q_PER_ROUND must be positive and divisible by number_of_GPUs" >&2
  exit 2
fi
Q_PER_GPU=$((Q_PER_ROUND / NGPU))
if (( TRAIN_GLOBAL_BATCH % (TRAIN_BATCH_SIZE_PER_GPU * NGPU) != 0 )); then
  echo "TRAIN_GLOBAL_BATCH must divide by TRAIN_BATCH_SIZE_PER_GPU*number_of_GPUs" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((TRAIN_GLOBAL_BATCH / (TRAIN_BATCH_SIZE_PER_GPU * NGPU)))
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

echo "ONLINE_DUAL_RFT_CONFIG infer_gpus=$INFER_GPU_IDS q_per_round=$Q_PER_ROUND q_per_gpu=$Q_PER_GPU inference_batch_per_gpu=$INFER_BATCH_SIZE_PER_GPU candidates_per_q=$CANDIDATES_PER_Q buffer_capacity=$BUFFER_CAPACITY pseudo_epochs_per_update=$PSEUDO_EPOCHS_PER_UPDATE real_fraction=$REAL_FRACTION train_batch_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU train_global_batch=$TRAIN_GLOBAL_BATCH gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS activation_checkpointing=$TRAIN_ACTIVATION_CHECKPOINTING update_steps=$UPDATE_STEPS max_updates=$MAX_UPDATES model_save_every_updates=$MODEL_SAVE_EVERY_UPDATES action_gate_policy=$ACTION_GATE_POLICY action_workspace_scope=$ACTION_WORKSPACE_SCOPE min_action_score=$MIN_ACTION_SCORE min_process_score=$MIN_PROCESS_SCORE max_pseudo_per_context=$MAX_PSEUDO_PER_CONTEXT swanlab_run_id=$SWANLAB_RUN_ID"

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
  PROJECT_ROOT="$PROJECT_ROOT" \
  LINGBOT_ROOT="$LINGBOT_ROOT" \
  PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
  REAL_DATA_ROOT="$REAL_DATA_ROOT" \
  PART2_ROOT="$PART2_ROOT" \
  STAGE1_CHECKPOINT="$current_model" \
  PSEUDO_JSONL="$ready" \
  NGPU="$NGPU" \
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
  bash script/run_robotwin_joint_rft.sh
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.stage_updated_model \
    --base-model "$current_model" \
    --transformer "$train_root/checkpoints/checkpoint_step_${UPDATE_STEPS}/transformer" \
    --output "$staged_model" \
    --move-transformer
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.online_iteration complete-update \
    --state "$STATE" \
    --model "$staged_model"

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

while true; do
  if run_ready_update; then
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
      --workers "$NGPU" \
      --q-per-worker "$Q_PER_GPU"
  fi
  echo "ONLINE_COLLECT_START index=$collect_index model=$current_model"

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

  merge_args=()
  for worker in "${!GPUS[@]}"; do
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
    --workers "$NGPU"

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

  merge_args=()
  for worker in "${!GPUS[@]}"; do
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
    --capacity "$BUFFER_CAPACITY" \
    --min-action-score "$MIN_ACTION_SCORE" \
    --min-process-score "$MIN_PROCESS_SCORE" \
    --max-per-context "$MAX_PSEUDO_PER_CONTEXT" \
    --split-manifest "$SPLIT_MANIFEST"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.summarize_online_task_retention \
    --collect-root "$ONLINE_ROOT/collect" \
    --output "$TASK_RETENTION_SUMMARY"
  log_collection_swanlab "$collect_index"
  echo "ONLINE_COLLECT_OK index=$collect_index"
done
