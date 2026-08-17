#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$LINGBOT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to the RFT training initializer}"
PSEUDO_JSONL="${PSEUDO_JSONL:-$PART2_ROOT/dual_rft_selected.jsonl}"
LOCAL_NGPU="${LOCAL_NGPU:-${NGPU:-8}}"
NNODES="${NNODES:-1}"
NODE_RANK_WAS_SET="${NODE_RANK+x}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
HETEROGENEOUS_RANK_LAUNCH="${HETEROGENEOUS_RANK_LAUNCH:-0}"
RANK_OFFSET="${RANK_OFFSET:-}"
RANK_LAYOUT="${RANK_LAYOUT:-}"
RANK_LAYOUT_FILE="${RANK_LAYOUT_FILE:-}"
if [[ "$HETEROGENEOUS_RANK_LAUNCH" != "0" && "$HETEROGENEOUS_RANK_LAUNCH" != "1" ]]; then
  echo "HETEROGENEOUS_RANK_LAUNCH must be 0 or 1" >&2
  exit 2
fi
if [[ "$HETEROGENEOUS_RANK_LAUNCH" == "1" ]]; then
  : "${WORLD_SIZE:?Heterogeneous launch requires WORLD_SIZE}"
  : "${RANK_OFFSET:?Heterogeneous launch requires RANK_OFFSET}"
  : "${RANK_LAYOUT:?Heterogeneous launch requires RANK_LAYOUT}"
  for rank_value_name in WORLD_SIZE RANK_OFFSET LOCAL_NGPU; do
    rank_value="${!rank_value_name}"
    if [[ ! "$rank_value" =~ ^[0-9]+$ ]]; then
      echo "$rank_value_name must be a decimal integer, got: $rank_value" >&2
      exit 2
    fi
  done
fi
if [[ "$HETEROGENEOUS_RANK_LAUNCH" == "1" && -z "$NODE_RANK_WAS_SET" ]]; then
  NODE_RANK="${RANK_OFFSET:-0}"
fi
USE_HETEROGENEOUS_HELPER="$HETEROGENEOUS_RANK_LAUNCH"
if [[ "$HETEROGENEOUS_RANK_LAUNCH" == "1" ]]; then
  # Uniform layouts are ordinary homogeneous multi-node jobs.  Keep those on
  # torchrun so PyTorch owns the complete rank/group environment used by FSDP2.
  # The custom helper is reserved for layouts torchrun cannot represent.
  layout_node_rank=-1
  layout_node_count=0
  layout_uniform_count=""
  layout_uniform=1
  layout_cursor=0
  layout_local_matches=0
  IFS=',' read -r -a layout_ranges <<< "$RANK_LAYOUT"
  for layout_range in "${layout_ranges[@]}"; do
    if [[ ! "$layout_range" =~ ^([0-9]+):([0-9]+)$ ]]; then
      echo "Invalid RANK_LAYOUT entry: $layout_range" >&2
      exit 2
    fi
    layout_start=$((10#${BASH_REMATCH[1]}))
    layout_count=$((10#${BASH_REMATCH[2]}))
    if (( layout_count < 1 || layout_start != layout_cursor )); then
      echo "RANK_LAYOUT must be contiguous with positive counts: $RANK_LAYOUT" >&2
      exit 2
    fi
    if [[ -z "$layout_uniform_count" ]]; then
      layout_uniform_count="$layout_count"
    elif (( layout_count != layout_uniform_count )); then
      layout_uniform=0
    fi
    if (( layout_start == 10#$RANK_OFFSET && layout_count == 10#$LOCAL_NGPU )); then
      layout_node_rank="$layout_node_count"
      layout_local_matches=$((layout_local_matches + 1))
    fi
    layout_cursor=$((layout_cursor + layout_count))
    layout_node_count=$((layout_node_count + 1))
  done
  if (( layout_cursor != 10#$WORLD_SIZE || layout_local_matches != 1 )); then
    echo "RANK_LAYOUT must cover WORLD_SIZE and contain local slice exactly once" >&2
    exit 2
  fi
  if [[ -n "$RANK_LAYOUT_FILE" ]]; then
    if [[ ! -r "$RANK_LAYOUT_FILE" ]] || [[ "$(<"$RANK_LAYOUT_FILE")" != "$RANK_LAYOUT" ]]; then
      echo "RANK_LAYOUT does not match readable RANK_LAYOUT_FILE=$RANK_LAYOUT_FILE" >&2
      exit 2
    fi
  fi
  if (( layout_uniform == 1 )); then
    if (( layout_node_rank < 0 || layout_uniform_count != 10#$LOCAL_NGPU )); then
      echo "Local slice $RANK_OFFSET:$LOCAL_NGPU is absent from uniform RANK_LAYOUT=$RANK_LAYOUT" >&2
      exit 2
    fi
    USE_HETEROGENEOUS_HELPER=0
    NNODES="$layout_node_count"
    NODE_RANK="$layout_node_rank"
  fi
fi
if [[ "${RFT_LAUNCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'launch_backend=%s nnodes=%s node_rank=%s local_ngpu=%s world_size=%s\n' \
    "$([[ "$USE_HETEROGENEOUS_HELPER" == "1" ]] && printf heterogeneous-helper || printf torchrun)" \
    "$NNODES" "$NODE_RANK" "$LOCAL_NGPU" "${WORLD_SIZE:-$((LOCAL_NGPU * NNODES))}"
  exit 0
fi
RFT_LAUNCH_PLAN_ONLY="${RFT_LAUNCH_PLAN_ONLY:-0}"
if [[ "$RFT_LAUNCH_PLAN_ONLY" != "0" && "$RFT_LAUNCH_PLAN_ONLY" != "1" ]]; then
  echo "RFT_LAUNCH_PLAN_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "$RFT_LAUNCH_PLAN_ONLY" == "1" ]]; then
  if [[ "$USE_HETEROGENEOUS_HELPER" == "1" ]]; then
    launch_backend=per-rank-helper
  else
    launch_backend=torchrun
  fi
  echo "launch_backend=$launch_backend"
  echo "world_size=${WORLD_SIZE:-$((LOCAL_NGPU * NNODES))}"
  echo "nnodes=$NNODES"
  echo "node_rank=$NODE_RANK"
  echo "local_ngpu=$LOCAL_NGPU"
  echo "rank_offset=${RANK_OFFSET:-0}"
  echo "rank_layout=${RANK_LAYOUT:-homogeneous}"
  exit 0
fi
SCHEDULE_MODE="${RFT_SCHEDULE_MODE:-steps}"
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
PSEUDO_GLOBAL_BATCH="${PSEUDO_GLOBAL_BATCH:-8}"
PSEUDO_LOSS_WEIGHT="${PSEUDO_LOSS_WEIGHT:-0.25}"
PSEUDO_LOSS_WARMUP_STEPS="${PSEUDO_LOSS_WARMUP_STEPS:-3000}"
PSEUDO_SAMPLER_SEED="${PSEUDO_SAMPLER_SEED:-43}"
LEGACY_PSEUDO_ACTION_WAIVER_SHA256="${LEGACY_PSEUDO_ACTION_WAIVER_SHA256:-}"
LEGACY_PSEUDO_ACTION_WAIVER_ROWS="${LEGACY_PSEUDO_ACTION_WAIVER_ROWS:-}"
REAL_DATA_ROOT="${REAL_DATA_ROOT:-}"
OUTER_STEP="${RFT_OUTER_STEP:-0}"
SWANLAB_STEP_OFFSET="${RFT_SWANLAB_STEP_OFFSET:-0}"
SMOKE_MODE=0
BASE_AUXILIARY_SMOKE_STEPS=0
BASE_REAL_REGRESSION_STEPS="${RFT_BASE_REAL_REGRESSION_STEPS:-0}"
BASE_AUXILIARY_ACTIVATION_CHECKPOINTING="${RFT_BASE_AUXILIARY_ACTIVATION_CHECKPOINTING:-0}"
if [[ "$BASE_AUXILIARY_ACTIVATION_CHECKPOINTING" != "0" \
      && "$BASE_AUXILIARY_ACTIVATION_CHECKPOINTING" != "1" ]]; then
  echo "RFT_BASE_AUXILIARY_ACTIVATION_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi
BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING="${RFT_BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING:-0}"
if [[ "$BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING" != "0" \
      && "$BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING" != "1" ]]; then
  echo "RFT_BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$BASE_REAL_REGRESSION_STEPS" =~ ^[0-9]+$ ]] \
    || (( 10#$BASE_REAL_REGRESSION_STEPS > 2000 )); then
  echo "RFT_BASE_REAL_REGRESSION_STEPS must be a decimal integer from 0 through 2000" >&2
  exit 2
fi
BASE_REAL_REGRESSION_STEPS=$((10#$BASE_REAL_REGRESSION_STEPS))
if [[ -n "${RFT_BASE_AUXILIARY_SMOKE_STEPS+x}" ]]; then
  if [[ "${RFT_SMOKE_MODE:-}" != "1" ]]; then
    echo "RFT_BASE_AUXILIARY_SMOKE_STEPS is accepted only when RFT_SMOKE_MODE=1" >&2
    exit 2
  fi
  if [[ ! "$RFT_BASE_AUXILIARY_SMOKE_STEPS" =~ ^[0-9]+$ ]] \
      || (( 10#$RFT_BASE_AUXILIARY_SMOKE_STEPS < 1 \
            || 10#$RFT_BASE_AUXILIARY_SMOKE_STEPS > 1000 )); then
    echo "RFT_BASE_AUXILIARY_SMOKE_STEPS must be a decimal integer from 1 through 1000" >&2
    exit 2
  fi
  if [[ "$SCHEDULE_MODE" != "base-auxiliary-pseudo" ]]; then
    echo "RFT_BASE_AUXILIARY_SMOKE_STEPS requires RFT_SCHEDULE_MODE=base-auxiliary-pseudo" >&2
    exit 2
  fi
  SMOKE_MODE=1
  BASE_AUXILIARY_SMOKE_STEPS=$((10#$RFT_BASE_AUXILIARY_SMOKE_STEPS))
elif [[ "${RFT_SMOKE_MODE:-0}" == "1" ]]; then
  echo "RFT_SMOKE_MODE=1 requires RFT_BASE_AUXILIARY_SMOKE_STEPS" >&2
  exit 2
fi
if (( BASE_REAL_REGRESSION_STEPS > 0 )); then
  if [[ "$SCHEDULE_MODE" != "base-auxiliary-pseudo" ]]; then
    echo "RFT_BASE_REAL_REGRESSION_STEPS requires RFT_SCHEDULE_MODE=base-auxiliary-pseudo" >&2
    exit 2
  fi
  if (( SMOKE_MODE == 1 )); then
    echo "real-only regression and smoke mode are mutually exclusive" >&2
    exit 2
  fi
  "$WAM_PYTHON" -c 'import sys; assert float(sys.argv[1]) == 0.0' "$PSEUDO_LOSS_WEIGHT" || {
    echo "real-only regression requires PSEUDO_LOSS_WEIGHT=0" >&2
    exit 2
  }
fi
if { (( NNODES > 1 )) || [[ "$HETEROGENEOUS_RANK_LAUNCH" == "1" ]]; } \
    && [[ -z "${RUN_ID:-}" && -z "${OUT:-}" ]]; then
  echo "Multi-node runs must set the same RUN_ID or OUT on every node" >&2
  exit 2
fi
RUN_ID="${RUN_ID:-robotwin_dual_rft_joint_full_${NUM_STEPS}steps_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$LINGBOT_ROOT/train_out/robotwin/$RUN_ID}"
RFT_INVOCATION_ID="${RFT_INVOCATION_ID:-$($WAM_PYTHON -c 'import uuid; print(uuid.uuid4())')}"
WAIVER_ARGS=()
REGRESSION_ARGS=()
BASE_AUXILIARY_ARGS=()
if [[ -n "$LEGACY_PSEUDO_ACTION_WAIVER_SHA256" || -n "$LEGACY_PSEUDO_ACTION_WAIVER_ROWS" ]]; then
  WAIVER_ARGS+=(--legacy-pseudo-action-waiver-sha256 "$LEGACY_PSEUDO_ACTION_WAIVER_SHA256")
  WAIVER_ARGS+=(--legacy-pseudo-action-waiver-rows "$LEGACY_PSEUDO_ACTION_WAIVER_ROWS")
fi
if [[ "$BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING" == "1" ]]; then
  REGRESSION_ARGS+=(--base-real-regression-activation-checkpointing)
fi
if [[ "$BASE_AUXILIARY_ACTIVATION_CHECKPOINTING" == "1" ]]; then
  BASE_AUXILIARY_ARGS+=(--base-auxiliary-activation-checkpointing)
fi

cd "$PROJECT_ROOT"
test -x "$WAM_PYTHON"
test -s "$STAGE1_CHECKPOINT/transformer/config.json"
test -s "$PSEUDO_JSONL"
test -d "$PREPARED_DATA_ROOT/stage1"
test -s "$PREPARED_DATA_ROOT/stage1/empty_emb.pt"
if [[ "$SCHEDULE_MODE" == "base-auxiliary-pseudo" ]]; then
  PSEUDO_EXPECTED_LATENT_FRAMES=2
  PSEUDO_ACTION_PER_FRAME=16
  PSEUDO_JSONL_RESOLVED="$("$WAM_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$PSEUDO_JSONL")"
  OUT_RESOLVED="$("$WAM_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUT")"
  PSEUDO_CONTRACT_DIR="$("$WAM_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).parent)' "$PSEUDO_JSONL_RESOLVED")"
  test -d "$PSEUDO_CONTRACT_DIR"
  PSEUDO_CONTRACT_KEY="$("$WAM_PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(b"\0".join(value.encode() for value in sys.argv[1:])).hexdigest())' "$OUT_RESOLVED" "$PSEUDO_JSONL_RESOLVED" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES")"
  PSEUDO_CONTRACT_SENTINEL="$PSEUDO_CONTRACT_DIR/.pseudo_action_contract.${PSEUDO_CONTRACT_KEY}.success.json"
if (( NODE_RANK == 0 )); then
  rm -f -- "$PSEUDO_CONTRACT_SENTINEL"
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.pseudo_action_contract \
    "$PSEUDO_JSONL_RESOLVED" \
    --expected-latent-frames "$PSEUDO_EXPECTED_LATENT_FRAMES" \
    --action-per-frame "$PSEUDO_ACTION_PER_FRAME" \
    "${WAIVER_ARGS[@]}"
  "$WAM_PYTHON" - "$PSEUDO_JSONL_RESOLVED" "$PSEUDO_CONTRACT_SENTINEL" "$PSEUDO_CONTRACT_KEY" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

pseudo = Path(sys.argv[1])
sentinel = Path(sys.argv[2])
stat = pseudo.stat()
payload = {
    "action_per_frame": 16,
    "expected_latent_frames": 2,
    "file_size": stat.st_size,
    "mtime_ns": stat.st_mtime_ns,
    "resolved_pseudo_path": str(pseudo),
    "run_identity": sys.argv[3],
}
temporary_name = None
try:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=sentinel.parent,
        prefix=f".{sentinel.name}.", delete=False,
    ) as handle:
        temporary_name = handle.name
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, sentinel)
    temporary_name = None
finally:
    if temporary_name is not None:
        Path(temporary_name).unlink(missing_ok=True)
PY
else
  deadline=$(( $(date +%s) + 300 ))
  while ! "$WAM_PYTHON" - "$PSEUDO_JSONL_RESOLVED" "$PSEUDO_CONTRACT_SENTINEL" "$PSEUDO_CONTRACT_KEY" <<'PY'
import json
from pathlib import Path
import sys

pseudo = Path(sys.argv[1])
try:
    stat = pseudo.stat()
    with Path(sys.argv[2]).open(encoding="utf-8") as handle:
        actual = json.load(handle)
except (OSError, ValueError):
    raise SystemExit(1)
expected = {
    "action_per_frame": 16,
    "expected_latent_frames": 2,
    "file_size": stat.st_size,
    "mtime_ns": stat.st_mtime_ns,
    "resolved_pseudo_path": str(pseudo),
    "run_identity": sys.argv[3],
}
if actual != expected:
    raise SystemExit(1)
PY
  do
    if (( $(date +%s) >= deadline )); then
      echo "Timed out waiting for rank-0 pseudo action contract validation" >&2
      exit 2
    fi
    sleep 1
  done
fi
fi
case "$REAL_DATA_MODE" in
  stage1)
    REAL_DATA_ROOT="${REAL_DATA_ROOT:-$PREPARED_DATA_ROOT/stage1}"
    ;;
  stage2)
    REAL_DATA_ROOT="${REAL_DATA_ROOT:-$($WAM_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "$PREPARED_DATA_ROOT/split_manifest.json")}"
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
if [[ "$HETEROGENEOUS_RANK_LAUNCH" == "1" ]]; then
  WORLD_SIZE="$WORLD_SIZE"
else
  WORLD_SIZE=$((LOCAL_NGPU * NNODES))
fi
if [[ ! "$PSEUDO_LOSS_WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
  echo "PSEUDO_LOSS_WARMUP_STEPS must be a non-negative decimal integer" >&2
  exit 2
fi
case "$SCHEDULE_MODE" in
  steps|base-auxiliary-pseudo|epochs) ;;
  *) echo "Unsupported RFT_SCHEDULE_MODE: $SCHEDULE_MODE" >&2; exit 2 ;;
esac
if [[ "$SCHEDULE_MODE" == "base-auxiliary-pseudo" ]]; then
  if [[ "$REAL_DATA_MODE" == "stage2" || "$REAL_DATA_MODE" == "stage1-stage2" ]]; then
    required_real_root="$(realpath -m "$($WAM_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_root"])' "$PREPARED_DATA_ROOT/split_manifest.json")")"
  else
    required_real_root="$(realpath -m "$PREPARED_DATA_ROOT/stage1")"
  fi
  actual_real_root="$(realpath -m "$REAL_DATA_ROOT")"
  if [[ "$REAL_DATA_MODE" != "stage1" && "$REAL_DATA_MODE" != "stage2" && "$REAL_DATA_MODE" != "stage1-stage2" ]] || [[ "$actual_real_root" != "$required_real_root" ]]; then
    echo "base-auxiliary-pseudo requires REAL_DATA_MODE=stage1, stage2, or stage1-stage2 with its canonical root; got mode=$REAL_DATA_MODE root=$actual_real_root expected_root=$required_real_root" >&2
    exit 2
  fi
  TRAIN_BATCH_SIZE_PER_GPU="${RFT_BASE_AUXILIARY_BATCH_SIZE_PER_GPU:-1}"
  TARGET_GLOBAL_BATCH=64
  REAL_CHUNK_MODE=full
  ACTIVATION_CHECKPOINTING="$BASE_AUXILIARY_ACTIVATION_CHECKPOINTING"
  WARMUP_STEPS=10
  LR_SCHEDULER=constant
  MAX_EPISODE_FRAMES=1000000000
  NUM_STEPS="${RFT_BASE_AUXILIARY_STEPS:-15000}"
  SAVE_INTERVAL="${RFT_BASE_AUXILIARY_SAVE_INTERVAL:-3000}"
  if [[ ! "$NUM_STEPS" =~ ^[1-9][0-9]*$ \
        || ! "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ \
        || 10#$SAVE_INTERVAL -gt 10#$NUM_STEPS ]]; then
    echo "RFT_BASE_AUXILIARY_STEPS and RFT_BASE_AUXILIARY_SAVE_INTERVAL must be positive decimals with save interval <= steps" >&2
    exit 2
  fi
  NUM_STEPS=$((10#$NUM_STEPS))
  SAVE_INTERVAL=$((10#$SAVE_INTERVAL))
  if (( SMOKE_MODE == 1 )); then
    NUM_STEPS="$BASE_AUXILIARY_SMOKE_STEPS"
    SAVE_INTERVAL="$BASE_AUXILIARY_SMOKE_STEPS"
  elif (( BASE_REAL_REGRESSION_STEPS > 0 )); then
    NUM_STEPS="$BASE_REAL_REGRESSION_STEPS"
    SAVE_INTERVAL="$BASE_REAL_REGRESSION_STEPS"
    ACTIVATION_CHECKPOINTING="$BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING"
  fi
  MIXING_MODE=auxiliary
  if [[ ! "$TRAIN_BATCH_SIZE_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "RFT_BASE_AUXILIARY_BATCH_SIZE_PER_GPU must be a positive decimal integer" >&2
    exit 2
  fi
  TRAIN_BATCH_SIZE_PER_GPU=$((10#$TRAIN_BATCH_SIZE_PER_GPU))
  BASE_AUXILIARY_DENOMINATOR=$((TRAIN_BATCH_SIZE_PER_GPU * WORLD_SIZE))
  if (( BASE_AUXILIARY_DENOMINATOR > 64 || 64 % BASE_AUXILIARY_DENOMINATOR != 0 )); then
    echo "base-auxiliary-pseudo requires batch_per_gpu*WORLD_SIZE to divide global batch 64" >&2
    exit 2
  fi
  GRADIENT_ACCUMULATION_STEPS=$((64 / BASE_AUXILIARY_DENOMINATOR))
  if (( PSEUDO_GLOBAL_BATCH <= 0 || PSEUDO_GLOBAL_BATCH % WORLD_SIZE != 0 )); then
    echo "PSEUDO_GLOBAL_BATCH must be positive and divisible by WORLD_SIZE ($WORLD_SIZE)" >&2
    exit 2
  fi
elif [[ "$SCHEDULE_MODE" == "epochs" && "$NUM_EPOCHS" -le 0 ]]; then
  echo "epochs schedule requires RFT_NUM_EPOCHS > 0" >&2
  exit 2
fi
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PSEUDO_EXPECTED_SAMPLES="$($WAM_PYTHON -c \
  'import sys; print(int(sys.argv[2]) * int(sys.argv[3]) if float(sys.argv[1]) > 0 else 0)' \
  "$PSEUDO_LOSS_WEIGHT" "$NUM_STEPS" "$PSEUDO_GLOBAL_BATCH")"

if (( NODE_RANK == 0 )); then
{
  echo "run_id=$RUN_ID"
  echo "rft_training_initializer=$STAGE1_CHECKPOINT"
  echo "real_dataset=$REAL_DATA_ROOT"
  echo "real_data_mode=$REAL_DATA_MODE"
  echo "pseudo_jsonl=$PSEUDO_JSONL"
  echo "legacy_pseudo_action_waiver_sha256=${LEGACY_PSEUDO_ACTION_WAIVER_SHA256:-disabled}"
  echo "legacy_pseudo_action_waiver_rows=${LEGACY_PSEUDO_ACTION_WAIVER_ROWS:-disabled}"
  echo "selection_mode=$RFT_SELECTION_MODE"
  echo "real_source_update_ratio=$REAL_FRACTION"
  echo "real_data_fraction=$REAL_DATA_FRACTION"
  echo "data_fraction_seed=$DATA_FRACTION_SEED"
  echo "mixing_mode=$MIXING_MODE"
  echo "schedule_mode=$SCHEDULE_MODE"
  echo "num_epochs=$NUM_EPOCHS"
  echo "objective=joint_video_action_flow_matching"
  echo "trainable_scope=full_transformer"
  echo "real_chunk_mode=$REAL_CHUNK_MODE"
  echo "train_batch_size_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU"
  echo "nnodes=$NNODES"
  echo "local_ngpu=$LOCAL_NGPU"
  echo "world_size=$WORLD_SIZE"
  echo "heterogeneous_rank_launch=$HETEROGENEOUS_RANK_LAUNCH"
  echo "heterogeneous_helper_used=$USE_HETEROGENEOUS_HELPER"
  echo "effective_nnodes=$NNODES"
  echo "effective_node_rank=$NODE_RANK"
  echo "rank_offset=${RANK_OFFSET:-homogeneous}"
  echo "rank_layout=${RANK_LAYOUT:-homogeneous}"
  echo "rank_layout_file=${RANK_LAYOUT_FILE:-not-set}"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "nccl_net=${NCCL_NET-<inherited-unset>}"
  echo "nccl_socket_ifname=${NCCL_SOCKET_IFNAME-<inherited-unset>}"
  echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  echo "activation_checkpointing=$ACTIVATION_CHECKPOINTING"
  echo "dataset_init_workers=${DATASET_INIT_WORKERS:-64}"
  echo "train_load_workers=${TRAIN_LOAD_WORKERS:-16}"
  echo "gc_interval=${GC_INTERVAL:-50}"
  echo "learning_rate=1e-5"
  echo "warmup_steps=${WARMUP_STEPS:-1000}"
  echo "lr_scheduler=${LR_SCHEDULER:-constant}"
  echo "num_steps=$NUM_STEPS"
  echo "smoke_mode=$SMOKE_MODE"
  echo "smoke_steps=$BASE_AUXILIARY_SMOKE_STEPS"
  echo "real_only_regression=$((BASE_REAL_REGRESSION_STEPS > 0))"
  echo "real_only_regression_steps=$BASE_REAL_REGRESSION_STEPS"
  echo "real_only_regression_activation_checkpointing=$BASE_REAL_REGRESSION_ACTIVATION_CHECKPOINTING"
  echo "base_auxiliary_activation_checkpointing=$BASE_AUXILIARY_ACTIVATION_CHECKPOINTING"
  if (( ACTIVATION_CHECKPOINTING == 1 )); then
    echo "base_execution_deviation=activation_checkpointing_enabled_for_2xh200_memory"
  else
    echo "base_execution_deviation=none"
  fi
  echo "save_interval=$SAVE_INTERVAL"
  echo "save_steps=$SAVE_STEPS"
  echo "saved_step_directory_kind=inference_model_snapshot"
  echo "resumable_optimizer_checkpoint=false"
  echo "invocation_id=$RFT_INVOCATION_ID"
  echo "global_batch=$TARGET_GLOBAL_BATCH"
  echo "expected_composition=reported_in_rft_dataset_report.json"
  case "$SCHEDULE_MODE" in
    base-auxiliary-pseudo)
      echo "pseudo_global_batch=$PSEUDO_GLOBAL_BATCH"
      echo "pseudo_microbatches_per_rank_update=$((PSEUDO_GLOBAL_BATCH / WORLD_SIZE))"
      echo "pseudo_loss_weight=$PSEUDO_LOSS_WEIGHT"
      echo "pseudo_loss_warmup_steps=$PSEUDO_LOSS_WARMUP_STEPS"
      echo "pseudo_sampler_seed=$PSEUDO_SAMPLER_SEED"
      echo "loss_metrics=real_latent_unscaled,real_action_unscaled,pseudo_latent_unscaled,pseudo_action_unscaled,combined_update_loss"
      echo "expected_real_samples=$((NUM_STEPS * 64))"
      echo "expected_pseudo_samples=$PSEUDO_EXPECTED_SAMPLES"
      echo "observed_sample_counts_output=$OUT/rft_source_counts.json"
      echo "gradient_objective=mean_64_real_plus_linear_warmup_to_${PSEUDO_LOSS_WEIGHT}_target_coefficient_times_mean_${PSEUDO_GLOBAL_BATCH}_pseudo"
      echo "fsdp_sync=base_real_boundary_then_separate_pseudo_boundary"
      echo "trajectory_note=pseudo_gradients_necessarily_diverge_from_base_model_trajectory"
      ;;
    steps)
      echo "schedule_exposure_claim=configured_optimizer_steps_no_base_exposure_guarantee"
      ;;
    epochs)
      echo "schedule_exposure_claim=exact_union_epochs_no_base_exposure_guarantee"
      ;;
  esac
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
if [[ "$USE_HETEROGENEOUS_HELPER" == "1" ]]; then
  HETERO_LAUNCH_MANIFEST="$OUT/heterogeneous_launch.offset_${RANK_OFFSET}.txt" \
  WORLD_SIZE="$WORLD_SIZE" \
  RANK_OFFSET="$RANK_OFFSET" \
  LOCAL_NGPU="$LOCAL_NGPU" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  RANK_LAYOUT="$RANK_LAYOUT" \
  RANK_LAYOUT_FILE="$RANK_LAYOUT_FILE" \
  HETERO_LOG_DIR="$OUT/heterogeneous_rank_logs.offset_${RANK_OFFSET}" \
  bash script/run_heterogeneous_local_ranks.sh -- \
  "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.train_joint_rft \
  --config-name robotwin_train \
  --pseudo-jsonl "$PSEUDO_JSONL" \
  "${WAIVER_ARGS[@]}" \
  --split-manifest "$PREPARED_DATA_ROOT/split_manifest.json" \
  --expected-selection-mode "$RFT_SELECTION_MODE" \
  --real-fraction "$REAL_FRACTION" \
  --real-data-fraction "$REAL_DATA_FRACTION" \
  --data-fraction-seed "$DATA_FRACTION_SEED" \
  --mixing-mode "$MIXING_MODE" \
  --schedule-mode "$SCHEDULE_MODE" \
  --base-auxiliary-smoke-steps "$BASE_AUXILIARY_SMOKE_STEPS" \
  --base-real-regression-steps "$BASE_REAL_REGRESSION_STEPS" \
  "${BASE_AUXILIARY_ARGS[@]}" \
  "${REGRESSION_ARGS[@]}" \
  --num-epochs "$NUM_EPOCHS" \
  --real-data-mode "$REAL_DATA_MODE" \
  --real-chunk-mode "$REAL_CHUNK_MODE" \
  --pseudo-global-batch "$PSEUDO_GLOBAL_BATCH" \
  --pseudo-loss-weight "$PSEUDO_LOSS_WEIGHT" \
  --pseudo-loss-warmup-steps "$PSEUDO_LOSS_WARMUP_STEPS" \
  --pseudo-sampler-seed "$PSEUDO_SAMPLER_SEED" \
  --outer-step "$OUTER_STEP" \
  --swanlab-step-offset "$SWANLAB_STEP_OFFSET" \
  --save-root "$OUT" \
  --invocation-id "$RFT_INVOCATION_ID" \
  2>&1 | tee "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
else
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
  "${WAIVER_ARGS[@]}" \
  --split-manifest "$PREPARED_DATA_ROOT/split_manifest.json" \
  --expected-selection-mode "$RFT_SELECTION_MODE" \
  --real-fraction "$REAL_FRACTION" \
  --real-data-fraction "$REAL_DATA_FRACTION" \
  --data-fraction-seed "$DATA_FRACTION_SEED" \
  --mixing-mode "$MIXING_MODE" \
  --schedule-mode "$SCHEDULE_MODE" \
  --base-auxiliary-smoke-steps "$BASE_AUXILIARY_SMOKE_STEPS" \
  --base-real-regression-steps "$BASE_REAL_REGRESSION_STEPS" \
  "${BASE_AUXILIARY_ARGS[@]}" \
  "${REGRESSION_ARGS[@]}" \
  --num-epochs "$NUM_EPOCHS" \
  --real-data-mode "$REAL_DATA_MODE" \
  --real-chunk-mode "$REAL_CHUNK_MODE" \
  --pseudo-global-batch "$PSEUDO_GLOBAL_BATCH" \
  --pseudo-loss-weight "$PSEUDO_LOSS_WEIGHT" \
  --pseudo-loss-warmup-steps "$PSEUDO_LOSS_WARMUP_STEPS" \
  --pseudo-sampler-seed "$PSEUDO_SAMPLER_SEED" \
  --outer-step "$OUTER_STEP" \
  --swanlab-step-offset "$SWANLAB_STEP_OFFSET" \
  --save-root "$OUT" \
  --invocation-id "$RFT_INVOCATION_ID" \
  2>&1 | tee "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
fi
set -e
if (( NODE_RANK == 0 && rc == 0 )); then
  AUDIT_STEP="$NUM_STEPS"
  if [[ "$SCHEDULE_MODE" != "steps" ]]; then
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
