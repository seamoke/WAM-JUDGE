#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

LOCAL_CONFIG="${ROBOTWIN_STAGE2_RFT_CONFIG:-$PROJECT_ROOT/.local/robotwin_stage2_rft.env}"
if [[ -s "$LOCAL_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_CONFIG"
fi

STAGE2_DATA_ROOT="${STAGE2_DATA_ROOT:-}"
ORIGINAL_ROBOTWIN_ROOT="${ORIGINAL_ROBOTWIN_ROOT:-}"
REAL_DATA_ROOT="${REAL_DATA_ROOT:-}"
REAL_DATA_LINK_MODE="${REAL_DATA_LINK_MODE:-hardlink}"
REAL_DATA_MODE="${REAL_DATA_MODE:-stage1-stage2-visible}"
WAM_MODEL="${WAM_MODEL:-}"
VLAC_MODEL="${VLAC_MODEL:-}"
BASE_MODEL="${BASE_MODEL:-$LINGBOT_ROOT/models/lingbot-va-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
ONLINE_ROOT="${ONLINE_ROOT:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-lingbot-va-robotwin}"
SWANLAB_GROUP="${SWANLAB_GROUP:-robotwin-stage1-real-stage2-pseudo}"
SWANLAB_NAME="${SWANLAB_NAME:-robotwin-stage2-online-dual-rft}"
SWANLAB_API_KEY_FILE="${SWANLAB_API_KEY_FILE:-$PROJECT_ROOT/.secrets/swanlab_api_key}"
EXPECTED_PER_DOMAIN_TOTAL="${EXPECTED_PER_DOMAIN_TOTAL:-50}"
EXPECTED_STAGE1_PER_DOMAIN="${EXPECTED_STAGE1_PER_DOMAIN:-30}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-1024}"
Q_PER_ROUND="${Q_PER_ROUND:-128}"
CANDIDATES_PER_Q="${CANDIDATES_PER_Q:-8}"
INFERENCE_BATCH_SIZE_PER_GPU="${INFERENCE_BATCH_SIZE_PER_GPU:-2}"
VLAC_BATCH_SIZE_PER_GPU="${VLAC_BATCH_SIZE_PER_GPU:-4}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-8}"
TRAIN_GLOBAL_BATCH="${TRAIN_GLOBAL_BATCH:-32}"
PSEUDO_EPOCHS_PER_UPDATE="${PSEUDO_EPOCHS_PER_UPDATE:-3}"
REAL_FRACTION="${REAL_FRACTION:-0.5}"
MAX_UPDATES="${MAX_UPDATES:-1000}"
MODEL_SAVE_EVERY_UPDATES="${MODEL_SAVE_EVERY_UPDATES:-50}"
MIN_ACTION_SCORE="${MIN_ACTION_SCORE:-0.75}"
MIN_PROCESS_SCORE="${MIN_PROCESS_SCORE:-5.0}"
MAX_PSEUDO_PER_CONTEXT="${MAX_PSEUDO_PER_CONTEXT:-0}"
HISTORY_FRAMES="${HISTORY_FRAMES:-4}"
CONTEXT_POOL_MULTIPLIER="${CONTEXT_POOL_MULTIPLIER:-2.0}"
MAX_EPISODE_FRAMES="${MAX_EPISODE_FRAMES:-500}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
FRESH="${FRESH:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash script/run_robotwin_stage2_online_rft_pipeline.sh \
    --stage2-data-root /path/to/prepared-dataset/stage2 \
    --wam-model /path/to/checkpoint_step_15000 \
    --vlac-model /path/to/vlac-checkpoint \
    [--output-root /path/to/output]

Required:
  --stage2-data-root  Prepared split root or its stage2/ directory. Its sibling
                      stage1/ must contain the action-labeled calibration data.
  --original-robotwin-root PATH
                      Original RoboTwin root used to restore the selected
                      Stage-2 action labels. Optional if manifest path exists.
  --wam-model         Complete WAM root, transformer checkpoint root, or the
                      transformer/ directory itself.
  --vlac-model        Trained VLAC checkpoint containing config.json.

Common options:
  --base-model PATH   Complete WAM skeleton used with transformer-only checkpoints.
  --real-data-root PATH
                      Existing selected action-visible replay root. If absent,
                      it is built at PREPARED_ROOT/action_visible_real.
  --real-data-link-mode hardlink|copy|symlink (use copy before migration).
  --output-root PATH  Run directory. Defaults to a timestamped train_out path.
  --gpu-ids IDS       Comma-separated GPU IDs (default: 0,1,2,3).
  --fresh             Delete OUTPUT_ROOT before starting. Never implied.
  --prepare-only      Build Action Critic and Stage-2 contexts, then exit.
  --buffer-capacity N --q-per-round N --candidates-per-q N
  --train-global-batch N --real-fraction X --max-updates N
  --model-save-every-updates N
  --swanlab-project NAME --swanlab-group NAME --swanlab-name NAME
  --swanlab-api-key-file PATH

With no arguments, values are read from:
  code/.local/robotwin_stage2_rft.env
Override this path with ROBOTWIN_STAGE2_RFT_CONFIG.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage2-data-root) require_value "$@"; STAGE2_DATA_ROOT="$2"; shift 2 ;;
    --original-robotwin-root) require_value "$@"; ORIGINAL_ROBOTWIN_ROOT="$2"; shift 2 ;;
    --real-data-root) require_value "$@"; REAL_DATA_ROOT="$2"; shift 2 ;;
    --real-data-link-mode) require_value "$@"; REAL_DATA_LINK_MODE="$2"; shift 2 ;;
    --wam-model) require_value "$@"; WAM_MODEL="$2"; shift 2 ;;
    --vlac-model) require_value "$@"; VLAC_MODEL="$2"; shift 2 ;;
    --base-model) require_value "$@"; BASE_MODEL="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu-ids) require_value "$@"; GPU_IDS="$2"; shift 2 ;;
    --buffer-capacity) require_value "$@"; BUFFER_CAPACITY="$2"; shift 2 ;;
    --q-per-round) require_value "$@"; Q_PER_ROUND="$2"; shift 2 ;;
    --candidates-per-q) require_value "$@"; CANDIDATES_PER_Q="$2"; shift 2 ;;
    --inference-batch-size-per-gpu) require_value "$@"; INFERENCE_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
    --vlac-batch-size-per-gpu) require_value "$@"; VLAC_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
    --train-batch-size-per-gpu) require_value "$@"; TRAIN_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
    --train-global-batch) require_value "$@"; TRAIN_GLOBAL_BATCH="$2"; shift 2 ;;
    --pseudo-epochs-per-update) require_value "$@"; PSEUDO_EPOCHS_PER_UPDATE="$2"; shift 2 ;;
    --real-fraction) require_value "$@"; REAL_FRACTION="$2"; shift 2 ;;
    --max-updates) require_value "$@"; MAX_UPDATES="$2"; shift 2 ;;
    --model-save-every-updates) require_value "$@"; MODEL_SAVE_EVERY_UPDATES="$2"; shift 2 ;;
    --min-action-score) require_value "$@"; MIN_ACTION_SCORE="$2"; shift 2 ;;
    --min-process-score) require_value "$@"; MIN_PROCESS_SCORE="$2"; shift 2 ;;
    --max-pseudo-per-context) require_value "$@"; MAX_PSEUDO_PER_CONTEXT="$2"; shift 2 ;;
    --expected-per-domain-total) require_value "$@"; EXPECTED_PER_DOMAIN_TOTAL="$2"; shift 2 ;;
    --expected-stage1-per-domain) require_value "$@"; EXPECTED_STAGE1_PER_DOMAIN="$2"; shift 2 ;;
    --history-frames) require_value "$@"; HISTORY_FRAMES="$2"; shift 2 ;;
    --context-pool-multiplier) require_value "$@"; CONTEXT_POOL_MULTIPLIER="$2"; shift 2 ;;
    --max-episode-frames) require_value "$@"; MAX_EPISODE_FRAMES="$2"; shift 2 ;;
    --swanlab-project) require_value "$@"; SWANLAB_PROJECT="$2"; shift 2 ;;
    --swanlab-group) require_value "$@"; SWANLAB_GROUP="$2"; shift 2 ;;
    --swanlab-name) require_value "$@"; SWANLAB_NAME="$2"; shift 2 ;;
    --swanlab-api-key-file) require_value "$@"; SWANLAB_API_KEY_FILE="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$STAGE2_DATA_ROOT" || -z "$WAM_MODEL" || -z "$VLAC_MODEL" ]]; then
  usage >&2
  exit 2
fi

cd "$PROJECT_ROOT"
test -x "$PYTHON"
STAGE2_DATA_ROOT="$(realpath "$STAGE2_DATA_ROOT")"
WAM_MODEL="$(realpath "$WAM_MODEL")"
VLAC_MODEL="$(realpath "$VLAC_MODEL")"
if [[ -e "$BASE_MODEL" ]]; then
  BASE_MODEL="$(realpath "$BASE_MODEL")"
fi

if [[ -s "$STAGE2_DATA_ROOT/split_manifest.json" ]]; then
  PREPARED_DATA_ROOT="$STAGE2_DATA_ROOT"
  STAGE2_DATA_ROOT="$PREPARED_DATA_ROOT/stage2"
elif [[ "$(basename "$STAGE2_DATA_ROOT")" == "stage2" && -s "$(dirname "$STAGE2_DATA_ROOT")/split_manifest.json" ]]; then
  PREPARED_DATA_ROOT="$(dirname "$STAGE2_DATA_ROOT")"
else
  echo "--stage2-data-root must be a prepared split root or its stage2/ directory" >&2
  exit 2
fi

test -d "$PREPARED_DATA_ROOT/stage1"
test -d "$STAGE2_DATA_ROOT"
test -s "$PREPARED_DATA_ROOT/split_manifest.json"
test -s "$PREPARED_DATA_ROOT/PREPARATION_COMPLETE.json"
test -s "$VLAC_MODEL/config.json"

if [[ -z "$REAL_DATA_ROOT" ]]; then
  REAL_DATA_ROOT="$PREPARED_DATA_ROOT/action_visible_real"
fi
REAL_DATA_ROOT="$(realpath -m "$REAL_DATA_ROOT")"

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$LINGBOT_ROOT/train_out/critic/robotwin/stage2_online_rft_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
if [[ "$FRESH" == "1" && -e "$OUTPUT_ROOT" ]]; then
  case "$OUTPUT_ROOT" in
    /|"$PROJECT_ROOT"|"$LINGBOT_ROOT"|"$PREPARED_DATA_ROOT"|"$STAGE2_DATA_ROOT"|"$REAL_DATA_ROOT"|"$WAM_MODEL"|"$VLAC_MODEL")
      echo "Refusing to delete protected path: $OUTPUT_ROOT" >&2
      exit 2
      ;;
  esac
  rm -rf -- "$OUTPUT_ROOT"
fi
mkdir -p "$OUTPUT_ROOT/part2"
PART2_ROOT="$OUTPUT_ROOT/part2"
MASTER_LOG="$OUTPUT_ROOT/stage2_online_rft.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "STAGE2_PIPELINE_START $(date -Is)"
echo "prepared_data_root=$PREPARED_DATA_ROOT"
echo "stage2_data_root=$STAGE2_DATA_ROOT"
echo "wam_model=$WAM_MODEL"
echo "vlac_model=$VLAC_MODEL"
echo "real_data_root=$REAL_DATA_ROOT"
echo "output_root=$OUTPUT_ROOT"

if [[ "$REAL_DATA_MODE" == "stage1-stage2-visible" ]]; then
  if [[ ! -s "$REAL_DATA_ROOT/ACTION_VISIBLE_COMPLETE.json" ]]; then
    source_args=()
    if [[ -n "$ORIGINAL_ROBOTWIN_ROOT" ]]; then
      source_args+=(--source-root "$ORIGINAL_ROBOTWIN_ROOT")
    fi
    "$PYTHON" -m robotwin_critic.two_stage_rft.prepare_action_visible_real \
      --prepared-root "$PREPARED_DATA_ROOT" \
      --output-root "$REAL_DATA_ROOT" \
      --link-mode "$REAL_DATA_LINK_MODE" \
      "${source_args[@]}"
  fi
  "$PYTHON" -m robotwin_critic.two_stage_rft.prepare_action_visible_real \
    --prepared-root "$PREPARED_DATA_ROOT" \
    --output-root "$REAL_DATA_ROOT" \
    --verify-only
fi

ACTION_PROFILE="$PART2_ROOT/stage1_kinematic_profile.json"
CONTEXTS="$PART2_ROOT/stage2_video_contexts.jsonl"
BUDGET="$PART2_ROOT/stage2_chunk_budget.json"

if [[ ! -s "$ACTION_PROFILE" ]]; then
  "$PYTHON" -m robotwin_critic.two_stage_rft.calibrate_action_critic \
    --prepared-root "$PREPARED_DATA_ROOT" \
    --output "$ACTION_PROFILE" \
    --soft-quantile 0.99 \
    --hard-quantile 0.999 \
    --minimum-score "$MIN_ACTION_SCORE" \
    --expected-per-domain-total "$EXPECTED_PER_DOMAIN_TOTAL" \
    --expected-stage1-per-domain "$EXPECTED_STAGE1_PER_DOMAIN"
fi

if [[ ! -s "$CONTEXTS" ]]; then
  "$PYTHON" -m robotwin_critic.two_stage_rft.build_video_contexts \
    --prepared-root "$PREPARED_DATA_ROOT" \
    --output "$CONTEXTS" \
    --history-frames "$HISTORY_FRAMES" \
    --max-episode-frames "$MAX_EPISODE_FRAMES" \
    --context-pool-multiplier "$CONTEXT_POOL_MULTIPLIER" \
    --expected-per-domain-total "$EXPECTED_PER_DOMAIN_TOTAL" \
    --expected-stage1-per-domain "$EXPECTED_STAGE1_PER_DOMAIN"
fi

if [[ ! -s "$BUDGET" ]]; then
  "$PYTHON" -m robotwin_critic.two_stage_rft.count_pseudo_budget \
    --prepared-root "$PREPARED_DATA_ROOT" \
    --output "$BUDGET" \
    --max-episode-frames "$MAX_EPISODE_FRAMES" \
    --expected-per-domain-total "$EXPECTED_PER_DOMAIN_TOTAL" \
    --expected-stage1-per-domain "$EXPECTED_STAGE1_PER_DOMAIN"
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "STAGE2_PREPARE_ONLY_DONE action_profile=$ACTION_PROFILE contexts=$CONTEXTS"
  exit 0
fi

ONLINE_ROOT="${ONLINE_ROOT:-$OUTPUT_ROOT/online}"
ONLINE_ROOT="$(realpath -m "$ONLINE_ROOT")"
mkdir -p "$ONLINE_ROOT"
if [[ -s "$WAM_MODEL/vae/config.json" && -s "$WAM_MODEL/transformer/config.json" ]]; then
  INITIAL_MODEL="$WAM_MODEL"
else
  if [[ -s "$WAM_MODEL/transformer/config.json" ]]; then
    WAM_TRANSFORMER="$WAM_MODEL/transformer"
  elif [[ -s "$WAM_MODEL/config.json" && "$(basename "$WAM_MODEL")" == "transformer" ]]; then
    WAM_TRANSFORMER="$WAM_MODEL"
  else
    echo "WAM model is neither a complete root nor a transformer checkpoint: $WAM_MODEL" >&2
    exit 2
  fi
  test -s "$BASE_MODEL/vae/config.json"
  test -s "$BASE_MODEL/transformer/config.json"
  INITIAL_MODEL="$ONLINE_ROOT/initial_model"
  if [[ ! -s "$INITIAL_MODEL/online_rft_model.json" ]]; then
    "$PYTHON" -m robotwin_critic.two_stage_rft.stage_updated_model \
      --base-model "$BASE_MODEL" \
      --transformer "$WAM_TRANSFORMER" \
      --output "$INITIAL_MODEL"
  fi
fi

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  test -s "$SWANLAB_API_KEY_FILE"
  export SWANLAB_API_KEY="$(tr -d '[:space:]' < "$SWANLAB_API_KEY_FILE")"
fi

export PROJECT_ROOT LINGBOT_ROOT PREPARED_DATA_ROOT PART2_ROOT ONLINE_ROOT
export REAL_DATA_ROOT REAL_DATA_MODE
export INITIAL_MODEL VLAC_MODEL
export CONTEXTS ACTION_PROFILE
export INFER_GPU_IDS="$GPU_IDS"
export INFER_BATCH_SIZE_PER_GPU="$INFERENCE_BATCH_SIZE_PER_GPU"
export Q_PER_ROUND INFER_BATCH_SIZE_PER_GPU CANDIDATES_PER_Q VLAC_BATCH_SIZE_PER_GPU
export BUFFER_CAPACITY TRAIN_BATCH_SIZE_PER_GPU TRAIN_GLOBAL_BATCH
export PSEUDO_EPOCHS_PER_UPDATE REAL_FRACTION MAX_UPDATES MODEL_SAVE_EVERY_UPDATES
export MIN_ACTION_SCORE MIN_PROCESS_SCORE MAX_PSEUDO_PER_CONTEXT
export ACTION_GATE_POLICY=score_with_safety_gates
export ACTION_WORKSPACE_SCOPE=global

echo "STAGE2_RFT_START $(date -Is)"
echo "action_profile=$ACTION_PROFILE"
echo "contexts=$CONTEXTS"
echo "online_root=$ONLINE_ROOT"

exec "$PYTHON" -m robotwin_critic.two_stage_rft.run_online_rft_swanlab \
  --online-root "$ONLINE_ROOT" \
  --project "$SWANLAB_PROJECT" \
  --group "$SWANLAB_GROUP" \
  --name "$SWANLAB_NAME" \
  --log-dir "$ONLINE_ROOT/swanlab" \
  --max-images-per-collect 4 \
  -- bash "$PROJECT_ROOT/script/run_robotwin_online_dual_rft.sh"
