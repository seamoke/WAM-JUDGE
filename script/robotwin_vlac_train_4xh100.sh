#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
CRITIC_ROOT="${CRITIC_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin}"
VENV_DIR="${VENV_DIR:-$CRITIC_ROOT/envs/vlac}"
MODEL_PATH="${MODEL_PATH:-$CRITIC_ROOT/models/VLAC-2B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
RUN_EVAL="${RUN_EVAL:-1}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

case "$MODE" in
  smoke)
    DATA_DIR="${DATA_DIR:-$CRITIC_ROOT/vlac_finetune/smoke_2task}"
    OUTPUT_DIR="${OUTPUT_DIR:-$CRITIC_ROOT/vlac_finetune/vlac_2b_full_smoke_4xh100}"
    MAX_STEPS="${MAX_STEPS:-10}"
    EVAL_STEPS="${EVAL_STEPS:-10}"
    SAVE_STEPS="${SAVE_STEPS:-10}"
    TRAIN_LENGTH_ARGS=(--max_steps "$MAX_STEPS")
    ;;
  full)
    DATA_DIR="${DATA_DIR:-$CRITIC_ROOT/vlac_finetune/full}"
    OUTPUT_DIR="${OUTPUT_DIR:-$CRITIC_ROOT/vlac_finetune/vlac_2b_full_4xh100}"
    EVAL_STEPS="${EVAL_STEPS:-1000}"
    SAVE_STEPS="${SAVE_STEPS:-1000}"
    if [[ -n "${MAX_STEPS:-}" ]]; then
      TRAIN_LENGTH_ARGS=(--max_steps "$MAX_STEPS")
    else
      NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
      TRAIN_LENGTH_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
    fi
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

PYTHON="$VENV_DIR/bin/python"
VAL_DATASET="$DATA_DIR/val_train.jsonl"
[[ -s "$VAL_DATASET" ]] || VAL_DATASET="$DATA_DIR/val.jsonl"
[[ -x "$PYTHON" ]] || { echo "VLAC Python is missing: $PYTHON" >&2; exit 1; }
[[ -s "$MODEL_PATH/config.json" ]] || { echo "VLAC model is incomplete: $MODEL_PATH" >&2; exit 1; }
[[ -s "$DATA_DIR/train.jsonl" ]] || { echo "Training manifest is missing: $DATA_DIR/train.jsonl" >&2; exit 1; }
[[ -s "$VAL_DATASET" ]] || { echo "Validation manifest is missing: $VAL_DATASET" >&2; exit 1; }

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$GPU_COUNT" -ne 4 ]]; then
  echo "Expected exactly four visible GPUs, found $GPU_COUNT" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; then
  echo "Refusing to start: at least one GPU compute process is already running." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2 || true
  exit 3
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

if [[ "$MODE" == "smoke" && "$RUN_BASELINE_EVAL" == "1" ]]; then
  BASELINE_EVAL_DIR="$CRITIC_ROOT/vlac_finetune/vlac_2b_eval_baseline_4xh100"
  "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.evaluate_vlac \
    --model "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --output-dir "$BASELINE_EVAL_DIR" \
    --device cuda:0 \
    --batch-size "${EVAL_BATCH_SIZE:-4}" \
    --max-pairs "${MAX_EVAL_PAIRS:-64}" \
    --max-trajectories "${MAX_EVAL_TRAJECTORIES:-4}"
fi

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "${MASTER_PORT:-29537}" \
  -m robotwin_critic.vlac_finetune.sft_compat \
  --model "$MODEL_PATH" \
  --model_type internvl2 \
  --template internvl2 \
  --train_type full \
  --dataset "$DATA_DIR/train.jsonl" \
  --val_dataset "$VAL_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --torch_dtype bfloat16 \
  --freeze_vit false \
  --freeze_aligner false \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length 4096 \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --logging_steps 1 \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --ddp_find_unused_parameters false \
  --dataloader_num_workers 4 \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --report_to tensorboard \
  --seed 42

if [[ "$MODE" == "smoke" && "$RUN_EVAL" == "1" ]]; then
  SMOKE_MODEL="$(
    find "$OUTPUT_DIR" -type f -path '*/checkpoint-*/config.json' -printf '%h\n' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "$SMOKE_MODEL" ]]; then
    echo "Smoke training finished but no full-model checkpoint was found under $OUTPUT_DIR" >&2
    exit 4
  fi
  EVAL_DIR="${EVAL_DIR:-$CRITIC_ROOT/vlac_finetune/vlac_2b_eval_full_smoke_4xh100}"
  "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.evaluate_vlac \
    --model "$SMOKE_MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$EVAL_DIR" \
    --device cuda:0 \
    --batch-size "${EVAL_BATCH_SIZE:-4}" \
    --max-pairs "${MAX_EVAL_PAIRS:-64}" \
    --max-trajectories "${MAX_EVAL_TRAJECTORIES:-4}"
  "$VENV_DIR/bin/python" - "$EVAL_DIR/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    summary = json.load(handle)
if not summary.get("smoke_passed", False):
    raise SystemExit(f"VLAC smoke gates failed: {summary.get('smoke_gates')}")
print("VLAC smoke gates passed:", json.dumps(summary["smoke_gates"], sort_keys=True))
PY
  BASELINE_SUMMARY="$CRITIC_ROOT/vlac_finetune/vlac_2b_eval_baseline_4xh100/summary.json"
  if [[ -s "$BASELINE_SUMMARY" ]]; then
    "$VENV_DIR/bin/python" - \
      "$BASELINE_SUMMARY" \
      "$EVAL_DIR/summary.json" \
      "$EVAL_DIR/comparison_vs_baseline.json" <<'PY'
import json
import sys

metrics = (
    "pair_numeric_parse_rate",
    "pair_mae",
    "pair_sign_accuracy",
    "pair_accuracy",
    "pair_macro_f1",
    "pair_macro_ovr_auc",
    "pair_target_spearman",
    "mean_voc",
    "mean_vroc",
    "mean_voc_f1",
    "mean_antisymmetry_mae",
    "trajectory_numeric_rate",
)
with open(sys.argv[1]) as handle:
    baseline = json.load(handle)
with open(sys.argv[2]) as handle:
    finetuned = json.load(handle)
comparison = {
    "baseline_summary": sys.argv[1],
    "finetuned_summary": sys.argv[2],
    "metrics": {
        metric: {
            "baseline": baseline[metric],
            "finetuned": finetuned[metric],
            "delta": finetuned[metric] - baseline[metric],
        }
        for metric in metrics
    },
}
with open(sys.argv[3], "w") as handle:
    json.dump(comparison, handle, indent=2)
print("VLAC baseline comparison:", json.dumps(comparison["metrics"], sort_keys=True))
PY
  fi
fi
