#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-0}"
RUN_DIR="${2:-}"
LINGBOT_ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
PROJECT_ROOT="${PROJECT_ROOT:-$LINGBOT_ROOT/code}"
CRITIC_ROOT="${CRITIC_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin}"
VENV_DIR="${VENV_DIR:-$CRITIC_ROOT/envs/vlac}"
MODEL_PATH="${MODEL_PATH:-$CRITIC_ROOT/models/VLAC-2B}"
DATA_DIR="${DATA_DIR:-$CRITIC_ROOT/vlac_finetune/full}"
EVAL_ROOT="${EVAL_ROOT:-$CRITIC_ROOT/vlac_finetune}"
MAX_EVAL_PAIRS="${MAX_EVAL_PAIRS:-4096}"
MAX_EVAL_TRAJECTORIES="${MAX_EVAL_TRAJECTORIES:-256}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

[[ -x "$VENV_DIR/bin/python" ]] || {
  echo "VLAC Python is missing: $VENV_DIR/bin/python" >&2
  exit 1
}
[[ -s "$MODEL_PATH/config.json" ]] || {
  echo "VLAC base model is incomplete: $MODEL_PATH" >&2
  exit 1
}
[[ -s "$DATA_DIR/val.jsonl" ]] || {
  echo "Validation manifest is missing: $DATA_DIR/val.jsonl" >&2
  exit 1
}

if [[ "$TRAIN_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Waiting for training PID $TRAIN_PID"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
  done
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(
    find "$CRITIC_ROOT/vlac_finetune/vlac_2b_full_4xh100" \
      -mindepth 1 -maxdepth 1 -type d -name 'v0-*' -printf '%T@ %p\n' \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )"
fi
[[ -d "$RUN_DIR" ]] || {
  echo "Training run directory is missing: $RUN_DIR" >&2
  exit 2
}

LATEST_CHECKPOINT="$(
  find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    -printf '%f %p\n' \
    | sort -t- -k2,2n \
    | tail -n 1 \
    | cut -d' ' -f2-
)"
[[ -s "$LATEST_CHECKPOINT/trainer_state.json" ]] || {
  echo "No completed checkpoint found under $RUN_DIR" >&2
  exit 3
}

read -r GLOBAL_STEP MAX_STEPS BEST_CHECKPOINT < <(
  "$VENV_DIR/bin/python" - "$LATEST_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
with (checkpoint / "trainer_state.json").open() as handle:
    state = json.load(handle)
best = state.get("best_model_checkpoint") or str(checkpoint)
print(state["global_step"], state["max_steps"], best)
PY
)
if (( GLOBAL_STEP < MAX_STEPS )); then
  echo "Training did not finish: global_step=$GLOBAL_STEP max_steps=$MAX_STEPS" >&2
  exit 4
fi
[[ -s "$BEST_CHECKPOINT/model.safetensors" ]] || {
  echo "Best full-model checkpoint is incomplete: $BEST_CHECKPOINT" >&2
  exit 5
}

cd "$PROJECT_ROOT"
COMMON_EVAL_ARGS=(
  --data-dir "$DATA_DIR"
  --device cuda:0
  --batch-size 4
  --max-pairs "$MAX_EVAL_PAIRS"
  --max-trajectories "$MAX_EVAL_TRAJECTORIES"
  --sampling stratified
  --sample-seed "$SAMPLE_SEED"
)
BASELINE_DIR="$EVAL_ROOT/vlac_2b_eval_baseline_full_stratified"
FINAL_DIR="$EVAL_ROOT/vlac_2b_eval_full_4xh100"

if [[ ! -s "$BASELINE_DIR/summary.json" ]]; then
  "$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.evaluate_vlac \
    --model "$MODEL_PATH" \
    --output-dir "$BASELINE_DIR" \
    "${COMMON_EVAL_ARGS[@]}"
fi
"$VENV_DIR/bin/python" -m robotwin_critic.vlac_finetune.evaluate_vlac \
  --model "$BEST_CHECKPOINT" \
  --output-dir "$FINAL_DIR" \
  "${COMMON_EVAL_ARGS[@]}"

"$VENV_DIR/bin/python" - \
  "$BASELINE_DIR/summary.json" \
  "$FINAL_DIR/summary.json" \
  "$FINAL_DIR/comparison_vs_baseline.json" \
  "$SAMPLE_SEED" <<'PY'
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
    "sample_seed": int(sys.argv[4]),
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
print(json.dumps(comparison, indent=2))
PY

echo "Post-training evaluation completed: $FINAL_DIR"
