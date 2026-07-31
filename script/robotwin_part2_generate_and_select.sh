#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va/code}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$(dirname "$PROJECT_ROOT")}"
PART2_ROOT="${PART2_ROOT:-$LINGBOT_ROOT/train_out/critic/robotwin/part2_rft}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to M30 checkpoint_step_15000}"
VLAC_MODEL="${VLAC_MODEL:?Set VLAC_MODEL to the trained VLAC checkpoint}"
VLAC_ADAPTER="${VLAC_ADAPTER:-}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29621}"

cd "$PROJECT_ROOT"
test -x "$PYTHON"
test -s "$STAGE1_CHECKPOINT/transformer/config.json"
test -s "$PART2_ROOT/stage2_video_contexts.jsonl"
test -s "$PART2_ROOT/stage1_kinematic_profile.json"
test -s "$PART2_ROOT/stage2_chunk_budget.json"
mkdir -p "$PART2_ROOT"

export WAN_VA_MODEL_PATH="$STAGE1_CHECKPOINT"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NGPU" \
  --redirects=3 \
  --tee=3 \
  --master_port "$MASTER_PORT" \
  -m robotwin_critic.two_stage_rft.generate_wam_candidates \
  --contexts "$PART2_ROOT/stage2_video_contexts.jsonl" \
  --model "$STAGE1_CHECKPOINT" \
  --output-dir "$PART2_ROOT/wam_candidates" \
  --candidates-per-context "${CANDIDATES_PER_CONTEXT:-8}" \
  --resume \
  ${GENERATION_LIMIT_ARGS:-}

CUDA_VISIBLE_DEVICES="${DECODE_CUDA_DEVICE:-0}" \
"$PYTHON" -m robotwin_critic.two_stage_rft.decode_wam_candidates \
  --input "$PART2_ROOT/wam_candidates/candidates.jsonl" \
  --model "$STAGE1_CHECKPOINT" \
  --device cuda:0

"$PYTHON" -m robotwin_critic.two_stage_rft.score_action_chunks \
  --input "$PART2_ROOT/wam_candidates/candidates.jsonl" \
  --profile "$PART2_ROOT/stage1_kinematic_profile.json" \
  --output "$PART2_ROOT/wam_candidates_action_scored.jsonl" \
  --min-score "${MIN_ACTION_SCORE:-0.5}"

VLAC_ARGS=()
if [[ -n "$VLAC_ADAPTER" ]]; then
  VLAC_ARGS+=(--adapter "$VLAC_ADAPTER")
fi
"$PYTHON" -m robotwin_critic.two_stage_rft.score_vlac_candidates \
  --input "$PART2_ROOT/wam_candidates_action_scored.jsonl" \
  --model "$VLAC_MODEL" \
  --output "$PART2_ROOT/wam_candidates_dual_scored.jsonl" \
  --device "${VLAC_DEVICE:-cuda:0}" \
  --batch-size "${VLAC_BATCH_SIZE:-4}" \
  "${VLAC_ARGS[@]}"

for mode in naive process action dual; do
  "$PYTHON" -m robotwin_critic.two_stage_rft.select_rft_candidates \
    --input "$PART2_ROOT/wam_candidates_dual_scored.jsonl" \
    --budget "$PART2_ROOT/stage2_chunk_budget.json" \
    --output "$PART2_ROOT/${mode}_rft_selected.jsonl" \
    --mode "$mode" \
    --min-action-score "${MIN_ACTION_SCORE:-0.5}"
done

echo "ROBOTWIN_PART2_SELECTION_OK"
