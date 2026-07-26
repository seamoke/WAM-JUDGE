#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/lingbot-va}"
MODEL_PATH="${MODEL_PATH:-/data/lingbot-va/models/vlac/VLAC-2B}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/smoke_2task}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/vlac_2b_lora_smoke}"
MAX_STEPS="${MAX_STEPS:-10}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

command -v swift >/dev/null || {
  echo "ms-swift CLI is missing; install the VLAC requirement ms-swift>=3.3" >&2
  exit 1
}
[[ -d "$MODEL_PATH" ]] || { echo "VLAC-2B model is missing: $MODEL_PATH" >&2; exit 1; }
[[ -s "$DATA_DIR/train.jsonl" ]] || { echo "Training manifest is missing" >&2; exit 1; }
[[ -s "$DATA_DIR/val.jsonl" ]] || { echo "Validation manifest is missing" >&2; exit 1; }

exec swift sft \
  --model "$MODEL_PATH" \
  --model_type internvl2 \
  --template internvl2 \
  --train_type lora \
  --dataset "$DATA_DIR/train.jsonl" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --output_dir "$OUTPUT_DIR" \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length 4096 \
  --max_steps "$MAX_STEPS" \
  --eval_strategy steps \
  --eval_steps "$MAX_STEPS" \
  --save_strategy steps \
  --save_steps "$MAX_STEPS" \
  --save_total_limit 2 \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2 \
  --report_to tensorboard \
  --seed 42

