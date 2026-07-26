#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/lingbot-va}"
MODEL_PATH="${MODEL_PATH:-/data/lingbot-va/models/vlac/VLAC-8B}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/full}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/train_out/critic/robotwin/vlac_finetune/vlac_8b_full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE

command -v swift >/dev/null || {
  echo "ms-swift CLI is missing; install the VLAC requirement ms-swift>=3.3" >&2
  exit 1
}
[[ -d "$MODEL_PATH" ]] || {
  echo "VLAC-8B model is missing or gated access has not been granted: $MODEL_PATH" >&2
  exit 1
}
[[ -s "$DATA_DIR/train.jsonl" ]] || { echo "Full training manifest is missing" >&2; exit 1; }
[[ -s "$DATA_DIR/val.jsonl" ]] || { echo "Full validation manifest is missing" >&2; exit 1; }

exec swift sft \
  --model "$MODEL_PATH" \
  --model_type internvl2 \
  --template internvl2 \
  --train_type full \
  --dataset "$DATA_DIR/train.jsonl" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --output_dir "$OUTPUT_DIR" \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --lr_scheduler_type cosine \
  --warmup_steps 100 \
  --max_length 4096 \
  --num_train_epochs 1 \
  --eval_strategy steps \
  --eval_steps 250 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --gradient_checkpointing true \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --deepspeed zero3 \
  --report_to tensorboard \
  --seed 42

