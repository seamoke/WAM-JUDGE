#!/usr/bin/env bash
set -euo pipefail

# Drift-control experiment: continue the Stage-1 15k model on the union of
# Stage-1 and Stage-2 real data, with the pseudo stream fully disabled.
RUN_ID="${RUN_ID:-stage1_stage2_real_only_from_stage1_15k_finalonly_8xmi355_$(date -u +%Y%m%dT%H%M%SZ)}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$HIP_VISIBLE_DEVICES}"
export PYTHONUNBUFFERED=1
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/lingbot-training}"
export LINGBOT_ROOT="${LINGBOT_ROOT:-/workspace/lingbot-va}"
export PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$PROJECT_ROOT/datasets/robotwin-clean-aug-two-stage-seed42}"
export WAM_PYTHON="${WAM_PYTHON:-$PROJECT_ROOT/.runtime/.venv/bin/python}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$LINGBOT_ROOT/models/lingbot-va-stage1-checkpoints/checkpoint_step_15000}"
# The generic launcher validates the artifact contract at startup even when its
# loss weight is zero. The training step itself performs no pseudo forward pass.
export PSEUDO_JSONL="${PSEUDO_JSONL:-$PROJECT_ROOT/datasets/robotwin-stage2-oneshot-pseudo-chunks/one_shot_pseudo_buffer.validated.jsonl}"

export RUN_ID
export OUT="${OUT:-$LINGBOT_ROOT/train_out/robotwin/$RUN_ID}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export LOCAL_NGPU="${LOCAL_NGPU:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29741}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LINGBOT_DATASET_POOL_DEFER_EMPTY_EMB=1

export RFT_SCHEDULE_MODE=base-auxiliary-pseudo
export RFT_BASE_AUXILIARY_STEPS="${RFT_BASE_AUXILIARY_STEPS:-15000}"
export RFT_BASE_AUXILIARY_SAVE_INTERVAL="${RFT_BASE_AUXILIARY_SAVE_INTERVAL:-15000}"
export RFT_BASE_AUXILIARY_ACTIVATION_CHECKPOINTING=0
export RFT_BASE_AUXILIARY_BATCH_SIZE_PER_GPU="${RFT_BASE_AUXILIARY_BATCH_SIZE_PER_GPU:-1}"
export REAL_DATA_MODE=stage1-stage2
export REAL_CHUNK_MODE=full
export REAL_DATA_FRACTION=1.0
export TARGET_GLOBAL_BATCH=64
export PSEUDO_GLOBAL_BATCH=64
export PSEUDO_LOSS_WEIGHT=0
export PSEUDO_LOSS_WARMUP_STEPS=0
export DATASET_INIT_WORKERS="${DATASET_INIT_WORKERS:-128}"
export TRAIN_LOAD_WORKERS="${TRAIN_LOAD_WORKERS:-8}"
export LEGACY_PSEUDO_ACTION_WAIVER_SHA256="${LEGACY_PSEUDO_ACTION_WAIVER_SHA256:-a330e3004fbb9eb30213751d981cfd908971d3baa8806e035cad296adc43bd39}"
export LEGACY_PSEUDO_ACTION_WAIVER_ROWS="${LEGACY_PSEUDO_ACTION_WAIVER_ROWS:-20477}"

export ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
export LINGBOT_SWANLAB_MODE="${LINGBOT_SWANLAB_MODE:-online}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
export LINGBOT_SWANLAB_PROJECT="${LINGBOT_SWANLAB_PROJECT:-lingbot-va-robotwin}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$RUN_ID}"

exec bash "$PROJECT_ROOT/script/run_robotwin_joint_rft.sh"
