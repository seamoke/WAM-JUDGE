#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT}"
MULTINODE_QUEUE_ROOT="${MULTINODE_QUEUE_ROOT:?Set MULTINODE_QUEUE_ROOT}"
TRAIN_NNODES="${TRAIN_NNODES:-2}"
TRAIN_LOCAL_NGPU="${TRAIN_LOCAL_NGPU:-2}"
TRAIN_MASTER_ADDR="${TRAIN_MASTER_ADDR:?Set TRAIN_MASTER_ADDR to the master Pod IP}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29641}"
WAM_PYTHON="${WAM_PYTHON:-$PROJECT_ROOT/.runtime/.venv/bin/python}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

if (( TRAIN_NNODES != 2 )); then
  echo "The shared-queue launcher currently requires TRAIN_NNODES=2" >&2
  exit 2
fi

job_id="train_${RUN_ID:-update}_${RFT_OUTER_STEP:-0}_$(date +%s)_$$"
worker_env=(
  NNODES="$TRAIN_NNODES"
  NODE_RANK=1
  LOCAL_NGPU="$TRAIN_LOCAL_NGPU"
  MASTER_ADDR="$TRAIN_MASTER_ADDR"
  MASTER_PORT="$TRAIN_MASTER_PORT"
  CUDA_VISIBLE_DEVICES="${TRAIN_REMOTE_GPU_IDS:-0,1}"
)

env "${worker_env[@]}" "$WAM_PYTHON" \
  -m robotwin_critic.two_stage_rft.multinode_worker submit \
  --queue-root "$MULTINODE_QUEUE_ROOT" \
  --job-id "$job_id" \
  --cwd "$PROJECT_ROOT" \
  -- bash script/run_robotwin_joint_rft.sh

started="$MULTINODE_QUEUE_ROOT/started/$job_id.json"
deadline=$(( $(date +%s) + 300 ))
while [[ ! -s "$started" ]]; do
  if (( $(date +%s) >= deadline )); then
    echo "Timed out waiting for remote training node: $job_id" >&2
    exit 1
  fi
  sleep 1
done

master_rc=0
env \
  NNODES="$TRAIN_NNODES" \
  NODE_RANK=0 \
  LOCAL_NGPU="$TRAIN_LOCAL_NGPU" \
  MASTER_ADDR="$TRAIN_MASTER_ADDR" \
  MASTER_PORT="$TRAIN_MASTER_PORT" \
  CUDA_VISIBLE_DEVICES="${TRAIN_LOCAL_GPU_IDS:-0,1}" \
  bash script/run_robotwin_joint_rft.sh || master_rc=$?

worker_rc=0
"$WAM_PYTHON" -m robotwin_critic.two_stage_rft.multinode_worker wait \
  --queue-root "$MULTINODE_QUEUE_ROOT" \
  --job-id "$job_id" \
  --timeout 0 || worker_rc=$?

if (( master_rc != 0 || worker_rc != 0 )); then
  echo "MULTINODE_TRAIN_FAILED master_rc=$master_rc worker_rc=$worker_rc job=$job_id" >&2
  exit 1
fi
echo "MULTINODE_TRAIN_OK job=$job_id world_size=$((TRAIN_NNODES * TRAIN_LOCAL_NGPU))"
