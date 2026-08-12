#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT}"
STAGE="${STAGE:?Set STAGE to generate, decode, or vlac}"
COLLECT_DIR="${COLLECT_DIR:?Set COLLECT_DIR}"
CURRENT_MODEL="${CURRENT_MODEL:?Set CURRENT_MODEL}"
REMOTE_WORKER_OFFSET="${REMOTE_WORKER_OFFSET:?Set REMOTE_WORKER_OFFSET}"
REMOTE_GPU_IDS="${REMOTE_GPU_IDS:-0,1}"
WAM_PYTHON="${WAM_PYTHON:-$PROJECT_ROOT/.runtime/.venv/bin/python}"
VLAC_PYTHON="${VLAC_PYTHON:-$WAM_PYTHON}"
WORKER_MASTER_PORT_BASE="${WORKER_MASTER_PORT_BASE:-29700}"

IFS=',' read -r -a GPUS <<< "$REMOTE_GPU_IDS"
if (( ${#GPUS[@]} < 1 )); then
  echo "REMOTE_GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

pids=()
for local_worker in "${!GPUS[@]}"; do
  gpu="${GPUS[$local_worker]}"
  worker=$((REMOTE_WORKER_OFFSET + local_worker))
  worker_name="$(printf '%02d' "$worker")"
  worker_dir="$COLLECT_DIR/worker_$worker_name"
  case "$STAGE" in
    generate)
      CUDA_VISIBLE_DEVICES="$gpu" \
      MASTER_ADDR=127.0.0.1 \
      MASTER_PORT="$((WORKER_MASTER_PORT_BASE + worker))" \
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      WAN_VA_DISABLE_WORLD1_FSDP=1 \
      "$WAM_PYTHON" -m robotwin_critic.two_stage_rft.generate_wam_candidates \
        --contexts "$COLLECT_DIR/contexts_worker_$worker_name.jsonl" \
        --model "$CURRENT_MODEL" \
        --output-dir "$worker_dir" \
        --candidates-per-context "$CANDIDATES_PER_Q" \
        --inference-batch-size "$INFER_BATCH_SIZE_PER_GPU" \
        --base-seed "$((BASE_SEED + COLLECT_INDEX * 1000000 + worker * 10000))" \
        --resume \
        > "$COLLECT_DIR/generate_worker_$worker_name.log" 2>&1 &
      ;;
    decode)
      CUDA_VISIBLE_DEVICES="$gpu" "$WAM_PYTHON" \
        -m robotwin_critic.two_stage_rft.decode_wam_candidates \
        --input "$worker_dir/candidates.jsonl" \
        --model "$CURRENT_MODEL" \
        --device cuda:0 \
        > "$COLLECT_DIR/decode_worker_$worker_name.log" 2>&1 &
      ;;
    vlac)
      adapter_args=()
      if [[ -n "${VLAC_ADAPTER:-}" ]]; then
        adapter_args+=(--adapter "$VLAC_ADAPTER")
      fi
      CUDA_VISIBLE_DEVICES="$gpu" "$VLAC_PYTHON" \
        -m robotwin_critic.two_stage_rft.score_vlac_candidates \
        --input "$COLLECT_DIR/vlac_shards/shard_$worker_name.jsonl" \
        --model "$VLAC_MODEL" \
        --output "$COLLECT_DIR/vlac_scored_$worker_name.jsonl" \
        --device cuda:0 \
        --batch-size "$VLAC_BATCH_SIZE_PER_GPU" \
        "${adapter_args[@]}" \
        > "$COLLECT_DIR/vlac_worker_$worker_name.log" 2>&1 &
      ;;
    *)
      echo "Unsupported remote collection stage: $STAGE" >&2
      exit 2
      ;;
  esac
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=$?
done
exit "$rc"
