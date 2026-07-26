#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/lingbot-va}
cd "${ROOT}"

MODEL_PATH=${MODEL_PATH:-${ROOT}/models/lingbot-va-posttrain-robotwin}
SEED_CACHE=${SEED_CACHE:-${ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json}
TEST_NUM=${TEST_NUM:-20}
NGPU=${NGPU:-5}
PHYSICAL_GPUS=6
PROMPT_SERVICE_GPU=5
PROMPT_SERVICE_PORT=${PROMPT_SERVICE_PORT:-31056}
PROMPT_CACHE_DIR=${PROMPT_CACHE_DIR:-${ROOT}/train_out/robotwin/prompt_embed_cache/demo_clean_n20_official}
RUN_ID=${RUN_ID:-robotwin_official_posttrain_easy_n${TEST_NUM}_${NGPU}shards_${PHYSICAL_GPUS}x5090_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-${ROOT}/logs/robotwin_eval_official_posttrain_easy_n${TEST_NUM}/${RUN_ID}}
RESULTS_ROOT=${RESULTS_ROOT:-${ROOT}/train_out/robotwin-posttrain-official/eval_results_easy_n${TEST_NUM}/${RUN_ID}}
VIS_ROOT=${VIS_ROOT:-${ROOT}/train_out/robotwin-posttrain-official/eval_visualization_easy_n${TEST_NUM}/${RUN_ID}}

OFFICIAL_TASKS="stack_bowls_three,handover_block,hanging_mug,scan_object,lift_pot,put_object_cabinet,stack_blocks_three,place_shoe,adjust_bottle,place_mouse_pad,dump_bin_bigbin,move_pillbottle_pad,pick_dual_bottles,shake_bottle,place_fan,turn_switch,shake_bottle_horizontally,place_container_plate,rotate_qrcode,place_object_stand,put_bottles_dustbin,move_stapler_pad,place_burger_fries,place_bread_basket,pick_diverse_bottles,open_microwave,beat_block_hammer,press_stapler,click_bell,move_playingcard_away,open_laptop,move_can_pot,stack_bowls_two,place_a2b_right,stamp_seal,place_object_basket,handover_mic,place_bread_skillet,stack_blocks_two,place_cans_plasticbox,click_alarmclock,blocks_ranking_size,place_phone_stand,place_can_basket,place_object_scale,place_a2b_left,grab_roller,place_dual_shoes,place_empty_cup,blocks_ranking_rgb"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${VIS_ROOT}"

for component in vae tokenizer text_encoder transformer; do
  test -d "${MODEL_PATH}/${component}"
done
test -s "${MODEL_PATH}/transformer/diffusion_pytorch_model.safetensors.index.json"
test -s "${SEED_CACHE}"
test -s "${PROMPT_CACHE_DIR}/negative.pt"

if pgrep -af "[s]cripts.gpu_guard.holder_worker" >/dev/null; then
  echo "Refusing to start while gpu-guard holder workers are active." >&2
  echo "Use the controller-side gpu-guard release command first; do not kill holders." >&2
  exit 1
fi

if pgrep -af "[r]un_robotwin_eval.sh|[r]un_server_ckpt.py|[e]val_polict_client_openpi.py" >/dev/null; then
  echo "Refusing to start while another RoboTwin evaluation is running." >&2
  exit 1
fi

python - "${SEED_CACHE}" "${OFFICIAL_TASKS}" <<'PY'
import json
import sys

cache_path, task_csv = sys.argv[1:]
tasks = task_csv.split(",")
with open(cache_path, encoding="utf-8") as f:
    payload = json.load(f)
assert payload.get("task_config") == "demo_clean", payload.get("task_config")
entries = payload.get("tasks", {})
assert set(entries) == set(tasks), (sorted(set(tasks) - set(entries)), sorted(set(entries) - set(tasks)))
for task in tasks:
    rows = entries[task]
    assert len(rows) >= 20, (task, len(rows))
    seeds = [int(row["seed"]) for row in rows]
    assert len(seeds) == len(set(seeds)), task
    assert all(isinstance(row.get("episode_info"), dict) for row in rows), task
print(f"seed cache OK: {len(tasks)} tasks, >=20 unique valid seeds/task")
PY

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "model_path=${MODEL_PATH}"
  echo "model_transformer_config_sha256=$(sha256sum "${MODEL_PATH}/transformer/config.json" | awk '{print $1}')"
  echo "seed_cache=${SEED_CACHE}"
  echo "seed_cache_sha256=$(sha256sum "${SEED_CACHE}" | awk '{print $1}')"
  echo "lingbot_repo_commit=7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
  echo "robotwin_commit=2eeec322d95799f537cbfe5f291a8220d965ccb8"
  echo "task_config=demo_clean"
  echo "tasks=50"
  echo "episodes_per_task=${TEST_NUM}"
  echo "ngpu=${NGPU}"
  echo "physical_gpus=${PHYSICAL_GPUS}"
  echo "prompt_service_gpu=${PROMPT_SERVICE_GPU}"
  echo "prompt_service_port=${PROMPT_SERVICE_PORT}"
  echo "clients_per_gpu=1"
  echo "protocol=official_config_frame_chunk2_video_steps25_action_steps50"
  echo "guidance=video5_action1"
  echo "render=fast0_low_render0_policy_cameras_only0_defer0_recreate0"
  echo "offload=vae0_text_encoder_service_gpu5_prompt_cache_strict1"
  echo "resume=seed_cache_ordered_subsequence_timing_chunk1_skip_existing"
  echo "results_root=${RESULTS_ROOT}"
  echo "log_dir=${LOG_DIR}"
} > "${LOG_DIR}/meta.txt"
echo running > "${LOG_DIR}/status.txt"

PROMPT_SERVICE_PID=""
cleanup_prompt_service() {
  if [[ -n "${PROMPT_SERVICE_PID}" ]] && kill -0 "${PROMPT_SERVICE_PID}" 2>/dev/null; then
    kill -TERM "${PROMPT_SERVICE_PID}" 2>/dev/null || true
    wait "${PROMPT_SERVICE_PID}" 2>/dev/null || true
  fi
}
trap cleanup_prompt_service EXIT INT TERM

CUDA_VISIBLE_DEVICES="${PROMPT_SERVICE_GPU}" \
NVIDIA_VISIBLE_DEVICES="${PROMPT_SERVICE_GPU}" \
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
"${ROOT}/.venv/bin/python" -u script/robotwin_prompt_embedding_service.py \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${PROMPT_CACHE_DIR}" \
  --host 127.0.0.1 \
  --port "${PROMPT_SERVICE_PORT}" \
  --device cuda:0 \
  > "${LOG_DIR}/prompt_service.log" 2>&1 &
PROMPT_SERVICE_PID=$!
echo "${PROMPT_SERVICE_PID}" > "${LOG_DIR}/prompt_service.pid"

prompt_service_ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "${PROMPT_SERVICE_PID}" 2>/dev/null; then
    break
  fi
  if python - "${PROMPT_SERVICE_PORT}" <<'PY'
import json
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1) as sock:
    sock.sendall(b'{"op":"ping"}\n')
    response = json.loads(sock.makefile("rb").readline())
assert response.get("ok") and response.get("status") == "ready", response
PY
  then
    prompt_service_ready=1
    break
  fi
  sleep 2
done
if [[ "${prompt_service_ready}" -ne 1 ]]; then
  echo "Prompt embedding service failed to become ready." >&2
  tail -100 "${LOG_DIR}/prompt_service.log" >&2 || true
  exit 1
fi

set +e
FULL_MODEL_PATH="${MODEL_PATH}" \
CHECKPOINT_PATH="${MODEL_PATH}" \
BASE_MODEL="${MODEL_PATH}" \
EVAL_CONFIG_NAME=robotwin \
TASK_CONFIG=demo_clean \
RUN_HARD=0 \
NGPU="${NGPU}" \
CLIENTS_PER_GPU=1 \
ROBOTWIN_DYNAMIC_SHARDS=1 \
ROBOTWIN_SIM_FOLLOWS_SERVER_GPU=1 \
ROBOTWIN_TASKS="${OFFICIAL_TASKS}" \
TEST_NUM="${TEST_NUM}" \
SEED=0 \
ROBOTWIN_SEED_CACHE="${SEED_CACHE}" \
ROBOTWIN_EXPERT_CHECK=1 \
ROBOTWIN_FAST=0 \
ROBOTWIN_EVAL_LOW_RENDER=0 \
ROBOTWIN_POLICY_CAMERAS_ONLY=0 \
ROBOTWIN_DEFER_RENDER_UPDATES=0 \
ROBOTWIN_RECREATE_CAMERAS_EVERY=0 \
ROBOTWIN_EVAL_TIMING=1 \
ROBOTWIN_RESUME_PARTIAL=1 \
ROBOTWIN_CLIENT_EPISODE_CHUNK=1 \
ROBOTWIN_CLIENT_TIMEOUT_SEC=900 \
ROBOTWIN_CLIENT_NO_PROGRESS_RETRIES=3 \
ROBOTWIN_EVAL_VIDEO_LOG=0 \
ROBOTWIN_SAVE_COMPARISON_VIDEO=0 \
ROBOTWIN_SAVE_VISUALIZATION=0 \
WAN_VA_ENABLE_OFFLOAD=0 \
WAN_VA_OFFLOAD_VAE=0 \
WAN_VA_OFFLOAD_TEXT_ENCODER=1 \
WAN_VA_SWAP_TEXT_ENCODER_FOR_PROMPT=0 \
WAN_VA_PROMPT_EMBED_CACHE_DIR=/workspace/lingbot-va/train_out/robotwin/prompt_embed_cache/demo_clean_n20_official \
WAN_VA_PROMPT_EMBED_CACHE_STRICT=1 \
WAN_VA_PROMPT_EMBED_SERVICE="127.0.0.1:${PROMPT_SERVICE_PORT}" \
WAN_VA_PROMPT_EMBED_SERVICE_TIMEOUT=300 \
WAN_VA_SAVE_INFER_DEBUG=0 \
SKIP_EXISTING=1 \
MAX_CHECKPOINTS=1 \
START_PORT=30056 \
MASTER_PORT_BASE=30161 \
RESULTS_ROOT="${RESULTS_ROOT}" \
LOG_DIR="${LOG_DIR}" \
VIS_ROOT="${VIS_ROOT}" \
bash script/run_robotwin_eval.sh
rc=$?
set -e

echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "${LOG_DIR}/meta.txt"
echo "exit_code=${rc}" >> "${LOG_DIR}/meta.txt"
if [[ "${rc}" -eq 0 ]]; then
  echo done > "${LOG_DIR}/status.txt"
else
  echo "failed rc=${rc}" > "${LOG_DIR}/status.txt"
fi
exit "${rc}"
