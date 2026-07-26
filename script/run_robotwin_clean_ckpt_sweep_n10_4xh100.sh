#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
CODE_ROOT="${PROJECT_ROOT}/code"
TRAIN_RUN="${PROJECT_ROOT}/train_out/robotwin/robotwin_clean_4xh100_b1_ga32_cosine_2000steps_ckpt400_swanlab_20260724_0230"
BASE_MODEL="${PROJECT_ROOT}/models/lingbot-va-base"
SEED_CACHE="${CODE_ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json"
PROMPT_SERVICE_PORT="${PROMPT_SERVICE_PORT:-31056}"
RT_DENOISER="${ROBOTWIN_RT_DENOISER:-optix}"
PIPELINE_ID="${PIPELINE_ID:-robotwin_clean_ckpt800_1200_1600_2000_easy_n10_4xh100_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_ROOT="${PROJECT_ROOT}/train_out/robotwin-easy-n10-sweep/${PIPELINE_ID}"
PIPELINE_LOG_ROOT="${PROJECT_ROOT}/logs/robotwin-easy-n10-sweep/${PIPELINE_ID}"
PROMPT_CACHE="${PROJECT_ROOT}/train_out/robotwin/prompt_embed_cache/${PIPELINE_ID}"
EVAL_MODEL_CACHE="${PIPELINE_ROOT}/eval_model_symlinks"
STEPS=(800 1200 1600 2000)

FORMAL_TASKS="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_mic,lift_pot,move_can_pot,move_playingcard_away,move_stapler_pad,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_skillet,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,rotate_qrcode,scan_object,stamp_seal,turn_switch"
SMOKE_TASKS="click_bell,move_stapler_pad,place_shoe,place_object_stand,place_object_scale"

mkdir -p "${PIPELINE_ROOT}" "${PIPELINE_LOG_ROOT}" "${EVAL_MODEL_CACHE}"
source "${PROJECT_ROOT}/activate_lingbot.sh"
cd "${CODE_ROOT}"

write_pipeline_status() {
  printf '%s\n' "$1" > "${PIPELINE_ROOT}/pipeline_status.txt"
}

PROMPT_SERVICE_PID=""
stop_prompt_service() {
  if [[ -n "${PROMPT_SERVICE_PID}" ]] && kill -0 "${PROMPT_SERVICE_PID}" 2>/dev/null; then
    kill -TERM "${PROMPT_SERVICE_PID}" 2>/dev/null || true
    wait "${PROMPT_SERVICE_PID}" 2>/dev/null || true
  fi
}
trap stop_prompt_service EXIT INT TERM

write_pipeline_status "preflight"

if pgrep -af "[r]un_robotwin_eval.sh|[r]un_server_ckpt.py|[e]val_polict_client_openpi.py" >/dev/null; then
  echo "Refusing to start while another RoboTwin evaluation is active." >&2
  exit 1
fi

if [[ "$(nvidia-smi -L | wc -l | tr -d ' ')" -lt 4 ]]; then
  echo "Need at least four visible GPUs." >&2
  exit 1
fi

test -s "${SEED_CACHE}"
test -d "${BASE_MODEL}/vae"
test -d "${BASE_MODEL}/tokenizer"
test -d "${BASE_MODEL}/text_encoder"
test -d "${BASE_MODEL}/transformer"
test -d "${CODE_ROOT}/third_party/RoboTwin/assets/background_texture"
test -d "${CODE_ROOT}/third_party/RoboTwin/assets/embodiments"
test -d "${CODE_ROOT}/third_party/RoboTwin/assets/objects"

for step in "${STEPS[@]}"; do
  checkpoint="${TRAIN_RUN}/checkpoints/checkpoint_step_${step}"
  test -s "${checkpoint}/transformer/config.json"
  test -s "${checkpoint}/transformer/diffusion_pytorch_model.safetensors"
done

python - "${SEED_CACHE}" "${FORMAL_TASKS}" <<'PY'
import json
import sys

cache_path, task_csv = sys.argv[1:]
expected = task_csv.split(",")
with open(cache_path, encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload.get("task_config") == "demo_clean", payload.get("task_config")
tasks = payload["tasks"]
assert set(expected).issubset(tasks), sorted(set(expected) - set(tasks))
for task in expected:
    rows = tasks[task]
    assert len(rows) >= 10, (task, len(rows))
    seeds = [int(row["seed"]) for row in rows[:10]]
    assert len(seeds) == len(set(seeds)), task
    assert all(
        isinstance(row.get("episode_info"), dict) and row["episode_info"]
        for row in rows[:10]
    ), task
print(
    f"Seed cache OK: formal_tasks={len(expected)}, "
    f"cache_tasks={len(tasks)}, >=10 unique valid candidates/task"
)
PY

{
  echo "pipeline_id=${PIPELINE_ID}"
  echo "started_at=$(date -Is)"
  echo "project_root=${PROJECT_ROOT}"
  echo "code_root=${CODE_ROOT}"
  echo "train_run=${TRAIN_RUN}"
  echo "base_model=${BASE_MODEL}"
  echo "seed_cache=${SEED_CACHE}"
  echo "seed_cache_sha256=$(sha256sum "${SEED_CACHE}" | awk '{print $1}')"
  echo "prompt_cache=${PROMPT_CACHE}"
  echo "prompt_service=127.0.0.1:${PROMPT_SERVICE_PORT}"
  echo "steps=${STEPS[*]}"
  echo "task_config=demo_clean"
  echo "formal_tasks=32"
  echo "episodes_per_task=10"
  echo "ngpu=4"
  echo "clients_per_gpu=1"
  echo "protocol=frame_chunk2_video_steps25_action_steps50_video_guidance5_action_guidance1"
  echo "render=rt_spp32_path_depth8_denoiser${RT_DENOISER}_fast0_low_render0_policy_cameras_only0_defer0_recreate0"
  echo "offload=enable0_vae0_text_encoder1_prompt_cache_strict1"
  echo "resume=timing_seed_subsequence_chunk1_skip_existing"
  echo "run_robotwin_eval_sha256=$(sha256sum "${CODE_ROOT}/script/run_robotwin_eval.sh" | awk '{print $1}')"
  echo "eval_client_sha256=$(sha256sum "${CODE_ROOT}/evaluation/robotwin/eval_polict_client_openpi.py" | awk '{print $1}')"
  echo "prepare_eval_model_sha256=$(sha256sum "${CODE_ROOT}/evaluation/libero/prepare_eval_model.py" | awk '{print $1}')"
  echo "robotwin_base_task_sha256=$(sha256sum "${CODE_ROOT}/third_party/RoboTwin/envs/_base_task.py" | awk '{print $1}')"
  git -C "${CODE_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/code_commit=/' || true
  python --version 2>&1 | sed 's/^/python=/'
  python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()}")
PY
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
} > "${PIPELINE_ROOT}/pipeline_meta.txt"

prompt_cache_complete() {
  python - "${PROMPT_CACHE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
if not manifest_path.is_file() or not (root / "negative.pt").is_file():
    raise SystemExit(1)
with manifest_path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
if not manifest.get("complete"):
    raise SystemExit(1)
if manifest.get("task_config") != "demo_clean" or manifest.get("test_num") != 10:
    raise SystemExit(1)
expected = int(manifest["unique_prompts"]) + 1
actual = len(list(root.glob("*.pt")))
if actual != expected:
    raise SystemExit(1)
print(
    f"Prompt cache OK: tasks={manifest['tasks']} "
    f"episodes={manifest['episodes']} unique={manifest['unique_prompts']} files={actual}"
)
PY
}

if ! prompt_cache_complete; then
  write_pipeline_status "building_prompt_cache"
  CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 \
    python script/precompute_robotwin_prompt_embeddings.py \
      --model-path "${BASE_MODEL}" \
      --seed-cache "${SEED_CACHE}" \
      --robotwin-root "${CODE_ROOT}/third_party/RoboTwin" \
      --output-dir "${PROMPT_CACHE}" \
      --test-num 10 \
      --device cuda:0
  prompt_cache_complete
fi

prompt_service_ready() {
  python - "${PROMPT_SERVICE_PORT}" <<'PY'
import json
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1) as sock:
    sock.sendall(b'{"op":"ping"}\n')
    response = json.loads(sock.makefile("rb").readline())
assert response.get("ok") and response.get("status") == "ready", response
PY
}

start_prompt_service() {
  local service_log="${PIPELINE_LOG_ROOT}/prompt_service.log"
  if prompt_service_ready 2>/dev/null; then
    echo "Refusing to reuse an unowned prompt service on port ${PROMPT_SERVICE_PORT}." >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES=0 \
  NVIDIA_VISIBLE_DEVICES=0 \
  PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" \
    python -u script/robotwin_prompt_embedding_service.py \
      --model-path "${BASE_MODEL}" \
      --cache-dir "${PROMPT_CACHE}" \
      --host 127.0.0.1 \
      --port "${PROMPT_SERVICE_PORT}" \
      --device cuda:0 \
      > "${service_log}" 2>&1 &
  PROMPT_SERVICE_PID=$!
  printf '%s\n' "${PROMPT_SERVICE_PID}" > "${PIPELINE_ROOT}/prompt_service.pid"

  for _ in $(seq 1 150); do
    if ! kill -0 "${PROMPT_SERVICE_PID}" 2>/dev/null; then
      tail -n 100 "${service_log}" >&2 || true
      return 1
    fi
    if prompt_service_ready 2>/dev/null; then
      echo "Prompt service ready: pid=${PROMPT_SERVICE_PID} port=${PROMPT_SERVICE_PORT}"
      return 0
    fi
    sleep 2
  done

  echo "Prompt service did not become ready." >&2
  tail -n 100 "${service_log}" >&2 || true
  return 1
}

write_pipeline_status "starting_prompt_service"
start_prompt_service

audit_results() {
  local results_root=$1
  local step=$2
  local expected_num=$3
  local task_csv=$4
  python - "${results_root}" "${step}" "${expected_num}" "${task_csv}" "${SEED_CACHE}" <<'PY'
import json
import sys
from pathlib import Path

results_root, step_text, expected_text, task_csv, cache_path = sys.argv[1:]
expected = int(expected_text)
tasks = task_csv.split(",")
seed_root = (
    Path(results_root)
    / "demo_clean"
    / f"checkpoint_step_{step_text}"
    / "stseed-10000"
)
with open(cache_path, encoding="utf-8") as handle:
    cache = json.load(handle)["tasks"]

total_success = 0
for task in tasks:
    res_path = seed_root / "metrics" / task / "res.json"
    timing_path = seed_root / "eval_timing" / f"{task}.jsonl"
    with res_path.open(encoding="utf-8") as handle:
        res = json.load(handle)
    records = [
        json.loads(line)
        for line in timing_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert int(res["total_num"]) == expected, (task, res)
    assert len(records) == expected, (task, len(records))
    assert [int(row["episode"]) for row in records] == list(
        range(1, expected + 1)
    ), task
    observed = [int(row["seed"]) for row in records]
    assert len(observed) == len(set(observed)), (task, observed)
    cached = [int(row["seed"]) for row in cache[task]]
    positions = []
    start = 0
    for seed in observed:
        pos = cached.index(seed, start)
        positions.append(pos)
        start = pos + 1
    assert positions == sorted(positions), (task, positions)
    successes = sum(bool(row["success"]) for row in records)
    assert int(res["succ_num"]) == successes, (task, res, successes)
    total_success += successes

print(
    f"AUDIT_OK step={step_text} tasks={len(tasks)} "
    f"episodes={len(tasks) * expected} successes={total_success}"
)
PY
}

run_eval() {
  local step=$1
  local test_num=$2
  local task_csv=$3
  local run_id=$4
  local start_port=$5
  local master_port_base=$6
  local checkpoint="${TRAIN_RUN}/checkpoints/checkpoint_step_${step}"
  local run_root="${PIPELINE_ROOT}/runs/${run_id}"
  local log_dir="${PIPELINE_LOG_ROOT}/${run_id}"
  local results_root="${run_root}/results"
  local vis_root="${run_root}/visualization"

  mkdir -p "${run_root}" "${log_dir}" "${results_root}" "${vis_root}"
  {
    echo "run_id=${run_id}"
    echo "started_at=$(date -Is)"
    echo "checkpoint_step=${step}"
    echo "checkpoint_path=${checkpoint}"
    echo "checkpoint_weight_sha256=$(sha256sum "${checkpoint}/transformer/diffusion_pytorch_model.safetensors" | awk '{print $1}')"
    echo "checkpoint_config_sha256=$(sha256sum "${checkpoint}/transformer/config.json" | awk '{print $1}')"
    echo "base_model=${BASE_MODEL}"
    echo "seed_cache=${SEED_CACHE}"
    echo "seed_cache_sha256=$(sha256sum "${SEED_CACHE}" | awk '{print $1}')"
    echo "task_config=demo_clean"
    echo "tasks=${task_csv}"
    echo "test_num=${test_num}"
    echo "rt_denoiser=${RT_DENOISER}"
    echo "results_root=${results_root}"
    echo "log_dir=${log_dir}"
  } > "${run_root}/meta.txt"
  echo "running" > "${run_root}/status.txt"

  set +e
  FULL_MODEL_PATH= \
  CHECKPOINT_PATH="${checkpoint}" \
  BASE_MODEL="${BASE_MODEL}" \
  EVAL_MODEL_CACHE="${EVAL_MODEL_CACHE}" \
  EVAL_CONFIG_NAME=robotwin \
  TASK_CONFIG=demo_clean \
  RUN_HARD=0 \
  NGPU=4 \
  CLIENTS_PER_GPU=1 \
  ROBOTWIN_DYNAMIC_SHARDS=1 \
  ROBOTWIN_SIM_FOLLOWS_SERVER_GPU=1 \
  ROBOTWIN_TASKS="${task_csv}" \
  TEST_NUM="${test_num}" \
  SEED=0 \
  ROBOTWIN_SEED_CACHE="${SEED_CACHE}" \
  ROBOTWIN_EXPERT_CHECK=1 \
  ROBOTWIN_FAST=0 \
  ROBOTWIN_EVAL_LOW_RENDER=0 \
  ROBOTWIN_POLICY_CAMERAS_ONLY=0 \
  ROBOTWIN_DEFER_RENDER_UPDATES=0 \
  ROBOTWIN_RECREATE_CAMERAS_EVERY=0 \
  ROBOTWIN_RT_DENOISER="${RT_DENOISER}" \
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
  WAN_VA_PROMPT_EMBED_CACHE_DIR="${PROMPT_CACHE}" \
  WAN_VA_PROMPT_EMBED_CACHE_STRICT=1 \
  WAN_VA_PROMPT_EMBED_SERVICE="127.0.0.1:${PROMPT_SERVICE_PORT}" \
  WAN_VA_PROMPT_EMBED_SERVICE_TIMEOUT=300 \
  WAN_VA_SAVE_INFER_DEBUG=0 \
  STRICT_CUROBO_COMPAT=1 \
  SKIP_EXISTING=1 \
  MAX_CHECKPOINTS=1 \
  START_PORT="${start_port}" \
  MASTER_PORT_BASE="${master_port_base}" \
  RESULTS_ROOT="${results_root}" \
  LOG_DIR="${log_dir}" \
  VIS_ROOT="${vis_root}" \
    bash script/run_robotwin_eval.sh
  local rc=$?
  set -e

  echo "finished_at=$(date -Is)" >> "${run_root}/meta.txt"
  echo "exit_code=${rc}" >> "${run_root}/meta.txt"
  printf '%s\n' "${rc}" > "${run_root}/exit_code"
  if [[ "${rc}" -eq 0 ]]; then
    echo "done" > "${run_root}/status.txt"
  else
    echo "failed rc=${rc}" > "${run_root}/status.txt"
    return "${rc}"
  fi

  audit_results "${results_root}" "${step}" "${test_num}" "${task_csv}" \
    | tee "${run_root}/audit.txt"
}

write_pipeline_status "smoke_checkpoint_800"
SMOKE_RUN_ID="${PIPELINE_ID}_smoke_step800_n1"
run_eval 800 1 "${SMOKE_TASKS}" "${SMOKE_RUN_ID}" 32056 32161

write_pipeline_status "formal_evaluations"
for step in "${STEPS[@]}"; do
  run_id="${PIPELINE_ID}_step${step}_easy_n10"
  write_pipeline_status "evaluating_checkpoint_step_${step}"
  run_eval "${step}" 10 "${FORMAL_TASKS}" "${run_id}" 33056 33161
done

{
  echo "pipeline_id=${PIPELINE_ID}"
  echo "completed_at=$(date -Is)"
  for step in "${STEPS[@]}"; do
    echo "${PIPELINE_ROOT}/runs/${PIPELINE_ID}_step${step}_easy_n10"
  done
} > "${PIPELINE_ROOT}/completed_runs.txt"

printf '0\n' > "${PIPELINE_ROOT}/pipeline_exit_code"
write_pipeline_status "done"
echo "PIPELINE_DONE ${PIPELINE_ID}"
