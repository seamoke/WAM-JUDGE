#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${LINGBOT_ROOT:-/inspire/hdd/project/sais-auto-scientist/public/Lingbot-va}"
CODE_ROOT="${PROJECT_ROOT}/code"
TRAIN_RUN="${PROJECT_ROOT}/train_out/robotwin/robotwin_clean_4xh100_b1_ga32_cosine_2000steps_ckpt400_swanlab_20260724_0230"
BASE_MODEL="${PROJECT_ROOT}/models/lingbot-va-base"
OFFICIAL_MODEL="${PROJECT_ROOT}/models/lingbot-va-posttrain-robotwin"
SEED_CACHE="${CODE_ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json"
PROMPT_CACHE="${PROJECT_ROOT}/train_out/robotwin/prompt_embed_cache/robotwin_clean_ckpt800_1200_1600_2000_easy_n10_4xh100_optix_20260725_1517"
PROMPT_SERVICE_PORT="${PROMPT_SERVICE_PORT:-31056}"
RT_DENOISER="${ROBOTWIN_RT_DENOISER:-optix}"
PIPELINE_ID="${PIPELINE_ID:-robotwin_official_then_clean_ckpt2000_1600_1200_800_easy_n5_4xh100_optix_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_ROOT="${PROJECT_ROOT}/train_out/robotwin-easy-n5-calibration/${PIPELINE_ID}"
PIPELINE_LOG_ROOT="${PROJECT_ROOT}/logs/robotwin-easy-n5-calibration/${PIPELINE_ID}"
EVAL_MODEL_CACHE="${PIPELINE_ROOT}/eval_model_symlinks"

FORMAL_TASKS="adjust_bottle,beat_block_hammer,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_mic,lift_pot,move_can_pot,move_playingcard_away,move_stapler_pad,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_skillet,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,rotate_qrcode,scan_object,stamp_seal,turn_switch"
SMOKE_TASKS="click_bell,move_stapler_pad,place_shoe,place_object_stand,place_object_scale"
USER_STEPS=(2000 1600 1200 800)

mkdir -p "${PIPELINE_ROOT}" "${PIPELINE_LOG_ROOT}" "${EVAL_MODEL_CACHE}"
source "${PROJECT_ROOT}/activate_lingbot.sh"
cd "${CODE_ROOT}"

write_status() {
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

write_status "preflight"

if pgrep -af "[r]un_robotwin_eval.sh|[r]un_server_ckpt.py|[e]val_polict_client_openpi.py" >/dev/null; then
  echo "Refusing to start while another RoboTwin evaluation is active." >&2
  exit 1
fi

if [[ "$(nvidia-smi -L | wc -l | tr -d ' ')" -lt 4 ]]; then
  echo "Need at least four visible GPUs." >&2
  exit 1
fi

for component in vae tokenizer text_encoder transformer; do
  test -d "${BASE_MODEL}/${component}"
  test -d "${OFFICIAL_MODEL}/${component}"
done
test -s "${OFFICIAL_MODEL}/transformer/config.json"
test -s "${OFFICIAL_MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"
test -s "${SEED_CACHE}"
test -s "${PROMPT_CACHE}/manifest.json"
test -s "${PROMPT_CACHE}/negative.pt"

for step in "${USER_STEPS[@]}"; do
  checkpoint="${TRAIN_RUN}/checkpoints/checkpoint_step_${step}"
  test -s "${checkpoint}/transformer/config.json"
  test -s "${checkpoint}/transformer/diffusion_pytorch_model.safetensors"
done

python - "${OFFICIAL_MODEL}" "${SEED_CACHE}" "${PROMPT_CACHE}" "${FORMAL_TASKS}" <<'PY'
import json
import sys
from pathlib import Path

model_root, seed_path, prompt_root = map(Path, sys.argv[1:4])
task_csv = sys.argv[4]
with (model_root / "transformer" / "config.json").open(encoding="utf-8") as handle:
    config = json.load(handle)
assert config.get("attn_mode") == "torch", config.get("attn_mode")

with seed_path.open(encoding="utf-8") as handle:
    seed_payload = json.load(handle)
assert seed_payload.get("task_config") == "demo_clean"
tasks = task_csv.split(",")
assert len(tasks) == 32 and len(set(tasks)) == 32
for task in tasks:
    rows = seed_payload["tasks"][task]
    assert len(rows) >= 5
    seeds = [int(row["seed"]) for row in rows[:5]]
    assert len(seeds) == len(set(seeds))
    assert all(isinstance(row.get("episode_info"), dict) and row["episode_info"] for row in rows[:5])

with (prompt_root / "manifest.json").open(encoding="utf-8") as handle:
    prompt_manifest = json.load(handle)
assert prompt_manifest.get("complete")
assert prompt_manifest.get("task_config") == "demo_clean"
assert int(prompt_manifest.get("test_num", 0)) >= 5
print("Preflight model/seed/prompt validation OK")
PY

python - <<'PY'
import diffusers
import sapien
import torch
import transformers

assert sapien.__version__ == "3.0.0b1", sapien.__version__
assert transformers.__version__ == "4.55.2", transformers.__version__
assert diffusers.__version__ == "0.36.0", diffusers.__version__
assert torch.cuda.device_count() >= 4
print(
    f"Runtime OK: sapien={sapien.__version__} transformers={transformers.__version__} "
    f"diffusers={diffusers.__version__} torch={torch.__version__} gpus={torch.cuda.device_count()}"
)
PY

{
  echo "pipeline_id=${PIPELINE_ID}"
  echo "started_at=$(date -Is)"
  echo "purpose=official_model_calibration_then_user_checkpoints_reverse_order"
  echo "official_model=${OFFICIAL_MODEL}"
  echo "official_transformer_config_sha256=$(sha256sum "${OFFICIAL_MODEL}/transformer/config.json" | awk '{print $1}')"
  echo "train_run=${TRAIN_RUN}"
  echo "base_model=${BASE_MODEL}"
  echo "seed_cache=${SEED_CACHE}"
  echo "seed_cache_sha256=$(sha256sum "${SEED_CACHE}" | awk '{print $1}')"
  echo "prompt_cache=${PROMPT_CACHE}"
  echo "prompt_service=127.0.0.1:${PROMPT_SERVICE_PORT}"
  echo "task_config=demo_clean"
  echo "formal_tasks=32"
  echo "episodes_per_task=5"
  echo "user_steps=${USER_STEPS[*]}"
  echo "ngpu=4"
  echo "clients_per_gpu=1"
  echo "protocol=frame_chunk2_video_steps25_action_steps50_video_guidance5_action_guidance1"
  echo "render=rt_spp32_path_depth8_denoiser${RT_DENOISER}_fast0_low_render0_policy_cameras_only0_defer0_recreate0"
  echo "official_gate_min_sr=0.80"
  echo "run_robotwin_eval_sha256=$(sha256sum "${CODE_ROOT}/script/run_robotwin_eval.sh" | awk '{print $1}')"
  echo "eval_client_sha256=$(sha256sum "${CODE_ROOT}/evaluation/robotwin/eval_polict_client_openpi.py" | awk '{print $1}')"
  echo "prepare_eval_model_sha256=$(sha256sum "${CODE_ROOT}/evaluation/libero/prepare_eval_model.py" | awk '{print $1}')"
  echo "robotwin_base_task_sha256=$(sha256sum "${CODE_ROOT}/third_party/RoboTwin/envs/_base_task.py" | awk '{print $1}')"
} > "${PIPELINE_ROOT}/pipeline_meta.txt"

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
      --model-path "${OFFICIAL_MODEL}" \
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
      return 0
    fi
    sleep 2
  done
  return 1
}

audit_results() {
  local results_root=$1
  local result_label=$2
  local expected_num=$3
  local task_csv=$4
  local summary_path=$5
  python - "${results_root}" "${result_label}" "${expected_num}" "${task_csv}" "${SEED_CACHE}" "${summary_path}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

results_root, label, expected_text, task_csv, cache_path, summary_path = sys.argv[1:]
expected = int(expected_text)
tasks = task_csv.split(",")
seed_root = Path(results_root) / "demo_clean" / label / "stseed-10000"
with open(cache_path, encoding="utf-8") as handle:
    cache = json.load(handle)["tasks"]

rows = []
timings = []
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
    assert [int(row["episode"]) for row in records] == list(range(1, expected + 1)), task
    observed = [int(row["seed"]) for row in records]
    assert len(observed) == len(set(observed)), (task, observed)
    cached = [int(row["seed"]) for row in cache[task]]
    start = 0
    positions = []
    for seed in observed:
        pos = cached.index(seed, start)
        positions.append(pos)
        start = pos + 1
    successes = sum(bool(row["success"]) for row in records)
    assert int(res["succ_num"]) == successes, (task, res, successes)
    timings.extend(float(row["total_sec"]) for row in records)
    rows.append({"task": task, "successes": successes, "total": expected, "sr": successes / expected})

successes = sum(row["successes"] for row in rows)
episodes = len(rows) * expected
summary = {
    "label": label,
    "tasks": len(rows),
    "episodes": episodes,
    "successes": successes,
    "sr": successes / episodes,
    "timing_mean_s": statistics.mean(timings),
    "timing_median_s": statistics.median(timings),
    "per_task": rows,
}
Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(
    f"AUDIT_OK label={label} tasks={len(rows)} episodes={episodes} "
    f"successes={successes} sr={successes / episodes:.4f}"
)
PY
}

run_eval() {
  local kind=$1
  local model_or_checkpoint=$2
  local result_label=$3
  local test_num=$4
  local task_csv=$5
  local run_id=$6
  local start_port=$7
  local master_port_base=$8
  local run_root="${PIPELINE_ROOT}/runs/${run_id}"
  local log_dir="${PIPELINE_LOG_ROOT}/${run_id}"
  local results_root="${run_root}/results"
  local vis_root="${run_root}/visualization"

  mkdir -p "${run_root}" "${log_dir}" "${results_root}" "${vis_root}"
  {
    echo "run_id=${run_id}"
    echo "started_at=$(date -Is)"
    echo "kind=${kind}"
    echo "model_or_checkpoint=${model_or_checkpoint}"
    echo "result_label=${result_label}"
    echo "task_config=demo_clean"
    echo "tasks=${task_csv}"
    echo "test_num=${test_num}"
    echo "rt_denoiser=${RT_DENOISER}"
    echo "results_root=${results_root}"
    echo "log_dir=${log_dir}"
  } > "${run_root}/meta.txt"
  echo "running" > "${run_root}/status.txt"

  local full_model_path=""
  local checkpoint_path="${model_or_checkpoint}"
  if [[ "${kind}" == "official_full_model" ]]; then
    full_model_path="${model_or_checkpoint}"
  fi

  set +e
  FULL_MODEL_PATH="${full_model_path}" \
  CHECKPOINT_PATH="${checkpoint_path}" \
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
  if [[ "${rc}" -ne 0 ]]; then
    echo "failed rc=${rc}" > "${run_root}/status.txt"
    return "${rc}"
  fi

  audit_results "${results_root}" "${result_label}" "${test_num}" "${task_csv}" \
    "${run_root}/summary.json" | tee "${run_root}/audit.txt"
  echo "done" > "${run_root}/status.txt"
}

write_status "starting_prompt_service"
start_prompt_service

write_status "official_smoke"
official_smoke_id="${PIPELINE_ID}_official_smoke_n1"
run_eval official_full_model "${OFFICIAL_MODEL}" "lingbot-va-posttrain-robotwin" 1 \
  "${SMOKE_TASKS}" "${official_smoke_id}" 34056 34161

write_status "official_formal_n5"
official_run_id="${PIPELINE_ID}_official_easy_n5"
run_eval official_full_model "${OFFICIAL_MODEL}" "lingbot-va-posttrain-robotwin" 5 \
  "${FORMAL_TASKS}" "${official_run_id}" 35056 35161

python - "${PIPELINE_ROOT}/runs/${official_run_id}/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
if float(summary["sr"]) < 0.80:
    raise SystemExit(
        f"Official calibration SR {summary['sr']:.4f} is below 0.80; "
        "refusing to evaluate user checkpoints with an uncalibrated evaluator."
    )
print(f"Official calibration gate passed: SR={summary['sr']:.4f}")
PY

for step in "${USER_STEPS[@]}"; do
  write_status "user_checkpoint_step_${step}_n5"
  checkpoint="${TRAIN_RUN}/checkpoints/checkpoint_step_${step}"
  run_id="${PIPELINE_ID}_user_step${step}_easy_n5"
  run_eval user_checkpoint "${checkpoint}" "checkpoint_step_${step}" 5 \
    "${FORMAL_TASKS}" "${run_id}" 36056 36161
done

{
  echo "pipeline_id=${PIPELINE_ID}"
  echo "completed_at=$(date -Is)"
  echo "${PIPELINE_ROOT}/runs/${official_run_id}"
  for step in "${USER_STEPS[@]}"; do
    echo "${PIPELINE_ROOT}/runs/${PIPELINE_ID}_user_step${step}_easy_n5"
  done
} > "${PIPELINE_ROOT}/completed_runs.txt"

printf '0\n' > "${PIPELINE_ROOT}/pipeline_exit_code"
write_status "done"
echo "PIPELINE_DONE ${PIPELINE_ID}"
