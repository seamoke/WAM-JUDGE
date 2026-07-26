#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/lingbot-va
RUN_ID=robotwin_rt_cpg1_official_step600_n20_2x5090_20260718_0233
RUN_LOG_DIR="${ROOT}/logs/robotwin_eval_rt_cpg1_official_step600_n20/${RUN_ID}"
SAVE_ROOT="${ROOT}/train_out/robotwin-short-3gpu/eval_results_rt_cpg1_official_step600_n20/${RUN_ID}/demo_clean/checkpoint_step_18000"
TASK_QUEUE="${RUN_LOG_DIR}/task_queue_checkpoint_step_18000_demo_clean_20260718_023254.txt"
RECOVERY_LOG="${RUN_LOG_DIR}/recovery_shard0_gpu0.log"
TEST_NUM=20
PORT=29056
QUEUE_RESERVE=3

cd "${ROOT}"
source .venv/bin/activate
[[ -f script/.robotwin_eval_env ]] && source script/.robotwin_eval_env

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export EVAL_MODEL_CACHE="${TMPDIR:-/tmp}/lingbot_robotwin_eval_symlinks"
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export ROBOTWIN_VULKAN_GPU=0
export NVIDIA_VISIBLE_DEVICES=0
export ROBOTWIN_VULKAN_SIM_SLOTS=1
export ROBOTWIN_EXPERT_CHECK=1
export ROBOTWIN_EVAL_VIDEO_LOG=0
export ROBOTWIN_SAVE_COMPARISON_VIDEO=0
export ROBOTWIN_SAVE_VISUALIZATION=0
export ROBOTWIN_EVAL_LOW_RENDER=0
export ROBOTWIN_POLICY_CAMERAS_ONLY=0
export ROBOTWIN_DEFER_RENDER_UPDATES=0
export ROBOTWIN_RECREATE_CAMERAS_EVERY=0
export ROBOTWIN_RESUME_PARTIAL=1
export ROBOTWIN_CLIENT_EPISODE_CHUNK=1
export ROBOTWIN_EVAL_TIMING=1
export ROBOTWIN_SEED_CACHE="${ROOT}/train_out/robotwin/eval_seed_cache/demo_clean_seed0_n100.json"
export WAN_VA_ENABLE_OFFLOAD=0
export WAN_VA_OFFLOAD_VAE=0
export WAN_VA_OFFLOAD_TEXT_ENCODER=1

result_total() {
  python - "${1}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as result_file:
        print(int(json.load(result_file).get("total_num", 0)))
except (OSError, TypeError, ValueError):
    print(0)
PY
}

run_task() {
  local task_name=$1
  local result_file="${SAVE_ROOT}/stseed-10000/metrics/${task_name}/res.json"
  local no_progress_retries=0

  while true; do
    local before after client_rc
    before="$(result_total "${result_file}")"
    if [[ "${before}" -ge "${TEST_NUM}" ]]; then
      echo "recovery skip task=${task_name} complete=${before}/${TEST_NUM}"
      return 0
    fi

    echo "recovery task=${task_name} port=${PORT} resume=${before}/${TEST_NUM}"
    set +e
    (
      cd "${ROOT}/third_party/RoboTwin"
      PYTHONUNBUFFERED=1 \
      PYTHONWARNINGS=ignore::UserWarning \
      timeout --signal=TERM --kill-after=30s 900 \
        python -m evaluation.robotwin.eval_polict_client_openpi \
        --config policy/ACT/deploy_policy.yml \
        --overrides \
        --task_name "${task_name}" \
        --task_config demo_clean \
        --train_config_name 0 \
        --model_name 0 \
        --ckpt_setting 0 \
        --seed 0 \
        --policy_name ACT \
        --save_root "${SAVE_ROOT}" \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num "${TEST_NUM}" \
        --port "${PORT}"
    )
    client_rc=$?
    set -e

    after="$(result_total "${result_file}")"
    if [[ "${after}" -ge "${TEST_NUM}" ]]; then
      echo "recovery task=${task_name} complete=${after}/${TEST_NUM}"
      return 0
    fi
    if [[ "${after}" -gt "${before}" ]]; then
      no_progress_retries=0
      echo "recovery task=${task_name} progress=${before}->${after}"
      continue
    fi

    no_progress_retries=$((no_progress_retries + 1))
    echo "recovery task=${task_name} rc=${client_rc} no-progress=${before}->${after} retry=${no_progress_retries}/3" >&2
    if [[ "${no_progress_retries}" -ge 3 ]]; then
      return 1
    fi
    sleep 5
  done
}

claim_task_with_reserve() {
  python - "${TASK_QUEUE}" "${QUEUE_RESERVE}" <<'PY'
import fcntl
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
reserve = int(sys.argv[2])
lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
lock_path.parent.mkdir(parents=True, exist_ok=True)

with open(lock_path, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    if not queue_path.exists():
        raise SystemExit(1)
    tasks = [line.strip() for line in queue_path.read_text().splitlines() if line.strip()]
    if len(tasks) <= reserve:
        raise SystemExit(1)
    task = tasks.pop(0)
    queue_path.write_text("".join(f"{remaining}\n" for remaining in tasks))
    print(task)
PY
}

{
  echo "recovery_started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "protocol=official RT CPG1 DEFER0 RECREATE0 FAST0 LOW_RENDER0 POLICY_CAMERAS_ONLY0"
  echo "patch_sha256=$(sha256sum "${ROOT}/third_party/RoboTwin/envs/place_object_scale.py" | awk '{print $1}')"
  run_task place_object_scale

  while task_name="$(claim_task_with_reserve)"; do
    echo "recovery claimed task=${task_name}; reserving ${QUEUE_RESERVE} queued tasks for shard1"
    run_task "${task_name}"
  done

  echo "recovery_finished_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} >> "${RECOVERY_LOG}" 2>&1
