cd /workspace/lingbot-va
source .venv/bin/activate
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Thorough cleanup before starting eval from tmux (12-shard ports: 29056-29067 / 29561-29572).
kill_stale_eval() {
  [[ -f logs/gpu_occupy.pid ]] && kill -9 "$(cat logs/gpu_occupy.pid)" 2>/dev/null || true
  rm -f logs/gpu_occupy.pid
  pkill -9 -f gpu_guard.sh 2>/dev/null || true
  pkill -9 -f gpu_occupy 2>/dev/null || true

  local -a patterns=(
    run_robotwin_eval.sh
    run_libero_eval.sh
    eval_polict_client_openpi
    evaluation/libero/run_server_ckpt
    'torch.distributed.run.*run_server_ckpt'
    run_server_ckpt.py
    'tee logs/robotwin_eval/all_checkpoints'
    'tee logs/libero_eval/all_checkpoints'
    'nohup.*run_robotwin_eval'
    'nohup.*run_libero_eval'
  )
  local round pat pid port leftover
  for round in 1 2 3; do
    for pat in "${patterns[@]}"; do
      pkill -9 -f "${pat}" 2>/dev/null || true
    done
    sleep 2
    if ! pgrep -f 'run_robotwin_eval|run_libero_eval|eval_polict_client|run_server_ckpt|torch.distributed.run.*run_server_ckpt' >/dev/null 2>&1; then
      break
    fi
  done

  for port in $(seq 29056 29067) $(seq 29561 29572); do
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
      while read -r pid; do
        [[ -n "${pid}" ]] && kill -9 "${pid}" 2>/dev/null || true
      done < <(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
    fi
  done
  sleep 3

  leftover=$(pgrep -af 'run_robotwin_eval|run_libero_eval|eval_polict_client|run_server_ckpt|torch.distributed.run.*run_server_ckpt' 2>/dev/null || true)
  if [[ -n "${leftover}" ]]; then
    echo "WARNING: force-killing leftover eval processes:" >&2
    echo "${leftover}" >&2
    while read -r pid; do
      [[ -n "${pid}" ]] && kill -9 "${pid}" 2>/dev/null || true
    done < <(echo "${leftover}" | awk '{print $1}')
    sleep 2
  fi

  if pgrep -f 'run_robotwin_eval|run_libero_eval|eval_polict_client|run_server_ckpt|torch.distributed.run.*run_server_ckpt' >/dev/null 2>&1; then
    echo "ERROR: stale eval processes still running. Run:" >&2
    echo "  pgrep -af 'run_robotwin_eval|eval_polict_client|run_server_ckpt'" >&2
    exit 1
  fi
  echo "Stale eval processes cleared."
}
kill_stale_eval
sleep 2
bash script/fix_robotwin_curobo.sh

CHECKPOINT_DIR=/workspace/lingbot-va/train_out/robotwin/checkpoints \
TEST_NUM=10 \
SKIP_EXISTING=1 \
ROBOTWIN_FAST=1 \
ROBOTWIN_EVAL_LOW_RENDER=1 \
CLIENTS_PER_GPU=3 \
bash script/run_robotwin_eval.sh 2>&1 | tee logs/robotwin_eval/all_checkpoints_$(date +%Y%m%d_%H%M%S).log


# kill $(cat logs/gpu_occupy.pid) 2>/dev/null; pkill -f gpu_guard.sh 2>/dev/null
# sleep 90

# CHECKPOINT_DIR=/workspace/lingbot-va/train_out/libero/checkpoints \
# LIBERO_FAST=1 \
# TEST_NUM=50 \
# MAX_ENV_STEPS=400 \
# SKIP_EXISTING=0 \
# LIBERO_RENDER_GL=egl \
# bash script/run_libero_eval.sh 2>&1 | tee logs/libero_eval/all_checkpoints_fast_$(date +%Y%m%d_%H%M%S).log
