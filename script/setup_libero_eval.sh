#!/usr/bin/env bash
# Install LIBERO sim deps for evaluation (no torch downgrade).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

LIBERO_DIR="${LIBERO_DIR:-${ROOT}/third_party/LIBERO}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
GITHUB_MIRROR="${GITHUB_MIRROR:-https://ghfast.top/https://github.com}"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [1/6] Clone LIBERO (if missing)"
if [[ ! -d "${LIBERO_DIR}/libero" ]]; then
  mkdir -p "$(dirname "${LIBERO_DIR}")"
  git clone --depth 1 "${GITHUB_MIRROR}/Lifelong-Robot-Learning/LIBERO.git" "${LIBERO_DIR}"
fi

echo "==> [2/6] Install system GL/EGL libs (MuJoCo offscreen rendering)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get install -y libegl1 libgl1-mesa-glx libosmesa6
else
  echo "WARN: apt-get not found; install libegl1 libgl1-mesa-glx libosmesa6 manually if client EGL fails" >&2
fi

echo "==> [3/6] Install sim deps (skip torch)"
if command -v uv >/dev/null 2>&1; then
  uv pip install robosuite==1.4.0 bddl==1.0.1 "gym>=0.25" "mujoco>=3.3.0,<3.10" imageio-ffmpeg pyyaml future \
    --index-url "${PYPI_MIRROR}"
else
  pip install robosuite==1.4.0 bddl==1.0.1 "gym>=0.25" "mujoco>=3.3.0,<3.10" imageio-ffmpeg pyyaml future \
    --index-url "${PYPI_MIRROR}"
fi

echo "==> [4/6] Install LIBERO package (editable, source tree)"
# Remove stale partial installs that shadow the real package (namespace-only libero/libero).
pip uninstall -y libero 2>/dev/null || true
LIBERO_SITE_PKG="$(python -c "import site; print(site.getsitepackages()[0])")/libero"
rm -rf "${LIBERO_SITE_PKG}" 2>/dev/null || true
if command -v uv >/dev/null 2>&1; then
  uv pip install -e "${LIBERO_DIR}"
else
  pip install -e "${LIBERO_DIR}"
fi

echo "==> [5/6] Write LIBERO config (non-interactive)"
LIBERO_PKG="${LIBERO_DIR}/libero/libero"
python - <<PY
import os, yaml
root = os.path.abspath("${LIBERO_PKG}")
cfg_dir = os.path.expanduser("~/.libero")
os.makedirs(cfg_dir, exist_ok=True)
paths = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "bddl_files"),
    "init_states": os.path.join(root, "init_files"),
    "datasets": os.path.join(os.path.dirname(root), "datasets"),
    "assets": os.path.join(root, "assets"),
}
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
    yaml.dump(paths, f)
print("Wrote", os.path.join(cfg_dir, "config.yaml"))
PY

echo "==> [6/6] Verify import"
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
# LIBERO repo root must be on PYTHONPATH (not .../libero, which maps inner libero/ as top-level).
export PYTHONPATH="${LIBERO_DIR}:${PYTHONPATH:-}"
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.pop("PYOPENGL_PLATFORM", None)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
print("LIBERO OK:", sorted(benchmark.get_benchmark_dict().keys()))
PY

# Persist env for eval scripts (editable install is primary; PYTHONPATH is fallback).
ENV_SNIPPET="${ROOT}/script/.libero_eval_env"
cat > "${ENV_SNIPPET}" <<EOF
export MUJOCO_GL=osmesa
export PYTHONPATH="\${PYTHONPATH:-}:${LIBERO_DIR}"
EOF

cat <<EOF

LIBERO eval setup complete.

Run all checkpoints (4 GPUs, 4 benchmarks):
  bash script/run_libero_eval.sh
EOF
