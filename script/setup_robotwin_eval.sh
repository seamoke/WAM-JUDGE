#!/usr/bin/env bash
# Install RoboTwin-2.0 sim environment for evaluation (lingbot-va server stays in .venv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

ROBOTWIN_DIR="${ROBOTWIN_DIR:-${ROOT}/third_party/RoboTwin}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-2eeec322}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.org/simple}"
GITHUB_MIRROR="${GITHUB_MIRROR:-https://github.com}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# shellcheck disable=SC1091
source .venv/bin/activate
export HF_ENDPOINT

echo "==> [1/7] System deps (Vulkan + build tools)"
if command -v apt-get >/dev/null 2>&1; then
  APT_GET=(apt-get)
  if [[ "$(id -u)" -ne 0 ]]; then
    APT_GET=(sudo apt-get)
  fi
  "${APT_GET[@]}" install -y \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    libegl1 libgl1-mesa-glx libosmesa6 \
    build-essential git wget ffmpeg
else
  echo "WARN: apt-get not found; install Vulkan/GL libs manually if render fails" >&2
fi

echo "==> [2/7] Clone RoboTwin @ ${ROBOTWIN_COMMIT}"
if [[ -d "${ROBOTWIN_DIR}/.git" ]]; then
  git -C "${ROBOTWIN_DIR}" fetch --depth 1 origin "${ROBOTWIN_COMMIT}" 2>/dev/null || true
  git -C "${ROBOTWIN_DIR}" checkout "${ROBOTWIN_COMMIT}"
elif [[ -f "${ROBOTWIN_DIR}/envs/_base_task.py" ]]; then
  echo "Using vendored RoboTwin source snapshot at ${ROBOTWIN_DIR}"
else
  mkdir -p "$(dirname "${ROBOTWIN_DIR}")"
  git clone "${GITHUB_MIRROR}/RoboTwin-Platform/RoboTwin.git" "${ROBOTWIN_DIR}"
  git -C "${ROBOTWIN_DIR}" fetch --depth 1 origin "${ROBOTWIN_COMMIT}" 2>/dev/null || true
  git -C "${ROBOTWIN_DIR}" checkout "${ROBOTWIN_COMMIT}"
fi

echo "==> [3/7] Patch RoboTwin requirements (per README)"
cat > "${ROBOTWIN_DIR}/script/requirements.txt" <<'EOF'
transforms3d==0.4.2
sapien==3.0.0b1
scipy==1.10.1
mplib==0.2.1
gymnasium==0.29.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
pydantic
zarr
openai
huggingface_hub==0.36.2
h5py
azure==4.0.0
azure-ai-inference
pyglet<2
wandb
moviepy
imageio
termcolor
av
matplotlib
ffmpeg
EOF

if grep -q 'pip install torch' "${ROBOTWIN_DIR}/script/_install.sh"; then
  sed -i 's|pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation|pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation|' \
    "${ROBOTWIN_DIR}/script/_install.sh" || true
fi
sed -i '8s|.*|pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation|' \
  "${ROBOTWIN_DIR}/script/_install.sh"

echo "==> [4/7] Install RoboTwin Python deps into lingbot-va .venv"
pip install -q 'setuptools<81'  # sapien needs pkg_resources
pushd "${ROBOTWIN_DIR}" >/dev/null
if command -v uv >/dev/null 2>&1; then
  uv pip install -r script/requirements.txt --index-url "${PYPI_MIRROR}"
else
  pip install -r script/requirements.txt --index-url "${PYPI_MIRROR}"
fi
if ! python -c "import pytorch3d" 2>/dev/null; then
  pip install "git+${GITHUB_MIRROR}/facebookresearch/pytorch3d.git@stable" --no-build-isolation
fi
# sapien/mplib patches + curobo (mirror-friendly; skip RoboTwin _install.sh re-installs)
SAPIEN_LOCATION="$(pip show sapien | awk '/^Location:/ {print $2}')/sapien"
URDF_LOADER="${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
[[ -f "${URDF_LOADER}" ]] && sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
MPLIB_LOCATION="$(pip show mplib | awk '/^Location:/ {print $2}')/mplib"
PLANNER="${MPLIB_LOCATION}/planner.py"
[[ -f "${PLANNER}" ]] && sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.7}"
pushd envs >/dev/null
if [[ -d curobo/.git ]]; then
  if ! git -C curobo describe --tags --exact-match 2>/dev/null | grep -q "${CUROBO_TAG#v}"; then
    echo "WARN: vendored cuRobo git checkout is not exactly ${CUROBO_TAG}" >&2
  fi
elif [[ -f curobo/src/curobo/__init__.py ]] && \
     [[ -d curobo/src/curobo/content/assets ]]; then
  echo "Using vendored cuRobo source snapshot"
else
  echo "Vendored cuRobo source/assets are incomplete; fetching ${CUROBO_TAG}"
  rm -rf curobo
  git clone --depth 1 --branch "${CUROBO_TAG}" "${GITHUB_MIRROR}/NVlabs/curobo.git" curobo
fi
pip install -e curobo --no-build-isolation
sed -i 's/wp\.torch\.device_from_torch/wp.device_from_torch/g' \
  "${ROBOTWIN_DIR}/envs/curobo/src/curobo/geom/sdf/world_mesh.py"
popd >/dev/null
popd >/dev/null

echo "==> [5/7] Download RoboTwin assets (HF mirror)"
bash "${ROOT}/script/download_robotwin_sim_assets.sh"

echo "==> [6/7] Verify imports"
export ROBOTWIN_DIR
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python - <<'PY'
import os
os.environ.setdefault("ROBOTWIN_DIR", os.environ["ROBOTWIN_DIR"])
import sapien.core  # noqa: F401
from evaluation.robotwin.test_render import Sapien_TEST
Sapien_TEST()
print("RoboTwin render OK")
PY

echo "==> [7/7] Write eval env snippet"
bash "${ROOT}/script/install_robotwin_vulkan_icd.sh"

ENV_SNIPPET="${ROOT}/script/.robotwin_eval_env"
cat > "${ENV_SNIPPET}" <<EOF
export ROBOTWIN_DIR="${ROBOTWIN_DIR}"
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH="\${PYTHONPATH:-}:${ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT}"
EOF

cat <<EOF

RoboTwin eval setup complete.

Before eval (after training finishes):
  # stop training / gpu guard, then:
  bash script/run_robotwin_eval.sh

Quick smoke test (10 rollouts/task):
  TEST_NUM=10 MAX_CHECKPOINTS=1 bash script/run_robotwin_eval.sh
EOF
