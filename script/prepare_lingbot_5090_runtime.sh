#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/lingbot-va}
RUNTIME_ROOT=${RUNTIME_ROOT:-/scratch/lingbot-va-runtime}
VENV_ARCHIVE=${VENV_ARCHIVE:-${ROOT}/.runtime/venv.tar}
VENV_DIR=${RUNTIME_ROOT}/.venv
CHECKPOINTS_DIR=${RUNTIME_ROOT}/checkpoints
BASE_MODEL=${CHECKPOINTS_DIR}/lingbot-va-base
ROBOTWIN_DIR=${ROOT}/third_party/RoboTwin
ROBOTWIN_ASSETS=${RUNTIME_ROOT}/robotwin-assets
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu128}
TORCH_VERSION=${TORCH_VERSION:-2.9.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.24.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.9.0}
NUMPY_VERSION=${NUMPY_VERSION:-1.26.4}
PREPARE_MODEL_ASSETS=${PREPARE_MODEL_ASSETS:-1}

mkdir -p "${RUNTIME_ROOT}" "${CHECKPOINTS_DIR}" "${ROBOTWIN_ASSETS}"

if [[ ! -f "${VENV_ARCHIVE}" ]]; then
  echo "Missing venv archive: ${VENV_ARCHIVE}" >&2
  exit 1
fi

archive_fingerprint="$(stat -c '%s:%Y' "${VENV_ARCHIVE}")"
if [[ ! -f "${RUNTIME_ROOT}/.venv-ready" ]] || \
   [[ "$(cat "${RUNTIME_ROOT}/.venv-ready" 2>/dev/null || true)" != "${archive_fingerprint}" ]]; then
  rm -rf "${VENV_DIR}"
  tar -xf "${VENV_ARCHIVE}" -C "${RUNTIME_ROOT}"
  printf '%s\n' "${archive_fingerprint}" > "${RUNTIME_ROOT}/.venv-ready"
fi

ln -sfn "${VENV_DIR}" "${ROOT}/.venv"
ln -sfn "${CHECKPOINTS_DIR}" "${ROOT}/checkpoints"
ln -sfn "${ROBOTWIN_ASSETS}" "${ROBOTWIN_DIR}/assets"

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export ROBOTWIN_DIR

if ! python - <<'PY'
import torch
raise SystemExit(0 if "sm_120" in torch.cuda.get_arch_list() else 1)
PY
then
  python -m pip install --no-cache-dir --upgrade --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
fi

# MPLib's toppra extension in the archived environment was built against the
# NumPy 1.x C API and cannot be imported with NumPy 2.x.
if ! python -c "import numpy, mplib; raise SystemExit(0 if numpy.__version__ == '${NUMPY_VERSION}' else 1)"; then
  python -m pip install --no-cache-dir --upgrade --force-reinstall \
    "numpy==${NUMPY_VERSION}"
fi

if [[ "${PREPARE_MODEL_ASSETS}" == "1" ]]; then
  if [[ ! -f "${BASE_MODEL}/vae/diffusion_pytorch_model.safetensors" ]] || \
     [[ ! -f "${BASE_MODEL}/text_encoder/model.safetensors.index.json" ]]; then
    bash "${ROOT}/script/download_cn.sh" lingbot-va-base "${BASE_MODEL}"
  fi

  if [[ ! -d "${ROBOTWIN_ASSETS}/embodiments" ]] || \
     [[ ! -d "${ROBOTWIN_ASSETS}/objects" ]] || \
     [[ ! -d "${ROBOTWIN_ASSETS}/background_texture" ]]; then
    bash "${ROOT}/script/download_robotwin_sim_assets.sh"
  fi
fi

bash "${ROOT}/script/install_robotwin_vulkan_icd.sh"

python - <<'PY'
import os
import subprocess

import torch

expected = int(os.environ.get("EXPECTED_GPUS", "2"))
count = torch.cuda.device_count()
if count < expected:
    raise SystemExit(f"Need at least {expected} GPUs, found {count}")
if "sm_120" not in torch.cuda.get_arch_list():
    raise SystemExit(f"PyTorch build lacks sm_120: {torch.cuda.get_arch_list()}")

for index in range(count):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    value = torch.ones(32, device=f"cuda:{index}").sum().item()
    print(f"gpu={index} name={name} capability={capability} cuda_sum={value}")

import sapien.core  # noqa: F401,E402
import mplib  # noqa: F401,E402
from curobo.types.math import Pose  # noqa: F401,E402
import pytorch3d._C  # noqa: F401,E402
import warp as wp  # noqa: E402

wp.init()
for index in range(count):
    device = wp.get_device(f"cuda:{index}")
    print(f"warp_gpu={index} device={device}")

subprocess.run(["nvidia-smi", "-L"], check=True)
print("LingBot 5090 runtime preflight OK")
PY

if [[ "${PREPARE_MODEL_ASSETS}" == "1" ]]; then
  test -f "${BASE_MODEL}/vae/diffusion_pytorch_model.safetensors"
  test -f "${BASE_MODEL}/text_encoder/model.safetensors.index.json"
  test -d "${ROBOTWIN_ASSETS}/embodiments"
  test -d "${ROBOTWIN_ASSETS}/objects"
  test -d "${ROBOTWIN_ASSETS}/background_texture"
fi

cat > "${ROOT}/script/.robotwin_eval_env" <<EOF
export ROBOTWIN_DIR="${ROBOTWIN_DIR}"
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH="\${PYTHONPATH:-}:${ROOT}"
export HF_ENDPOINT="${HF_ENDPOINT}"
EOF

echo "Runtime ready under ${RUNTIME_ROOT}"
