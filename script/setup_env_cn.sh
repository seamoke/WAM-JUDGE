#!/usr/bin/env bash
# LingBot-VA environment setup for China network (no direct GitHub/HuggingFace).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

# Mirrors
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://mirror.sjtu.edu.cn/pytorch-wheels/cu126"
GITHUB_MIRROR="https://ghfast.top/https://github.com"

echo "==> [1/6] Ensure uv + Python 3.10.16"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv python install 3.10.16
if [[ ! -d .venv ]]; then
  uv venv --python 3.10.16 .venv
else
  echo "Reusing existing .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python --version

echo "==> [2/6] PyTorch 2.9.0 + CUDA 12.6 (SJTU mirror)"
uv pip install \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url "${TORCH_INDEX}"

echo "==> [3/6] Core dependencies (Tsinghua PyPI)"
uv pip install \
  websockets einops "diffusers==0.36.0" "transformers==4.55.2" \
  accelerate msgpack opencv-python matplotlib ftfy easydict \
  "numpy==1.26.4" tqdm "imageio[ffmpeg]" safetensors Pillow \
  --index-url "${PYPI_MIRROR}"

echo "==> [4/6] flash-attn (build; may take several minutes)"
uv pip install pip setuptools wheel --index-url "${PYPI_MIRROR}"
if ! uv pip install flash-attn --no-build-isolation --index-url "${PYPI_MIRROR}"; then
  echo "[warn] PyPI flash-attn failed, trying GitHub mirror..."
  uv pip install "flash-attn" --no-build-isolation \
    --index-url "${PYPI_MIRROR}" \
    -f "${GITHUB_MIRROR}/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3" || \
  uv pip install "git+${GITHUB_MIRROR}/Dao-AILab/flash-attention.git" --no-build-isolation
fi

echo "==> [5/6] ModelScope + install package"
uv pip install modelscope huggingface_hub --index-url "${PYPI_MIRROR}"
uv pip install -e . --index-url "${PYPI_MIRROR}"

echo "==> [6/6] Verify"
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
try:
    import flash_attn; print('flash_attn', flash_attn.__version__)
except Exception as e:
    print('flash_attn: not ready -', e)
"

cat <<'EOF'

Done. Activate:
  source .venv/bin/activate

Download model (ModelScope):
  bash script/download_model_ms.sh lingbot-va-base

Set model path in wan_va/configs/va_*_cfg.py:
  wan22_pretrained_model_name_or_path = "/workspace/lingbot-va/checkpoints/lingbot-va-base"
EOF
