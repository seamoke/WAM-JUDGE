#!/usr/bin/env bash
# Install NVIDIA Vulkan ICD so SAPIEN uses the system driver (not bundled fallback).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ICD_SRC="${ROOT}/script/nvidia_icd.json"
ICD_DST="/usr/share/vulkan/icd.d/nvidia_icd.json"
EGL_VENDOR_DIR="/usr/share/glvnd/egl_vendor.d"
EGL_VENDOR_DST="${EGL_VENDOR_DIR}/10_nvidia.json"

if [[ ! -f "${ICD_SRC}" ]]; then
  echo "Missing ${ICD_SRC}" >&2
  exit 1
fi

mkdir -p /usr/share/vulkan/icd.d
if [[ ! -f "${ICD_DST}" ]] || ! cmp -s "${ICD_SRC}" "${ICD_DST}"; then
  cp "${ICD_SRC}" "${ICD_DST}"
  echo "Installed NVIDIA Vulkan ICD -> ${ICD_DST}"
else
  echo "NVIDIA Vulkan ICD already installed at ${ICD_DST}"
fi

mkdir -p "${EGL_VENDOR_DIR}"
if [[ ! -f "${EGL_VENDOR_DST}" ]]; then
  cat > "${EGL_VENDOR_DST}" <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0"
  }
}
EOF
  echo "Installed NVIDIA EGL vendor config -> ${EGL_VENDOR_DST}"
else
  echo "NVIDIA EGL vendor config already installed at ${EGL_VENDOR_DST}"
fi
