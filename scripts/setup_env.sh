#!/bin/bash
# POLARIS environment setup (one-command, ordered install).
#
# Recreates the exact public POLARIS environment from a clean machine:
#   1. conda env create from environment.yml   (python 3.10 + pip only)
#   2. pip install --no-deps -r requirements-lock.txt  (exact pinned closure, torch 2.10.0+cu128)
#   3. compile flash_attn 2.8.3 from source     (no prebuilt wheel exists for torch 2.10)
#   4. pip install -e ./verl --no-deps          (editable public VERL fork)
#   5. import smoke test                         (torch+cuda, vllm, flash_attn, verl)
#
# The lock is installed with --no-deps on purpose: it is a complete, already
# resolved closure, and the working env intentionally violates some upstream
# metadata pins (transformers 5.3.0.dev0 vs vllm's declared transformers<5),
# which a normal resolve would reject.
#
# Prerequisites:
#   - conda / miniconda available on PATH (or set POLARIS_CONDA_BIN)
#   - Linux x86_64 with an NVIDIA GPU and CUDA 12.x driver
#   - A CUDA 12.x toolkit providing nvcc for the flash_attn build. On HPC this is
#     usually `module load cuda/12.8`; set POLARIS_CUDA_MODULE to override, or
#     POLARIS_SKIP_CUDA_MODULE=1 if nvcc is already on PATH / CUDA_HOME is set.
#
# Usage:
#   bash scripts/setup_env.sh                 # creates env named "polaris"
#   POLARIS_ENV_NAME=myenv bash scripts/setup_env.sh
#
# Step 4 (the cuda/vllm/flash_attn import check) requires a visible GPU, so run
# this on a GPU node (e.g. inside an sbatch allocation). The conda create and the
# flash_attn compile themselves only need CPUs + nvcc.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${POLARIS_ENV_NAME:-polaris}"
CONDA_BIN="${POLARIS_CONDA_BIN:-conda}"
FLASH_ATTN_VERSION="${POLARIS_FLASH_ATTN_VERSION:-2.8.3}"
CUDA_MODULE="${POLARIS_CUDA_MODULE:-cuda/12.8}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}"

# Which CUDA architectures to compile flash_attn for. By default flash_attn 2.8.3
# builds kernels for EVERY supported arch (sm_80/90/100/120...), which is a
# multi-HOUR build. Limiting to the target arch(s) cuts it to ~20-40 min. We
# auto-detect from a visible GPU when present; otherwise default to A100+H100
# (8.0;9.0), which matches the POLARIS training hardware. Override with
# POLARIS_FLASH_ATTN_ARCHS, e.g. "8.0" for A100 only or "8.9" for L40S.
if [[ -z "${POLARIS_FLASH_ATTN_ARCHS:-}" ]]; then
  _caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' - || true)"
  POLARIS_FLASH_ATTN_ARCHS="${_caps:-8.0;9.0}"
fi
TORCH_CUDA_ARCH_LIST="${POLARIS_FLASH_ATTN_ARCHS}"            # dotted, e.g. "8.0;9.0"
FLASH_ATTN_CUDA_ARCHS="$(echo "${POLARIS_FLASH_ATTN_ARCHS}" | tr -d '.')"  # undotted, e.g. "80;90"
export TORCH_CUDA_ARCH_LIST FLASH_ATTN_CUDA_ARCHS

echo "=== POLARIS setup ==="
echo "repo root  : ${ROOT}"
echo "env name   : ${ENV_NAME}"
echo "flash_attn : ${FLASH_ATTN_VERSION} (compiled from source)"
echo "flash archs: ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS   : ${MAX_JOBS}"

# --- nvcc / CUDA toolkit for the flash_attn build -----------------------------
if [[ "${POLARIS_SKIP_CUDA_MODULE:-0}" != "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    echo "=== Loading CUDA toolkit module: ${CUDA_MODULE} ==="
    module load "${CUDA_MODULE}" || {
      echo "WARNING: could not 'module load ${CUDA_MODULE}'. Ensure nvcc is on PATH." >&2
    }
  fi
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "WARNING: nvcc not found on PATH. The flash_attn compile will fail unless" >&2
  echo "         CUDA_HOME points at a CUDA 12.x toolkit. Set POLARIS_CUDA_MODULE" >&2
  echo "         or load a CUDA toolkit before running this script." >&2
else
  echo "nvcc: $(command -v nvcc) ($(nvcc --version | tail -1))"
fi

# --- 1. create the conda env --------------------------------------------------
echo "=== 1/5 Creating conda env '${ENV_NAME}' (python 3.10 + pip) ==="
cd "${ROOT}"
if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Env '${ENV_NAME}' already exists; reusing it."
else
  "${CONDA_BIN}" env create -n "${ENV_NAME}" -f environment.yml
fi

# Resolve the env's interpreter deterministically from the conda prefix.
# (Do NOT rely on `conda run python`: in a non-interactive batch shell it can
#  fall back to the system python, which then trips PEP 668 / externally-managed.)
CONDA_BASE="$("${CONDA_BIN}" info --base)"
PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: env python not found at ${PY}" >&2
  exit 1
fi
echo "env python: ${PY}"
"${PY}" --version

# Put the env's bin on PATH so the `ninja` executable (shipped by the ninja pip
# package) is found by torch's BuildExtension. Without this, the flash_attn
# build silently falls back to the SERIAL distutils backend and takes many
# hours instead of ~10-20 min.
export PATH="$(dirname "${PY}"):${PATH}"

# --- 2. install the exact pinned closure (no resolver) ------------------------
echo "=== 2/5 Installing pinned closure (pip --no-deps -r requirements-lock.txt) ==="
echo "    --no-deps reproduces the working env, which intentionally violates some"
echo "    upstream metadata pins (transformers 5.3.0.dev0 vs vllm transformers<5)."
"${PY}" -m pip install --no-deps -r "${ROOT}/requirements-lock.txt"

# --- 3. compile flash_attn from source ----------------------------------------
# Needs nvcc + torch (step 2), but NOT a GPU — run this on a CPU node. The arch
# list above keeps the compile to ~20-40 min instead of multiple hours.
echo "=== 3/5 Compiling flash_attn==${FLASH_ATTN_VERSION} from source (--no-build-isolation) ==="
echo "    archs=${TORCH_CUDA_ARCH_LIST}  MAX_JOBS=${MAX_JOBS}  (CPU+nvcc; no GPU needed)"
if ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: 'ninja' not on PATH; the flash_attn build would fall back to the" >&2
  echo "       serial distutils backend (hours, not minutes). Expected it at" >&2
  echo "       $(dirname "${PY}")/ninja" >&2
  exit 1
fi
echo "    ninja: $(command -v ninja) ($(ninja --version))"
MAX_JOBS="${MAX_JOBS}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
"${PY}" -m pip install \
  "flash_attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-deps

# --- 4. editable verl fork ----------------------------------------------------
echo "=== 4/5 Installing public VERL fork (editable, --no-deps) ==="
"${PY}" -m pip install -e "${ROOT}/verl" --no-deps

# --- 5. import smoke test -----------------------------------------------------
# CPU-safe: torch.cuda.is_available() is just reported, not asserted, so this
# passes on a CPU build node. Run the GPU smoke training separately.
echo "=== 5/5 Import smoke test ==="
"${PY}" - <<'INNERPY'
import inspect, sys
import torch
print("python      :", sys.executable)
print("torch       :", torch.__version__)
print("torch.cuda  :", torch.version.cuda, "| available:", torch.cuda.is_available())
import vllm;          print("vllm        :", vllm.__version__)
import flash_attn;    print("flash_attn  :", flash_attn.__version__)
import transformers;  print("transformers:", transformers.__version__)
import verl;          print("verl        :", inspect.getfile(verl))
print("OK: core imports succeeded.")
INNERPY

echo "=== Done. Activate with: conda activate ${ENV_NAME} ==="
