#!/bin/bash -l
# Step 1 of 2: build the POLARIS environment from the shipped files.
#
# This is a Slurm TEMPLATE — adjust the #SBATCH directives (partition, account,
# resources) to your own cluster before submitting.
#
# Building the env (conda create + pinned --no-deps install + flash_attn compile +
# editable ./verl) needs nvcc and CPUs but NOT a GPU, so it can run on a CPU
# partition. flash_attn is compiled only for the target compute capabilities
# (default 8.0;9.0) to keep the build to tens of minutes instead of hours.
#
# Submit from the POLARIS repo root, then run slurm/smoke_from_file_env.sh:
#   sbatch slurm/build_env_from_file.sh
#
#SBATCH --job-name=polaris_build
#SBATCH --partition=cpu          # set to a CPU partition on your cluster
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/polaris_build_%j.out

set -euo pipefail

ROOT="${POLARIS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${ROOT}"
echo "=== POLARIS env build (CPU) ==="
echo "repo root: ${ROOT}"

MINICONDA="${POLARIS_MINICONDA:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"
CONDA_BIN="${POLARIS_CONDA_BIN:-${MINICONDA}/bin/conda}"
ENV_NAME="${POLARIS_ENV_NAME:-polaris}"

# Fresh build every time.
echo "=== Removing any stale '${ENV_NAME}' env ==="
"${CONDA_BIN}" env remove -n "${ENV_NAME}" -y 2>/dev/null || true
/bin/rm -rf "${MINICONDA}/envs/${ENV_NAME}"

# conda create + pinned closure + flash_attn compile + editable verl + import test.
POLARIS_ENV_NAME="${ENV_NAME}" \
POLARIS_CONDA_BIN="${CONDA_BIN}" \
POLARIS_CUDA_MODULE="${POLARIS_CUDA_MODULE:-cuda/12.8}" \
POLARIS_FLASH_ATTN_ARCHS="${POLARIS_FLASH_ATTN_ARCHS:-8.0;9.0}" \
MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-16}}" \
bash "${ROOT}/scripts/setup_env.sh"

PY="${MINICONDA}/envs/${ENV_NAME}/bin/python"
echo "=== Verify install ==="
POLARIS_ROOT="${ROOT}" POLARIS_PYTHON="${PY}" bash "${ROOT}/scripts/verify_install.sh"

echo "=== BUILD COMPLETE. Env: ${MINICONDA}/envs/${ENV_NAME} ==="
echo "Next: sbatch slurm/smoke_from_file_env.sh   (runs the GPU smoke training)"
