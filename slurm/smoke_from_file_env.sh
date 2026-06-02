#!/bin/bash -l
# Step 2 of 2: run the GPU smoke training on the env built by build_env_from_file.sh.
#
# This is a Slurm TEMPLATE — adjust the #SBATCH directives (partition, account,
# GPU type/count, resources) to your own cluster before submitting.
#
# It runs a CUDA import check plus the polaris_smoke_test run (Story Quality
# judge disabled, so no external API). Keep the allocation short — the env is
# already built.
#
# Submit from the POLARIS repo root AFTER build_env_from_file.sh finishes:
#   sbatch slurm/smoke_from_file_env.sh
#
#SBATCH --job-name=polaris_smoke
#SBATCH --partition=gpu          # set to a GPU partition on your cluster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4             # 4 Ampere-or-newer GPUs (pin a type if your scheduler needs it)
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G               # covers FSDP offload + the 9B checkpoint gather to CPU; lower for smaller models
#SBATCH --time=00:45:00
#SBATCH --output=slurm/polaris_smoke_%j.out

set -euo pipefail

ROOT="${POLARIS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${ROOT}"
echo "=== POLARIS GPU smoke ==="
echo "repo root: ${ROOT}"; nvidia-smi -L || true

MINICONDA="${POLARIS_MINICONDA:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"
ENV_NAME="${POLARIS_ENV_NAME:-polaris}"
PY="${MINICONDA}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: env not found at ${PY}. Run slurm/build_env_from_file.sh first." >&2
  exit 1
fi

# CUDA-backed import check (this is what genuinely needs a GPU).
echo "=== CUDA import check ==="
"${PY}" - <<'INNERPY'
import torch, flash_attn, vllm
assert torch.cuda.is_available(), "CUDA not available on this node"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| devices:", torch.cuda.device_count())
print("flash_attn:", flash_attn.__version__, "| vllm:", vllm.__version__)
print("OK: GPU imports succeeded.")
INNERPY

echo "=== Verify install ==="
POLARIS_ROOT="${ROOT}" POLARIS_PYTHON="${PY}" bash "${ROOT}/scripts/verify_install.sh"

echo "=== polaris_smoke_test training run ==="
export POLARIS_ROOT="${ROOT}"
export HF_HOME="${HF_HOME:-${ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export VLLM_ALLOW_TRUST_REMOTE_CODE=1
export MKL_THREADING_LAYER=GNU
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
mkdir -p "${HF_HOME}"

# Optional GPU-count override (the shipped config uses 4). Set POLARIS_SMOKE_NGPUS
# to match your --gres; ppo_mini_batch_size (4) and train_batch_size (8) stay
# valid for 1/2/4-way data parallelism. Note: the config's memory budget is tuned
# for 4-way FSDP sharding, so fewer GPUs may need optimizer offload or a smaller
# gpu_memory_utilization.
NGPU_OVERRIDE=()
if [[ -n "${POLARIS_SMOKE_NGPUS:-}" ]]; then
  NGPU_OVERRIDE=(trainer.n_gpus_per_node="${POLARIS_SMOKE_NGPUS}")
  echo "Overriding trainer.n_gpus_per_node=${POLARIS_SMOKE_NGPUS}"
fi

cd "${ROOT}/verl"
HYDRA_FULL_ERROR=1 "${PY}" -m verl.trainer.main_ppo \
  --config-path "${ROOT}/configs/train" \
  --config-name polaris_smoke_test \
  trainer.logger=[console] \
  "${NGPU_OVERRIDE[@]}"

echo "=== VALIDATION COMPLETE: env built from file + smoke training ran ==="
