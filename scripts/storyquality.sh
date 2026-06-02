#!/bin/bash -l
#SBATCH -J polaris
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --output=slurm/polaris_%j.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 04:00:00

set -euo pipefail

POLARIS_ROOT="${POLARIS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${POLARIS_PYTHON:-python}"
CONFIG_PATH="${POLARIS_CONFIG_PATH:-$POLARIS_ROOT/configs/train}"
CONFIG_NAME="${POLARIS_CONFIG_NAME:-polaris_main}"
RUN_TAG="${POLARIS_RUN_TAG:-}"
EXTRA_ARGS="${POLARIS_EXTRA_ARGS:-}"

while getopts "l:p:n:r:a:" opt; do
  case "$opt" in
    l) PYTHON_BIN="$OPTARG" ;;
    p) CONFIG_PATH="$OPTARG" ;;
    n) CONFIG_NAME="$OPTARG" ;;
    r) RUN_TAG="$OPTARG" ;;
    a) EXTRA_ARGS="$OPTARG" ;;
    \?) echo "Usage: $0 [-l python_bin] [-p config_path] [-n config_name] [-r run_tag] [-a extra_args]"; exit 1 ;;
  esac
done

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  JOB_TAG="${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
else
  JOB_TAG="${SLURM_JOB_ID:-local}"
fi

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$CONFIG_NAME"
fi

export POLARIS_ROOT
export VLLM_ALLOW_TRUST_REMOTE_CODE=1
export MKL_THREADING_LAYER=GNU
export HF_HOME="${HF_HOME:-$POLARIS_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

if [[ -n "${POLARIS_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$POLARIS_CONDA_ENV"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-polaris}"
  export WANDB_NAME="${RUN_TAG}-${JOB_TAG}"
fi

echo "Running POLARIS training with ${PYTHON_BIN}"
HYDRA_FULL_ERROR=1 "${PYTHON_BIN}" -m verl.trainer.main_ppo   --config-path "${CONFIG_PATH}"   --config-name "${CONFIG_NAME}"   trainer.project_name="${WANDB_PROJECT:-polaris}"   trainer.experiment_name="${WANDB_NAME:-${RUN_TAG}-${JOB_TAG}}"   ${EXTRA_ARGS}
