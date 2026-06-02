#!/bin/bash -l
#SBATCH -J polaris_merge
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH --output=slurm/polaris_merge_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00

set -euo pipefail

POLARIS_ROOT="${POLARIS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${POLARIS_PYTHON:-python}"
MERGER_SCRIPT="${POLARIS_MERGER_SCRIPT:-$POLARIS_ROOT/verl/scripts/legacy_model_merger.py}"

CKPT_DIR="${1:?Usage: sbatch merge_ckpt.sh <actor_dir> <target_dir>}"
TARGET_DIR="${2:?Usage: sbatch merge_ckpt.sh <actor_dir> <target_dir>}"

echo "Merging FSDP checkpoint: ${CKPT_DIR} -> ${TARGET_DIR}"
"${PYTHON_BIN}" "${MERGER_SCRIPT}" merge   --backend fsdp   --local_dir "${CKPT_DIR}"   --target_dir "${TARGET_DIR}"

echo "Done. Merged model at: ${TARGET_DIR}"
