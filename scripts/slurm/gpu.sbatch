#!/bin/bash
# Reusable GPU job on Rivanna: runs the given command from the repo root.
#   mkdir -p /scratch/$USER/epiForecasting/logs        (once; Slurm does not create it)
#   sbatch -J nn scripts/slurm/gpu.sbatch .venv/bin/python -m pytest tests/agent/ -q
# bii-gpu is V100 32 GB except one A100 node; ask for it with `--gres=gpu:a100:1`.
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/epiForecasting/logs/%x-%j.out
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch [options] $0 <command> [args...]" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"
# Without this, xgboost and torch each start one thread per core of the whole node.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) commit=$(git rev-parse --short HEAD)"
exec "$@"
