#!/bin/bash
# Reusable CPU job on Rivanna: runs the given command from the repo root.
#   mkdir -p /scratch/$USER/epiForecasting/logs        (once; Slurm does not create it)
#   sbatch -J tests scripts/slurm/cpu.sbatch .venv/bin/python -m pytest tests/agent/ -q
# Override any default on the command line, e.g. `sbatch -c 16 -t 12:00:00 ...`.
#SBATCH --account=nssac_students
#SBATCH --partition=bii
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
echo "job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK commit=$(git rev-parse --short HEAD)"
exec "$@"
