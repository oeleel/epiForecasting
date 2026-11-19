#!/bin/bash
# GPU job on Rivanna with its own LLM server: starts Ollama on the job's node, waits
# until it answers, then runs the given command from the repo root. The server dies
# with the job, so nothing is left running.
#   sbatch -J improve scripts/slurm/llm.sbatch .venv/bin/python -m agent improve \
#       --cutoff-date 2025-12-27 --auto-apply --exclude-locations US
# The command sees LLM_BASE_URL and LLM_MODEL (read by `python -m agent`) and
# OPENAI_API_BASE / OPENAI_API_KEY (read by TimeSeriesScientist).
# One-time setup: unpack an Ollama Linux release into ~/opt/ollama (QUICKSTART has the lines).
# Override: `sbatch --export=ALL,LLM_MODEL=qwen3:14b ...`, `--gres=gpu:a100:1`, `-t 1-00:00:00`.
#SBATCH --account=nssac_students
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/epiForecasting/logs/%x-%j.out
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch [options] $0 <command> [args...]" >&2
    exit 2
fi

OLLAMA_HOME="${OLLAMA_HOME:-$HOME/opt/ollama}"
LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
# TimeSeriesScientist pastes the whole input series into each prompt; Ollama's default
# 4096-token context truncates it.
LLM_CONTEXT_TOKENS="${LLM_CONTEXT_TOKENS:-16384}"
SERVER_START_TIMEOUT_SECONDS=120
# Several jobs can share a node, so the port comes from the job id, not a constant.
port=$((20000 + SLURM_JOB_ID % 20000))
server_log="/scratch/$USER/epiForecasting/logs/ollama-$SLURM_JOB_ID.log"

if [ ! -x "$OLLAMA_HOME/bin/ollama" ]; then
    echo "no Ollama at $OLLAMA_HOME/bin/ollama; see the Rivanna section of QUICKSTART.md" >&2
    exit 1
fi

export OLLAMA_HOST="127.0.0.1:$port"
export OLLAMA_MODELS="$OLLAMA_HOME/models"
export OLLAMA_CONTEXT_LENGTH="$LLM_CONTEXT_TOKENS"
export OLLAMA_KEEP_ALIVE=-1

"$OLLAMA_HOME/bin/ollama" serve > "$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

waited=0
until curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Ollama exited during start; see $server_log" >&2
        exit 1
    fi
    if [ "$waited" -ge "$SERVER_START_TIMEOUT_SECONDS" ]; then
        echo "Ollama did not answer on $OLLAMA_HOST after ${SERVER_START_TIMEOUT_SECONDS}s; see $server_log" >&2
        exit 1
    fi
    sleep 2
    waited=$((waited + 2))
done
# No-op when the model is already in $OLLAMA_MODELS.
# The progress bar goes to stderr and would flood the job log; keep the last line only.
"$OLLAMA_HOME/bin/ollama" pull "$LLM_MODEL" 2>&1 | tail -n 1

export LLM_BASE_URL="http://$OLLAMA_HOST/v1"
export LLM_MODEL
export OPENAI_API_BASE="$LLM_BASE_URL"
export OPENAI_BASE_URL="$LLM_BASE_URL"
export OPENAI_API_KEY=ollama
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# The GPU is for the LLM server. The command runs on CPU unless COMMAND_USES_GPU=1: the
# torch build in .venv (2.15, CUDA 13) has no kernels for the V100s that fill bii-gpu, and
# a neural family then dies with "no kernel image is available". Set it to 1 on the A100 node.
if [ "${COMMAND_USES_GPU:-0}" != "1" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) llm=$LLM_MODEL ctx=$LLM_CONTEXT_TOKENS url=$LLM_BASE_URL commit=$(git rev-parse --short HEAD)"
# Not exec: the EXIT trap must run to stop the server.
"$@"
