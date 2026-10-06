#!/bin/bash
#SBATCH --job-name=eduardo-transfer-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --partition=normal
#SBATCH --time=11:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Build the Big-Math-RL-transfer dataset (near-transfer rewrites + k=16 student
# solve rates) with eduardo/create_transfer_dataset.py. API client only; needs
# a running student server, and a judge server unless Gemini is the rewriter.
# Rows are checkpointed, so resubmitting with the same --num-problems/--seed
# resumes the run.
#
# Env vars:
#   SERVING_API_KEY     bearer token of the vLLM serving endpoints
#   WORK_DIR            scratch dir for results and caches
#   EDUARDO_RECIPE_DIR  path to eduardo/ in the checkout
#   TRANSFER_OUT_DIR    optional: output dir
#   TRANSFER_PUSH       optional: 1 to push to the HF hub (needs HF_TOKEN)
#   TRANSFER_GEMINI     optional: 1 to rewrite/verify with Gemini (needs
#                       GEMINI_API_KEY; same as passing --gemini)
#
# Usage:
#   sbatch eduardo/scripts/transfer_dataset_slurm.sh --num-problems 100
#   sbatch --export=ALL,TRANSFER_PUSH=1 eduardo/scripts/transfer_dataset_slurm.sh
#   sbatch --export=ALL,TRANSFER_GEMINI=1 eduardo/scripts/transfer_dataset_slurm.sh

set -euo pipefail

# ── Required env ────────────────────────────────────────────────────────────
: "${SERVING_API_KEY:?set SERVING_API_KEY before sbatch}"
: "${WORK_DIR:=${SCRATCH:-$HOME/scratch}/eduardo}"
: "${EDUARDO_RECIPE_DIR:=$HOME/eduardo/eduardo}"
: "${TRANSFER_OUT_DIR:=${WORK_DIR}/data/bigmath_transfer}"
export WORK_DIR EDUARDO_RECIPE_DIR

EXTRA_ARGS=("$@")

# ── Container (pyxis) ───────────────────────────────────────────────────────
# Image built from the repo's Dockerfile. Set CONTAINER_ARGS="" to run on the host.
: "${CONTAINER_IMAGE:=${WORK_DIR}/eduardo.sqsh}"
: "${CONTAINER_MOUNTS:=${HOME}:${HOME},${WORK_DIR}:${WORK_DIR}}"
: "${CONTAINER_ARGS=--container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS}}"

if [[ "${TRANSFER_PUSH:-0}" == "1" ]]; then
  : "${HF_TOKEN:?TRANSFER_PUSH=1 requires HF_TOKEN for the private HF upload}"
  # HF_HOME is overridden below, so the cached CLI login token is not found.
  export HF_TOKEN
  PUSH_ARGS="--push"
  echo "[transfer_slurm] will push result to the private HF repo (see --hf-repo default in create_transfer_dataset.py)"
else
  PUSH_ARGS=""
  echo "[transfer_slurm] TRANSFER_PUSH not set — writing locally only"
fi

# ── Resolve serving endpoints ───────────────────────────────────────────────
resolve_serving_node() {
  local prefix="$1"
  squeue --user="${USER}" --noheader --format="%j %N" \
    | grep "^${prefix}" | head -1 | awk '{print $2}'
}

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Llama-3.1-8B-Instruct-${USER}}"
STUDENT_NODE=$(resolve_serving_node "serve_${STUDENT_MODEL_NAME}")
if [[ -z "${STUDENT_NODE}" ]]; then
  echo "[transfer_slurm] ERROR: no running SLURM job matching 'serve_${STUDENT_MODEL_NAME}*'. Launch the student server first." >&2
  exit 1
fi
export STUDENT_API_BASE="http://${STUDENT_NODE}:8080/v1"
export STUDENT_MODEL_NAME
echo "[transfer_slurm] Student endpoint: ${STUDENT_API_BASE} (node=${STUDENT_NODE})"

# Gemini mode (TRANSFER_GEMINI=1 or --gemini): no judge server needed.
GEMINI_ARGS=""
if [[ "${TRANSFER_GEMINI:-0}" == "1" || " ${EXTRA_ARGS[*]:-} " == *" --gemini "* ]]; then
  : "${GEMINI_API_KEY:?gemini mode requires GEMINI_API_KEY before sbatch}"
  export GEMINI_API_KEY
  [[ " ${EXTRA_ARGS[*]:-} " == *" --gemini "* ]] || GEMINI_ARGS="--gemini"
  echo "[transfer_slurm] Rewriter: Gemini API (skipping judge server resolution)"
else
  JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3.6-27B-${USER}}"
  JUDGE_NODE=$(resolve_serving_node "serve_${JUDGE_MODEL_NAME}")
  if [[ -z "${JUDGE_NODE}" ]]; then
    echo "[transfer_slurm] ERROR: no running SLURM job matching 'serve_${JUDGE_MODEL_NAME}*'. Launch it with eduardo/scripts/launch_judge_server.sh." >&2
    exit 1
  fi
  export JUDGE_API_BASE="http://${JUDGE_NODE}:8080/v1"
  export JUDGE_MODEL_NAME
  echo "[transfer_slurm] Judge endpoint:   ${JUDGE_API_BASE} (node=${JUDGE_NODE})"
fi

# ── Caches on scratch, never in $HOME ───────────────────────────────────────
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
export HF_HOME="${WORK_DIR}/huggingface"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/inductor"
export WANDB_DIR="${WORK_DIR}/wandb"
export WANDB_CACHE_DIR="${XDG_CACHE_HOME}/wandb"
export WANDB_CONFIG_DIR="${XDG_CACHE_HOME}/wandb-config"
mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
         "${WANDB_CONFIG_DIR}"

# ── Environment ─────────────────────────────────────────────────────────────
export PYTHONPATH="${EDUARDO_RECIPE_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
ulimit -c 0

mkdir -p "${TRANSFER_OUT_DIR}" "logs"

echo "[transfer_slurm] Output dir: ${TRANSFER_OUT_DIR}"

srun --nodes=1 \
     --ntasks-per-node=1 \
     --export=ALL \
     ${CONTAINER_ARGS} \
     bash -lc "cd ${EDUARDO_RECIPE_DIR}/.. && python -m eduardo.create_transfer_dataset \
       --out-dir ${TRANSFER_OUT_DIR} \
       ${PUSH_ARGS} ${GEMINI_ARGS} ${EXTRA_ARGS[*]:-}"
